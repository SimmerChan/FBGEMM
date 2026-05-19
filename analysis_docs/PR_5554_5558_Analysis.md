# FBGEMM PR #5554 & #5558 分析报告

## 概览

这两个 PR 属于同一功能系列 —— **TurboSSD v2 推理服务化**，共同为 FBGEMM 的 SSD Table-Batched Embedding (TBE) 推理引擎增加流式更新和服务化封装能力。目标是让 Video Retrieval HSTU 模型（VDD、New2、IFU）能用 FBGEMM SSD TBE 替代 EmbeddingDB。

| 属性 | PR #5554 | PR #5558 |
|------|----------|----------|
| 标题 | Add streaming_update() and load_snapshot() for inference | Add TurboSSDInferenceModule for HSTU serving integration |
| 合并日期 | 2026-04-09 | 2026-04-09 |
| 变更规模 | +2111 / -2 行 | +545 / -0 行 |
| 涉及文件 | 3 个文件 | 3 个文件 |
| 定位 | 底层原语（推理引擎核心能力） | 上层封装（服务化模块） |
| 依赖关系 | 独立 | 依赖 #5554 |

---

## PR #5554: 流式更新原语

### 功能目标

为 SSD TBE 推理算子 (`SSDIntNBitTableBatchedEmbeddingBags`) 添加流式增量更新和快照切换能力，对齐 EmbeddingDB 的 streaming delta table 特性。

### 新增核心 API

#### 1. `streaming_update(indices, weights)`

- **作用**: 在推理期间写入更新后的 embedding 行到 RocksDB，并使对应的 HBM 缓存条目失效
- **流程**: 写 RocksDB → 向量化缓存失效（set-associative cache invalidation）
- **线程安全**: 获取排他写锁 + CUDA 同步

#### 2. `load_snapshot(ssd_storage_directory, ...)`

- **作用**: 无停机切换到新的 RocksDB 快照
- **流程**: flush 当前 DB → 打开新 DB 实例 → 全量清除 HBM 缓存
- **限制**: 不支持 parameter server 后端（仅支持本地 RocksDB）

### 关键技术实现

#### 读写锁 `_RWLock`

```
读者 (prefetch/forward)    → 共享读锁（~2µs 开销）
写者 (streaming_update/
      load_snapshot)       → 排他写锁 + CUDA synchronize
```

- **写者优先策略**: 一旦有写者等待，新读者被阻塞直到写者完成，防止持续读负载下写者饿死
- 使用计数器（非布尔值）跟踪等待写者数量，避免多个排队的写者互相饿死
- 写锁上下文管理器在获取/释放时均执行 `torch.cuda.synchronize()`

#### HBM 缓存失效

`_invalidate_cache()` 使用向量化操作：
1. 计算每个 index 的 cache set (`index % max_cache_sets`)
2. Gather 相关 cache set 的 ASSOC 个槽位
3. 找到匹配槽位并置为 -1

#### AMD/ROCm 适配

- 检测 `IS_ROCM` 标志
- 构造时发出警告：`streaming_update`/`load_snapshot` 可用，但 `prefetch`/`forward` 的 C++ 内核未移植到 HIP（warp size 64 vs ASSOC=32 不匹配）
- 文档明确标注各方法的 AMD 支持状态

### 变更文件

| 文件 | 变更 | 说明 |
|------|------|------|
| `fbgemm_gpu/fbgemm_gpu/tbe/ssd/inference.py` | +234 / -2 | 核心实现：RWLock、streaming_update、load_snapshot、cache invalidation |
| `fbgemm_gpu/test/tbe/ssd/ssd_rwlock_test.py` | +1177 | RWLock 并发测试 |
| `fbgemm_gpu/test/tbe/ssd/ssd_split_tbe_inference_test.py` | +700 | 流式更新/快照切换集成测试 |

---

## PR #5558: 服务化封装模块

### 功能目标

在 PR #5554 的底层原语之上，构建一个面向服务的封装模块 `TurboSSDInferenceModule`，使 HSTU 模型能以最小改动从 EmbeddingDB 迁移到 FBGEMM SSD TBE。

### 新增核心组件

#### `TurboSSDInferenceModule` 类

一个 `nn.Module` 子类，包装 `SSDIntNBitTableBatchedEmbeddingBags`，提供以下能力：

##### 1. 单调用 forward（自动 prefetch + lookup）

```python
output = module(indices, offsets)  # 自动 prefetch 然后 lookup
```

替代 EmbeddingDB 的同步 SSD 读取模式，利用 GPU HBM 缓存 + LRU 淘汰。

##### 2. 工厂方法 `from_embedding_specs()`

- 根据 embedding 规格、目标命中率、HBM 预算自动计算 cache_sets
- 内部调用 `_compute_cache_sets()` 基于 ASSOC=32 路组相联缓存模型
- 适用于容量规划：H100 (96 GB)、MI350X (288 GB)

##### 3. `estimate_hbm_gb()` 静态方法

- 根据给定的 embedding 规格和目标命中率估算 HBM 使用量
- 无需实例化模块即可进行容量规划

##### 4. 流式更新代理

- `streaming_update()` → 委托到底层 TBE
- `load_snapshot()` → 委托到底层 TBE

### 变更文件

| 文件 | 变更 | 说明 |
|------|------|------|
| `fbgemm_gpu/fbgemm_gpu/tbe/ssd/inference_serving.py` | +260 (新文件) | TurboSSDInferenceModule 实现 |
| `fbgemm_gpu/fbgemm_gpu/tbe/ssd/__init__.py` | +1 | 导出新模块 |
| `fbgemm_gpu/test/tbe/ssd/ssd_split_tbe_inference_test.py` | +284 | 封装模块测试 |

### 测试覆盖

PR #5558 新增了 `TurboSSDInferenceModuleTest` 测试类，包含 8 个测试用例：

| 测试 | 验证内容 |
|------|----------|
| `test_from_embedding_specs_creates_module` | 工厂方法基本创建 |
| `test_from_embedding_specs_multi_table` | 多表场景（HSTU 风格） |
| `test_forward_correctness` | 封装 forward 与原始 TBE 输出一致性 |
| `test_streaming_update_through_wrapper` | 通过封装进行流式更新 |
| `test_load_snapshot_through_wrapper` | 通过封装进行快照切换 |
| `test_estimate_hbm_gb` | HBM 估算合理性（1.6B 行 INT8 场景 ~190GB） |
| `test_estimate_hbm_gb_multi_table` | 多表 HBM 估算 |
| `test_hbm_budget_cap` | HBM 预算上限约束 |
| `test_full_hstu_flow` | 端到端 HSTU 流程：创建→加载→forward→delta更新→验证 |

---

## 架构关系

```
┌─────────────────────────────────────────────────────┐
│              HSTU Serving (TGIF 框架)                 │
│  通过 DIShardingPass 替换 SSDEmbeddingDBSplitTable-  │
│  BatchedEmbeddingBagsCodegen                         │
└────────────────────┬────────────────────────────────┘
                     │ 调用
┌────────────────────▼────────────────────────────────┐
│       TurboSSDInferenceModule  (PR #5558)            │
│  ┌─────────────────────────────────────────────┐    │
│  │  from_embedding_specs()  — 工厂方法          │    │
│  │  forward()               — 自动 prefetch    │    │
│  │  streaming_update()      — 增量更新代理      │    │
│  │  load_snapshot()         — 快照切换代理      │    │
│  │  estimate_hbm_gb()       — 容量规划          │    │
│  └─────────────────────────────────────────────┘    │
└────────────────────┬────────────────────────────────┘
                     │ 委托
┌────────────────────▼────────────────────────────────┐
│  SSDIntNBitTableBatchedEmbeddingBags  (PR #5554)     │
│  ┌─────────────────────────────────────────────┐    │
│  │  _RWLock          — 读写锁（写者优先）       │    │
│  │  prefetch()       — HBM 缓存预取            │    │
│  │  forward()        — lookup + 反量化          │    │
│  │  streaming_update()  — RocksDB 写 + 缓存失效 │    │
│  │  load_snapshot()  — 无停机 DB 切换           │    │
│  │  _invalidate_cache() — 向量化缓存失效        │    │
│  └─────────────────────────────────────────────┘    │
│                                                      │
│  存储层: RocksDB (SSD) + GPU HBM Cache (LRU)         │
└──────────────────────────────────────────────────────┘
```

---

## 总结

| 维度 | 分析 |
|------|------|
| **设计模式** | 经典的底层原语 + 上层封装分层架构。#5554 提供原子能力，#5558 组合为服务化 API |
| **迁移路径** | 替代 EmbeddingDB：DIShardingPass 在模型图中替换算子，保持上层接口不变 |
| **并发安全** | 写者优先 RWLock 保证推理延迟不受影响，同时支持 45 分钟级别的增量发布周期 |
| **硬件适配** | 明确标注 NVIDIA/AMD 支持矩阵，ROCm 路径有限但流式 API 可用 |
| **测试质量** | 包含单元测试、并发测试、集成测试和端到端 HSTU 流程测试，覆盖充分 |

---

## 技术性质：纯 Python 业务逻辑封装，无新增 CUDA/C++ 算子

**结论：这两个 PR 没有新增任何 CUDA kernel 或自定义 C++ 算子，属于纯 Python 层的业务逻辑封装。**

### 文件类型分析

| PR | 变更文件 | 语言 |
|----|----------|------|
| #5554 | `inference.py`, `ssd_rwlock_test.py`, `ssd_split_tbe_inference_test.py` | Python |
| #5558 | `inference_serving.py` (新), `__init__.py`, `ssd_split_tbe_inference_test.py` | Python |

无任何 `.cu`、`.cuh`、`.cpp`、`.h` 文件被修改或新增。

### 具体技术成分拆解

**PR #5554 — 所有新增能力均基于已有的 Python/C++ 基础设施组合：**

| 新增能力 | 实现方式 | 底层依赖 |
|----------|----------|----------|
| `_RWLock` | 纯 Python `threading.Condition` + `threading.Lock` | 无 GPU 依赖 |
| `streaming_update()` | 调用已有 `self.ssd_db.set()` (C++ RocksDB binding) + PyTorch tensor 运算 | 已有算子 |
| `_invalidate_cache()` | PyTorch 标准运算：`%`、gather、`==`、`nonzero`、索引赋值 | 已有算子 |
| `load_snapshot()` | 创建已有 `EmbeddingRocksDBWrapper` 实例 + `lxu_cache_state.fill_(-1)` | 已有算子 |
| prefetch/forward 加锁 | 在已有 `prefetch()`/`forward()` 外包一层 `read_lock` 上下文管理器 | 已有算子 |

**PR #5558 — 纯粹的封装层：**

| 新增能力 | 实现方式 |
|----------|----------|
| `TurboSSDInferenceModule` | `nn.Module` 包装已有的 `SSDIntNBitTableBatchedEmbeddingBags` |
| `from_embedding_specs()` | Python 数学计算 cache_sets + 调用已有构造函数 |
| `forward()` | 串行调用已有的 `tbe.prefetch()` + `tbe()` |
| `estimate_hbm_gb()` | 纯 Python 算术计算 |
| `streaming_update()` / `load_snapshot()` | 直接委托给底层 TBE（即 #5554 的实现） |

### 依赖的已有底层算子

这两个 PR 的功能完全建立在以下**已存在的** C++/CUDA 算子之上：

- `torch.ops.fbgemm.linearize_cache_indices` — 索引线性化 (CUDA)
- `torch.classes.fbgemm.EmbeddingRocksDBWrapper` — RocksDB 读写接口 (C++)
- `torch.ops.fbgemm.ssd_cache_populate_actions` — 缓存填充 (CUDA)
- `torch.ops.fbgemm.masked_index_put` — 掩码写入 (CUDA)
- `lxu_cache_state` — GPU HBM 缓存状态管理 (CUDA tensor)

这些底层算子在此前的 PR 中已经实现并稳定运行，#5554 和 #5558 仅在 Python 层对它们进行了新的编排和组合。
