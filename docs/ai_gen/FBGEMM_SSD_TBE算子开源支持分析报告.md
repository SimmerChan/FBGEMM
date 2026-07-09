# FBGEMM SSD TBE 算子开源支持分析报告

> **分析日期**: 2026-06-23
> **分析分支**: v1.5.0-release
> **分析范围**: FBGEMM_GPU SSD (Sequential Sparse Densed) Table Batched Embedding 算子的开源实现完整性
> **关联 Issue**: [pytorch/FBGEMM#5666](https://github.com/pytorch/FBGEMM/issues/5666)

---

## 摘要

**FBGEMM 开源版本不完整支持 SSD TBE 算子。** 准确说:SSD 的**源代码完整存在于仓库**(真实实现,非桩),但这些手写的 SSD 源文件**从未被纳入开源的 CMake / setup.py 编译列表**——这是一种**结构性禁用**,既不是 `#ifdef` 条件编译,也不是 feature flag,而是源文件清单层面的遗漏。

社区 issue #5666 的结论是正确的,但措辞 "not supported" 需要精确化:**代码有,构建不编**。

要在 GPU 上完整实现 SSD TBE 特性(核心训练/推理路径),需要补齐 5 大类内容,工程量约 2-3 周。真正的硬天花板是 `dram_kv_embedding_cache` 模块依赖的 4 个 Meta 内部 enrichment 头(开源仓库物理不存在),但 SSD 核心功能不依赖它。

---

## 1. 问题背景

### 1.1 Issue #5666 现象

用户运行 `fbgemm_gpu/test/tbe/ssd/kv_backend_test.py` 时报错:

```
AttributeError: '_OpNamespace' 'fbgemm' object has no attribute
'get_bucket_sorted_indices_and_bucket_tensor'
```

用户疑惑 SSD TBE 算子是否不支持,但注意到 torchrec 仍使用 `from fbgemm_gpu.tbe.ssd import SSDTableBatchedEmbeddingBags`。

### 1.2 根因

该算子的**注册点**在 `ssd_split_table_batched_embeddings.cpp`(`TORCH_LIBRARY_FRAGMENT(fbgemm, m)` + `DISPATCH_TO_CPU`),**真实 CPU 实现**在 `kv_db_cpp_utils.cpp`(用 `std::map` + `folly::CPUThreadPoolExecutor` 做 bucket 排序)。但 `ssd_split_table_batched_embeddings.cpp` 这个文件**不在任何构建列表里**。

所以这份 `.cpp` 从未被编译成 `.so`,算子自然未注册。**无论用什么 build target 安装,该算子都不会存在**——这是确定性的,与用户环境无关。

---

## 2. SSD TBE 存储栈架构

```
SSDTableBatchedEmbeddingBags  (fbgemm_gpu/tbe/ssd/training.py:230)
  │
  ├─ 构造 → torch.classes.fbgemm.EmbeddingRocksDBWrapper      [❌ 未编译]
  │           └─ EmbeddingKVDB (kv_db_table_batched_embeddings.cpp)
  │                ├─ RocksDB        SSD / 磁盘层         需链接 librocksdb
  │                ├─ l2_cache_      DRAM L2 (CacheLib)   [❌ cachelib_cache.cpp 未编译]
  │                └─ dram_kv        DRAM hot 层          [❌ dram_kv_embedding_cache 未编译, fbcode 重]
  │
  ├─ forward / backward GPU 计算 kernel (codegen)            [✅ 已编译进包]
  │
  └─ 权重搬运 ops: ssd_cache_populate_actions 等 5 个         [❌ 未编译]
```

**关键发现**: 所谓 "SSD" 实际是**基于 RocksDB(文件系统层)的 KV 存储**,通过 CPU 线程池异步读写、经 `cudaLaunchHostFunc` 与 GPU 流同步。整个仓库 `grep -rni "nvme|spdk|cufile|gpudirect"` **零命中**——**没有 GPU-initiated NVMe DMA / GPUDirect Storage**。开源和 Meta 内部用的是**同一套存储路径**。

---

## 3. 开源版本支持现状(分层)

| 组件 | 代码是否存在 | 开源是否编译 | 可用 |
|---|---|---|---|
| Codegen 生成的 SSD forward/backward kernel(纯 GPU 计算) | ✅ 真实实现 | ✅ 编译(tbe_sources.py 无条件纳入) | ⚠️ kernel 在二进制里,但无调用入口 |
| 通用 LRU cache kernels(`lru_cache_find/populate`) | ✅ | ✅ 编译 | ✅(非 SSD 专属) |
| **手写 SSD C++ 实现**(`EmbeddingRocksDB`、op 注册) | ✅ 真实实现 | ❌ **不编译** | ❌ |
| **SSD cache CUDA kernel**(`ssd_cache_populate_actions` 等) | ✅ 真实 kernel | ❌ **不编译** | ❌ |
| RocksDB / folly 存储后端 | ✅ 真实实现 | ❌ 无 `find_package` | ❌ |
| `kv_tensor_wrapper_cpu.cpp`(CPU fallback) | ✅ 存在 | — | ❌ 全是 `FBEXCEPTION("Not implemented")` 桩 |
| Python `SSDTableBatchedEmbeddingBags` | ✅ 完整类 | 随源打包 | ❌ import 后调用底层 C++ op 失败 |
| SSD 测试(`test/tbe/ssd/*.py`) | ✅ | — | ⏭️ `@unittest.skipIf(*running_in_oss)` 全跳过 |
| NVMe 直读路径 | ❌ **完全缺失** | — | ❌(根本不存在,非 fbcode 独占) |

---

## 4. 为什么开源"不支持"——三道闸

### 闸 1:源文件清单排除(最致命)

`FbgemmGpu.cmake` 的 `fbgemm_gpu_sources_cpu_static` / `fbgemm_gpu_sources_gpu_static` 列表里**完全没有** `src/ssd_split_embeddings_cache/` 目录的任何文件。在所有 `.cmake` / `CMakeLists.txt` / `setup.py` 中穷举搜索 `ssd_split`、`kv_db_cpp_utils`、`cachelib_cache`、`dram_kv` **全部零命中**。这才是根因——不是宏,不是 flag。

### 闸 2:build target 无 SSD 选项

`setup.py` 的 `--build-target` 只有 `none/default/genai/hstu`,没有一个会拉入 SSD 源文件。也没有 `FBGEMM_GPU_SSD` 之类的 CMake `option`(grep 零命中)。

### 闸 3:Python import guard + 测试 skip

`tbe/ssd/__init__.py` 注释明确写 `common/inference/training` "ship only with the heavy `:ssd_split_table_batched_embeddings_ops` target"——这个 `:ssd_..._ops` 是 **Meta 内部 buck2 target**,开源根本不用 buck2(仓库内无任何 `BUCK*` 文件)。`open_source` 硬编码为 `True`(`__init__.py:117-119`)触发 `running_in_oss`(`common.py:31-48`),所有 SSD 测试被 `@skipIf`(`kv_backend_test.py:56`)。

### 关于 torchrec 的引用

torchrec 依赖的是 **Meta 内部构建的 fbgemm_gpu wheel**(包含 `:ssd_split_table_batched_embeddings_ops` heavy target,拉了 RocksDB/folly 并编译了 `src/ssd_split_embeddings_cache/`)。同一份源码在 PyPI/开源构建里就缺这些算子——**同一份代码,两条构建链路,开源那条少了 SSD 的编译单元**。

---

## 5. 完整实现所需缺失内容(5 大类)

### 类 1:构建系统未编译(表层,必须先做)

3 个未编译目录共 **9 个 .cpp/.cu** 要加入 CMake:

| 文件 | 依赖 |
|---|---|
| `src/ssd_split_embeddings_cache/ssd_split_table_batched_embeddings.cpp`(op/class 注册) | rocksdb, folly, json |
| `src/ssd_split_embeddings_cache/kv_db_table_batched_embeddings.cpp` | rocksdb, folly |
| `src/ssd_split_embeddings_cache/kv_db_cuda_utils.cpp`、`ssd_scratch_pad_indices_queue.cpp`、`ssd_split_embeddings_cache_cuda.cu` | CUDA, folly |
| `src/ssd_split_embeddings_cache/kv_tensor_wrapper_cpu.cpp` | torch, json |
| `src/split_embeddings_cache/cachelib_cache.cpp`、`kv_db_cpp_utils.cpp` | cachelib, folly, glog |
| `src/dram_kv_embedding_cache/` 的可编译部分(见 §7) | folly |

**改法**:仿照 `fbgemm_gpu_tbe_cache` 模式新建 `cmake/TbeSsd.cmake`,定义 `gpu_cpp_library(fbgemm_gpu_tbe_ssd ...)`,在 `TbeTraining.cmake` 末尾 `include`,加入 `fbgemm_gpu_py`(`FbgemmGpu.cmake:198-207`)的 DEPS,在 `__init__.py:138-153` 的 `fbgemm_gpu_libraries` 加 `"fbgemm_gpu_tbe_ssd"`。

> `gpu_cpp_library` 函数定义在 `cmake/modules/GpuCppLibrary.cmake:143`,**不支持**自动链接 RocksDB/folly,但调用后 target 名会 export,可直接追加 `target_link_libraries(fbgemm_gpu_tbe_ssd PRIVATE ${ROCKSDB_LIBRARIES} ${FOLLY_LIBRARIES})`。

### 类 2:第三方依赖(全部开源可获取)

| 库 | 用途 | 获取方式 |
|---|---|---|
| **rocksdb** | SSD 持久化后端核心 | conda/vcpkg/apt/源码(facebook/rocksdb) |
| **folly** | 异步执行器、coro、F14Map、并发队列(**编译期硬依赖**) | conda/vcpkg/源码(facebook/folly) |
| **glog** | 日志断言 | apt/vcpkg |
| **nlohmann/json** | KV wrapper 序列化 | header-only |
| **CacheLib** | DRAM L2 cache | 开源(facebook/CacheLib),提供 `LruAllocator`/`CacheAllocator` |
| **Intel MKL**(仅 x86) | `kv_db_table_batched_embeddings.h:13` 条件引入 | apt/源码(可禁用) |

folly 传递依赖 gflags/double-conversion/libevent/openssl 等。建议用 conda 安装,避免从源码编译 folly 依赖链的痛苦。

### 类 3:fbcode 硬阻断 shim(机械替换)

见 §6 详表。

### 类 4:dram_kv_embedding_cache 模块(最大工作量 + 部分不可得)

见 §7 详述。

### 类 5:测试侧(最后跑通)

13 处 `@unittest.skipIf(*running_in_oss)` 要删(保留 `gpu_unavailable` 的 skip)。集中位置:`ssd_split_tbe_training_test.py:43`、`kv_backend_test.py:56`、`ssd_split_tbe_inference_test.py:43`、`ssd_tbe_training_adam_test.py:39` 等。测试用**进程内嵌入式 RocksDB**(写到 `tempfile` 临时目录),无需启动独立服务/端口。

---

## 6. fbcode 硬阻断点详细清单

### 6.1 `#include` 头文件硬阻断(无 ifdef 包裹)

| # | 文件:行号 | include 内容 | OSS shim 建议 |
|---|---|---|---|
| 1 | `ssd_split_embeddings_cache/kv_db_table_batched_embeddings.cpp:13` | `common/time/Time.h`(WallClockUtil) | 删除,改用 `std::chrono::steady_clock` |
| 2 | `ssd_split_embeddings_cache/kv_tensor_wrapper_cpu.cpp:14` | `common/base/Exception.h`(FBEXCEPTION) | 删除,改用 `TORCH_CHECK(false, ...)` |
| 3 | `split_embeddings_cache/kv_db_cpp_utils.cpp:11` | `common/base/Proc.h`(getCpuInfo) | 删除,改用 `std::thread::hardware_concurrency()` |

### 6.2 代码内裸调用(无 ifdef 包裹)

| # | 文件:行号 | 调用内容 | OSS shim 建议 |
|---|---|---|---|
| 4 | `kv_db_table_batched_embeddings.cpp` 全文 ~22 处(L173/176/444/454/477/481/492/499/505/517/527/530/542/552/607/623/629/649/701/714/745) | `facebook::WallClockUtil::NowInUsecFast()` | `std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()` |
| 5 | `ssd_table_batched_embeddings.h:990` | `facebook::strings::generateUUID()`(**裸调用,无 ifdef**) | `boost::uuids` / `stduuid` / `getpid()+时间戳` |
| 6 | `split_embeddings_cache/kv_db_cpp_utils.cpp:114` | `facebook::Proc::getCpuInfo().numCpuCores` | `std::thread::hardware_concurrency()` |

### 6.3 CacheLib 的 CacheAdmin(僵尸死代码,可剥离)

`cachelib_cache.h` include 了两个头:

```cpp
#include <cachelib/allocator/CacheAllocator.h>      // 开源 CacheLib 有 ✓
#include <cachelib/facebook/admin/CacheAdmin.h>      // 开源 CacheLib 没有 ✗
```

- 第一个(`CacheAllocator.h` + `LruAllocator`)在开源 [facebook/CacheLib](https://github.com/facebook/CacheLib) 仓库的 `cachelib/allocator/` 目录**完全开源**。
- 第二个(`CacheAdmin.h`)那个 `cachelib/facebook/admin/` 路径——开源 CacheLib 的 `cachelib/` 目录树**没有 `facebook/` 子目录**(只有 allocator/common/navy/compact_cache 等),是 fbcode 内部专有。

**但 `CacheAdmin` 在 FBGEMM 里是僵尸死代码**:`admin_` 成员**只在构造函数初始化列表赋值一次**(`cachelib_cache.cpp:24`),**整个 278 行文件里没有任何 `admin_->` 的使用**,`createCacheAdmin()`(L108-114)也**只在构造函数被调用一次**,无其他调用点。它的唯一内容是 `adminConfig.oncall = "mvai"`——Meta 内部 oncall 监控配置,纯运维用途。

**改造**:用 `#ifdef FBGEMM_FBCODE` 包住 4 处即可:

| 位置 | 内容 |
|---|---|
| `cachelib_cache.h:12` | `#include <cachelib/facebook/admin/CacheAdmin.h>` |
| `cachelib_cache.h:50` | `createCacheAdmin` 声明 |
| `cachelib_cache.h:154` | `admin_` 成员 |
| `cachelib_cache.cpp:24` + `L108-114` | 构造函数初始化 + 函数定义 |

### 6.4 已正确 ifdef 包裹的(非阻断,仅记录)

- `ssd_table_batched_embeddings.h:23-29` 的 5 个 fbcode include(UUID.h/Time.h/DBMonitor.h/FbRocksDb.h/FB303Stats.h)已包裹。
- `ssd_table_batched_embeddings.h:410-427/431-440/481-493/504-515` 的 `DBMonitorOptions`、`generateUUID`(L417)、`openRocksDB` 均已包裹。
- `raw_embedding_streamer.cpp` 全文(L9-440)所有 XLOG/stop_watch/coro/ServiceRouter/ODS 调用均已包裹。
- `XLOG` 来自 `<folly/logging/xlog.h>`,是 folly 的一部分,**OSS 完全可用**,不是阻断点。

---

## 7. dram_kv_embedding_cache 模块分析

### 7.1 模块定位

被 `ssd_split_table_batched_embeddings.cpp:16` 和 `kv_db_table_batched_embeddings.h:43` 引用。13 个文件,约 7300 行。**不依赖 cachelib、不依赖 rocksdb**——纯 DRAM 层,基于 `folly::F14FastMap` + 自研 `fixed_block_pool.h` 内存池 + `feature_evict.h` 淘汰策略。

### 7.2 两层划分

**纯净层(~2500 行,可直接 OSS 编译)**:

| 文件 | 行数 | 说明 |
|---|---|---|
| `feature_evict.h` | 1762 | 特征淘汰策略(LFU/LRU),纯算法,零 fbcode |
| `fixed_block_pool.h` | 437 | 固定块内存池,零 fbcode |
| `SynchronizedShardedMap.h` | 114 | 分片哈希表(folly F14FastMap) |
| `enrichment_config.h`、`inference_feature_evict.h`、`inference_fixed_block_pool.h`、`InferenceSynchronizedShardedMap.h`、`kv_inference_embedding_interface.h` | — | 依赖上述纯净层 |

**fbcode 重依赖层(~3500 行,部分不可得)**:

| 文件 | 行数 | 重依赖 |
|---|---|---|
| `dram_kv_embedding_cache.h` | 2506 | Thrift/ServiceRouter/fb303 + 4 个 fb-only enrichment 头 |
| `dram_kv_inference_embedding.h` | 966 | Thrift/ServiceRouter/fb303 |
| `dram_kv_embedding_cache_wrapper.h`、`dram_kv_embedding_inference_wrapper.cpp/.h` | — | 传递依赖上述 |

### 7.3 真正的硬天花板

`dram_kv_embedding_cache.h:33-36` 依赖 4 个 `deeplearning/fbgemm/fbgemm_gpu/fb/...` 路径的 enrichment 头:

| 头文件 |
|---|
| `fake_enrichment.h`、`feature_store_enrichment.h`、`igr_enrichment.h`、`oneflow_enrichment.h` |

这些**在开源仓库物理不存在**——是 Meta 内部特征处理管线源码。要么用 `#ifdef FBGEMM_FBCODE` 完全排除 enrichment 子系统(走无 enrichment 的纯 DRAM 缓存路径,需确认 wrapper 默认配置能绕过),要么放弃 dram_kv 的 enrichment 功能。

### 7.4 enrichment 功能详解(语义 / 配置 / 使用)

#### 7.4.1 enrichment 是什么

正常情况下,DRAM KV 缓存(或 SSD 后端)查不到某个 embedding ID 时(cache miss),只能返回**零向量或随机初始化**,对冷启动/未见过的特征不友好。

**enrichment 解决这个问题**:cache miss 时不直接返回零,而是**异步从一个外部模型服务(Laser / OpenTab / Feature Store)查询该 ID 的预训练 embedding,写回缓存**,后续命中即用真实值。本质是"用外部 retrieval 服务给冷特征做 warm-start"。

```
cache miss(ID 不在 DRAM/SSD)
   │
   ├─ 同步路径:返回零向量(训练继续,不阻塞)
   │
   └─ 异步 enrichment(enrichment_executor_, 4 线程,低优先级,不影响 forward/backward):
        fetch(ID) → prepare → set_kv_db_async_on_enrichment_executor() 写回 cache
        └─ 后续 iteration 命中 → 返回 enriched embedding
```

代码证据:
- 专用线程池 `dram_kv_embedding_cache.h:161-165`(`"Only created when enrichment is configured"`,`folly::CPUThreadPoolExecutor(4)`)。
- 异步写回 `set_kv_db_async_on_enrichment_executor`(`dram_kv_embedding_cache.h:572`),经 `enrichment_query_stream`(`training.py:1032`)。
- 3 个统计指标:`enrichment_query_count`(尝试)、`enrichment_empty_count`(空返回)、`success_rate_pct`(`training.py:208-216`)。

#### 7.4.2 5 种 EnrichmentType 与 fb 头的对应

| 值 | EnrichmentType | 外部后端 | 对应 fb 头 | 开源可用 |
|---|---|---|---|---|
| 0 | `IGR_LASER_EMBEDDING` | Meta Laser 向量检索 | `igr_enrichment.h` | ❌ fb/ 路径 |
| 1 | `IGR_LASER_SID` | Meta Laser(SID 模式) | `igr_enrichment.h` | ❌ |
| 2 | `ONEFLOW_OPENTAB_SID` | Meta OpenTab/Maple | `oneflow_enrichment.h` | ❌ |
| 3 | `ONEFLOW_FEATURE_STORE_SID` | Meta Feature Store | `feature_store_enrichment.h` | ❌ |
| 4 | `IN_MEMORY_TEST_ONLY` | 进程内确定性 fake(测试用) | `fake_enrichment.h` | ❌ 连此也在 fb/ 路径 |

`dram_kv_embedding_cache.h:33-36` 这 4 个头**全部**在 `deeplearning/fbgemm/fbgemm_gpu/fb/src/...` 路径下——包括"唯一看似可开源"的测试桩 `fake_enrichment.h`。所以 5 种 enrichment provider 实现**开源一个都拿不到**。

但**配置骨架是开源的**:`enrichment_config.h`(dram_kv 目录)定义了 `EnrichmentConfig` 这个 TorchScript custom class;Python 侧 `EnrichmentPolicy`(`ssd_config.py:228`)NamedTuple 也完整开源。**配置入口在,provider 实现不在。**

#### 7.4.3 用户配置链路

```
EnrichmentPolicy (NamedTuple)        # 用户在 Python 侧填
  → 放进 KVZCHParams.enrichment_policy 或 KVZCHTBEConfig.enrichment_policy
    → training.py 构造 torch.classes.fbgemm.EnrichmentConfig
      → 传给 DramKVEmbeddingCacheWrapper (C++ 入口)
```

**配置示例(Laser IGR 向量检索场景)**:

```python
from fbgemm_gpu.tbe.ssd.ssd_config import (
    EnrichmentType, EnrichmentResponseFormat, EnrichmentPolicy, KVZCHParams,
)

enrichment_policy = EnrichmentPolicy(
    enrichment_type=EnrichmentType.IGR_LASER_EMBEDDING,  # 选哪个外部服务
    provider_name="laser_prod",   # Laser provider 名
    client_id="my_rec_model",     # 客户端标识(服务端鉴权/计费)
    enrichment_dim=64,            # 外部服务返回的向量维度
    response_format=EnrichmentResponseFormat.JSON,
    laser_batch_size=0,           # 0=不批量,所有 miss ID 一个 RPC
)

kv_zch_params = KVZCHParams(
    bucket_offsets=..., bucket_sizes=...,
    enrichment_policy=enrichment_policy,
)
```

不同 `enrichment_type` 用不同的专用参数(`ssd_config.py:239-256`):

- **ONEFLOW_OPENTAB_SID**(类型 2):`opentab_tier_name`、`opentab_payload_ids="31739"`、`opentab_payload_types="2"`、`opentab_column_group_ids="12"` 等逗号分隔的 OpenTab schema。
- **ONEFLOW_FEATURE_STORE_SID**(类型 3):`fs_tier`、`fs_caller_id`、`fs_feature_group_id`、`fs_feature_name` 等 Feature Store 配置。
- **IGR_LASER_***(类型 0/1):`provider_name` + `client_id` + `laser_batch_size`,C++ 侧构造时预初始化 LaserClient(`dram_kv_embedding_cache.h:166-172`)。

#### 7.4.4 触发条件(3 个同时满足才启用)

代码在 `training.py:962-966`:

```python
if (
    self.kv_zch_params                                    # ① 用了 KVZCH
    and self.kv_zch_params.enrichment_policy is not None  # ② 显式配置
    and self._embedding_cache_mode                        # ③ embedding_cache_mode=True
):
    enrichment_config = torch.classes.fbgemm.EnrichmentConfig(...)
```

**不满足则 `enrichment_config=None`,C++ 侧不创建 enrichment_executor_,整条 enrichment 路径完全不激活。** 这就是 enrichment 是"可选增强"的根因。

#### 7.4.5 开源下的状态(对接"完整实现 SSD TBE")

1. **enrichment 不是 SSD 核心功能**——它是 KVZCH `embedding_cache_mode` 下的可选增强,默认关闭(`enrichment_policy=None`)。不配它,DRAM KV / SSD 训练推理完全正常,仅 cache miss 返回零向量。
2. **开源可完全排除**:用 `#ifdef FBGEMM_FBCODE` 包住 3 处即可剥离,不影响核心:
   - `dram_kv_embedding_cache.h:33-36` 的 4 个 fb include
   - `dram_kv_embedding_cache.h:161-180` 的客户端预初始化(LaserClient/OpenTabReader)
   - `dram_kv_embedding_cache.h:812-1046` 的 `dispatchEnrichmentAsync` / `fetch_sids_sync` 分发逻辑
   
   剥离后 `enrichment_config` 参数虽传但不消费,DRAM KV 缓存照常工作。
3. **若想开源用 enrichment**:5 种 provider 实现都在 `fb/` 路径拿不到。唯一出路是**自行实现一个 enrichment provider 接口**(参考 `fake_enrichment.h` 的契约:输入 unhashed IDs,输出 `optional<EnrichmentResult>`),接到 `dispatchEnrichmentAsync`(`dram_kv_embedding_cache.h:818`)的回调签名 `const PayloadMap&) -> optional<EnrichmentResult>`。工作量中等,但需对接自己的外部 retrieval 服务。
4. **测试场景**:即使开源编译通过,`IN_MEMORY_TEST_ONLY`(类型 4)的 `fake_enrichment.h` 也在 fb/ 路径——要用它做 hermetic 测试,得照 `enrichment_config.h:26-29` 的注释契约自行复刻一个"deterministic embedding from ID"的 fake。

> **小结**:enrichment = "cache miss 时从外部 retrieval 服务异步补 embedding 值并写回缓存"的 warm-start 增强。**可选、默认关闭、属 KVZCH 高级特性**;配置入口(`EnrichmentPolicy`)开源但 5 种 provider 实现全在 fb/ 路径。对"完整实现 SSD TBE 核心"而言,enrichment 可整体 ifdef 排除,不构成功能阻塞——这正是 §10 把它列为"部分不可得但核心不依赖"的依据。

---

## 8. CacheLib 与 cachelib_cache.h 使用分析

### 8.1 使用点

唯一引用:`kv_db_table_batched_embeddings.h:44` → `cachelib_cache.h`。作用是 SSD TBE 的 **L2 DRAM 缓存**(RocksDB 是磁盘层,CacheLib 是内存层)。

### 8.2 L2 cache 的可关闭性

`EmbeddingKVDB` 类持有 `l2_cache_` 成员(`kv_db_table_batched_embeddings.h:549`),在 `kv_db_table_batched_embeddings.cpp:99-108` 根据参数构造:

```cpp
if (cache_size_gb > 0) {
    l2_cache_ = std::make_unique<l2_cache::CacheLibCache>(cache_config, unique_id);
} else {
    l2_cache_ = nullptr;
}
```

所有 L2 cache 使用点都有 `if (l2_cache_)` 守卫(L178/207/385/549/641),`nullptr` 时直接跳过。所以 `cache_size_gb=0` 时 Cachelib 整条路径都不走。

### 8.3 实现文件状态

- `cachelib_cache.cpp`(278 行)完整存在于开源仓库。
- 但**未编译**:grep 所有 `.cmake`/`CMakeLists.txt`/`setup.py`,`cachelib_cache` 零命中。对照组 `raw_embedding_streamer.cpp` 在 `TbeInference.cmake:25`,说明同目录的 cachelib 是**被刻意漏掉**的。

### 8.4 结论

CacheLib 从一度被认为的"致命硬阻断"降级为"普通开源第三方库依赖":真正需要的是开源 CacheLib 的 `LruAllocator` + `CacheAllocator`(开源仓库有),`CacheAdmin` 是可剥离的死代码。

---

## 9. Python 侧 + Codegen 侧分析

### 9.1 codegen SSD kernel 已编译(关键好消息)

`tbe_sources.py` 的 `SSD_OPTIMIZERS = ["rowwise_adagrad"]`(L44)、`DENSE_OPTIONS` 含 `"ssd"`(L84)会生成 `gen_embedding_forward_ssd_*` / `gen_embedding_backward_ssd_*` 等文件,经 `TbeTraining.cmake` 的 `get_tbe_sources_list` **全部编译进** `fbgemm_gpu_tbe_training_forward/backward`。

这些是 **SSDTableBatchedEmbeddingBags 的前向/后向/优化器计算 kernel**——embedding lookup 的 GPU 计算、梯度计算、rowwise_adagrad 状态更新。**GPU 计算侧已就绪,缺的只是数据搬运层 + 存储后端。**

### 9.2 Python import guard 行为

`tbe/ssd/__init__.py` 用 4 个 try/except 块包裹 import。**关键**:import 时(模块加载阶段)通常不会失败;失败发生在**构造 TBE 实例时**——`SSDTableBatchedEmbeddingBags.__init__`(`training.py:837`)调用 `torch.classes.fbgemm.EmbeddingRocksDBWrapper(...)`,该 class 不存在时抛 `RuntimeError`(非 ModuleNotFoundError,不被 guard 捕获)。

**C++ 算子编译进包后,`__init__.py` 本身无需修改**(guard 会自动放行成功的 import)。

### 9.3 未编译的 torch.classes(8 个,全部定义在未编译目录)

| 类名 | 定义位置 |
|---|---|
| `EmbeddingRocksDBWrapper` | `src/ssd_split_embeddings_cache/` |
| `DramKVEmbeddingCacheWrapper` | `src/dram_kv_embedding_cache/` |
| `FeatureEvictConfig`、`EnrichmentConfig`、`KVTensorWrapper`、`SSDScratchPadIndicesQueue` | `src/ssd_split_embeddings_cache/` |
| `EmbeddingParameterServerWrapper`、`AtomicCounter` | `src/ps_split_embeddings_cache/`(PS 后端,可选) |

---

## 10. 可行性判断

### 10.1 能否构建出 Meta 内部等价的 whl?

**能,但不是"开箱即用",而是"中等工程量的改造"(2-3 周)。** 而且关键的好消息是——不存在 fbcode 独占的 NVMe 直读代码(开源和内部用同一套 RocksDB 路径)。所以本质是"补齐开源构建链 + 3 处 fbcode 头 shim + CacheLib",不是"拿到 Meta 的秘密代码"。

### 10.2 "完整实现"分两个层次

| 目标 | 可行性 | 工程量 |
|---|---|---|
| **SSD 核心训练/推理跑通**(RocksDB + CacheLib L2 + 无 enrichment 的 DRAM 缓存) | ✅ **可行** | 2-3 周 |
| **含 dram_kv enrichment 子系统 + PS parameter server** | ⚠️ **部分不可得** | 4 个 fb-only 头源码缺失 |

### 10.3 核心路径可行的依据

1. GPU 计算内核已编译(codegen)。
2. CacheLib、RocksDB、folly 全部开源。
3. fbcode 阻断是机械 shim 替换 + 僵尸死代码剥离。
4. enrichment 是可选增强,可 ifdef 排除。

**真正的硬天花板**只有 dram_kv 的 4 个 `fb/` enrichment 头——Meta 内部特征处理管线,开源仓库没有源码。但 SSD TBE 的核心功能不依赖它。

---

## 11. 分阶段实施计划

| 阶段 | 目标 | 工作量 | 风险 |
|---|---|---|---|
| **阶段 1** | 只编译 `ssd_scratch_pad_indices_queue.cpp` + `ssd_split_embeddings_cache_cuda.cu`(零 fbcode、纯 CUDA/folly),验证 CMake 链路通 | 1 天 | 低 |
| **阶段 2** | 加 3 个 fbcode shim,编译 `ssd_split_table_batched_embeddings.cpp` + `kv_db_cpp_utils.cpp`,让 `get_bucket_sorted_indices_and_bucket_tensor` 等 op 注册成功(issue #5666 目标) | 3-5 天 | 低 |
| **阶段 3** | 链接 librocksdb + folly,编译 `kv_db_table_batched_embeddings.cpp`,让 `EmbeddingRocksDBWrapper` class 可构造 | 1 周 | 中(rocksdb/folly ABI 匹配) |
| **阶段 4** | 编译 `cachelib_cache.cpp`(剥离 CacheAdmin),用开源 CacheLib | 3-5 天 | 低 |
| **阶段 5** | 先只编译 dram_kv 纯净层(feature_evict 等),enrichment 用 ifdef 排除 | 1-2 周 | 中 |

**最小验证路径**:阶段 1 的 CMake 改造最确定、最低风险,能立刻验证整个构建链是否通。

---

## 12. 关键文件路径速查表

### 未编译源文件(需加入 CMake)

```
src/ssd_split_embeddings_cache/
├── ssd_split_table_batched_embeddings.cpp   # op/class 注册(TORCH_LIBRARY_FRAGMENT)
├── ssd_table_batched_embeddings.h            # EmbeddingRocksDB 类(header-only, 含 fbcode 分支)
├── ssd_split_embeddings_cache_cuda.cu        # SSD cache CUDA kernel
├── ssd_scratch_pad_indices_queue.cpp         # 零 fbcode, 干净
├── kv_db_table_batched_embeddings.cpp        # EmbeddingKVDB(含 WallClockUtil 裸调用)
├── kv_db_table_batched_embeddings.h          # 传递引入 cachelib + rocksdb
├── kv_tensor_wrapper_cpu.cpp                 # CPU fallback(全是 stub)
├── kv_db_cuda_utils.cpp / .h                 # CUDA host func 同步
├── embedding_rocksdb_wrapper.h               # RocksDB wrapper
├── initializer.h                             # folly::USPSCQueue
└── kv_tensor_wrapper.h                       # 干净

src/split_embeddings_cache/
├── cachelib_cache.cpp                        # CacheLib L2(278 行, 完整开源实现)
├── kv_db_cpp_utils.cpp                       # bucket sort CPU 实现(含 Proc.h 阻断)
└── raw_embedding_streamer.cpp                # ✅ 已编译(TbeInference.cmake:25)

src/dram_kv_embedding_cache/                  # DRAM hot 层(13 文件, 约 7300 行)
├── feature_evict.h / fixed_block_pool.h / SynchronizedShardedMap.h  # 纯净层
└── dram_kv_embedding_cache.h / dram_kv_inference_embedding.h         # fbcode 重依赖层
```

### 构建系统关键文件

```
fbgemm_gpu/FbgemmGpu.cmake                    # fbgemm_gpu_py library 定义(L185-209)
fbgemm_gpu/CMakeLists.txt                     # BUILD_TARGET_VALUES(L26-29), target 分派(L284-315)
fbgemm_gpu/cmake/TbeTraining.cmake            # TBE training library 组织
fbgemm_gpu/cmake/tbe_sources.py               # codegen 文件清单(SSD_OPTIMIZERS L44)
cmake/modules/GpuCppLibrary.cmake             # gpu_cpp_library 函数定义(L143)
fbgemm_gpu/setup.py                           # cmake_args 注入(L265-404), --build-target(L56)
fbgemm_gpu/fbgemm_gpu/__init__.py             # fbgemm_gpu_libraries(L138-153), open_source(L117-119)
fbgemm_gpu/fbgemm_gpu/tbe/ssd/__init__.py     # import guard(L22-53)
```

### 第三方依赖

- [facebook/rocksdb](https://github.com/facebook/rocksdb) — SSD 持久化后端
- [facebook/folly](https://github.com/facebook/folly) — 异步/并发基础设施(编译期硬依赖)
- [facebook/CacheLib](https://github.com/facebook/CacheLib) — DRAM L2 cache
- glog、nlohmann/json、gflags、(x86) Intel MKL

---

## 13. 结论

**一句话总结**: SSD TBE 算子在仓库里**代码相当完整**(真实 RocksDB + folly + CacheLib 实现 + codegen GPU kernel),但被开源构建系统**结构性禁用**——手写 SSD 源文件未进 CMake 编译列表 → `TORCH_LIBRARY_FRAGMENT` 注册不进二进制 → Python 调用报 `AttributeError`。

要完整实现 SSD TBE:
1. **缺的不是算法秘密**,而是构建链补全(CMake target) + 第三方依赖(rocksdb/folly/cachelib) + fbcode 头 shim(6 处)。
2. **真正的硬天花板**是 dram_kv 的 4 个 `fb/` enrichment 头(开源仓库物理不存在),但 SSD 核心功能不依赖它。
3. **核心训练/推理路径可行**,工程量约 2-3 周,分 5 个阶段递进验证。

对**昇腾 950 / CANN 移植**的建议:SSD 这条路径重依赖 RocksDB + folly + `cudaLaunchHostFunc` 流同步,移植成本远高于 MANAGED_CACHING(后者只需 `aclrtMallocManaged` + `aclrtMemcpyAsync`)。SSD 应作为昇腾移植的 P2/P3,而非 P0。

---

*本报告基于 FBGEMM v1.5.0-release 分支源码静态分析,所有结论附文件路径与行号证据。第三方库开源状态已通过 [github.com/facebook/CacheLib](https://github.com/facebook/CacheLib) 目录树核实。*
