# FBGEMM 中 UVM (Unified Virtual Memory) 分析报告

> **报告范围**: 全面解析 FBGEMM (尤其 FBGEMM_GPU) 中 UVM 的作用、涉及的算子/接口、业务背景与工程权衡。
> **报告时间**: 2026-06-08
> **基于仓库版本**: 当前 v1.5.0-release 分支（FBGEMM_GPU 1.5.0 / FBGEMM 1.4.0）
> **核心结论**: FBGEMM 通过 `cudaMallocManaged` + `cudaMemAdvise` + `cudaMemPrefetchAsync` 三件套，把"超大规模稀疏 embedding 表 + 长尾访问"这一业务难题包装成 `EmbeddingLocation.MANAGED` / `MANAGED_CACHING` 两个开箱即用开关。

---

## 目录

1. [UVM 的作用（基础原理）](#1-uvm-的作用基础原理)
2. [FBGEMM 中的 UVM 实现层（cumem_utils 工具）](#2-fbgemm-中的-uvm-实现层cumem_utils-工具)
3. [FBGEMM 使用 UVM 的算子/接口](#3-fbgemm-使用-uvm-的算子接口)
4. [业务场景与触发条件](#4-业务场景与触发条件)
5. [三种存储方案对比（HBM / Host + memcpy / UVM）](#5-三种存储方案对比hbm--host--memcpy--uvm)
6. [性能调优与最佳实践](#6-性能调优与最佳实践)
7. [GenAI / 大模型场景下 UVM 的角色](#7-genai--大模型场景下-uvm-的角色)
8. [总结与决策建议](#8-总结与决策建议)
9. [附：关键源码/文件路径速查](#9-附关键源码文件路径速查)

---

## 1. UVM 的作用（基础原理）

### 1.1 一句话定义

> **UVM (Unified Virtual Memory)，CUDA 中又称 "managed memory"**：一份虚拟地址空间同时被 CPU 与所有可见 GPU 共享，由 CUDA 驱动 + Linux 内核 HMM (Heterogeneous Memory Management) 子系统联合管理，以 **4 KB / 2 MB 页**为粒度在 CPU DRAM、GPU HBM 之间**按需自动迁移**。

**关键点**：UVM **不是零拷贝**（与早期 `cudaHostAlloc(..., cudaHostAllocMapped)` 不同），它会**真的在 PCIe / NVLink 上搬运页面**；只是把搬运细节对程序员隐藏。

### 1.2 UVM 与传统 `cudaMalloc` 对比

| 特性 | `cudaMalloc` (传统) | `cudaMallocManaged` (UVM) |
|---|---|---|
| 可见性 | 仅当前 device | **所有** CPU + 可见 GPU |
| 数据搬运 | 程序员手写 `cudaMemcpy` | **驱动**按 page fault 触发迁移 |
| 编程模型 | 显式数据搬运 | 隐式（page fault 自动处理） |
| 默认性能 | 稳定可预测 | 未调优时波动大（fault 抖动） |
| 适用规模 | 中小张量（受 HBM 容量限制） | 大到无法装入 HBM / 稀疏访问 |

### 1.3 关键 API 三件套

FBGEMM 全部用 **runtime API**（**未使用** `__managed__` 编译期变量、`cuMemAllocManaged` driver API、`cudaStreamAttachMemAsync`）：

| API | 作用 | FBGEMM 是否使用 |
|---|---|---|
| `cudaMallocManaged(&ptr, size)` | 分配 UVM 内存 | ✅ 核心入口 |
| `cudaMemAdvise(ptr, ..., SetPreferredLocation, dev)` | 声明"数据归属哪个 device/host" | ✅ 分配后立即调用 |
| `cudaMemAdvise(ptr, ..., SetAccessedBy, dev)` | 声明"某 device 即将访问"，建立 direct mapping **避免 page fault** | ✅ 分配后立即调用 |
| `cudaMemAdvise(ptr, ..., SetReadMostly)` | 标记只读热点，pin 在某 device 上 | ⚠️ 暴露给 Python，未默认使用 |
| `cudaMemPrefetchAsync(ptr, ..., dstDev, stream)` | **主动**把页搬到目标 device | ✅ 暴露给 Python (`uvm_cuda_mem_prefetch_async`) |
| `madvise(MADV_DONTFORK)` | Workaround fork 场景下 UVM 的问题 | ✅ `uvm_mem_advice_dont_fork` |
| `cudaMemAttachGlobal` / `cudaMemAttachHost` | 限制 UVM 可见范围 | ❌ 未使用（默认 Global） |
| `__managed__` 编译期变量 | 编译期声明 UVM 变量 | ❌ 未使用 |

#### 1.3.1 `cudaMemAdvise` 的本质：**性能优化提示（hint），不是访问控制**

> ⚠️ **这是最容易被误解的语义**。在 FBGEMM 文档和很多社区文章中，"把 UVM 内存 advise 给 X 卡"的说法容易让读者以为 X 卡获得了某种"专属访问权"。**这是错的**。

| 误解 | 真相 |
|---|---|
| "advise 给卡 0 之后，卡 1 就不能访问了" | ❌ 错。卡 1 **仍能访问**，只是访问时**没有享受任何性能优化** |
| "advise 是访问控制" | ❌ 错。UVM 的访问控制由 **OS 页表** 负责（read/write/exec 权限），与 advise 无关 |
| "不 advise 给卡 1，卡 1 访问会失败" | ❌ 不会失败，但首次访问会触发 **page fault**（几十~几百 μs 抖动） |
| "advise 是声明给 driver 的 hint" | ✅ **正确**。告诉 driver "这片内存将来被谁用，请按这个意图优化内存布局" |

**6 种 advice 的本质语义分类**：

| Advice | 实际语义 | driver 收到 hint 后会做什么 |
|---|---|---|
| `SetPreferredLocation` | "这片内存的**家**在 X"（X 可以是 device 或 host） | page fault 时**优先把页迁回 X**；避免乱迁移 |
| `UnsetPreferredLocation` | 解除上一条 | — |
| `SetAccessedBy` | "X device 将来要访问它" | 预先在 X device 上建立 **direct mapping**，访问时**不触发 page fault** |
| `UnsetAccessedBy` | 解除上一条 | — |
| `SetReadMostly` | "只读热点" | 在多 device 场景下**复制**到 X device 的 cache，避免反复迁移 |
| `UnsetReadMostly` | 解除上一条 | — |

**类比**：
- `SetPreferredLocation` ≈ "建议仓库把货放在 A 货架"（管理建议）
- `SetAccessedBy` ≈ "建议给 B 工人办一张 A 货架通行证"（让 B 来去自如）
- 但 C 工人**理论上仍能进仓库**——只是门卫要现场查证件、还要临时给他办手续（page fault + migrate）

#### 1.3.2 FBGEMM 单卡边界的真正成因（不是被技术阻止，是被设计选择）

```cpp
// memory_utils.cu:202-227
gpuMemAdvise(ptr, size_bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
gpuMemAdvise(ptr, size_bytes, cudaMemAdviseSetAccessedBy, current_device());
//                                                  ^^^^^^^^^^^^^^^
//                                                  永远是本进程当前那张卡
```

**事实**：
- FBGEMM 只 `SetAccessedBy` 给本进程当前那张卡
- **理论可能性**（FBGEMM 没用）：可以循环 N 次 `SetAccessedBy(card_0), SetAccessedBy(card_1), ...` 让同节点所有卡都建立 direct mapping
- **结果**：
  - 本卡访问：零 page fault（享受 advise 优化）
  - 其他卡访问：每次 fault（未享受 advise 优化）

> 💡 **关键洞察**：FBGEMM UVM 的"单卡"边界**不是被 CUDA 技术阻止**的，而是**被 FBGEMM 主动设计**出来的——FBGEMM 主动选择只 advise 给本卡。如果需要多卡共享一片 UVM 内存，代码上完全可以做到，只是 FBGEMM 选择了不这么做（这与"FBGEMM 是单 rank 算子库"的设计定位一致）。

#### 1.3.3 跨卡访问的合法路径（既然 advise 不是访问控制，那跨卡访问怎么办？）

由于 advise **不阻止**其他卡访问，技术上可以走两条路：

| 路径 | 机制 | 性能 | FBGEMM 用法 |
|---|---|---|---|
| **A. 让其他卡也走 UVM 路径** | 在其他卡进程中也 `SetAccessedBy` 同一片 UVM 内存 + 接受 page fault 代价 | 差（每次 fault） | ❌ 不用 |
| **B. 显式 P2P `cudaMemcpy`** | `cudaMemcpy(device_to_device)` 配合 P2P 启用（`cudaDeviceEnablePeerAccess`） | 好（同节点 NVLink 直连） | ❌ 不用 |
| **C. NCCL collective** | `torch.distributed.all_to_all_single` 等 | 优（NVLink / IB 优化路径） | ✅ 实际做法（**通过 TorchRec 调用**） |

FBGEMM 选择路径 C（实际是 TorchRec 在调），把跨卡/跨节点通信完全交给上层框架。UVM 在 FBGEMM 中只负责"**本卡 HBM 装不下时**的容量扩展"，**不**参与跨卡协调。

### 1.4 硬件代价（理解 UVM 为何需要调优）

- **PCIe Gen4 ×16 ≈ 32 GB/s**；**PCIe Gen5 ×16 ≈ 64 GB/s**
- **NVLink 3.0 ≈ 600 GB/s**（GPU-GPU）；**NVLink 4 ≈ 900 GB/s**
- **单次 page fault 成本**：Pascal+ 上**几十到几百 μs/页**；大量 fault 直接毁掉 kernel
- **未调优的 UVM vs 显式 `cudaMemcpy`**：0.3~0.7× 性能（多卡可掉到 0.1~0.3×）
- **调优后 + NVLink 上**：0.85~1.0× 显式性能
- **Linux THP / 2 MB huge page**：能进一步减少 30~50% fault 开销

---

## 2. FBGEMM 中的 UVM 实现层（cumem_utils 工具）

FBGEMM_GPU 有一个**独立的 UVM 工具子模块** `cumem_utils`，作为 TBE 等上层算子与 CUDA 驱动之间的**唯一** UVM 抽象层。

### 2.1 模块结构

| 文件 | 角色 |
|---|---|
| `fbgemm_gpu/include/fbgemm_gpu/cumem_utils.h` | 公共 API 头文件（Doxygen 文档） |
| `fbgemm_gpu/src/memory_utils/memory_utils.cu` | **核心实现**：`cudaMallocManaged` + `cudaMemAdvise` |
| `fbgemm_gpu/src/memory_utils/memory_utils.cpp` | Meta/CPU dispatch 占位 |
| `fbgemm_gpu/src/memory_utils/memory_utils_ops.cpp` | TorchScript op 注册（CPU 端） |
| `fbgemm_gpu/src/memory_utils/memory_utils_ops.cu` | TorchScript op 注册（CUDA 端） |
| `fbgemm_gpu/src/memory_utils/memory_utils_fallback.cpp` | 非 GPU 构建的回退 |
| `fbgemm_gpu/fbgemm_gpu/uvm.py` | Python 用户态封装（`cudaMemAdvise` / `cudaMemPrefetchAsync`） |

### 2.2 公共 API

| Python/C++ 接口 | 作用 | 关键文件 |
|---|---|---|
| `new_managed_tensor(self, sizes)` | 分配一个 UVM tensor，**分配后立即**调 `MemAdvise(SetPreferredLocation=CPU, SetAccessedBy=GPU)` | `memory_utils.cu:192-238` |
| `new_vanilla_managed_tensor(self, sizes)` | 分配 UVM tensor 但不调 advise（"vanilla" 路径） | `memory_utils.cu:240-248` |
| `new_host_mapped_tensor(self, sizes)` | 走 `cudaHostRegister` 路径（host-mapped，零拷贝） | `memory_utils.cu:250-293` |
| `new_unified_tensor(self, sizes, is_host_mapped)` | 总入口，根据 `is_host_mapped` 分流到上述两条路径 | `memory_utils.cu:295-306` |
| `uvm_storage(t)` / `is_uvm_tensor(t)` | 通过 deleter 类型判断 tensor 是否 UVM | `memory_utils.cu:308-319` |
| `uvm_to_cpu(t)` / `uvm_to_device(t)` | 把 UVM storage **重新解释**为指定 device 的 tensor（**不复制数据**） | `memory_utils.cu:321-373` |
| `uvm_cuda_mem_advise(t, advice)` | 调 `cudaMemAdvise` | `memory_utils.cu:397-426` |
| `uvm_cuda_mem_prefetch_async(t, device_t)` | 调 `cudaMemPrefetchAsync` | `memory_utils.cu:428-464` |
| `uvm_mem_advice_dont_fork(t)` | `madvise(MADV_DONTFORK)` workaround | `memory_utils.cu:466-486` |
| `uvm_to_cpu_clone(t)` | `memcpy` 到新 CPU tensor（真正的复制） | `memory_utils.cu:488-500` |

### 2.3 核心实现摘录（`memory_utils.cu:192-238`）

```cpp
// new_managed_tensor: 分配 UVM 后立即做"两步 advise"，
// 避免运行时 page fault
AT_CUDA_CHECK(cudaMallocManaged(&ptr, size_bytes));   // ① 分配

// ② 把"数据归属"声明在 CPU DRAM
gpuMemAdvise(ptr, size_bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);

// ③ 告诉 GPU"你将来要访问它，请建立 direct mapping"
gpuMemAdvise(ptr, size_bytes, cudaMemAdviseSetAccessedBy, current_device);
//    ↑ 关键：建立 direct mapping 后, GPU 访问不会触发 page fault

// ④ Workaround fork 后 cuda context 不匹配的问题
madvise(ptr, size_bytes, MADV_DONTFORK);
```

源码中的注释原文（`memory_utils.cu:212-227`）：

> *"Set preferred memory location to host memory"*
> *"User hints with 'accessed by': GPU will establish direct mapping of data in CPU memory, **no page faults will be generated**"*

这两行 `SetPreferredLocation=CPU` + `SetAccessedBy=GPU` 的组合，是 FBGEMM 性能稳定的**关键**——单纯 `cudaMallocManaged` 不带 advise 会导致运行时大规模 page fault。

### 2.4 Python 用户态封装（`fbgemm_gpu/uvm.py`）

```python
# fbgemm_gpu/uvm.py
from torch.ops.fbgemm import (
    new_managed_tensor, new_unified_tensor, new_vanilla_managed_tensor,
    new_host_mapped_tensor, cuda_mem_advise, cuda_mem_prefetch_async,
    uvm_mem_advice_dont_fork, uvm_to_cpu, uvm_to_device,
    is_uvm_tensor, uvm_storage,
)
from torch.ops.fbgemm import fbgemm_gpu_uvm_enum_query
create_enums(globals(), fbgemm_gpu_uvm_enum_query)  # 6 个 cudaMemoryAdvise 枚举

def cudaMemAdvise(t, advice):
    return torch.ops.fbgemm.cuda_mem_advise(t, advice.value)

def cudaMemPrefetchAsync(t, device_t=None):
    return torch.ops.fbgemm.cuda_mem_prefetch_async(t, device_t)
```

### 2.5 兼容性处理

`memory_utils.cu` 显式适配 **CUDA 13 / ROCm 7** 的新 `cudaMemLocation` 签名：

- `#define gpuMemAdvise` 宏适配 `cudaMemAdvise_v2` / `hipMemAdvise_v2`
- `new_mem_location_from_device` / `new_mem_location_cpu` 辅助函数

注释（`memory_utils.cu:414`）提到：
> *"FIXME: some advanced cudaMemAdvise flags are not supported by HIP"*

即 FBGEMM 在 ROCm 上对部分 advanced advise flag 提供的是 best-effort 支持。

### 2.6 构建系统

- `fbgemm_gpu/FbgemmGpu.cmake` 第 27-28、78-79 行：把 `memory_utils.cpp` / `memory_utils.cu` / `memory_utils_ops.cpp` / `memory_utils_ops.cu` 加入 `gpu_cpp_library`
- **无独立的 UVM 编译开关**——UVM 是 TBE 的基础能力，**始终编译进去**
- `setup.py` 和 `fbgemm_gpu/cmake/*.cmake` 中**无 UVM 相关编译选项**

---

## 3. FBGEMM 使用 UVM 的算子/接口

FBGEMM 中 **UVM 的使用 100% 集中在 TBE (Table Batched Embeddings) 模块及其衍生算子**。其他模块（quantize_ops、jagged_tensor_ops、gen_ai、hstu 等）**完全不用 UVM**。

### 3.1 顶层分类

| 算子类别 | 是否用 UVM | UVM 角色 |
|---|---|---|
| **TBE 训练** (`SplitTableBatchedEmbeddingsTraining`) | ✅ 大量 | 存放超大 embedding 表 + 优化器 state |
| **TBE 推理** (`IntNBitTableBatchedEmbeddings`) | ✅ 大量 | 存放 INT4/INT2/FP8 量化表 |
| **SSD TBE 训练/推理** | ✅ 大量 | embedding 存 NVMe，UVM 作为 DRAM 层 |
| **KV 缓存推理** (`kv_embedding_ops_inference`) | ✅ | DRAM 级 embedding cache |
| **LXU / LRU 缓存** (`lxu_cache.cu`、`lru_cache_populate.cu`) | ✅ | UVM 表 + HBM cache + 6 项 cache stats |
| **Embedding Inplace Update** | ✅ | 直接写回 UVM 权重 |
| **Raw Embedding Streamer / SSD embedding streamer** | ✅ | 处理 UVM tensor 上的 `count` 等小标量 |
| **sparse_ops**（`int_nbit_split_embedding_codegen_lookup_function`） | ⚠️ 透传 | 接收 `uvm_weights` 形参 |
| **faster_hash / zero_collision_hash**（zch） | ⚠️ 透传 | 测试用例 `test_zch_output_on_uvm` 把输出写到 UVM |
| **quantize_ops** | ❌ | 不使用 |
| **jagged_tensor_ops** | ❌ | 不使用 |
| **experimental/gen_ai**（Llama 3/4） | ❌ | 不使用，靠量化 + TP/PP |
| **experimental/hstu** | ❌ | 不使用 |

### 3.2 TBE 算子家族（最核心）

#### 3.2.1 `EmbeddingLocation` 枚举（4 种 placement）

`fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_common.py:29-43`：

```python
class EmbeddingLocation(IntEnum):
    DEVICE = 0           # GPU HBM
    MANAGED = 1          # ← UVM（GPU/CPU 共享）
    MANAGED_CACHING = 2  # ← UVM + HBM 当 cache
    HOST = 3             # CPU DRAM
    MTIA = 4             # 异构加速器（实验性）
```

`fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_inference.py:201-214` 的 docstring 原文定义：

```
(1) DEVICE        = placing an embedding table in the GPU global memory (HBM)
(2) MANAGED       = placing an embedding in the unified virtual memory
                    (accessible from both GPU and CPU)
(3) MANAGED_CACHING = placing an embedding table in the unified virtual memory
                    and using the GPU global memory (HBM) as a cache
(4) HOST          = placing an embedding table in the CPU memory (DRAM)
```

#### 3.2.1.1 MANAGED 与 MANAGED_CACHING 的物理存储位置（关键澄清，2026-06-10 增补）

> **docstring 容易让人误解的一句话**：
> > `MANAGED = placing an embedding in the unified virtual memory (accessible from both GPU and CPU)`
>
> "accessible from both GPU and CPU" 指的是**虚拟地址层面的可访问性**（同一个指针，CPU 和 GPU 都能解引用），**不是说"数据物理上同时存在于 HBM 和 CPU"**。UVM 的本质是：每个 page 任意时刻只能在一个物理位置（CPU DRAM 或 GPU HBM），driver/HMM 根据访问模式自动迁移。

| 维度 | MANAGED | MANAGED_CACHING |
|---|---|---|
| `weights_uvm`（UVM 主存池） | ✅ 100% 权重都在这 | ✅ 100% 权重都在这 |
| `lxu_cache_weights`（FBGEMM 显式 HBM cache 池） | ❌ **不分配**（`numel() == 0`） | ✅ **显式分配**（`cache_load_factor × total_rows × D`） |
| HBM 中**是否有数据** | ✅ 有（driver 自动迁移的 hot page） | ✅ 有（FBGEMM 显式 cache 拷贝） |
| HBM 占用**是否可调** | ❌ 不可调（driver 启发式决定） | ✅ `cache_load_factor`（默认 0.2）直接控制 |
| 热点由谁管理 | CUDA driver + Linux HMM（启发式） | FBGEMM LRU/LFU/Direct-mapped kernel（精确） |
| `SetAccessedBy(GPU)` 角色 | **关键**——预装 direct mapping 避免首次 fault | 不重要（kernel 自己会处理 miss） |
| forward kernel 路径 | `use_lxu_cache = false`（直读 `weights_uvm`） | `use_lxu_cache = true`（先查 cache，再 fallback 到 `weights_uvm`） |

**MANAGED 模式的物理位置演化**：

```
        分配完成                  GPU 反复访问 hot page 后
        ┌─────────────┐           ┌─────────────┐
        │  CPU DRAM   │           │  CPU DRAM   │  ← 冷 page
        │  (100% 权重) │           │  (冷 page)   │
        └─────────────┘           └─────────────┘
                                  ┌─────────────┐
                                  │  GPU HBM    │  ← driver 隐式迁过来的 hot page
                                  │ (working set)│
                                  └─────────────┘
```

- `cudaMallocManaged` 初始分配 100% 在 CPU DRAM
- `SetPreferredLocation(CPU)` 是 **hint**（不是强约束），driver 可能把 hot page 迁到 HBM
- `SetAccessedBy(GPU)` 预装 direct mapping，**首次访问不 fault**，但走 PCIe（~32 GB/s）读 CPU 数据；之后 driver 决定是否迁到 HBM
- HBM 中**也会有数据**——只是 FBGEMM **不能控制**哪部分进 HBM、进多少、何时进出

**MANAGED_CACHING 模式的物理位置**：

```
        ┌─────────────┐
        │  CPU DRAM   │  ← 80% 权重（UVM 主存）
        │  (uvm)       │
        └─────────────┘
        ┌─────────────┐
        │  GPU HBM    │  ← 20% 权重（FBGEMM 显式 cache）
        │  (lxu_cache) │     由 LRU/LFU kernel 精确管理
        └─────────────┘
```

- FBGEMM 显式分配**两个池**：`weights_uvm`（UVM 存主存）+ `lxu_cache_weights`（HBM 存 cache）
- kernel **先查 cache**：命中走 HBM（~2-3 TB/s）；未命中走 UVM（可能 PCIe 也可能 HBM，取决于 driver 状态）
- `cache_load_factor`（默认 0.2）直接控制 HBM cache 大小

**两者的本质区别不是"用不用 HBM"，而是"由谁管 HBM 中的热点子集"**：
- MANAGED = **driver 管**热点（你不能调）
- MANAGED_CACHING = **FBGEMM 管**热点（你可以通过 `cache_load_factor`、`CacheAlgorithm.LRU/LFU/Direct-mapped` 调）

源码佐证：
- `lxu_cache_weights` 分配条件：[`split_table_batched_embeddings_ops_training.py:3541-3580, 3602-3630`](fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_training.py#L3541)（仅在 `cache_load_factor > 0` 即 `MANAGED_CACHING` 模式下分配）
- forward kernel 是否用 cache：[`embedding_forward_split_template.cu:602-605`](fbgemm_gpu/codegen/training/forward/embedding_forward_split_template.cu#L602-L605)（`use_lxu_cache = lxu_cache_weights.numel() > 0`）
- `SetAccessedBy` 的语义：[`memory_utils.cu:212-227`](fbgemm_gpu/src/memory_utils/memory_utils.cu#L212-L227)（"no page faults will be generated"）

#### 3.2.2 `MANAGED_CACHING` 的关键参数

- **`cache_load_factor`**（默认 `0.2`）：HBM cache 大小 = `cache_load_factor` × 所有表行总数
- **`uvm_size`**：所有 MANAGED / MANAGED_CACHING 表占用的总字节数（写入 `SplitState`）
- **`uvm_host_mapped`**：`True` 走 `cudaHostAlloc`（零拷贝），`False` 走 `cudaMallocManaged`（按需分页）
- **`gather_uvm_cache_stats`**：是否收集 6 项 cache 命中率统计

#### 3.2.3 UVMCacheStatsIndex（6 项统计指标）

`fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_training.py:184-190`：

```python
class UVMCacheStatsIndex(enum.IntEnum):
    num_calls = 0                 # cache 访问次数
    num_requested_indices = 1     # 请求的行数（含重复）
    num_unique_indices = 2        # 去重后行数
    num_unique_misses = 3         # 唯一缺失行数（要 page fault）
    num_conflict_unique_misses = 4  # 唯一缺失的 hash 冲突
    num_conflict_misses = 5       # 冲突未命中
```

通过 `emb.print_uvm_cache_stats()` / `emb.get_uvm_cache_stats()` / `emb.reset_uvm_cache_stats()` 读取。

### 3.3 算子-文件对照表

| 算子 / 接口 | 文件 | UVM 形参 / 缓冲 |
|---|---|---|
| `SplitTableBatchedEmbeddingsTraining` (构造) | `fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_training.py:685-` | `weights_uvm` / `momentum1_uvm` / `momentum2_uvm` / `prev_iter_uvm` / `row_counter_uvm` |
| 训练 TBE 分配 UVM | `…_ops_training.py:357-401`（`apply_split_helper` 中 `if split.uvm_size > 0`） | 调 `torch.ops.fbgemm.new_managed_tensor` / `new_unified_tensor` |
| `IntNBitTableBatchedEmbeddings` (构造) | `fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_inference.py:374` | `uvm_host_mapped: bool = False` |
| 推理 TBE 权重缓冲 | `…_ops_inference.py:530-532` | `self.weights_uvm: torch.Tensor = torch.empty(0, …)` |
| 推理 TBE 取权重 | `…_ops_inference.py:1625` | `else: weights = self.weights_uvm` |
| `print_uvm_cache_stats` | `…_ops_inference.py:717-739` | 打印 6 项统计 |
| UVM 内存报告 | `…_ops_training.py:1829-1869`（`_report_uvm_breakdown`） | 发送 `tbe.uvm.{embeddings, optimizer_states, cache, total_static_sparse, ephemeral}` 事件 |
| SSD TBE 训练 | `fbgemm_gpu/fbgemm_gpu/tbe/ssd/training.py:173-175` | `uvm_host_mapped: bool = False` 形参 |
| SSD TBE 推理 | `fbgemm_gpu/fbgemm_gpu/tbe/ssd/inference.py:327` | `"weights_uvm"` state dict key |
| KV 缓存推理 | `fbgemm_gpu/fbgemm_gpu/tbe/cache/kv_embedding_ops_inference.py:75` | `uvm_host_mapped: bool = False` 形参 |
| `lru_cache_populate(_byte)` | `fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate(_byte).cu:274-292, 506-521` | `std::optional<Tensor> uvm_cache_stats` |
| `lxu_cache_flush` | `fbgemm_gpu/src/split_embeddings_cache/lxu_cache.cu:84-145` | `Tensor uvm_weights`（把 LXU cache 写回 UVM） |
| `lxu_cache_lookup/populate` | `fbgemm_gpu/src/split_embeddings_cache/lxu_cache.cu:404-501` | `std::optional<Tensor> uvm_cache_stats` |
| `reset_weight_momentum` | `fbgemm_gpu/src/split_embeddings_cache/reset_weight_momentum.cu:171-173` | `weights = &uvm_weights[…]` + `MANAGED_CACHING` 分支 |
| `embedding_inplace_update` | `fbgemm_gpu/src/embedding_inplace_ops/embedding_inplace_update.cu:93` | `weights_placement == PlacementType::MANAGED_CACHING` 时直接写 UVM |
| `raw_embedding_streamer` | `fbgemm_gpu/src/split_embeddings_cache/raw_embedding_streamer.cpp:21-28` | `get_maybe_uvm_scalar()`（绕过 `.item()` 限制） |
| `kv_db_table_batched_embeddings` (SSD) | `fbgemm_gpu/src/ssd_split_embeddings_cache/kv_db_table_batched_embeddings.cpp:21-28` | 同上 |
| Codegen 模板（forward） | `fbgemm_gpu/codegen/training/forward/embedding_forward_split_kernel*.cu` | 形参 `weights_uvm` / `uvm_cache_stats` |
| Codegen 模板（backward） | `fbgemm_gpu/codegen/training/backward/embedding_backward_split_*.cu/.cpp/.cuh` | 同上 + `MANAGED_CACHING` 分支 |
| Codegen 模板（optimizer） | `fbgemm_gpu/codegen/training/optimizer/embedding_optimizer_split_*.cu/.cpp/.cuh` | `TENSOR_ON_CUDA_GPU({{ tensor }}_uvm)` 等 |
| Codegen 模板（PT2） | `fbgemm_gpu/codegen/training/pt2/embedding_split_host_pt2_autograd_template.cpp` | `weights_uvm` 形参 |
| `int_nbit_split_embedding_codegen_lookup_function` | `fbgemm_gpu/fbgemm_gpu/sparse_ops.py:427` | `uvm_weights: torch.Tensor` 形参 |

### 3.4 代码生成器（Codegen）中的 UVM

`fbgemm_gpu/codegen/genscript/optimizer_args.py` 定义的所有 UVM 形参：

```python
"weights_uvm":     "(c!)",   # 权重 UVM tensor
"uvm_cache_stats": "(f!)",   # 6 项 cache 命中率
"momentum1_uvm":   "(i!)",   # 一阶动量
"momentum2_uvm":   "(l!)",   # 二阶动量
"prev_iter_uvm":   "(o!)",   # 上一轮迭代（rowwise adagrad）
"row_counter_uvm": "(r!)",   # 行计数器
```

Jinja 模板（`embedding_forward_split_kernel_v2_template.cu:282`）注释明确：

> *"If UVM cache is used, fall back to the generic function"*

`embedding_backward_split_indice_weights_template.cu:204, 219, 354` 中 `placement == PlacementType::MANAGED_CACHING` 分支控制从 HBM 还是从 UVM 读取权重：

```cpp
if (placement == PlacementType::MANAGED_CACHING) {
    // 走 cache 路径
} else if (placement == PlacementType::MANAGED) {
    // 走 UVM 直读路径
}
```

### 3.5 UVM tensor 的 `.item()` 限制

`raw_embedding_streamer.cpp:21-28` 和 `kv_db_table_batched_embeddings.cpp:21-28` 中都有一个**特殊工具函数**：

```cpp
// 注释原文：
// "Read a scalar value from a tensor that is maybe a UVM tensor.
//  Note that `tensor.item<type>()` is not allowed on a UVM tensor in PyTorch"
inline int64_t get_maybe_uvm_scalar(const at::Tensor& tensor) {
    return *tensor.data_ptr<int>();  // 绕过 .item() 限制
}
```

**坑点**：PyTorch 的 `.item()` 内部会做 device→host 同步，对 UVM tensor 不支持。FBGEMM 用 `*data_ptr<T>()` 绕过。SSD TBE 的 `count` 等标量读取都走这个工具函数。

---

## 4. 业务场景与触发条件

### 4.1 FBGEMM 典型部署场景的内存特征

| 场景 | Embedding 行数 | 单行维度 | 单表容量 | 一次 batch 实际访问 |
|---|---|---|---|---|
| 推荐系统（DLRM / 排序 / 召回） | 1e8 – 1e10 | 16 – 256 | **几 GB – 几百 GB** | **< 5%**（长尾 0.1% – 2%） |
| GenAI / Llama 3/4 推理 | 7B – 405B+ 参 | 隐藏维度 | 量化后 35 – 200 GB | **全表都访问**（dense） |
| HSTU 序列推荐 | 1e8 – 1e10 | 16 – 256 | 几 GB – 几百 GB | 长尾 + 序列特征 |
| MoE LLM（Llama-4 Maverick 等） | 几十–几百 expert × 几十 B | 隐藏维度 | 100+ GB | 一次只激活 2–8 个 expert |

### 4.2 内存层级与带宽

```
HBM (GPU 显存)   : 40 – 80 GB / 卡 (消费级 ~24 GB),  带宽 2 – 3 TB/s
CPU DRAM         : 单节点几百 GB – 几 TB,             带宽 100 GB/s
NVMe SSD         : 几十 TB,                            带宽 3 – 7 GB/s
PCIe Gen4 ×16    : 32 GB/s  (连接 CPU↔GPU)
PCIe Gen5 ×16    : 64 GB/s
NVLink 3.0       : 600 GB/s (GPU↔GPU)
NVLink 4         : 900 GB/s
```

**结论**：单卡 HBM 装不下百 GB 级 embedding 表，必须用 DRAM/NVMe 当"溢出层"。

### 4.3 "Use UVM when..." 决策表

> ⚠️ **本表是 2026-06-10 更正版**：之前版本中"多卡 / 多节点共享同一张表 → UVM page migration 避免多卡显式同步"的说法**是错误的**——UVM 在 FBGEMM 中只有"**单进程 / 单节点**"半径（详见 §10 边界澄清）。

| 触发条件 | 说明 | 半径 |
|---|---|---|
| **嵌入表总容量 ≫ 单卡 HBM** | 100 GB 表 + 80 GB HBM 是最直接触发条件 | 单卡 |
| **访问极端稀疏且长尾** | batch 触达 < 5%，HBM 命中率 < 5%，硬塞 HBM 是浪费 | 单卡 |
| **冷热分层访问** | 少量行频繁（热），大量行偶尔（冷），非常契合 `MANAGED_CACHING` + LRU/LFU | 单卡 |
| **单节点内多卡共享同一张表** | ⚠️ UVM 在单节点内通过 P2P/NVLink 间接可能，但 FBGEMM **不直接用** | 单节点多卡（理论） |
| **跨节点共享同一张表** | ❌ UVM **不适用**——由 TorchRec + NCCL/HCCL 做 sharding + all_to_all 解决 | **UVM 范围外** |
| **训练 + 推理同一份参数** | 优化器在 CPU update、forward 在 GPU lookup，UVM 同一份指针即可 | 单卡 |
| **不想写复杂分片/分桶代码** | 原型期 / 跨业务线共用表 | 单卡 |
| **NVMe-backed 之外的次选** | SSD cache 之外的 DRAM 级折中 | 单卡 |
| **需要 host-mapped 行为** | `uvm_host_mapped=True` 走 `cudaHostRegister`，对 fork 更友好 | 单卡 |

### 4.4 "不要使用 UVM" 反例

| 场景 | 原因 |
|---|---|
| **小表 + 高频访问** | 整张表能放下且全访问，UVM 反而引入 page-fault 抖动；用 `EmbeddingLocation.DEVICE` |
| **极致低延迟（RTB / P99 < 5ms）** | UVM page fault 不可预测，应**预热 + lock** 或全 HBM |
| **计算密集、显存吃紧 + KV cache 卸载** | UVM 与显存的 LRU 互相干扰，更建议显式管 cache |
| **CPU 内存也吃紧** | UVM 爆 DRAM 一样 OOM；走 NVMe/SSD 路径 |
| **跨进程 / fork / 多租户** | managed memory 在 fork、跨进程传递时易触发"释放时 cuda context 不匹配"等坑（`memory_utils.cu:17-45` 注释专门讲） |

---

## 5. 三种存储方案对比（HBM / Host + memcpy / UVM）

以 **embedding lookup** 场景为基准：

| 维度 | 方案 A：纯 HBM (`DEVICE`) | 方案 B：Host DRAM + 显式 `cudaMemcpy` (`HOST`) | 方案 C：UVM `MANAGED` / `MANAGED_CACHING` |
|---|---|---|---|
| **对应 FBGEMM 开关** | `EmbeddingLocation.DEVICE` | `EmbeddingLocation.HOST` + 手动 `to("cuda")` | `EmbeddingLocation.MANAGED` / `MANAGED_CACHING` |
| **支持的表规模** | 严格 ≤ HBM（通常 ≤ 80 GB / 卡） | 不限（受 CPU DRAM 限制） | 不限（受 CPU DRAM / NVMe 限制） |
| **HBM 中是否有数据** | 100% 都在 HBM | 0%（全在 CPU） | **MANAGED**：HBM 也有数据（driver 自动迁移的 hot page，不可控）；**MANAGED_CACHING**：HBM 有显式 cache 池（`cache_load_factor` 决定，可控） |
| **访问延迟** | 最低（一次 memcpy 之后都很快） | 高（每次访问都要 memcpy） | 中——MANAGED 首次走 PCIe 直读 CPU（~32 GB/s），稳态 hot page 走 HBM ≈ 方案 A；MANAGED_CACHING 命中走 HBM cache（≈ 方案 A），未命中走 UVM |
| **PCIe 带宽利用率** | 一次性摊销 | 每次访问都吃 PCIe，**极差** | 驱动按页粒度合并，**高** |
| **代码复杂度** | 低 | 高（显式编排传输） | 中（少量 `cudaMemAdvise` / `prefetch`） |
| **缓存策略可调** | 难（必须分片/分桶） | 难 | **MANAGED 不可调**（driver 决定）；**MANAGED_CACHING 易调**——`cache_load_factor` + LRU/LFU/Direct-mapped |
| **首次访问抖动** | 无 | 无 | 有（GPU 大模型预热时可见；MANAGED_CACHING 抖动更小） |
| **是否需要分桶/sharding** | **是** | 否 | 否（与方案 A 比，少分桶） |
| **典型权衡** | 容量天花板最低、性能最稳 | 实现最复杂、浪费最多带宽 | **首推**——容量大、代码简单、驱动调优 |

**一句话总结**：

- 方案 A：装得下就用 HBM，最稳。
- 方案 B：理论上"都行"，但实际会**把 PCIe 打满** + 白白烧掉延迟。
- 方案 C：**让 CUDA driver 替你搬运**，业务代码看不到传输，FBGEMM TBE 把它做成 `MANAGED_CACHING` 的"开箱即用"开关。

---

## 6. 性能调优与最佳实践

### 6.1 FBGEMM 已做的关键调优

`memory_utils.cu:202-227` 注释原文：

```cpp
// 1. Set preferred memory location to host memory
gpuMemAdvise(ptr, size_bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);

// 2. User hints with 'accessed by': GPU will establish direct mapping
//    of data in CPU memory, no page faults will be generated
gpuMemAdvise(ptr, size_bytes, cudaMemAdviseSetAccessedBy, current_device);

// 3. Fork-safe: 告诉内核 fork 时不要复制这片 region
madvise(ptr, size_bytes, MADV_DONTFORK);
```

这三步等价于：**UVM 数据"家"在 CPU，但 GPU 已经准备好了 direct mapping，访问时不会触发 page fault**。这是 FBGEMM 性能稳定的核心。

### 6.2 用户可调的关键参数

| 参数 | 默认值 | 调优建议 |
|---|---|---|
| `cache_load_factor` | `0.2` | HBM cache 大小 = `0.2 × 总行数`。命中率 < 90% 可调高；命中率 > 99% 可调低以释放 HBM |
| `uvm_host_mapped` | `False` | `True` 走 `cudaHostAlloc`（零拷贝），对 fork 场景更友好；但 `cudaHostAlloc` 在大块分配时可能锁全局 |
| `gather_uvm_cache_stats` | `False` | 生产期关闭，benchmark/调优期开启 |
| `EmbeddingLocation` | `DEVICE` | 表 < 80 GB 用 `DEVICE`；表 80 GB – 几百 GB 用 `MANAGED_CACHING`；表 > 几 TB 考虑 SSD TBE |
| `cudaMemAdvise(SetReadMostly)` | 未默认开启 | embedding 表是只读热点，可考虑开启（多卡场景） |
| `cudaMemPrefetchAsync` | 调 `forward` 前手动调 | 推荐在前向/反向前 `uvm_cuda_mem_prefetch_async(t, current_device)` |

### 6.3 性能 benchmark

`fbgemm_gpu/bench/tbe/tbe_inference_benchmark.py:1022-1291` 提供 `nbit_uvm` benchmark，量化对比 device 表 / uvm 表 / 混合三种部署：

- `nbit_uvm`：纯 UVM 路径
- `nbit_uvm_compare_direct_mapped`：与 LRU/Direct-mapped cache 对比
- `print_uvm_cache_stats` 输出 6 项统计
- `bench_uvm_cls("HBM")` / `"32way"` / `"1way"` 对比不同 cache 配置

`tbe_cache_benchmark.py:392, 439` 注释明确：

> *"Replay to figure out UVM access BW, which would be PCIe bound"*
> *"BW (just UVM accesses): ..."*

即 UVM 的极限性能受 **PCIe 带宽**制约——benchmark 输出会直接展示这一点。

### 6.4 经验性能数据

| 场景 | UVM 未调优 | UVM 调优 + NVLink | 显式 `cudaMemcpy` |
|---|---|---|---|
| 单卡 256 MB GEMM | 0.7~0.9× | ~1.0× | 基准 |
| 单卡 8 GB embedding 查表 | 0.5~0.7× | 0.85~0.95× | 基准 |
| 多卡 8 卡 + 反复访问 | 0.1~0.3× | 0.5~0.8× | 基准 |
| TB 级 spill（无 HBM 装得下） | **唯一可行** | **唯一可行** | 不可行 |

> *量级为经验值；具体依赖硬件、驱动版本、page size。* 启用 Linux THP / 2 MB huge page 能进一步减少 30~50% fault 开销。

### 6.5 调优流程建议

```
1. 选 EmbeddingLocation：
   表 ≤ HBM 80% → DEVICE
   表 > HBM     → MANAGED_CACHING（cache_load_factor=0.2 起步）

2. 开 gather_uvm_cache_stats=True，跑到稳态，调 cache_load_factor：
   - 命中率 < 90% → 调高 cache_load_factor
   - 命中率 > 99% → 调低 cache_load_factor 释放 HBM

3. 调 uvm_host_mapped：
   - 常规用 False（cudaMallocManaged + 上面两步 advise）
   - 多进程 / fork 场景用 True（cudaHostAlloc 零拷贝）

4. 性能仍不达预期时：
   - 检查 PCIe 带宽是否打满（nvprof / nsys）
   - 启用 2 MB huge page（Linux THP）
   - 在 forward 前调 uvm_cuda_mem_prefetch_async
   - 考虑切回 DEVICE + 显式 sharding
```

---

## 7. GenAI / 大模型场景下 UVM 的角色

### 7.1 INT4/INT8/FP8 量化后是否还需要 UVM？

**绝大多数情况下：不需要。**

- 70B 模型 INT4 ≈ 35 GB、405B 模型 INT4 ≈ 200 GB
- H100 / H200 已有 80 – 141 GB HBM，配合 tensor parallel + pipeline parallel 单卡都能装下
- 量化权重的访问模式是"全表都需要"（dense model），**没有长尾可言**

**例外（仍然需要 UVM 思想）**：
- **超大 MoE**（Mixtral 8x22B、Llama-4 Maverick、Llama-4 Behemoth），专家权重 200+ GB
- **KV cache 卸载**（vLLM 的 `cpu offload`、FBGEMM GenAI 的 `kv_cache.cu`）—— 激活/中间张量可以用 UVM 思想搬运

### 7.2 KV cache、激活张量是否受益 UVM？

| 张量类型 | 生命周期 | 访问模式 | 是否适合 UVM |
|---|---|---|---|
| **量化后的 LLM 权重** | 长（数小时） | dense（每 token 都访问） | ❌ 用 TP/PP 分片 |
| **KV cache** | 中（每个请求） | 半密集（每 token 写一次） | ⚠️ 思想可借鉴，但 FBGEMM 选**显式 `kv_cache.cu` 自管** |
| **激活张量** | 短（一个 layer） | dense + 大块 | ❌ 抖动不可接受 |
| **MoE 专家权重** | 长（模型加载时） | **极端稀疏**（top_k=2/4/8） | ✅ 强烈适合 |

### 7.3 MoE：UVM 在 GenAI 的"复兴"场景

这是 **UVM 真正有可能在 GenAI 中复兴的场景**：

- MoE 每次只激活 2–8 个专家（`top_k=2/4/8`），但全部专家权重可能 100+ GB
- 行为非常像 DLRM：**超大表 + 稀疏访问 + 长尾**
- FBGEMM `MANAGED_CACHING` + `cache_load_factor` 思想**完全可以平移**到 MoE：
  - `cache_load_factor = 活跃专家比例 × 专家总数`
  - HBM cache 装热点专家权重
  - UVM/DRAM 装全量专家权重
  - 触发专家 → cache hit → 走 HBM；未触发 → page fault → 走 UVM

### 7.4 FBGEMM_GPU 的 GenAI 模块当前状态

- `fbgemm_gpu/experimental/gen_ai/`（Llama 3/4 量化推理）—— **不使用 UVM**，靠 `quantize_ops` + TP/PP
- `fbgemm_gpu/experimental/hstu/`（层次序列转换）—— **不使用 UVM**
- `fbgemm_gpu/experimental/simplicial_attention/`、`gemm/`、`example/` —— 都不使用

MoE 专家权重的 UVM 卸载目前在 FBGEMM_GPU 中**尚未实现**，是潜在演进方向。

---

## 8. 总结与决策建议

### 8.1 核心因果链

```
推荐系统 sparse embedding 容量 ≫ HBM
  → 必须把"全集"放到 DRAM/NVMe
    → 但一次 batch 只访问 < 5%（长尾）
      → 全量 memcpy = 浪费 PCIe
        → 自然的解法: "按页粒度按需搬运"
          → CUDA 的 UVM (managed memory) + cudaMemAdvise/cudaMemPrefetchAsync
            → FBGEMM 把它包装成 EmbeddingLocation.MANAGED / MANAGED_CACHING
              → 业务方只需设置 cache_load_factor，不必管传输
                → 代码最简、性能足够稳
```

```
GenAI dense LLM:
  70B+ 模型权重 量化后仍 > 100 GB
    → 单卡 HBM 装不下
      → 主流解: TP/PP + 量化
        → 量化后若装得下，就不需要 UVM

GenAI MoE:
  权重全集/激活子集重新出现
    → UVM/DRAM 思想 + 显式 cache 又成为合理选择
```

### 8.2 决策速查表

| 场景 | 推荐方案 | FBGEMM 开关 | 半径 |
|---|---|---|---|
| 推荐系统 embedding 表 ≤ 80 GB，单卡 HBM 装得下 | **HBM** | `EmbeddingLocation.DEVICE` | 单卡 |
| 推荐系统 embedding 表 80 GB – 几百 GB，单 batch 触达 < 5% | **UVM + HBM cache**（首推） | `EmbeddingLocation.MANAGED_CACHING` + `cache_load_factor=0.2` | 单卡 |
| 推荐系统 embedding 表 > 几百 GB，DRAM 也不够 | **SSD TBE + UVM** | `ssd_split_embeddings_cache` + `EmbeddingLocation.MANAGED` | 单卡 |
| GenAI 量化 LLM 推理（dense） | **TP/PP + 量化** | 不使用 UVM | — |
| GenAI MoE 推理（专家稀疏激活） | **未来 UVM 复兴场景** | （FBGEMM_GPU 尚未原生支持） | — |
| 单节点内多卡共享大表 | UVM + `SetReadMostly` + `SetAccessedBy` | `MANAGED` + `cudaMemAdvise` | 单节点（理论可行，FBGEMM 不直接用） |
| **跨节点分布式训练（4 节点 32 卡）** | **TorchRec sharding + NCCL/HCCL all_to_all** | FBGEMM 接收 `embedding_shard_info` 元组，**不参与跨节点通信** | **UVM 范围外** |
| 极致低延迟（< 5ms P99） | **预热 + 全 HBM** | `EmbeddingLocation.DEVICE` | 单卡 |

### 8.3 一页速查

```
FBGEMM 中 UVM 的 5 件事:

1. 唯一分配点:  memory_utils.cu::new_managed_tensor_internal (cudaMallocManaged)
                + new_managed_tensor (分配后立即 SetPreferredLocation=CPU + SetAccessedBy=GPU)

2. 唯一使用方:  TBE (Table Batched Embeddings) — 推荐系统稀疏 embedding

3. 核心开关:    EmbeddingLocation.MANAGED / MANAGED_CACHING
                + cache_load_factor (默认 0.2) 调 HBM cache 大小

4. 关键调优:    cudaMemAdvise(SetPreferredLocation, SetAccessedBy) — 避免 page fault
                cudaMemPrefetchAsync — 主动预取
                2 MB huge page — 减少 fault 次数

5. 业务价值:    让百 GB 级 embedding 表在 HBM/DRAM 之间按需分页
                单 batch < 5% 触达时, 比"全量 memcpy"省 10×+ 带宽
```

---

## 9. 附：关键源码/文件路径速查

### 9.1 核心 UVM 实现

| 文件 | 行号 | 关键内容 |
|---|---|---|
| `fbgemm_gpu/include/fbgemm_gpu/cumem_utils.h` | 全文 | UVM 公共 API 头（Doxygen） |
| `fbgemm_gpu/src/memory_utils/memory_utils.cu` | 88-131 | `new_managed_tensor_internal`（核心 `cudaMallocManaged`） |
| 同上 | 192-238 | `new_managed_tensor`（分配后两步 advise） |
| 同上 | 240-248 | `new_vanilla_managed_tensor` |
| 同上 | 250-293 | `new_host_mapped_tensor` |
| 同上 | 295-306 | `new_unified_tensor`（总入口） |
| 同上 | 308-319 | `uvm_storage` / `is_uvm_tensor` |
| 同上 | 321-373 | `uvm_to_cpu` / `uvm_to_device` |
| 同上 | 397-426 | `uvm_cuda_mem_advise` |
| 同上 | 428-464 | `uvm_cuda_mem_prefetch_async` |
| 同上 | 466-486 | `uvm_mem_advice_dont_fork` |
| 同上 | 548-575 | `FBGEMM_GPU_ENUM_GLOBAL(uvm)` 枚举导出 |
| `fbgemm_gpu/src/memory_utils/memory_utils.cpp` | 全文 | Meta/CPU dispatch |
| `fbgemm_gpu/src/memory_utils/memory_utils_ops.cpp` | 18-28 | TorchLibrary 注册（CPU 端） |
| `fbgemm_gpu/src/memory_utils/memory_utils_ops.cu` | 全文 | TorchLibrary 注册（CUDA 端） |
| `fbgemm_gpu/src/memory_utils/memory_utils_fallback.cpp` | 全文 | 非 GPU 构建回退 |
| `fbgemm_gpu/fbgemm_gpu/uvm.py` | 全文 | Python 用户态封装 |

### 9.2 TBE 算子

| 文件 | 行号 | 关键内容 |
|---|---|---|
| `fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_common.py` | 29-43 | `EmbeddingLocation` 枚举 |
| 同上 | 384 | `uvm_size` 字段 |
| 同上 | 407-474 | `construct_cache_state` / `get_new_embedding_location` |
| `fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_training.py` | 184-190 | `UVMCacheStatsIndex` |
| 同上 | 286-401 | `apply_split_helper`（核心 UVM 分配） |
| 同上 | 1665-1697 | UVM state tensor 注释 |
| 同上 | 1765-1781 | `_categorize_memory_usage`（HBM/UVM 拆分） |
| 同上 | 1829-1869 | `_report_uvm_breakdown` |
| `fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_inference.py` | 201-214 | 4 种 placement docstring |
| 同上 | 317, 374 | `uvm_host_mapped: bool = False` 形参 |
| 同上 | 530-532 | `self.weights_uvm` buffer |
| 同上 | 692-739 | `reset_uvm_cache_stats` / `print_uvm_cache_stats` |
| 同上 | 1589, 1625 | `MANAGED_CACHING` 分支 / `weights = self.weights_uvm` |
| `fbgemm_gpu/fbgemm_gpu/split_embedding_configs.py` | 263 | `uvm_size=0` 字段 |
| `fbgemm_gpu/fbgemm_gpu/tbe/ssd/training.py` | 173-175 | `uvm_host_mapped` 形参 |
| `fbgemm_gpu/fbgemm_gpu/tbe/ssd/inference.py` | 327, 489-493 | `"weights_uvm"` state dict key |
| `fbgemm_gpu/fbgemm_gpu/tbe/cache/kv_embedding_ops_inference.py` | 75, 240 | `uvm_host_mapped` / `uvm_weights=self.weights_uvm` |
| `fbgemm_gpu/fbgemm_gpu/tbe/bench/embedding_ops_common_config.py` | 36-37, 94-97 | `--emb-uvm-host-mapped` CLI |
| `fbgemm_gpu/fbgemm_gpu/sparse_ops.py` | 427, 452 | `int_nbit_split_embedding_codegen_lookup_function` 透传 |

### 9.3 缓存层

| 文件 | 行号 | 关键内容 |
|---|---|---|
| `fbgemm_gpu/src/split_embeddings_cache/split_embeddings_cache_ops.cpp` | 20-37 | 算子 schema 含 `uvm_cache_stats` |
| `fbgemm_gpu/src/split_embeddings_cache/lru_cache_find.cu` | 全文 | UVM cache 统计 |
| `fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate.cu` | 274-292 | `lru_cache_populate_cuda` |
| `fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate_byte.cu` | 506-521 | `lru_cache_populate_byte_cuda` |
| `fbgemm_gpu/src/split_embeddings_cache/lxu_cache.cu` | 84-145 | `lxu_cache_flush_cuda`（写回 UVM） |
| 同上 | 404-501 | `lxu_cache_populate_cuda` / `lxu_cache_lookup_cuda` |
| `fbgemm_gpu/src/split_embeddings_cache/reset_weight_momentum.cu` | 171-173 | `MANAGED_CACHING` 分支 |
| `fbgemm_gpu/src/split_embeddings_cache/raw_embedding_streamer.cpp` | 21-28 | `get_maybe_uvm_scalar` |
| `fbgemm_gpu/src/ssd_split_embeddings_cache/kv_db_table_batched_embeddings.cpp` | 21-28 | `get_maybe_uvm_scalar` |

### 9.4 Embedding Inplace

| 文件 | 行号 | 关键内容 |
|---|---|---|
| `fbgemm_gpu/src/embedding_inplace_ops/embedding_inplace_update.cu` | 93 | `MANAGED_CACHING` 分支 |

### 9.5 代码生成（Codegen）

| 文件 | 行号 | 关键内容 |
|---|---|---|
| `fbgemm_gpu/codegen/genscript/optimizer_args.py` | 55-74 | UVM 形参定义 |
| `fbgemm_gpu/codegen/genscript/optimizers.py` | 1071 | `row_counter = &row_counter_uvm[...]` |
| `fbgemm_gpu/codegen/genscript/generate_backward_split.py` | 427 | `"uvm_cache_stats"` schema |
| `fbgemm_gpu/codegen/training/forward/embedding_forward_split_kernel_v2_template.cu` | 282, 412, 619 | "If UVM cache is used" 注释 |
| `fbgemm_gpu/codegen/training/forward/embedding_forward_split_kernel_template.cu` | 747, 751 | "Load every row from HBM or UVM" |
| `fbgemm_gpu/codegen/training/backward/embedding_backward_split_template.cu` | 81, 174, 257, 528, 621, 1118, 1293, 1393 | `uvm_weights` 形参 |
| `fbgemm_gpu/codegen/training/backward/embedding_backward_split_kernel_cta_template.cu` | 97, 367, 469 | `uvm_weights` |
| `fbgemm_gpu/codegen/training/backward/embedding_backward_split_kernel_warp_template.cu` | 90, 285, 389, 571, 709 | `uvm_weights` |
| `fbgemm_gpu/codegen/training/backward/embedding_backward_split_indice_weights_template.cu` | 82, 159, 161, 204, 219, 354 | `MANAGED_CACHING` 分支 |
| `fbgemm_gpu/codegen/training/backward/embedding_backward_split_grad_template.cu` | 80, 103-111, 136 | `gpuAtomicAdd` 到 UVM |
| `fbgemm_gpu/codegen/training/backward/embedding_backward_split_host_template.cpp` | 全文 | CPU stub 含 `uvm_weights` / `uvm_cache_stats` |
| `fbgemm_gpu/codegen/training/backward/embedding_backward_split_meta_template.cpp` | 64 | meta 路径 |
| `fbgemm_gpu/codegen/training/optimizer/embedding_optimizer_split_host_template.cpp` | 25, 37 | CPU 优化器 |
| `fbgemm_gpu/codegen/training/optimizer/embedding_optimizer_split_template.cu` | 115, 128 | `TENSOR_ON_CUDA_GPU({{ tensor }}_uvm)` |
| `fbgemm_gpu/codegen/training/optimizer/embedding_optimizer_split_device_kernel_template.cuh` | 99 | `{{ tensor }} = &{{ tensor }}_uvm[{{ tensor }}_offset]` |
| `fbgemm_gpu/codegen/training/pt2/embedding_split_host_pt2_autograd_template.cpp` | 全文 | PT2 路径 |
| `fbgemm_gpu/codegen/training/python/lookup_args.template` | 91 | `uvm: torch.Tensor` 形参 |
| `fbgemm_gpu/codegen/training/python/split_embedding_codegen_lookup_invoker.template` | 50, 76, 135, 310, 318, 334 | Python invoker |
| `fbgemm_gpu/codegen/training/python/split_embedding_optimizer_codegen.template` | 145-147, 237-238, 243, 290 | UVM/LXU 注释 |

### 9.6 测试

| 文件 | 行号 | 关键内容 |
|---|---|---|
| `fbgemm_gpu/test/uvm/uvm_test.py` | 34-224 | UVM 单元测试（`test_is_uvm_tensor`、`test_cudaMemAdvise`、`test_cudaMemPrefetchAsync`、`test_uvm_slice`、`test_uvm_memadviceDontFork`） |
| `fbgemm_gpu/test/uvm/copy_test.py` | 41-152 | `new_unified_tensor` / `new_managed_tensor` 复制测试 |
| `fbgemm_gpu/test/uvm/ops_load_test.py` | 36 | op 加载测试 |
| `fbgemm_gpu/test/uvm/cache_miss_emulate_test.cpp` | 33-63 | cache miss 模拟测试 |
| `fbgemm_gpu/test/tbe/common.py` | 270-400, 580-594 | TBE 测试 UVM 形参 |
| `fbgemm_gpu/test/tbe/cache/cache_test.py` | 405, 636, 997 | `tbe.total_uvm_usage`、`MANAGED_CACHING` |
| `fbgemm_gpu/test/tbe/cache/cache_overflow_test.py` | 35 | "TBE with UVM caching" 注释 |
| `fbgemm_gpu/test/tbe/training/forward_test.py` | 167, 217, 229, 1343-1399, 1586, 1591 | "Raw embedding streaming requires UVM cache" |
| `fbgemm_gpu/test/tbe/training/backward_optimizers_test.py` | 108-1154 | `uvm_non_rowwise_momentum` 测试 |
| `fbgemm_gpu/test/tbe/training/backward_none_test.py` | 266 | `MANAGED_CACHING` |
| `fbgemm_gpu/test/tbe/training/backward_adagrad_common.py` | 206, 216 | `MANAGED_CACHING` / `MANAGED` |
| `fbgemm_gpu/test/tbe/training/store_prefetched_tensors_test.py` | 211, 362, 402, 409 | "tables have UVM cache enabled" |
| `fbgemm_gpu/test/tbe/training/merge_vbe_test.py` | 169-203 | `mixed_uvm: bool` |
| `fbgemm_gpu/test/tbe/inference/nbit_cache_test.py` | 101-482 | 各种 `MANAGED_CACHING` / `MANAGED` 测试 |
| `fbgemm_gpu/test/tbe/inference/nbit_forward_test.py` | 767 | `MANAGED_CACHING` |
| `fbgemm_gpu/test/tbe/utils/split_embeddings_test.py` | 210 | `MANAGED_CACHING` |
| `fbgemm_gpu/test/faster_hash_test.py` | 668-693 | `test_zch_output_on_uvm`（zch 输出到 UVM） |

### 9.7 Benchmarks

| 文件 | 行号 | 关键内容 |
|---|---|---|
| `fbgemm_gpu/bench/tbe/tbe_inference_benchmark.py` | 388-390, 1022-1291, 1354-1791 | `nbit_uvm` / `nbit_uvm_compare_direct_mapped` |
| `fbgemm_gpu/bench/tbe/tbe_ssd_benchmark.py` | 365-366, 559-595, 658-671, 739, 759, 773 | "UVM" / "UVM_CACHING" generator |
| `fbgemm_gpu/bench/tbe/tbe_cache_benchmark.py` | 87, 129-562 | UVM 访问 BW 测量 |
| `fbgemm_gpu/bench/tbe/tbe_utils_benchmark.py` | 486, 598 | `op.weights_uvm` |
| `fbgemm_gpu/bench/tbe/split_table_batched_embeddings_benchmark.py` | 217-1077 | 各种 MANAGED/UVM 用法 |

### 9.8 构建系统

| 文件 | 行号 | 关键内容 |
|---|---|---|
| `fbgemm_gpu/FbgemmGpu.cmake` | 27-28, 78-79 | `memory_utils.cpp` / `memory_utils.cu` 加入构建 |

### 9.9 外部参考

- **FBGEMM 论文**（Khudia et al., 2021）：https://arxiv.org/pdf/2101.05615.pdf
- **DLRM 论文**（Naumov et al., 2019）：https://arxiv.org/abs/1906.00091
- **HSTU 论文**（Zhai et al., 2024）：https://arxiv.org/abs/2402.17149
- **PyTorch/FBGEMM 官方文档**：https://pytorch.org/FBGEMM
- **PyTorch/FBGEMM GitHub**：https://github.com/pytorch/FBGEMM
- **NVIDIA CUDA Unified Memory Programming Guide**：https://docs.nvidia.com/cuda/cuda-c-programming-guide/04-special-topics/unified-memory.html
- **Linux HMM 文档**：`Documentation/mm/hmm.rst`
- **Meta Engineering Blog**（FBGEMM 介绍）：https://engineering.fb.com/2021/06/15/open-source/fbgemm/

---

## 10. UVM 在 FBGEMM 中的边界澄清（**2026-06-10 增补**）

> **本节是对全文中"UVM 跨卡 / 跨节点"相关说法的正式澄清。** 之前的 §1.4、§4.3、§5 表格中部分表述容易给读者造成"UVM 可以解决跨节点访问"的误解，本节明确边界。

### 10.1 核心结论（一句话）

> **UVM 在 FBGEMM 中只有"单进程 / 单节点"的半径。** 跨节点（多机）embedding 访问由 **TorchRec sharding + NCCL/HCCL all_to_all** 解决，**与 UVM 完全无关**。

### 10.2 用户原问题答复

> **Q: 32 卡训练（4 节点 × 8 卡），卡 0 在服务器 1，卡 31 在服务器 4，卡 0 能直接访问卡 31 的 UVM 中 emb 数据么？**

**A: 不能。** 具体见下方三层解释。

#### 10.2.1 FBGEMM 源码层：UVM 路径不感知"其他节点"

- [`fbgemm_gpu/fbgemm_gpu/uvm.py`](fbgemm_gpu/fbgemm_gpu/uvm.py) 全文 41 行，只暴露 `cudaMemAdvise` 和 `cudaMemPrefetchAsync` 两个包装函数
- [`fbgemm_gpu/src/memory_utils/memory_utils.cu`](fbgemm_gpu/src/memory_utils/memory_utils.cu:147-186) 中 `cudaMemLocation` 的 type 字段**只支持 `Device` 和 `Host` 两种**——没有 `cudaMemLocationTypeDeviceNvidiaPeer`、没有 NUMA 节点、没有跨节点概念
- FBGEMM 整个仓库（`fbgemm_gpu/src/` 和 `fbgemm_gpu/fbgemm_gpu/`）grep `torch.distributed` / `init_process_group` / `nccl*` / `all_to_all` / `world_size` / `process_group` —— **全部 0 命中**（排除 `experimental/gen_ai/`）
- `MANAGED` / `MANAGED_CACHING` 的 `cudaMemAdvise` 目标位置永远是：`cudaCpuDeviceId`（**当前节点 host**）或 `t.get_device()`（**当前进程当前 GPU**）

#### 10.2.2 FBGEMM 架构层：FBGEMM 是"单 rank 算子库"

- [`split_table_batched_embeddings_ops_training.py:646-649`](fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_training.py#L646) 接收 `embedding_shard_info` 元组 `(preshard_table_height, preshard_table_dim, height_offset, dim_offset)`，告诉它"我这个 rank 上这张表被切成什么样子"
- FBGEMM 假设整张全局表已经按 row-wise / table-wise / column-wise 在所有 rank 上静态划分完毕，**它自己不做 sharding 决策**
- [`fbgemm_gpu/bench/README.md:5-6`](fbgemm_gpu/bench/README.md#L5) 原文：
  > *"[Torchrec](https://pytorch.org/torchrec/) uses fbgemm_gpu embedding and embedding bag implementations for Fused, Batched, Quantized versions of embedding and embeddingbag (in addition to other kernels)."*
  即 **FBGEMM 是被 TorchRec 调用的 kernel 提供方**，分布式编排交给 TorchRec

#### 10.2.3 NVIDIA 架构层：CUDA UVM 不跨节点

- CUDA Managed Memory 的 page table 由单节点内统一虚拟地址空间管理（同一 OS 进程内）
- 跨节点需要走 NCCL/Gloo + RDMA/IB——**UVM 的 page fault/migrate 机制不延伸到跨节点 GPU**
- 即便 NVIDIA 12.x/13.x 引入了 NUMA 改进，也仍是"单节点内多 socket"层级，**不延伸到多机**
- 这是 NVIDIA 文档与驱动的硬约束，不是 FBGEMM 的实现选择

### 10.3 实际跨节点访问流程（4 节点 32 卡示例）

以 DLRM 32 卡训练为例，假设 batch 分散到 32 个 rank 上：

```
                     节点 1                              节点 4
                ┌──────────────┐                   ┌──────────────┐
                │ 卡 0..7      │                   │ 卡 24..31    │
                │ emb[0..7/32] │                   │ emb[24..31/32]│
                └──────────────┘                   └──────────────┘
                       │                                  ▲
                       │  step 1: torch.distributed.all_to_all_single
                       │         把 indices 按 rank 路由 (底层走 IB/RoCE via NCCL)
                       ▼                                  │
                ┌──────────────────────────────────────────┐
                │  各 rank 本地 lookup, 走 FBGEMM TBE      │
                │  (本 rank 内的 idx 查本 rank 的 emb 分片)  │
                │  → 返回局部 pooled embedding              │
                └──────────────────────────────────────────┘
                                │
                                ▼
                ┌──────────────────────────────────────────┐
                │  torch.distributed.all_to_all_single (回程)│
                │  把 pooled embedding 按原 batch 顺序回拼   │
                └──────────────────────────────────────────┘
```

**关键**：
- 跨节点走 **NCCL**（GPU backend），底层走 **InfiniBand / RoCE**；NCCL 内部用 `libnccl`，**完全脱离 FBGEMM 的 UVM 路径**
- FBGEMM 只负责"本 rank 内的 lookup"，对 ranks 间的分布"完全无知"
- **没有任何代码路径让卡 0 通过 UVM 直接读到卡 31 的内存**

### 10.4 UVM 半径决策表（最终版）

| 半径 | 是否能用 UVM | FBGEMM 是否实际使用 | 备注 |
|---|---|---|---|
| **同一进程 + 同一节点：CPU ↔ 单卡 HBM** | ✅ | ✅ `MANAGED` / `MANAGED_CACHING` | 唯一主战场 |
| **同一进程 + 同一节点：CPU ↔ 8 卡 HBM（NVLink）** | ✅ 理论 | ⚠️ 通过 `SetAccessedBy` 多卡 hint | FBGEMM 不强制使用 |
| **同一进程 + 同一节点：卡 0 ↔ 卡 1（P2P）** | ✅ 理论可行 | ❌ 不直接使用 | 由 `nv_peer_mem` 兜底 |
| **不同进程 + 同一节点** | ❌ | — | 进程隔离 |
| **跨节点（卡 0 在服务器 1，卡 31 在服务器 4）** | ❌ | ❌ | **由 TorchRec + NCCL/HCCL all_to_all 解决** |

### 10.5 错误说法清单（已修正）

| 位置 | 原错误说法 | 修正后 |
|---|---|---|
| §4.3 决策表 | "多卡 / 多节点共享同一张表 → UVM page migration 避免多卡显式同步" | 拆为"单节点多卡（理论）"和"跨节点（UVM 范围外，由 TorchRec 解决）"两行 |
| §8.2 决策速查表 | "多卡共享大表 → UVM + SetReadMostly" | 限定为"**单节点内**多卡"；新增"跨节点分布式训练 → TorchRec sharding + NCCL/HCCL"行 |
| §1.4 战略价值 | "UVM 是 FBGEMM / TorchRec 的事实依赖" | 限定为"**单卡 HBM 装不下**时的依赖"；跨节点由 NCCL/HCCL 解决 |

### 10.6 启示

1. **FBGEMM UVM 需求基线以"单卡 HBM 装不下"为核心**，不要被"统一内存"字面意思误导到"跨节点"
2. **跨节点需求不在 FBGEMM 范围内**，由 TorchRec + 集合通信解决
3. **昇腾 950 系列补 UVM 能力**也只解决"单卡 HBM 容量"问题；**跨节点通信的 RDMA / IB / HCCL over RoCE 是另一条独立产品线**（属于 HCCL/集合通信范畴）
4. 写"UVM 需求"文档时，**不要把跨节点需求与 UVM 混在一起**——这是两个独立的软件栈

---

*报告结束。本报告基于 FBGEMM v1.5.0-release 分支源码、CUDA 官方文档及 FBGEMM 公开论文综合整理；具体性能数字会因驱动版本、CUDA 版本、硬件代际不同而浮动，使用前请以本地 benchmark 为准。*
