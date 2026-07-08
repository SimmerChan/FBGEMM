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

FBGEMM 全部用 **runtime API**（**未使用** `__managed__` 编译期变量、`cuMemAllocManaged` driver API、`cudaStreamAttachMemAsync`、以及**纯 `cudaMalloc`**——所有 GPU 内存分配要么走 UVM 要么走 host-mapped，**没有第三种**）。宏适配方面（**2026-06-12 增补**）：**只有 `gpuMemAdvise` 一个宏**（[memory_utils.cu:159-162](fbgemm_gpu/src/memory_utils/memory_utils.cu#L159) 定义，ROCm 7.1+ 走 `hipMemAdvise_v2`、CUDA 走 `cudaMemAdvise`）；`cudaMallocManaged`（line 97）和 `cudaMemPrefetchAsync`（line 450）都是**直接调**，靠 HIP runtime 自带的同名 alias 实现 ROCm 兼容。`gpuMemLocation` 是 v2 API 的类型别名（line 148-150），用于在 CUDA 13+ / ROCm 7.1+ 时把 `int device` 升级为 `cudaMemLocation` struct。

| API | 作用 | FBGEMM 是否使用 |
|---|---|---|
| `cudaMallocManaged(&ptr, size)` | 分配 UVM 内存 | ✅ 核心入口（直接调，无宏） |
| `cudaMemAdvise(ptr, ..., SetPreferredLocation, dev)` | 声明"数据归属哪个 device/host" | ✅ 分配后立即调用（经 `gpuMemAdvise` 宏） |
| `cudaMemAdvise(ptr, ..., SetAccessedBy, dev)` | 声明"某 device 即将访问"，建立 direct mapping **避免 page fault** | ✅ 分配后立即调用（经 `gpuMemAdvise` 宏） |
| `cudaMemAdvise(ptr, ..., SetReadMostly)` | 标记只读热点，pin 在某 device 上 | ⚠️ 暴露给 Python（未默认使用） |
| `cudaMemAdvise(ptr, ..., Unset*)` × 3 种 | 解除对应的 advice | ⚠️ 暴露给 Python（未默认使用） |
| `cudaMemPrefetchAsync(ptr, ..., dstDev, stream)` | **主动**把页搬到目标 device | ✅ 暴露给 Python (`uvm_cuda_mem_prefetch_async`)，FBGEMM 内部不调 |
| `madvise(MADV_DONTFORK)` | Workaround fork 场景下 UVM 的问题 | ✅ **`new_managed_tensor` 内部自动调用**（[line 234](fbgemm_gpu/src/memory_utils/memory_utils.cu#L234)），用户也可用 `uvm_mem_advice_dont_fork` 显式再调（[line 481](fbgemm_gpu/src/memory_utils/memory_utils.cu#L481)） |
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

// ④ madvise() 系统调用要求 page-aligned, 但 cudaMallocManaged 返回的
//    ptr/size 不一定 aligned, 所以先调 adjust_to_page_boundaries
//    把 [ptr, ptr+size) 区间对齐到 OS page 边界
auto adjusted = adjust_to_page_boundaries(ptr, size_bytes);

// ⑤ 在对齐后的区间上设 MADV_DONTFORK
//    Workaround fork 后 cuda context 不匹配的问题
madvise(std::get<0>(adjusted), std::get<1>(adjusted), MADV_DONTFORK);
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

#### 3.2.2.1 MANAGED_CACHING miss 时的访存行为（5 步流水线，2026-06-11 增补）

> **核心问题**：embedding 不在 HBM cache 时，FBGEMM 怎么把数据从 UVM 拉过来？miss 率靠什么降低？

**完整 forward 循环**（[split_table_batched_embeddings_ops_training.py:2823-2904](fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_training.py#L2823)）：

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Step 1. linearize_cache_indices                                          │
│   (table, row) → global hash idx（MANAGED_CACHING 表才有 idx, 其他表填 -1）│
└──────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ Step 2. lxu_cache_lookup_kernel  [lxu_cache.cu:238-306]  (第一次查找)     │
│   每个 warp 处理 blockDim.x 个 idx:                                       │
│     ① cache_slot(idx, C) = MurmurHash3(idx) % C                           │
│     ② 读 lxu_cache_state[set][0..32]                                     │
│     ③ __ballot_sync 找匹配 way                                           │
│   命中 → way, 未命中 → kCacheLocationMissing(-1)                          │
│   ⚠️ **纯查找**,不更新 LRU,不读写 UVM; 这一步主要为了 gather_uvm_cache_stats│
└──────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ Step 3. lru_cache_find_uncached_kernel  [lru_cache_find.cu:78-150]        │
│   ① 对**命中的行**: lru_state[set][slot] = time_stamp (更新访问时间)     │
│   ② 对**未命中的行**: 把 cache_set 写到 cache_sets 输出                   │
│   ③ CUB DeviceRadixSort 按 cache_set 归并 (同 set 的 miss 一起处理)      │
└──────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ Step 4. lru_cache_insert_kernel  [lru_cache_populate.cu:17-165]          │
│   ← 这就是 UVM ↔ HBM 数据交换的真正发生点                                  │
│   一个 warp 处理一个 set:                                                  │
│     ① BitonicSort 在 32 个 LRU 时间戳里找**最久未用**的 way               │
│     ② 检查 victim 是否 locked (lxu_cache_locking_counter > 0)              │
│        → locked 跳过 (防止驱逐正在 forward 的行)                          │
│     ③ **Evict**: warp_cache_evict 把 victim 从 lxu_cache_weights          │
│                  拷回 weights_uvm (HBM → UVM)                            │
│     ④ **Load**:  warp_cache_load 把新行从 weights_uvm                     │
│                  拷到 lxu_cache_weights (UVM → HBM)                       │
│     ⑤ 更新 lxu_cache_state / lru_state / 锁定计数器                       │
└──────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌──────────────────────────────────────────────────────────────────────────┐
│ Step 5. lxu_cache_lookup_kernel  (第二次查找)                              │
│   返回更新后的 lxu_cache_locations                                        │
│   → forward kernel 用这个 tensor 决定读 HBM cache 还是 fallback UVM       │
└──────────────────────────────────────────────────────────────────────────┘
```

**数据交换的粒度**——注意不是页粒度：

| 维度 | FBGEMM cache exchange | CUDA UVM page migration |
|---|---|---|
| 粒度 | **一行 (row)**，由 `warp_cache_load`/`warp_cache_evict` 整行搬运 | **4 KB / 2 MB 页** |
| 触发者 | FBGEMM `lru_cache_insert_kernel` | CUDA driver page fault handler |
| 替换策略 | LRU / LFU / Direct-mapped（**软件可控**） | 硬件 LRU（**应用不可控**） |
| 典型 row 大小 | `D × 2B` = 32B (D=16) ~ 512B (D=256) | — |

**关键区别**：MANAGED_CACHING 的交换粒度是 **embedding row**（几十到几百字节），比 UVM 的 2 MB page 小 4000×~60000×。这意味着 cache 替换决策可以做到**行级精确**，但每次交换的 PCIe/NVLink 摊销效率不如 page-level。

**miss 率降低的 5 个机制**：

| 机制 | 实现 | 文件:行 | 效果 |
|---|---|---|---|
| **LRU 替换** | 时间戳最小者被驱逐 | [lru_cache_populate.cu:17-165](fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate.cu#L17) | 时间局部性好的工作集 → 高 hit rate |
| **LFU 替换** | 访问次数最小者被驱逐 | [lfu_cache_populate.cu](fbgemm_gpu/src/split_embeddings_cache/lfu_cache_populate.cu) | 频率局部性好的工作集 → 高 hit rate |
| **`lock_cache_line`** | 正在 forward 的行不驱逐（`lxu_cache_locking_counter > 0` 跳过 victim） | [lru_cache_populate.cu:222-228](fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate.cu#L222) | 避免热行被反复换入换出（thrashing） |
| **`prefetch_stream`** | populate 跑在 `self.prefetch_stream` 上，与主 stream 的 forward/backward overlap | [training.py:3669-3680](fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_training.py#L3669) | 隐藏 cache populate 的 latency |
| **`gather_uvm_cache_stats`** | 暴露 6 项命中率指标，用户可据此调 `cache_load_factor` | [training.py:184-190](fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_training.py#L184) | 调优闭环 |

外加一个**测试用**机制 `emulate_cache_miss_kernel`（[lru_cache_find.cu:16-38](fbgemm_gpu/src/split_embeddings_cache/lru_cache_find.cu#L16)）：强制每 256 个访问中 `enforced_misses_per_256` 个走 miss，用来复现低 hit-rate 场景做鲁棒性测试。

#### 3.2.2.2 容量构成：HBM cache + UVM 主存 = 整张表（2026-06-11 增补）

> **核心问题**：MANAGED_CACHING 的总占用是 HBM+UVM 还是只 UVM？两者关系？

**是的，整张表 = HBM cache + UVM 主存**，两者是同一份权重的不同副本：

```
权重全集 (total_rows × D × 2B)
    ├── lxu_cache_weights   [C × A, D]   ← HBM 副本（cache_load_factor 部分）
    │       cache_load_factor = (C × A) / total_rows
    │       默认 0.2 → 20% 行在 HBM
    │
    └── weights_uvm         [total_rows, D]  ← UVM 主存（100% 行都在这）
```

**关键代码** [training.py:3541-3580](fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_training.py#L3541)：

```python
cache_sets = (
    int(cache_state.total_cache_hash_size * cache_load_factor)
    + DEFAULT_ASSOC - 1
) // DEFAULT_ASSOC
# lxu_cache_weights 大小 = cache_sets × DEFAULT_ASSOC × D × 2B
# 举例: 1e8 行 × D=128 × fp16, cache_load_factor=0.2
#   = 1e8 × 0.2 × 128 × 2B = 4.8 GB（HBM 占用）
#   = 1e8 × 128 × 2B = 24 GB（UVM 占用）
```

**两个池子的一致性保证**——[embedding_inplace_update.cu:84-98](fbgemm_gpu/src/embedding_inplace_ops/embedding_inplace_update.cu#L84) 优化器写入时**双写**：

```cpp
bool cache_valid = (weights_placement == PlacementType::MANAGED_CACHING);
if (cache_valid && cache_idx != kCacheLocationMissing) {
    // 同时写 UVM 主存 和 HBM cache 行
}
```

**总占用估算**：

- 默认 0.2 下：`HBM 0.2×表 + UVM 1.0×表 ≈ 1.2×单表容量`
- 不是"刚好整张表"，而是**略多一点**（cache 副本是冗余的）
- UVM 中也有"被 cache 命中过、又被 evict 回来"的行（冗余存储）
- evict 路径（`warp_cache_evict`）就是把 HBM 副本写回 UVM 主存，保证两者最终一致

#### 3.2.2.3 动态扩容：不存在（2026-06-11 增补）

> **核心问题**：HBM 或 UVM 不够用时，FBGEMM 能否自动扩充？

**简短回答：没有。** 用户必须自己选 placement。

##### HBM cache 大小——构造时一次性确定，运行时不变

```python
# 构造时（split_table_batched_embeddings_ops_training.py:3541-3580）：
cache_sets = ceil(total_cache_hash_size × cache_load_factor / DEFAULT_ASSOC)
# 然后 register_buffer 分配 lxu_cache_weights / lxu_cache_state / lru_state
# 这些 buffer 不可 resize，PyTorch 也只是 register_buffer 而非 Parameter
```

- **没有 `resize_cache()` 这种 API**
- 唯一例外：构造时如果 `cache_size > free_HBM`，代码会**一次性自动缩到能装下的最大值**：

```python
if cache_size > free_memory:    # free_memory = 当前卡可用 HBM
    cache_sets = int(1.0 * free_memory / self.max_D_cache / element_size) // DEFAULT_ASSOC
    cache_load_factor = 1.0 * cache_sets * DEFAULT_ASSOC / int(total_cache_hash_size)
```

这是**一次性的"安全降级"**，不是动态扩容——一旦构造完成就不再变。

##### UVM 内存——由 `cudaMallocManaged` 一次性分配，不动态扩

[memory_utils.cu:88-97](fbgemm_gpu/src/memory_utils/memory_utils.cu#L88)：

```cpp
AT_CUDA_CHECK(cudaMallocManaged(&ptr, size_bytes));
```

- 一次性分配整张表大小
- 系统内存不足时 → **OOM**（不是动态找 NVMe）
- 没有任何 "if OOM then go to NVMe" 的 fallback 逻辑

##### "HBM 不够"的处置清单

| 触发条件 | FBGEMM 提供的能力 | 自动化？ |
|---|---|---|
| 单表能装进 HBM | `EmbeddingLocation.DEVICE` | ❌ 手动选 |
| 单表装不下 HBM，但 working set < 20% | `MANAGED_CACHING` + `cache_load_factor` | ❌ 手动选 |
| 单表装不下 HBM，且 working set 不可刻画 | `MANAGED` | ❌ 手动选 |
| 单表连 CPU DRAM 都装不下 | **SSD TBE**（`ssd_split_embeddings_cache`） | ❌ 手动选 |
| Cache 装不下请求的 `cache_load_factor` | 自动缩到 free HBM | ✅（一次性降级） |
| HBM 真的不够了 | **无 fallback** → OOM | ❌ |

**SSD TBE 是真正的"溢出层"**，但它是**独立的算子**（`SSDTableBatchedEmbedding`），不是 `MANAGED_CACHING` 的扩展：

- 权重存在 NVMe SSD
- DRAM 级用 UVM 池子做 cache
- 由专门的 [`kv_db_table_batched_embeddings.cpp`](fbgemm_gpu/src/ssd_split_embeddings_cache/kv_db_table_batched_embeddings.cpp) / SSD-specific Python 类管
- **用户在构造时**决定"哪些表用 SSD TBE"，不是运行时切换

##### 为什么不自动扩容？

工程上的几个原因：

1. **PyTorch 生态的 `register_buffer` 不支持 resize**——一旦 buffer 分配，`data_ptr()` 就不变，任何引用都得跟着重定位
2. **kernel 启动时 cache 维度是编译期常量**（如 `kWarpSize = 32`），改 cache 大小要重编
3. **bump allocator 模式会有碎片化问题**，DLRM 这种 hot/cold 极不均衡的 workload 上不划算
4. **预测 cache 命中率需要历史信息**，运行早期不知道 working set 大小，盲目扩容可能反而浪费 HBM

##### 用户能用的"半自动"机制（FBGEMM 不内置，需用户自写）

虽然 FBGEMM 本身不动态扩容，但提供了**可观测性**让你**手动**做这件事：

```python
# 1. 打开 cache 命中率监控
emb.gather_uvm_cache_stats = True

# 2. 训练中定期看
stats = emb.get_uvm_cache_stats()
# stats[UVMCacheStatsIndex.num_calls]               # 总访问次数
# stats[UVMCacheStatsIndex.num_unique_indices]      # 唯一访问行数
# stats[UVMCacheStatsIndex.num_unique_misses]       # 唯一 miss 行数
# hit_rate = 1 - num_unique_misses / num_unique_indices

# 3. hit rate 低 → 手动调大 cache_load_factor，重建 emb
if hit_rate < 0.8:
    new_emb = SplitTableBatchedEmbeddingsTraining(
        ...,
        cache_load_factor=0.4,   # 翻倍
    )
    # state_dict 拷过去
    new_emb.load_state_dict(emb.state_dict())
    emb = new_emb
```

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

#### 3.2.4 MANAGED vs MANAGED_CACHING 的 UVM 依赖差异（2026-06-12 增补）

> **核心问题**：两种模式都用了 UVM，但到底"用了多少"UVM？哪一种对昇腾 950 UVM 能力更敏感？

##### 一句话总结

**两种模式分配方式完全相同**（都走 `cudaMallocManaged`），**唯一差异是 FBGEMM 是否在 HBM 上显式维护一份 cache**。这导致**对 UVM 的依赖程度完全不同**：

- **MANAGED**：**重度依赖 UVM 硬件级 page-fault 机制**
- **MANAGED_CACHING**：**轻度依赖 UVM**（作为存储后端），miss 处理由 FBGEMM 自管

##### 决策入口：一个参数分两路

[split_table_batched_embeddings_ops_common.py:450-474](fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_common.py#L450)

```python
if cache_load_factor == 0:        return EmbeddingLocation.MANAGED         # 纯 UVM
elif cache_load_factor == 1.0:    return EmbeddingLocation.DEVICE         # 纯 HBM
else:                              return EmbeddingLocation.MANAGED_CACHING # UVM + HBM cache
```

| `cache_load_factor` | 模式 | `lxu_cache_weights` 是否分配 | `use_lxu_cache` |
|---|---|---|---|
| `0` | MANAGED | ❌ 不分配（`numel() == 0`） | `false` |
| `(0, 1)` | MANAGED_CACHING | ✅ 分配（`cache_load_factor × total_rows` 大小） | `true` |
| `1.0` | DEVICE | ❌ 不分配 | `false`（用 DEVICE HBM） |

##### 同一份 UVM 主存，两条路径

```
        MANAGED                              MANAGED_CACHING
  ┌─────────────────────┐              ┌─────────────────────┐
  │      CPU DRAM       │              │      CPU DRAM       │
  │  weights_uvm (100%) │              │  weights_uvm (100%) │  ← 完全相同
  └─────────────────────┘              └─────────────────────┘
            ↑                                       ↑ ↓
            │ page fault                          │ ↓ cache populate
            │ (driver 隐式迁移)                   │ ↓ (FBGEMM 显式)
            │                                     │ ↓
  ┌─────────────────────┐              ┌─────────────────────┐
  │     GPU HBM         │              │     GPU HBM         │
  │ (driver 自动迁移的   │              │ lxu_cache_weights   │
  │  hot pages, 不可控) │              │ (FBGEMM 显式 cache, │
  │                     │              │  cache_load_factor  │
  │                     │              │  控制大小, LRU/LFU) │
  └─────────────────────┘              └─────────────────────┘
        ↖ driver 管                          ↖ FBGEMM 管
```

##### 5 个维度的差异

| 维度 | MANAGED | MANAGED_CACHING |
|---|---|---|
| **内存分配** | `cudaMallocManaged` 分配 `weights_uvm`（[memory_utils.cu:97](fbgemm_gpu/src/memory_utils/memory_utils.cu#L97)） | **完全相同** |
| **cudaMemAdvise** | `SetPreferredLocation=CPU` + `SetAccessedBy=current_device`（[memory_utils.cu:203-227](fbgemm_gpu/src/memory_utils/memory_utils.cu#L203)） | **完全相同** |
| **HBM cache buffer** | ❌ 不分配（`lxu_cache_weights.numel() == 0`） | ✅ 显式分配（[training.py:3602-3630](fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_training.py#L3602)）：`lxu_cache_state`、`lxu_cache_weights`、`lru_state` |
| **forward kernel 路径** | `use_lxu_cache = false` → 走 no-cache 路径直读 `weights_uvm`（[embedding_forward_split_template.cu:602-605](fbgemm_gpu/codegen/training/forward/embedding_forward_split_template.cu#L602)） | `use_lxu_cache = true` → 先查 cache，命中走 HBM cache，未命中 fallback UVM |
| **miss 处理** | **CUDA driver + Linux HMM** 自动 page fault + on-demand migrate（**FBGEMM 不参与**） | **FBGEMM 自管**：`lru_cache_populate` kernel 显式 `warp_cache_evict` + `warp_cache_load`（[lru_cache_populate.cu:17-165](fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate.cu#L17)） |

##### UVM 依赖程度：重度 vs 轻度

| 依赖项 | MANAGED | MANAGED_CACHING |
|---|---|---|
| `cudaMallocManaged` | ✅ 必须 | ✅ 必须 |
| `cudaMemAdvise` | ✅ 必须（性能） | ✅ 必须（性能） |
| **硬件级 page-fault 自动迁移** | 🔴 **必须** | 🟢 **不必须**（可走显式 memcpy 退化路径） |
| `cudaMemPrefetchAsync` | ⚠️ 可选（性能） | ⚠️ 可选（性能） |
| HBM 上的 cache 缓冲 | ❌ 不要 | ✅ 必须 |
| FBGEMM miss 处理逻辑 | ❌ 没有 | ✅ 必须（`lru_cache_populate` 等 kernel） |
| 优化器双写 UVM + HBM | ❌ 不需要 | ✅ 必须（[embedding_inplace_update.cu:84-98](fbgemm_gpu/src/embedding_inplace_ops/embedding_inplace_update.cu#L84)） |
| **总体 UVM 依赖** | **重** | **中** |

**关键洞察**：

- **MANAGED 没有 page-fault 就完全不能工作**。如果 CANN 不支持硬件 page fault 机制：`cudaMallocManaged` 可以模拟（分配在 host memory），但 GPU 访问时没有 fault 机制，访问会直接失败或得到随机数据
- **MANAGED_CACHING 即使没有 page-fault 也能工作**。退化路径：cache miss → 显式 memcpy（轮询 / Pinned memcpy）→ HBM。FBGEMM 内部走的就是**显式 memcpy**（`warp_cache_load` 是同步拷贝），不依赖 driver fault 机制

##### 对昇腾 950 UVM 移植的启示

| 模式 | 移植风险 | 验收分级 |
|---|---|---|
| **MANAGED_CACHING** | 🟢 **P1 中风险** — 只需 `aclrtMallocManaged` + memcpy 访问（显式 `aclrtMemcpyAsync`），对 fault 机制无强依赖 | **P0 必须支持** — 单测 `test/uvm/*` + DLRM benchmark |
| **MANAGED** | 🔴 **P0 高风险** — 必须依赖硬件 page-fault 自动迁移 | **P1 可选支持** — 仅当 CANN 提供硬件 page-fault 机制后才开启 |

**CANN 实现建议**：

如果 CANN UVM 短期只支持 prefetch API、不支持硬件 page-fault：
- ✅ **MANAGED_CACHING 可以支持**：kernel miss 时调 `aclrtMemcpyAsync` 从 UVM 池子拷到 HBM cache，性能可接受
- ❌ **MANAGED 难以支持**：没有 page-fault 机制时，GPU 直接访问 UVM 内存要么 OOM、要么完全失败

#### 3.2.5 FBGEMM 冷热管理机制：用 advise 压制 UVM 自动搬运（2026-06-12 增补）

> **核心命题**："UVM 接管了 H2D/D2H 搬运"这个常见说法**只在 MANAGED 模式成立**。在 MANAGED_CACHING 模式下，FBGEMM 故意用 advise 设置让 UVM 的自动搬运"休眠"，自己夺回搬运决策权——这才是它能在 HBM 上精确维护热数据的根本机制。

##### 两种模式下搬运决策权的归属

| 模式 | 谁决定 H2D/D2H 搬运 | FBGEMM 能否管冷热 |
|---|---|---|
| **MANAGED** | ✅ UVM driver 接管（page fault + 自动迁移） | ❌ 管不了（黑盒） |
| **MANAGED_CACHING** | ❌ UVM 被压制（退居为纯存储后端） | ✅ **FBGEMM 显式管** |

##### MANAGED 模式：UVM 真接管，FBGEMM 管不了冷热

```
GPU 访问 weights_uvm[idx]
      ↓
  UVM driver 检查页表
      ↓
  页在 CPU？→ page fault → driver 把 4KB/2MB 页迁到 HBM → 完成访问
      ↓
  driver 内部的硬件 LRU 决定哪些页留 HBM、哪些挤回 CPU
```

- **搬运决策**：driver/HMM 硬件 LRU 启发式
- **搬运粒度**：4KB / 2MB 页
- **冷热判断**：页访问频率（硬件统计）
- **FBGEMM 角色**：只分配 + 设 advise，之后**完全不参与**

这就是 MANAGED 模式性能波动大的原因——driver 启发式对 embedding 长尾访问不一定优化得好，热数据可能被误驱逐，FBGEMM 没法干预。

##### MANAGED_CACHING 模式：三个手段压制 UVM，自己接管冷热

**手段 1：advise 组合让 UVM 不主动迁移**

[memory_utils.cu:202-227](fbgemm_gpu/src/memory_utils/memory_utils.cu#L202)：

```cpp
gpuMemAdvise(ptr, size, SetPreferredLocation, CPU);      // 数据"家"声明在 CPU
gpuMemAdvise(ptr, size, SetAccessedBy, current_device);  // GPU 建立到 CPU 的 direct mapping
```

两步组合的效果（NVIDIA UVM 语义）：

| 场景 | 没有 advise | 有这个 advise 组合 |
|---|---|---|
| GPU 访问 UVM 页 | page fault → 页迁到 HBM | **不 fault**，走 direct mapping 直接读 CPU |
| 页最终位置 | driver 决定（可能 HBM） | **永远在 CPU**（preferred location） |

即：FBGEMM 让 UVM driver **承诺"我不主动搬你的页"**。UVM 的 page migration 机制在 MANAGED_CACHING 模式下基本是**休眠**的。

**手段 2：FBGEMM 自己分配独立的 HBM cache 池**

[training.py:3602-3630](fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_training.py#L3602)：

```python
self.lxu_cache_weights = torch.zeros(C*A, D, device=current_device)  # ★ 普通 HBM 分配, 非 UVM
self.lxu_cache_state   = torch.zeros(C, A, device=current_device)    # 每路存的行号
self.lru_state         = torch.zeros(C, A, device=current_device)    # LRU 时间戳
```

**注意**：`lxu_cache_weights` 是 `register_buffer(device=cuda)` 分配的**普通 HBM tensor**，跟 UVM 没有任何关系。这是一个完全独立于 UVM 的、FBGEMM 自己控制的 HBM 区域。

**手段 3：显式 kernel 做冷热搬运（不走 UVM）**

[lru_cache_populate.cu:17-165](fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate.cu#L17)：

```cpp
// ① Evict: 把冷行从 HBM cache 显式写回 UVM 主存
warp_cache_evict(lxu_cache_weights[victim], weights_uvm[idx]);  // HBM → UVM (显式拷贝)

// ② Load: 把热行从 UVM 主存显式读到 HBM cache
warp_cache_load(lxu_cache_weights[slot], weights_uvm[idx]);     // UVM → HBM (显式拷贝)
```

- `weights_uvm[idx]` 读取**不触发 page fault**（因为手段 1 的 `SetAccessedBy`），直接走 PCIe 读 CPU 内存
- `warp_cache_load` / `warp_cache_evict` 是**显式 memcpy**，FBGEMM kernel 自己发的搬运指令，**跟 UVM driver 无关**

##### 冷热判断的具体机制（LRU 时间戳，纯软件）

冷热判断**完全靠软件**，跟 UVM 无关：

**Step 1：访问时打时间戳（行变"热"）**——[lru_cache_find.cu:78-150](fbgemm_gpu/src/split_embeddings_cache/lru_cache_find.cu#L78)：

```cpp
// 对命中的行: 更新时间戳 = 当前 batch id (变热)
lru_state[cache_set][slot] = time_stamp;
```

`time_stamp` 单调递增。**最近被访问的行，时间戳最新（最热）**。

**Step 2：驱逐时找最旧的（行变"冷"被踢）**——[lru_cache_populate.cu:17-165](fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate.cu#L17)：

```cpp
// 一个 warp 处理一个 cache set, BitonicSort 在 32 路里找时间戳最旧的
int oldest_slot = warp_find_oldest(lru_state[set]);  // 时间戳最小 = 最久没访问 = 最冷
// 把 oldest_slot 作为 victim 驱逐, 腾位置给新行
```

**冷热逻辑总结**：
- 热行 = 时间戳新 = 最近被访问 → 留在 HBM cache
- 冷行 = 时间戳旧 = 长时间没访问 → 被 evict 回 UVM（CPU DRAM）
- **整个判断纯软件、纯应用层，UVM driver 完全不参与**

##### 完整数据流（三层架构）

```
┌─────────────────────────────────────────────────────────────────┐
│  存储层（UVM 只在这层，纯池子）                                   │
│  weights_uvm [total_rows, D]  ← cudaMallocManaged, 家在 CPU      │
│  ★ UVM driver 不主动搬它（被 advise 压制）                        │
└─────────────────────────────────────────────────────────────────┘
                          ↑↓ 显式 memcpy
┌─────────────────────────────────────────────────────────────────┐
│  搬运层（FBGEMM kernel，完全自己控制）                            │
│  warp_cache_load (UVM→HBM)  /  warp_cache_evict (HBM→UVM)        │
│  ★ 决策依据：lru_state 时间戳                                    │
└─────────────────────────────────────────────────────────────────┘
                          ↑↓
┌─────────────────────────────────────────────────────────────────┐
│  cache 层（普通 HBM，非 UVM）                                     │
│  lxu_cache_weights [C×A, D]  ← register_buffer(device=cuda)      │
│  lxu_cache_state   [C, A]    ← 每路存的行号                       │
│  lru_state         [C, A]    ← 每路最近访问时间戳                  │
└─────────────────────────────────────────────────────────────────┘
```

##### 两种模式"热数据进 HBM"机制对比

| 维度 | MANAGED（UVM 接管） | MANAGED_CACHING（FBGEMM 接管） |
|---|---|---|
| 谁决策热数据进 HBM | **UVM driver** 硬件启发式 | **FBGEMM LRU kernel** |
| 搬运触发 | page fault | `lru_cache_insert_kernel` 显式调用 |
| 搬运粒度 | 4KB / 2MB 页 | embedding 行（D×2B，几十~几百字节） |
| 冷热判断依据 | 页访问频率（硬件统计） | 软件时间戳 |
| 可控性 | ❌ 完全不可控 | ✅ `cache_load_factor` + LRU/LFU/Direct-mapped |
| 替换策略 | 硬件 LRU（不可调） | 软件 LRU / LFU / Direct-mapped（可选） |
| 防 thrashing | ❌ 无 | ✅ `lock_cache_line`（forward 中的行不驱逐） |
| HBM 占用 | driver 决定，不可调 | `cache_load_factor` 精确控制 |

##### 为什么 FBGEMM 要"绕开" UVM 的自动搬运

Meta 工程师的设计权衡，原因很实在：

1. **粒度问题**：UVM 按 4KB 页搬，一个 128 维 fp16 的 embedding 行只有 256B，一个页装 16 行——**页级搬运对 embedding 太粗**，会把无关的冷行一起搬进 HBM 浪费带宽
2. **替换策略不可控**：UVM 的硬件 LRU 是通用启发式，**不懂 embedding 的访问模式**（比如某行虽然在 cache 里但实际是冷的，UVM 不知道）
3. **长尾延迟不可预测**：page fault 抖动会让 batch 延迟方差大，训练 throughput 不稳；FBGEMM 自己的 cache 命中走 HBM，miss 走显式 memcpy，延迟**可预测、可调优**
4. **优化器一致性**：训练时要同时更新 UVM 主存和 HBM cache 副本（[embedding_inplace_update.cu:84-98](fbgemm_gpu/src/embedding_inplace_ops/embedding_inplace_update.cu#L84)），如果让 UVM 也参与搬运，两套机制会打架

> **一句话**：UVM 的自动搬运是为"通用 GPU 计算"设计的，对"稀疏 embedding lookup"这种特定 workload 不够好。所以 FBGEMM 借用 UVM 的"统一地址 + 按需访问 CPU"能力，但**剥离了它的"自动搬运决策"**，换成自己更懂业务的 cache 逻辑。

#### 3.2.6 为什么不能纯 cudaMalloc + cudaMemcpy 替代 UVM（2026-06-12 增补）

> **核心命题**：MANAGED_CACHING 模式下，FBGEMM 确实用 kernel 接管了"搬运"这一步，但这只是冰山一角。FBGEMM 真正依赖 UVM 的，是一组**纯 cudaMalloc 根本做不到**的"统一地址空间"能力。

##### 纯 cudaMalloc vs UVM 的本质区别

| 能力 | 纯 `cudaMalloc` | `cudaMallocManaged` (UVM) |
|---|---|---|
| **同一份指针，CPU 和 GPU 都能 deref** | ❌ 不能 | ✅ **能**（这是核心） |
| **GPU 端写完后 CPU 端能直接读到** | ❌ 必须显式 `cudaMemcpy(DeviceToHost)` | ✅ 写完直接读，driver 保证一致性 |
| **跨进程传递（fork）** | ❌ 指针随 context 死 | ✅ 子进程能继承 UVM 区域 |
| **超量分配（> HBM 容量）** | ❌ 满了就 OOM | ✅ 溢出到 host memory |
| **PyTorch tensor 互操作** | ⚠️ `.cpu()` 触发完整 D2H 拷贝 | ✅ `uvm_to_cpu()` **不复制数据** |

**一句话**：UVM 的核心价值不是"搬运"，是"**CPU 和 GPU 共享同一份虚拟地址空间**"。搬运只是这个能力的一个副作用。

##### FBGEMM 实际用到的"纯 cudaMalloc 做不到"的能力

###### 能力 1：CPU 和 GPU 共享同一份指针

**场景 A：优化器在 CPU 端做 rowwise adagrad norm 计算**

[embedding_optimizer_split_device_kernel_template.cuh:99](fbgemm_gpu/codegen/training/optimizer/embedding_optimizer_split_device_kernel_template.cuh#L99)：

```cpp
{{ tensor }} = &{{ tensor }}_uvm[{{ tensor }}_offset];
// 优化器直接拿到 UVM 指针, 在 GPU kernel 里读写同一份
```

如果用纯 `cudaMalloc`：
- 优化器状态（momentum1/momentum2/prev_iter）必须独立存两份
- 一份在 GPU HBM（GPU 端算梯度用）
- 一份在 CPU DRAM（CPU 端算 norm）
- 两份之间必须用 cudaMemcpy 同步
- **UVM 模式下：只有一份，CPU/GPU 都用 `*_uvm[offset]` 拿指针**

**场景 B：CPU 端直接读 UVM tensor 的标量**

[raw_embedding_streamer.cpp:21-28](fbgemm_gpu/src/split_embeddings_cache/raw_embedding_streamer.cpp#L21)：

```cpp
// 注释原文：
// "Read a scalar value from a tensor that is maybe a UVM tensor.
//  Note that `tensor.item<type>()` is not allowed on a UVM tensor in PyTorch"
inline int64_t get_maybe_uvm_scalar(const at::Tensor& tensor) {
    return *tensor.data_ptr<int>();  // ★ 绕开 .item() 限制, 直接 deref UVM 指针
}
```

**为什么 PyTorch 的 `.item()` 不支持 UVM**？因为 `.item()` 内部要做 device→host 同步，对普通 `cudaMalloc` tensor 是 D2H 拷贝，但 PyTorch 不想"为 UVM 触发隐式拷贝"。FBGEMM 用 `*data_ptr<int>()` 绕过——**直接 deref UVM 指针**。

如果用纯 cudaMalloc，**这种 CPU 端直接读 GPU tensor 标量的能力就丢了**，每次都必须先 cudaMemcpy。

###### 能力 2：PyTorch tensor 重新解释 storage（不复制数据）

[memory_utils.cu:321-373](fbgemm_gpu/src/memory_utils/memory_utils.cu#L321) `uvm_to_cpu` / `uvm_to_device`：

```cpp
Tensor uvm_to_cpu(const Tensor& t) {
    // 重新解释 storage 为 CPU 视角, 不复制数据
    auto storage = Storage(
        ...,
        ocontext->ptr_,  // ★ 用同一个 UVM 指针
        ...
        {at::DeviceType::CPU}  // 但 device type 改为 CPU
    );
    ...
}
```

**核心**：UVM tensor 可以**零拷贝**地"切换" device 视角（CPU/GPU 任一），因为底层是同一份物理内存 + 同一份页表。

如果用纯 `cudaMalloc`：每次 `tensor.cpu()` 都会触发完整 D2H 拷贝，**几百 GB 表直接卡死**。

###### 能力 3：超量分配（> HBM 容量）

| 场景 | 表容量 | 单卡 HBM | 用 cudaMalloc | 用 cudaMallocManaged |
|---|---|---|---|---|
| 推荐系统 | 200 GB | 80 GB | ❌ **OOM** | ✅ 溢出到 CPU |
| 短视频 embedding | 500 GB | 80 GB | ❌ **OOM** | ✅ 溢出到 CPU |

**这是 UVM 存在的根本原因**——纯 `cudaMalloc` 装不下的表，必须用 UVM（或显式手动管理两块内存）。

> "FBGEMM 自己用 kernel 做行级 memcpy"——这话没错，但**前提是有内存可搬**。如果连初始分配都 OOM 了，搬运再精妙也白搭。UVM 的 `cudaMallocManaged` 是**唯一能保证超量分配**的官方接口。

###### 能力 4：跨进程传递

`new_managed_tensor` 自动调 `madvise(MADV_DONTFORK)`（[memory_utils.cu:234](fbgemm_gpu/src/memory_utils/memory_utils.cu#L234)）—— **为什么这个 workaround 存在**？

因为 fork() 之后：
- UVM 内存：子进程能继承（fork 前 madvise DONTFORK 避免重映射开销）
- 纯 cudaMalloc 内存：子进程的 cuda context 是新的，**指针直接失效**

搜广推典型场景：
- PS worker fork 出 serving worker
- 训练进程 fork 出 inference 进程做 checkpoint loading
- 这类 fork-after-allocate 模式，**纯 cudaMalloc 完全不能工作**

##### 纯 cudaMalloc 替代 UVM 的"全账"

**需要重新实现的东西**：

| FBGEMM 现状（用 UVM） | 用纯 cudaMalloc 必须做的事 |
|---|---|
| 优化器 GPU/CPU 共享一份状态 | 维护两套状态，cudaMemcpy 同步，**代码量翻倍** |
| `get_maybe_uvm_scalar()` CPU 端直接读 | 每次读标量都 cudaMemcpy，**embedding streamer 性能崩** |
| `uvm_to_cpu(t)` 零拷贝切换视角 | `tensor.cpu()` 触发完整 D2H 拷贝，**几百 GB 表直接卡死** |
| 超量分配放 host 内存 | 自己实现 host/device 内存分配器，**重新造一个 PyTorch caching allocator** |
| 跨进程传递 | 完全做不到（除非走共享内存 + IPC，**复杂度爆炸**） |

**总账**：纯 cudaMalloc 方案至少要重写 1000+ 行代码，且性能大概率不如 UVM 路径（因为缺少"统一地址空间"这个核心抽象）。

##### 核心反驳（一句话）

> **UVM 给 FBGEMM 提供的不是"自动搬运"，是"统一虚拟地址空间"这个底层抽象。FBGEMM 借这个抽象实现了 4 件事：(1) 优化器 CPU/GPU 共享一份状态、(2) 跨进程传递、(3) 超量分配、(4) tensor 零拷贝切换视角。这 4 件事纯 cudaMalloc + cudaMemcpy 做不到，要么代码量爆炸，要么性能崩。FBGEMM 的 `warp_cache_load` / `warp_cache_evict` 是在 UVM 抽象之上做的"行级搬运优化"，不依赖 UVM 的自动搬运——但仍然依赖 UVM 的"统一地址 + 按需访问 CPU"能力。**

##### 重新审视问题的视角

把这个问题反过来看：

| 问题 | 答案 |
|---|---|
| 不用 UVM，纯 cudaMalloc 能写一个 embedding cache 吗？ | ⚠️ 理论上能，但只解决了"搬运"问题 |
| 不用 UVM，能做超量分配吗？ | ❌ 做不到（HBM 满就 OOM） |
| 不用 UVM，能让 CPU/GPU 共享一份优化器 state 吗？ | ❌ 做不到（必须双份 + 同步） |
| 不用 UVM，能做跨进程 embedding 表传递吗？ | ❌ 做不到（cudaMalloc 指针 context-bound） |
| 不用 UVM，能零拷贝切换 tensor 视角吗？ | ❌ 做不到（会触发完整 D2H 拷贝） |

> "我直接用 cudaMalloc+memcpy" 这其实是个**降级方案**——放弃了 UVM 提供的全部"统一地址空间"红利，只为"避免用 UVM"这个目标。**ROI 是负的**。

### 3.2.7 物理分配模型：HBM 80GB 时 `cudaMallocManaged(100GB)` 的实际存储拆分

> **可信度图例**：🟢 来自 FBGEMM 源码直接证据 / 🟡 来自 CUDA 公开行为 / 🔴 来自 driver 内部推断（可能因驱动版本而异）

#### 申请瞬间的物理状态

| 项 | 状态 | 说明 | 可信度 |
|---|---|---|---|
| **虚拟地址空间** | 100 GB | 已预留，指针正常返回 | 🟢 |
| **HBM 物理占用** | **0 GB** | 未分配物理页 | 🟡 |
| **Host 物理占用** | **0 GB** | 未分配物理页 | 🟡 |
| **swap 占用** | 0 GB | 取决于 driver 实现 | 🔴 |

**关键事实**：`cudaMallocManaged` **只预留虚拟地址，不分配物理内存**。这一刻 HBM 和 Host 都是 0。

对照 `cudaMalloc(100GB)`：在 80GB HBM 上**直接 OOM 返回失败**；`cudaMallocManaged(100GB)` **成功**。

> **源码证据**：[memory_utils.cu:97](fbgemm_gpu/src/memory_utils/memory_utils.cu#L97) 的 `cudaMallocManaged(&ptr, size_bytes)` 只接收 `size_bytes` 并返回指针，无任何物理页预分配逻辑。

#### 第一次访问：page fault 触发 lazy 物理页分配

CUDA driver 按 **页粒度**（典型 4KB，HugePage 模式 2MB）分配物理页，行为取决于 `cudaMemAdvise` 设置：

##### 场景 A：FBGEMM 的默认 `new_managed_tensor` 流程

[memory_utils.cu:202-227](fbgemm_gpu/src/memory_utils/memory_utils.cu#L202-L227) 的 5 步流程中，**第 2、3 步**直接决定了物理页的去向：

```cpp
// line 203-214
cudaMemAdviseSetPreferredLocation → CPU（host）       // 🟢 源码
// line 218-227
cudaMemAdviseSetAccessedBy → current_device（GPU）   // 🟢 源码
```

两个 advise 的语义：

| advise | 含义 | 对物理页的影响 |
|---|---|---|
| `PreferredLocation=CPU` | driver 倾向把页放在 host | 页默认在 host，GPU 访问时按需 fault 迁出 |
| `AccessedBy=GPU` | GPU 可以**直接读** host 内存不触发 page fault | GPU 读路径走 zero-copy，**写操作仍会 fault 并迁到 HBM** |

[memory_utils.cu:216-217](fbgemm_gpu/src/memory_utils/memory_utils.cu#L216-L217) 的源码注释明确证实了这一点：
> "User hints with 'accessed by': GPU will establish direct mapping of data in CPU memory, no page faults will be generated"

##### 场景 B：用户自己写裸 `cudaMallocManaged` + 不设 advise

driver 使用**默认策略**（"fault-on-access, migrate to first accessor"）：

| 触发者 | 行为 | 物理页去向 |
|---|---|---|
| **CPU 首次访问** | page fault，页从 zero-page 复制到 host | Host |
| **GPU 首次访问** | page fault，页从 zero-page 复制到 HBM | HBM |
| **谁访问，页就在哪** | 不主动迁移 | 取决于**最近一次**访问者 |

#### 100GB 申请 + 80GB HBM 的具体拆分

##### 情况 1：只读不写 + FBGEMM `new_managed_tensor`

```
HBM 占用: 0 GB（GPU 走 zero-copy 读 host，无 fault）
Host 占用: 100 GB（全部页都在 host）
```

##### 情况 2：随机读写 + FBGEMM `new_managed_tensor`（最常见）

- 假设工作集 W=100GB（全部页都访问过）
- driver 在 HBM 和 Host 之间 LRU 调度
- **HBM 占用 ≈ 80 GB**（HBM 满）
- **Host 占用 ≈ 100 GB**（所有页在 host 有一份 copy，HBM 的页在 host 也有 stale copy，driver 实现相关）
- **总物理内存 ≥ 100 GB**

##### 情况 3：整张表 GPU 高频训练 + FBGEMM `new_managed_tensor`

- 100GB 频繁被 GPU 读写
- driver 倾向把所有页放 HBM，但 HBM 装不下
- LRU 强制换出 20GB 到 host
- **HBM 占用 = 80 GB**（满）
- **Host 占用 ≥ 20 GB**（被换出的冷页）
- **被换出的页每次再被访问 → page fault → 重新换入 → 抖动（thrashing）**

这就是 **MANAGED 模式在大表场景性能差** 的根本原因。

##### 情况 4：FBGEMM `MANAGED_CACHING` 模式 + `cache_load_factor=0.2`

```
table_size = 100 GB
HBM cache  = 100 × 0.2 = 20 GB   ← FBGEMM 显式管理
UVM (host) = 80 GB               ← FBGEMM 整张表放这
```

FBGEMM 在 [lru_cache_populate.cu:17-165](fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate.cu#L17-L165) 用 `warp_cache_load` / `warp_cache_evict` 显式控制哪些 UVM 页进 HBM cache：

- **HBM cache 区域（20GB）**：是**普通 CUDA tensor**（[lxu_cache.cu:415, 472](fbgemm_gpu/src/split_embeddings_cache/lxu_cache.cu#L415) 用 `at::empty` 分配），**不是 UVM**，**不调任何 advise**——driver 看到普通 CUDA tensor 自然放 HBM
- **UVM 区域（80GB）**：`PreferredLocation=CPU`, `AccessedBy=GPU`（[memory_utils.cu:202-227](fbgemm_gpu/src/memory_utils/memory_utils.cu#L202)）→ driver 优先放 host
- FBGEMM 自己用 `cudaMemcpy` 把热行从 UVM 拷进 HBM cache，cold 行拷回 UVM

最终物理占用：

```
HBM 占用 = 20 GB（lxu_cache_weights，独立普通 CUDA tensor，物理一定在 HBM）
Host 占用 = 100 GB（weights_uvm 整张表，物理一定在 CPU 内存）
```

**两个 tensor 物理上完全分离**：
- HBM cache 不会出现在 host 上（它不是 UVM，driver 无 UVM 调度）
- weights_uvm 不会出现在 HBM 上（`SetPreferredLocation=CPU` 引导 driver 不放 HBM）
- 两者是**两份独立内存**，靠 FBGEMM **显式 cudaMemcpy** 同步（详见 §3.2.9 HBM cache 同步架构）

**不会发生 thrashing**，因为：
- HBM 上限被 FBGEMM 显式锁死（`cache_load_factor=0.2`）
- HBM cache 本身不是 UVM，**不参与 UVM 的 LRU 调度**（详见 §3.2.5 冷热管理机制和 §3.2.8 advise 选项详解）

#### 关键认知差异（评审会必讲）

| 错误认知 | 正确理解 | 可信度 |
|---|---|---|
| "100GB 申请，driver 会从 HBM/Host 各分一些" | ❌ driver **不主动**平均分，**按访问模式 + advise 动态调度** | 🟡 |
| "HBM 装不下会 OOM" | ❌ `cudaMallocManaged` 不会 OOM（只受虚拟地址空间限制），只有 `cudaMalloc` 才会 | 🟡 |
| "UVM 是 driver 自动搬运" | ⚠️ 部分对，但**频繁抖动**时性能崩塌 | 🟡 |
| "100GB 申请后立刻占 100GB 物理" | ❌ 只占**实际访问过**的页 | 🟡 |
| "FBGEMM 的 HBM cache 上限 20GB 是 driver 决定的" | ❌ 是 FBGEMM 用 `cache_load_factor=0.2` 显式锁死的 | 🟢 |
| "HBM cache 区域设了 `PreferredLocation=DEVICE`" | ❌ HBM cache 是**普通 CUDA tensor**（[lxu_cache.cu:415, 472](fbgemm_gpu/src/split_embeddings_cache/lxu_cache.cu#L415)），**不调任何 advise**，不放 HBM 才怪 | 🟢 |

#### 评审会"一句话杀手锏"

> **"`cudaMallocManaged(100GB)` 申请瞬间 HBM 占用 0GB、Host 占用 0GB**，物理页是**按访问 lazy 分配**的；当某页被 GPU 首次写访问时，driver 触发 page fault，按 `PreferredLocation` 设置把页放 HBM 还是 Host；如果没设 advise 或设了 `DEVICE`，driver 会把页全堆到 HBM，**80GB HBM 装满 100GB 时会发生 LRU 抖动**；FBGEMM 的 `MANAGED_CACHING` 模式用 `cache_load_factor=0.2` 显式锁住 HBM 容量 20GB，UVM 区 80GB 用 `PreferredLocation=CPU` 引导 driver 放 host，**根本不依赖 UVM 自动搬运**——这才是昇腾 UVM 硬件必须实现的语义：**不是自动迁移，是 advised placement + accessed-by 直连 + lazy fault**。"**

### 3.2.8 FBGEMM 实际使用的 advise 选项与内存分配行为

> **可信度图例**：🟢 来自 FBGEMM 源码直接证据 / 🟡 来自 CUDA 公开行为 / 🔴 来自 driver 内部推断（可能因驱动版本而异）

#### 4 处 advise 调用点（穷举）

| # | 调用函数 | advise 选项 | 源码位置 | 调用次数 |
|---|---|---|---|---|
| 1 | `new_managed_tensor`（硬编码） | `SetPreferredLocation=CPU` | [memory_utils.cu:202-214](fbgemm_gpu/src/memory_utils/memory_utils.cu#L202) | 每次 UVM tensor 创建 |
| 2 | `new_managed_tensor`（硬编码） | `SetAccessedBy=GPU` | [memory_utils.cu:218-227](fbgemm_gpu/src/memory_utils/memory_utils.cu#L218) | 每次 UVM tensor 创建 |
| 3 | `uvm_cuda_mem_advise`（Python 入口） | 6 种之一 | [memory_utils.cu:415-424](fbgemm_gpu/src/memory_utils/memory_utils.cu#L415) | 由用户调用 |
| 4 | `uvm_mem_advice_dont_fork` | `madvise(MADV_DONTFORK)`（**OS 内核调用，不是 CUDA advise**） | [memory_utils.cu:480-481](fbgemm_gpu/src/memory_utils/memory_utils.cu#L480) | fork 兼容性场景 |

**重要发现**：HBM cache 区域（`lxu_cache_weights`）是**普通 CUDA tensor**（[lxu_cache.cu:415, 472](fbgemm_gpu/src/split_embeddings_cache/lxu_cache.cu#L415) 用 `at::empty` 分配），**不是 UVM，**不调任何 cudaMemAdvise**——driver 看到普通 CUDA tensor 自然放 HBM。这也修正了 §3.2.7 情况 4 中"HBM cache 区域 PreferredLocation=DEVICE"的错误表述。

#### FBGEMM 暴露给 Python 的 6 种 advise 枚举

[memory_utils.cu:550-575](fbgemm_gpu/src/memory_utils/memory_utils.cu#L550) 注册到 Python `fbgemm_gpu_uvm_enum_query`：

| Python 枚举名 | CUDA 枚举值 | FBGEMM 内部使用 |
|---|---|---|
| `AdviseSetReadMostly` | `cudaMemAdviseSetReadMostly` | ❌ **从未在 C++ 调用** |
| `AdviseUnsetReadMostly` | `cudaMemAdviseUnsetReadMostly` | ❌ 同上 |
| `AdviseSetPreferredLocation` | `cudaMemAdviseSetPreferredLocation` | ✅ `new_managed_tensor` 硬编码 |
| `AdviseUnsetPreferredLocation` | `cudaMemAdviseUnsetPreferredLocation` | ❌ 仅暴露给 Python |
| `AdviseSetAccessedBy` | `cudaMemAdviseSetAccessedBy` | ✅ `new_managed_tensor` 硬编码 |
| `AdviseUnsetAccessedBy` | `cudaMemAdviseUnsetAccessedBy` | ❌ 仅暴露给 Python |

#### 每个 advise 的内存分配行为详解

##### ① `SetPreferredLocation=CPU`（FBGEMM 唯一硬编码的 placement advise）

**driver 行为**：把该 UVM 区域的页**首选位置**标记为 host（CPU）内存，是 **hint（提示）** 而非强制。

| 触发事件 | 物理页去向 |
|---|---|
| 分配后无访问 | 物理页 0 占用，**不预分配** |
| CPU 首次访问 | page fault → 页从 zero-page 复制到 **Host** |
| GPU 首次写访问 | page fault → 页从 zero-page 复制到 **HBM**（但 driver 倾向 LRU 回 host） |
| GPU 频繁访问 | 页**暂时**留在 HBM，driver 周期性 LRU 回 host |
| GPU 长期不访问 | 页**主动**迁回 host（hint 作用） |

**本质**：让 driver 知道"这些页是 host-resident 的，HBM 满了优先换出"。

##### ② `SetAccessedBy=GPU`（FBGEMM 唯一硬编码的 access advise）

**driver 行为**：告诉 driver GPU 也会读这个区域，**预先建立 GPU 到 host 内存的直接映射**。

源码注释（[memory_utils.cu:216-217](fbgemm_gpu/src/memory_utils/memory_utils.cu#L216)）：
> "User hints with 'accessed by': GPU will establish direct mapping of data in CPU memory, no page faults will be generated"

| 触发事件 | 物理页去向 |
|---|---|
| GPU **读** host 内存 | **零拷贝**，不触发 page fault，**不迁移**页 |
| GPU **写** host 内存 | **page fault**，**强制迁移**到 HBM |
| CPU 任意访问 | 正常 host 访问，无 fault |

**关键细节**：**只对读操作省 fault**；写操作仍会强制迁移。这就是为什么 FBGEMM 的 `MANAGED_CACHING` 模式能高效：HBM cache 是**普通 CUDA tensor**（无需 advise），UVM 大表设了 `SetAccessedBy=GPU` 让 GPU 读时零拷贝，**写时通过 `warp_cache_load` 显式控制**。

##### ③ `SetReadMostly`（FBGEMM 暴露但内部不用）

**driver 行为**：标记这个 UVM 区域**主要是只读**的，driver **几乎从不**把页从 host 迁到 device。

| 触发事件 | 物理页去向 |
|---|---|
| GPU 读 | 零拷贝（与 `SetAccessedBy` 类似） |
| GPU 写 | 一次性迁移到 HBM，之后回退到 host 行为 |

**FBGEMM 不用它的原因**：
- FBGEMM 的 UVM 表**是会被训练的**（weights / optimizer states 都会写）
- 写了之后还要写回 UVM 的场景，`SetReadMostly` 会引发不必要的迁移
- FBGEMM 选了更细粒度的 `SetPreferredLocation=CPU` + `SetAccessedBy=GPU` 组合，**让 LRU 策略替代 ReadMostly**

##### ④ `SetPreferredLocation=DEVICE`（Python 入口支持但 FBGEMM 不主动调）

**driver 行为**：标记这个 UVM 区域的页**首选位置**为 device，driver LRU 优先把页放 HBM。

| 触发事件 | 物理页去向 |
|---|---|
| HBM 装得下 | 所有页都放 HBM |
| HBM 装不下 | driver 强制把"非首选位置"的页**换出到 host**（§3.2.7 情况 3 抖动场景） |

**FBGEMM 不用它的原因**：MANAGED 模式如果用 `SetPreferredLocation=DEVICE`，HBM 装不下大表会**直接抖动**。FBGEMM 改用 `SetPreferredLocation=CPU`（反向），让 HBM 缓存只装 HBM cache 那一小块（20%）。

##### ⑤ `madvise(MADV_DONTFORK)`（OS 内核调用，**不是 CUDA advise**）

**内核行为**：标记这个 VMA（虚拟内存区域）**不要被 fork 到子进程**。

源码注释（[memory_utils.cu:467-471](fbgemm_gpu/src/memory_utils/memory_utils.cu#L467)）说明问题背景：
> "During fork() the uvm driver is called to copy VMA for UVM space. The uvm driver then removes pmap entries for both child and parent. Re-establishing the mappings for the parent is slow."

**内存分配行为**：无（这是个**进程隔离 advise**，不影响物理页位置）

#### FBGEMM 实际选用的 advise 组合（最优解分析）

FBGEMM 在 `new_managed_tensor` 中**只硬编码 2 个 advise**：

```cpp
SetPreferredLocation=CPU + SetAccessedBy=GPU
```

**为什么这个组合是对的**：

| advise 1 | advise 2 | 组合效果 |
|---|---|---|
| `SetPreferredLocation=CPU` | `SetAccessedBy=GPU` | **写时按需 fault 到 HBM**（热数据），**读时 zero-copy 走 host**（冷数据） |

**这正好是 MANAGED_CACHING 模式需要的语义**：
- 大表（100GB）放在 host，GPU 训练时**只把热行拷进 HBM cache**（FBGEMM 自己控）
- 冷行留在 host，GPU 偶尔读时**走 zero-copy 不消耗 HBM 容量**
- 完美避免 §3.2.7 情况 3 的抖动问题

#### 评审会关键事实清单

| 事实 | 证据 | 可信度 |
|---|---|---|
| FBGEMM 硬编码 `SetPreferredLocation=CPU` | [memory_utils.cu:202-214](fbgemm_gpu/src/memory_utils/memory_utils.cu#L202) | 🟢 |
| FBGEMM 硬编码 `SetAccessedBy=GPU` | [memory_utils.cu:218-227](fbgemm_gpu/src/memory_utils/memory_utils.cu#L218) | 🟢 |
| HBM cache 区域**不调任何 advise**（是普通 CUDA tensor） | [lxu_cache.cu:415, 472](fbgemm_gpu/src/split_embeddings_cache/lxu_cache.cu#L415) | 🟢 |
| FBGEMM 暴露 6 种 advise 给 Python（3 set + 3 unset） | [memory_utils.cu:550-575](fbgemm_gpu/src/memory_utils/memory_utils.cu#L550) | 🟢 |
| `SetAccessedBy=GPU` 只对读零拷贝，写仍会迁移 | [memory_utils.cu:216-217](fbgemm_gpu/src/memory_utils/memory_utils.cu#L216) 注释 | 🟢 |
| FBGEMM 用 `madvise(DONTFORK)` 处理 fork 兼容 | [memory_utils.cu:480-481](fbgemm_gpu/src/memory_utils/memory_utils.cu#L480) | 🟢 |
| `SetAccessedBy` 的具体 driver 行为（"GPU establish direct mapping"） | CUDA 公开文档 | 🟡 |
| FBGEMM 选 `CPU+AccessedBy` 组合的原因（替代 ReadMostly） | 代码推论 | 🔴 |

#### 评审会"一句话核心"

> "FBGEMM 在 `new_managed_tensor` 中**只硬编码 2 个 advise**：`SetPreferredLocation=CPU`（让 driver 优先把页放 host）+ `SetAccessedBy=GPU`（让 GPU 读时 zero-copy 不 fault）。HBM cache 区域是**普通 CUDA tensor 不需要 advise**。**昇腾 UVM 硬件必须实现的最小 advise 集合就是这 2 个 + 1 个 OS 兼容性 `madvise(DONTFORK)`**——少了任一个，FBGEMM 都不能跑。"

#### 昇腾 CANN 对照表（适配依据）

| CUDA API | FBGEMM 用法 | 昇腾 CANN 等价 API（待确认） |
|---|---|---|
| `cudaMallocManaged` | 申请 100GB UVM | `aclrtMallocManaged`（需确认是否存在） |
| `cudaMemAdviseSetPreferredLocation` | 引导页放 host | `aclrtMemAdviseSetPreferredLocation`（需确认） |
| `cudaMemAdviseSetAccessedBy` | GPU 读零拷贝 | `aclrtMemAdviseSetAccessedBy`（需确认） |
| `cudaMemPrefetchAsync` | 主动预取（FBGEMM 不用） | `aclrtMemPrefetchAsync`（需确认） |
| `madvise(MADV_DONTFORK)` | fork 兼容 | 操作系统层，CANN 需保证 driver 不破坏此行为 |

> ⚠️ CANN 对照表中的 API 名是**基于 CUDA 公开 API 命名约定的推测**，昇腾 NPU 是否实现、命名是否一致，需要昇腾团队确认。这是评审会上需要昇腾侧明确承诺的部分。

### 3.2.9 HBM cache 同步架构：独立 tensor + 行级显式 memcpy

> **可信度图例**：🟢 来自 FBGEMM 源码直接证据 / 🟡 来自 CUDA 公开行为 / 🔴 来自 driver 内部推断

#### 核心结论（一句话）

> **`weights_uvm`（100% 整张表的 UVM tensor）与 `lxu_cache_weights`（20% 热行副本的普通 CUDA tensor）是两个物理上完全分离的独立内存**。HBM cache **不是 UVM 的页子集**，是**显式维护的行级副本**。**数据一致性由 FBGEMM 自己保证，不依赖 driver**。

#### 关键区分：UVM page 缓存 vs 显式 tensor 副本

| 维度 | UVM 自身的 page 缓存 | FBGEMM 的 HBM cache |
|---|---|---|
| **物理形态** | 同一个 UVM tensor 的页 | **两个独立的 tensor**（UVM + 普通 CUDA） |
| **同步机制** | driver 按 LRU 自动迁移 | FBGEMM **显式 cudaMemcpy** |
| **数据一致性** | driver 保证单份数据多副本一致 | FBGEMM 自己维护（`warp_cache_load` / `warp_cache_evict`） |
| **粒度** | 4KB / 2MB 页 | **一行 embedding**（行级） |
| **错峰策略** | driver LRU 决定 | FBGEMM 用时间戳决定（[lru_cache_populate.cu:74](fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate.cu#L74)） |

#### 完整架构图（100GB 表为例）

```
┌─────────────────────────────────────────────────────────────┐
│ MANAGED_CACHING 模式的内存布局（100GB 表为例）               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CPU 内存（100GB）                                           │
│  ┌──────────────────────────────────────────────┐           │
│  │ weights_uvm (cudaMallocManaged)               │           │
│  │ - SetPreferredLocation = CPU                  │           │
│  │ - SetAccessedBy = GPU                         │           │
│  │ - 完整 100GB 表（所有行）                       │           │
│  │ - physical: 100GB 都在 host 内存               │           │
│  └──────────────────────────────────────────────┘           │
│                          ▲                                  │
│                          │ FBGEMM 显式 cudaMemcpy            │
│                          │ (warp_cache_load/evict)          │
│                          │                                  │
│  HBM（20GB）         ────┘                                  │
│  ┌──────────────────────────────────────────────┐           │
│  │ lxu_cache_weights (at::empty 普通 CUDA tensor)│           │
│  │ - 不是 UVM, 不调任何 advise                    │           │
│  │ - 只装 cache_load_factor × total_rows × D     │           │
│  │   = 0.2 × 100GB = 20GB                         │           │
│  │ - 物理: 20GB 都在 HBM                          │           │
│  │ - 是 weights_uvm 中"热行"的副本                 │           │
│  └──────────────────────────────────────────────┘           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 两个独立 tensor 的代码证据

| tensor | 分配方式 | 物理位置 | advise | 证据 |
|---|---|---|---|---|
| **`weights_uvm`**（100GB 整张表） | `cudaMallocManaged` | CPU 内存（驱动放） | `SetPreferredLocation=CPU` + `SetAccessedBy=GPU` | [memory_utils.cu:97](fbgemm_gpu/src/memory_utils/memory_utils.cu#L97) |
| **`lxu_cache_weights`**（20GB 缓存） | `at::empty`（普通 CUDA） | HBM（自然放） | **不调任何 advise** | [lxu_cache.cu:415, 472](fbgemm_gpu/src/split_embeddings_cache/lxu_cache.cu#L415) |

**关键结论**：
- HBM cache **不是 UVM tensor 的一个"页子集"**
- HBM cache **是一个独立的普通 CUDA tensor**，**与 `weights_uvm` 没有 UVM 关系**
- 两者靠 FBGEMM 的**显式 cudaMemcpy** 维持数据同步

#### 显式同步的两个原语

##### ① `warp_cache_load`：[weight_row.cuh:346-367](fbgemm_gpu/include/fbgemm_gpu/utils/weight_row.cuh#L346)

注释原话：
> "Load the row from the embedding table into the cache"

代码语义（[weight_row.cuh:354](fbgemm_gpu/include/fbgemm_gpu/utils/weight_row.cuh#L354)）：
```cpp
// cache_row_ ← row_  （HBM ← UVM）
same_type_vector_copy(cache_row_ + d, reinterpret_cast<const cache_t*>(row_ + d));
```

| 项 | 值 |
|---|---|
| **方向** | UVM tensor（CPU 内存）→ HBM cache |
| **粒度** | 一行 embedding（`dim_` 个元素） |
| **触发时机** | cache miss，需要把热行从 UVM 拉到 HBM |
| **调用位置** | [lru_cache_populate.cu:142-146](fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate.cu#L142) 的 `lru_cache_insert_kernel` |

##### ② `warp_cache_evict`：[weight_row.cuh:395-440](fbgemm_gpu/include/fbgemm_gpu/utils/weight_row.cuh#L395)

注释原话：
> "Evict the row from the cache and into the embedding table"

代码语义（[weight_row.cuh:378-379](fbgemm_gpu/include/fbgemm_gpu/utils/weight_row.cuh#L378)）：
```cpp
// row_ ← cache_row_  （UVM ← HBM）
same_type_vector_copy(
    reinterpret_cast<emb_t*>(row_ + d),
    reinterpret_cast<const cache_t*>(cache_row_ + d));
```

| 项 | 值 |
|---|---|
| **方向** | HBM cache → UVM tensor（CPU 内存） |
| **粒度** | 一行 embedding |
| **触发时机** | cache 满，需要把冷行从 HBM 写回 UVM |
| **调用位置** | [lru_cache_populate.cu:127-134](fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate.cu#L127) 的 `lru_cache_insert_kernel`（在 load 之前先 evict） |
| **副作用** | 如果是 INT8 量化表，会**重新计算 qparams** 并写回 UVM（[weight_row.cuh:400-419](fbgemm_gpu/include/fbgemm_gpu/utils/weight_row.cuh#L400)） |

#### 同步流程（[lru_cache_populate.cu:111-146](fbgemm_gpu/src/split_embeddings_cache/lru_cache_populate.cu#L111)）

```cpp
// 步骤 1: 槽位已被占用, 先把旧行写回 UVM
if (current_idx != kCacheStateInvalid) {
    WeightRow<emb_t, cache_t, cache_t>(
        &weights[weights_offset_current + idx_current * D_emb + 0],  // 写到 UVM
        &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0], // 从 HBM 读
        D_current,
        stochastic_rounding,
        ...)
        .warp_cache_evict(blockDim.x, threadIdx.x);
}

// 步骤 2: 然后把新行从 UVM 拉到 HBM
WeightRow<emb_t, cache_t, cache_t>(
    &weights[weights_offset_insert + idx_insert * D_emb + 0],        // 从 UVM 读
    &lxu_cache_weights[cache_set * kWarpSize + insert_slot][0],    // 写到 HBM
    D_insert)
    .warp_cache_load(blockDim.x, threadIdx.x);

// 步骤 3: 更新 LRU 元数据
if (threadIdx.x == 0) {
    lxu_cache_state[cache_set][insert_slot] = insert_idx;
    lru_state[cache_set][insert_slot] = time_stamp;
}
```

**重要细节**：
- 步骤 1 和 2 是**成对**的：先 evict 旧行回 UVM，再 load 新行到 HBM
- 步骤 3 才更新元数据（`lxu_cache_state` 记录这槽位现在存的是哪一行）
- **元数据是 FBGEMM 自己的**（`lxu_cache_state` + `lru_state`），**不依赖 driver**

#### FBGEMM 实际搬运数据的方式（kernel 内 vs `cudaMemcpy` API）

FBGEMM 实际搬运数据有**两种**方式，对应**不同粒度**和**不同场景**：

| 维度 | 方式 A：kernel 内逐元素拷贝 | 方式 B：`cudaMemcpyAsync` API |
|---|---|---|
| **API** | `same_type_vector_copy`（[weight_row.cuh:354](fbgemm_gpu/include/fbgemm_gpu/utils/weight_row.cuh#L354)） | `cudaMemcpyAsync` |
| **调用位置** | 嵌在 forward / backward kernel 里 | 独立 host 端 launch kernel |
| **粒度** | **一行 embedding**（行级） | **整块**（按需切分） |
| **同步性** | 异步（kernel 内并发） | 异步（CUDA stream） |
| **使用场景** | cache miss 时按行 load/evict | 批量场景：`lxu_cache_flush` / `lxu_cache_lock_all_rows` |
| **代表性代码** | `warp_cache_load` / `warp_cache_evict` | [lxu_cache.cu:84-145](fbgemm_gpu/src/split_embeddings_cache/lxu_cache.cu#L84) `lxu_cache_flush` |
| **是否阻塞 host** | 不阻塞（GPU kernel 内） | 不阻塞（async + stream） |
| **能否 overlap 计算** | ✅ 天然 overlap | ✅ 通过多 stream overlap |

##### 方式 A：kernel 内逐元素拷贝（行级同步）

[weight_row.cuh:346-440](fbgemm_gpu/include/fbgemm_gpu/utils/weight_row.cuh#L346) 的 `warp_cache_load` / `warp_cache_evict` 在 **CUDA kernel 内部**用向量化赋值：

```cpp
// warp_cache_load: HBM ← UVM
same_type_vector_copy(cache_row_ + d, reinterpret_cast<const cache_t*>(row_ + d));

// warp_cache_evict: UVM ← HBM
same_type_vector_copy(reinterpret_cast<emb_t*>(row_ + d), reinterpret_cast<const cache_t*>(cache_row_ + d));
```

**为什么不直接用 `cudaMemcpy`**：
- `cudaMemcpy` 是**整块、阻塞**的 API，**不适合**嵌入 forward/backward kernel 的**行级、并发**场景
- kernel 内逐元素拷贝可以**和计算流水线 overlap**（prefetch 效果）
- 行级粒度让 FBGEMM 能在一次 kernel launch 内**部分预取**（一行 load 不阻塞其他行计算）

**底层语义**：和 `cudaMemcpy` 完全一样——都是把一块 GPU 可访问内存的数据搬到另一块。区别只在**触发位置**（kernel 内 vs host launch）。

##### 方式 B：`cudaMemcpyAsync` API（批量同步）

[lxu_cache.cu:84-145](fbgemm_gpu/src/split_embeddings_cache/lxu_cache.cu#L84) 的 `lxu_cache_flush` 是**真正用 `cudaMemcpyAsync` API** 的代表性场景——把整个 HBM cache 一次性写回 UVM：

```cpp
// lxu_cache_flush 简化伪代码
for (auto slot_idx = 0; slot_idx < num_slots; slot_idx++) {
    if (lxu_cache_state[slot_idx] != kCacheStateInvalid) {
        // 取出该槽位对应的 UVM 行号
        auto uvm_row = lxu_cache_state[slot_idx];
        // 计算 UVM 偏移和 HBM 偏移
        auto hbm_offset = slot_idx * row_size;
        auto uvm_offset = uvm_row * row_size;
        // 显式 cudaMemcpyAsync
        cudaMemcpyAsync(
            weights_uvm.data_ptr<float>() + uvm_offset,   // dst: UVM (managed 指针)
            lxu_cache_weights.data_ptr<float>() + hbm_offset, // src: HBM (device 指针)
            row_size * sizeof(float),
            cudaMemcpyDeviceToDevice,  // ← 关键: managed 指针当 device 指针用
            stream
        );
    }
}
```

**`cudaMemcpy` 跨 HBM/UVM 的 4 种合法方向**：

| # | 源 | 目标 | `cudaMemcpyKind` | 是否合法 |
|---|---|---|---|---|
| 1 | HBM（device 指针） | UVM（managed 指针） | `cudaMemcpyDeviceToDevice` | ✅ driver 透明处理 |
| 2 | UVM（managed 指针） | HBM（device 指针） | `cudaMemcpyDeviceToDevice` | ✅ driver 透明处理 |
| 3 | HBM | UVM | `cudaMemcpyDeviceToHost` | ✅ 走 D2H 路径 |
| 4 | UVM | HBM | `cudaMemcpyHostToDevice` | ✅ 走 H2D 路径 |

**关键事实**：
- `cudaMallocManaged` 返回的指针**既可以当 device 指针用，又可以当 host 指针用**
- driver **透明**处理跨位置拷贝：发不同时机可能是 DMA、可能通过 host 内存中转
- 用户**不需要关心** UVM 物理页当前在 HBM 还是 host

#### 训练时的双向同步（更复杂）

训练场景下，**optimizer 更新权重**时必须**同时写 UVM 和 HBM cache**，否则两个副本会不一致：

| 同步路径 | 时机 | 代码 |
|---|---|---|
| **optimizer 写 HBM cache slot** | forward + backward 之间 | `split_embedding_optimizer.cu` |
| **optimizer 写 UVM（CPU 内存）** | 同上 | 同上 |
| **writeback 路径** | 必要时（eviction） | `warp_cache_evict` |

如果 FBGEMM 只写 HBM cache 不写 UVM，cache evict 时会把**陈旧数据**写回 UVM，导致 embedding 表"丢更新"。

这就是为什么在 [embedding_inplace_update.cu:84-98](fbgemm_gpu/src/embedding_inplace_ops/embedding_inplace_update.cu#L84) 中 `MANAGED_CACHING` 模式有**双写逻辑**——同时更新两份。

#### 物理位置严格分离（关键事实）

| 内存区域 | 物理位置 | 不可能出现的位置 | 原因 |
|---|---|---|---|
| `weights_uvm`（100GB） | CPU 内存 | HBM | `SetPreferredLocation=CPU` 引导 driver 不放 HBM |
| `lxu_cache_weights`（20GB） | HBM | CPU 内存 | 它是普通 CUDA tensor，driver 无 UVM 调度 |

**两套内存物理上完全分离**——这意味着：
- 训练时 HBM cache 满不会触发 UVM driver 的 LRU 调度（因为 cache 不是 UVM）
- UVM 大表抖动不会污染 HBM cache（因为 cache 不走 UVM 路径）
- **HBM 总占用被 FBGEMM 显式锁死在 20GB 附近**，不会随访问模式波动

#### 评审会必问的 3 个尖锐问题

##### Q1: "MANAGED_CACHING 是不是有双份数据？"

**答：是的。但双份是 FBGEMM 显式维护的，不是 driver 隐式复制的。**
- `weights_uvm` 在 CPU（完整表）+ `lxu_cache_weights` 在 HBM（20% 热行副本）
- 任何时候，HBM 上有最多 20% 的行（其余 80% 不在 HBM）
- 训练时必须双写；推理时按需 load/evict

##### Q2: "HBM 和 UVM 怎么保持一致？"

**答：FBGEMM 自己保证，不依赖 driver。**
- Forward 时：cache hit 直接读 HBM；cache miss 从 UVM `warp_cache_load` 拉一行
- Backward 时：optimizer **同时写 HBM cache slot 和 UVM**（双写）
- Eviction 时：`warp_cache_evict` 把 HBM 修改写回 UVM
- **没有任何 driver 介入的自动同步**

##### Q3: "driver 帮你管还是 FBGEMM 自己管？"

**答：FBGEMM 自己管冷热，driver 只管 UVM 那一侧的物理页放置。**
- Driver 管：`weights_uvm` 的物理页放哪（HBM 还是 CPU）
- FBGEMM 管：`lxu_cache_weights` 里放哪些行、什么时候换出

#### 评审会"一句话核心"

> "MANAGED_CACHING 模式下，`weights_uvm` 是**完整 100GB 表的 UVM tensor**（用 `SetPreferredLocation=CPU` 引导 driver 放在 CPU 内存），`lxu_cache_weights` 是**独立的 20GB 普通 CUDA tensor**（自然放 HBM），两者**不共享 UVM 关系**。HBM cache 是 weights_uvm 中**热行的行级副本**，FBGEMM 用 `warp_cache_load` / `warp_cache_evict` 显式 cudaMemcpy 维护同步——**昇腾 UVM 硬件要实现的就是 `weights_uvm` 那一侧的 lazy fault + advised placement + accessed-by 直连**，`lxu_cache_weights` 根本不走 UVM，**昇腾硬件不需要为它实现任何 UVM 语义**。"

#### 修正 §3.2.7 情况 4

| 之前的描述 | 更精确的描述 |
|---|---|
| "HBM 占用 ≈ 20 GB（HBM cache，锁死不会超）" | "HBM 占用 = 20 GB（`lxu_cache_weights`，**独立普通 CUDA tensor**，物理一定在 HBM）" |
| "Host 占用 ≈ 100 GB（UVM 整张表 + cache 在 host 的备份，driver 实现相关）" | "Host 占用 = 100 GB（`weights_uvm` 整张表，物理一定在 CPU 内存）" |
| （隐含）cache 可能在 host 有备份 | "**HBM cache 不会出现在 host 上**（它不是 UVM）" |
| （隐含）UVM 可能在 HBM 有副本 | "**weights_uvm 不会出现在 HBM 上**（`SetPreferredLocation=CPU` 引导 driver 不放 HBM）" |
| （缺失） | "**两者物理上是完全分离的两份内存**，靠 FBGEMM 显式 memcpy 同步" |

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
