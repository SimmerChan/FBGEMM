# SIMT接口使用分析报告

本文档分析了两个仓库（HierarchicalKV-ascend 和 RecSDK/cust_op）中使用SIMT（Single Instruction Multiple Threads）接口的算子情况。

## 仓库信息

| 仓库名称 | 路径 |
|---------|------|
| HierarchicalKV-ascend | `/Users/huangshilei/Documents/pythonprojects/HierarchicalKV-ascend` |
| RecSDK/cust_op | `/Users/huangshilei/Documents/cppprojects/RecSDK/cust_op` |

---

## 一、SIMT接口汇总表

### 1.1 核心声明与启动接口

| SIMT接口 | 涉及的算子 | 算子对应实现的文件路径 | SIMT接口使用方法 | SIMT接口作用 |
|---------|-----------|----------------------|-----------------|-------------|
| `__simt_vf__` | 所有SIMT算子 | HierarchicalKV: `hkv_hashtable/*/v35/*.h`<br>RecSDK: `ascendc_op/ai_core_op/*/c310/op_kernel/*.h` | `__simt_vf__ __aicore__ LAUNCH_BOUND(N) inline void kernel_func(...)` | 声明SIMT向量函数，标记函数为SIMT执行模式，允许在AICore上以SIMT方式运行 |
| `AscendC::Simt::VF_CALL<KernelFunc>()` | 所有SIMT算子 | RecSDK: `split_embedding_kernel_simt.h`<br>`backward_codegen_*_kernel.h`<br>`block_bucketize_sparse_features_kernel.h` | `AscendC::Simt::VF_CALL<KernelName>(AscendC::Simt::Dim3{threadNum, 1, 1}, args...);` | 启动SIMT kernel，第一个参数指定线程维度，后续参数传递给kernel函数 |
| `Simt::VF_CALL<KernelFunc>()` | HKV所有算子 | HierarchicalKV: `simt_vf_dispatcher.h` | `Simt::VF_CALL<kernel_vf<K, V, S, STRATEGY>>(args...);` | 启动SIMT kernel，用于HierarchicalKV-ascend项目的kernel调度 |

### 1.2 线程索引与维度接口

| SIMT接口 | 涉及的算子 | 算子对应实现的文件路径 | SIMT接口使用方法 | SIMT接口作用 |
|---------|-----------|----------------------|-----------------|-------------|
| `AscendC::Simt::GetThreadIdx<0>()` | split_embedding, backward_codegen, asynchronous_cumsum, block_bucketize, invert_permute | `split_embedding_kernel_simt.h:109`<br>`backward_codegen_unweighted_exact_kernel.h:136`<br>`asynchronous_complete_cumsum_kernel.h:100` | `int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();` | 获取当前线程在块内的索引（x维度），用于确定线程处理的数据位置 |
| `AscendC::Simt::GetThreadNum<0>()` | split_embedding, backward_codegen, asynchronous_cumsum, block_bucketize | `split_embedding_kernel_simt.h:112`<br>`backward_codegen_unweighted_exact_kernel.h:137`<br>`asynchronous_complete_cumsum_kernel.h:102` | `int32_t blockThreads = AscendC::Simt::GetThreadNum<0>();` | 获取当前块内的线程总数（x维度），用于计算循环步长和任务分配 |
| `AscendC::Simt::GetBlockIdx()` | split_embedding, asynchronous_cumsum, block_bucketize, invert_permute | `split_embedding_kernel_simt.h:119`<br>`asynchronous_complete_cumsum_kernel.h:101`<br>`invert_permute_kernel.h:35` | `int32_t blockIdx = AscendC::Simt::GetBlockIdx();` | 获取当前块的索引，用于确定块处理的数据范围 |
| `AscendC::Simt::GetBlockNum()` | split_embedding | `split_embedding_kernel_simt.h:118` | `int64_t totalWarps = static_cast<int64_t>(AscendC::Simt::GetBlockNum()) * warpsPerBlock;` | 获取块的总数量，用于计算全局任务分配 |
| `AscendC::Simt::Dim3` | 所有使用VF_CALL的算子 | `split_embedding_kernel_simt.h:271`<br>`block_bucketize_sparse_features_kernel.h` | `AscendC::Simt::Dim3 simtDim{threadNum, 1, 1};` | 定义SIMT kernel启动的维度结构体，包含x、y、z三个维度 |

### 1.3 线程同步接口

| SIMT接口 | 涉及的算子 | 算子对应实现的文件路径 | SIMT接口使用方法 | SIMT接口作用 |
|---------|-----------|----------------------|-----------------|-------------|
| `AscendC::Simt::ThreadBarrier()` | backward_codegen, asynchronous_cumsum, block_bucketize | `backward_codegen_rowwise_adagrad_unweighted_exact_kernel.h:142`<br>`asynchronous_complete_cumsum_kernel.h:79,87`<br>`block_bucketize_sparse_features_kernel.h` | `AscendC::Simt::ThreadBarrier();` | 块内线程同步屏障，确保所有线程执行到此点后再继续，用于协调多线程数据访问 |
| `__threadfence()` | insert_or_assign | `insert_or_assign_kernel.h:243` | `__threadfence();` | 线程围栏，确保此点之前的所有内存写入对其他线程可见，用于保证内存一致性 |

### 1.4 原子操作接口

| SIMT接口 | 涉及的算子 | 算子对应实现的文件路径 | SIMT接口使用方法 | SIMT接口作用 |
|---------|-----------|----------------------|-----------------|-------------|
| `AscendC::Simt::AtomicCas()` | backward_codegen_dedup, insert_or_assign | `backward_codegen_unweighted_exact_kernel.h:197`<br>`insert_or_assign_kernel.h:90,100` | `uint32_t old = AscendC::Simt::AtomicCas(ptr, expected, desired);` | 原子比较并交换，如果*ptr等于expected则设置为desired，返回原值；用于无锁并发控制 |
| `AscendC::Simt::AtomicAdd()` | backward_codegen_dedup | `backward_codegen_unweighted_exact_kernel.h:202` | `uint32_t pos = AscendC::Simt::AtomicAdd(ptr, value);` | 原子加法操作，将value加到*ptr并返回旧值；用于并发计数 |
| `Simt::AtomicCas()` | insert_or_assign | `insert_or_assign_kernel.h:90,100,111` | `auto try_key = Simt::AtomicCas(ptr, compare, val);` | 原子比较交换操作，用于并发场景下的键值抢占 |
| `Simt::AtomicExch()` | insert_or_assign | `insert_or_assign_kernel.h:138,244` | `Simt::AtomicExch(ptr, new_val);` | 原子交换操作，将ptr指向的值原子地替换为new_val |
| `atomicAdd()` | insert_or_assign | `insert_or_assign_kernel.h:96,117` | `atomicAdd(bucket_size, 1);` | 原子加法操作，用于更新桶大小计数器 |

### 1.5 线程束操作接口

| SIMT接口 | 涉及的算子 | 算子对应实现的文件路径 | SIMT接口使用方法 | SIMT接口作用 |
|---------|-----------|----------------------|-----------------|-------------|
| `AscendC::Simt::WarpReduceAddSync()` | split_embedding | `split_embedding_kernel_simt.h:177` | `float sum = AscendC::Simt::WarpReduceAddSync(local[dim]);` | 线程束内归约加法，将warp内所有线程的值相加；用于高效的并行归约计算 |
| `AscendC::Simt::WarpShflUpSync()` | asynchronous_cumsum, block_bucketize | `asynchronous_complete_cumsum_kernel.h:41` | `T temp = AscendC::Simt::WarpShflUpSync(val, offset);` | 线程束内向上洗牌，从线程束内偏移offset的线程获取值；用于前缀和计算 |
| `__shfl()` | insert_or_assign | `insert_or_assign_kernel.h:121,133,145` | `auto res_sync = __shfl(occupy_result, i, GROUP_SIZE);` | 线程束洗牌，从指定线程获取变量值；用于线程组内数据交换 |
| `__shfl_xor()` | insert_or_assign | `insert_or_assign_kernel.h:152` | `S other_score = __shfl_xor(min_score, offset, GROUP_SIZE);` | 线程束异或洗牌，用于分治法并行计算最小值 |

### 1.6 位操作接口

| SIMT接口 | 涉及的算子 | 算子对应实现的文件路径 | SIMT接口使用方法 | SIMT接口作用 |
|---------|-----------|----------------------|-----------------|-------------|
| `Simt::Ffs()` | insert_or_assign | `insert_or_assign_kernel.h:87,105` | `uint32_t index = (Simt::Ffs(static_cast<int32_t>(cmp_result)) - 1) >> 3;` | 查找第一个设置的位（Find First Set），返回最低设置位的位置；用于快速定位digest比较结果 |

### 1.7 内存访问接口

| SIMT接口 | 涉及的算子 | 算子对应实现的文件路径 | SIMT接口使用方法 | SIMT接口作用 |
|---------|-----------|----------------------|-----------------|-------------|
| `__ldg<>()` | insert_or_assign | `insert_or_assign_kernel.h:139,141,167,169` | `S current_score = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(ptr);` | 带缓存提示的全局内存加载，可指定L1/L2缓存策略；用于优化内存访问性能 |
| `__stg<>()` | insert_or_assign | `insert_or_assign_kernel.h:186,188` | `__stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(ptr, val);` | 带缓存提示的全局内存存储，可指定缓存策略；用于优化写入性能 |

---

## 二、HierarchicalKV-ascend 算子分析

### 2.1 算子列表

| 算子名称 | 文件路径 | 主要功能 |
|---------|---------|---------|
| find_ptr_kernel | `hkv_hashtable/find_ptr_kernel/v35/find_ptr_kernel.h` | 查找哈希表中key对应的value指针 |
| insert_or_assign_kernel | `hkv_hashtable/insert_or_assign_kernel/v35/insert_or_assign_kernel.h` | 插入或更新键值对，支持淘汰策略 |
| insert_and_evict_kernel | `hkv_hashtable/insert_and_evict_kernel/v35/insert_and_evict_kernel.h` | 插入数据并执行淘汰操作 |
| assign_scores_kernel | `hkv_hashtable/assign_scores_kernel/v35/assign_scores_kernel.h` | 分配分数用于淘汰策略 |
| rehash_kernel | `hkv_hashtable/rehash_kernel/v35/rehash_kernel.h` | 重哈希操作，扩展哈希表容量 |
| clear_kernel | `hkv_hashtable/clear_kernel/v35/clear_kernel.h` | 清空哈希表 |
| dump_kernel | `hkv_hashtable/dump_kernel/v35/dump_kernel.h` | 导出哈希表数据 |
| init_table_kernel | `hkv_hashtable/init_table_kernel/v35/init_table_kernel.h` | 初始化哈希表 |
| find_and_update_kernel | `hkv_hashtable/find_and_update_kernel/v35/find_and_update_kernel.h` | 查找并更新元素 |
| find_or_insert_ptr_kernel | `hkv_hashtable/find_or_insert_ptr_kernel/v35/find_or_insert_ptr_kernel.h` | 查找或插入指针 |
| utils_kernel | `hkv_hashtable/utils_kernel/v35/utils_kernel.h` | 工具函数kernel |

### 2.2 核心SIMT接口使用示例

```cpp
// insert_or_assign_kernel.h - 典型的SIMT kernel定义
template <typename K, typename V, typename S, int32_t Strategy>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void insert_or_assign_kernel_with_digest_vf(...) {
    // 使用原子操作进行并发控制
    auto try_key = Simt::AtomicCas(current_key_ptr, key, static_cast<K>(LOCKED_KEY));
    
    // 使用线程束洗牌进行组内通信
    auto res_sync = __shfl(occupy_result, i, GROUP_SIZE);
    
    // 使用原子交换完成最终写入
    Simt::AtomicExch(bucket_keys + key_pos, key);
}
```

---

## 三、RecSDK/cust_op 算子分析

### 3.1 算子列表

| 算子名称 | 文件路径 | 主要功能 |
|---------|---------|---------|
| split_embedding_codegen_forward_unweighted | `ascendc_op/ai_core_op/split_embedding_codegen_forward_unweighted/c310/` | 分割嵌入表前向传播（无权重） |
| backward_codegen_adagrad_unweighted_exact | `ascendc_op/ai_core_op/backward_codegen_adagrad_unweighted_exact/c310/` | Adagrad优化器反向传播 |
| backward_codegen_rowwise_adagrad_unweighted_exact | `ascendc_op/ai_core_op/backward_codegen_adagrad_unweighted_exact/c310/` | 行级Adagrad反向传播 |
| backward_codegen_unweighted_exact | `ascendc_op/ai_core_op/backward_codegen_adagrad_unweighted_exact/c310/` | 通用反向传播（含去重） |
| asynchronous_complete_cumsum | `ascendc_op/ai_core_op/asynchronous_complete_cumsum/c310/` | 异步完全前缀和计算 |
| block_bucketize_sparse_features | `ascendc_op/ai_core_op/block_bucketize_sparse_features/c310/` | 块分桶稀疏特征处理 |
| invert_permute | `ascendc_op/ai_core_op/invert_permute/c310/` | 逆置换操作 |
| expand_into_jagged_permute | `ascendc_op/ai_core_op/expand_into_jagged_permute/c310/` | 扩展到锯齿置换 |

### 3.2 核心SIMT接口使用示例

```cpp
// split_embedding_kernel_simt.h - SIMT kernel定义与启动
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_LAUNCH_BOUND)
inline void PoolingSimt(__gm__ float* devWeights, ...) {
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t warpId = threadIdx / warpSize;
    int32_t laneId = threadIdx % warpSize;
    int32_t warpsPerBlock = static_cast<int32_t>(AscendC::Simt::GetThreadNum<0>()) / warpSize;
    
    // 线程束归约
    float sum = AscendC::Simt::WarpReduceAddSync(local[dim]);
}

// kernel启动
__aicore__ inline void LaunchSimtKernel(Args &args) {
    AscendC::Simt::Dim3 simtDim{static_cast<uint32_t>(simtThreadNum), 1, 1};
    AscendC::Simt::VF_CALL<PoolingSimt>(
        simtDim,
        devWeightsPtr,
        weightsOffsetsPtr,
        ...);
}
```

```cpp
// backward_codegen_unweighted_exact_kernel.h - 去重操作
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK) 
inline void SimtDedupIndices(...) {
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t numThreads = AscendC::Simt::GetThreadNum<0>();
    
    // 原子操作实现去重
    uint32_t oldFlag = AscendC::Simt::AtomicCas(workspace + thisIndForTotalTable, 0, 1);
    if (oldFlag == 0) {
        uint32_t uniqPos = AscendC::Simt::AtomicAdd(validListLenPtr, static_cast<uint32_t>(1));
        validList[uniqPos] = thisIndForTotalTable;
    }
    
    AscendC::Simt::ThreadBarrier();
}
```

---

## 四、SIMT编程模式总结

### 4.1 典型SIMT Kernel结构

```cpp
// 1. 声明SIMT函数
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS) 
inline void kernel_function(参数列表) {
    // 2. 获取线程索引
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    int32_t blockDim = AscendC::Simt::GetThreadNum<0>();
    
    // 3. 计算全局索引
    int64_t globalIdx = blockIdx * blockDim + threadIdx;
    
    // 4. 并行处理数据
    for (int64_t i = globalIdx; i < totalSize; i += blockDim * gridDim) {
        // 处理逻辑
    }
    
    // 5. 必要时同步
    AscendC::Simt::ThreadBarrier();
}

// 6. 启动kernel
__aicore__ inline void LaunchKernel() {
    AscendC::Simt::Dim3 dim{threadsPerBlock, 1, 1};
    AscendC::Simt::VF_CALL<kernel_function>(dim, args...);
}
```

### 4.2 关键特性

| 特性 | 说明 |
|-----|------|
| **SIMT执行模型** | 单指令多线程，所有线程执行相同代码但处理不同数据 |
| **线程层级** | Grid -> Block -> Warp -> Thread，类似CUDA编程模型 |
| **原子操作支持** | 提供CAS、Add、Exch等原子操作，支持无锁并发编程 |
| **线程束操作** | Warp级别归约、洗牌操作，提高并行效率 |
| **内存访问优化** | 支持缓存提示的加载/存储，优化内存性能 |

### 4.3 适用场景

1. **高并行度任务**：大量独立元素需要相同处理逻辑
2. **稀疏数据处理**：Embedding查找、稀疏特征处理
3. **哈希表操作**：并发插入、查找、淘汰
4. **前缀和计算**：利用Warp级别操作高效实现
5. **置换与重排**：数据重排序操作

---

## 五、两个仓库SIMT使用对比

| 对比维度 | HierarchicalKV-ascend | RecSDK/cust_op |
|---------|----------------------|----------------|
| **主要应用场景** | 哈希表操作（查找、插入、淘汰） | 推荐系统算子（Embedding、优化器） |
| **原子操作使用** | 高频使用（并发键值操作） | 中等使用（去重、计数） |
| **线程束操作** | 用于组内通信和分治计算 | 用于归约和前缀和 |
| **内存访问模式** | 随机访问（哈希表） | 混合访问（Embedding表+顺序输出） |
| **同步策略** | 组级同步 + 围栏 | 块级屏障 |

---

## 六、结论

两个仓库充分利用了昇腾AICore的SIMT编程能力，实现了高效的并行计算：

1. **HierarchicalKV-ascend** 专注于高性能哈希表实现，大量使用原子操作实现无锁并发控制，通过线程束洗牌操作实现高效的组内通信，适用于推荐系统中的特征存储和查找场景。

2. **RecSDK/cust_op** 覆盖了推荐系统的核心算子，包括Embedding查找、优化器更新、稀疏特征处理等，SIMT接口的使用使这些算子能够在AICore上高效并行执行。

SIMT接口为昇腾NPU提供了一种类CUDA的编程范式，使得GPU算子能够相对容易地迁移到NPU平台，同时保持较高的并行计算效率。
