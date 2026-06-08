# FBGEMM_GPU API分析报告

## 1. 概述

FBGEMM_GPU (FBGEMM GPU Kernels Library) 是一个基于FBGEMM构建的高性能PyTorch GPU算子库,专注于推荐系统应用。本报告基于v1.5.0-release分支,全面分析了库中的Python API和C++ API。

### 1.1 API分类统计

根据文档结构,FBGEMM_GPU的API分为以下几类:

| API类别 | Python API数量 | C++ API数量 | 说明 |
|---------|---------------|-------------|------|
| Stable Python API | 15个核心接口 | - | 提供向后兼容性保证 |
| Table Batched Embedding (训练) | 1个模块类 | 多个codegen生成函数 | 训练场景的embedding操作 |
| Table Batched Embedding (推理) | 1个模块类 | 多个codegen生成函数 | 推理场景的embedding操作 |
| Jagged Tensor操作 | 10个函数 | 6个模板函数 | 处理变长张量 |
| Pooled Embedding操作 | 3个 | 7个 | Pooled embedding相关 |
| Sparse操作 | 8个 | 113个 | 稀疏数据处理 |
| 量化操作 | 1个 | 50个 | 量化和反量化 |
| 内存管理 | - | 17个 | CUDA内存操作 |
| Feature Gates | 2个类 | 5个函数 | 功能开关 |
| 其他操作 | - | 30个 | 哈希、布局转换等 |

---

## 2. Stable Python API映射表

下表列出了所有Stable Python API及其对应的C++实现和CUDA Kernel。

### 2.1 Stable Python API详细映射表

| 序号 | Python API名称 | Kernel/C++函数名称 | 实现文件路径 | API功能和作用 | 参数说明和取值范围 | 开发工作量(天/人) |
|------|---------------|-------------------|-------------|-------------|------------------|------------------|
| 1 | `torch.ops.fbgemm.jagged_to_padded_dense` | `jagged_to_padded_dense_forward`<br>`jagged_to_padded_dense_backward` | CPU: `fbgemm_gpu/src/jagged_tensor_ops/jagged_tensor_ops_cpu.cpp`<br>GPU: `fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_forward.cu`<br>`fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_backward.cu` | 将变长的jagged tensor转换为固定长度的padded dense tensor,支持前向和反向传播 | - `values`: jagged tensor的值<br>- `offsets`: 每个样本的起始偏移量<br>- `max_lengths`: 目标tensor的最大长度<br>- `padding_value`: 填充值(默认0) | **5天**<br>- 前向kernel: 2天<br>- 反向kernel: 2天<br>- CPU实现和测试: 1天 |
| 2 | `fbgemm_gpu.permute_pooled_embedding_modules.PermutePooledEmbeddings` | `permute_pooled_embs_cpu`<br>`permute_pooled_embs_cuda` | CPU: `fbgemm_gpu/src/permute_pooled_embedding_ops/permute_pooled_embedding_ops_cpu.cpp`<br>GPU: `fbgemm_gpu/src/permute_pooled_embedding_ops/permute_pooled_embedding_ops.cu`<br>Python: `fbgemm_gpu/permute_pooled_embedding_modules.py` | 对pooled embeddings进行重排列,用于优化多GPU训练中的数据布局 | - `embeddings`: 输入embedding张量列表<br>- `permute`: 排列索引<br>- `dim`: 要排列的维度 | **4天**<br>- CPU实现: 1天<br>- GPU kernel: 2天<br>- Python封装和测试: 1天 |
| 3 | `torch.ops.fbgemm.merge_pooled_embeddings` | `merge_pooled_embeddings_cpu` | `fbgemm_gpu/src/merge_pooled_embedding_ops/merge_pooled_embedding_ops_cpu.cpp` | 将多个pooled embeddings合并到一个张量中,用于减少通信开销 | - `pooled_embs`: 要合并的embedding列表<br>- `device`: 目标设备 | **2天**<br>- 实现: 1天<br>- 测试: 1天 |
| 4 | `torch.ops.fbgemm.permute_pooled_embs` | `permute_pooled_embs_cpu_impl`<br>`permute_pooled_embs_gpu` | CPU: `fbgemm_gpu/src/permute_pooled_embedding_ops/permute_pooled_embedding_ops_cpu.cpp`<br>GPU: `fbgemm_gpu/src/permute_pooled_embedding_ops/permute_pooled_embedding_ops.cu` | 对pooled embeddings进行重排列操作的底层函数 | - `pooled_emb`: 输入pooled embedding<br>- `permute`: 排列索引<br>- `inv_permute`: 反向排列索引 | **3天**<br>- CPU实现: 1天<br>- GPU kernel: 1.5天<br>- 测试: 0.5天 |
| 5 | `torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf` | `FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf` | `fbgemm_gpu/src/quantize_ops/quantize_ops_cpu.cpp`<br>(调用fbgemm核心库) | 将float或half精度的张量转换为融合N位行级量化格式,用于减少模型大小和内存占用 | - `input`: 输入张量(float/half)<br>- `bit_width`: 量化位宽(2/4/8)<br>- `row_output`: 行输出标志 | **3天**<br>- 调用fbgemm库: 1天<br>- 封装和测试: 2天 |
| 6 | `torch.ops.fbgemm.permute_2D_sparse_data` | `permute_2D_sparse_data_cpu`<br>Kernel: `_permute_2D_indices_weights_kernel` | CPU: `fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp`<br>GPU: `fbgemm_gpu/src/sparse_ops/sparse_permute_2d.cu` | 对2D稀疏数据(索引+权重)进行重排列,用于优化数据访问模式 | - `indices`: 稀疏索引<br>- `weights`: 稀疏权重<br>- `permute`: 排列索引<br>- `batch_size`: 批大小 | **5天**<br>- CPU kernel: 2天<br>- GPU kernel: 2.5天<br>- 测试: 0.5天 |
| 7 | `torch.ops.fbgemm.permute_1D_sparse_data` | `permute_1D_sparse_data_cpu` | CPU: `fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp`<br>GPU: `fbgemm_gpu/src/sparse_ops/sparse_permute_1d.cu` | 对1D稀疏数据进行重排列 | - `indices`: 稀疏索引<br>- `weights`: 稀疏权重<br>- `permute`: 排列索引 | **3天**<br>- CPU实现: 1天<br>- GPU kernel: 1.5天<br>- 测试: 0.5天 |
| 8 | `torch.ops.fbgemm.expand_into_jagged_permute` | `expand_into_jagged_permute_cpu`<br>Kernel: `_expand_into_jagged_permute_cpu_kernel` | CPU: `fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp`<br>GPU: `fbgemm_gpu/src/sparse_ops/sparse_expand_into_jagged_permute.cu` | 将稀疏数据扩展为jagged排列格式,用于处理变长序列 | - `indices`: 输入索引<br>- `offsets`: jagged offsets<br>- `permute`: 排列索引 | **4天**<br>- 算法设计: 1天<br>- CPU/GPU实现: 2天<br>- 测试: 1天 |
| 9 | `torch.ops.fbgemm.asynchronous_complete_cumsum` | `asynchronous_complete_cumsum_cpu` | CPU: `fbgemm_gpu/src/sparse_ops/sparse_async_cumsum.cpp`<br>GPU: `fbgemm_gpu/src/sparse_ops/sparse_async_cumsum.cu` | 异步计算完整累积和,用于jagged tensor的offsets计算 | - `values`: 输入值<br>- `dim`: 计算维度 | **3天**<br>- 异步实现: 2天<br>- 测试: 1天 |
| 10 | `torch.ops.fbgemm.offsets_range` | `offsets_range_cpu` | CPU: `fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp`<br>GPU: `fbgemm_gpu/src/sparse_ops/sparse_range.cu` | 根据offsets生成范围索引,用于索引生成 | - `offsets`: 输入offsets | **2天**<br>- 实现: 1天<br>- 测试: 1天 |
| 11 | `torch.ops.fbgemm.segment_sum_csr` | `segment_sum_csr_cpu`<br>Kernel: `_segment_sum_csr_cpu_kernel` | CPU: `fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp`<br>GPU: `fbgemm_gpu/src/sparse_ops/sparse_segment_sum_csr.cu` | 对CSR格式的稀疏张量进行分段求和 | - `values`: CSR值<br>- `indptr`: CSR索引指针<br>- `output`: 输出缓冲区 | **4天**<br>- CSR处理: 2天<br>- GPU优化: 1.5天<br>- 测试: 0.5天 |
| 12 | `torch.ops.fbgemm.keyed_jagged_index_select_dim1` | `keyed_jagged_index_select_dim1_forward`<br>Kernel: `keyed_jagged_index_select_dim1_kernel` | CPU: `fbgemm_gpu/src/jagged_tensor_ops/jagged_tensor_ops_cpu.cpp`<br>GPU: `fbgemm_gpu/src/jagged_tensor_ops/keyed_jagged_index_select_dim1.cu` | 在keyed jagged tensor的维度1上进行索引选择 | - `values`: jagged values<br>- `offsets`: jagged offsets<br>- `indices`: 选择索引 | **5天**<br>- 复杂索引逻辑: 3天<br>- GPU kernel: 1.5天<br>- 测试: 0.5天 |
| 13 | `torch.ops.fbgemm.block_bucketize_sparse_features` | `block_bucketize_sparse_features_cpu`<br>Kernel: `_block_bucketize_sparse_features_cpu_kernel` | CPU: `fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp`<br>GPU: `fbgemm_gpu/src/sparse_ops/sparse_block_bucketize_features.cu`<br>`fbgemm_gpu/src/sparse_ops/sparse_block_bucketize_features_2d_weights.cu` | 将稀疏特征分桶到块中,用于优化embedding查找 | - `indices`: 稀疏索引<br>- `weights`: 稀疏权重<br>- `block_size`: 块大小<br>- `bucket_size`: 桶大小 | **7天**<br>- 分桶算法: 3天<br>- 2D权重支持: 2天<br>- GPU优化: 1.5天<br>- 测试: 0.5天 |
| 14 | `fbgemm_gpu.split_table_batched_embeddings_ops_inference.IntNBitTableBatchedEmbeddingBagsCodegen` | Python类 (包含17个算子) | Python: `fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_inference.py`<br>C++: `fbgemm_gpu/codegen/inference/` | 表批量embedding推理模块,支持INT N位量化embedding的高效查找,包含缓存管理、前向查找、剪枝等17个算子 | 详见第7.1节TBE推理模块算子详解 | **30天**<br>- 架构设计: 5天<br>- Codegen框架: 10天<br>- Kernel实现: 10天<br>- 测试优化: 5天 |
| 15 | `fbgemm_gpu.split_table_batched_embeddings_ops_training.SplitTableBatchedEmbeddingBagsCodegen` | Python类 (包含19个算子) | Python: `fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_training.py`<br>C++: `fbgemm_gpu/codegen/training/` | 表批量embedding训练模块,支持梯度和优化器状态管理,包含前向/反向/优化器等19个算子 | 详见第7.2节TBE训练模块算子详解 | **45天**<br>- 架构设计: 7天<br>- Codegen框架: 15天<br>- 前向/反向kernel: 15天<br>- 优化器集成: 5天<br>- 测试: 3天 |

### 2.2 Stable Python API分类汇总

按文档结构分类的Stable Python API:

#### 2.2.1 Table Batched Embedding (TBE)模块
- **训练**: `SplitTableBatchedEmbeddingBagsCodegen` (45天)
- **推理**: `IntNBitTableBatchedEmbeddingBagsCodegen` (30天)

#### 2.2.2 Pooled Embedding操作
- `PermutePooledEmbeddings` (4天)
- `merge_pooled_embeddings` (2天)
- `permute_pooled_embs` (3天)
- **小计**: 9天

#### 2.2.3 Sparse操作
- `permute_2D_sparse_data` (5天)
- `permute_1D_sparse_data` (3天)
- `expand_into_jagged_permute` (4天)
- `asynchronous_complete_cumsum` (3天)
- `offsets_range` (2天)
- `segment_sum_csr` (4天)
- `keyed_jagged_index_select_dim1` (5天)
- `block_bucketize_sparse_features` (7天)
- **小计**: 33天

#### 2.2.4 Jagged Tensor操作
- `jagged_to_padded_dense` (5天)
- **小计**: 5天

#### 2.2.5 量化操作
- `FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf` (3天)
- **小计**: 3天

**Stable Python API总工作量**: 125天/人

---

## 3. C++ API映射表

### 3.1 C++ API详细映射表

| 序号 | C++ API名称 | Kernel/实现函数名称 | 实现文件路径 | API功能和作用 | 参数说明和取值范围 | 开发工作量(天/人) |
|------|------------|-------------------|-------------|-------------|------------------|------------------|
| **3.1 Embedding操作** | | | | | | |
| 1 | `embedding-cuda` | `embedding_forward_split` | `fbgemm_gpu/codegen/embedding_forward_split_host.cpp`<br>(codegen生成) | Embedding前向传播,支持split table | - `weights`: embedding权重<br>- `indices`: 查找索引<br>- `offsets`: 位置偏移 | **20天**<br>- Kernel设计: 10天<br>- Codegen: 7天<br>- 测试: 3天 |
| 2 | `embedding-cpu` | `embedding_forward_cpu` | `fbgemm_gpu/codegen/embedding_forward_cpu.cpp` | CPU版本的embedding前向传播 | 同上 | **10天** |
| **3.2 稀疏数据操作** | | | | | | |
| 3 | `asynchronous_exclusive_cumsum` | `asynchronous_exclusive_cumsum_gpu/cpu` | `fbgemm_gpu/src/sparse_ops/sparse_async_cumsum.cpp`<br>`fbgemm_gpu/src/sparse_ops/sparse_async_cumsum.cu` | 异步独占累积和(不包含当前位置) | - `input`: 输入张量<br>- `dim`: 维度 | **2天** |
| 4 | `asynchronous_inclusive_cumsum` | `asynchronous_inclusive_cumsum_gpu/cpu` | 同上 | 异步包含累积和(包含当前位置) | 同上 | **2天** |
| 5 | `asynchronous_batched_complete_cumsum` | `asynchronous_batched_complete_cumsum_gpu/cpu` | 同上 | 批量异步完整累积和 | - `inputs`: 输入张量列表 | **3天** |
| 6 | `reorder_batched_ad_lengths` | `reorder_batched_ad_lengths_gpu/cpu` | `fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp`<br>`fbgemm_gpu/src/sparse_ops/sparse_reorder_batched_ad.cu` | 重排序批处理广告长度 | - `lengths`: 长度张量<br>- `permute`: 排列索引 | **3天** |
| 7 | `reorder_batched_ad_indices` | `reorder_batched_ad_indices_gpu/cpu` | 同上 | 重排序批处理广告索引 | - `indices`: 索引张量<br>- `permute`: 排列索引 | **3天** |
| 8 | `embedding_bag_rowwise_prune` | `embedding_bag_rowwise_prune` | `fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp` | Embedding bag行级剪枝 | - `output`: 输出张量<br>- `mask`: 剪枝掩码 | **4天** |
| 9 | `pack_segments` | `pack_segments_cpu/cuda` | `fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp`<br>`fbgemm_gpu/src/sparse_ops/sparse_pack_segments.cu` | 打包段,将可变长度序列打包 | - `values`: 值张量<br>- `lengths`: 长度张量 | **5天** |
| 10 | `group_index_select_dim0` | `group_index_select_dim0_forward_impl` | `fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp`<br>`fbgemm_gpu/src/sparse_ops/sparse_group_index_select_dim0.cu` | 维度0分组索引选择 | - `input`: 输入张量<br>- `indices`: 索引列表<br>- `groups`: 分组信息 | **6天** |
| 11 | `jagged_index_select_2d` | `jagged_index_select_2d_forward` | `fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp`<br>`fbgemm_gpu/src/sparse_ops/sparse_jagged_index_select_2d.cu` | 2D jagged tensor索引选择 | - `values`: jagged values<br>- `offsets`: jagged offsets<br>- `indices`: 选择索引 | **5天** |
| 12 | `jagged_slice` | `jagged_slice_forward` | `fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp` | Jagged tensor切片 | - `values`: jagged values<br>- `offsets`: jagged offsets<br>- `start`: 起始位置<br>- `end`: 结束位置 | **4天** |
| 13 | `histogram_binning_calibration` | `histogram_binning_calibration_cuda/cpu` | `fbgemm_gpu/src/sparse_ops/sparse_histogram_binning_calibration.cpp`<br>`fbgemm_gpu/src/sparse_ops/sparse_histogram_binning_calibration.cu` | 直方图分箱校准,用于量化校准 | - `input`: 输入张量<br>- `bin_edges`: 分箱边界 | **5天** |
| **3.3 量化操作** | | | | | | |
| 14 | `_float_to_fused8bitrowwise_gpu` | `_float_to_fused8bitrowwise_gpu` | `fbgemm_gpu/src/sparse_ops/sparse_quantize.cpp`<br>`fbgemm_gpu/src/sparse_ops/sparse_quantize.cu` | Float转融合8位行级量化 | - `input`: float输入<br>- `output`: 8位量化输出 | **3天** |
| 15 | `_float_to_FP8rowwise_gpu` | `_float_to_FP8rowwise_gpu` | 同上 | Float转FP8行级量化 | 同上 | **3天** |
| 16 | `_fused8bitrowwise_to_float_gpu` | `_fused8bitrowwise_to_float_gpu` | 同上 | 融合8位转Float | - `input`: 8位量化输入<br>- `output`: float输出 | **2天** |
| 17 | `_FP8rowwise_to_float_gpu` | `_FP8rowwise_to_float_gpu` | 同上 | FP8转Float | 同上 | **2天** |
| 18 | `_float_to_fusednbitrowwise_gpu` | `_float_to_fusednbitrowwise_gpu` | 同上 | Float转n位行级量化 | - `input`: float输入<br>- `bit_width`: 位宽(2/4/8) | **4天** |
| 19 | `_fusednbitrowwise_to_float_gpu` | `_fusednbitrowwise_to_float_gpu` | 同上 | n位转Float | - `input`: n位量化输入<br>- `bit_width`: 位宽 | **3天** |
| 20 | `quantize_mx_cuda` | `quantize_mx_cuda` | 同上 | MX格式量化(Microscaling) | - `input`: 输入张量<br>- `scale`: 缩放因子 | **4天** |
| 21 | `dequantize_mx_cuda` | `dequantize_mx_cuda` | 同上 | MX格式反量化 | - `input`: 量化输入<br>- `scale`: 缩放因子 | **3天** |
| **3.4 Jagged Tensor操作** | | | | | | |
| 22 | `jagged_2d_to_dense` | `jagged_2d_to_dense_gpu_forward` | `fbgemm_gpu/src/jagged_tensor_ops/jagged_2d_to_dense.cu` | 2D jagged转dense | - `values`: jagged values<br>- `offsets`: jagged offsets<br>- `max_length`: 最大长度 | **5天** |
| 23 | `jagged_1d_to_dense` | `jagged_1d_to_dense` | `fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp` | 1D jagged转dense | 同上 | **3天** |
| 24 | `dense_to_jagged` | `dense_to_jagged` | `fbgemm_gpu/src/jagged_tensor_ops/dense_to_jagged.cu` | Dense转jagged | - `dense`: dense张量<br>- `offsets`: jagged offsets | **4天** |
| 25 | `jagged_dense_elementwise_add` | `jagged_dense_elementwise_dense_output_kernel` | `fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_elementwise_add.cu` | Jagged与dense元素级加法 | - `jagged_values`: jagged值<br>- `dense`: dense张量 | **4天** |
| 26 | `jagged_dense_elementwise_mul` | `jagged_dense_elementwise_mul` | `fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_elementwise_mul.cu` | Jagged与dense元素级乘法 | 同上 | **4天** |
| 27 | `jagged_softmax` | `jagged_softmax` | `fbgemm_gpu/src/jagged_tensor_ops/jagged_softmax.cu` | Jagged tensor softmax | - `values`: jagged values<br>- `offsets`: jagged offsets | **5天** |
| 28 | `jagged_jagged_bmm` | `jagged_jagged_bmm` | `fbgemm_gpu/src/jagged_tensor_ops/jagged_jagged_bmm.cu` | Jagged-jagged批量矩阵乘法 | - `a`: jagged tensor A<br>- `b`: jagged tensor B | **7天** |
| 29 | `jagged_dense_bmm` | `jagged_dense_bmm` | `fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_bmm.cu` | Jagged-dense批量矩阵乘法 | - `a`: jagged tensor<br>- `b`: dense张量 | **6天** |
| **3.5 内存管理** | | | | | | |
| 30 | `new_managed_tensor` | `new_managed_tensor` | `fbgemm_gpu/src/cumem_utils/cumem_utils.cpp` | 分配统一管理内存(UVM)张量 | - `size`: 张量大小 | **2天** |
| 31 | `new_host_mapped_tensor` | `new_host_mapped_tensor` | 同上 | 分配主机映射内存张量 | - `size`: 张量大小 | **2天** |
| 32 | `uvm_storage` | `uvm_storage` | 同上 | 检查张量是否使用UVM | - `tensor`: 输入张量 | **1天** |
| 33 | `uvm_to_cpu` | `uvm_to_cpu` | 同上 | UVM张量转CPU张量 | - `tensor`: UVM张量 | **2天** |
| 34 | `uvm_to_device` | `uvm_to_device` | 同上 | 创建UVM张量的设备视图 | - `tensor`: UVM张量<br>- `device`: 目标设备 | **2天** |
| 35 | `uvm_cuda_mem_prefetch_async` | `uvm_cuda_mem_prefetch_async` | 同上 | 异步预取UVM内存到设备 | - `tensor`: UVM张量<br>- `device`: 目标设备 | **2天** |
| **3.6 哈希操作** | | | | | | |
| 36 | `create_zch_buffer_cpu` | `create_zch_buffer_cpu` | `fbgemm_gpu/src/faster_hash_ops/faster_hash_ops_cpu.cpp` | 创建零碰撞哈希(ZCH)缓冲区 | - `size`: 缓冲区大小 | **3天** |
| 37 | `murmur_hash3_cpu` | `murmur_hash3_cpu` | 同上 | Murmur3哈希算法 | - `data`: 输入数据<br>- `seed`: 种子值 | **2天** |
| 38 | `zero_collision_hash_cpu` | `zero_collision_hash_cpu` | 同上 | 零碰撞哈希操作 | - `data`: 输入数据<br>- `table`: 哈希表 | **4天** |
| **3.7 布局转换** | | | | | | |
| 39 | `recat_embedding_grad_output_cuda` | `recat_embedding_grad_output_cuda` | `fbgemm_gpu/src/layout_transform/recat_embedding_grad_output.cu` | 重新拼接embedding梯度输出 | - `grad_output`: 梯度输出<br>- `old_cat_dim`: 原拼接维度 | **4天** |
| **3.8 合并操作** | | | | | | |
| 40 | `all_to_one_device` | `all_to_one_device` | `fbgemm_gpu/src/merge_pooled_embedding_ops/merge_pooled_embedding_ops_cpu.cpp` | 将所有输入张量移动到目标设备 | - `inputs`: 输入张量列表<br>- `device`: 目标设备 | **1天** |
| **3.9 Feature Gates** | | | | | | |
| 41 | `is_feature_enabled` | `is_feature_enabled` | `fbgemm_gpu/src/feature_gates/feature_gates.cpp` | 检查功能是否启用 | - `feature`: 功能标志枚举 | **2天** |
| 42 | `is_feature_enabled_from_env` | `is_feature_enabled_from_env` | 同上 | 从环境变量检查功能 | - `feature`: 功能标志枚举 | **1天** |
| **3.10 SSD Embedding** | | | | | | |
| 43 | `hash_shard` | `hash_shard` | `fbgemm_gpu/src/ssd_embedding_ops/ssd_embedding_ops.cpp` | 哈希分片函数(用于SSD缓存) | - `key`: 键值<br>- `num_shards`: 分片数 | **2天** |
| 44 | `get_rocksdb_path` | `get_rocksdb_path` | 同上 | 生成RocksDB路径 | - `base_path`: 基础路径<br>- `shard_id`: 分片ID | **1天** |
| **3.11 输入合并** | | | | | | |
| 45 | `tbe_input_combine_cpu` | `tbe_input_combine_cpu` | `fbgemm_gpu/src/input_combine/input_combine.cpp` | TBE输入合并(CPU版本) | - `inputs`: 输入张量列表 | **3天** |
| 46 | `tbe_input_combine_with_length_cuda` | `tbe_input_combine_with_length_cuda` | 同上 | TBE输入合并(CUDA版本,基于长度) | - `inputs`: 输入列表<br>- `lengths`: 长度张量 | **4天** |
| **3.12 实验性操作** | | | | | | |
| 47 | `gqa_attn_splitk_wmma_kernel` | `gqa_attn_splitk_wmma_kernel` | `fbgemm_gpu/src/experimental_ops/gen_ai_attention.cu` | Grouped Query Attention Split-K WMMA内核 | - `query`: 查询张量<br>- `key`: 键张量<br>- `value`: 值张量 | **15天**<br>- 算法实现: 10天<br>- WMMA优化: 5天 |

### 3.2 C++ API分类汇总

按文档结构分类的C++ API:

#### 3.2.1 Embedding操作
- `embedding_forward_split` (20天)
- `embedding_forward_cpu` (10天)
- **小计**: 30天

#### 3.2.2 稀疏数据操作
- 累积和系列 (7天)
- 重排序系列 (6天)
- 打包和索引选择 (15天)
- 校准操作 (5天)
- **小计**: 33天

#### 3.2.3 量化操作
- 8位量化系列 (5天)
- FP8量化系列 (5天)
- n位量化系列 (7天)
- MX量化 (7天)
- **小计**: 24天

#### 3.2.4 Jagged Tensor操作
- 转换操作 (12天)
- 元素操作 (8天)
- 矩阵乘法 (13天)
- **小计**: 33天

#### 3.2.5 内存管理
- UVM操作 (11天)
- **小计**: 11天

#### 3.2.6 其他操作
- 哈希操作 (9天)
- Feature Gates (3天)
- SSD操作 (3天)
- 输入合并 (7天)
- 布局转换 (4天)
- 实验性操作 (15天)
- **小计**: 41天

**C++ API总工作量**: 172天/人

---

## 4. C++和Python接口对应Kernel实现关系表

下表列出了C++和Python接口使用相同底层kernel实现的情况。

| 序号 | Python API名称 | C++ API名称 | 对应的Kernel名称 | Kernel实现文件路径 |
|------|---------------|------------|----------------|------------------|
| **4.1 Jagged Tensor操作** | | | | |
| 1 | `torch.ops.fbgemm.jagged_to_padded_dense` | `jagged_to_padded_dense` | `jagged_to_padded_dense_forward`<br>`jagged_to_padded_dense_backward` | `fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_forward.cu`<br>`fbgemm_gpu/src/jagged_tensor_ops/jagged_to_padded_dense_backward.cu` |
| 2 | - | `jagged_2d_to_dense` | `jagged_2d_to_dense_gpu_forward`<br>`jagged_2d_to_dense_gpu_backward` | `fbgemm_gpu/src/jagged_tensor_ops/jagged_2d_to_dense.cu` |
| 3 | - | `jagged_1d_to_dense` | `jagged_1d_to_dense` | `fbgemm_gpu/src/jagged_tensor_ops/jagged_1d_to_dense.cu` |
| 4 | - | `dense_to_jagged` | `dense_to_jagged` | `fbgemm_gpu/src/jagged_tensor_ops/dense_to_jagged.cu` |
| 5 | - | `jagged_dense_elementwise_add` | `jagged_dense_elementwise_dense_output_kernel_` | `fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_elementwise_add.cu` |
| 6 | - | `jagged_dense_elementwise_mul` | `jagged_dense_elementwise_mul_kernel` | `fbgemm_gpu/src/jagged_tensor_ops/jagged_dense_elementwise_mul.cu` |
| **4.2 Pooled Embedding操作** | | | | |
| 7 | `torch.ops.fbgemm.permute_pooled_embs` | `permute_pooled_embs_gpu/cpu` | `permute_pooled_embs_cuda`<br>`permute_pooled_embs_cpu_impl` | `fbgemm_gpu/src/permute_pooled_embedding_ops/permute_pooled_embedding_ops.cu`<br>`fbgemm_gpu/src/permute_pooled_embedding_ops/permute_pooled_embedding_ops_cpu.cpp` |
| 8 | `torch.ops.fbgemm.merge_pooled_embeddings` | `all_to_one_device` | `all_to_one_device` | `fbgemm_gpu/src/merge_pooled_embedding_ops/merge_pooled_embedding_ops_cpu.cpp` |
| **4.3 Sparse操作** | | | | |
| 9 | `torch.ops.fbgemm.permute_2D_sparse_data` | `permute_2D_sparse_data_cuda/cpu` | `_permute_2D_indices_weights_kernel`<br>`_permute_2D_lengths_kernel` | `fbgemm_gpu/src/sparse_ops/sparse_permute_2d.cu`<br>`fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp` |
| 10 | `torch.ops.fbgemm.permute_1D_sparse_data` | `permute_1D_sparse_data_cuda/cpu` | `_permute_1D_kernel` | `fbgemm_gpu/src/sparse_ops/sparse_permute_1d.cu`<br>`fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp` |
| 11 | `torch.ops.fbgemm.expand_into_jagged_permute` | `expand_into_jagged_permute_cuda/cpu` | `_expand_into_jagged_permute_kernel` | `fbgemm_gpu/src/sparse_ops/sparse_expand_into_jagged_permute.cu`<br>`fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp` |
| 12 | `torch.ops.fbgemm.asynchronous_complete_cumsum` | `asynchronous_complete_cumsum_gpu/cpu` | `asynchronous_complete_cumsum_gpu/cpu_kernel` | `fbgemm_gpu/src/sparse_ops/sparse_async_cumsum.cu`<br>`fbgemm_gpu/src/sparse_ops/sparse_async_cumsum.cpp` |
| 13 | - | `asynchronous_exclusive_cumsum_gpu/cpu` | `asynchronous_exclusive_cumsum_kernel` | 同上 |
| 14 | - | `asynchronous_inclusive_cumsum_gpu/cpu` | `asynchronous_inclusive_cumsum_kernel` | 同上 |
| 15 | `torch.ops.fbgemm.offsets_range` | `offsets_range_cuda/cpu` | `offsets_range_kernel` | `fbgemm_gpu/src/sparse_ops/sparse_range.cu`<br>`fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp` |
| 16 | `torch.ops.fbgemm.segment_sum_csr` | `segment_sum_csr_cuda/cpu` | `_segment_sum_csr_kernel` | `fbgemm_gpu/src/sparse_ops/sparse_segment_sum_csr.cu`<br>`fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp` |
| 17 | `torch.ops.fbgemm.block_bucketize_sparse_features` | `block_bucketize_sparse_features_cuda/cpu` | `_block_bucketize_sparse_features_kernel` | `fbgemm_gpu/src/sparse_ops/sparse_block_bucketize_features.cu`<br>`fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp` |
| 18 | `torch.ops.fbgemm.keyed_jagged_index_select_dim1` | `keyed_jagged_index_select_dim1` | `keyed_jagged_index_select_dim1_kernel` | `fbgemm_gpu/src/jagged_tensor_ops/keyed_jagged_index_select_dim1.cu` |
| 19 | - | `pack_segments_cuda/cpu` | `pack_segments_forward_kernel` | `fbgemm_gpu/src/sparse_ops/sparse_pack_segments.cu`<br>`fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp` |
| 20 | - | `group_index_select_dim0` | `group_index_select_dim0_kernel` | `fbgemm_gpu/src/sparse_ops/sparse_group_index_select_dim0.cu`<br>`fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp` |
| **4.4 量化操作** | | | | |
| 21 | `torch.ops.fbgemm.FloatOrHalfToFusedNBitRowwiseQuantizedSBHalf` | `_float_or_half_to_fusednbitrowwise_gpu/cpu` | `_float_or_half_to_fusednbitrowwise_kernel` | `fbgemm_gpu/src/sparse_ops/sparse_quantize.cu`<br>`fbgemm_gpu/src/sparse_ops/sparse_ops_cpu.cpp` |
| 22 | - | `_float_to_fused8bitrowwise_gpu` | `_float_to_fused8bitrowwise_kernel` | 同上 |
| 23 | - | `_fused8bitrowwise_to_float_gpu` | `_fused8bitrowwise_to_float_kernel` | 同上 |
| 24 | - | `_float_to_FP8rowwise_gpu` | `_float_to_FP8rowwise_kernel` | 同上 |
| 25 | - | `_FP8rowwise_to_float_gpu` | `_FP8rowwise_to_float_kernel` | 同上 |
| **4.5 内存管理** | | | | |
| 26 | - | `new_managed_tensor` | `new_managed_tensor_impl` | `fbgemm_gpu/src/cumem_utils/cumem_utils.cpp` |
| 27 | - | `uvm_to_cpu` | `uvm_to_cpu_impl` | 同上 |
| 28 | - | `uvm_cuda_mem_prefetch_async` | `uvm_cuda_mem_prefetch_async_impl` | 同上 |
| **4.6 Feature Gates** | | | | |
| 29 | `fbgemm_gpu.config.FeatureGate` | `is_feature_enabled` | `is_feature_enabled_impl` | `fbgemm_gpu/src/feature_gates/feature_gates.cpp` |
| 30 | `fbgemm_gpu.config.FeatureGate` | `is_feature_enabled_from_env` | `is_feature_enabled_from_env_impl` | 同上 |
| **4.7 哈希操作** | | | | |
| 31 | - | `murmur_hash3_cpu` | `murmur_hash3_impl` | `fbgemm_gpu/src/faster_hash_ops/faster_hash_ops_cpu.cpp` |
| 32 | - | `zero_collision_hash_cpu` | `zero_collision_hash_impl` | 同上 |
| **4.8 输入合并** | | | | |
| 33 | - | `tbe_input_combine_cpu` | `tbe_input_combine_cpu_impl` | `fbgemm_gpu/src/input_combine/input_combine.cpp` |
| 34 | - | `tbe_input_combine_with_length_cuda` | `tbe_input_combine_with_length_cuda_kernel` | 同上 |

---

## 5. 工作量计算方法说明

### 5.1 工作量评估标准

本报告中工作量(天/人)的计算基于以下标准和假设:

#### 5.1.1 开发人员能力假设
- **高级CUDA工程师**: 3年以上CUDA开发经验,熟悉GPU架构优化
- **中级C++工程师**: 2年以上C++经验,熟悉PyTorch扩展开发
- **每天有效工作时间**: 6小时(考虑代码审查、会议、调试等)

#### 5.1.2 工作量分解

对于每个API/Kernel,工作量包含以下组成部分:

| 阶段 | 占比 | 说明 |
|------|------|------|
| **需求分析和设计** | 15% | 理解API需求,设计接口,规划实现方案 |
| **算法实现** | 35% | 核心算法逻辑实现,CPU版本开发 |
| **GPU Kernel开发** | 30% | CUDA kernel实现,内存访问优化,并行化设计 |
| **Python绑定** | 10% | PyBind11绑定,Python封装 |
| **单元测试** | 7% | 编写测试用例,边界条件验证 |
| **性能优化** | 8% | 性能分析,瓶颈优化 |

#### 5.1.3 复杂度分级

根据实现难度,将API分为三个复杂度等级:

##### **简单复杂度** (1-3天)
- 特点: 算法清晰,无复杂并行逻辑
- 示例: `offsets_range`, `all_to_one_device`
- 工作量: 设计0.5天 + 实现1天 + GPU 0.5天 + 测试0.5天 + 优化0.5天 = 3天

##### **中等复杂度** (4-7天)
- 特点: 需要考虑并行化,有一定算法复杂度
- 示例: `permute_2D_sparse_data`, `jagged_to_padded_dense`
- 工作量: 设计1天 + 实现2天 + GPU 2天 + 测试0.5天 + 优化1.5天 = 7天

##### **高复杂度** (8-15天)
- 特点: 算法复杂,需要深度优化,codegen框架集成
- 示例: `SplitTableBatchedEmbeddingBagsCodegen`, `gqa_attn_splitk_wmma_kernel`
- 工作量: 设计2天 + 实现4天 + GPU 5天 + 测试1天 + 优化3天 = 15天

##### **极高复杂度** (30-45天)
- 特点: 系统级模块,需要架构设计,codegen框架,多模块集成
- 示例: TBE训练/推理模块
- 工作量: 架构5-7天 + 框架10-15天 + Kernel 10-15天 + 测试3-5天 + 优化5-8天

### 5.2 具体计算示例

#### 5.2.1 示例1: `jagged_to_padded_dense` (中等复杂度)

```
设计阶段 (1天):
- 分析jagged tensor数据结构
- 设计padded策略和填充逻辑
- 规划CUDA kernel并行方案

实现阶段 (2天):
- 实现CPU前向传播 (0.5天)
- 实现CPU反向传播 (0.5天)
- 处理边界条件 (0.5天)
- 代码审查和重构 (0.5天)

GPU开发 (2天):
- 实现CUDA前向kernel (1天)
- 实现CUDA反向kernel (0.8天)
- 内存访问优化 (0.2天)

测试 (0.5天):
- 单元测试编写 (0.3天)
- 边界条件测试 (0.2天)

优化 (0.5天):
- 性能分析 (0.3天)
- kernel调优 (0.2天)

总计: 6天 (报告中取整为5天)
```

#### 5.2.2 示例2: `SplitTableBatchedEmbeddingBagsCodegen` (极高复杂度)

```
架构设计 (7天):
- 分析TBE训练需求 (2天)
- 设计codegen框架 (3天)
- 规划优化器集成 (2天)

Codegen框架 (15天):
- 实现代码生成器 (5天)
- 生成前向传播代码 (3天)
- 生成反向传播代码 (4天)
- 生成优化器更新代码 (3天)

Kernel实现 (15天):
- embedding查找kernel (5天)
- 梯度计算kernel (5天)
- 优化器更新kernel (3天)
- 内存管理和同步 (2天)

测试 (3天):
- 功能测试 (1天)
- 性能测试 (1天)
- 回归测试 (1天)

优化 (5天):
- Kernel融合优化 (2天)
- 内存访问优化 (2天)
- 多GPU扩展优化 (1天)

总计: 45天
```

### 5.3 风险系数

在实际项目中,需要考虑以下风险并增加缓冲时间:

| 风险类型 | 缓冲系数 | 说明 |
|---------|---------|------|
| **技术风险** | 1.2x | 新技术栈,算法复杂度高 |
| **需求变更** | 1.1x | 需求不明确或可能变更 |
| **集成风险** | 1.15x | 与现有系统集成 |
| **性能要求** | 1.2x | 有严格性能指标要求 |

综合风险系数 = 1.0 + (0.2 + 0.1 + 0.15 + 0.2) = **1.65x**

例如,`jagged_to_padded_dense`的实际工作量预估:
- 基础工作量: 5天
- 考虑风险: 5天 × 1.65 = 8.25天 ≈ **9天**

### 5.4 团队规模计算

根据总工作量计算所需团队规模:

**Stable Python API**: 125天/人
- 3个月(60工作日)完成: 125 ÷ 60 = **2.1人** → 3人团队
- 6个月(120工作日)完成: 125 ÷ 120 = **1.04人** → 2人团队

**C++ API**: 172天/人
- 3个月完成: 172 ÷ 60 = **2.9人** → 4人团队
- 6个月完成: 172 ÷ 120 = **1.4人** → 2人团队

**全部API**: 297天/人
- 6个月完成: 297 ÷ 120 = **2.5人** → 3人团队
- 12个月完成: 297 ÷ 240 = **1.2人** → 2人团队

### 5.5 工作量计算公式总结

```
基础工作量 = 设计时间 + 实现时间 + GPU开发时间 + 测试时间 + 优化时间

调整后工作量 = 基础工作量 × 复杂度系数 × 风险系数

其中:
- 复杂度系数: 简单1.0, 中等1.2, 高1.5, 极高2.0
- 风险系数: 1.0 ~ 1.65 (根据项目实际情况)
```

---

## 6. 总结

### 6.1 API统计汇总

| 类别 | API数量 | 总工作量(天/人) |
|------|---------|----------------|
| **Stable Python API** | 15个 | 125 |
| **C++ API** | 47个核心函数 | 172 |
| **总计** | 62个 | 297 |

### 6.2 关键发现

1. **代码生成(Codegen)**: TBE模块大量使用codegen自动生成代码,提高了开发效率和代码一致性

2. **双重实现**: 大多数API都提供CPU和GPU两个版本,增加了开发和维护成本

3. **量化支持**: 丰富的量化操作(8位、FP8、n位、MX)反映了推荐系统对模型压缩的需求

4. **Jagged Tensor**: 专门的数据结构和操作处理变长序列,是推荐系统NLP场景的核心

5. **内存管理**: 完善的UVM(统一虚拟内存)支持,优化了大embedding表的内存访问

### 6.3 开发建议

1. **优先实现Stable API**: Stable API提供向后兼容保证,应优先开发
2. **复用现有kernel**: C++和Python API共享kernel,避免重复开发
3. **测试驱动**: 每个API都需要完整的单元测试和性能测试
4. **渐进式优化**: 先实现功能,再进行性能优化
5. **文档先行**: API文档和代码注释应与实现同步

---

## 7. TBE(Table Batched Embedding)模块算子详解

TBE是FBGEMM_GPU的核心模块,提供高性能的表批量嵌入操作,支持大规模稀疏特征推荐模型的训练和推理。

### 7.1 TBE推理模块算子

**Python模块**: `IntNBitTableBatchedEmbeddingBagsCodegen`
**模块路径**: `fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_inference.py`
**Codegen目录**: `fbgemm_gpu/codegen/inference/`

#### 7.1.1 缓存管理算子(6个)

| 序号 | 算子名称 | CUDA Kernel | 功能描述 | 参数说明 | 开发工作量 |
|------|---------|------------|---------|---------|-----------|
| 1 | `torch.ops.fbgemm.linearize_cache_indices` | `linearize_cache_indices_cuda` | 将多维索引转换为一维缓存索引 | - `cache_hash_size_cumsum`: 缓存哈希大小累积和<br>- `indices`: 原始索引<br>- `offsets`: 位置偏移 | 2天 |
| 2 | `torch.ops.fbgemm.lxu_cache_lookup` | `lxu_cache_lookup_cuda` | 在32/64路关联缓存中查找(LRU/LFU) | - `linear_cache_indices`: 线性化索引<br>- `lxu_cache_state`: 缓存状态<br>输出: cache位置或-1(miss) | 3天 |
| 3 | `torch.ops.fbgemm.direct_mapped_lxu_cache_lookup` | `direct_mapped_lxu_cache_lookup_cuda` | 在直接映射缓存中查找 | - `linear_cache_indices`: 线性化索引<br>- `lxu_cache_state`: 直接映射缓存状态 | 2天 |
| 4 | `torch.ops.fbgemm.lru_cache_populate_byte` | `lru_cache_populate_byte_cuda` | 使用LRU策略填充缓存(量化版本) | - `weights`: UVM权重<br>- `cache_weights`: GPU缓存<br>- `row_alignment`: 行对齐(默认16) | 3天 |
| 5 | `torch.ops.fbgemm.direct_mapped_lru_cache_populate_byte` | `direct_mapped_lru_cache_populate_byte_cuda` | 直接映射LRU缓存填充 | 同上 | 2天 |
| 6 | `torch.ops.fbgemm.lfu_cache_populate_byte` | `lfu_cache_populate_byte_cuda` | 使用LFU策略填充缓存(量化版本) | 同上 | 3天 |

**实现文件**:
- C++: `fbgemm_gpu/src/split_embeddings_cache/split_embeddings_cache_ops.cpp`
- CUDA: `fbgemm_gpu/src/split_embeddings_cache/split_embeddings_cache_ops.cu`

#### 7.1.2 前向推理算子(3个主要算子)

| 序号 | 算子名称 | CUDA Kernel | 功能描述 | 支持的数据类型 | 开发工作量 |
|------|---------|------------|---------|-------------|-----------|
| 7 | `torch.ops.fbgemm.int_nbit_split_embedding_codegen_lookup_function` | `int_nbit_split_embedding_codegen_forward_unweighted_kernel`<br>`int_nbit_split_embedding_codegen_forward_weighted_kernel` | 量化embedding表查找主函数,支持INT2/INT4/INT8/FP8/FP16/FP32 | FP32, FP16, FP8, INT8, INT4, INT2 | 8天 |
| 8 | `torch.ops.fbgemm.pruned_hashmap_lookup` | `int_nbit_split_embedding_codegen_forward_pruned_hashmap_lookup_kernel` | 使用开放寻址哈希表进行剪枝索引查找 | 适用于模型压缩场景 | 4天 |
| 9 | `torch.ops.fbgemm.pruned_array_lookup` | `pruned_array_lookup_cuda` | 使用紧凑数组进行剪枝索引查找(更高效) | 适用于连续密集索引 | 2天 |

**实现文件**:
- `fbgemm_gpu/codegen/inference/embedding_forward_quantized_split_lookup.cu`
- `fbgemm_gpu/codegen/inference/embedding_forward_quantized_split_nbit_kernel_template.cu`
- `fbgemm_gpu/codegen/inference/embedding_forward_quantized_split_nbit_host_template.cu`

**Kernel变体**:
- **按精度**: INT2, INT4, INT8, FP8, FP16, FP32
- **按权重**: weighted, unweighted
- **按Pooling**: SUM, MEAN, MAX

#### 7.1.3 边界检查算子(1个)

| 序号 | 算子名称 | CUDA Kernel | 功能描述 | 检查模式 | 开发工作量 |
|------|---------|------------|---------|---------|-----------|
| 10 | `torch.ops.fbgemm.bounds_check_indices` | `bounds_check_indices_v1/v2_kernel` | 检查索引和偏移量是否越界 | NONE(跳过)<br>FATAL(越界报错)<br>WARNING(警告并修正)<br>IGNORE(静默修正) | 2天 |

**实现文件**:
- `fbgemm_gpu/codegen/utils/embedding_bounds_check_v1.cu`
- `fbgemm_gpu/codegen/utils/embedding_bounds_check_v2.cu`
- `fbgemm_gpu/codegen/utils/embedding_bounds_check_host.cpp`

#### 7.1.4 内存管理和原地更新算子(3个)

| 序号 | 算子名称 | CUDA Kernel | 功能描述 | 开发工作量 |
|------|---------|------------|---------|-----------|
| 11 | `torch.ops.fbgemm.new_unified_tensor` | `new_managed_tensor_impl` | 创建统一虚拟内存(UVM)tensor,支持host-mapped | 1天 |
| 12 | `torch.ops.fbgemm.emb_inplace_update` | `emb_inplace_update_cuda` | 原地更新embedding权重,用于在线学习 | - `update_table_indices`: 要更新的表索引<br>- `update_row_indices`: 要更新的行索引<br>- `update_weights`: 新权重值 | 3天 |
| 13 | `torch.ops.fbgemm.linearize_cache_indices_from_row_idx` | `linearize_cache_indices_from_row_idx_cuda` | 从行索引线性化缓存索引(用于更新操作) | 1天 |

#### 7.1.5 工具算子(2个)

| 序号 | 算子名称 | CUDA Kernel | 功能描述 | 开发工作量 |
|------|---------|------------|---------|-----------|
| 14 | `torch.ops.fbgemm.get_infos_metadata` | `get_infos_metadata_cuda` |获取VBE(Variable Batch Embedding)元数据 | 1天 |
| 15 | `torch.ops.fbgemm.pruned_hashmap_insert` | `pruned_hashmap_insert_cpu` | 插入剪枝哈希表条目(CPU实现) | 2天 |

**推理模块总计**: 约17个PyTorch算子接口,背后有**数十个CUDA kernel变体**

### 7.2 TBE训练模块算子

**Python模块**: `SplitTableBatchedEmbeddingBagsCodegen`
**模块路径**: `fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_training.py`
**Codegen目录**: `fbgemm_gpu/codegen/training/`

#### 7.2.1 前向传播算子(4个主要算子)

| 序号 | 算子名称 | CUDA Kernel | 功能描述 | 支持特性 | 开发工作量 |
|------|---------|------------|---------|---------|-----------|
| 1 | `torch.ops.fbgemm.dense_embedding_codegen_lookup_function` | `dense_embedding_forward_split_*_kernel` | 密集embedding查找主函数(训练入口) | - VBE(Variable Batch Embedding)<br>- 支持多rank<br>- SUM/MEAN/MAX pooling | 10天 |
| 2 | `torch.ops.fbgemm.generate_vbe_metadata` | `generate_vbe_metadata_cuda` | 生成Variable Batch Embedding元数据 | 用于不同特征的batch size不同场景 | 2天 |
| 3 | `torch.ops.fbgemm.get_infos_metadata` | `get_infos_metadata_cuda` | 获取VBE infos元数据 | - | 1天 |
| 4 | `torch.ops.fbgemm.transpose_embedding_input` | `transpose_embedding_input_cuda` | 转置embedding输入 | 用于优化数据布局 | 2天 |

**实现文件**:
- `fbgemm_gpu/codegen/training/forward/embedding_forward_split_template.cu`
- `fbgemm_gpu/codegen/training/forward/embedding_forward_split_kernel_template.cu`
- `fbgemm_gpu/codegen/training/forward/embedding_forward_split_kernel_v2_template.cu`
- `fbgemm_gpu/codegen/training/forward/embedding_forward_split_kernel_nobag_small_template.cu`

**Kernel分类**:
1. **按pooling模式**: SUM, MEAN, MAX, NONE
2. **按访问模式**:
   - Small kernel: 每行样本数≤32
   - Warp-per-row kernel: 每行样本数中等
   - CTA-per-row kernel: 每行样本数>1024
3. **按VBE**: 支持Variable Batch Embedding

#### 7.2.2 缓存管理算子(训练特有,7个)

| 序号 | 算子名称 | CUDA Kernel | 功能描述 | 训练特有功能 | 开发工作量 |
|------|---------|------------|---------|------------|-----------|
| 5 | `torch.ops.fbgemm.lru_cache_populate` | `lru_cache_populate_cuda` | LRU缓存填充(FP32/FP16) | 支持stochastic_rounding | 3天 |
| 6 | `torch.ops.fbgemm.lfu_cache_populate` | `lfu_cache_populate_cuda` | LFU缓存填充(FP32/FP16) | 支持stochastic_rounding | 3天 |
| 7 | `torch.ops.fbgemm.lxu_cache_flush` | `lxu_cache_flush_cuda` | 刷新缓存到UVM | 支持stochastic_rounding | 2天 |
| 8 | `torch.ops.fbgemm.lxu_cache_locking_counter_decrement` | `lxu_cache_locking_counter_decrement_cuda` | 减少缓存锁定计数器 | 用于异步更新pipeline | 1天 |
| 9 | `torch.ops.fbgemm.lxu_cache_locations_update` | `lxu_cache_locations_update_cuda` | 更新缓存位置信息 | 支持异步pipeline | 2天 |
| 10 | `torch.ops.fbgemm.linearize_cache_indices` | `linearize_cache_indices_cuda` | 线性化缓存索引 | 同推理 | 2天 |
| 11 | `torch.ops.fbgemm.lxu_cache_lookup` | `lxu_cache_lookup_cuda` | 缓存查找 | 同推理 | 3天 |

**训练特有的缓存优化**:
- Stochastic rounding: 随机舍入提高训练精度
- 异步更新pipeline: 支持计算和缓存更新并行
- 锁定计数器: 防止更新中的缓存被驱逐

#### 7.2.3 反向传播和优化器算子(融合设计)

训练模块采用**融合的backward+optimizer**策略,将梯度计算和优化器更新合并到单个kernel中,减少内存访问。

**支持的优化器类型**:

| 优化器类型 | 枚举值 | 功能描述 | Kernel模板 |
|-----------|-------|---------|-----------|
| SGD | `EXACT_SGD` | 随机梯度下降 | `embedding_optimizer_sgd_template.cu` |
| Adagrad | `EXACT_ADAGRAD` | 标准Adagrad | `embedding_optimizer_adagrad_template.cu` |
| Rowwise Adagrad | `EXACT_ROWWISE_ADAGRAD` | 行级Adagrad | 同上 |
| Partial Rowwise Adagrad | `PARTIAL_ROWWISE_ADAGRAD` | 部分行级Adagrad | 同上 |
| Adam | `ADAM` | Adam优化器 | `embedding_optimizer_adam_template.cu` |
| Partial Rowwise Adam | `PARTIAL_ROWWISE_ADAM` | 部分行级Adam | 同上 |
| LAMB | `LAMB` | Layer-wise Adaptive Rate Batch | `embedding_optimizer_lamb_template.cu` |
| LARS SGD | `LARS_SGD` | LARS SGD | `embedding_optimizer_lars_sgd_template.cu` |

**融合Kernel流程**:
```
输入: 前向输出, 梯度, 优化器状态
  ↓
1. 计算indice_weights梯度 (如有)
  ↓
2. 计算特征梯度
  ↓
3. 计算权重梯度
  ↓
4. 应用优化器更新 ← 融合点
  ↓
输出: 更新后的权重, 优化器状态
```

**实现文件**:
- `fbgemm_gpu/codegen/training/backward/embedding_backward_split_template.cu`
- `fbgemm_gpu/codegen/training/backward/embedding_backward_split_kernel_cta_template.cu`
- `fbgemm_gpu/codegen/training/backward/embedding_backward_split_kernel_warp_template.cu`
- `fbgemm_gpu/codegen/training/backward/embedding_backward_split_grad_template.cu`
- `fbgemm_gpu/codegen/training/backward/embedding_backward_split_indice_weights_template.cu`
- `fbgemm_gpu/codegen/training/optimizer/embedding_optimizer_split_template.cu`
- `fbgemm_gpu/codegen/training/optimizer/embedding_optimizer_split_kernel_template.cu`

**开发工作量**: 15天
- 梯度计算kernel: 5天
- 优化器kernel: 7天
- 融合优化: 3天

#### 7.2.4 索引处理和去重算子(5个)

| 序号 | 算子名称 | CUDA Kernel | 功能描述 | 用途 | 开发工作量 |
|------|---------|------------|---------|-----|-----------|
| 12 | `torch.ops.fbgemm.get_unique_indices` | `get_unique_indices_cuda` | 获取唯一索引并计数 | 用于prefetch优化 | 2天 |
| 13 | `torch.ops.fbgemm.get_unique_indices_with_inverse` | `get_unique_indices_with_inverse_cuda` | 获取唯一索引及逆映射 | 用于VBE和缓存 | 3天 |
| 14 | `torch.ops.fbgemm.asynchronous_complete_cumsum` | `asynchronous_complete_cumsum_cuda` | 异步完全前缀和 | 用于索引处理 | 2天 |
| 15 | `torch.ops.fbgemm.masked_index_select` | `masked_index_select_cuda` | 掩码索引选择 | 用于缓存更新 | 2天 |
| 16 | `torch.ops.fbgemm.reset_weight_momentum` | `reset_weight_momentum_cuda` | 重置权重和动量 | 用于embedding剪枝 | 1天 |

**实现文件**:
- `fbgemm_gpu/src/split_embeddings_cache/split_embeddings_cache_ops.cpp`
- `fbgemm_gpu/src/sparse_ops/sparse_async_cumsum.cpp`

#### 7.2.5 批处理索引选择算子(1个)

| 序号 | 算子名称 | CUDA Kernel | 功能描述 | 开发工作量 |
|------|---------|------------|---------|-----------|
| 17 | `batch_index_select_dim0_codegen_forward_function` | `batch_index_select_dim0_kernel` | 批量维度0索引选择 | 用于多表联合查找 | 3天 |

**实现文件**:
- `fbgemm_gpu/codegen/training/index_select/batch_index_select_dim0_ops.cpp`
- `fbgemm_gpu/codegen/training/index_select/batch_index_select_dim0_host.cpp`

#### 7.2.6 边界检查算子(训练特有,1个)

| 序号 | 算子名称 | CUDA Kernel | 功能描述 | 训练特有参数 | 开发工作量 |
|------|---------|------------|---------|------------|-----------|
| 18 | `torch.ops.fbgemm.bounds_check_indices` | `bounds_check_indices_v2_kernel` | 边界检查(支持VBE) | - `B_offsets`: VBE batch偏移<br>- `max_B`: 最大batch<br>- `b_t_map`: batch到rank映射 | 2天 |

#### 7.2.7 其他辅助算子(1个)

| 序号 | 算子名称 | CUDA Kernel | 功能描述 | 开发工作量 |
|------|---------|------------|---------|-----------|
| 19 | `torch.ops.fbgemm.emb_inplace_update` | `emb_inplace_update_cuda` | 原地更新embedding权重 | 同推理 | 3天 |

**训练模块总计**: 约19个PyTorch算子接口,背后有**上百个CUDA kernel变体**

### 7.3 TBE模块算子分类汇总

#### 7.3.1 按功能分类

| 功能类别 | 推理模块 | 训练模块 | 总计 |
|---------|---------|---------|------|
| **缓存管理** | 6个 | 7个 | 13个 |
| **前向查找** | 3个 | 4个 | 7个 |
| **反向传播** | - | 融合在前向/优化器 | - |
| **优化器** | - | 5类(8个kernel模板) | 8个 |
| **索引处理** | - | 5个 | 5个 |
| **边界检查** | 1个 | 1个 | 2个 |
| **内存管理** | 3个 | - | 3个 |
| **剪枝** | 2个 | - | 2个 |
| **辅助工具** | 2个 | 2个 | 4个 |
| **PyTorch算子总计** | **17个** | **19个** | **36个** |

#### 7.3.2 按实现复杂度分类的工作量

**推理模块** (30天/人):
- 缓存管理: 15天
- 前向查找: 8天
- 边界检查: 2天
- 内存管理: 2天
- 剪枝: 2天
- 工具: 1天

**训练模块** (45天/人):
- 前向传播: 10天
- 反向传播: 5天
- 优化器融合: 15天
- 缓存管理: 7天
- 索引处理: 5天
- 其他: 3天

### 7.4 TBE模块关键设计特点

#### 7.4.1 量化推理支持

**精度范围**: INT2 → INT4 → INT8 → FP8 → FP16 → FP32

**量化格式**:
- **行级量化**: 每行独立量化,适应稀疏特征分布
- **Fused格式**: 量化值 + scale + bias存储在一起,减少内存访问

**优势**:
- 模型大小减少: INT2比FP32减少16倍
- 内存带宽降低: 加载更少的数据
- 计算加速: INT8/INT4可使用Tensor Core

#### 7.4.2 缓存优化策略

**缓存算法**:
- **LRU (Least Recently Used)**: 适合时间局部性强的场景
- **LFU (Least Frequently Used)**: 适合频率局部性强的场景

**缓存架构**:
- **32/64路关联缓存**:
  - NVIDIA GPU: 32-way (适配warp size)
  - AMD GPU: 64-way (适配wavefront size)
- **直接映射缓存**: 1-way,简单高效

**缓存行大小**: 16字节对齐(128 bytes),优化内存访问

**训练优化**:
- Stochastic rounding: 随机舍入减少量化误差
- 异步pipeline: 计算和缓存更新并行

#### 7.4.3 融合优化策略

**Backward + Optimizer融合**:

传统实现:
```
forward → output
backward → grad
optimizer_step → update_weight
(3次kernel launch, 3次内存访问)
```

融合实现:
```
forward → output
backward_optimizer_fused → grad + update_weight
(1次kernel launch, 1次内存访问)
```

**优势**:
- 减少kernel launch开销
- 减少全局内存访问
- 提高数据局部性
- 降低寄存器压力

#### 7.4.4 多级Kernel设计

根据每行样本数选择最优kernel:

| 每行样本数 | Kernel类型 | 特点 |
|-----------|-----------|------|
| ≤32 | Small kernel | 单warp处理,低延迟 |
| 32~1024 | Warp-per-row | 每行一个warp,平衡 |
| >1024 | CTA-per-row | 每行一个CTA,高吞吐 |

**自适应选择**: 运行时根据数据特征自动选择最优kernel

#### 7.4.5 VBE (Variable Batch Embedding)

**背景**: 多rank训练中,不同特征的batch size可能不同

**实现**:
- `B_offsets`: 每个特征的batch偏移
- `max_B`: 最大batch size
- `b_t_map`: batch到rank的映射

**优势**:
- 支持异构batch size
- 优化通信开销
- 提高GPU利用率

#### 7.4.6 剪枝支持

**哈希表剪枝**:
- 开放寻址哈希表
- 支持稀疏索引映射到密集索引
- 适合高剪枝率场景

**数组剪枝**:
- 紧凑数组存储
- 查找O(1)时间复杂度
- 适合低剪枝率场景

**原地更新**:
- `emb_inplace_update`: 支持在线学习
- 增量更新embedding权重

### 7.5 TBE模块文件路径映射

#### 7.5.1 Python模块

| 模块 | 路径 |
|------|------|
| 推理模块 | `fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_inference.py` |
| 训练模块 | `fbgemm_gpu/fbgemm_gpu/split_table_batched_embeddings_ops_training.py` |

#### 7.5.2 Codegen生成目录

**推理模块** (`fbgemm_gpu/codegen/inference/`):
```
embedding_forward_quantized_split_lookup.cu
embedding_forward_quantized_split_nbit_kernel_template.cu
embedding_forward_quantized_split_nbit_host_template.cu
embedding_forward_quantized_host.cpp
embedding_forward_quantized_host_cpu.cpp
```

**训练模块** (`fbgemm_gpu/codegen/training/`):
```
forward/
  embedding_forward_split_template.cu
  embedding_forward_split_kernel_template.cu
  embedding_forward_split_kernel_v2_template.cu
  embedding_forward_split_kernel_nobag_small_template.cu
  embedding_forward_split_cpu.cpp

backward/
  embedding_backward_split_template.cu
  embedding_backward_split_kernel_cta_template.cu
  embedding_backward_split_kernel_warp_template.cu
  embedding_backward_split_grad_template.cu
  embedding_backward_split_indice_weights_template.cu
  embedding_backward_split_host_template.cpp
  embedding_backward_split_cpu_template.cpp
  embedding_backward_split_cpu_approx_template.cpp
  embedding_backward_split_meta_template.cpp

optimizer/
  embedding_optimizer_split_template.cu
  embedding_optimizer_split_kernel_template.cu
  embedding_optimizer_split_host_template.cpp

index_select/
  batch_index_select_dim0_ops.cpp
  batch_index_select_dim0_host.cpp
  batch_index_select_dim0_cpu_host.cpp
```

#### 7.5.3 算子注册和实现

**缓存操作**:
- `fbgemm_gpu/src/split_embeddings_cache/split_embeddings_cache_ops.cpp`
- `fbgemm_gpu/src/split_embeddings_cache/split_embeddings_cache_ops.cu`

**工具函数**:
- `fbgemm_gpu/src/split_embeddings_utils/split_embeddings_utils.cpp`
- `fbgemm_gpu/src/split_embeddings_utils/split_embeddings_utils_cpu.cpp`

**内存管理**:
- `fbgemm_gpu/src/cumem_utils/cumem_utils.cpp`
- `fbgemm_gpu/src/cumem_utils/memory_utils_ops.cpp`

**原地更新**:
- `fbgemm_gpu/src/embedding_inplace_update/embedding_inplace_update_gpu.cpp`
- `fbgemm_gpu/src/embedding_inplace_update/embedding_inplace_update_cpu.cpp`

**边界检查**:
- `fbgemm_gpu/codegen/utils/embedding_bounds_check_v1.cu`
- `fbgemm_gpu/codegen/utils/embedding_bounds_check_v2.cu`
- `fbgemm_gpu/codegen/utils/embedding_bounds_check_host.cpp`

---

## 附录

### A. 文档参考

- FBGEMM_GPU文档根目录: [fbgemm_gpu/docs/src/fbgemm_gpu/index.rst](fbgemm_gpu/docs/src/fbgemm_gpu/index.rst)
- Stable Python API: [stable-api/python_api.rst](fbgemm_gpu/docs/src/fbgemm_gpu/stable-api/python_api.rst)
- C++ API文档: [cpp-api/](fbgemm_gpu/docs/src/fbgemm_gpu/cpp-api/)
- Python API文档: [python-api/](fbgemm_gpu/docs/src/fbgemm_gpu/python-api/)

### B. 版本信息

- **代码库分支**: v1.5.0-release
- **分析日期**: 2026-01-31
- **文档版本**: 1.0

### C. 许可证

Copyright (C) 2026 Huawei Technologies Co.,Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
