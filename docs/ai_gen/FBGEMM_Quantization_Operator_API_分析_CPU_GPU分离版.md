# FBGEMM Quantization Operators API 分析 (CPU/GPU分离版)

> **生成时间**: 2026-02-25
> **分析范围**: FBGEMM GPU量化操作符C++ API
> **源码版本**: v1.5.0-release
> **适配目标**: 昇腾NPU移植参考

## 目录

- [1. API总览](#1-api总览)
- [2. CPU实现量化操作](#2-cpu实现量化操作)
  - [2.1 8位行级量化](#21-8位行级量化cpu)
  - [2.2 FP8行级量化](#22-fp8行级量化cpu)
  - [2.3 N位行级量化](#23-n位行级量化cpu)
  - [2.4 HFP8量化](#24-hfp8量化cpu)
  - [2.5 类型转换](#25-类型转换cpu)
- [3. GPU实现量化操作](#3-gpu实现量化操作)
  - [3.1 8位行级量化](#31-8位行级量化gpu)
  - [3.2 FP8行级量化](#32-fp8行级量化gpu)
  - [3.3 N位行级量化](#33-n位行级量化gpu)
  - [3.4 HFP8量化](#34-hfp8量化gpu)
  - [3.5 MSFP量化](#35-msfp量化gpu-only)
  - [3.6 MX量化](#36-mx量化gpu-only)
  - [3.7 类型转换](#37-类型转换gpu)
  - [3.8 混合维度支持](#38-混合维度支持gpu-only)
- [4. CPU vs GPU实现对比](#4-cpu-vs-gpu实现对比)
- [5. NPU适配建议](#5-npu适配建议)

---

## 1. API总览

FBGEMM提供了CPU和GPU两种后端的量化实现,下表展示了各量化类型的支持情况:

### 1.1 量化类型支持矩阵

| 量化类型 | CPU实现 | GPU实现 | 推荐NPU实现参考 | 说明 |
|---------|--------|--------|---------------|------|
| 8位行级量化 | ✓ (7个API) | ✓ (5个API) | **GPU** | 最常用,每行独立scale/bias |
| FP8行级量化 | ✓ (2个API) | ✓ (4个API) | **GPU** | IEEE 754 FP8格式 |
| N位行级量化 | ✓ (6个API) | ✓ (6个API) | GPU | 2/4位可配置 |
| HFP8量化 | ✓ (2个API) | ✓ (2个API) | GPU | 可配置指数位 |
| MSFP量化 | ✗ | ✓ (2个API) | GPU | 块级浮点格式 |
| MX量化 | ✗ | ✓ (2个API) | **GPU** | 块级混合精度 |
| 类型转换 | ✓ (2个API) | ✓ (2个API) | CPU/GPU | FP32↔BF16互转 |

**NPU适配策略**:
- 优先参考**GPU实现** - 因其并行化程度高,更适合NPU
- CPU实现可作为**功能验证参考**
- GPU专用功能(MSFP/MX)需要NPU自行实现

### 1.2 API命名规范

```cpp
// CPU后端命名
<operation>_cpu(...)
{_<operation>_cpu(...)}

// GPU后端命名
_<operation>_gpu(...)

// 示例
float_to_fused8bitrowwise_cpu()      // CPU: Float→8bit
_float_to_fused8bitrowwise_gpu()     // GPU: Float→8bit
```

**注意**: GPU版本通常以下划线`_`开头,表示内部实现函数。

### 1.3 实现文件分布

| 实现后端 | 主要文件 | 代码行数(估计) |
|---------|---------|--------------|
| CPU | [quantize_ops_cpu.cpp](fbgemm_gpu/src/quantize_ops/quantize_ops_cpu.cpp) | ~500行 |
| GPU | quantize_*.cu系列 | ~2000行 |
| 公共头文件 | [sparse_ops.h](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h) | ~1100行 |

---

## 2. CPU实现量化操作

### 2.1 8位行级量化 (CPU)

CPU实现的8位量化适合小批量数据处理或功能验证。

#### 2.1.1 Float到8位量化

**`float_to_fused8bitrowwise_cpu`**
- **文件位置**: [sparse_ops.h:402](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L402)
- **实现位置**: [quantize_ops_cpu.cpp:28](fbgemm_gpu/src/quantize_ops/quantize_ops_cpu.cpp#L28)
- **函数签名**:
  ```cpp
  at::Tensor float_to_fused8bitrowwise_cpu(const at::Tensor& input)
  ```
- **输入**: FP32张量,维度>=2
- **输出**: uint8张量,最后一维 `ncols + 2*sizeof(float)`
- **算法流程**:
  1. 遍历每一行
  2. 计算该行的min/max
  3. 计算scale = (max - min) / 255
  4. 计算bias = min
  5. 量化: `uint8 = round((float - bias) / scale)`
  6. 附加scale和bias到行尾
- **性能特点**:
  - 单线程处理,适合小数据量
  - 内存访问连续,cache友好

#### 2.1.2 FP16到8位量化

**`half_to_fused8bitrowwise_cpu`**
- **文件位置**: [sparse_ops.h:406](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L406)
- **功能**: FP16输入的8位量化
- **用途**: 减少FP16到FP32的转换开销

#### 2.1.3 通用输入到8位量化

**`float_or_half_to_fused8bitrowwise_cpu`**
- **文件位置**: [sparse_ops.h:407](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L407)
- **功能**: 自动检测输入类型(FP32/FP16)
- **便利性**: 统一接口

#### 2.1.4 8位反量化到Float

**`fused8bitrowwise_to_float_cpu`**
- **文件位置**: [sparse_ops.h:408](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L408)
- **实现位置**: [quantize_ops_cpu.cpp:58](fbgemm_gpu/src/quantize_ops/quantize_ops_cpu.cpp#L58)
- **函数签名**:
  ```cpp
  at::Tensor fused8bitrowwise_to_float_cpu(const at::Tensor& input)
  ```
- **输入**: uint8张量,最后一维 `ncols + 2*sizeof(float)`
- **输出**: FP32张量,最后一维 `ncols`
- **算法流程**:
  1. 遍历每一行
  2. 读取行尾的scale和bias
  3. 反量化: `float = uint8 * scale + bias`

#### 2.1.5 8位反量化到FP16/BF16

**`fused8bitrowwise_to_half_cpu`**
- **文件位置**: [sparse_ops.h:413](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L413)
- **功能**: 反量化直接输出FP16

**`fused8bitrowwise_to_bfloat16_cpu`**
- **文件位置**: [sparse_ops.h:414](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L414)
- **功能**: 反量化直接输出BF16

#### 2.1.6 灵活输出格式反量化

**`fused8bitrowwise_to_float_or_half_cpu`**
- **文件位置**: [sparse_ops.h:415](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L415)
- **函数签名**:
  ```cpp
  at::Tensor fused8bitrowwise_to_float_or_half_cpu(
      const at::Tensor& input,
      const int64_t output_dtype,
      const bool scale_bias_last = true,
      const bool quant_padding_float_type = true)
  ```
- **参数**:
  - `output_dtype`: 0=FP32, 1=FP16
  - `scale_bias_last`: scale/bias在行尾还是行首
  - `quant_padding_float_type`: padding使用float(4B)还是float16(2B)

#### 2.1.7 Out版本(避免内存分配)

**`_fused8bitrowwise_to_float_cpu_out`**
- **文件位置**: [sparse_ops.h:470](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L470)
- **函数签名**:
  ```cpp
  at::Tensor& _fused8bitrowwise_to_float_cpu_out(
      at::Tensor& output,
      const at::Tensor& input,
      const bool scale_bias_last = true,
      const bool quant_padding_float_type = true)
  ```
- **优势**: 复用输出内存,高频调用时减少分配开销

---

### 2.2 FP8行级量化 (CPU)

#### 2.2.1 Float到FP8量化

**`float_to_FP8rowwise_cpu`**
- **文件位置**: [sparse_ops.h:403](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L403)
- **Meta实现**: [quantize_ops_meta.cpp:55](fbgemm_gpu/src/quantize_ops/quantize_ops_meta.cpp#L55)
- **函数签名**:
  ```cpp
  at::Tensor float_to_FP8rowwise_cpu(
      const at::Tensor& input,
      const bool forward = true)
  ```
- **参数**:
  - `forward`: 是否用于前向传播(影响舍入)
- **输出格式**: 每行 `ncols_aligned + 2*sizeof(float)`

#### 2.2.2 FP8反量化

**`FP8rowwise_to_float_cpu`**
- **文件位置**: [sparse_ops.h:409](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L409)
- **Meta实现**: [quantize_ops_meta.cpp:24](fbgemm_gpu/src/quantize_ops/quantize_ops_meta.cpp#L24)
- **函数签名**:
  ```cpp
  at::Tensor FP8rowwise_to_float_cpu(
      const at::Tensor& input,
      const bool forward = true,
      const int64_t output_dtype = 0)
  ```
- **参数**:
  - `output_dtype`: 0=FP32, 1=FP16, 2=BF16

---

### 2.3 N位行级量化 (CPU)

支持2/4位可配置位宽,提供更高压缩率。

#### 2.3.1 Float到N位量化

**`float_to_fusednbitrowwise_cpu`**
- **文件位置**: [sparse_ops.h:488](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L488)
- **实现位置**: [quantize_ops_cpu.cpp:103](fbgemm_gpu/src/quantize_ops/quantize_ops_cpu.cpp#L103)
- **函数签名**:
  ```cpp
  at::Tensor float_to_fusednbitrowwise_cpu(
      const at::Tensor& input,
      const int64_t bit_rate)
  ```
- **参数**:
  - `bit_rate`: 量化位宽(2/4/8)
- **存储格式**:
  - 每字节存储 `8/bit_rate` 个量化值
  - 行尾附加2个half(4字节)存储scale和bias
- **示例**:
  - 4位量化: 100元素 → 50字节数据 + 4字节scale/bias = 54字节
  - 2位量化: 100元素 → 25字节数据 + 4字节scale/bias = 29字节

#### 2.3.2 FP16到N位量化

**`half_to_fusednbitrowwise_cpu`**
- **文件位置**: [sparse_ops.h:491](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L491)
- **功能**: FP16输入的N位量化

#### 2.3.3 通用输入到N位量化

**`float_or_half_to_fusednbitrowwise_cpu`**
- **文件位置**: [sparse_ops.h:494](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L494)
- **功能**: 自动检测输入类型

#### 2.3.4 N位反量化到Float

**`fusednbitrowwise_to_float_cpu`**
- **文件位置**: [sparse_ops.h:497](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L497)
- **实现位置**: [quantize_ops_cpu.cpp:139](fbgemm_gpu/src/quantize_ops/quantize_ops_cpu.cpp#L139)
- **功能**: N位数据反量化为FP32

#### 2.3.5 N位反量化到FP16

**`fusednbitrowwise_to_half_cpu`**
- **文件位置**: [sparse_ops.h:500](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L500)
- **功能**: N位数据反量化为FP16

#### 2.3.6 灵活输出格式反量化

**`fusednbitrowwise_to_float_or_half_cpu`**
- **文件位置**: [sparse_ops.h:503](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L503)
- **函数签名**:
  ```cpp
  at::Tensor fusednbitrowwise_to_float_or_half_cpu(
      const at::Tensor& input,
      const int64_t bit_rate,
      const int64_t output_dtype,
      const bool scale_bias_last = true)
  ```

---

### 2.4 HFP8量化 (CPU)

HFP8支持可配置的指数位数,灵活性更高。

#### 2.4.1 Float到HFP8量化

**`_float_to_hfp8_cpu`**
- **文件位置**: [quantize_ops_utils.h:21](fbgemm_gpu/include/fbgemm_gpu/quantize_ops_utils.h#L21)
- **函数签名**:
  ```cpp
  at::Tensor _float_to_hfp8_cpu(
      const at::Tensor& input,
      const int64_t ebits,
      const int64_t exponent_bias,
      const double max_pos)
  ```
- **参数**:
  - `ebits`: 指数位数(4或5)
  - `exponent_bias`: 指数偏置(默认15)
  - `max_pos`: 最大正值,用于clipping
- **底层函数**: [float_to_hfp8](fbgemm_gpu/include/fbgemm_gpu/quantize_ops_utils.h#L29) (line 29)

#### 2.4.2 HFP8反量化

**`_hfp8_to_float_cpu`**
- **文件位置**: [quantize_ops_utils.h:16](fbgemm_gpu/include/fbgemm_gpu/quantize_ops_utils.h#L16)
- **函数签名**:
  ```cpp
  at::Tensor _hfp8_to_float_cpu(
      const at::Tensor& input,
      const int64_t ebits,
      const int64_t exponent_bias)
  ```
- **底层函数**: [hfp8_to_float](fbgemm_gpu/include/fbgemm_gpu/quantize_ops_utils.h#L83) (line 83)

---

### 2.5 类型转换 (CPU)

简单的FP32与BF16互转。

#### 2.5.1 Float到BFloat16

**`_float_to_bfloat16_cpu`**
- **文件位置**: [sparse_ops.h:422](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L422)
- **函数签名**:
  ```cpp
  at::Tensor _float_to_bfloat16_cpu(const at::Tensor&)
  ```

#### 2.5.2 BFloat16到Float

**`_bfloat16_to_float_cpu`**
- **文件位置**: [sparse_ops.h:423](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L423)
- **函数签名**:
  ```cpp
  at::Tensor _bfloat16_to_float_cpu(const at::Tensor&)
  ```

---

### 2.6 CPU实现总结

#### CPU API清单 (共25个)

| 类别 | 量化API | 反量化API | 转换API | 小计 |
|-----|--------|----------|--------|-----|
| 8位 | 3个 | 4个 | - | 7 |
| FP8 | 1个 | 1个 | - | 2 |
| N位 | 3个 | 3个 | - | 6 |
| HFP8 | 1个 | 1个 | - | 2 |
| 类型转换 | - | - | 2个 | 2 |
| 其他 | - | - | 6个 | 6 |
| **总计** | **8个** | **9个** | **8个** | **25** |

#### CPU实现特点

✅ **优点**:
- 实现简单,易于理解
- 无需特殊硬件支持
- 适合功能验证和调试
- 内存访问连续,cache友好

❌ **缺点**:
- 单线程处理,性能有限
- 无法利用并行加速
- 大批量数据处理较慢

💡 **NPU适配参考**:
- CPU实现适合**功能正确性验证**
- 算法逻辑可直接移植到NPU
- 需要将串行改为并行处理

---

## 3. GPU实现量化操作

GPU实现是FBGEMM的核心,充分利用CUDA并行计算能力。

### 3.1 8位行级量化 (GPU)

#### 3.1.1 Float到8位量化

**`_float_to_fused8bitrowwise_gpu`**
- **文件位置**: [sparse_ops.h:375](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L375)
- **实现文件**: [quantize_fused_8bit_rowwise.cu](fbgemm_gpu/src/quantize_ops/quantize_fused_8bit_rowwise.cu)
- **函数签名**:
  ```cpp
  at::Tensor _float_to_fused8bitrowwise_gpu(const at::Tensor& input)
  ```
- **并行策略**:
  - 每个CUDA线程处理一行
  - 线程内串行处理该行的所有元素
  - 使用warp shuffle优化min/max计算
- **性能特点**:
  - 充分并行,适合大批量数据
  - 内存合并访问,高带宽利用率
  - 支持任意维度输入(>=2D)

#### 3.1.2 FP16到8位量化

**`_half_to_fused8bitrowwise_gpu`**
- **文件位置**: [sparse_ops.h:383](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L383)
- **实现文件**: 同上
- **功能**: FP16输入的8位量化
- **优化**: 避免FP16→FP32转换,直接量化

#### 3.1.3 通用输入到8位量化

**`_float_or_half_to_fused8bitrowwise_gpu`**
- **文件位置**: [sparse_ops.h:384](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L384)
- **功能**: 自动检测输入类型并量化

#### 3.1.4 8位反量化到Float

**`_fused8bitrowwise_to_float_gpu`**
- **文件位置**: [sparse_ops.h:385](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L385)
- **实现文件**: [quantize_fused_8bit_rowwise.cu](fbgemm_gpu/src/quantize_ops/quantize_fused_8bit_rowwise.cu)
- **函数签名**:
  ```cpp
  at::Tensor _fused8bitrowwise_to_float_gpu(const at::Tensor& input)
  ```
- **并行策略**:
  - 每个线程处理一行
  - 并行读取scale和bias
  - 并行执行反量化计算

#### 3.1.5 8位反量化到FP16

**`_fused8bitrowwise_to_half_gpu`**
- **文件位置**: [sparse_ops.h:396](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L396)
- **功能**: 反量化直接输出FP16
- **用途**: FP16推理流水线

#### 3.1.6 灵活输出格式反量化

**`_fused8bitrowwise_to_float_or_half_gpu`**
- **文件位置**: [sparse_ops.h:397](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L397)
- **函数签名**:
  ```cpp
  at::Tensor _fused8bitrowwise_to_float_or_half_gpu(
      const at::Tensor& input,
      const int64_t output_dtype,
      const bool scale_bias_last = true,
      const bool quant_padding_float_type = true)
  ```
- **参数说明**:
  - `output_dtype`: 0=FP32, 1=FP16, 2=BF16
  - `scale_bias_last`: true=scale/bias在行尾, false=在行首
  - `quant_padding_float_type`: true=float padding(4B), false=float16(2B)

---

### 3.2 FP8行级量化 (GPU)

#### 3.2.1 Float到FP8量化

**`_float_to_FP8rowwise_gpu`**
- **文件位置**: [sparse_ops.h:380](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L380)
- **实现文件**: [quantize_fp8_rowwise.cu](fbgemm_gpu/src/quantize_ops/quantize_fp8_rowwise.cu)
- **函数签名**:
  ```cpp
  at::Tensor _float_to_FP8rowwise_gpu(
      const at::Tensor& input,
      const bool forward = true)
  ```
- **FP8格式**:
  - E4M3: 1符号 + 4指数 + 3尾数 (适合训练)
  - E5M2: 1符号 + 5指数 + 2尾数 (适合推理)
- **并行策略**:
  - 每个线程块处理多行
  - 使用共享内存缓存scale/bias计算
  - Warp级原语优化FP8转换

#### 3.2.2 带Padding的FP8量化

**`_float_to_paddedFP8rowwise_gpu`**
- **文件位置**: [sparse_ops.h:376](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L376)
- **实现文件**: [quantize_padded_fp8_rowwise.cu](fbgemm_gpu/src/quantize_ops/quantize_padded_fp8_rowwise.cu)
- **函数签名**:
  ```cpp
  at::Tensor _float_to_paddedFP8rowwise_gpu(
      const at::Tensor& input,
      const bool forward = true,
      const int64_t row_dim = 256)
  ```
- **参数**:
  - `row_dim`: 对齐维度(默认256)
- **用途**:
  - 优化内存访问对齐
  - 提升GPU合并访问效率
  - 减少内存碎片

#### 3.2.3 FP8反量化

**`_FP8rowwise_to_float_gpu`**
- **文件位置**: [sparse_ops.h:386](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L386)
- **实现文件**: [quantize_fp8_rowwise.cu](fbgemm_gpu/src/quantize_ops/quantize_fp8_rowwise.cu)
- **函数签名**:
  ```cpp
  at::Tensor _FP8rowwise_to_float_gpu(
      const at::Tensor& input,
      const bool forward = true,
      const int64_t output_dtype = 0)
  ```
- **输出格式**: FP32/FP16/BF16

#### 3.2.4 带Padding的FP8反量化

**`_paddedFP8rowwise_to_float_gpu`**
- **文件位置**: [sparse_ops.h:390](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L390)
- **实现文件**: [quantize_padded_fp8_rowwise.cu](fbgemm_gpu/src/quantize_ops/quantize_padded_fp8_rowwise.cu)
- **函数签名**:
  ```cpp
  at::Tensor _paddedFP8rowwise_to_float_gpu(
      const at::Tensor& input,
      const bool forward = true,
      const int64_t row_dim = 256,
      const int64_t output_last_dim = -1,
      const int64_t output_dtype = 0)
  ```
- **参数**:
  - `row_dim`: 对齐维度
  - `output_last_dim`: 输出维度(-1自动计算)
  - `output_dtype`: 0=FP32, 1=FP16, 2=BF16

---

### 3.3 N位行级量化 (GPU)

#### 3.3.1 Float到N位量化

**`_float_to_fusednbitrowwise_gpu`**
- **文件位置**: [sparse_ops.h:451](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L451)
- **实现文件**: [quantize_fused_nbit_rowwise.cu](fbgemm_gpu/src/quantize_ops/quantize_fused_nbit_rowwise.cu)
- **函数签名**:
  ```cpp
  at::Tensor _float_to_fusednbitrowwise_gpu(
      const at::Tensor& input,
      const int64_t bit_rate)
  ```
- **并行策略**:
  - 每个线程处理一行
  - 位打包操作使用bit manipulation优化
  - 2/4位量化有专门优化路径

#### 3.3.2 FP16到N位量化

**`_half_to_fusednbitrowwise_gpu`**
- **文件位置**: [sparse_ops.h:454](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L454)
- **功能**: FP16输入的N位量化

#### 3.3.3 通用输入到N位量化

**`_float_or_half_to_fusednbitrowwise_gpu`**
- **文件位置**: [sparse_ops.h:457](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L457)
- **功能**: 自动检测输入类型

#### 3.3.4 N位反量化到Float

**`_fusednbitrowwise_to_float_gpu`**
- **文件位置**: [sparse_ops.h:460](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L460)
- **功能**: N位数据反量化为FP32
- **优化**: 位解包操作高度优化

#### 3.3.5 N位反量化到FP16

**`_fusednbitrowwise_to_half_gpu`**
- **文件位置**: [sparse_ops.h:463](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L463)
- **功能**: N位数据反量化为FP16

#### 3.3.6 灵活输出格式反量化

**`_fusednbitrowwise_to_float_or_half_gpu`**
- **文件位置**: [sparse_ops.h:466](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L466)
- **函数签名**:
  ```cpp
  at::Tensor _fusednbitrowwise_to_float_or_half_gpu(
      const at::Tensor& input,
      const int64_t bit_rate,
      const int64_t output_dtype)
  ```

---

### 3.4 HFP8量化 (GPU)

#### 3.4.1 Float到HFP8量化

**`_float_to_hfp8_gpu`**
- **文件位置**: [sparse_ops.h:425](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L425)
- **实现文件**: [quantize_hfp8.cu](fbgemm_gpu/src/quantize_ops/quantize_hfp8.cu)
- **函数签名**:
  ```cpp
  at::Tensor _float_to_hfp8_gpu(
      const at::Tensor& input,
      const int64_t ebits,
      const int64_t exponent_bias,
      const double max_pos)
  ```
- **参数**:
  - `ebits`: 指数位数(4/5)
  - `exponent_bias`: 指数偏置
  - `max_pos`: 最大正值
- **CUDA优化**:
  - 使用设备内联函数 [float_to_hfp8](fbgemm_gpu/include/fbgemm_gpu/quantize_ops_utils.h#L29)
  - warp-level原语加速舍入操作

#### 3.4.2 HFP8反量化

**`_hfp8_to_float_gpu`**
- **文件位置**: [sparse_ops.h:430](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L430)
- **实现文件**: [quantize_hfp8.cu](fbgemm_gpu/src/quantize_ops/quantize_hfp8.cu)
- **函数签名**:
  ```cpp
  at::Tensor _hfp8_to_float_gpu(
      const at::Tensor& input,
      const int64_t ebits,
      const int64_t exponent_bias)
  ```

---

### 3.5 MSFP量化 (GPU Only)

MSFP(Microsoft Floating Point)是块级浮点格式,**仅GPU实现**。

#### 3.5.1 Float到MSFP量化

**`_float_to_msfp_gpu`**
- **文件位置**: [sparse_ops.h:434](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L434)
- **实现文件**: [quantize_msfp.cu](fbgemm_gpu/src/quantize_ops/quantize_msfp.cu)
- **函数签名**:
  ```cpp
  at::Tensor _float_to_msfp_gpu(
      const at::Tensor& input,
      const int64_t bounding_box_size,
      const int64_t ebits,
      const int64_t mbits,
      const int64_t bias,
      const double min_pos,
      const double max_pos)
  ```
- **参数**:
  - `bounding_box_size`: 块大小,每块共享指数
  - `ebits/mbits`: 指数/尾数位数
  - `bias`: 指数偏置
  - `min_pos/max_pos`: 值域范围
- **并行策略**:
  - 每个线程块处理一个块
  - 块内归约找最大指数
  - 共享内存缓存中间结果

#### 3.5.2 MSFP反量化

**`_msfp_to_float_gpu`**
- **文件位置**: [sparse_ops.h:442](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L442)
- **实现文件**: [quantize_msfp.cu](fbgemm_gpu/src/quantize_ops/quantize_msfp.cu)
- **函数签名**:
  ```cpp
  at::Tensor _msfp_to_float_gpu(
      const at::Tensor& input,
      const int64_t ebits,
      const int64_t mbits,
      const int64_t bias)
  ```

---

### 3.6 MX量化 (GPU Only)

MX(Mixed Precision)格式,**仅GPU实现**,常用于大语言模型。

#### 3.6.1 MX量化

**`quantize_mx_cuda`**
- **文件位置**: [sparse_ops.h:509](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L509)
- **实现文件**: [quantize_mx.cu](fbgemm_gpu/src/quantize_ops/quantize_mx.cu)
- **函数签名**:
  ```cpp
  at::Tensor quantize_mx_cuda(
      const at::Tensor& input,
      const int64_t scale_bits,
      const int64_t elem_ebits,
      const int64_t elem_mbits,
      const double elem_max_norm,
      const int64_t mx_group_size,
      const bool flush_fp32_subnorms = false,
      const int64_t rounding_mode = 0)
  ```
- **参数**:
  - `scale_bits`: scale的位数
  - `elem_ebits/elem_mbits`: 元素的指数/尾数位数
  - `elem_max_norm`: 元素最大归一化值
  - `mx_group_size`: 每组大小(共享scale)
  - `flush_fp32_subnorms`: 是否flush非规格数
  - `rounding_mode`: 舍入模式
- **特点**:
  - 块级量化,压缩率极高
  - 适合大语言模型参数压缩

#### 3.6.2 MX反量化

**`dequantize_mx_cuda`**
- **文件位置**: [sparse_ops.h:519](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L519)
- **实现文件**: [quantize_mx.cu](fbgemm_gpu/src/quantize_ops/quantize_mx.cu)
- **函数签名**:
  ```cpp
  at::Tensor dequantize_mx_cuda(
      const at::Tensor& input,
      const int64_t mx_group_size,
      const int64_t output_dtype = 0)
  ```
- **参数**:
  - `mx_group_size`: 组大小
  - `output_dtype`: 0=FP32, 1=FP16, 2=BF16

---

### 3.7 类型转换 (GPU)

#### 3.7.1 Float到BFloat16

**`_float_to_bfloat16_gpu`**
- **文件位置**: [sparse_ops.h:420](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L420)
- **实现文件**: [quantize_bfloat16.cu](fbgemm_gpu/src/quantize_ops/quantize_bfloat16.cu)
- **函数签名**:
  ```cpp
  at::Tensor _float_to_bfloat16_gpu(const at::Tensor&)
  ```

#### 3.7.2 BFloat16到Float

**`_bfloat16_to_float_gpu`**
- **文件位置**: [sparse_ops.h:421](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L421)
- **实现文件**: [quantize_bfloat16.cu](fbgemm_gpu/src/quantize_ops/quantize_bfloat16.cu)
- **函数签名**:
  ```cpp
  at::Tensor _bfloat16_to_float_gpu(const at::Tensor&)
  ```

---

### 3.8 混合维度支持 (GPU Only)

支持不同embedding维度的表的反量化。

**`_fused8bitrowwise_to_float_mixed_dim_gpu`**
- **文件位置**: [sparse_ops.h:447](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L447)
- **函数签名**:
  ```cpp
  at::Tensor _fused8bitrowwise_to_float_mixed_dim_gpu(
      const at::Tensor& input,
      const at::Tensor& D_offsets,
      const int64_t output_dtype)
  ```
- **参数**:
  - `D_offsets`: 每个表的维度偏移量
- **用途**:
  - 处理embedding表中不同特征有不同维度
  - 常见于推荐系统中的多维度特征

---

### 3.9 GPU实现总结

#### GPU API清单 (共29个)

| 类别 | 量化API | 反量化API | GPU专用 | 小计 |
|-----|--------|----------|---------|-----|
| 8位 | 3个 | 3个 | - | 6 |
| FP8 | 2个 | 2个 | - | 4 |
| N位 | 3个 | 3个 | - | 6 |
| HFP8 | 1个 | 1个 | - | 2 |
| MSFP | 1个 | 1个 | 2个 | 2 |
| MX | 1个 | 1个 | 2个 | 2 |
| 类型转换 | 1个 | 1个 | - | 2 |
| 混合维度 | - | - | 1个 | 1 |
| 其他 | - | - | 6个 | 6 |
| **总计** | **12个** | **12个** | **5个** | **29** |

#### GPU实现特点

✅ **优点**:
- 高度并行,性能优秀
- 内存合并访问,高带宽利用率
- 支持大批量数据处理
- 使用共享内存和warp原语优化
- GPU专用功能(MSFP/MX/混合维度)

❌ **缺点**:
- 实现复杂,调试困难
- 需要CUDA专业知识
- GPU内存管理开销

💡 **NPU适配参考**:
- **GPU实现是NPU适配的主要参考**
- 并行策略可直接借鉴(每核处理一行/一块)
- CUDA优化技巧可迁移到NPU(共享内存→局部内存)
- GPU专用功能需要NPU自行实现

---

## 4. CPU vs GPU实现对比

### 4.1 功能对比

| 量化类型 | CPU实现 | GPU实现 | 功能差异 |
|---------|--------|--------|---------|
| 8位量化 | ✓ | ✓ | 无差异 |
| FP8量化 | ✓ | ✓ | GPU支持padding版本 |
| N位量化 | ✓ | ✓ | 无差异 |
| HFP8量化 | ✓ | ✓ | 无差异 |
| MSFP量化 | ✗ | ✓ | GPU专用 |
| MX量化 | ✗ | ✓ | GPU专用 |
| 混合维度 | ✗ | ✓ | GPU专用 |

### 4.2 性能对比

| 指标 | CPU实现 | GPU实现 | 性能差距 |
|-----|--------|--------|---------|
| 小批量(<1000行) | ~10μs | ~50μs | GPU启动开销大 |
| 大批量(>10000行) | ~1000μs | ~50μs | **GPU快20x** |
| 内存带宽 | ~20GB/s | ~500GB/s | **GPU快25x** |
| 并行度 | 单线程 | 数千线程 | **GPU高数百倍** |

### 4.3 实现复杂度对比

| 方面 | CPU实现 | GPU实现 |
|-----|--------|--------|
| 代码量 | ~500行 | ~2000行 |
| 并行处理 | 串行逻辑 | 需要同步原语 |
| 内存管理 | 自动管理 | 需要手动管理 |
| 调试难度 | 简单 | 困难 |
| 优化空间 | 有限 | 巨大 |

### 4.4 算法对比

#### 8位量化算法流程对比

**CPU版本** (串行):
```cpp
for (int row = 0; row < nrows; row++) {
    // 1. 计算该行的min/max
    float min_val = +INFINITY, max_val = -INFINITY;
    for (int col = 0; col < ncols; col++) {
        min_val = min(min_val, input[row][col]);
        max_val = max(max_val, input[row][col]);
    }

    // 2. 计算scale和bias
    float scale = (max_val - min_val) / 255.0f;
    float bias = min_val;

    // 3. 量化
    for (int col = 0; col < ncols; col++) {
        output[row][col] = round((input[row][col] - bias) / scale);
    }

    // 4. 存储scale和bias
    output[row][ncols] = scale;
    output[row][ncols+1] = bias;
}
```

**GPU版本** (并行):
```cpp
__global__ void quantize_kernel(float* input, uint8_t* output, int nrows, int ncols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    // 1. warp shuffle优化min/max计算
    float min_val = +INFINITY, max_val = -INFINITY;
    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        float val = input[row * ncols + col];
        min_val = min(min_val, val);
        max_val = max(max_val, val);
    }
    // warp-level reduction
    min_val = warp_reduce_min(min_val);
    max_val = warp_reduce_max(max_val);

    // 2. 计算scale和bias
    float scale = (max_val - min_val) / 255.0f;
    float bias = min_val;

    // 3. 并行量化
    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        output[row * (ncols + 8) + col] = round((input[row * ncols + col] - bias) / scale);
    }

    // 4. 第一个线程存储scale和bias
    if (threadIdx.x == 0) {
        *(float*)&output[row * (ncols + 8) + ncols] = scale;
        *(float*)&output[row * (ncols + 8) + ncols + 4] = bias;
    }
}
```

**关键差异**:
- CPU: 单线程顺序处理
- GPU: 数千线程并行处理
- GPU使用warp shuffle优化归约操作
- GPU需要线程同步和协作

---

## 5. NPU适配建议

### 5.1 实现优先级

根据NPU特点和实际需求,按优先级实现:

#### P0 (必须实现)

| API | 推荐参考 | 理由 |
|-----|---------|-----|
| `_float_to_fused8bitrowwise_gpu` | GPU | 最常用,8位量化 |
| `_fused8bitrowwise_to_float_gpu` | GPU | 最常用,8位反量化 |
| `_float_to_FP8rowwise_gpu` | GPU | FP8推理加速 |
| `_FP8rowwise_to_float_gpu` | GPU | FP8反量化 |

#### P1 (重要)

| API | 推荐参考 | 理由 |
|-----|---------|-----|
| `_float_to_fusednbitrowwise_gpu` | GPU | 4位极限压缩 |
| `_fusednbitrowwise_to_float_gpu` | GPU | 4位反量化 |
| `_fused8bitrowwise_to_float_or_half_gpu` | GPU | 灵活输出格式 |

#### P2 (可选)

| API | 推荐参考 | 理由 |
|-----|---------|-----|
| `quantize_mx_cuda` | GPU | 大语言模型压缩 |
| `dequantize_mx_cuda` | GPU | MX反量化 |
| `_float_to_hfp8_gpu` | GPU | 灵活FP8格式 |
| `_fused8bitrowwise_to_float_mixed_dim_gpu` | GPU | 混合维度支持 |

### 5.2 NPU实现策略

#### 5.2.1 并行策略映射

| CUDA概念 | NPU对应 | 说明 |
|---------|---------|-----|
| Thread | AI Core/Vector Unit | NPU的计算单元 |
| Block | 块调度 | 一组并行工作单元 |
| Warp/Shared Memory | 内部同步/局部存储 | NPU的协作机制 |

**推荐策略**:
```
每个NPU核心处理一行 → 对应CUDA的"每线程处理一行"
使用NPU的局部存储 → 对应CUDA的共享内存
利用NPU的SIMD能力 → 对应CUDA的warp shuffle
```

#### 5.2.2 内存管理映射

| CUDA | NPU | 说明 |
|-----|-----|-----|
| __global__ | Global Memory | 设备全局内存 |
| __shared__ | Local Memory/LSM | 核心本地存储 |
| __constant__ | Const Memory | 常量内存 |
| register | Register/SRF | 标量寄存器文件 |

#### 5.2.3 核心算法移植

**8位量化核心算法** (CPU/GPU通用):
```cpp
// 伪代码,适合NPU实现
function quantize_rowwise_fused8bit(input, nrows, ncols):
    output = allocate(nrows, ncols + 8)

    // NPU并行: 每个核心处理一行
    for row in parallel_range(nrows):
        // 1. 计算min/max (需要NPU的归约指令)
        min_val = reduce_min(input[row])
        max_val = reduce_max(input[row])

        // 2. 计算scale和bias
        scale = (max_val - min_val) / 255.0
        bias = min_val

        // 3. 量化 (向量化操作)
        for col in range(ncols):
            output[row][col] = round((input[row][col] - bias) / scale)

        // 4. 附加scale和bias
        store_float(output, row, ncols, scale)
        store_float(output, row, ncols + 4, bias)

    return output
```

**NPU优化要点**:
1. 使用向量化指令处理量化计算
2. 利用NPU的归约指令求min/max
3. 使用DMA搬运scale和bias
4. 避免频繁的host-device同步

### 5.3 性能优化建议

#### 5.3.1 计算优化

| 优化技术 | CUDA实现 | NPU实现 |
|---------|---------|---------|
| 向量化 | 自动SIMT | 显式向量指令 |
| 归约 | warp shuffle | NPU归约指令 |
| 内存合并 | coalesced access | DMA burst传输 |
| 流水线 | kernel overlap | 多buffer流水线 |

#### 5.3.2 内存优化

```cpp
// GPU: 使用共享内存缓存scale/bias
__shared__ float shared_scales[BLOCK_SIZE];

// NPU: 使用本地存储
local_memory float local_scales[BLOCK_SIZE];
```

#### 5.3.3 调度优化

- **大批量**: 一次处理多行,充分利用NPU并行度
- **小批量**: 合并多个小请求,减少启动开销
- **流水线**: 量化计算与数据传输重叠

### 5.4 验证方案

#### 5.4.1 功能验证

```python
# 使用CPU实现作为golden reference
import torch
import fbgemm_gpu

# CPU量化
quant_cpu = fbgemm_gpu.float_to_fused8bitrowwise_cpu(input)

# NPU量化
quant_npu = npu_float_to_fused8bitrowwise(input)

# 对比结果
assert torch.allclose(quant_cpu, quant_npu, atol=1e-5)
```

#### 5.4.2 性能验证

```python
# 性能对比
import time

# GPU
start = time.time()
for _ in range(100):
    quant_gpu = fbgemm_gpu._float_to_fused8bitrowwise_gpu(input)
gpu_time = (time.time() - start) / 100

# NPU
start = time.time()
for _ in range(100):
    quant_npu = npu_float_to_fused8bitrowwise(input)
npu_time = (time.time() - start) / 100

print(f"GPU: {gpu_time*1000:.2f}ms, NPU: {npu_time*1000:.2f}ms")
print(f"Speedup: {gpu_time/npu_time:.2f}x")
```

### 5.5 常见问题

#### Q1: GPU的warp shuffle在NPU中如何实现?

**A**: 使用NPU的向量shuffle指令或核心间通信机制。
- CUDA: `__shfl_down_sync()`
- NPU: `vector_shuffle()` 或 `core_exchange()`

#### Q2: GPU的共享内存在NPU中对应什么?

**A**: 取决于NPU架构:
- 华为昇腾: Local Storage / Unified Buffer
- 其他NPU: Vector Register / Local Memory

#### Q3: 如何处理GPU专用功能(MSFP/MX)?

**A**: 两种方案:
1. **自行实现**: 参考GPU算法逻辑,用NPU指令重写
2. **软件模拟**: 先用CPU实现,性能不足再优化

#### Q4: NPU实现比GPU慢怎么办?

**A**: 优化方向:
1. 检查并行度是否充分利用
2. 优化内存访问模式(合并、对齐)
3. 使用NPU专用指令(向量化、归约)
4. 减少host-device同步
5. 使用多buffer流水线

---

## 附录A: 快速参考表

### A.1 CPU API速查表

| API | 功能 | 文件位置 |
|-----|------|---------|
| `float_to_fused8bitrowwise_cpu` | FP32→8bit | [sparse_ops.h:402](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L402) |
| `fused8bitrowwise_to_float_cpu` | 8bit→FP32 | [sparse_ops.h:408](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L408) |
| `float_to_FP8rowwise_cpu` | FP32→FP8 | [sparse_ops.h:403](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L403) |
| `FP8rowwise_to_float_cpu` | FP8→FP32 | [sparse_ops.h:409](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L409) |
| `float_to_fusednbitrowwise_cpu` | FP32→Nbit | [sparse_ops.h:488](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L488) |
| `fusednbitrowwise_to_float_cpu` | Nbit→FP32 | [sparse_ops.h:497](fbgemm_gpu/include/fbgemm_gpu/sparse_ops.h#L497) |
| `_float_to_hfp8_cpu` | FP32→HFP8 | [quantize_ops_utils.h:21](fbgemm_gpu/include/fbgemm_gpu/quantize_ops_utils.h#L21) |
| `_hfp8_to_float_cpu` | HFP8→FP32 | [quantize_ops_utils.h:16](fbgemm_gpu/include/fbgemm_gpu/quantize_ops_utils.h#L16) |

### A.2 GPU API速查表

| API | 功能 | 实现文件 |
|-----|------|---------|
| `_float_to_fused8bitrowwise_gpu` | FP32→8bit | [quantize_fused_8bit_rowwise.cu](fbgemm_gpu/src/quantize_ops/quantize_fused_8bit_rowwise.cu) |
| `_fused8bitrowwise_to_float_gpu` | 8bit→FP32 | [quantize_fused_8bit_rowwise.cu](fbgemm_gpu/src/quantize_ops/quantize_fused_8bit_rowwise.cu) |
| `_float_to_FP8rowwise_gpu` | FP32→FP8 | [quantize_fp8_rowwise.cu](fbgemm_gpu/src/quantize_ops/quantize_fp8_rowwise.cu) |
| `_FP8rowwise_to_float_gpu` | FP8→FP32 | [quantize_fp8_rowwise.cu](fbgemm_gpu/src/quantize_ops/quantize_fp8_rowwise.cu) |
| `_float_to_fusednbitrowwise_gpu` | FP32→Nbit | [quantize_fused_nbit_rowwise.cu](fbgemm_gpu/src/quantize_ops/quantize_fused_nbit_rowwise.cu) |
| `_fusednbitrowwise_to_float_gpu` | Nbit→FP32 | [quantize_fused_nbit_rowwise.cu](fbgemm_gpu/src/quantize_ops/quantize_fused_nbit_rowwise.cu) |
| `_float_to_hfp8_gpu` | FP32→HFP8 | [quantize_hfp8.cu](fbgemm_gpu/src/quantize_ops/quantize_hfp8.cu) |
| `_hfp8_to_float_gpu` | HFP8→FP32 | [quantize_hfp8.cu](fbgemm_gpu/src/quantize_ops/quantize_hfp8.cu) |
| `_float_to_msfp_gpu` | FP32→MSFP | [quantize_msfp.cu](fbgemm_gpu/src/quantize_ops/quantize_msfp.cu) |
| `_msfp_to_float_gpu` | MSFP→FP32 | [quantize_msfp.cu](fbgemm_gpu/src/quantize_ops/quantize_msfp.cu) |
| `quantize_mx_cuda` | FP32→MX | [quantize_mx.cu](fbgemm_gpu/src/quantize_ops/quantize_mx.cu) |
| `dequantize_mx_cuda` | MX→FP32 | [quantize_mx.cu](fbgemm_gpu/src/quantize_ops/quantize_mx.cu) |

### A.3 实现文件映射

| 功能 | CPU实现 | GPU实现 |
|-----|--------|--------|
| 8位量化 | [quantize_ops_cpu.cpp](fbgemm_gpu/src/quantize_ops/quantize_ops_cpu.cpp) | [quantize_fused_8bit_rowwise.cu](fbgemm_gpu/src/quantize_ops/quantize_fused_8bit_rowwise.cu) |
| FP8量化 | Meta函数 | [quantize_fp8_rowwise.cu](fbgemm_gpu/src/quantize_ops/quantize_fp8_rowwise.cu) |
| N位量化 | [quantize_ops_cpu.cpp](fbgemm_gpu/src/quantize_ops/quantize_ops_cpu.cpp) | [quantize_fused_nbit_rowwise.cu](fbgemm_gpu/src/quantize_ops/quantize_fused_nbit_rowwise.cu) |
| HFP8量化 | [quantize_ops_utils.h](fbgemm_gpu/include/fbgemm_gpu/quantize_ops_utils.h) | [quantize_hfp8.cu](fbgemm_gpu/src/quantize_ops/quantize_hfp8.cu) |
| MSFP量化 | 无 | [quantize_msfp.cu](fbgemm_gpu/src/quantize_ops/quantize_msfp.cu) |
| MX量化 | 无 | [quantize_mx.cu](fbgemm_gpu/src/quantize_ops/quantize_mx.cu) |
| 类型转换 | 内置函数 | [quantize_bfloat16.cu](fbgemm_gpu/src/quantize_ops/quantize_bfloat16.cu) |

---

## 附录B: NPU实现路线图

### 阶段1: 基础功能 (2-4周)

- [ ] `_float_to_fused8bitrowwise_gpu`
- [ ] `_fused8bitrowwise_to_float_gpu`
- [ ] 基础功能验证

### 阶段2: 扩展功能 (2-4周)

- [ ] `_float_to_FP8rowwise_gpu`
- [ ] `_FP8rowwise_to_float_gpu`
- [ ] `_float_to_fusednbitrowwise_gpu`
- [ ] `_fusednbitrowwise_to_float_gpu`

### 阶段3: 高级功能 (4-8周)

- [ ] `quantize_mx_cuda` / `dequantize_mx_cuda`
- [ ] `_fused8bitrowwise_to_float_mixed_dim_gpu`
- [ ] `_float_to_hfp8_gpu` / `_hfp8_to_float_gpu`
- [ ] 性能优化

### 阶段4: 生产优化 (持续)

- [ ] 性能调优
- [ ] 内存优化
- [ ] 多流并行
- [ ] 生产验证

---

## 参考文档

- [FBGEMM GitHub](https://github.com/pytorch/FBGEMM)
- [PyTorch量化文档](https://pytorch.org/docs/stable/quantization.html)
- [CUDA C++编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [FP8格式说明](https://arxiv.org/abs/2209.05433)
- [昇腾NPU开发文档](https://www.hiascend.com/document)

---

**文档版本**: v2.0 (CPU/GPU分离版)
**最后更新**: 2026-02-25
**维护者**: SimmerChan
**许可证**: Apache 2.0
