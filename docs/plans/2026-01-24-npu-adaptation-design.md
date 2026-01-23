# FBGEMM_GPU昇腾NPU适配方案设计文档

**项目**: FBGEMM_GPU Ascend NPU Backend Adaptation
**日期**: 2026-01-24
**版本**: v1.0
**状态**: 设计阶段

---

## 文档修订历史

| 版本 | 日期 | 作者 | 变更说明 |
|------|------|------|---------|
| v1.0 | 2026-01-24 | Claude | 初始版本 |

---

## 目录

1. [总体架构设计](#1-总体架构设计)
2. [核心组件设计](#2-核心组件设计)
3. [编译系统设计](#3-编译系统设计)
4. [类型系统设计](#4-类型系统设计)
5. [算子注册和分发机制](#5-算子注册和分发机制)
6. [错误处理和日志](#6-错误处理和日志)
7. [测试策略](#7-测试策略)
8. [实施路线图](#8-实施路线图)

---

## 1. 总体架构设计

### 1.1 设计目标

FBGEMM_GPU昇腾NPU适配框架的核心目标是：**在不破坏现有代码结构和编译流程的前提下，提供完整的NPU后端支持**。关键设计原则包括：

1. **完全解耦**：NPU代码独立于CUDA/CPU实现，昇腾团队可以自主开发维护
2. **零侵入**：不修改现有CUDA/CPU源文件，通过分发器实现路由
3. **性能优先**：C++层分发保证最小开销（~10ns）
4. **渐进式集成**：NPU作为可选后端，默认关闭，不影响其他开发者

### 1.2 分层架构

采用四层架构设计，从上到下依次为：

```
┌─────────────────────────────────────────┐
│  PyTorch用户层                           │
│  torch.ops.fbgemm.xxx(tensor)           │
└──────────────┬──────────────────────────┘
               │ 自动设备检测
┌──────────────▼──────────────────────────┐
│  统一分发层 (Dispatcher)                 │
│  根据tensor.device()路由到对应后端       │
└──────────────┬──────────────────────────┘
               │
     ┌─────────┼─────────┬─────────┐
     │         │         │         │
┌────▼───┐ ┌──▼────┐ ┌──▼────┐ ┌──▼────┐
│ CUDA   │ │  CPU  │ │  NPU   │ │ ROCm  │
│ 实现   │ │ 实现  │ │ 实现   │ │ 实现  │
└────────┘ └───────┘ └───────┘ └───────┘
```

### 1.3 目录组织

```
fbgemm_gpu/
├── src/
│   ├── sparse_ops/           # 现有CUDA实现（不变）
│   ├── npu/                  # NPU后端（新增）
│   │   ├── dispatch/         # 分发器层
│   │   ├── sparse_ops/       # NPU算子实现
│   │   ├── quantize_ops/     # NPU量化算子
│   │   ├── utils/            # NPU工具函数
│   │   ├── ascendc/          # AscendC kernel（独立编译）
│   │   └── CMakeLists.txt    # NPU构建配置
│   └── ...
├── include/fbgemm_gpu/       # 公共头文件
│   └── npu/                  # NPU公共接口（新增）
├── test/
│   └── npu/                  # NPU测试（新增）
└── CMakeLists.txt            # 主构建配置（最小修改）
```

### 1.4 设计决策总结

| 决策项 | 选择方案 | 理由 |
|-------|---------|------|
| 代码组织 | 扁平化架构（npu/目录） | 符合FBGEMM模式，易于合入 |
| 编译集成 | 集成到主CMake | 统一构建流程 |
| 命名空间 | 扩展fbgemm命名空间 | 用户API一致 |
| 设备检测 | 自动设备检测 | 用户无感知 |
| 依赖管理 | 可选依赖（USE_NPU） | 不影响其他开发者 |
| 算子路由 | 统一接口分发器 | 性能+可维护性最佳 |
| 编译方式 | 两阶段编译 | 完全解耦，独立工具链 |
| 类型系统 | 映射到FBGEMM类型系统 | 保持编程模型一致 |
| AscendC目录 | 每个算子独立目录 | 完全隔离，便于并行开发 |

---

## 2. 核心组件设计

### 2.1 统一分发器 (Dispatcher)

**文件位置**: `fbgemm_gpu/src/npu/dispatch/dispatcher.h`

分发器提供统一的算子入口，使用前向声明避免循环依赖。通过`inline`函数保证零额外开销（~10ns）。分发逻辑根据`tensor.device().type()`判断：

```cpp
inline Tensor invert_permute(const Tensor& permute) {
  if (permute.device().type() == c10::kPrivateUse1) {
    return npu::invert_permute_npu(permute);
  } else if (permute.device().type() == c10::kCUDA) {
    return cuda::invert_permute_cuda(permute);
  }
  return cpu::invert_permute_cpu(permute);
}
```

### 2.2 NPU算子三层架构

**Layer 1: PyTorch Wrapper层** (`fbgemm_gpu/src/npu/sparse_ops/*.cpp`)
- 提供符合FBGEMM规范的C++接口
- 使用`AT_DISPATCH_*`宏进行类型调度
- 处理张量连续性、设备检查等前置条件

**Layer 2: AscendC Host层** (每个算子独立的`op_host/`)
- `.cpp`: Tiling函数、Shape推导、类型推导
- `_tiling.h`: Tiling数据结构定义
- 计算AI Core数量、线程数、数据分块策略

**Layer 3: AscendC Kernel层** (每个算子独立的`op_kernel/`)
- `.cpp`: NPU计算逻辑（AscendC API）
- `_kernel.h`: Kernel函数声明
- 从tiling参数读取配置并执行

### 2.3 AscendC目录结构（每个算子独立）

```
fbgemm_gpu/src/npu/ascendc/
├── invert_permute/
│   ├── op_host/
│   │   ├── invert_permute.cpp
│   │   └── invert_permute_tiling.h
│   └── op_kernel/
│       ├── invert_permute.cpp
│       └── invert_permute_kernel.h
├── permute_1d/
│   ├── op_host/
│   │   ├── permute_1d.cpp
│   │   └── permute_1d_tiling.h
│   └── op_kernel/
│       ├── permute_1d.cpp
│       └── permute_1d_kernel.h
├── ...
└── CMakeLists.txt
```

**优势**：完全解耦、独立演进、符合AscendC实践、易于合入。

### 2.4 Tiling机制

Tiling是AscendC算子的核心机制，负责：
- 计算需要的AI Core数量（blockDim）
- 计算每个block的线程数（threadsPerBlock）
- 根据数据类型选择kernel实例（通过TilingKey）
- 将参数传递给Kernel执行

类似于CUDA的grid/block配置，但更复杂，因为NPU有多个AI Core需要协同工作。

---

## 3. 编译系统设计

### 3.1 两阶段编译架构

采用完全独立的编译流程，AscendC算子与FBGEMM主库分开编译：

```
┌─────────────────────────────────────────────────┐
│  阶段1: AscendC算子编译                          │
│  工具: AscendC Compiler (msopgen + soc)        │
│  输入: op_host/*.cpp + op_kernel/*.cpp          │
│  输出: libinvert_permute.so, libpermute_1d.so   │
│  触发: cmake -DUSE_NPU=ON                       │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│  阶段2: NPU Wrapper编译                         │
│  工具: CMake + CANN                             │
│  输入: npu/sparse_ops/*.cpp + 阶段1的.so        │
│  输出: libfbgemm_npu.so                         │
│  链接: libfbgemm_gpu.so                         │
└─────────────────────────────────────────────────┘
```

### 3.2 CMake配置结构

**主CMakeLists.txt修改** (最小侵入):
```cmake
# fbgemm_gpu/CMakeLists.txt

# NPU支持 - 可选依赖
option(USE_NPU "Build with Ascend NPU support" OFF)

if(USE_NPU)
  # 检查CANN工具链
  find_package(CANN REQUIRED)

  # 添加NPU子目录
  add_subdirectory(src/npu)

  # 链接NPU wrapper
  target_link_libraries(fbgemm_gpu PRIVATE fbgemm_npu_wrapper)
endif()
```

**NPU CMakeLists.txt**:
```cmake
# fbgemm_gpu/src/npu/CMakeLists.txt

# 编译AscendC算子
add_subdirectory(ascendc)

# NPU Wrapper库
add_library(fbgemm_npu_wrapper STATIC
  dispatch/sparse_ops.cpp
  sparse_ops/invert_permute.cpp
  sparse_ops/permute_1d.cpp
)

target_link_libraries(fbgemm_npu_wrapper
  PUBLIC
    fbgemm_gpu::utils
    torch::torch
    ${CANN_LIBRARIES}
    ascendc_ops
)

target_include_directories(fbgemm_npu_wrapper
  PUBLIC
    ${CANN_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
```

**AscendC CMakeLists.txt**:
```cmake
# fbgemm_gpu/src/npu/ascendc/CMakeLists.txt

# 为每个算子创建独立库
foreach(op ${NPU_ASCENDC_OPS})
  add_subdirectory(${op})
endforeach()

# 聚合所有AscendC算子
add_library(ascendc_ops INTERFACE)
target_link_libraries(ascendc_ops INTERFACE ${NPU_ASCENDC_LIBS})
```

### 3.3 工具链检测和配置

```cmake
# 检测CANN安装路径
if(NOT DEFINED CANN_PATH)
  set(CANN_PATH "/usr/local/Ascend" CACHE PATH "CANN installation path")
endif()

# 检测AscendC编译器
find_program(ASCENDC_COMPILER
  msopgen
  PATHS ${CANN_PATH}/bin
  NO_DEFAULT_PATH
)

if(NOT ASCENDC_COMPILER)
  message(WARNING "AscendC compiler not found. NPU support will be disabled.")
  set(USE_NPU OFF)
endif()
```

### 3.4 编译产物

```
fbgmm_gpu/build/
├── lib/
│   ├── libfbgemm_gpu.so              # 主库（无NPU依赖）
│   ├── libfbgemm_npu_wrapper.so       # NPU wrapper（需USE_NPU=ON）
│   └── npu_ops/                       # AscendC算子库
│       ├── libinvert_permute.so
│       ├── libpermute_1d.so
│       └── ...
└── npu/
    └── ascendc/
        ├── invert_permute/
        ├── permute_1d/
        └── ...
```

---

## 4. 类型系统设计

### 4.1 FBGEMM类型系统映射

FBGEMM使用PyTorch的`AT_DISPATCH_*`宏进行类型调度。NPU wrapper需要映射到FBGEMM的类型系统：

```cpp
// fbgemm_gpu/src/npu/sparse_ops/invert_permute.cpp

Tensor invert_permute_npu(const Tensor& permute) {
  // 使用FBGEMM的类型调度宏
  AT_DISPATCH_INDEX_TYPES(
      permute.scalar_type(),
      "invert_permute_npu",
      [&] {
        return invert_permute_npu_impl<index_t>(permute);
      });
}

template <typename index_t>
Tensor invert_permute_npu_impl(const Tensor& permute) {
  // 映射到AscendC的数据类型
  if constexpr (std::is_same_v<index_t, int32_t>) {
    return call_ascendc_op<int32_t>(permute);
  } else if constexpr (std::is_same_v<index_t, int64_t>) {
    return call_ascendc_op<int64_t>(permute);
  }
}
```

### 4.2 类型映射表

| FBGEMM/C++类型 | PyTorch ScalarType | AscendC类型 | ACLNN API |
|---------------|-------------------|-----------|-----------|
| `int32_t` | `kInt` | `ge::DT_INT32` | `aclnnInvertPermuteInt32` |
| `int64_t` | `kLong` | `ge::DT_INT64` | `aclnnInvertPermuteInt64` |
| `float` | `kFloat` | `ge::DT_FLOAT` | `aclnnXxxFloat` |
| `float16` | `kHalf` | `ge::DT_FLOAT16` | `aclnnXxxFloat16` |
| `bfloat16` | `kBFloat16` | `ge::DT_BF16` | `aclnnXxxBf16` |

### 4.3 NPU类型工具函数

```cpp
// fbgemm_gpu/src/npu/utils/npu_type_utils.h

namespace fbgemm_gpu::npu {

// 类型映射：c10::ScalarType -> AscendC ge::DataType
inline ge::DataType get_ge_dtype(c10::ScalarType type) {
  switch(type) {
    case c10::kInt: return ge::DT_INT32;
    case c10::kLong: return ge::DT_INT64;
    case c10::kFloat: return ge::DT_FLOAT;
    case c10::kHalf: return ge::DT_FLOAT16;
    case c10::kBFloat16: return ge::DT_BF16;
    default: TORCH_CHECK(false, "Unsupported dtype for NPU");
  }
}

// 设备检查
inline bool is_npu_tensor(const Tensor& tensor) {
  return tensor.device().type() == c10::kPrivateUse1;
}

// NPU设备保护（类似CUDA_DEVICE_GUARD）
class NPUDeviceGuard {
  c10::Device prev_device;
public:
  explicit NPUDeviceGuard(const Tensor& tensor) {
    prev_device = c10::current_device();
    c10::set_device(tensor.device());
  }
  ~NPUDeviceGuard() {
    c10::set_device(prev_device);
  }
};

} // namespace fbgemm_gpu::npu
```

### 4.4 类型调度宏扩展

复用FBGEMM现有的DISPATCH宏系统，最小化学习成本：

```cpp
// 扩展现有的DISPATCH宏以支持NPU类型检查
#ifdef USE_NPU
#define TENSOR_ON_NPU(tensor)                                         \
  TORCH_CHECK(                                                       \
      fbgemm_gpu::npu::is_npu_tensor(tensor),                        \
      "Tensor must be on NPU device but is on device: ",           \
      (tensor).device());
#else
#define TENSOR_ON_NPU(tensor)                                         \
  TORCH_CHECK(false, "NPU support not compiled in");
#endif
```

---

## 5. 算子注册和分发机制

### 5.1 CUDA现有注册方式

FBGEMM的CUDA算子使用简单的`DISPATCH_TO_CUDA`宏逐个注册：

```cpp
// fbgemm_gpu/src/sparse_ops/sparse_ops_gpu.cpp
TORCH_LIBRARY_IMPL(fbgemm, CUDA, m) {
  DISPATCH_TO_CUDA("invert_permute", fbgemm_gpu::invert_permute_cuda);
  DISPATCH_TO_CUDA("permute_1d", fbgemm_gpu::permute_1d_cuda);
  // ... 逐个注册
}
```

`DISPATCH_TO_CUDA`宏定义：
```cpp
#define DISPATCH_TO_CUDA(name, function) \
  m.impl(name, torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(function)))
```

**特点**：
- ✅ 简单直接
- ✅ 每个算子独立注册
- ✅ 没有批量宏系统

### 5.2 NPU分发器注册（与CUDA保持一致）

NPU使用与CUDA完全一致的注册风格：

```cpp
// fbgemm_gpu/src/npu/dispatch/sparse_ops.cpp
#include "fbgemm_gpu/src/npu/dispatch/dispatcher.h"
#include <torch/library.h>

// 使用CatchAll分发键注册分发器
// 分发器内部会根据设备类型路由到对应后端
TORCH_LIBRARY_IMPL(fbgemm, CatchAll, m) {
  m.impl("invert_permute", &fbgemm_gpu::dispatch::invert_permute);
  m.impl("permute_1d", &fbgemm_gpu::dispatch::permute_1d);
  m.impl("permute_2d", &fbgemm_gpu::dispatch::permute_2d);
  m.impl("index_select_dim0", &fbgemm_gpu::dispatch::index_select_dim0);
  // ... 逐个注册，与CUDA风格一致
}
```

### 5.3 分发器完整代码

**文件**: `fbgemm_gpu/src/npu/dispatch/dispatcher.h`
```cpp
#pragma once

#include <ATen/Tensor.h>
#include <torch/library.h>

namespace fbgemm_gpu::dispatch {

// 前向声明CUDA实现
namespace cuda {
Tensor invert_permute_cuda(const Tensor& permute);
Tensor permute_1d_cuda(const Tensor& input, const Tensor& indices);
} // namespace cuda

// 前向声明CPU实现
namespace cpu {
Tensor invert_permute_cpu(const Tensor& permute);
Tensor permute_1d_cpu(const Tensor& input, const Tensor& indices);
} // namespace cpu

// NPU实现声明和定义
#ifdef USE_NPU
namespace npu {
Tensor invert_permute_npu(const Tensor& permute);
Tensor permute_1d_npu(const Tensor& input, const Tensor& indices);
} // namespace npu
#endif

// 统一分发函数
inline Tensor invert_permute(const Tensor& permute) {
  auto device_type = permute.device().type();

  if (device_type == c10::kPrivateUse1) {
#ifdef USE_NPU
    return npu::invert_permute_npu(permute);
#else
    TORCH_CHECK(false, "NPU support not compiled in. Please rebuild with -DUSE_NPU=ON");
#endif
  } else if (device_type == c10::kCUDA) {
    return cuda::invert_permute_cuda(permute);
  } else {
    return cpu::invert_permute_cpu(permute);
  }
}

inline Tensor permute_1d(const Tensor& input, const Tensor& indices) {
  auto device_type = input.device().type();

  if (device_type == c10::kPrivateUse1) {
#ifdef USE_NPU
    return npu::permute_1d_npu(input, indices);
#else
    TORCH_CHECK(false, "NPU support not compiled in");
#endif
  } else if (device_type == c10::kCUDA) {
    return cuda::permute_1d_cuda(input, indices);
  } else {
    return cpu::permute_1d_cpu(input, indices);
  }
}

} // namespace fbgemm_gpu::dispatch
```

### 5.4 注册优先级说明

PyTorch的调度优先级：
```
1. 具体设备类型 (CUDA/CPU)  ← 优先级高
2. CatchAll                   ← 优先级低，但可以覆盖所有情况
```

通过`CatchAll`注册分发器后：
- CUDA张量：分发器检测到`device.type() == CUDA`，调用`cuda::op_cuda()`
- NPU张量：分发器检测到`device.type() == PrivateUse1`，调用`npu::op_npu()`
- CPU张量：分发器检测到`device.type() == CPU`，调用`cpu::op_cpu()`

这样**无需修改CUDA现有注册代码**，完全解耦。

---

## 6. 错误处理和日志

### 6.1 错误处理策略

NPU算子的错误处理需要与FBGEMM现有风格保持一致：

**FBGEMM现有风格**：
```cpp
// CUDA算子的错误处理
Tensor invert_permute_cuda(const Tensor& permute) {
  TENSOR_ON_CUDA_GPU(permute);  // 设备检查
  CUDA_DEVICE_GUARD(permute);    // 设备保护

  TORCH_CHECK_VALUE(
      permute.dim() == 1,
      "invert_permute expects 1D tensor, got ",
      permute.dim());
}
```

**NPU采用相同风格**：
```cpp
// fbgemm_gpu/src/npu/sparse_ops/invert_permute.cpp

Tensor invert_permute_npu(const Tensor& permute) {
#ifdef USE_NPU
  // 设备检查
  TENSOR_ON_NPU(permute);

  // 设备保护
  NPUDeviceGuard guard(permute);

  // 参数验证
  TORCH_CHECK_VALUE(
      permute.dim() == 1,
      "invert_permute expects 1D tensor, got ",
      permute.dim());

  TORCH_CHECK_VALUE(
      permute.numel() > 0 && permute.numel() <= (1L << 31),
      "invert_permute: tensor size must be in [1, 2^31], got ",
      permute.numel());

  // 调用AscendC算子
  Tensor output = at::empty_like(permute);
  EXEC_NPU_CMD(aclnnInvertPermute, permute, output);

  return output;
#else
  TORCH_CHECK(false, "NPU support not compiled in");
#endif
}
```

### 6.2 NPU专用宏定义

```cpp
// fbgemm_gpu/src/npu/utils/npu_ops_utils.h

#pragma once

#include <ATen/Tensor.h>
#include <c10/core/DeviceType.h>
#include <torch/library.h>

#ifdef USE_NPU

namespace fbgemm_gpu::npu {

// 设备检查宏
#define TENSOR_ON_NPU(tensor)                                         \
  TORCH_CHECK(                                                       \
      (tensor).device().type() == c10::kPrivateUse1,                 \
      "Tensor must be on NPU device but is on device: ",            \
      (tensor).device());

// NPU设备保护类
class NPUDeviceGuard {
  c10::Device prev_device_;
public:
  explicit NPUDeviceGuard(const at::Tensor& tensor)
    : prev_device_(c10::current_device()) {
    c10::set_device(tensor.device());
  }

  ~NPUDeviceGuard() {
    c10::set_device(prev_device_);
  }

  NPUDeviceGuard(const NPUDeviceGuard&) = delete;
  NPUDeviceGuard& operator=(const NPUDeviceGuard&) = delete;
};

// ACLNN错误检查宏
#define CHECK_NPU_CMD(cmd)                                          \
  do {                                                              \
    auto ret = (cmd);                                               \
    if (ret != 0) {                                                 \
      TORCH_CHECK(false, "NPU command failed: " #cmd ", ret=", ret); \
    }                                                               \
  } while(0)

} // namespace fbgemm_gpu::npu

#endif // USE_NPU
```

### 6.3 日志系统

复用FBGEMM现有的日志系统（如果有），或使用PyTorch标准日志：

```cpp
// 使用PyTorch的logging系统
#include <ATen/Logging.h>

Tensor invert_permute_npu(const Tensor& permute) {
#ifdef USE_NPU
  // 调试日志
  VLOG(1) << "invert_permute_npu called with tensor shape: "
          << permute.sizes();

  // 性能日志
  at::record_function_start("invert_permute_npu");

  // 执行算子
  Tensor output = ...;

  at::record_function_end("invert_permute_npu");

  return output;
#else
  TORCH_CHECK(false, "NPU support not compiled in");
#endif
}
```

### 6.4 错误信息一致性

所有错误信息需要遵循FBGEMM的命名和格式规范：

```cpp
// 好的错误信息
TORCH_CHECK(
    permute.is_contiguous(),
    "invert_permute_npu: input tensor must be contiguous, got stride=",
    permute.strides());

// 避免的错误信息（太简单）
TORCH_CHECK(permute.is_contiguous(), "tensor not contiguous");

// 避免的错误信息（太复杂）
TORCH_CHECK(
    permute.is_contiguous(),
    "The input tensor provided to the invert_permute operation on the NPU "
    "device must be contiguous in memory layout...");
```

### 6.5 边界条件处理

参考AscendC样例的约束：

```cpp
Tensor invert_permute_npu(const Tensor& permute) {
  // 类型检查
  TORCH_CHECK(
      permute.scalar_type() == at::kInt ||
      permute.scalar_type() == at::kLong,
      "invert_permute_npu only supports int32 and int64, got ",
      permute.scalar_type());

  // 维度检查
  TORCH_CHECK(
      permute.dim() == 1,
      "invert_permute_npu expects 1D tensor, got dim=",
      permute.dim(),
      ", shape=",
      permute.sizes());

  // 长度检查（参考AscendC约束）
  int64_t length = permute.numel();
  TORCH_CHECK(
      length >= 1 && length <= (1L << 31),
      "invert_permute_npu: tensor length must be in [1, 2^31], got ",
      length);

  // 空张量特殊处理
  if (length == 0) {
    return at::empty_like(permute);
  }

  // ... 正常逻辑
}
```

---

## 7. 测试策略

### 7.1 现实约束

**实际部署场景**：
- GPU环境：使用CUDA实现
- NPU环境：使用NPU实现
- **没有GPU+NPU混合部署场景**

**测试策略调整**：
- 不能在同一用例中对比GPU和NPU结果
- 需要分别验证各设备的正确性
- 使用CPU作为参考验证

### 7.2 测试目录结构

**符合FBGEMM现有结构**：
```
fbgemm_gpu/test/
├── npu/                           # NPU专用测试目录（新增）
│   ├── sparse_ops_test.py         # 稀疏算子测试
│   ├── quantize_ops_test.py       # 量化算子测试
│   └── dispatch_test.py           # 分发器路由测试
├── combine/                       # 现有测试目录
├── permute/                       # 现有测试目录
└── test_utils.py                  # 扩展公共工具
```

### 7.3 使用unittest + Hypothesis框架

**FBGEMM现有风格**：
```python
# fbgemm_gpu/test/npu/sparse_ops_test.py

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import unittest
import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity

try:
    from fbgemm_gpu import open_source
    from test_utils import gpu_unavailable
except Exception:
    from fbgemm_gpu.test.test_utils import gpu_unavailable

# NPU不可用时的跳过装饰器
npu_unavailable = (
    not hasattr(torch, 'npu') or not torch.npu.is_available(),
    "NPU is not available"
)

class NPUInvertPermuteTest(unittest.TestCase):
    """invert_permute算子的NPU测试"""

    @unittest.skipIf(*npu_unavailable)
    @given(
        size=st.integers(min_value=1, max_value=10000),
        dtype=st.sampled_from([torch.int32, torch.int64]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_invert_permute_correctness(self, size: int, dtype: torch.dtype) -> None:
        """测试NPU与CPU结果一致性"""
        import torch_npu

        # 生成输入
        x = torch.randperm(size).to(dtype)

        # CPU计算（参考）
        y_cpu = torch.ops.fbgemm.invert_permute(x)

        # NPU计算
        x_npu = x.npu()
        y_npu = torch.ops.fbgemm.invert_permute(x_npu)

        # 验证结果一致
        torch.testing.assert_close(
            y_npu.cpu(),
            y_cpu,
            rtol=1e-5,
            atol=1e-5,
            msg=f"NPU和CPU结果不一致"
        )

    @unittest.skipIf(*npu_unavailable)
    def test_invert_permute_empty_tensor(self) -> None:
        """测试空张量边界条件"""
        x = torch.tensor([], dtype=torch.int32)
        x_npu = x.npu()

        y_npu = torch.ops.fbgemm.invert_permute(x_npu)
        y_cpu = torch.ops.fbgemm.invert_permute(x)

        torch.testing.assert_close(y_npu.cpu(), y_cpu.cpu())
```

### 7.4 扩展test_utils.py

**在现有test_utils.py中添加NPU支持**：
```python
# fbgemm_gpu/test/test_utils.py (新增)

# NPU设备检测（与gpu_unavailable风格一致）
npu_unavailable: tuple[bool, str] = (
    not hasattr(torch, 'npu') or not torch.npu.is_available(),
    "NPU is not available"
)

npu_available: bool = not npu_unavailable[0]
```

### 7.5 分发器集成测试

```python
# fbgemm_gpu/test/npu/dispatch_test.py

import unittest
from test_utils import npu_unavailable

class DispatcherTest(unittest.TestCase):
    """测试设备路由逻辑"""

    @unittest.skipIf(*npu_unavailable)
    def test_npu_routing(self) -> None:
        """测试NPU张量路由到NPU实现"""
        x = torch.tensor([1, 2, 0]).npu()
        y = torch.ops.fbgemm.invert_permute(x)

        # 验证输出在NPU上
        self.assertEqual(y.device.type, 'privateuseone')
        self.assertEqual(y.device.index, 0)

    def test_cpu_routing(self) -> None:
        """测试CPU张量路由到CPU实现"""
        x = torch.tensor([1, 2, 0])
        y = torch.ops.fbgemm.invert_permute(x)

        self.assertEqual(y.device.type, 'cpu')
```

### 7.6 CI/CD分离

**不同的CI环境**：
```yaml
# GPU环境CI（现有）
name: GPU Tests
runs-on: [self-hosted, gpu]
steps:
  - run: python -m unittest fbgemm_gpu.test.sparse_ops_test

# NPU环境CI（新增）
name: NPU Tests
runs-on: [self-hosted, npu]
steps:
  - run: python -m unittest fbgemm_gpu.test.npu.sparse_ops_test
```

### 7.7 测试覆盖率要求

| 测试类型 | 覆盖率要求 | 说明 |
|---------|----------|------|
| **单元测试** | 100% | 每个NPU算子必须有测试 |
| **类型覆盖** | 4种类型 | int32, int64, float, half |
| **边界条件** | 5+场景 | 空张量、单元素、大张量、非连续、错误输入 |
| **设备路由** | NPU+CPU | NPU环境验证，CPU作为参考 |
| **性能测试** | 5个规模 | 不同规模张量的性能测试（独立运行） |

---

## 8. 实施路线图

### 8.1 总体时间规划

```
阶段1: 框架搭建 (2-3周)
  ├── CMake配置
  ├── 目录结构创建
  ├── 分发器实现
  └── 工具函数库

阶段2: 示例算子实现 (1-2周)
  ├── 选取1-2个简单算子
  ├── AscendC实现
  ├── Wrapper实现
  └── 单元测试

阶段3: 核心算子适配 (4-6周)
  ├── 稀疏算子
  ├── 量化算子
  └── 性能优化

阶段4: 测试和验证 (2-3周)
  ├── 完善单元测试
  ├── 集成测试
  └── 性能调优

阶段5: 文档和发布 (1-2周)
  ├── 开发文档
  ├── 用户文档
  └── 提交PR
```

### 8.2 阶段1：框架搭建

**目标**：建立完整的NPU适配框架，无功能算子

**任务清单**：

1. **CMake配置**
   - [ ] 添加`USE_NPU`选项
   - [ ] CANN工具链检测
   - [ ] 创建`fbgemm_gpu/src/npu/CMakeLists.txt`
   - [ ] 创建`fbgemm_gpu/src/npu/ascendc/CMakeLists.txt`

2. **目录结构创建**
   ```bash
   mkdir -p fbgemm_gpu/src/npu/{dispatch,sparse_ops,quantize_ops,utils}
   mkdir -p fbgemm_gpu/src/npu/ascendc/invert_permute/{op_host,op_kernel}
   ```

3. **分发器实现**
   - [ ] `fbgemm_gpu/src/npu/dispatch/dispatcher.h`
   - [ ] `fbgemm_gpu/src/npu/dispatch/sparse_ops.cpp`
   - [ ] 测试分发器路由逻辑

4. **工具函数库**
   - [ ] `fbgemm_gpu/src/npu/utils/npu_ops_utils.h`
   - [ ] `fbgemm_gpu/src/npu/utils/npu_type_utils.h`
   - [ ] `TENSOR_ON_NPU`宏
   - [ ] `NPUDeviceGuard`类

5. **测试框架**
   - [ ] 创建`fbgemm_gpu/test/npu/`目录
   - [ ] 扩展`test_utils.py`添加`npu_unavailable`
   - [ ] 创建CI配置模板

**验收标准**：
- ✅ `cmake -DUSE_NPU=ON`可以成功配置
- ✅ 编译生成`libfbgemm_npu_wrapper.a`
- ✅ 分发器可以正确路由CPU张量

### 8.3 阶段2：示例算子实现

**目标**：实现1-2个简单算子，验证完整流程

**选择算子**：`invert_permute`（简单、无状态）

**任务清单**：

1. **AscendC实现**
   - [ ] `op_host/invert_permute.cpp`
   - [ ] `op_host/invert_permute_tiling.h`
   - [ ] `op_kernel/invert_permute.cpp`
   - [ ] `op_kernel/invert_permute_kernel.h`
   - [ ] 独立编译测试

2. **NPU Wrapper实现**
   - [ ] `sparse_ops/invert_permute.cpp`
   - [ ] 类型调度（int32/int64）
   - [ ] 边界检查

3. **分发器集成**
   - [ ] 在`dispatcher.h`中声明
   - [ ] 在`sparse_ops.cpp`中注册

4. **单元测试**
   - [ ] 正确性测试（vs CPU）
   - [ ] 边界条件测试
   - [ ] 错误处理测试

**验收标准**：
- ✅ `torch.ops.fbgemm.invert_permute(npu_tensor)`正确执行
- ✅ 单元测试通过率100%
- ✅ 与CPU结果数值一致

### 8.4 阶段3：核心算子适配

**目标**：适配推荐系统核心算子

**优先级排序**：

| 优先级 | 算子类型 | 典型算子 | 复杂度 |
|-------|---------|---------|--------|
| P0 | 稀疏算子 | permute_1d, permute_2d | 低 |
| P0 | 稀疏算子 | index_select_dim0 | 中 |
| P1 | 稀疏算子 | pack_segments | 中 |
| P1 | 量化算子 | quantize, dequantize | 低 |
| P2 | Embedding | split_embedding_lookup | 高 |

**并行开发策略**：
- 昇腾团队：2-3人并行开发不同算子
- 每完成一个算子立即提交测试
- 每周同步进度和问题

### 8.5 阶段4：测试和验证

**目标**：确保质量和性能

**任务清单**：

1. **单元测试完善**
   - [ ] 每个算子覆盖4种类型
   - [ ] 边界条件测试
   - [ ] 错误输入测试

2. **集成测试**
   - [ ] 分发器路由测试
   - [ ] 多算子组合测试
   - [ ] 端到端场景测试

3. **性能测试**
   - [ ] 不同数据规模测试
   - [ ] Tiling参数调优
   - [ ] vs CPU性能对比

4. **CI/CD集成**
   - [ ] NPU CI环境配置
   - [ ] 自动化测试流程
   - [ ] 性能回归检测

**验收标准**：
- ✅ 单元测试覆盖率 > 90%
- ✅ 所有集成测试通过
- ✅ 性能达到预期目标

### 8.6 阶段5：文档和发布

**目标**：准备合入主分支

**任务清单**：

1. **开发文档**
   - [ ] 架构设计文档（本文档）
   - [ ] 编译指南
   - [ ] 算子开发指南
   - [ ] 调试指南

2. **用户文档**
   - [ ] NPU支持说明
   - [ ] 安装指南
   - [ ] 迁移指南（CUDA→NPU）

3. **代码审查准备**
   - [ ] 清理临时代码
   - [ ] 代码注释完善
   - [ ] 合入commit message规范

4. **提交PR**
   - [ ] 分拆为多个小PR（按模块）
   - [ ] 每个PR独立可测试
   - [ ] 提供详细的变更说明

### 8.7 风险和应对

| 风险 | 影响 | 概率 | 应对措施 |
|------|------|------|---------|
| AscendC工具链问题 | 高 | 中 | 提前验证工具链，准备备选方案 |
| 性能不达标 | 高 | 中 | 预留性能优化时间，邀请NPU专家 |
| 与FBGEMM设计冲突 | 中 | 低 | 早期与FBGEMM团队对齐设计 |
| CI环境不稳定 | 中 | 中 | 建立稳定的NPU测试环境 |
| 资源不足 | 高 | 低 | 合理规划优先级，聚焦核心算子 |

### 8.8 成功标准

**框架合入标准**：
- ✅ 零侵入现有CUDA/CPU代码
- ✅ 编译可选（`USE_NPU=OFF`不影响）
- ✅ 代码符合FBGEMM规范
- ✅ 通过所有CI测试

**功能完成标准**（MVP）：
- ✅ 支持5个核心稀疏算子
- ✅ 支持2个量化算子
- ✅ 单元测试覆盖率 > 90%
- ✅ 在实际模型中验证通过

---

## 附录

### A. 参考文档

1. **FBGEMM_GPU项目**
   - GitHub: https://github.com/pytorch/FBGEMM
   - 文档: https://fbgemm.readthedocs.io/

2. **AscendC开发指南**
   - AscendC算子样例: `/home/hsl/RecSDK/cust_op`
   - CANN文档: 华为官方文档

3. **PyTorch扩展**
   - PyTorch C++ Extension: https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html
   - TORCH_LIBRARY: https://pytorch.org/tutorials/advanced/torch_script_custom_classes.html

### B. 术语表

| 术语 | 说明 |
|------|------|
| NPU | Neural Processing Unit，昇腾AI处理器 |
| AscendC | 昇腾AI处理器的C++编程语言 |
| CANN | Compute Architecture for Neural Networks，昇腾AI计算架构 |
| ACLNN | Ascend Computing Library for Neural Networks |
| Tiling | 数据分块策略，将数据分配到多个AI Core |
| AI Core | 昇腾NPU的计算核心单元 |
| PrivateUse1 | PyTorch中用于自定义设备的设备类型 |
| CatchAll | PyTorch dispatch key，捕获所有设备类型 |

### C. 联系方式

**项目相关**：
- FBGEMM GitHub Issues: https://github.com/pytorch/FBGEMM/issues
- 昇腾技术支持: 华为官方支持渠道

---

**文档结束**
