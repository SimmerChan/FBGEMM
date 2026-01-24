# NPU适配方案对比 - 以invert_permute算子为例

## 前置条件：现有CUDA实现

**文件位置**: `fbgemm_gpu/src/sparse_ops/sparse_invert_permute.cu`

```cpp
namespace fbgemm_gpu {

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void invert_permute_kernel(
    int32_t permute_size,
    const index_t* __restrict__ permute,
    index_t* __restrict__ inversed_permute) {
  CUDA_KERNEL_LOOP(i, permute_size) {
    inversed_permute[permute[i]] = i;
  }
}

DLL_PUBLIC Tensor invert_permute_cuda(const Tensor& permute) {
  TENSOR_ON_CUDA_GPU(permute);
  CUDA_DEVICE_GUARD(permute);

  const auto permute_contig = permute.contiguous();
  const auto permute_size = permute.numel();
  Tensor inversed_permute = at::empty_like(permute);

  if (permute_size == 0) {
    return inversed_permute;
  }

  constexpr int32_t threads_1 = kMaxThreads;
  const auto blocks_1 = cuda_calc_xblock_count(permute_size, threads_1);
  AT_DISPATCH_INDEX_TYPES(permute.scalar_type(), "invert_permute_kernel", [&] {
    FBGEMM_LAUNCH_KERNEL(
        (invert_permute_kernel<index_t>),
        blocks_1,
        threads_1,
        0,
        at::cuda::getCurrentCUDAStream(),
        permute_size,
        permute_contig.data_ptr<index_t>(),
        inversed_permute.data_ptr<index_t>());
  });
  return inversed_permute;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(CUDA, "invert_permute", fbgemm_gpu::invert_permute_cuda);
```

---

## 方案A: 独立的NPU算子入口 (Python层分发)

### 目录结构
```
fbgemm_gpu/
└── src/
    └── npu/
        ├── sparse_ops/
        │   ├── invert_permute.cpp          # NPU算子实现
        │   └── ascendc_kernels/
        │       └── invert_permute.ascendc  # AscendC kernel
        └── torch_bindings/
            └── npu_ops.cpp                 # PyTorch绑定
```

### NPU算子实现
**文件**: `fbgemm_gpu/src/npu/sparse_ops/invert_permute.cpp`
```cpp
#include <torch/library.h>
#include <torch_npu/inc/cann_npu.h>
#include "fbgemm_gpu/utils/ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu::npu {

DLL_PUBLIC Tensor invert_permute_npu(const Tensor& permute) {
  // NPU设备检查
  TORCH_CHECK(
      permute.device().type() == at::kPrivateUse1,
      "invert_permute_npu expects NPU tensor");

  // 调用AscendC算子（通过ACLNN API）
  Tensor inversed_permute = at::empty_like(permute);

  // 执行NPU算子
  EXEC_NPU_CMD(aclnnInvertPermute, permute, inversed_permute);

  return inversed_permute;
}

} // namespace fbgemm_gpu::npu
```

### PyTorch绑定
**文件**: `fbgemm_gpu/src/npu/torch_bindings/npu_ops.cpp`
```cpp
#include <torch/library.h>

namespace fbgemm_gpu::npu {

// 声明NPU实现
Tensor invert_permute_npu(const Tensor& permute);

} // namespace fbgemm_gpu::npu

// 注册到fbgemm命名空间，设备类型为PrivateUse1 (NPU)
TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m) {
  m.impl("invert_permute", &fbgemm_gpu::npu::invert_permute_npu);
}
```

### Python层调用
```python
import torch
import torch_npu
import fbgemm_gpu

# 用户代码 - 完全透明，自动选择后端
def invert_permute(permute: Tensor) -> Tensor:
    """
    根据张量所在设备自动选择后端
    - NPU张量 → NPU算子
    - CUDA张量 → CUDA算子
    - CPU张量 → CPU算子
    """
    return torch.ops.fbgemm.invert_permute(permute)

# 使用示例
x_cuda = torch.tensor([1, 2, 0]).cuda()
y_cuda = invert_permute(x_cuda)  # 自动调用CUDA实现

x_npu = torch.tensor([1, 2, 0]).npu()
y_npu = invert_permute(x_npu)    # 自动调用NPU实现
```

### 优点
- ✅ **完全解耦**：NPU代码独立，不影响CUDA实现
- ✅ **独立开发**：昇腾团队可以独立开发和迭代
- ✅ **零侵入**：不修改现有CUDA/CPU代码

### 缺点
- ❌ **Python层开销**：多一次Python→C++调用（~1-5μs，但相比算子执行时间可忽略）
- ❌ **重复接口**：每个算子需要单独注册

---

## 方案B: 在DISPATCH宏中集成 (C++层集成)

### 目录结构
```
fbgemm_gpu/
└── src/
    └── npu/
        ├── sparse_ops/
        │   └── ascendc_kernels/
        │       └── invert_permute.ascendc
        └── utils/
            └── npu_ops_utils.h
```

### 修改现有CUDA文件
**文件**: `fbgemm_gpu/src/sparse_ops/sparse_invert_permute.cu` (需要修改现有文件)

```cpp
#include "fbgemm_gpu/src/npu/utils/npu_ops_utils.h"  // 新增

namespace fbgemm_gpu {

template <typename index_t>
__global__ __launch_bounds__(kMaxThreads) void invert_permute_kernel(
    int32_t permute_size,
    const index_t* __restrict__ permute,
    index_t* __restrict__ inversed_permute) {
  CUDA_KERNEL_LOOP(i, permute_size) {
    inversed_permute[permute[i]] = i;
  }
}

// 新增NPU实现
#ifdef USE_NPU
namespace npu {
Tensor invert_permute_npu_impl(const Tensor& permute) {
  EXEC_NPU_CMD(aclnnInvertPermute, permute, at::empty_like(permute));
}
} // namespace npu
#endif

DLL_PUBLIC Tensor invert_permute_cuda(const Tensor& permute) {
  // 修改：移除CUDA设备强制检查
  // TENSOR_ON_CUDA_GPU(permute);  // 删除

  // 新增：根据设备类型分发
  if (permute.device().type() == at::kPrivateUse1) {
#ifdef USE_NPU
    return npu::invert_permute_npu_impl(permute);
#else
    TORCH_CHECK(false, "NPU support not compiled in");
#endif
  }

  // 原有CUDA逻辑
  TENSOR_ON_CUDA_GPU(permute);
  CUDA_DEVICE_GUARD(permute);

  const auto permute_contig = permute.contiguous();
  const auto permute_size = permute.numel();
  Tensor inversed_permute = at::empty_like(permute);

  if (permute_size == 0) {
    return inversed_permute;
  }

  constexpr int32_t threads_1 = kMaxThreads;
  const auto blocks_1 = cuda_calc_xblock_count(permute_size, threads_1);
  AT_DISPATCH_INDEX_TYPES(permute.scalar_type(), "invert_permute_kernel", [&] {
    FBGEMM_LAUNCH_KERNEL(
        (invert_permute_kernel<index_t>),
        blocks_1,
        threads_1,
        0,
        at::cuda::getCurrentCUDAStream(),
        permute_size,
        permute_contig.data_ptr<index_t>(),
        inversed_permute.data_ptr<index_t>());
  });
  return inversed_permute;
}

} // namespace fbgemm_gpu

FBGEMM_OP_DISPATCH(CUDA, "invert_permute", fbgemm_gpu::invert_permute_cuda);
```

### 优点
- ✅ **性能最优**：C++层分发，零额外开销
- ✅ **统一接口**：单一函数入口

### 缺点
- ❌ **高度侵入**：需要修改所有现有CUDA实现文件
- ❌ **维护困难**：NPU逻辑和CUDA代码混在一起
- ❌ **冲突风险**：每次CUDA更新都可能产生冲突
- ❌ **Review负担**：FBGEMM owner需要review大量修改
- ❌ **不符合独立团队协作模式**

---

## 方案C: 统一接口分发器 (推荐)

### 目录结构
```
fbgemm_gpu/
└── src/
    └── npu/
        ├── dispatch/              # 分发器层
        │   ├── dispatcher.h
        │   └── sparse_ops.cpp
        ├── sparse_ops/
        │   ├── invert_permute.cpp          # NPU算子实现
        │   └── ascendc_kernels/
        │       └── invert_permute.ascendc
        └── utils/
            └── npu_ops_utils.h
```

### 分发器接口定义
**文件**: `fbgemm_gpu/src/npu/dispatch/dispatcher.h`
```cpp
#pragma once

#include <ATen/Tensor.h>
#include <torch/library.h>

namespace fbgemm_gpu::dispatch {

// 前向声明各后端实现
namespace cuda {
Tensor invert_permute_cuda(const Tensor& permute);
} // namespace cuda

namespace cpu {
Tensor invert_permute_cpu(const Tensor& permute);
} // namespace cpu

#ifdef USE_NPU
namespace npu {
Tensor invert_permute_npu(const Tensor& permute);
} // namespace npu
#endif

// 统一分发器
inline Tensor invert_permute(const Tensor& permute) {
  auto device_type = permute.device().type();

  // 根据设备类型路由到对应后端
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

} // namespace fbgemm_gpu::dispatch
```

### NPU算子实现
**文件**: `fbgemm_gpu/src/npu/sparse_ops/invert_permute.cpp`
```cpp
#include "fbgemm_gpu/src/npu/dispatch/dispatcher.h"
#include <torch_npu/inc/cann_npu.h>
#include "fbgemm_gpu/utils/ops_utils.h"

using Tensor = at::Tensor;

namespace fbgemm_gpu::npu {

DLL_PUBLIC Tensor invert_permute_npu(const Tensor& permute) {
  TORCH_CHECK(
      permute.device().type() == at::kPrivateUse1,
      "invert_permute_npu expects NPU tensor");

  Tensor inversed_permute = at::empty_like(permute);
  EXEC_NPU_CMD(aclnnInvertPermute, permute, inversed_permute);
  return inversed_permute;
}

} // namespace fbgemm_gpu::npu
```

### 注册分发器
**文件**: `fbgemm_gpu/src/npu/dispatch/sparse_ops.cpp`
```cpp
#include "fbgemm_gpu/src/npu/dispatch/dispatcher.h"
#include <torch/library.h>

// 注册分发器到fbgemm命名空间
TORCH_LIBRARY_IMPL(fbgemm, CatchAll, m) {
  m.impl("invert_permute", &fbgemm_gpu::dispatch::invert_permute);
}
```

### 用户代码 (完全透明)
```python
import torch
import torch_npu
import fbgemm_gpu

# 用户无需关心底层实现
x = torch.tensor([1, 2, 0])

# CUDA
x_cuda = x.cuda()
y_cuda = torch.ops.fbgemm.invert_permute(x_cuda)  # 自动使用CUDA

# NPU
x_npu = x.npu()
y_npu = torch.ops.fbgemm.invert_permute(x_npu)    # 自动使用NPU

# CPU
y_cpu = torch.ops.fbgemm.invert_permute(x)        # 自动使用CPU
```

### 扩展到多算子
**文件**: `fbgemm_gpu/src/npu/dispatch/dispatcher.h`
```cpp
namespace fbgemm_gpu::dispatch {

// 支持的算子列表
#define FOR_EACH_NPU_OP(_) \
  _(invert_permute) \
  _(permute_1d) \
  _(permute_2d) \
  _(index_select_dim0) \
  _(pack_segments)

// 自动生成分发器宏
#define DECLARE_DISPATCH_OP(op_name) \
  namespace cuda { Tensor op_name##_cuda(const Tensor&); } \
  namespace cpu { Tensor op_name##_cpu(const Tensor&); } \
  FOR_EACH_NPU_OP(DECLARE_DISPATCH_OP)

// 统一分发器
inline Tensor invert_permute(const Tensor& x) {
  return dispatch_op("invert_permute", x);
}

} // namespace fbgemm_gpu::dispatch
```

### 优点
- ✅ **性能优秀**：C++层分发，开销可忽略（~10ns）
- ✅ **解耦彻底**：不修改现有CUDA/CPU代码
- ✅ **可维护性高**：分发逻辑集中管理
- ✅ **易于扩展**：添加新算子只需在dispatcher中注册
- ✅ **清晰分层**：NPU、CUDA、CPU实现完全分离
- ✅ **便于合入**：FBGEMM owner只需review新增的dispatcher代码

### 缺点
- ⚠️ **额外抽象层**：增加一层函数调用（但开销极小）

---

## 方案对比总结

| 维度 | 方案A (Python层) | 方案B (DISPATCH集成) | 方案C (统一分发器) |
|------|----------------|-------------------|------------------|
| **性能开销** | ~1-5μs | ~10ns | ~10ns |
| **代码侵入** | 无 | 高 | 低 |
| **可维护性** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **独立开发** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| **FBGEMM接受度** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ |
| **扩展性** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 推荐方案

**选择方案C - 统一接口分发器**

理由：
1. 性能和可维护性的最佳平衡
2. 符合独立团队协作模式
3. 易于被FBGEMM owner接受和合入
4. 为未来扩展预留空间
