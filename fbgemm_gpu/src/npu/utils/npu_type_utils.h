// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <ATen/Tensor.h>
#include <c10/core/ScalarType.h>
#include <torch/library.h>

#ifdef USE_NPU

// Note: AscendC headers will be included when CANN is available
// For now, we use placeholder types
namespace ge {
  enum DataType {
    DT_INT32 = 3,
    DT_INT64 = 9,
    DT_FLOAT = 0,
    DT_FLOAT16 = 1,
    DT_BF16 = 27
  };
}

namespace fbgemm_gpu::npu {

// Type mapping: PyTorch ScalarType -> AscendC ge::DataType
inline ge::DataType get_ge_dtype(c10::ScalarType type) {
  switch(type) {
    case c10::kInt:
      return ge::DT_INT32;
    case c10::kLong:
      return ge::DT_INT64;
    case c10::kFloat:
      return ge::DT_FLOAT;
    case c10::kHalf:
      return ge::DT_FLOAT16;
    case c10::kBFloat16:
      return ge::DT_BF16;
    default:
      TORCH_CHECK(false, "Unsupported dtype for NPU: ", toString(type));
  }
}

// Device detection - check if tensor is on NPU device
inline bool is_npu_tensor(const at::Tensor& tensor) {
  return tensor.device().type() == c10::kPrivateUse1;
}

// Get tensor device index
inline int8_t get_npu_device_index(const at::Tensor& tensor) {
  TORCH_CHECK(
    is_npu_tensor(tensor),
    "get_npu_device_index: tensor is not on NPU device"
  );
  return tensor.device().index();
}

} // namespace fbgemm_gpu::npu

#endif // USE_NPU
