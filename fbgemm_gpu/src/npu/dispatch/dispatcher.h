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
#include <torch/library.h>

namespace fbgemm_gpu::dispatch {

// Forward declarations for CUDA implementations
namespace cuda {
at::Tensor invert_permute_cuda(const at::Tensor& permute);
at::Tensor permute_1d_cuda(
  const at::Tensor& input,
  const at::Tensor& indices);
at::Tensor permute_2d_cuda(
  const at::Tensor& input,
  const at::Tensor& indices);
at::Tensor index_select_dim0_cuda(
  const at::Tensor& input,
  const at::Tensor& indices);
} // namespace cuda

// Forward declarations for CPU implementations
namespace cpu {
at::Tensor invert_permute_cpu(const at::Tensor& permute);
at::Tensor permute_1d_cpu(
  const at::Tensor& input,
  const at::Tensor& indices);
at::Tensor permute_2d_cpu(
  const at::Tensor& input,
  const at::Tensor& indices);
at::Tensor index_select_dim0_cpu(
  const at::Tensor& input,
  const at::Tensor& indices);
} // namespace cpu

// Forward declarations for NPU implementations
#ifdef USE_NPU
namespace npu {
at::Tensor invert_permute_npu(const at::Tensor& permute);
at::Tensor permute_1d_npu(
  const at::Tensor& input,
  const at::Tensor& indices);
at::Tensor permute_2d_npu(
  const at::Tensor& input,
  const at::Tensor& indices);
at::Tensor index_select_dim0_npu(
  const at::Tensor& input,
  const at::Tensor& indices);
} // namespace npu
#endif

////////////////////////////////////////////////////////////////////////////////
// Unified dispatch functions
// These inline functions route operations to the appropriate backend
// based on tensor device type. The inline ensures zero overhead (~10ns).
////////////////////////////////////////////////////////////////////////////////

inline at::Tensor invert_permute(const at::Tensor& permute) {
  auto device_type = permute.device().type();

  if (device_type == c10::kPrivateUse1) {
#ifdef USE_NPU
    return npu::invert_permute_npu(permute);
#else
    TORCH_CHECK(
      false,
      "NPU support not compiled in. Please rebuild with -DUSE_NPU=ON");
#endif
  } else if (device_type == c10::kCUDA) {
    return cuda::invert_permute_cuda(permute);
  } else {
    return cpu::invert_permute_cpu(permute);
  }
}

inline at::Tensor permute_1d(
  const at::Tensor& input,
  const at::Tensor& indices) {
  auto device_type = input.device().type();

  if (device_type == c10::kPrivateUse1) {
#ifdef USE_NPU
    return npu::permute_1d_npu(input, indices);
#else
    TORCH_CHECK(
      false,
      "NPU support not compiled in. Please rebuild with -DUSE_NPU=ON");
#endif
  } else if (device_type == c10::kCUDA) {
    return cuda::permute_1d_cuda(input, indices);
  } else {
    return cpu::permute_1d_cpu(input, indices);
  }
}

inline at::Tensor permute_2d(
  const at::Tensor& input,
  const at::Tensor& indices) {
  auto device_type = input.device().type();

  if (device_type == c10::kPrivateUse1) {
#ifdef USE_NPU
    return npu::permute_2d_npu(input, indices);
#else
    TORCH_CHECK(
      false,
      "NPU support not compiled in. Please rebuild with -DUSE_NPU=ON");
#endif
  } else if (device_type == c10::kCUDA) {
    return cuda::permute_2d_cuda(input, indices);
  } else {
    return cpu::permute_2d_cpu(input, indices);
  }
}

inline at::Tensor index_select_dim0(
  const at::Tensor& input,
  const at::Tensor& indices) {
  auto device_type = input.device().type();

  if (device_type == c10::kPrivateUse1) {
#ifdef USE_NPU
    return npu::index_select_dim0_npu(input, indices);
#else
    TORCH_CHECK(
      false,
      "NPU support not compiled in. Please rebuild with -DUSE_NPU=ON");
#endif
  } else if (device_type == c10::kCUDA) {
    return cuda::index_select_dim0_cuda(input, indices);
  } else {
    return cpu::index_select_dim0_cpu(input, indices);
  }
}

} // namespace fbgemm_gpu::dispatch
