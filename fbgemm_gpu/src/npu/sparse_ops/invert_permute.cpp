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

#include "fbgemm_gpu/npu/utils/npu_ops_utils.h"
#include "fbgemm_gpu/npu/utils/npu_type_utils.h"
#include <ATen/Tensor.h>
#include <torch/library.h>

#ifdef USE_NPU

namespace fbgemm_gpu::npu {

at::Tensor invert_permute_npu(const at::Tensor& permute) {
  // Device check
  TENSOR_ON_NPU(permute);

  // Device guard
  NPUDeviceGuard guard(permute);

  // Parameter validation
  TORCH_CHECK(
    permute.dim() == 1,
    "invert_permute_npu expects 1D tensor, got dim=",
    permute.dim(),
    ", shape=",
    permute.sizes());

  TORCH_CHECK(
    permute.scalar_type() == at::kInt ||
    permute.scalar_type() == at::kLong,
    "invert_permute_npu only supports int32/int64, got ",
    permute.scalar_type());

  // Size validation (AscendC constraints)
  int64_t length = permute.numel();
  TORCH_CHECK(
    length >= 0 && length <= (1L << 31),
    "invert_permute_npu: tensor length must be in [0, 2^31], got ",
    length);

  // Empty tensor special case
  if (length == 0) {
    return at::empty_like(permute);
  }

  // Type dispatch
  return AT_DISPATCH_INDEX_TYPES(
    permute.scalar_type(),
    "invert_permute_npu",
    [&] {
      return invert_permute_npu_impl<index_t>(permute);
    });
}

template <typename index_t>
at::Tensor invert_permute_npu_impl(const at::Tensor& permute) {
  // Create output tensor
  at::Tensor output = at::empty_like(permute);

  // TODO: Call AscendC kernel when implemented
  // For now, this is a placeholder that will be replaced
  // with actual ACLNN API calls or custom AscendC .so calls

  // Example of what the actual call will look like:
  // if constexpr (std::is_same_v<index_t, int32_t>) {
  //   EXEC_NPU_CMD(aclnnInvertPermuteInt32, permute, output);
  // } else {
  //   EXEC_NPU_CMD(aclnnInvertPermuteInt64, permute, output);
  // }

  // Placeholder: copy to CPU, compute, copy back
  // This will be removed once AscendC kernel is implemented
  at::Tensor permute_cpu = permute.to(at::kCPU);
  at::Tensor output_cpu = at::empty_like(permute_cpu);

  // Simple CPU implementation for testing dispatcher
  auto permute_acc = permute_cpu.accessor<index_t, 1>();
  auto output_acc = output_cpu.accessor<index_t, 1>();

  for (int64_t i = 0; i < permute.numel(); ++i) {
    output_acc[permute_acc[i]] = static_cast<index_t>(i);
  }

  output = output_cpu.to(permute.device());

  return output;
}

} // namespace fbgemm_gpu::npu

#endif // USE_NPU
