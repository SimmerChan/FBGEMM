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
#include <c10/core/DeviceType.h>
#include <torch/library.h>

#ifdef USE_NPU

namespace fbgemm_gpu::npu {

// Device check macro - verifies tensor is on NPU device
#define TENSOR_ON_NPU(tensor)                                         \
  TORCH_CHECK(                                                       \
    (tensor).device().type() == c10::kPrivateUse1,                  \
    "Tensor must be on NPU device but is on device: ",              \
    (tensor).device());

// NPU device guard (RAII pattern)
// Automatically saves and restores the current NPU device
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

// ACLNN error checking macro
// Checks the return code from NPU API calls and throws if non-zero
#define CHECK_NPU_CMD(cmd)                                          \
  do {                                                              \
    auto ret = (cmd);                                               \
    if (ret != 0) {                                                 \
      TORCH_CHECK(false, "NPU command failed: " #cmd ", ret=", ret);\
    }                                                               \
  } while(0)

// NPU execution macro (placeholder for actual ACLNN API)
// This will be replaced with actual AscendC API calls
#define EXEC_NPU_CMD(op, ...)                                       \
  do {                                                              \
    CHECK_NPU_CMD(op(__VA_ARGS__));                                 \
  } while(0)

} // namespace fbgemm_gpu::npu

#endif // USE_NPU
