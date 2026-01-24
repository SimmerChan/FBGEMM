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

#include "fbgemm_gpu/src/npu/dispatch/dispatcher.h"
#include <torch/library.h>

// Register dispatcher functions for quantization operations
// These will be implemented when quantization operators are added
TORCH_LIBRARY_IMPL(fbgemm, CatchAll, m) {
  // Quantization operations - to be implemented
  // m.impl("quantize", &fbgemm_gpu::dispatch::quantize);
  // m.impl("dequantize", &fbgemm_gpu::dispatch::dequantize);
  // m.impl("float_to_bfloat16", &fbgemm_gpu::dispatch::float_to_bfloat16);
}
