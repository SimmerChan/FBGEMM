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

// Register dispatcher functions using CatchAll dispatch key
// The dispatcher itself will route to CUDA/CPU/NPU based on tensor device
TORCH_LIBRARY_IMPL(fbgemm, CatchAll, m) {
  // Sparse operations
  m.impl("invert_permute", &fbgemm_gpu::dispatch::invert_permute);
  m.impl("permute_1d", &fbgemm_gpu::dispatch::permute_1d);
  m.impl("permute_2d", &fbgemm_gpu::dispatch::permute_2d);
  m.impl("index_select_dim0", &fbgemm_gpu::dispatch::index_select_dim0);

  // Add more operators here as they are implemented
  // m.impl("pack_segments", &fbgemm_gpu::dispatch::pack_segments);
}
