# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

################################################################################
# AscendC Kernels Build Configuration
################################################################################

# List of AscendC operators to compile
# Each operator will be in its own subdirectory: <op_name>/{op_host,op_kernel}
set(NPU_ASCENDC_OPS
  # Add operators as they are implemented
  # invert_permute
  # permute_1d
  # permute_2d
  # index_select_dim0
  # pack_segments
)

set(NPU_ASCENDC_LIBS)

# Build each operator independently
foreach(op ${NPU_ASCENDC_OPS})
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${op})
    add_subdirectory(${op})
    list(APPEND NPU_ASCENDC_LIBS ascendc_${op})
  endif()
endforeach()

# Create aggregate interface library for all AscendC operators
add_library(ascendc_ops INTERFACE)
if(NPU_ASCENDC_LIBS)
  target_link_libraries(ascendc_ops INTERFACE ${NPU_ASCENDC_LIBS})
  message(STATUS "AscendC kernels: ${NPU_ASCENDC_LIBS}")
else()
  message(STATUS "No AscendC kernels configured yet")
endif()
