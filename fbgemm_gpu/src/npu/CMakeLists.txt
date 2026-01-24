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
# NPU Backend Support for FBGEMM_GPU
################################################################################

# NPU is completely optional - only build if USE_NPU=ON
if(NOT USE_NPU)
  return()
endif()

# Find CANN toolkit
if(NOT DEFINED CANN_PATH)
  set(CANN_PATH "/usr/local/Ascend" CACHE PATH "CANN installation path")
endif()

# Check for CANN installation
find_path(CANN_INCLUDE_DIRS
  NAMES
    acl/acl.h
    ascenddc.h
  PATHS
    ${CANN_PATH}/include
    ${CANN_PATH}/ascend/toolkit/latest/include
    /usr/local/Ascend/ascend/toolkit/latest/include
  NO_DEFAULT_PATH
)

if(NOT CANN_INCLUDE_DIRS)
  message(WARNING
    "CANN headers not found. NPU support will be disabled. "
    "Set CANN_PATH to point to CANN installation."
  )
  return()
endif()

# Find CANN libraries
find_library(CANN_ACL_LIB
  NAMES
    ascendcl
    acl
  PATHS
    ${CANN_PATH}/lib64
    ${CANN_PATH}/ascend/toolkit/latest/lib64
    /usr/local/Ascend/ascend/toolkit/latest/lib64
  NO_DEFAULT_PATH
)

if(NOT CANN_ACL_LIB)
  message(WARNING
    "CANN libraries not found. NPU support will be disabled."
  )
  return()
endif()

set(CANN_LIBRARIES ${CANN_ACL_LIB})
message(STATUS "Found CANN: ${CANN_PATH}")
message(STATUS "  CANN_INCLUDE_DIRS: ${CANN_INCLUDE_DIRS}")
message(STATUS "  CANN_LIBRARIES: ${CANN_LIBRARIES}")

# Compile AscendC kernels first
add_subdirectory(ascendc)

# NPU wrapper library
add_library(fbgemm_npu_wrapper STATIC
  dispatch/sparse_ops.cpp
  dispatch/quantize_ops.cpp
  sparse_ops/invert_permute.cpp
  # Add more operators as they are implemented
  # sparse_ops/permute_1d.cpp
  # quantize_ops/quantize.cpp
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
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

# Set compile definitions
target_compile_definitions(fbgemm_npu_wrapper PRIVATE USE_NPU)

# Alias for consistency
add_library(fbgemm_gpu::npu_wrapper ALIAS fbgemm_npu_wrapper)

message(STATUS "NPU backend enabled: fbgemm_npu_wrapper")
