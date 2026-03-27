# zero_collision_hash 昇腾NPU迁移实现方案

## 一、迁移概述

本文档详细描述了将FBGEMM的`zero_collision_hash`算子迁移到昇腾NPU平台的完整实现方案，参照**fbgemm-ascend**代码仓的架构和构建方式，包括目录结构设计、SIMT kernel实现、算子框架注册等。

### 1.1 算子功能说明

`zero_collision_hash`是FBGEMM中用于特征ID重新映射的无冲突哈希算法，主要功能：
- 使用MurmurHash3计算哈希值
- 通过探测（probing）解决哈希冲突
- 支持淘汰策略（eviction）管理哈希表
- 支持只读模式（推理）和训练模式

### 1.2 迁移策略

| 迁移内容 | 原始实现 | 迁移后实现 |
|---------|---------|-----------|
| 计算后端 | CUDA | AscendC SIMT |
| 算子框架 | PyTorch C++ Extension | Ascend OpDef |
| 并行模型 | CUDA Thread Block | SIMT VF_CALL |
| 原子操作 | atomicCAS/atomicExch | AscendC::Simt::AtomicCas/Exch |

### 1.3 芯片版本说明

- **c310**：代表昇腾**A5 950PR芯片**，**仅A5代际芯片支持SIMT接口**，A2/A3芯片不支持
- **SIMT接口适用范围**：仅A5代际（如c310）芯片可使用本方案中的SIMT实现
- **A2/A3芯片**：如需支持，需要使用非SIMT接口（见第4.5节）

### 1.4 开发环境

本算子的开发和调试在远端昇腾NPU服务器上进行，环境信息如下：

| 项目 | 信息 |
|-----|------|
| 服务器地址 | `192.168.13.158` |
| SSH端口 | `22` |
| 用户名 | `root` |
| 密码 | `DCauto1!2@` |
| 容器名称 | `hsl_pt` |
| 工作路径 | `/home/hsl` |
| 目标芯片 | A5 950PR (c310) |

**连接与开发流程**：

```bash
# 1. SSH连接到服务器
ssh root@192.168.13.158

# 2. 进入容器
docker exec -it hsl_pt bash

# 3. 进入工作目录
cd /home/hsl

# 4. 算子代码位于
# /home/hsl/zero_collision_hash/
```

**环境变量**（容器内已预装）：

```bash
# Ascend Toolkit 路径
export ASCEND_HOME=/usr/local/Ascend
export ASCEND_TOOLKIT_HOME=${ASCEND_HOME}/ascend-toolkit/latest
export PATH=${ASCEND_TOOLKIT_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${LD_LIBRARY_PATH}

# ccec编译器路径
# /usr/local/Ascend/ascend-toolkit/latest/bin/ccec
```

---

## 二、适配到 fbgemm-ascend 仓库的目录结构

### 2.1 fbgemm-ascend 仓库实际目录结构

基于 fbgemm-ascend（https://gitcode.com/Ascend/fbgemm-ascend）仓库的实际代码，其目录结构如下：

```
fbgemm-ascend/                                    # 项目根目录
├── bench/                                        # 性能基准测试脚本（按算子类别组织）
│   ├── sparse/                                   #   稀疏算子基准测试
│   ├── jagged/                                   #   锯齿张量基准测试
│   ├── pooled_embedding/                         #   池化嵌入基准测试
│   ├── tbe_inference/                            #   TBE推理基准测试
│   └── tbe_training/                             #   TBE训练基准测试
├── fbgemm_ascend/                                # Python包入口
│   ├── __init__.py                               #   环境探测、OPP路径自动设置
│   └── env_setup.sh                              #   环境变量设置脚本
├── include/                                      # C++/AscendC 公共头文件
│   └── fbgemm_ascend/                            #   公共接口头文件
├── src/                                          # 自定义算子实现、注册源码
│   ├── cmake/                                    #   CMake模块（func.cmake等）
│   ├── common/                                   #   公共工具（pytorch_npu_helper.hpp等）
│   ├── common_ops/                               #   Kernel公共头文件
│   │   ├── kernel_common_utils.h
│   │   ├── ops_log.h
│   │   └── workload_sharder.h
│   ├── sparse_ops/                               #   稀疏算子（按算子名组织）
│   │   ├── asynchronous_complete_cumsum/
│   │   │   ├── asynchronous_complete_cumsum.cpp  #   C++适配层（注册到torch.ops）
│   │   │   ├── c310/                            #     A5芯片 AscendC实现
│   │   │   │   ├── asynchronous_complete_cumsum.json
│   │   │   │   ├── op_host/                     #     Tiling + OpDef
│   │   │   │   ├── op_kernel/                   #     SIMT Kernel
│   │   │   │   ├── run.sh                       #     编译脚本
│   │   │   │   └── README.md
│   │   │   └── v220/                            #     A2/A3芯片实现
│   │   ├── block_bucketize_sparse_features/      #   （结构同上）
│   │   └── ...
│   ├── jagged_tensor_ops/                        #   锯齿张量算子
│   ├── pooled_embedding_ops/                     #   池化嵌入算子
│   ├── intraining_embedding_pruning_ops/         #   训练中嵌入剪枝算子
│   ├── tbe_inference/                            #   TBE推理算子
│   └── tbe_training/                             #   TBE训练算子
├── CMakeLists.txt                                # 顶层CMake（构建适配层.so）
├── FbgemmAscend.cmake                            # AscendC算子编译入口
├── setup.py                                      # Python包构建（scikit-build）
├── build_whl.sh                                  # whl包构建脚本
└── README.md
```

**关键架构说明**：

| 层次 | 说明 |
|-----|------|
| `src/<category>/<op_name>/<op_name>.cpp` | C++适配层，使用 `EXEC_NPU_CMD(aclnnXxx, ...)` 调用AscendC算子，并通过 `TORCH_LIBRARY_IMPL` 注册到 `torch.ops.fbgemm` |
| `src/<category>/<op_name>/c310/` | A5芯片的AscendC实现，包含 `op_host/`（Tiling + OpDef）和 `op_kernel/`（SIMT Kernel） |
| `src/<category>/<op_name>/v220/` | A2/A3芯片的AscendC实现（可选，不含SIMT） |
| `FbgemmAscend.cmake` | 统一管理所有AscendC算子的编译注册，新增算子只需在此文件中添加条目 |
| `src/common_ops/` | Kernel公共头文件（`kernel_common_utils.h`、`ops_log.h`等） |
| `src/common/` | C++适配层公共工具（`pytorch_npu_helper.hpp`、`common_utils.h`） |

### 2.2 单个算子的标准目录结构

以 `asynchronous_complete_cumsum` 为例，展示一个典型算子的完整文件组织：

```
src/sparse_ops/asynchronous_complete_cumsum/
├── asynchronous_complete_cumsum.cpp              # [适配层] C++注册 + EXEC_NPU_CMD调用
├── c310/                                         # A5芯片实现
│   ├── asynchronous_complete_cumsum.json         #   算子描述文件（输入/输出/属性）
│   ├── op_host/
│   │   ├── asynchronous_complete_cumsum_tiling.h #   Tiling数据结构定义
│   │   └── asynchronous_complete_cumsum.cpp      #   TilingFunc + OpDef注册
│   ├── op_kernel/
│   │   ├── asynchronous_complete_cumsum_kernel.h #   Kernel类定义
│   │   ├── asynchronous_complete_cumsum.cpp      #   Kernel入口
│   │   └── simt_kernel.h                        #   SIMT Kernel实现（A5独有）
│   ├── run.sh                                    #   编译脚本
│   └── README.md
└── v220/                                         # A2/A3芯片实现（可选）
    ├── asynchronous_complete_cumsum.json
    ├── op_host/
    ├── op_kernel/
    ├── run.sh
    └── README.md
```

### 2.3 zero_collision_hash 适配后的目录结构

`zero_collision_hash` 属于**稀疏算子（sparse_ops）**类别，适配后的目录结构：

```
src/sparse_ops/zero_collision_hash/
├── zero_collision_hash.cpp                       # [适配层] C++注册 + EXEC_NPU_CMD调用
├── c310/                                         # A5芯片实现（SIMT）
│   ├── zero_collision_hash.json                  #   算子描述文件
│   ├── op_host/
│   │   ├── zero_collision_hash_tiling.h          #   Tiling数据结构定义
│   │   └── zero_collision_hash.cpp               #   TilingFunc + OpDef注册
│   ├── op_kernel/
│   │   ├── zero_collision_hash_kernel.h          #   Kernel类定义
│   │   ├── zero_collision_hash.cpp               #   Kernel入口
│   │   └── simt_kernel.h                        #   SIMT Kernel实现
│   ├── run.sh                                    #   编译脚本
│   └── README.md
└── v220/                                         # A2/A3芯片（可选，不含SIMT）
    └── ...（如需支持A2/A3芯片）
```

### 2.4 迁移集成步骤

**步骤一：在仓库中创建算子目录**

```bash
cd /path/to/fbgemm-ascend
mkdir -p src/sparse_ops/zero_collision_hash/c310
```

**步骤二：复制AscendC实现文件**

将本地 `zero_collision_hash_ascendc/c310/` 下的文件复制到 `src/sparse_ops/zero_collision_hash/c310/`：

```bash
cp zero_collision_hash_ascendc/c310/zero_collision_hash.json  src/sparse_ops/zero_collision_hash/c310/
cp -r zero_collision_hash_ascendc/c310/op_host                    src/sparse_ops/zero_collision_hash/c310/
cp -r zero_collision_hash_ascendc/c310/op_kernel                  src/sparse_ops/zero_collision_hash/c310/
cp zero_collision_hash_ascendc/c310/run.sh                        src/sparse_ops/zero_collision_hash/c310/
```

**步骤三：编写C++适配层**

创建 `src/sparse_ops/zero_collision_hash/zero_collision_hash.cpp`，参照仓库中其他算子的模式：

```cpp
// zero_collision_hash.cpp - C++适配层
#include <torch/library.h>
#include "../../common/pytorch_npu_helper.hpp"
#include "../../common/common_utils.h"

// NPU实现函数，通过EXEC_NPU_CMD调用AscendC算子
std::tuple<at::Tensor, at::Tensor> zero_collision_hash_npu(
    const at::Tensor& input,
    const at::Tensor& identities,
    ...
    int64_t max_probe) {
    // ... 参数检查和输出张量分配 ...
    EXEC_NPU_CMD(aclnnZeroCollisionHash, input, identities, output, evict_slots, ...);
    return std::make_tuple(output, evict_slots);
}

// 注册到torch.ops.fbgemm
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    m.def("zero_collision_hash(...)");
}
TORCH_LIBRARY_IMPL(fbgemm, PrivateUse1, m) {
    m.impl("zero_collision_hash", &zero_collision_hash_npu);
}
```

**步骤四：在 FbgemmAscend.cmake 中注册算子**

在 `FbgemmAscend.cmake` 的 `_ASCENDC_OPS` 列表中添加：

```cmake
list(APPEND _ASCENDC_OPS
    "zero_collision_hash|${FBGEMM_ASCEND_SOURCE_DIR}/src/sparse_ops/zero_collision_hash"
)
```

同时在 `FBGEMM_ASCEND_ADAPTER_SRCS` 中添加适配层源文件：

```cmake
list(APPEND FBGEMM_ASCEND_ADAPTER_SRCS
    src/sparse_ops/zero_collision_hash/zero_collision_hash.cpp
)
```

**步骤五：根据芯片支持策略更新编译配置**

根据 zero_collision_hash 仅支持A5（SIMT），在 `FbgemmAscend.cmake` 的 `ASCENDC_A5_ONLY_OPS` 列表中添加：

```cmake
list(APPEND ASCENDC_A5_ONLY_OPS
    zero_collision_hash
)
```

**步骤六：调整 run.sh 中的路径引用**

`run.sh` 中需要复制 `kernel_common_utils.h` 等公共头文件。参考现有算子的 run.sh 模式：

```bash
# 本仓库中 kernel_common_utils.h 在 src/common_ops/（c310 上四级到 src）
cp -rf ../../../common_ops/kernel_common_utils.h <build_dir>/op_kernel/
```

**步骤七：编译验证**

```bash
cd /path/to/fbgemm-ascend
bash build_whl.sh   # 构建完整的whl包
# 或单独编译算子：
cd src/sparse_ops/zero_collision_hash/c310
bash run.sh ai_core-Ascend950
```

### 2.5 芯片版本对应关系

| 芯片代际 | 目录名 | AI Core标识 | SIMT支持 | 说明 |
|---------|-------|------------|---------|------|
| **A5** | `c310` | `ai_core-Ascend950` | ✅ 支持 | 本方案主推 |
| **A2** | `v220` | `ai_core-Ascend910B2` | ❌ 不支持 | 需非SIMT实现 |
| **A3** | `v220` | `ai_core-Ascend910_93` | ❌ 不支持 | 需非SIMT实现 |

> 注意：`FbgemmAscend.cmake` 中 `_fbgemm_get_target_info()` 函数根据芯片变体自动映射到正确的目录和AI Core标识。

### 2.6 参考项目

- **fbgemm-ascend**: https://gitcode.com/Ascend/fbgemm-ascend（目标仓库）
- **HierarchicalKV-ascend**: https://gitcode.com/Ascend/HierarchicalKV-ascend（SIMT实现参考）

---

## 三、核心代码实现

### 3.1 算子配置文件 (zero_collision_hash.json)

```json
[
  {
    "op": "ZeroCollisionHash",
    "language": "cpp",
    "input_desc": [
      {
        "name": "input",
        "param_type": "required",
        "format": ["ND", "ND"],
        "type": ["int64", "int32"]
      },
      {
        "name": "identities",
        "param_type": "required",
        "format": ["ND", "ND"],
        "type": ["int64", "int32"]
      },
      {
        "name": "local_sizes",
        "param_type": "optional",
        "format": ["ND"],
        "type": ["int64"]
      },
      {
        "name": "offsets",
        "param_type": "optional",
        "format": ["ND"],
        "type": ["int64"]
      }
    ],
    "output_desc": [
      {
        "name": "output",
        "param_type": "required",
        "format": ["ND", "ND"],
        "type": ["int64", "int32"]
      },
      {
        "name": "evict_slots",
        "param_type": "required",
        "format": ["ND", "ND"],
        "type": ["int64", "int32"]
      }
    ],
    "attr": [
      {
        "name": "max_probe",
        "param_type": "required",
        "type": "int",
        "default_value": 128
      },
      {
        "name": "circular_probe",
        "param_type": "optional",
        "type": "bool",
        "default_value": false
      },
      {
        "name": "disable_fallback",
        "param_type": "optional",
        "type": "bool",
        "default_value": false
      },
      {
        "name": "hash_identity",
        "param_type": "optional",
        "type": "int",
        "default_value": 1
      }
    ]
  }
]
```

### 3.2 Tiling数据结构定义 (op_host/zero_collision_hash_tiling.h)

```cpp
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ZERO_COLLISION_HASH_TILING_H
#define ZERO_COLLISION_HASH_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(ZeroCollisionHashTilingData)
    // 输入输出维度参数
    TILING_DATA_FIELD_DEF(int64_t, inputLength);        // 输入元素数量
    TILING_DATA_FIELD_DEF(int64_t, modulo);             // 哈希表大小
    TILING_DATA_FIELD_DEF(int64_t, maxProbe);           // 最大探测次数
    
    // 分块参数
    TILING_DATA_FIELD_DEF(int64_t, totalBlocks);        // 总块数
    TILING_DATA_FIELD_DEF(int64_t, blocksPerCore);      // 每核块数
    TILING_DATA_FIELD_DEF(int32_t, remainderBlocks);    // 余数块数
    TILING_DATA_FIELD_DEF(int32_t, elementsPerBlock);   // 每块元素数
    
    // 算子配置参数
    TILING_DATA_FIELD_DEF(bool, circularProbe);         // 是否循环探测
    TILING_DATA_FIELD_DEF(bool, disableFallback);       // 是否禁用fallback
    TILING_DATA_FIELD_DEF(int32_t, hashIdentity);       // 哈希标识类型
    TILING_DATA_FIELD_DEF(bool, hasLocalSizes);         // 是否有local_sizes
    TILING_DATA_FIELD_DEF(bool, hasOffsets);            // 是否有offsets
    
    // Opt-in参数
    TILING_DATA_FIELD_DEF(int64_t, optInProb);          // opt-in概率
    TILING_DATA_FIELD_DEF(int64_t, numReservedSlots);   // 保留槽数量
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ZeroCollisionHash, ZeroCollisionHashTilingData)

}  // namespace optiling

#endif  // ZERO_COLLISION_HASH_TILING_H
```

### 3.3 Host端实现 (op_host/zero_collision_hash.cpp)

参考**fbgemm-ascend**代码仓中算子的Host实现模式，主要包含Tiling计算函数和算子注册类。

```cpp
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdint>
#include <cmath>
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "ops_log.h"
#include "zero_collision_hash_tiling.h"

namespace {
    constexpr int32_t MAX_THREADS_PER_BLOCK = 512;
    constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
    constexpr int DCACHE_SIZE = 128 * 1024;
}

namespace optiling {

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputShape", context->GetInputShape(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("inputTensor", context->GetInputTensor(0), return ge::GRAPH_FAILED);
    OPS_LOG_E_IF_NULL("identitiesShape", context->GetInputShape(1), return ge::GRAPH_FAILED);

    // 获取输入维度
    int64_t inputLength = context->GetInputShape(0)->GetOriginShape().GetShapeSize();
    auto inputTensor = context->GetInputTensor(0);
    ge::DataType inputDataType = inputTensor->GetDataType();

    // 获取identities维度
    int64_t modulo = context->GetInputShape(1)->GetOriginShape().GetDim(0);

    uint32_t dimNum = context->GetInputShape(0)->GetOriginShape().GetDimNum();
    OPS_LOG_E_IF(dimNum != 1, context, return ge::GRAPH_FAILED,
                 "[ERROR]ZeroCollisionHash required the dim of input-0 is 1");

    OPS_CHECK(inputDataType != ge::DT_INT32 && inputDataType != ge::DT_INT64,
              OPS_LOG_E("[ERROR]Invalid data type",
                        "ZeroCollisionHash only support int64 and int32."),
              return ge::GRAPH_FAILED);

    // 获取属性参数（参考fbgemm-ascend中算子的属性读取方式）
    auto attrs = context->GetAttrs();
    int64_t maxProbe = attrs->GetAttrValue<int64_t>(0);      // max_probe
    bool circularProbe = attrs->GetAttrValue<bool>(1);       // circular_probe
    bool disableFallback = attrs->GetAttrValue<bool>(2);     // disable_fallback
    int32_t hashIdentity = attrs->GetAttrValue<int>(3);      // hash_identity
    int64_t optInProb = attrs->GetAttrValue<int64_t>(4);     // opt_in_prob
    int64_t numReservedSlots = attrs->GetAttrValue<int64_t>(5); // num_reserved_slots

    // 检查可选输入
    bool hasLocalSizes = (context->GetInputShape(2) != nullptr);
    bool hasOffsets = (context->GetInputShape(3) != nullptr);

    // 计算分块参数（参考HierarchicalKV-ascend的tiling实现）
    auto ascendPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    size_t maxCores = ascendPlatform.GetCoreNumAiv();

    int32_t elementsPerBlock = MAX_THREADS_PER_BLOCK * MAX_ELEMENTS_PER_THREAD;
    int64_t totalBlocks = (inputLength + elementsPerBlock - 1) / elementsPerBlock;

    size_t coreNum = (totalBlocks < maxCores) ? totalBlocks : maxCores;
    if (coreNum == 0) {
        coreNum = 1;
    }

    int64_t blocksPerCore = totalBlocks / coreNum;
    int32_t remainderBlocks = totalBlocks % coreNum;

    // 设置workspace大小
    size_t* workspaceSize = context->GetWorkspaceSizes(1);
    OPS_LOG_E_IF_NULL("workspaceSize", workspaceSize, return ge::GRAPH_FAILED);
    size_t systemWorkspacesSize = ascendPlatform.GetLibApiWorkSpaceSize();
    workspaceSize[0] = systemWorkspacesSize;

    // 填充tiling数据
    ZeroCollisionHashTilingData tiling;
    tiling.set_inputLength(inputLength);
    tiling.set_modulo(modulo);
    tiling.set_maxProbe(maxProbe);
    tiling.set_totalBlocks(totalBlocks);
    tiling.set_blocksPerCore(blocksPerCore);
    tiling.set_remainderBlocks(remainderBlocks);
    tiling.set_elementsPerBlock(elementsPerBlock);
    tiling.set_circularProbe(circularProbe);
    tiling.set_disableFallback(disableFallback);
    tiling.set_hashIdentity(hashIdentity);
    tiling.set_hasLocalSizes(hasLocalSizes);
    tiling.set_hasOffsets(hasOffsets);
    tiling.set_optInProb(optInProb);
    tiling.set_numReservedSlots(numReservedSlots);

    context->SetBlockDim(coreNum);
    context->SetLocalMemorySize(DCACHE_SIZE);

    OPS_LOG_E_IF_NULL("raw tilingData", context->GetRawTilingData(), return ge::GRAPH_FAILED);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ge {

static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    OPS_LOG_E_IF_NULL("context", context, return ge::GRAPH_FAILED);

    const gert::Shape* inputShape = context->GetInputShape(0);
    OPS_LOG_E_IF_NULL("inputShape", inputShape, return ge::GRAPH_FAILED);

    // 输出output与输入input形状相同
    gert::Shape* outputShape = context->GetOutputShape(0);
    OPS_LOG_E_IF_NULL("outputShape", outputShape, return ge::GRAPH_FAILED);
    *outputShape = *inputShape;

    // evict_slots输出为空或与输入相同大小（推理模式下为空）
    gert::Shape* evictShape = context->GetOutputShape(1);
    OPS_LOG_E_IF_NULL("evictShape", evictShape, return ge::GRAPH_FAILED);
    evictShape->SetDimNum(1);
    evictShape->SetDim(0, 0);

    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    auto inputDataType = context->GetInputDataType(0);
    if (ge::GRAPH_SUCCESS != context->SetOutputDataType(0, ge::DT_INT64)) {
        return ge::GRAPH_FAILED;
    }
    if (ge::GRAPH_SUCCESS != context->SetOutputDataType(1, ge::DT_INT64)) {
        return ge::GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {

class ZeroCollisionHash : public OpDef {
public:
    explicit ZeroCollisionHash(const char* name) : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("identities")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Input("local_sizes")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});

        this->Input("offsets")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_INT64})
            .FormatList({ge::FORMAT_ND});

        this->Output("output")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        this->Output("evict_slots")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64, ge::DT_INT32})
            .FormatList({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});

        // 算子属性（与FBGEMM原始参数保持一致）
        this->Attr("max_probe", ge::AttrType::ATTR_INT, "128");
        this->Attr("circular_probe", ge::AttrType::ATTR_BOOL, "false");
        this->Attr("disable_fallback", ge::AttrType::ATTR_BOOL, "false");
        this->Attr("hash_identity", ge::AttrType::ATTR_INT, "1");
        this->Attr("opt_in_prob", ge::AttrType::ATTR_INT, "-1");
        this->Attr("num_reserved_slots", ge::AttrType::ATTR_INT, "-1");

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend950");  // 对应A5芯片
    }
};

OP_ADD(ZeroCollisionHash);

}  // namespace ops
```

### 3.4 SIMT Kernel实现 (op_kernel/simt_kernel.h)

```cpp
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SIMT_KERNEL_H
#define SIMT_KERNEL_H

#include "kernel_operator.h"
#include "simt_api/asc_simt.h"

using namespace AscendC;

constexpr int32_t MAX_THREADS_PER_BLOCK = 512;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t MAX_ELEMENTS_PER_THREAD = 4;
constexpr int64_t K_DEFAULT_TENSOR = -1;
constexpr int64_t K_MAX_IDENTITY_NUM = INT32_MAX;

namespace ZeroCollisionHashSimt {

// MurmurHash3 128位哈希函数 - 设备端实现
__aicore__ inline uint64_t MurmurHash3_2x64(uint64_t x, uint64_t y, uint64_t seed)
{
    const uint64_t c1 = 0x87c37b91114253d5ULL;
    const uint64_t c2 = 0x4cf5ad432745937fULL;
    
    uint64_t h1 = seed;
    uint64_t h2 = seed;
    
    // First 64-bit block
    uint64_t k1 = x;
    k1 *= c1;
    k1 = (k1 << 31) | (k1 >> (64 - 31));
    k1 *= c2;
    h1 ^= k1;
    h1 = (h1 << 27) | (h1 >> (64 - 27));
    h1 += h2;
    h1 = h1 * 5 + 0x52dce729ULL;
    
    // Second 64-bit block
    uint64_t k2 = y;
    k2 *= c2;
    k2 = (k2 << 33) | (k2 >> (64 - 33));
    k2 *= c1;
    h2 ^= k2;
    h2 = (h2 << 31) | (h2 >> (64 - 31));
    h2 += h1;
    h2 = h2 * 5 + 0x38495ab5ULL;
    
    // Finalization
    h1 ^= 16;
    h2 ^= 16;
    h1 += h2;
    h2 += h1;
    h1 ^= h1 >> 33;
    h1 *= 0xff51afd7ed558ccdULL;
    h1 ^= h1 >> 33;
    h1 *= 0xc4ceb9fe1a85ec53ULL;
    h1 ^= h1 >> 33;
    h2 ^= h2 >> 33;
    h2 *= 0xff51afd7ed558ccdULL;
    h2 ^= h2 >> 33;
    h2 *= 0xc4ceb9fe1a85ec53ULL;
    h2 ^= h2 >> 33;
    h1 += h2;
    h2 += h1;
    
    return h1 ^ h2;
}

// 计算下一个探测位置 - 循环探测
template <bool CIRCULAR_PROBE>
__aicore__ inline int64_t NextOutputIndex(
    int64_t outputIndex,
    int64_t modulo,
    int64_t& maxProbeLocal)
{
    if constexpr (CIRCULAR_PROBE) {
        return (outputIndex + 1) % modulo;
    } else {
        outputIndex = (outputIndex + 1) % modulo;
        if (outputIndex == 0) {
            maxProbeLocal = 0;  // 非循环探测，回到起点时退出
        }
        return outputIndex;
    }
}

// 只读模式下的哈希查找kernel
template <typename TInput, typename TIdentity, bool CIRCULAR_PROBE, bool DISABLE_FALLBACK, int32_t HASH_IDENTITY>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK)
inline void ZeroCollisionHashReadonlySimt(
    __gm__ TInput* input,
    __gm__ TIdentity* identities,
    __gm__ int64_t* output,
    __gm__ int64_t* localSizes,
    __gm__ int64_t* offsets,
    int64_t inputLength,
    int64_t modulo,
    int64_t maxProbe,
    int64_t optInProb,
    int64_t numReservedSlots,
    bool hasLocalSizes,
    bool hasOffsets)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    int32_t blockDim = AscendC::Simt::GetThreadNum<0>();
    int32_t gridDim = AscendC::Simt::GetBlockNum();
    
    // Stride loop模式
    for (int64_t processIndex = blockIdx * blockDim + threadIdx;
         processIndex < inputLength;
         processIndex += blockDim * gridDim) {
        
        TInput item = input[processIndex];
        
        // 获取当前元素的local_size和offset
        int64_t currentModulo = modulo;
        int64_t offset = 0;
        if (hasLocalSizes) {
            currentModulo = localSizes[processIndex];
        }
        if (hasOffsets) {
            offset = offsets[processIndex];
        }
        
        // 计算opt-in块大小
        int64_t optInBlockSize = (optInProb == -1) ? currentModulo : 
                                 (currentModulo - numReservedSlots);
        
        // 计算哈希值
        uint64_t hash = MurmurHash3_2x64(static_cast<uint64_t>(item), 0, 0);
        int64_t outputIndex = static_cast<int64_t>(hash % optInBlockSize);
        
        // 计算identity
        TIdentity identity;
        if constexpr (HASH_IDENTITY == 1) {
            identity = static_cast<TIdentity>(
                MurmurHash3_2x64(static_cast<uint64_t>(item), 0x17, 0) % K_MAX_IDENTITY_NUM);
        } else if constexpr (HASH_IDENTITY == 2) {
            identity = static_cast<TIdentity>(static_cast<uint64_t>(item) % K_MAX_IDENTITY_NUM);
        } else {
            identity = static_cast<TIdentity>(item);
        }
        
        // 探测查找
        int64_t maxProbeLocal = maxProbe;
        bool found = false;
        
        while (maxProbeLocal-- > 0) {
            int64_t insertIdx = outputIndex + offset;
            TIdentity currentSlotIdentity = identities[insertIdx];
            
            // 找到匹配的identity
            if (currentSlotIdentity == identity) {
                found = true;
                break;
            }
            
            // 遇到空槽，说明未找到（推理模式）
            if (currentSlotIdentity == static_cast<TIdentity>(K_DEFAULT_TENSOR)) {
                break;
            }
            
            outputIndex = NextOutputIndex<CIRCULAR_PROBE>(outputIndex, optInBlockSize, maxProbeLocal);
        }
        
        // 处理未找到的情况
        if (!found) {
            if constexpr (DISABLE_FALLBACK) {
                outputIndex = -1;
                offset = 0;
            } else {
                if (optInProb == -1) {
                    outputIndex = static_cast<int64_t>(hash % currentModulo);
                } else {
                    outputIndex = optInBlockSize + static_cast<int64_t>(hash % numReservedSlots);
                }
            }
        }
        
        output[processIndex] = outputIndex + offset;
    }
}

// 训练模式下的哈希插入kernel（支持原子操作）
template <typename TInput, typename TIdentity, bool CIRCULAR_PROBE, bool DISABLE_FALLBACK, int32_t HASH_IDENTITY>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREADS_PER_BLOCK)
inline void ZeroCollisionHashTrainSimt(
    __gm__ TInput* input,
    __gm__ TIdentity* identities,
    __gm__ int64_t* output,
    __gm__ int64_t* evictSlots,
    __gm__ int64_t* localSizes,
    __gm__ int64_t* offsets,
    int64_t inputLength,
    int64_t modulo,
    int64_t maxProbe,
    int64_t optInProb,
    int64_t numReservedSlots,
    bool hasLocalSizes,
    bool hasOffsets)
{
    int32_t threadIdx = AscendC::Simt::GetThreadIdx<0>();
    int32_t blockIdx = AscendC::Simt::GetBlockIdx();
    int32_t blockDim = AscendC::Simt::GetThreadNum<0>();
    int32_t gridDim = AscendC::Simt::GetBlockNum();
    
    for (int64_t processIndex = blockIdx * blockDim + threadIdx;
         processIndex < inputLength;
         processIndex += blockDim * gridDim) {
        
        TInput item = input[processIndex];
        
        int64_t currentModulo = modulo;
        int64_t offset = 0;
        if (hasLocalSizes) {
            currentModulo = localSizes[processIndex];
        }
        if (hasOffsets) {
            offset = offsets[processIndex];
        }
        
        int64_t optInBlockSize = (optInProb == -1) ? currentModulo : 
                                 (currentModulo - numReservedSlots);
        
        uint64_t hash = MurmurHash3_2x64(static_cast<uint64_t>(item), 0, 0);
        int64_t outputIndex = static_cast<int64_t>(hash % optInBlockSize);
        
        TIdentity identity;
        if constexpr (HASH_IDENTITY == 1) {
            identity = static_cast<TIdentity>(
                MurmurHash3_2x64(static_cast<uint64_t>(item), 0x17, 0) % K_MAX_IDENTITY_NUM);
        } else if constexpr (HASH_IDENTITY == 2) {
            identity = static_cast<TIdentity>(static_cast<uint64_t>(item) % K_MAX_IDENTITY_NUM);
        } else {
            identity = static_cast<TIdentity>(item);
        }
        
        int64_t maxProbeLocal = maxProbe;
        bool found = false;
        
        while (maxProbeLocal-- > 0) {
            int64_t insertIdx = outputIndex + offset;
            __gm__ TIdentity* slotPtr = &identities[insertIdx];
            
            // 原子比较交换尝试插入
            TIdentity oldValue = AscendC::Simt::AtomicCas(
                slotPtr,
                static_cast<TIdentity>(K_DEFAULT_TENSOR),
                identity);
            
            if (oldValue == identity) {
                // 已经存在此identity
                found = true;
                break;
            } else if (oldValue == static_cast<TIdentity>(K_DEFAULT_TENSOR)) {
                // 成功插入到空槽
                found = true;
                evictSlots[processIndex] = -1;  // 无淘汰
                break;
            }
            
            outputIndex = NextOutputIndex<CIRCULAR_PROBE>(outputIndex, optInBlockSize, maxProbeLocal);
        }
        
        if (!found) {
            if constexpr (DISABLE_FALLBACK) {
                outputIndex = -1;
                offset = 0;
            } else {
                if (optInProb == -1) {
                    outputIndex = static_cast<int64_t>(hash % currentModulo);
                } else {
                    outputIndex = optInBlockSize + static_cast<int64_t>(hash % numReservedSlots);
                }
            }
        }
        
        output[processIndex] = outputIndex + offset;
    }
}

// Kernel启动函数模板
template <typename TInput, typename TIdentity>
__aicore__ inline void LaunchZeroCollisionHashSimt(
    bool readonly,
    bool circularProbe,
    bool disableFallback,
    int32_t hashIdentity,
    __gm__ TInput* input,
    __gm__ TIdentity* identities,
    __gm__ int64_t* output,
    __gm__ int64_t* evictSlots,
    __gm__ int64_t* localSizes,
    __gm__ int64_t* offsets,
    int64_t inputLength,
    int64_t modulo,
    int64_t maxProbe,
    int64_t optInProb,
    int64_t numReservedSlots,
    bool hasLocalSizes,
    bool hasOffsets,
    int32_t threadsPerBlock)
{
    AscendC::Simt::Dim3 simtDim{static_cast<uint32_t>(threadsPerBlock), 1, 1};
    
    // 根据参数选择不同的kernel实例化
    // 这里简化为调用只读模式
    if (readonly) {
        if (circularProbe) {
            if (disableFallback) {
                if (hashIdentity == 1) {
                    AscendC::Simt::VF_CALL<ZeroCollisionHashReadonlySimt<TInput, TIdentity, true, true, 1>>(
                        simtDim, input, identities, output, localSizes, offsets,
                        inputLength, modulo, maxProbe, optInProb, numReservedSlots,
                        hasLocalSizes, hasOffsets);
                } else if (hashIdentity == 2) {
                    AscendC::Simt::VF_CALL<ZeroCollisionHashReadonlySimt<TInput, TIdentity, true, true, 2>>(
                        simtDim, input, identities, output, localSizes, offsets,
                        inputLength, modulo, maxProbe, optInProb, numReservedSlots,
                        hasLocalSizes, hasOffsets);
                } else {
                    AscendC::Simt::VF_CALL<ZeroCollisionHashReadonlySimt<TInput, TIdentity, true, true, 0>>(
                        simtDim, input, identities, output, localSizes, offsets,
                        inputLength, modulo, maxProbe, optInProb, numReservedSlots,
                        hasLocalSizes, hasOffsets);
                }
            } else {
                if (hashIdentity == 1) {
                    AscendC::Simt::VF_CALL<ZeroCollisionHashReadonlySimt<TInput, TIdentity, true, false, 1>>(
                        simtDim, input, identities, output, localSizes, offsets,
                        inputLength, modulo, maxProbe, optInProb, numReservedSlots,
                        hasLocalSizes, hasOffsets);
                } else if (hashIdentity == 2) {
                    AscendC::Simt::VF_CALL<ZeroCollisionHashReadonlySimt<TInput, TIdentity, true, false, 2>>(
                        simtDim, input, identities, output, localSizes, offsets,
                        inputLength, modulo, maxProbe, optInProb, numReservedSlots,
                        hasLocalSizes, hasOffsets);
                } else {
                    AscendC::Simt::VF_CALL<ZeroCollisionHashReadonlySimt<TInput, TIdentity, true, false, 0>>(
                        simtDim, input, identities, output, localSizes, offsets,
                        inputLength, modulo, maxProbe, optInProb, numReservedSlots,
                        hasLocalSizes, hasOffsets);
                }
            }
        } else {
            // 非循环探测模式
            if (disableFallback) {
                if (hashIdentity == 1) {
                    AscendC::Simt::VF_CALL<ZeroCollisionHashReadonlySimt<TInput, TIdentity, false, true, 1>>(
                        simtDim, input, identities, output, localSizes, offsets,
                        inputLength, modulo, maxProbe, optInProb, numReservedSlots,
                        hasLocalSizes, hasOffsets);
                } else if (hashIdentity == 2) {
                    AscendC::Simt::VF_CALL<ZeroCollisionHashReadonlySimt<TInput, TIdentity, false, true, 2>>(
                        simtDim, input, identities, output, localSizes, offsets,
                        inputLength, modulo, maxProbe, optInProb, numReservedSlots,
                        hasLocalSizes, hasOffsets);
                } else {
                    AscendC::Simt::VF_CALL<ZeroCollisionHashReadonlySimt<TInput, TIdentity, false, true, 0>>(
                        simtDim, input, identities, output, localSizes, offsets,
                        inputLength, modulo, maxProbe, optInProb, numReservedSlots,
                        hasLocalSizes, hasOffsets);
                }
            } else {
                if (hashIdentity == 1) {
                    AscendC::Simt::VF_CALL<ZeroCollisionHashReadonlySimt<TInput, TIdentity, false, false, 1>>(
                        simtDim, input, identities, output, localSizes, offsets,
                        inputLength, modulo, maxProbe, optInProb, numReservedSlots,
                        hasLocalSizes, hasOffsets);
                } else if (hashIdentity == 2) {
                    AscendC::Simt::VF_CALL<ZeroCollisionHashReadonlySimt<TInput, TIdentity, false, false, 2>>(
                        simtDim, input, identities, output, localSizes, offsets,
                        inputLength, modulo, maxProbe, optInProb, numReservedSlots,
                        hasLocalSizes, hasOffsets);
                } else {
                    AscendC::Simt::VF_CALL<ZeroCollisionHashReadonlySimt<TInput, TIdentity, false, false, 0>>(
                        simtDim, input, identities, output, localSizes, offsets,
                        inputLength, modulo, maxProbe, optInProb, numReservedSlots,
                        hasLocalSizes, hasOffsets);
                }
            }
        }
    } else {
        // 训练模式（类似结构）
        // ... 省略类似代码
    }
}

}  // namespace ZeroCollisionHashSimt

#endif  // SIMT_KERNEL_H
```

### 3.5 Kernel类定义 (op_kernel/zero_collision_hash_kernel.h)

```cpp
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ZERO_COLLISION_HASH_KERNEL_H
#define ZERO_COLLISION_HASH_KERNEL_H

#include "simt_kernel.h"
#include "kernel_common_utils.h"

struct Args {
    GM_ADDR input;
    GM_ADDR identities;
    GM_ADDR localSizes;
    GM_ADDR offsets;
    GM_ADDR output;
    GM_ADDR evictSlots;
    GM_ADDR tiling;
};

namespace ZeroCollisionHash {

constexpr int BUFFER_NUM = 2;

template <typename TInput, typename TIdentity>
class ZeroCollisionHashKernel {
public:
    __aicore__ inline ZeroCollisionHashKernel(Args& args)
    {
        GET_TILING_DATA(tilingData, args.tiling);
        InitTilingParams(tilingData);
        InitGmParams(args);
    }
    
    __aicore__ inline void Compute()
    {
        int32_t coreIdx = GetBlockIdx();
        
        // 计算当前核心处理的数据范围
        if (coreIdx < remainderBlocks) {
            blockCount = blocksPerCore + 1;
            blockStart = coreIdx * blockCount;
        } else {
            blockCount = blocksPerCore;
            blockStart = remainderBlocks * (blocksPerCore + 1) + 
                         (coreIdx - remainderBlocks) * blocksPerCore;
        }
        
        // 调用SIMT kernel
        ZeroCollisionHashSimt::LaunchZeroCollisionHashSimt<TInput, TIdentity>(
            true,    // readonly (推理模式)
            circularProbe,
            disableFallback,
            hashIdentity,
            reinterpret_cast<__gm__ TInput*>(inputPtr),
            reinterpret_cast<__gm__ TIdentity*>(identitiesPtr),
            reinterpret_cast<__gm__ int64_t*>(outputPtr),
            reinterpret_cast<__gm__ int64_t*>(evictSlotsPtr),
            reinterpret_cast<__gm__ int64_t*>(localSizesPtr),
            reinterpret_cast<__gm__ int64_t*>(offsetsPtr),
            inputLength,
            modulo,
            maxProbe,
            optInProb,
            numReservedSlots,
            hasLocalSizes,
            hasOffsets,
            elementsPerBlock);
    }
    
private:
    __aicore__ inline void InitTilingParams(const ZeroCollisionHashTilingData& tilingData)
    {
        inputLength = tilingData.inputLength;
        modulo = tilingData.modulo;
        maxProbe = tilingData.maxProbe;
        totalBlocks = tilingData.totalBlocks;
        blocksPerCore = tilingData.blocksPerCore;
        remainderBlocks = tilingData.remainderBlocks;
        elementsPerBlock = tilingData.elementsPerBlock;
        circularProbe = tilingData.circularProbe;
        disableFallback = tilingData.disableFallback;
        hashIdentity = tilingData.hashIdentity;
        hasLocalSizes = tilingData.hasLocalSizes;
        hasOffsets = tilingData.hasOffsets;
        optInProb = tilingData.optInProb;
        numReservedSlots = tilingData.numReservedSlots;
    }
    
    __aicore__ inline void InitGmParams(const Args& args)
    {
        inputPtr = args.input;
        identitiesPtr = args.identities;
        localSizesPtr = args.localSizes;
        offsetsPtr = args.offsets;
        outputPtr = args.output;
        evictSlotsPtr = args.evictSlots;
    }
    
private:
    // 全局内存指针
    __gm__ void* inputPtr;
    __gm__ void* identitiesPtr;
    __gm__ void* localSizesPtr;
    __gm__ void* offsetsPtr;
    __gm__ void* outputPtr;
    __gm__ void* evictSlotsPtr;
    
    // Tiling参数
    int64_t inputLength;
    int64_t modulo;
    int64_t maxProbe;
    int64_t totalBlocks;
    int64_t blocksPerCore;
    int32_t remainderBlocks;
    int32_t elementsPerBlock;
    bool circularProbe;
    bool disableFallback;
    int32_t hashIdentity;
    bool hasLocalSizes;
    bool hasOffsets;
    int64_t optInProb;
    int64_t numReservedSlots;
    
    // 运行时参数
    int64_t blockCount;
    int64_t blockStart;
};

}  // namespace ZeroCollisionHash

#endif  // ZERO_COLLISION_HASH_KERNEL_H
```

### 3.6 Kernel入口 (op_kernel/zero_collision_hash.cpp)

```cpp
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "zero_collision_hash_kernel.h"

extern "C" __global__ __aicore__ void zero_collision_hash(GM_ADDR input, GM_ADDR identities,
                                                          GM_ADDR localSizes, GM_ADDR offsets,
                                                          GM_ADDR output, GM_ADDR evictSlots,
                                                          GM_ADDR tiling)
{
    Args args;
    args.input = input;
    args.identities = identities;
    args.localSizes = localSizes;
    args.offsets = offsets;
    args.output = output;
    args.evictSlots = evictSlots;
    args.tiling = tiling;
    
    // 根据数据类型分发kernel
    GET_TILING_DATA(tilingData, tiling);
    auto inputDataType = tilingData.inputDataType;
    auto identityDataType = tilingData.identityDataType;
    
    if (inputDataType == ge::DT_INT64 && identityDataType == ge::DT_INT64) {
        ZeroCollisionHash::ZeroCollisionHashKernel<int64_t, int64_t> kernel(args);
        kernel.Compute();
    } else if (inputDataType == ge::DT_INT32 && identityDataType == ge::DT_INT32) {
        ZeroCollisionHash::ZeroCollisionHashKernel<int32_t, int32_t> kernel(args);
        kernel.Compute();
    } else if (inputDataType == ge::DT_INT64 && identityDataType == ge::DT_INT32) {
        ZeroCollisionHash::ZeroCollisionHashKernel<int64_t, int32_t> kernel(args);
        kernel.Compute();
    }
}
```

---

## 四、SIMT接口映射关系

本节基于 **fbgemm-ascend**、**RecSDK/cust_op** 和 **HierarchicalKV-ascend** 的实际代码，分析 `zero_collision_hash` 算子迁移所需的所有接口映射关系。

### 4.1 接口映射总览

| 映射状态 | 接口类别 | 说明 |
|---------|---------|------|
| ✅ **完全支持** | SIMT核心接口 | 线程索引、原子操作(CAS/Add/Exch)、同步、warp操作等 |
| ⚠️ **需要适配** | atomicMax | 需要用CAS循环实现，或用SetAtomicMax配合DataCopy |
| ✅ **框架层处理** | PyTorch/CUDA运行时 | 张量转换、Kernel启动、算子注册等 |

### 4.2 完全支持的SIMT接口

以下CUDA接口在昇腾SIMT中有直接对应的接口，可直接替换使用：

| CUDA接口 | 昇腾SIMT接口 | 功能说明 | 接口来源（实际使用项目） |
|---------|-------------|---------|------------------------|
| `threadIdx.x` | `AscendC::Simt::GetThreadIdx<0>()` | 获取线程索引 | RecSDK, HierarchicalKV |
| `blockIdx.x` | `AscendC::Simt::GetBlockIdx()` | 获取块索引 | RecSDK, HierarchicalKV |
| `blockDim.x` | `AscendC::Simt::GetThreadNum<0>()` | 获取块内线程数 | RecSDK, HierarchicalKV |
| `gridDim.x` | `AscendC::Simt::GetBlockNum()` | 获取块数量 | RecSDK, HierarchicalKV |
| `atomicCAS()` | `AscendC::Simt::AtomicCas()` | 原子比较交换（**支持64位**） | RecSDK, HierarchicalKV |
| `atomicExch()` | `AscendC::Simt::AtomicExch()` | 原子交换 | HierarchicalKV |
| `atomicAdd()` | `AscendC::Simt::AtomicAdd()` | 原子加 | RecSDK, HierarchicalKV |
| `__syncthreads()` | `AscendC::Simt::ThreadBarrier()` | 线程同步 | RecSDK |
| `__shfl()` | `__shfl()` | 线程束洗牌 | HierarchicalKV |
| `__shfl_xor()` | `__shfl_xor()` | 线程束异或洗牌 | HierarchicalKV |
| `WarpReduceAddSync()` | `AscendC::Simt::WarpReduceAddSync()` | 线程束归约加法 | RecSDK |
| `WarpShflUpSync()` | `AscendC::Simt::WarpShflUpSync()` | 线程束向上洗牌 | RecSDK |
| - | `AscendC::Simt::VF_CALL<Kernel>()` | 启动SIMT kernel | RecSDK, HierarchicalKV |
| - | `AscendC::Simt::Dim3{x, y, z}` | 定义线程维度 | RecSDK, HierarchicalKV |
| - | `__simt_vf__` | 声明SIMT函数 | RecSDK, HierarchicalKV |

**重要说明 - 64位AtomicCas已支持**：

GPU版本中使用的 `int64_t CAS` 不需要特殊处理。HierarchicalKV项目已验证 `AtomicCas` 支持 `uint64_t` 类型：

```cpp
// HierarchicalKV 实际代码 (insert_or_assign_kernel.h)
template <typename K = uint64_t, typename V = float, typename S = uint64_t>
// K 默认为 uint64_t，直接用于 AtomicCas
auto try_key = Simt::AtomicCas(current_key_ptr, key, static_cast<K>(LOCKED_KEY));
```

### 4.3 需要适配的接口

#### 4.3.1 atomicMax - 唯一需要适配的SIMT接口

**GPU使用位置**：`faster_hash.cu:63` - 更新metadata时间戳

```cpp
template <>
__device__ __inline__ void update_metadata<1>(
    int32_t* metadata,
    int64_t output_index,
    int32_t metadata_val) {
  atomicMax(metadata + output_index, metadata_val);  // 更新最后访问时间戳
}
```

**适配方案一：CAS循环实现（推荐，通用方案）**

```cpp
__aicore__ inline void AtomicMaxInt32(__gm__ int32_t* addr, int32_t value) {
    int32_t old = *addr;
    while (value > old) {
        int32_t expected = old;
        old = AscendC::Simt::AtomicCas(addr, expected, value);
        if (old == expected) break;  // CAS成功
    }
}
```

**适配方案二：SetAtomicMax + DataCopy（仅支持int8_t）**

RecSDK中的实际使用案例（`backward_codegen_unweighted_exact_kernel.h:146`）：

```cpp
SetAtomicMax<int8_t>();
DataCopy(this->updateMaskGT_[thisIndForTotalTable], newFlagOutLt, FLAG_LEN);
SetAtomicNone();
```

> ⚠️ **限制**：`SetAtomicMax<T>()` 仅支持 `int8_t` 等有限类型，不适用于 `int32_t` 的metadata更新场景。

### 4.4 框架层接口（非SIMT，方案明确）

以下接口不属于SIMT范畴，在框架层处理，参考项目中有明确的适配模式：

#### 一、PyTorch张量到ACL张量转换

| GPU接口 | 昂腾适配方案 | 参考来源 |
|---------|------------|---------|
| `at::Tensor` | `ConvertType(const at::Tensor&)` → `aclTensor*` | RecSDK: `pytorch_npu_helper.hpp` |
| `at::PackedTensorAccessor64` | 使用 `__gm__ T*` 原始指针 | HierarchicalKV: 所有kernel |
| `.data_ptr<T>()` | `DeviceTensor::get_data()` | HierarchicalKV: `aclnn_helper.h` |

**RecSDK 实际代码**：
```cpp
inline aclTensor* ConvertType(const at::Tensor& at_tensor) {
    auto acl_tensor = aclCreateTensor(
        at_tensor.sizes().data(), at_tensor.sizes().size(), 
        acl_data_type, at_tensor.strides().data(),
        at_tensor.storage_offset(), format, storageDims.data(), 
        storageDims.size(), const_cast<void*>(at_tensor.storage().data()));
    return acl_tensor;
}
```

#### 二、CUDA Kernel启动方式

| GPU方式 | 昂腾适配方案 | 参考来源 |
|--------|------------|---------|
| `<<<grid, block, shared_mem, stream>>>` | `Simt::VF_CALL<Kernel>(Dim3{...}, args...)` | HierarchicalKV |
| 自定义kernel | `extern "C" __global__ __aicore__ void kernel_name(...)` | RecSDK, HierarchicalKV |
| ACL NN算子 | `EXEC_NPU_CMD(aclnn_api, ...)` | RecSDK |

**HierarchicalKV 实际代码**：
```cpp
extern "C" __global__ __aicore__ void find_ptr_kernel(...) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
  Simt::VF_CALL<find_ptr_kernel_vf<...>>(
      Simt::Dim3{static_cast<uint32_t>(THREAD_NUM), 1, 1}, ...);
}
```

#### 三、PyTorch算子注册机制

| GPU接口 | 昂腾适配方案 | 说明 |
|---------|------------|------|
| `TORCH_LIBRARY_FRAGMENT` | **保持不变** | 与GPU相同 |
| `c10::DispatchKey::CUDA` | `c10::DispatchKey::Autograd` | NPU设备自动处理 |

**RecSDK 实际代码**：
```cpp
TORCH_LIBRARY_FRAGMENT(fbgemm, m) {
    m.def("split_embedding_codegen_forward_unweighted_cuda(...)");
    m.impl("split_embedding_codegen_forward_unweighted_cuda",
        torch::dispatch(c10::DispatchKey::Autograd,
                        TORCH_FN(fbgemm_npu_lookups::split_embedding_codegen_forward_unweighted_npu)));
}
```

#### 四、CUDA运行时接口

| GPU接口 | 昂腾适配方案 |
|---------|------------|
| `at::cuda::getCurrentCUDAStream()` | `c10_npu::getCurrentNPUStream().stream(false)` |
| `at::cuda::getCurrentDeviceProperties()` | `platform_ascendc::PlatformAscendC::GetCoreNumAiv()` |
| `cudaMalloc/cudaFree` | `aclrtMalloc/aclrtFree` |
| `cudaMemcpy` | `aclrtMemcpy(..., ACL_MEMCPY_DEVICE_TO_DEVICE)` |

### 4.5 接口适配状态总结

| 状态 | 接口 | 数量 | 说明 |
|-----|------|-----|------|
| ✅ 直接替换 | 线程索引、原子操作(CAS/Add/Exch)、同步、warp操作 | 14个 | 有直接对应的SIMT接口 |
| ⚠️ 需要适配 | atomicMax | 1个 | 用CAS循环实现 |
| ✅ 框架层处理 | 张量转换、Kernel启动、算子注册、运行时接口 | 多个 | 参考项目有明确模式 |

**结论**：`zero_collision_hash` 算子迁移所需的接口全部可以找到对应映射，**不存在完全不支持的接口**。唯一需要自行实现的是 `atomicMax`，可通过CAS循环简单实现。

### 4.6 其他发现的昇腾特有接口

| 接口 | 功能说明 | 使用位置 | 备注 |
|-----|---------|---------|------|
| `asc_atomic_add()` | UB（Unified Buffer）上的原子加，仅支持int32_t | RecSDK: `block_bucketize_sparse_features_kernel.h:571,719` | 用于本地缓冲区的原子计数，可替代部分atomicAdd场景 |
| `SetAtomicMax<T>()` | 设置全局内存原子Max模式 | RecSDK: `backward_codegen_unweighted_exact_kernel.h:146` | 配合DataCopy使用，仅支持int8_t等有限类型 |

---

## 五、编译构建与测试验证

### 5.1 构建系统说明

fbgemm-ascend 仓库采用统一的构建系统：

- **AscendC算子编译**：通过 `FbgemmAscend.cmake` 统一管理，使用 `msopgen` 生成代码 + `run.sh` 编译
- **C++适配层编译**：通过顶层 `CMakeLists.txt` 编译为 `.so` 库
- **Python包构建**：通过 `setup.py`（scikit-build）生成 `whl` 包

整体流程见上方第 2.4 节"迁移集成步骤"。

### 5.2 CMakeLists.txt 示例

```cmake
cmake_minimum_required(VERSION 3.16)
project(zero_collision_hash)

set(CMAKE_CXX_STANDARD 17)

# Ascend路径
set(ASCEND_HOME /usr/local/Ascend)
set(ASCEND_TOOLKIT_HOME ${ASCEND_HOME}/ascend-toolkit/latest)

# 包含目录
include_directories(${ASCEND_TOOLKIT_HOME}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/op_kernel)

# 源文件
set(KERNEL_SRCS
    op_kernel/zero_collision_hash.cpp
)

# 编译算子库
add_library(zero_collision_hash SHARED ${KERNEL_SRCS})

# 链接库
target_link_libraries(zero_collision_hash
    ${ASCEND_TOOLKIT_HOME}/lib/libascendcl.so
)
```

---

## 六、性能优化建议

### 6.1 内存访问优化

1. **合并内存访问**：确保相邻线程访问相邻内存地址
2. **缓存利用**：使用`__ldg`和`__stg`带缓存提示的访问
3. **数据预取**：对identities表进行预取优化

### 6.2 并行度优化

1. **线程块大小**：建议使用512线程/块，平衡占用率和资源
2. **Stride Loop**：使用grid-stride loop处理变长数据
3. **负载均衡**：合理分配数据到各核心

### 6.3 哈希表优化

1. **探测长度**：根据冲突率调整maxProbe参数
2. **表大小**：选择合适的哈希表大小，避免过度冲突
3. **内存布局**：优化identities表的内存布局

---

## 七、测试验证

### 7.1 测试策略

根据迁移指引，**zero_collision_hash的测试用例必须复用FBGEMM/fbgemm_gpu/test测试目录对应的测试用例**。测试用例通过后，认定迁移成功。

**参考位置**：
- **FBGEMM_GPU测试目录**: `FBGEMM/fbgemm_gpu/test/quantize_ops/` 或相关测试模块
- **原GPU算子测试**: 参考`test_zero_collision_hash*.py`（如果存在）或类似hash算子的测试

**测试流程**：
1. 将原GPU版本的测试用例适配到昇腾NPU环境
2. 修改设备为`npu`，并确保算子调用路径指向昇腾版本
3. 运行测试用例，验证功能正确性
4. 对比GPU版本输出，确保结果一致（在允许的误差范围内）

### 7.2 适配现有测试用例

在原FBGEMM_GPU测试代码基础上，主要修改点：

```python
import torch
import torch_npu  # 昇腾PyTorch扩展

# 原始GPU测试（FBGEMM/fbgemm_gpu/test/...）
# 修改为昇腾版本：
# 1. device='cuda' → device='npu'
# 2. torch.ops.fbgemm_gpu.xxx → torch.ops.fbgemm.xxx（若算子注册名一致）
# 3. 如有CUDA特有操作，需要替换为NPU等效操作

def test_zero_collision_hash_basic_npu():
    """基础功能测试 - 适配昇腾"""
    input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64, device='npu')
    identities = torch.full((1000, 1), -1, dtype=torch.int64, device='npu')

    # 调用昇腾算子
    output, evict_slots = torch.ops.fbgemm.zero_collision_hash(
        input_ids, identities, max_probe=128, readonly=True
    )

    print(f"Output: {output}")
    print(f"Evict slots: {evict_slots}")
    assert output.shape == input_ids.shape
    print("Basic NPU test passed!")

def test_zero_collision_hash_collision_npu():
    """冲突处理测试 - 适配昇腾"""
    input_ids = torch.tensor([1, 1001, 2001, 3001], dtype=torch.int64, device='npu')
    identities = torch.full((500, 1), -1, dtype=torch.int64, device='npu')

    output, _ = torch.ops.fbgemm.zero_collision_hash(
        input_ids, identities, max_probe=128, readonly=True
    )

    # 验证输出索引不重复
    unique_outputs = torch.unique(output)
    assert len(unique_outputs) == len(output), "Collision not handled properly"
    print("Collision NPU test passed!")

if __name__ == "__main__":
    test_zero_collision_hash_basic_npu()
    test_zero_collision_hash_collision_npu()
    print("All NPU tests passed!")
```

### 7.3 运行测试

```bash
# 进入fbgemm_gpu测试目录
cd FBGEMM/fbgemm_gpu/test

# 运行pytest（适配昇腾环境）
pytest test_quantize_ops/test_zero_collision_hash.py -v  # 如果存在独立测试文件
# 或
pytest test/ -k zero_collision_hash -v

# 单个测试文件
python test_zero_collision_hash.py
```

**验收标准**：
- 所有FBGEMM_GPU中原有的zero_collision_hash测试用例在昇腾NPU上通过
- 功能结果与GPU版本一致（考虑数据类型和精度差异）
- 无崩溃、数据越界等严重问题

---

### 7.2 性能基准测试

```python
import time
import torch

def benchmark_zero_collision_hash(batch_size=100000, table_size=1000000, iterations=10):
    """性能基准测试"""
    input_ids = torch.randint(0, table_size, (batch_size,), dtype=torch.int64, device='npu')
    identities = torch.full((table_size, 1), -1, dtype=torch.int64, device='npu')
    
    # 预热
    for _ in range(3):
        _ = torch.ops.fbgemm.zero_collision_hash(input_ids, identities, max_probe=128, readonly=True)
    
    torch.npu.synchronize()
    start = time.time()
    
    for _ in range(iterations):
        output, _ = torch.ops.fbgemm.zero_collision_hash(input_ids, identities, max_probe=128, readonly=True)
    
    torch.npu.synchronize()
    end = time.time()
    
    avg_time = (end - start) / iterations
    throughput = batch_size / avg_time
    
    print(f"Batch size: {batch_size}")
    print(f"Avg time: {avg_time*1000:.2f} ms")
    print(f"Throughput: {throughput:.0f} IDs/sec")

if __name__ == "__main__":
    benchmark_zero_collision_hash()
```

---

## 八、非SIMT接口（A2/A3芯片适配）

### 8.2 非SIMT API开发要点| 芯片代际 | 芯片型号 | SIMT支持 | 推荐实现方式 |
|---------|---------|---------|-------------|
| **A5代际** | c310 (950PR) | ✅ 支持 | 本方案主推的SIMT实现 |
| **A2代际** | c200 (310P) | ❌ 不支持 | 使用非SIMT接口（Stream/Vector融合） |
| **A3代际** | c250 (510) | ❌ 不支持 | 使用非SIMT接口（Stream/Vector融合） |

### 8.2 非SIMT API开发要点

对于A2/A3芯片，需要使用传统的**Stream（流多线程）+ Vector（向量化）**编程模型，而非SIMT（单指令多线程）模型。

**核心接口差异**：

| 功能 | SIMT (A5) | 非SIMT (A2/A3) |
|-----|-----------|---------------|
| 线程模型 | CUDA-like SIMT | Stream + Vector |
| 线程同步 | `AscendC::Simt::ThreadBarrier()` | `AscendC::PipeBarrier()` |
| 原子操作 | `AscendC::Simt::AtomicCas()` | `AscendC::AtomicAdd()` 等 |
| Kernel启动 | `Simt::VF_CALL<Kernel>()` | `Kernel::template Process()` |
| 线程索引 | `GetThreadIdx<0>()` | `GetBlockIdx()` + `GetThreadIdx()` |

**非SIMT编程模式**（参考文档）：
```cpp
// A2/A3非SIMT核函数结构
__aicore__ inline void ZeroCollisionHashKernel() {
    // 1. 获取线程/Stream索引
    int32_t blockIdx = GetBlockIdx();
    int32_t threadIdx = GetThreadIdx();

    // 2. 使用Pipe进行数据搬运（GM->UB, UB->GM）
    // ...

    // 3. 使用Vector指令进行并行计算
    // ...

    // 4. 使用PipeBarrier进行同步
    PipeBarrier(PIPE_MTE3);
}

// Host端启动（非SIMT）
extern "C" __global__ __aicore__ void zero_collision_hash(...) {
    // 初始化参数
    KernelArgs args = { ... };

    // 启动Stream内核
    ZeroCollisionHashKernel args;
}
```

### 8.3 非SIMT开发文档

详细开发指南请参考昇腾官方文档：

1. **AscendC算子开发（通用 - A2/A3）**
   - 文档地址: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html
   - 内容：A2/A3芯片的非SIMT算子开发全流程，包括数据搬运、计算核、内存管理等

2. **AscendC API参考（非SIMT）**
   - 文档地址: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0003.html
   - 内容：非SIMT接口的详细API说明，包括Pipe、DataCopy、Vector指令等

### 8.4 A2/A3适配建议

如果需要在A2/A3芯片上支持zero_collision_hash：

1. **重新实现kernel**：基于非SIMT编程模型重写`simt_kernel.h`中的计算核心
2. **数据搬运优化**：使用`Pipe/DataCopy`高效搬运数据到UB（Unified Buffer）
3. **向量化计算**：将探测逻辑向量化，每个Vector处理多个元素
4. **原子操作适配**：使用`AtomicAdd/AtomicCas`等非SIMT原子接口（注意A2/A3的原子操作支持能力可能不同）

鉴于A5芯片（c310）已经支持SIMT，**推荐优先在A5平台上部署**，A2/A3平台暂时不支持。

---

## 九、关键实现要点

### 8.1 MurmurHash3哈希函数移植

**原始CUDA实现**：
```cpp
__device__ __host__ __inline__ uint64_t murmur_hash3_2x64(uint64_t x, uint64_t y, uint64_t seed)
```

**昇腾适配要点**：
- 使用`__aicore__`宏替代`__device__ __host__`
- 哈希常量`c1=0x87c37b91114253d5`, `c2=0x4cf5ad432745937f`保持不变
- 位旋转操作需要使用标准C++实现，昇腾支持完整的位运算

**参考来源**：原始实现在`fbgemm_gpu/include/fbgemm_gpu/faster_hash_ops/common_utils.cuh`

### 9.2 探测模式模板化
**两种探测模式**：
| 模式 | CUDA实现 | 昇腾SIMT实现 |
|-----|---------|-------------|
| 循环探测 | `next_output_index<true>()` | 模板参数`CIRCULAR_PROBE=true` |
| 非循环探测 | `next_output_index<false>()` | 模板参数`CIRCULAR_PROBE=false` |

**实现要点**：
```cpp
// 非循环探测：到达末尾后设置max_probe_local=0退出
if constexpr (!CIRCULAR_PROBE) {
    if (output_index == 0) max_probe_local = 0;
}
// 循环探测：自动回绕
output_index = (output_index + 1) % modulo;
```

**参考来源**：RecSDK中asynchronous_cumsum使用类似的模板参数控制模式切换

### 8.3 原子操作并发控制

**核心原子操作使用**：

| 操作场景 | CUDA接口 | 昇腾SIMT接口 | 使用算子参考 |
|---------|---------|-------------|-------------|
| 键值槽位抢占 | `atomicCAS(&identities[idx], expected, desired)` | `AscendC::Simt::AtomicCas(ptr, expected, desired)` | HierarchicalKV: insert_or_assign<br>RecSDK: backward_codegen_dedup |
| 更新桶大小计数 | `atomicAdd(bucket_size, 1)` | `AscendC::Simt::AtomicAdd(ptr, val)` | HierarchicalKV: insert_or_assign |
| 释放锁/写入最终值 | `atomicExch(ptr, val)` | `AscendC::Simt::AtomicExch(ptr, val)` | HierarchicalKV: insert_or_assign |

**关键代码模式**：
```cpp
// 推理模式：只读检查，不使用原子操作
if constexpr (READONLY) {
    old_value = *identities_slot;
    return (old_value == identity);
}
// 训练模式：使用原子CAS抢占槽位
else {
    old_value = AscendC::Simt::AtomicCas(identities_slot, EMPTY_KEY, identity);
    return (old_value == EMPTY_KEY || old_value == identity);
}
```

### 8.4 类型泛化与模板实例化

**支持的类型组合**：
| 输入类型 | Identity类型 | hash_identity值 |
|---------|-------------|----------------|
| int64_t | int64_t | 0 (直接使用原值) |
| int32_t | int32_t | 0 |
| int64_t | int32_t | 1 (MurmurHash) 或 2 (取模) |

**模板参数设计**：
```cpp
template<
    typename TInput,           // 输入ID类型
    typename TIdentity,        // Identity类型
    bool CIRCULAR_PROBE,       // 探测模式
    bool DISABLE_FALLBACK,     // 是否禁用回退
    int32_t HASH_IDENTITY      // hash计算方式
>
__simt_vf__ __aicore__ void ZeroCollisionHashSimt(...) { ... }
```

**参考来源**：RecSDK中split_embedding使用类似的多模板参数设计处理不同pooling模式

### 8.5 内存访问优化

**昇腾特有优化**：

| 优化技术 | 接口 | 适用场景 | 来源 |
|---------|------|---------|------|
| 带缓存提示加载 | `__ldg<L2_CACHE_HINT, L1_CACHE_TYPE>(ptr)` | 高频读取的identities表 | HierarchicalKV: insert_or_assign |
| 带缓存提示存储 | `__stg<L2_CACHE_HINT, L1_CACHE_TYPE>(ptr, val)` | 写入输出结果 | HierarchicalKV: insert_or_assign |
| 向量加载 | `reinterpret_cast<__gm__ float4*>(ptr)[idx]` | 批量数据搬运 | RecSDK: split_embedding |

**示例代码**：
```cpp
// 带L2缓存提示的加载
auto current_key = __ldg<LD_L2CacheType::L2_CACHE_HINT_NORMAL_FV,
                         L1CacheType::NON_CACHEABLE>(key_ptr);
```

### 8.6 线程束级操作

**Warp操作用于分治计算**：

| 操作 | 用途 | 使用算子 |
|-----|------|---------|
| `__shfl(var, srcLane, width)` | 从指定线程获取变量值 | HierarchicalKV: insert_or_assign（组内同步状态） |
| `__shfl_xor(var, laneMask, width)` | 异或洗牌，用于分治归约 | HierarchicalKV: insert_or_assign（并行求最小值） |
| `AscendC::Simt::WarpReduceAddSync(val)` | 线程束内归约求和 | RecSDK: split_embedding |
| `AscendC::Simt::WarpShflUpSync(val, offset)` | 向上洗牌，用于前缀和 | RecSDK: asynchronous_cumsum |

**分治求最小值示例**（来自HierarchicalKV）：
```cpp
// 分治法求最小值，最终所有线程获得相同的min_score和min_pos
for (int32_t offset = GROUP_SIZE / 2; offset > 0; offset /= 2) {
    S other_score = __shfl_xor(min_score, offset, GROUP_SIZE);
    uint32_t other_pos = __shfl_xor(min_pos, offset, GROUP_SIZE);
    if (other_score < min_score) {
        min_score = other_score;
        min_pos = other_pos;
    }
}
```

---

## 十、总结

本文档详细描述了将FBGEMM的`zero_collision_hash`算子迁移到昇腾NPU平台的完整方案，参照**fbgemm-ascend**代码仓的架构和构建方式：

1. **目录结构**：遵循**fbgemm-ascend**的标准结构（c310目录用于A5芯片）
2. **SIMT实现**：使用`__simt_vf__`声明和`VF_CALL`启动kernel（适用于c310/A5）
3. **原子操作**：使用`AscendC::Simt::AtomicCas`等接口实现并发控制
4. **框架注册**：通过OpDef类完成算子注册和形状推导
5. **测试策略**：复用`FBGEMM/fbgemm_gpu/test`中的测试用例
6. **芯片适配**：c310（A5 950PR）为SIMT实现；如需支持A2/A3，需使用非SIMT接口

**关键参考项目**：
- **fbgemm-ascend**: https://gitcode.com/Ascend/fbgemm-ascend（参考架构和构建方式）
- **HierarchicalKV-ascend**: https://gitcode.com/Ascend/HierarchicalKV-ascend（参考SIMT实现细节）
- **A2/A3非SIMT文档**: 官方AscendC开发文档（见第8.3节）

迁移后的算子能够充分利用昇腾NPU的SIMT并行能力，保持与原始CUDA版本相同的功能语义。通过复用FBGEMM_GPU的测试用例进行验证，确保功能正确性。

---

## 附录

### 附录A：CMake Presets 配置示例

在 `fbgemm-ascend/c310/` 目录下可以使用 `CMakePresets.json` 简化构建：

```json
{
  "version": 1,
  "configurePresets": [
    {
      "name": "c310-default",
      "generator": "Unix Makefiles",
      "cacheVariables": {
        "CMAKE_CXX_COMPILER": "/usr/local/Ascend/ascend-toolkit/latest/bin/ccec",
        "CMAKE_C_COMPILER": "/usr/local/Ascend/ascend-toolkit/latest/bin/ccec",
        "CMAKE_BUILD_TYPE": "Release"
      }
    }
  ]
}
```

---

## 参考文献

### 项目仓库
- **fbgemm-ascend**: https://gitcode.com/Ascend/fbgemm-ascend
  - **目标仓库**，zero_collision_hash 算子最终适配的位置
  - 标准架构：`c310/ai_core_op/<op_name>/`

- **HierarchicalKV-ascend**: https://gitcode.com/Ascend/HierarchicalKV-ascend
  - SIMT kernel实现的详细样例
  - 包含原子操作、warp操作等高级特性

- **RecSDK/cust_op**: 昇腾自定义算子开发SDK
  - 提供算子开发的通用模式和最佳实践

### 官方文档
- **AscendC算子开发（A2/A3非SIMT）**: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html
  - A2/A3芯片的非SIMT算子开发指南

- **AscendC API参考**: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/API/ascendcopapi/atlasascendc_api_07_0003.html
  - 非SIMT接口的详细API说明

### 测试用例
- **FBGEMM_GPU测试目录**: `FBGEMM/fbgemm_gpu/test/`
  - 迁移测试用例的来源位置
  - 包含zero_collision_hash相关的功能测试
