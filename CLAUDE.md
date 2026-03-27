# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

FBGEMM（Facebook GEneral Matrix Multiplication）是一个高性能深度学习推理和训练库，包含三个主要组件：

1. **FBGEMM**: CPU端低精度矩阵乘法和卷积库（x86/ARM）
2. **FBGEMM_GPU**: GPU端高性能算子库，专注推荐系统
3. **FBGEMM_GenAI**: 面向生成式AI（Llama 3/4）的GPU算子库

- **当前版本**: FBGEMM_GPU 1.5.0, FBGEMM 1.4.0
- **许可证**: BSD-3-Clause
- **主仓库**: https://github.com/pytorch/FBGEMM
- **在线文档**: https://pytorch.org/FBGEMM

## 代码架构

### 核心组件

**FBGEMM (CPU)**
- 矩阵乘法: `src/Fbgemm*.cc`（通用、FP16、稀疏）
- 量化: `src/QuantUtils*.cc`（AVX2/AVX512/NEON优化）
- 嵌入: `src/EmbeddingSpMDM*.cc`（稀疏矩阵-稠密乘法）
- 卷积: `src/FbgemmI8Depthwise*.cc`, `src/GroupwiseConv*.cc`
- 打包: `src/Pack*.cc`（矩阵打包优化）

**FBGEMM_GPU**
- 量化算子: `fbgemm_gpu/src/quantize_ops/`
- 稀疏算子: `fbgemm_gpu/src/sparse_ops/`
- 锯齿张量: `fbgemm_gpu/src/jagged_tensor_ops/`
- 表批嵌入(TBE): `fbgemm_gpu/src/tbe/`
- 布局转换: `fbgemm_gpu/src/layout_transform_ops/`
- 代码生成器: `fbgemm_gpu/codegen/genscript/` (Jinja2模板)

**实验性特性**
- GenAI: `fbgemm_gpu/experimental/gen_ai/` (FP8/INT4量化、MoE)
- HSTU: `fbgemm_gpu/experimental/hstu/` (层次序列转换单元)

### 架构设计模式

- **编译时分发**: 通过CMake为不同指令集构建变体（AVX2/AVX512/NEON/SVE）
- **运行时分发**: 基于CPU/GPU能力动态选择内核
- **融合算子**: 减少内存访问，提升性能
- **多后端支持**: CUTLASS、Composable Kernel、Triton（GenAI）

## 常用开发命令

### 环境准备

```bash
# 初始化子模块（必需）
git submodule update --init --recursive
```

### FBGEMM (CPU库) 构建与测试

```bash
# 配置构建
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DFBGEMM_BUILD_TESTS=ON -DFBGEMM_BUILD_BENCHMARKS=ON

# 编译
make -j$(nproc)

# 运行测试
ctest -j$(nproc) --output-on-failure

# 运行单个测试
./test/EmbeddingSpMDMTest --gtest_filter=TestName
./test/QuantUtilsTest

# 运行基准测试
./bench/run_benchmarks
./bench/GEMMsBenchmark --benchmark_min_time=1.0
```

**CMake选项**:
- `-DFBGEMM_BUILD_TESTS=ON/OFF` (默认ON)
- `-DFBGEMM_BUILD_BENCHMARKS=ON/OFF` (默认ON)
- `-DFBGEMM_BUILD_DOCS=ON/OFF` (默认OFF)
- `-DFBGEMM_USE_IPO=ON/OFF` (默认OFF, 过程间优化)
- `-DCMAKE_BUILD_TYPE=Release/Debug/RelWithDebInfo`

### FBGEMM_GPU (GPU库) 构建与测试

```bash
cd fbgemm_gpu

# 安装依赖
pip install -r requirements.txt

# 开发模式安装（推荐用于开发）
pip install -e . --no-build-isolation

# 或使用setup.py
python setup.py develop

# 安装时指定构建目标
python setup.py install --build-target=genai  # GenAI算子
python setup.py install --build-target=hstu   # HSTU算子
python setup.py install --build-target=default # 默认

# 指定构建变体
python setup.py install --build-variant=cuda  # CUDA (默认)
python setup.py install --build-variant=rocm  # ROCm
python setup.py install --build-variant=cpu   # 仅CPU

# 指定CUDA架构（CUDA变体）
TORCH_CUDA_ARCH_LIST="8.0 9.0 10.0" python setup.py install

# Dry-run查看构建信息
python setup.py install --dryrun --verbose
```

**Python包导入**:
```python
import fbgemm_gpu
# 或特定模块
from fbgemm_gpu import quantize_ops, sparse_ops, tbe
```

**运行Python测试**:
```bash
cd fbgemm_gpu

# 使用pytest
pytest test/quantize/
pytest test/sparse/
pytest test/tbe/

# 运行单个测试文件
pytest test/quantize/test_quantize_ops.py -v

# 带详细输出
pytest test/ -vv --tb=short
```

### 代码生成（TBE）

```bash
cd fbgemm_gpu/codegen/genscript

# 生成前向量化算子
python generate_forward_quantized.py

# 生成前向/后向分割算子
python generate_forward_split.py
python generate_backward_split.py

# 生成优化器
python generate_embedding_optimizer.py

# 生成的代码输出到 fbgemm_gpu/src/tbe/
```

### Lint与静态分析

```bash
# CUDA HIPify检查（ROCm构建前）
python -m torch.utils.hipify.hipify --hipify_skip_files=".*test.*" --ci

# 使用clang-tidy（需要先构建）
cd build
clang-tidy -p . ../src/*.cc
```

### 文档构建

```bash
cd docs
pip install -r requirements.txt
make html
# 输出: _build/html/index.html
```

## 关键配置文件

- **CMakeLists.txt** (根目录): FBGEMM主构建文件
- **FbgemmGpu.cmake** (fbgemm_gpu/): FBGEMM_GPU库定义
- **setup.py** (fbbgemm_gpu/): Python包构建脚本
- **.bazelrc**, **BUILD.bazel**: Bazel构建（Facebook内部）
- **.clang-tidy**: 代码静态分析规则
- **cmake/modules/**: CMake模块
  - `CudaSetup.cmake`: CUDA配置
  - `RocmSetup.cmake`: ROCm/HIP配置
  - `PyTorchSetup.cmake`: PyTorch依赖定位
  - `FindAVX.cmake`: AVX指令集检测

## 开发工作流

### 添加新算子到FBGEMM_GPU

1. **实现C++/CUDA代码**
   - CPU: `fbgemm_gpu/src/<category>/new_op_cpu.cpp`
   - GPU: `fbgemm_gpu/src/<category>/new_op.cu`
   - 头文件: `fbgemm_gpu/src/<category>/new_op.h`

2. **注册到PyBind**
   - 在对应`.cc`文件中添加`PYBIND11_MODULE`绑定
   - 或修改现有绑定文件

3. **添加测试**
   - Python测试: `fbgemm_gpu/test/<category>/test_new_op.py`
   - C++测试（如需要）: `test/NewOpTest.cc`

4. **更新构建系统**
   - `fbgemm_gpu/FbgemmGpu.cmake`: 添加源文件到`gpu_cpp_library()`

5. **更新文档**
   - 相关README
   - API文档（Doxygen注释）

### 常见问题排查

**CUDA版本不匹配**:
```bash
nvcc --version  # 检查CUDA版本
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA版本
# 确保两者兼容
```

**NVML未找到**:
```bash
python setup.py install --nvml_lib_path /usr/lib/x86_64-linux-gnu/libnvidia-ml.so
```

**ROCm环境**:
```bash
# 安装tbb
conda install -c conda-forge tbb
# 指定HIP根目录
python setup.py install --build-variant=rocm -DHIP_ROOT_DIR=/opt/rocm
```

**内存不足**:
- 减少并行度: `export CMAKE_BUILD_PARALLEL_LEVEL=4`
- setup.py自动设置并行度

## 测试策略

- **新功能必须配套测试**
- C++使用Google Test框架
- Python使用pytest + unittest
- 测试文件位置:
  - C++: `test/*Test.cc`
  - Python: `fbbgemm_gpu/test/<category>/`
- CI覆盖多平台（Linux/macOS/Windows）、多配置

## 代码风格

- **C++标准**: C++17/C++20
- **Python**: 渐进式类型提示
- **Lint**: 严格（`-Werror`、`-Wextra`）
- **命名**: 遵循Google C++ Style Guide
- **文档**: 公共API必须有Doxygen注释

## CI/CD工作流

GitHub Actions配置文件位于 `.github/workflows/`:

- `fbgemm_ci.yml`: FBGEMM CI（多平台）
- `fbgemm_gpu_ci_cpu.yml`: FBGEMM_GPU CPU变体
- `fbgemm_gpu_ci_cuda.yml`: FBGEMM_GPU CUDA矩阵（PyTorch版本 × CUDA版本 × OS）
- `fbgemm_gpu_ci_rocm.yml`: ROCm变体CI
- `fbgemm_gpu_lint.yml`: 代码风格检查

## 发布流程

- 版本号: 基于git标签（setuptools_git_versioning）
- 变体后缀: `+cu118`, `+rocm6.2`
- 通道: `nightly`/`test`/`release`

包名示例:
- `fbgemm_gpu-1.5.0+cu118-cp38-cp38-linux_x86_64.whl`
- `fbgemm_gpu_genai-1.5.0+rocm6.2-cp39-cp39-manylinux2014_x86_64.whl`

## 项目结构快速参考

```
FBGEMM/
├── bench/                    # C++基准测试
├── cmake/modules/           # CMake构建模块
├── docs/                    # Sphinx文档
├── fbgemm_gpu/              # GPU算子库（可独立安装）
│   ├── src/
│   │   ├── quantize_ops/
│   │   ├── sparse_ops/
│   │   ├── tbe/
│   │   └── ...
│   ├── test/                # Python测试
│   ├── bench/               # Python基准测试
│   ├── codegen/             # 代码生成器
│   ├── experimental/        # GenAI, HSTU
│   ├── setup.py             # Python包构建
│   └── README.md
├── include/fbgemm/          # C++头文件
├── src/                     # FBGEMM C++实现
├── test/                    # FBGEMM C++测试
├── external/                # git submodule依赖
│   ├── asmjit/, cpuinfo/, cutlass/
│   ├── composable_kernel/, json/
│   └── googletest/
└── CMakeLists.txt           # FBGEMM主构建文件
```

## 外部资源

- 论文: https://arxiv.org/pdf/2101.05615.pdf
- 博客: https://code.fb.com/ml-applications/fbgemm
- 问题追踪: https://github.com/pytorch/FBGEMM/issues
- PyTorch Slack: `#fbgemm` 频道
- 贡献指南: `CONTRIBUTING.md`

## 重要提醒

- **子模块必须初始化**: `git submodule update --init --recursive`
- **开发FBGEMM_GPU**: 使用 `pip install -e . --no-build-isolation` 进行编辑模式安装
- **修改CUDA代码**: 注意HIPify转换（ROCm支持）
- **性能敏感代码**: 优先使用手写汇编/intrinsics而非普通C++
- **多架构兼容**: 新代码应考虑x86/ ARM/ CUDA/ROCm
- **测试覆盖**: 新功能必须添加测试并确保CI通过
- **文档更新**: API变更必须同步更新文档
