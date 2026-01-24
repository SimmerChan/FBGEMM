# Ralph Fix Plan - FBGEMM NPU Backend Adaptation

## High Priority - Phase 1: Framework Setup

- [ ] **Create NPU directory structure**
  - [ ] Create `fbgemm_gpu/src/npu/` with subdirectories: `dispatch/`, `sparse_ops/`, `quantize_ops/`, `utils/`, `ascendc/`
  - [ ] Create `fbgemm_gpu/test/npu/` for NPU-specific tests
  - [ ] Create `fbgemm_gpu/include/fbgemm_gpu/npu/` for public NPU headers

- [ ] **Configure CMake build system**
  - [ ] Add `USE_NPU` option to main CMakeLists.txt (default: OFF)
  - [ ] Add CANN toolkit detection (find_package, path detection)
  - [ ] Create `fbgemm_gpu/src/npu/CMakeLists.txt` for NPU wrapper library
  - [ ] Create `fbgemm_gpu/src/npu/ascendc/CMakeLists.txt` for AscendC kernels
  - [ ] Verify compilation works with both USE_NPU=ON and USE_NPU=OFF

- [ ] **Implement NPU utility functions**
  - [ ] Create `npu_ops_utils.h`: TENSOR_ON_NPU macro, NPUDeviceGuard class
  - [ ] Create `npu_type_utils.h`: Type mapping (c10::ScalarType → ge::DataType)
  - [ ] Add device detection functions: `is_npu_tensor()`, `get_ge_dtype()`
  - [ ] Add ACLNN error checking macros

- [ ] **Implement unified dispatcher layer**
  - [ ] Create `npu/dispatch/dispatcher.h` with inline dispatch functions
  - [ ] Add forward declarations for CUDA/CPU/NPU implementations
  - [ ] Create `npu/dispatch/sparse_ops.cpp` with TORCH_LIBRARY_IMPL registration
  - [ ] Use CatchAll dispatch key for device-based routing
  - [ ] Test dispatcher routing logic (CPU → CPU impl, NPU → NPU impl)

## High Priority - Phase 2: First Example Operator

- [ ] **Implement invert_permute as reference operator**
  - [ ] Create AscendC op_host/invert_permute.cpp (tiling logic)
  - [ ] Create AscendC op_host/invert_permute_tiling.h (tiling data structures)
  - [ ] Create AscendC op_kernel/invert_permute.cpp (AscendC kernel)
  - [ ] Create AscendC op_kernel/invert_permute_kernel.h (kernel declaration)
  - [ ] Compile AscendC kernel to libinvert_permute.so
  - [ ] Create NPU wrapper `npu/sparse_ops/invert_permute.cpp`
  - [ ] Add type dispatch (int32, int64) using AT_DISPATCH_INDEX_TYPES
  - [ ] Add parameter validation and error handling
  - [ ] Add dispatcher registration in dispatcher.h

- [ ] **Write comprehensive tests for invert_permute**
  - [ ] Create `test/npu/sparse_ops_test.py` with unittest + Hypothesis
  - [ ] Test correctness: compare NPU output vs CPU output
  - [ ] Test type coverage: int32, int64
  - [ ] Test edge cases: empty tensor, single element, large tensor
  - [ ] Test error handling: wrong device, wrong dtype, wrong dimensions
  - [ ] Verify all tests pass

## Medium Priority - Phase 3: Core Sparse Operators

- [ ] **Implement permute_1d operator**
  - [ ] AscendC kernel implementation (op_host + op_kernel)
  - [ ] NPU wrapper with type dispatch
  - [ ] Dispatcher registration
  - [ ] Unit tests (correctness, types, edge cases)

- [ ] **Implement permute_2d operator**
  - [ ] AscendC kernel implementation
  - [ ] NPU wrapper with type dispatch
  - [ ] Dispatcher registration
  - [ ] Unit tests

- [ ] **Implement index_select_dim0 operator**
  - [ ] AscendC kernel implementation
  - [ ] NPU wrapper with type dispatch
  - [ ] Dispatcher registration
  - [ ] Unit tests

- [ ] **Implement pack_segments operator**
  - [ ] AscendC kernel implementation
  - [ ] NPU wrapper with type dispatch
  - [ ] Dispatcher registration
  - [ ] Unit tests

## Medium Priority - Phase 4: Quantization Operators

- [ ] **Implement quantize operator**
  - [ ] AscendC kernel implementation
  - [ ] NPU wrapper supporting float/half/bfloat16 → int8
  - [ ] Dispatcher registration
  - [ ] Unit tests for all dtype combinations

- [ ] **Implement dequantize operator**
  - [ ] AscendC kernel implementation
  - [ ] NPU wrapper supporting int8 → float/half/bfloat16
  - [ ] Dispatcher registration
  - [ ] Unit tests

- [ ] **Implement float_to_bfloat16 operator**
  - [ ] AscendC kernel implementation
  - [ ] NPU wrapper
  - [ ] Dispatcher registration
  - [ ] Unit tests

## Low Priority - Phase 5: Advanced Features & Optimization

- [ ] **Performance testing and optimization**
  - [ ] Benchmark NPU vs CPU performance for core operators
  - [ ] Profile AscendC kernels and optimize tiling parameters
  - [ ] Add performance regression tests to CI

- [ ] **Extended operator support**
  - [ ] Implement remaining sparse operators as needed
  - [ ] Implement embedding lookup operators (if required)
  - [ ] Add custom NPU-specific operators for performance

- [ ] **Documentation**
  - [ ] Write NPU build instructions (README_NPU.md)
  - [ ] Write operator development guide (CONTRIBUTING_NPU.md)
  - [ ] Write user migration guide (CUDA → NPU)
  - [ ] Add inline code comments for complex logic

- [ ] **CI/CD integration**
  - [ ] Add NPU CI job (runs on NPU hardware)
  - [ ] Add pre-commit checks for NPU code quality
  - [ ] Set up automated testing pipeline

## Completed
- [x] Project initialization - analyzed design document and created Ralph specs

## Notes

### Architecture Principles
1. **Zero Intrusion**: Never modify existing CUDA/CPU source files in `src/sparse_ops/`, `src/quantize_ops/`
2. **Complete Isolation**: All NPU code lives in `src/npu/` directory
3. **Optional Compilation**: USE_NPU=OFF must work without CANN toolkit
4. **Automatic Routing**: Dispatcher detects tensor device and routes transparently
5. **Testing Strategy**: Compare NPU results against CPU (not GPU - different deployment environments)

### Operator Implementation Checklist
For each operator:
1. AscendC kernel in `ascendc/<op_name>/op_host/` and `op_kernel/`
2. NPU wrapper in `npu/sparse_ops/` or `npu/quantize_ops/`
3. Add to `npu/dispatch/dispatcher.h` with inline dispatch function
4. Register in `npu/dispatch/sparse_ops.cpp` or `quantize_ops.cpp`
5. Write tests in `test/npu/` using unittest + Hypothesis
6. Verify compilation with both USE_NPU=ON and USE_NPU=OFF
7. Run tests and ensure correctness vs CPU

### Type Support Matrix
Target dtype support for all operators:
- ✅ int32 (index types)
- ✅ int64 (index types)
- ✅ float32
- ✅ float16 (half)
- ✅ bfloat16

### Risk Mitigation
- **AscendC toolchain issues**: Validate early with simple operator (invert_permute)
- **Performance gaps**: Plan dedicated optimization phase after core features work
- **CI environment instability**: Use self-hosted NPU runners with stable setup
- **Resource constraints**: Prioritize P0 operators, defer P2 features

### Success Criteria
Phase 1-2 complete when:
- ✅ CMake builds with USE_NPU=ON and USE_NPU=OFF
- ✅ invert_permute works on NPU tensors
- ✅ Dispatcher correctly routes all device types
- ✅ Unit tests pass for invert_permute
- ✅ Zero CUDA/CPU code modified

Phase 3-4 complete when:
- ✅ All P0 sparse operators implemented
- ✅ Core quantization operators implemented
- ✅ Test coverage > 90% for NPU operators
- ✅ Performance validated on real NPU hardware

Project complete when:
- ✅ All MVP operators from requirements implemented
- ✅ All tests passing
- ✅ Documentation complete
- ✅ Ready for upstream PR to FBGEMM
