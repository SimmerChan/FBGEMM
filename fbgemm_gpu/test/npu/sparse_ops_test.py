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

# pyre-strict

import unittest
import hypothesis.strategies as st
import torch
from hypothesis import given, settings, Verbosity

try:
    from fbgemm_gpu import open_source
except Exception:
    pass

# NPU availability check
npu_unavailable = (
    not hasattr(torch, 'npu') or not torch.npu.is_available(),
    "NPU is not available"
)
npu_available = not npu_unavailable[0]


class NPUInvertPermuteTest(unittest.TestCase):
    """Test invert_permute operator on NPU"""

    @unittest.skipIf(*npu_unavailable)
    @given(
        size=st.integers(min_value=1, max_value=10000),
        dtype=st.sampled_from([torch.int32, torch.int64]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_invert_permute_correctness(
        self,
        size: int,
        dtype: torch.dtype,
    ) -> None:
        """Test NPU vs CPU correctness"""
        import torch_npu

        # Generate input: a permutation of [0, size)
        x = torch.randperm(size).to(dtype)

        # CPU reference
        y_cpu = torch.ops.fbgemm.invert_permute(x)

        # NPU computation
        x_npu = x.npu()
        y_npu = torch.ops.fbgemm.invert_permute(x_npu)

        # Verify results match
        torch.testing.assert_close(
            y_npu.cpu(),
            y_cpu,
            rtol=1e-5,
            atol=1e-5,
            msg=f"NPU and CPU results don't match for size={size}, dtype={dtype}"
        )

        # Verify device
        self.assertEqual(y_npu.device.type, 'privateuseone')

    @unittest.skipIf(*npu_unavailable)
    def test_invert_permute_empty_tensor(self) -> None:
        """Test empty tensor boundary case"""
        x = torch.tensor([], dtype=torch.int32)
        x_npu = x.npu()

        y_npu = torch.ops.fbgemm.invert_permute(x_npu)
        y_cpu = torch.ops.fbgemm.invert_permute(x)

        torch.testing.assert_close(y_npu.cpu(), y_cpu)

    @unittest.skipIf(*npu_unavailable)
    def test_invert_permute_single_element(self) -> None:
        """Test single element tensor"""
        x = torch.tensor([0], dtype=torch.int32)
        x_npu = x.npu()

        y_npu = torch.ops.fbgemm.invert_permute(x_npu)
        y_cpu = torch.ops.fbgemm.invert_permute(x)

        torch.testing.assert_close(y_npu.cpu(), y_cpu)

    @unittest.skipIf(*npu_unavailable)
    def test_invert_permute_error_handling(self) -> None:
        """Test error handling for invalid inputs"""
        import torch_npu

        # Test with wrong device (CPU tensor when expecting NPU)
        # This should fall back to CPU implementation, not error
        x_cpu = torch.tensor([1, 2, 0], dtype=torch.int32)
        y_cpu = torch.ops.fbgemm.invert_permute(x_cpu)
        self.assertEqual(y_cpu.device.type, 'cpu')

        # Test with wrong dtype (float instead of int)
        x_npu = torch.tensor([0, 1, 2], dtype=torch.float32).npu()
        with self.assertRaises(RuntimeError):
            torch.ops.fbgemm.invert_permute(x_npu)


if __name__ == "__main__":
    unittest.main()
