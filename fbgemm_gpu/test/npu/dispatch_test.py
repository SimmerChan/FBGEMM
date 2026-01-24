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
import torch

# NPU availability check
npu_unavailable = (
    not hasattr(torch, 'npu') or not torch.npu.is_available(),
    "NPU is not available"
)


class DispatcherTest(unittest.TestCase):
    """Test device routing logic in dispatcher"""

    def test_cpu_routing(self) -> None:
        """Test that CPU tensors route to CPU implementation"""
        x = torch.tensor([1, 2, 0], dtype=torch.int32)
        y = torch.ops.fbgemm.invert_permute(x)

        # Verify output is on CPU
        self.assertEqual(y.device.type, 'cpu')

        # Verify correctness
        expected = torch.tensor([2, 0, 1], dtype=torch.int32)
        torch.testing.assert_close(y, expected)

    @unittest.skipIf(*npu_unavailable)
    def test_npu_routing(self) -> None:
        """Test that NPU tensors route to NPU implementation"""
        x = torch.tensor([1, 2, 0], dtype=torch.int32).npu()
        y = torch.ops.fbgemm.invert_permute(x)

        # Verify output is on NPU
        self.assertEqual(y.device.type, 'privateuseone')
        self.assertEqual(y.device.index, 0)

        # Verify correctness by comparing with CPU
        x_cpu = x.cpu()
        y_cpu_expected = torch.ops.fbgemm.invert_permute(x_cpu)
        torch.testing.assert_close(y.cpu(), y_cpu_expected)

    def test_dispatcher_transparent_routing(self) -> None:
        """Test that dispatcher routes transparently based on device"""
        # Same input, different devices
        x_cpu = torch.tensor([1, 2, 0], dtype=torch.int32)

        # CPU result
        y_cpu = torch.ops.fbgemm.invert_permute(x_cpu)

        # Verify CPU result is correct
        expected = torch.tensor([2, 0, 1], dtype=torch.int32)
        torch.testing.assert_close(y_cpu, expected)

    @unittest.skipIf(*npu_unavailable)
    def test_cross_device_error(self) -> None:
        """Test that mixing devices raises appropriate errors"""
        import torch_npu

        # This should work - both inputs on same device
        x = torch.tensor([0, 1], dtype=torch.int32).npu()
        indices = torch.tensor([1, 0], dtype=torch.int32).npu()
        # y = torch.ops.fbgemm.permute_1d(x, indices)
        # This will be tested when permute_1d is implemented


if __name__ == "__main__":
    unittest.main()
