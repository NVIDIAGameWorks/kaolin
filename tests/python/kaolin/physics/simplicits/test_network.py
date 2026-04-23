# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

import pytest
import torch

from kaolin.physics.simplicits.network import SimplicitsMLP, SkinningModule
from kaolin.physics.utils.finite_diff import finite_diff_jac


class TestSkinningModule:

    @pytest.fixture
    def device(self):
        return 'cuda'

    @pytest.fixture
    def dtype(self):
        return torch.float

    def test_compute_skinning_weights_shape(self, device, dtype):
        # SimplicitsMLP with 5 handles: forward outputs 4, +1 constant = 5
        mlp = SimplicitsMLP(3, 32, 5, 3).to(device)
        pts = torch.rand(20, 3, device=device, dtype=dtype)
        w = mlp.compute_skinning_weights(pts)
        assert w.shape == (20, 5)

    def test_compute_dwdx_shape(self, device, dtype):
        mlp = SimplicitsMLP(3, 32, 5, 3).to(device)
        pts = torch.rand(20, 3, device=device, dtype=dtype)
        dwdx = mlp.compute_dwdx(pts)
        assert dwdx.shape == (20, 5, 3)

    def test_compute_dwdx_finite_diff(self, device):
        num_handles = 5
        mlp = SimplicitsMLP(3, 32, num_handles, 3).to(device=device, dtype=torch.float64)
        pts = torch.rand(10, 3, device=device, dtype=torch.float64)
        dwdx_analytic = mlp.compute_dwdx(pts)
        dwdx_fd = finite_diff_jac(mlp.compute_skinning_weights, pts)
        torch.testing.assert_close(dwdx_analytic, dwdx_fd, atol=1e-5, rtol=1e-4)

    def test_offset_scale(self, device, dtype):
        bb_min = torch.zeros(3, device=device, dtype=dtype)
        bb_max = torch.ones(3, device=device, dtype=dtype) * 2.0
        mod = SkinningModule.from_function(
            lambda x: torch.zeros(x.shape[0], 0, device=x.device, dtype=x.dtype),
            bb_min=bb_min, bb_max=bb_max)
        pts = torch.rand(10, 3, device=device, dtype=dtype)  # pts in [0, 1]
        scaled = mod._offset_scale(pts)
        assert scaled.max() <= 0.5 + 1e-6  # [0,1] mapped to [0, 0.5]
