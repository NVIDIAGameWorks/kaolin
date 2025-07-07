# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import os
import torch
import math
import kaolin

class TestGaussiansConversions:
    @pytest.fixture(autouse=True)
    def xyz(self):
        return torch.tensor([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, 0.5],
            [0.5, -0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, -0.5, -0.5]
        ], dtype=torch.float, device='cuda')

    @pytest.fixture(autouse=True)
    def scales(self):
        return torch.tensor([
            [0.2, 0.05, 0.05],
            [0.2, 0.05, 0.05],
            [0.2, 0.05, 0.05],
            [0.2, 0.05, 0.05],
            [0.2, 0.05, 0.05],
            [0.2, 0.05, 0.05],
            [0.2, 0.05, 0.05],
            [0.2, 0.05, 0.05]
        ], dtype=torch.float, device='cuda')

    @pytest.fixture(autouse=True)
    def rots(self):
        theta0 = math.acos(1 / math.sqrt(3)) / 2
        theta1 = math.acos(-1 / math.sqrt(3)) / 2
        c = 1 / math.sqrt(2)
        return torch.tensor([
            [math.cos(theta0), 0, c * math.sin(theta0), -c * math.sin(theta0)],
            [math.cos(theta1), 0, c * math.sin(theta1),  c * math.sin(theta1)],
            [math.cos(theta0), 0, c * math.sin(theta0),  c * math.sin(theta0)],
            [math.cos(theta1), 0, c * math.sin(theta1), -c * math.sin(theta1)],
            [math.cos(theta1), 0, c * math.sin(theta1), -c * math.sin(theta1)],
            [math.cos(theta0), 0, c * math.sin(theta0),  c * math.sin(theta0)],
            [math.cos(theta1), 0, c * math.sin(theta1),  c * math.sin(theta1)],
            [math.cos(theta0), 0, c * math.sin(theta0), -c * math.sin(theta0)]
        ], dtype=torch.float, device='cuda')

    @pytest.fixture(autouse=True)
    def opacities(self):
        return torch.tensor([
            [1.0],
            [0.8],
            [0.6],
            [0.4],
            [0.2],
            [0.1],
            [0.05],
            [0.01]
        ], dtype=torch.float, device='cuda')

    def test_gs_to_voxelgrid_0(self, xyz, scales, rots, opacities):
        voxels, merged_opacities = kaolin.ops.conversions.gs_to_voxelgrid(
            xyz, scales, rots, opacities, level=0, iso=11.345, tol=1. / 8., step=10)
        assert torch.equal(
            voxels, torch.tensor([[0, 0, 0]], device='cuda', dtype=torch.int16))
        assert torch.allclose(
            merged_opacities, torch.tensor([0.0678], device='cuda', dtype=torch.float),
            atol=1e-4, rtol=1e-4
        )

    def test_gs_to_voxelgrid_1(self, xyz, scales, rots, opacities):
        voxels, merged_opacities = kaolin.ops.conversions.gs_to_voxelgrid(
            xyz, scales, rots, opacities, level=1, iso=11.345, tol=1. / 8., step=10)
        assert torch.equal(
            voxels,
            torch.tensor([
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1]
            ], device='cuda', dtype=torch.int16))
        assert torch.allclose(
            merged_opacities,
            torch.tensor([
                0.0004, 0.0018, 0.0036, 0.0072,
                0.0144, 0.0216, 0.0288, 0.0359
            ], device='cuda', dtype=torch.float),
            atol=1e-4, rtol=1e-4
        )

    def test_gs_to_voxelgrid_large(self, xyz, scales, rots, opacities):
        voxels, merged_opacities = kaolin.ops.conversions.gs_to_voxelgrid(
            xyz, scales, rots, opacities, level=7, iso=11.345, tol=1. / 8., step=10)
        expected = torch.load(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir,
            os.pardir, 'samples', 'ops', 'conversions', 'gs_to_voxelgrid_large.pt'
        ))
        assert torch.allclose(voxels, expected['voxels'])
        assert torch.allclose(merged_opacities, expected['merged_opacities'], atol=1e-4, rtol=1e-4)

        

