# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
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

import math
import pytest
import os
import itertools

import torch

from kaolin.ops.spc import points_to_morton, morton_to_points, points_to_corners, \
                           coords_to_trilinear, quantize_points

class TestPoints:
    @pytest.fixture(autouse=True)
    def points(self):
        return torch.tensor([
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
            [0, 1, 0]], device='cuda', dtype=torch.int16)

    @pytest.fixture(autouse=True)
    def morton(self):
        return torch.tensor([0, 1, 8, 9, 2], device='cuda', dtype=torch.long)

    def test_quantize_points(self):
        x = torch.tensor([
            [-1.1, -1.1, -1.1],
            [-1., -1., -1.],
            [0., 0., 0.],
            [0.1, 0.3, 0.6],
            [0.1, -1.1, 1.1],
            [0.1, -1., 1.],
            [1., 1., 1.],
            [1.1, 1.1, 1.1]], device='cuda', dtype=torch.float)

        points = quantize_points(x, 3)
        expected_points = torch.tensor([
            [0, 0, 0],
            [0, 0, 0],
            [4, 4, 4],
            [4, 5, 6],
            [4, 0, 7],
            [4, 0, 7],
            [7, 7, 7],
            [7, 7, 7]], device='cuda', dtype=torch.int16)

        assert torch.equal(points, expected_points)
    def test_points_to_morton(self, points, morton):
        assert torch.equal(points_to_morton(points), morton)

    def test_morton_to_points(self, morton, points):
        assert torch.equal(morton_to_points(morton), points)

    def test_points_to_corners(self, points):
        expected_corners = []
        for offset in itertools.product([0, 1], repeat=3):
            expected_corners.append(points + torch.tensor([offset], device='cuda', dtype=torch.int16))
        expected_corners = torch.stack(expected_corners, dim=-2)
        assert torch.equal(points_to_corners(points), expected_corners)

    def test_coords_to_trilinear(self, points):
        w = torch.rand(points.shape, device='cuda')
        x = points + w
        expected_coeffs = torch.stack([
            (1 - w[:, 0]) * (1 - w[:, 1]) * (1 - w[:, 2]),
            (1 - w[:, 0]) * (1 - w[:, 1]) * w[:, 2],
            (1 - w[:, 0]) * w[:, 1] * (1 - w[:, 2]),
            (1 - w[:, 0]) * w[:, 1] * w[:, 2],
            w[:, 0] * (1 - w[:, 1]) * (1 - w[:, 2]),
            w[:, 0] * (1 - w[:, 1]) * w[:, 2],
            w[:, 0] * w[:, 1] * (1 - w[:, 2]),
            w[:, 0] * w[:, 1] * w[:, 2]
        ], dim=-1)
        assert torch.allclose(coords_to_trilinear(x, points), expected_coeffs)
