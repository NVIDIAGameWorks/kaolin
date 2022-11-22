# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.
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
import math
import torch
from kaolin.utils.testing import FLOAT_TYPES, check_tensor
from kaolin.render.mesh.utils import texture_mapping

@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
class TestTextureMapping:

    @pytest.fixture(autouse=True)
    def texture_map_1d(self, device, dtype):
        texture_map_l1 = torch.tensor([
            [11.0, 12.0, 13.0, 14.0, 15.0],
            [21.0, 22.0, 23.0, 24.0, 25.0],
            [31.0, 32.0, 33.0, 34.0, 35.0],
            [41.0, 42.0, 43.0, 44.0, 45.0]
        ], device=device, dtype=dtype)
        texture_map_ll2 = texture_map_l1 + 100
        texture_map = torch.stack((texture_map_l1, texture_map_ll2)).unsqueeze(1)
        return texture_map

    @pytest.fixture(autouse=True)
    def texture_map_3d(self, texture_map_1d):
        return torch.cat((texture_map_1d, -texture_map_1d, texture_map_1d), dim=1)

    @pytest.fixture(autouse=True)
    def sparse_coords_batch(self, device, dtype):
        texture_coordinates = torch.tensor([
            [0.0, 0.0], [1.0, 1.0], [0, 1.0], [0.5, 0.5]
        ], device=device, dtype=dtype)
        texture_coordinates = torch.stack((texture_coordinates,
                                          torch.flip(texture_coordinates, dims=(0,))))
        return texture_coordinates

    @pytest.fixture(autouse=True)
    def dense_coords_batch(self, device, dtype):
        texture_coordinates = torch.tensor([
            [[0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [0.75, 0.0], [1.0, 0.0]],
            [[0.0, 1/8], [0.25, 1/8], [0.5, 1/8], [0.75, 1/8], [1.0, 1/8]]
        ], device=device, dtype=dtype)
        texture_coordinates = torch.stack((texture_coordinates,
                                          torch.flip(texture_coordinates, dims=(0,))))
        return texture_coordinates

    @pytest.mark.parametrize('mode', ['nearest', 'bilinear'])
    def test_sparse_1d_texture_mapping(self, sparse_coords_batch, texture_map_1d, mode):
        interop = texture_mapping(texture_coordinates=sparse_coords_batch, texture_maps=texture_map_1d, mode=mode)

        if mode == 'nearest':
            expected = torch.tensor([[41, 15, 11, 33], [133, 111, 115, 141]]).unsqueeze(-1)
        elif mode == 'bilinear':
            expected = torch.tensor([[41, 15, 11, 28], [128, 111, 115, 141]]).unsqueeze(-1)
        expected = expected.to(texture_map_1d.device).type(texture_map_1d.dtype)
        assert check_tensor(interop, shape=(2,4,1), dtype=texture_map_1d.dtype)
        assert torch.equal(interop, expected)

    @pytest.mark.parametrize('mode', ['nearest', 'bilinear'])
    def test_sparse_3d_texture_mapping(self, sparse_coords_batch, texture_map_3d, mode):
        interop = texture_mapping(texture_coordinates=sparse_coords_batch,
                                  texture_maps=texture_map_3d,
                                  mode=mode)
        if mode == 'nearest':
            expected_d1 = torch.tensor([[41, 15, 11, 33], [133, 111, 115, 141]])
            expected_d2 = -torch.tensor([[41, 15, 11, 33], [133, 111, 115, 141]])
            expected_d3 = torch.tensor([[41, 15, 11, 33], [133, 111, 115, 141]])
            expected = torch.stack([expected_d1, expected_d2, expected_d3], dim=-1)
        elif mode == 'bilinear':
            expected_d1 = torch.tensor([[41, 15, 11, 28], [128, 111, 115, 141]])
            expected_d2 = -torch.tensor([[41, 15, 11, 28], [128, 111, 115, 141]])
            expected_d3 = torch.tensor([[41, 15, 11, 28], [128, 111, 115, 141]])
            expected = torch.stack([expected_d1, expected_d2, expected_d3], dim=-1)
        expected = expected.to(texture_map_3d.device).type(texture_map_3d.dtype)
        assert check_tensor(interop, shape=(2,4,3), dtype=texture_map_3d.dtype)
        assert torch.equal(interop, expected)

    @pytest.mark.parametrize('mode', ['nearest', 'bilinear'])
    def test_dense_3d_texture_mapping(self, dense_coords_batch, texture_map_3d,
                                      mode, device, dtype):
        interop = texture_mapping(texture_coordinates=dense_coords_batch, texture_maps=texture_map_3d, mode=mode)

        if mode == 'nearest':
            expected_base = torch.tensor([41., 42., 43., 44., 45.],
                                         device=device, dtype=dtype)
        elif mode == 'bilinear':
            expected_base = torch.tensor([41., 41.75, 43., 44.25, 45.],
                                         device=device, dtype=dtype)
        expected = torch.stack([expected_base, expected_base + 100], dim=0)
        expected = torch.stack([expected, -expected, expected],
                               dim=-1).reshape(2, 1, -1, 3).repeat(1, 2, 1, 1)

        assert torch.equal(interop, expected)
