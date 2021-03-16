# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

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

import torch

from kaolin.ops.conversions import pointclouds_to_voxelgrids
from kaolin.utils.testing import FLOAT_TYPES

@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
class TestPointcloudToVoxelgrid:

    def test_pointclouds_to_voxelgrids(self, device, dtype):
        pointclouds = torch.tensor([[[0, 0, 0],
                                     [1, 1, 1],
                                     [2, 2, 2],
                                     [0, 2, 2]],
                                    
                                    [[0, 1, 2],
                                     [2, 0, 0],
                                     [1, 2, 0],
                                     [1, 1, 2]]], device=device, dtype=dtype)
        
        expected_vg = torch.tensor([[[[1., 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., 1.]],

                                     [[0., 0., 0.],
                                      [0., 1., 0.],
                                      [0., 0., 0.]],

                                     [[0., 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., 1.]]],

                                    [[[0., 0., 0.],
                                      [0., 0., 1.],
                                      [0., 0., 0.]],

                                     [[0., 0., 0.],
                                      [0., 0., 1.],
                                      [1., 0., 0.]],

                                     [[1., 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., 0.]]]], device=device, dtype=dtype)
        
        output_vg = pointclouds_to_voxelgrids(pointclouds, 3)
        
        assert torch.equal(output_vg, expected_vg)

    def test_pointclouds_to_voxelgrids_origin(self, device, dtype):
        pointclouds = torch.tensor([[[0, 0, 0],
                                     [1, 1, 1],
                                     [2, 2, 2],
                                     [0, 2, 2]],
                                    
                                    [[0, 1, 2],
                                     [2, 0, 0],
                                     [1, 2, 0],
                                     [1, 1, 2]]], device=device, dtype=dtype)
        
        expected_vg = torch.tensor([[[[1., 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., 0.]],

                                     [[0., 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., 0.]],

                                     [[0., 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., 1.]]],

                                    [[[0., 0., 1.],
                                      [0., 0., 0.],
                                      [0., 0., 0.]],

                                     [[0., 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., 0.]],

                                     [[0., 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., 0.]]]], device=device, dtype=dtype)
        
        
        output_vg = pointclouds_to_voxelgrids(pointclouds, 3, origin=torch.ones((2, 3), device=device))

        assert torch.equal(output_vg, expected_vg)

    def test_pointclouds_to_voxelgrids_scale(self, device, dtype):
        pointclouds = torch.tensor([[[0, 0, 0],
                                     [1, 1, 1],
                                     [2, 2, 2],
                                     [0, 2, 2]],
                                    
                                    [[0, 1, 2],
                                     [2, 0, 0],
                                     [1, 2, 0],
                                     [1, 1, 2]]], device=device, dtype=dtype)
        
        expected_vg = torch.tensor([[[[1., 0., 0.],
                                      [0., 1., 0.],
                                      [0., 0., 0.]],

                                     [[0., 0., 0.],
                                      [0., 1., 0.],
                                      [0., 0., 0.]],

                                     [[0., 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., 0.]]],

                                    [[[0., 1., 0.],
                                      [1., 0., 0.],
                                      [0., 0., 0.]],

                                     [[1., 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., 0.]],

                                     [[0., 0., 0.],
                                      [0., 0., 0.],
                                      [0., 0., 0.]]]], device=device, dtype=dtype)
        
        output_vg = pointclouds_to_voxelgrids(pointclouds, 3, scale=torch.ones((2), device=device) * 4)

        print(output_vg)
        
        assert torch.equal(output_vg, expected_vg)
