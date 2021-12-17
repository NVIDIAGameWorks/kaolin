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

from kaolin.ops.conversions import pointclouds_to_voxelgrids, unbatched_pointcloud_to_spc
from kaolin.utils.testing import FLOAT_TYPES, BOOL_DTYPES, INT_DTYPES, FLOAT_DTYPES, ALL_DTYPES, check_spc_octrees

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
        
        
        output_vg = pointclouds_to_voxelgrids(pointclouds, 3, origin=torch.ones((2, 3), device=device, dtype=dtype))

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
        
        output_vg = pointclouds_to_voxelgrids(pointclouds, 3, scale=torch.ones((2), device=device, dtype=dtype) * 4)
    
        assert torch.equal(output_vg, expected_vg)


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('level', list(range(1, 6)))
class TestUnbatchedPointcloudToSpc:

    @pytest.fixture
    def pointcloud(self, device):
        return torch.tensor([[-1, -1, -1],
                             [-1, -1, 0],
                             [0, -1, -1],
                             [-1, 0, -1],
                             [0, 0, 0],
                             [1, 1, 1],
                             [0.999, 0.999, 0.999]], device=device)


    @pytest.fixture
    def typed_pointcloud(self, pointcloud, dtype):
        return pointcloud.to(dtype)

    @pytest.fixture(autouse=True)
    def expected_octree(self, device, level):
        level_cutoff_mapping = [1, 6, 12, 18, 24]
        level_cutoff = level_cutoff_mapping[level-1]
        full_octree = torch.tensor([151,
                                    1, 1, 1, 1, 129,
                                    1, 1, 1, 1, 1, 128,
                                    1, 1, 1, 1, 1, 128,
                                    1, 1, 1, 1, 1, 128], device=device)
        expected_octree = full_octree[:level_cutoff]
        return expected_octree.byte()

    @pytest.fixture(autouse=True)
    def bool_features(self):
        def _bool_features(device, booltype):
            return torch.tensor([[0],
                                 [1],
                                 [1],
                                 [1],
                                 [0],
                                 [1],
                                 [1]], device=device).to(booltype)
        return _bool_features

    @pytest.fixture(autouse=True)
    def expected_bool_features(self):
        def _expected_bool_features(device, booltype, level):
            if level == 1:
                return torch.tensor([[0],
                                     [1],
                                     [1],
                                     [1],
                                     [1]], device=device).to(booltype)
            else:
                return torch.tensor([[0],
                                     [1],
                                     [1],
                                     [1],
                                     [0],
                                     [1]], device=device).to(booltype)
        return _expected_bool_features

    @pytest.fixture(autouse=True)
    def int_features(self):
        def _int_features(device, inttype):
            return torch.tensor([[1],
                                 [4],
                                 [7],
                                 [10],
                                 [20],
                                 [37],
                                 [1]], device=device).to(inttype)
        return _int_features

    @pytest.fixture(autouse=True)
    def expected_int_features(self):
        def _expected_int_features(device, inttype, level):
            if level == 1:
                return torch.tensor([[1],
                                     [4],
                                     [10],
                                     [7],
                                     [19]], device=device).to(inttype)
            else:
                return torch.tensor([[1],
                                     [4],
                                     [10],
                                     [7],
                                     [20],
                                     [19]], device=device).to(inttype)
        return _expected_int_features

    @pytest.fixture(autouse=True)
    def fp_features(self):
        def _fp_features(device, fptype):
            return torch.tensor([[1, 2, 3],
                                 [4, 5, 6],
                                 [7, 8, 9],
                                 [10, 10, 10],
                                 [20, 20, 20],
                                 [37, 37, 37],
                                 [1, 2, 3]], device=device).to(fptype)
        return _fp_features

    @pytest.fixture(autouse=True)
    def expected_fp_features(self):
        def _expected_fp_features(device, fptype, level):
            if level == 1:
                return torch.tensor([[1, 2, 3],
                                     [4, 5, 6],
                                     [10, 10, 10],
                                     [7, 8, 9],
                                     [58/3, 59/3, 60/3]], device=device).to(fptype)
            else:
                return torch.tensor([[1, 2, 3],
                                     [4, 5, 6],
                                     [10, 10, 10],
                                     [7, 8, 9],
                                     [20, 20, 20],
                                     [19, 19.5, 20]], device=device).to(fptype)
        return _expected_fp_features


    @pytest.mark.parametrize('dtype', FLOAT_DTYPES)
    def test_unbatched_pointcloud_to_spc(self, typed_pointcloud, level, expected_octree):
        output_spc = unbatched_pointcloud_to_spc(typed_pointcloud, level)
        assert check_spc_octrees(output_spc.octrees, output_spc.lengths,
                                 batch_size=output_spc.batch_size,
                                 level=level,
                                 device=typed_pointcloud.device.type)
        assert torch.equal(output_spc.octrees, expected_octree)

    @pytest.mark.parametrize('booltype', BOOL_DTYPES)
    def test_unbatched_pointcloud_to_spc_with_bool_features(self, pointcloud, device, booltype, level,
                                                            bool_features, expected_bool_features):
        features_arg = bool_features(device, booltype)
        expected_features_arg = expected_bool_features(device, booltype, level)
        output_spc = unbatched_pointcloud_to_spc(pointcloud, level, features_arg)
        assert torch.equal(output_spc.features, expected_features_arg)

    @pytest.mark.parametrize('inttype', INT_DTYPES)
    def test_unbatched_pointcloud_to_spc_with_int_features(self, pointcloud, device, inttype, level,
                                                           int_features, expected_int_features):
        features_arg = int_features(device, inttype)
        expected_features_arg = expected_int_features(device, inttype, level)
        output_spc = unbatched_pointcloud_to_spc(pointcloud, level, features_arg)
        assert torch.equal(output_spc.features, expected_features_arg)

    @pytest.mark.parametrize('fptype', FLOAT_DTYPES)
    def test_unbatched_pointcloud_to_spc_with_fp_features(self, pointcloud, device, fptype, level,
                                                          fp_features, expected_fp_features):
        features_arg = fp_features(device, fptype)
        expected_features_arg = expected_fp_features(device, fptype, level)
        output_spc = unbatched_pointcloud_to_spc(pointcloud, level, features_arg)
        assert torch.allclose(output_spc.features, expected_features_arg)
