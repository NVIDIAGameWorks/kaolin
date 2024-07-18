# Copyright (c) 204 NVIDIA CORPORATION & AFFILIATES.
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
import os

import math
import torch
from kaolin.ops.gaussian import VolumeDensifier

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, os.pardir, os.pardir, os.pardir, 'samples/')
TEST_MODELS = ['hotdog_minimal.pt']
SUPPORTED_GSPLATS_DEVICES = ['cuda']
SUPPORTED_GSPLATS_DTYPES = [torch.float32]

@pytest.mark.parametrize('model_name', TEST_MODELS)
@pytest.mark.parametrize("device", SUPPORTED_GSPLATS_DEVICES)
@pytest.mark.parametrize("dtype", SUPPORTED_GSPLATS_DTYPES)
class TestVolumeDensifier:

    @pytest.fixture(autouse=True)
    def test_sample_points_in_volume(self, model_name, device, dtype):
        model_path = os.path.join(ROOT_DIR, model_name)
        gaussians = torch.load(model_path)
        pos = gaussians['position'].cuda()                  # model.get_xyz
        scale = gaussians['scale'].cuda()                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].cuda()             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).cuda()    # model.get_opacity

        densifier = VolumeDensifier()
        output = densifier.sample_points_in_volume(xyz=pos, scale=scale, rotation=rotation, opacity=opacity)

        assert output.ndim == 2
        assert output.shape[1] == 3
        assert output.device == device
        assert output.dtype == dtype
        assert (pos.min(dim=0)[0] <= output.min(dim=0)[0]).all()    # Check points are within shell
        assert (pos.max(dim=0)[0] >= output.max(dim=0)[0]).all()    # Check points are within shell

    @pytest.fixture(autouse=True)
    def test_sample_points_in_volume_for_density(self, model_name, device, dtype):
        model_path = os.path.join(ROOT_DIR, model_name)
        gaussians = torch.load(model_path)
        pos = gaussians['position'].cuda()                  # model.get_xyz
        scale = gaussians['scale'].cuda()                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].cuda()             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).cuda()    # model.get_opacity

        densifier = VolumeDensifier(resolution=6, jitter=True)
        output = densifier.sample_points_in_volume(xyz=pos, scale=scale, rotation=rotation, opacity=opacity)

        assert output.ndim == 2
        assert output.shape[1] == 3
        assert output.device == device
        assert output.dtype == dtype
        assert (pos.min(dim=0)[0] <= output.min(dim=0)[0]).all()    # Check points are within shell
        assert (pos.max(dim=0)[0] >= output.max(dim=0)[0]).all()    # Check points are within shell

        N = output.shape[0]
        pairwise_dists = torch.linalg.norm(output[None] - output[:, None], dim=2)
        pairwise_dists += output.max() * torch.eye(N, dtype=dtype, device=device)
        maximal_nearest_neighbor_distance = pairwise_dists.min(dim=0)[0].max()

        spc_cell_length = 2.0 / 2**6
        spc_diagonal_length = spc_cell_length * math.sqrt(3.0)
        min_allowed_pts_distance = 2.0 * spc_diagonal_length # two spc diagonals
        assert maximal_nearest_neighbor_distance <= min_allowed_pts_distance

    @pytest.fixture(autouse=True)
    def test_sample_points_in_volume_no_jitter_for_density(self, model_name, device, dtype):
        model_path = os.path.join(ROOT_DIR, model_name)
        gaussians = torch.load(model_path)
        pos = gaussians['position'].cuda()                  # model.get_xyz
        scale = gaussians['scale'].cuda()                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].cuda()             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).cuda()    # model.get_opacity

        densifier = VolumeDensifier(resolution=6, jitter=False)
        output = densifier.sample_points_in_volume(xyz=pos, scale=scale, rotation=rotation, opacity=opacity)

        assert output.ndim == 2
        assert output.shape[1] == 3
        assert output.device == device
        assert output.dtype == dtype
        assert (pos.min(dim=0)[0] <= output.min(dim=0)[0]).all()    # Check points are within shell
        assert (pos.max(dim=0)[0] >= output.max(dim=0)[0]).all()    # Check points are within shell

        N = output.shape[0]
        pairwise_dists = torch.linalg.norm(output[None] - output[:, None], dim=2)
        pairwise_dists += output.max() * torch.eye(N, dtype=dtype, device=device)
        maximal_nearest_neighbor_distance = pairwise_dists.min(dim=0)[0].max()

        # Distance is cell length
        spc_cell_length = 2.0 / 2**6
        assert maximal_nearest_neighbor_distance == spc_cell_length

    @pytest.fixture(autouse=True)
    def test_sample_points_in_volume_for_count(self, model_name, device, dtype):
        model_path = os.path.join(ROOT_DIR, model_name)
        gaussians = torch.load(model_path)
        pos = gaussians['position'].cuda()                  # model.get_xyz
        scale = gaussians['scale'].cuda()                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].cuda()             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).cuda()    # model.get_opacity

        densifier = VolumeDensifier()
        output = densifier.sample_points_in_volume(
            xyz=pos, scale=scale, rotation=rotation, opacity=opacity, count=5000
        )

        assert output.ndim == 2
        assert output.shape[1] == 5000
        assert output.shape[1] == 3
        assert output.device == device
        assert output.dtype == dtype
        assert (pos.min(dim=0)[0] <= output.min(dim=0)[0]).all()    # Check points are within shell
        assert (pos.max(dim=0)[0] >= output.max(dim=0)[0]).all()    # Check points are within shell
