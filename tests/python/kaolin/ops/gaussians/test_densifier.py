# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
from kaolin.utils.testing import check_tensor

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, os.pardir, os.pardir, os.pardir, 'samples')
TEST_MODELS = ['spheres', 'hotdog_minimal.pt']
SUPPORTED_GSPLATS_DEVICES = ['cuda']
SUPPORTED_GSPLATS_DTYPES = [torch.float32]


def validate_samples_inside_shell(samples, gaussian_xyz):
    assert (gaussian_xyz.min(dim=0)[0] <= samples.min(dim=0)[0]).all()  # Check points are within shell
    assert (gaussian_xyz.max(dim=0)[0] >= samples.max(dim=0)[0]).all()  # Check points are within shell


@pytest.mark.parametrize('model_name', TEST_MODELS)
@pytest.mark.parametrize("device", SUPPORTED_GSPLATS_DEVICES)
@pytest.mark.parametrize("dtype", SUPPORTED_GSPLATS_DTYPES)
class TestVolumeDensifier:

    @pytest.fixture(autouse=True)
    def gaussians(self, model_name):
        if model_name == 'spheres':
            N = 100000
            pts = torch.rand(N, 3)
            pts /= pts.norm(dim=1).unsqueeze(1)
            pts *= 6.0
            gaussian_fields = dict(
                position=pts,
                scale=torch.full([N, 3], 0.01),
                rotation=torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(N, 1),
                opacity=torch.full([N, 1], 0.80)
            )
        else:
            model_path = os.path.join(ROOT_DIR, 'gsplats', model_name)
            gaussian_fields = torch.load(os.path.abspath(model_path))
        gaussian_fields['model_name'] = model_name
        return gaussian_fields

    def test_sample_points_in_volume(self, gaussians, device, dtype):
        pos = gaussians['position'].cuda()                  # model.get_xyz
        scale = gaussians['scale'].cuda()                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].cuda()             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).cuda()    # model.get_opacity

        densifier = VolumeDensifier()
        output = densifier.sample_points_in_volume(xyz=pos, scale=scale, rotation=rotation, opacity=opacity)

        assert output.ndim == 2
        assert output.shape[1] == 3
        check_tensor(output, dtype=dtype, device=device)
        validate_samples_inside_shell(output, pos)

    def test_sample_points_in_volume_for_density(self, gaussians, device, dtype):
        pos = gaussians['position'].cuda()                  # model.get_xyz
        scale = gaussians['scale'].cuda()                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].cuda()             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).cuda()    # model.get_opacity

        resolution = 6
        densifier = VolumeDensifier(resolution=resolution, jitter=True)
        output = densifier.sample_points_in_volume(xyz=pos, scale=scale, rotation=rotation, opacity=opacity)

        assert output.ndim == 2
        assert output.shape[1] == 3
        check_tensor(output, dtype=dtype, device=device)
        validate_samples_inside_shell(output, pos)

        N = output.shape[0]
        pairwise_dists = torch.linalg.norm(output[None] - output[:, None], dim=2)
        pairwise_dists += output.max() * torch.eye(N, dtype=dtype, device=device)
        minimal_nearest_neighbor_distance = pairwise_dists.min(dim=0)[0].min()
        maximal_nearest_neighbor_distance = pairwise_dists.min(dim=0)[0].max()

        spc_cell_length = 2.0 / 2**resolution
        spc_diagonal_length = spc_cell_length * math.sqrt(3.0)
        max_allowed_pts_distance = 2.0 * spc_diagonal_length # two spc diagonals
        assert maximal_nearest_neighbor_distance <= max_allowed_pts_distance

        # Points should not be equi-distanced
        assert minimal_nearest_neighbor_distance != maximal_nearest_neighbor_distance

    def test_sample_points_in_volume_no_jitter_for_density(self, gaussians, device, dtype):
        pos = gaussians['position'].cuda()                  # model.get_xyz
        scale = gaussians['scale'].cuda()                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].cuda()             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).cuda()    # model.get_opacity

        resolution = 6
        densifier = VolumeDensifier(resolution=resolution, jitter=False, post_scale_factor=1.0)
        output = densifier.sample_points_in_volume(xyz=pos, scale=scale, rotation=rotation, opacity=opacity)

        assert output.ndim == 2
        assert output.shape[1] == 3
        check_tensor(output, dtype=dtype, device=device)
        validate_samples_inside_shell(output, pos)

        N = output.shape[0]
        pairwise_dists = torch.linalg.norm(output[None] - output[:, None], dim=2)
        pairwise_dists += output.max() * torch.eye(N, dtype=dtype, device=device)
        minimal_nearest_neighbor_distance = pairwise_dists.min(dim=0)[0].min()
        maximal_nearest_neighbor_distance = pairwise_dists.min(dim=0)[0].max()

        # Points should be equi-distanced
        assert torch.isclose(minimal_nearest_neighbor_distance, maximal_nearest_neighbor_distance)

    def test_sample_points_in_volume_for_count(self, gaussians, device, dtype):
        pos = gaussians['position'].cuda()                  # model.get_xyz
        scale = gaussians['scale'].cuda()                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].cuda()             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).cuda()    # model.get_opacity

        densifier = VolumeDensifier()
        output = densifier.sample_points_in_volume(
            xyz=pos, scale=scale, rotation=rotation, opacity=opacity, count=5000
        )

        assert output.ndim == 2
        check_tensor(output, shape=(5000, 3), dtype=dtype, device=device)
        validate_samples_inside_shell(output, pos)

    def test_sample_points_in_volume_without_opacity_threshold(self, gaussians, device, dtype):
        pos = gaussians['position'].cuda()  # model.get_xyz
        scale = gaussians['scale'].cuda()  # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].cuda()  # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).cuda()  # model.get_opacity

        densifier = VolumeDensifier(opacity_threshold=0.0)
        output = densifier.sample_points_in_volume(
            xyz=pos, scale=scale, rotation=rotation, opacity=opacity
        )

        assert output.ndim == 2
        assert output.shape[1] == 3
        check_tensor(output, dtype=dtype, device=device)
        validate_samples_inside_shell(output, pos)

    def test_sample_points_in_volume_with_high_opacity_threshold(self, gaussians, device, dtype):
        pos = gaussians['position'].cuda()  # model.get_xyz
        scale = gaussians['scale'].cuda()  # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].cuda()  # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).cuda()  # model.get_opacity

        densifier = VolumeDensifier(opacity_threshold=0.5)
        output = densifier.sample_points_in_volume(
            xyz=pos, scale=scale, rotation=rotation, opacity=opacity
        )

        assert output.ndim == 2
        assert output.shape[1] == 3
        check_tensor(output, dtype=dtype, device=device)
        validate_samples_inside_shell(output, pos)

    def test_sample_points_in_volume_with_mask(self, gaussians, device, dtype):
        pos = gaussians['position'].cuda()  # model.get_xyz
        scale = gaussians['scale'].cuda()  # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].cuda()  # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).cuda()  # model.get_opacity

        opacity_threshold = 0.35

        N = pos.shape[0]
        opacity_mask = opacity.reshape(-1) < opacity_threshold
        mask = pos.new_ones((N,), device=device, dtype=torch.bool)
        mask &= ~opacity_mask

        densifier1 = VolumeDensifier(opacity_threshold=0.0, jitter=False)
        output1 = densifier1.sample_points_in_volume(
            xyz=pos, scale=scale, rotation=rotation, opacity=opacity, mask=mask
        )

        densifier2 = VolumeDensifier(opacity_threshold=0.35, jitter=False)
        output2 = densifier2.sample_points_in_volume(
            xyz=pos, scale=scale, rotation=rotation, opacity=opacity,
        )

        assert output1.ndim == 2
        assert output1.shape[1] == 3
        check_tensor(output1, dtype=dtype, device=device)

        assert output2.ndim == 2
        assert output2.shape[1] == 3
        check_tensor(output2, dtype=dtype, device=device)
        validate_samples_inside_shell(output2, pos)

        assert output1.shape[0] >= output2.shape[0]

    def test_sample_points_in_volume_with_custom_viewpoints(self, gaussians, device, dtype):

        pos = gaussians['position'].cuda()                  # model.get_xyz
        scale = gaussians['scale'].cuda()                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].cuda()             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).cuda()    # model.get_opacity

        anchors = torch.tensor([
            [2.3, -2.3, 2.3],
            [0.0, 4.2, 0.0],
            [4.0, 0.0, 0.0],
            [-2.3, 2.3, 2.3],
            [0.0, -4.1, 0.0],
            [2.3, 2.3, 2.3],
            [-4.2, 0.0, 0.0],
            [-2.3, -2.3, 2.3]
        ])
        anchors = torch.cat([anchors, -anchors], dim=0)
        densifier = VolumeDensifier(viewpoints=anchors)
        output = densifier.sample_points_in_volume(xyz=pos, scale=scale, rotation=rotation, opacity=opacity)

        assert output.ndim == 2
        assert output.shape[1] == 3
        check_tensor(output, dtype=dtype, device=device)
        validate_samples_inside_shell(output, pos)

    def test_sample_points_in_volume_no_clipping(self, gaussians, device, dtype):
        pos = gaussians['position'].cuda()                  # model.get_xyz
        scale = gaussians['scale'].cuda()                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].cuda()             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).cuda()    # model.get_opacity

        densifier = VolumeDensifier(clip_samples_to_input_bbox=False)
        output = densifier.sample_points_in_volume(xyz=pos, scale=scale, rotation=rotation, opacity=opacity)

        assert output.ndim == 2
        assert output.shape[1] == 3
        check_tensor(output, dtype=dtype, device=device)
