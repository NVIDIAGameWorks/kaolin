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
import wget
import math
import torch
from pathlib import Path
import kaolin
from kaolin.ops.gaussian import sample_points_in_volume
from kaolin.utils.testing import check_tensor

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, os.pardir, os.pardir, os.pardir, 'samples', 'gsplats')
TEST_MODELS = ['dozer_minimal.pt']
BAD_TEST_MODELS = ['sparse_sphere']
SUPPORTED_GSPLATS_DEVICES = ['cuda']
SUPPORTED_GSPLATS_DTYPES = [torch.float32]

S3_MODEL_PATHS = [
    'https://nvidia-kaolin.s3.us-east-2.amazonaws.com/data/dozer_minimal.pt',
    'https://nvidia-kaolin.s3.us-east-2.amazonaws.com/data/ficus_minimal.pt',
    'https://nvidia-kaolin.s3.us-east-2.amazonaws.com/data/hotdog_minimal.pt',
    'https://nvidia-kaolin.s3.us-east-2.amazonaws.com/data/materials_minimal.pt',
]


@pytest.fixture(autouse=True, scope='class')
def setup():
    """ Fetches all large models from S3 before running the tests, and deletes them when done """
    os.makedirs(ROOT_DIR, exist_ok=True)
    downloaded_files = []
    try:
        for model_path in S3_MODEL_PATHS:
            local_asset_path = wget.download(model_path, out=ROOT_DIR)
            downloaded_files.append(local_asset_path)
        yield
    finally:
        for local_asset_path in downloaded_files:
            Path(local_asset_path).unlink(missing_ok=True)


def validate_samples_inside_shell(samples, gaussian_xyz):
    assert (gaussian_xyz.min(dim=0)[0] <= samples.min(dim=0)[0]).all()  # Check points are within shell
    assert (gaussian_xyz.max(dim=0)[0] >= samples.max(dim=0)[0]).all()  # Check points are within shell


@pytest.mark.parametrize('model_name', TEST_MODELS)
@pytest.mark.parametrize("device", SUPPORTED_GSPLATS_DEVICES)
@pytest.mark.parametrize("dtype", SUPPORTED_GSPLATS_DTYPES)
class TestVolumeDensifier:

    @pytest.fixture(autouse=True)
    def gaussians(self, model_name):
        model_path = os.path.join(ROOT_DIR, model_name)
        gaussian_fields = torch.load(os.path.abspath(model_path))
        gaussian_fields['model_name'] = model_name
        return gaussian_fields

    def test_sample_points_in_volume(self, gaussians, device, dtype):
        pos = gaussians['position'].to(device=device, dtype=dtype)                  # model.get_xyz
        scale = gaussians['scale'].to(device=device, dtype=dtype)                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].to(device=device, dtype=dtype)             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).to(device=device, dtype=dtype)    # model.get_opacity

        output = sample_points_in_volume(xyz=pos, scale=scale, rotation=rotation, opacity=opacity)

        assert output.ndim == 2
        assert output.shape[1] == 3
        check_tensor(output, dtype=dtype, device=device)
        validate_samples_inside_shell(output, pos)

    def test_sample_points_in_volume_for_density(self, gaussians, device, dtype):
        pos = gaussians['position'].to(device=device, dtype=dtype)                  # model.get_xyz
        scale = gaussians['scale'].to(device=device, dtype=dtype)                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].to(device=device, dtype=dtype)             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).to(device=device, dtype=dtype)    # model.get_opacity

        octree_level = 6
        output = sample_points_in_volume(xyz=pos, scale=scale, rotation=rotation, opacity=opacity,
                                         octree_level=octree_level, jitter=True)

        assert output.ndim == 2
        assert output.shape[1] == 3
        check_tensor(output, dtype=dtype, device=device)
        validate_samples_inside_shell(output, pos)

        N = output.shape[0]
        pairwise_dists = torch.linalg.norm(output[None] - output[:, None], dim=2)
        pairwise_dists += output.max() * torch.eye(N, dtype=dtype, device=device)
        minimal_nearest_neighbor_distance = pairwise_dists.min(dim=0)[0].min()
        maximal_nearest_neighbor_distance = pairwise_dists.min(dim=0)[0].max()

        # Points should not be equi-distanced
        assert minimal_nearest_neighbor_distance != maximal_nearest_neighbor_distance

        # Also check we don't perturb beyond cell boundary
        output = sample_points_in_volume(xyz=pos, scale=scale, rotation=rotation, opacity=opacity,
                                         octree_level=octree_level, jitter=False, post_scale_factor=1.0,
                                         clip_samples_to_input_bbox=True)
        jittered_output = kaolin.ops.gaussian.densifier._jitter(output, octree_level)
        spc_cell_length = 2.0 / 2**octree_level
        spc_diagonal_length = spc_cell_length * math.sqrt(3.0)
        max_allowed_pts_distance = 2.0 * spc_diagonal_length # two spc diagonals
        max_perturbation = torch.linalg.norm(jittered_output - output, dim=1).max()
        assert max_perturbation <= max_allowed_pts_distance

    def test_sample_points_in_volume_no_jitter_for_density(self, gaussians, device, dtype):
        pos = gaussians['position'].to(device=device, dtype=dtype)                  # model.get_xyz
        scale = gaussians['scale'].to(device=device, dtype=dtype)                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].to(device=device, dtype=dtype)             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).to(device=device, dtype=dtype)    # model.get_opacity

        octree_level = 6
        output = sample_points_in_volume(xyz=pos, scale=scale, rotation=rotation, opacity=opacity,
                                         octree_level=octree_level, jitter=False, post_scale_factor=1.0)

        assert output.ndim == 2
        assert output.shape[1] == 3
        check_tensor(output, dtype=dtype, device=device)
        validate_samples_inside_shell(output, pos)

        N = output.shape[0]
        pairwise_dists = torch.linalg.norm(output[None] - output[:, None], dim=2)
        pairwise_dists += output.max() * torch.eye(N, dtype=dtype, device=device)
        nearest_neighbor_distance = pairwise_dists.min(dim=0)[0]
        minimal_nearest_neighbor_distance = nearest_neighbor_distance.min()

        # most point should have nearest neighbor of same distance (except for few loners)
        assert torch.sum(torch.isclose(
            torch.full((output.shape[0],), minimal_nearest_neighbor_distance, device=device, dtype=dtype),
            nearest_neighbor_distance)) > 0.95 * output.shape[0]

    def test_sample_points_in_volume_for_num_samples(self, gaussians, device, dtype):
        pos = gaussians['position'].to(device=device, dtype=dtype)                  # model.get_xyz
        scale = gaussians['scale'].to(device=device, dtype=dtype)                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].to(device=device, dtype=dtype)             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).to(device=device, dtype=dtype)    # model.get_opacity

        output = sample_points_in_volume(
            xyz=pos, scale=scale, rotation=rotation, opacity=opacity, num_samples=5000
        )

        assert output.ndim == 2
        check_tensor(output, shape=(5000, 3), dtype=dtype, device=device)
        validate_samples_inside_shell(output, pos)

    def test_sample_points_in_volume_without_opacity_threshold(self, gaussians, device, dtype):
        pos = gaussians['position'].to(device=device, dtype=dtype)                  # model.get_xyz
        scale = gaussians['scale'].to(device=device, dtype=dtype)                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].to(device=device, dtype=dtype)             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).to(device=device, dtype=dtype)    # model.get_opacity

        output = sample_points_in_volume(
            xyz=pos, scale=scale, rotation=rotation, opacity=opacity, opacity_threshold=0.0
        )

        assert output.ndim == 2
        assert output.shape[1] == 3
        check_tensor(output, dtype=dtype, device=device)
        validate_samples_inside_shell(output, pos)

    def test_sample_points_in_volume_with_high_opacity_threshold(self, gaussians, device, dtype):
        pos = gaussians['position'].to(device=device, dtype=dtype)                  # model.get_xyz
        scale = gaussians['scale'].to(device=device, dtype=dtype)                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].to(device=device, dtype=dtype)             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).to(device=device, dtype=dtype)    # model.get_opacity

        output = sample_points_in_volume(
            xyz=pos, scale=scale, rotation=rotation, opacity=opacity, opacity_threshold=0.5
        )

        assert output.ndim == 2
        assert output.shape[1] == 3
        check_tensor(output, dtype=dtype, device=device)
        validate_samples_inside_shell(output, pos)

    def test_sample_points_in_volume_with_mask(self, gaussians, device, dtype):
        pos = gaussians['position'].to(device=device, dtype=dtype)                  # model.get_xyz
        scale = gaussians['scale'].to(device=device, dtype=dtype)                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].to(device=device, dtype=dtype)             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).to(device=device, dtype=dtype)    # model.get_opacity

        opacity_threshold = 0.35

        N = pos.shape[0]
        opacity_mask = opacity.reshape(-1) < opacity_threshold
        mask = pos.new_ones((N,), device=device, dtype=torch.bool)
        mask &= ~opacity_mask

        output1 = sample_points_in_volume(
            xyz=pos, scale=scale, rotation=rotation, opacity=opacity, mask=mask, opacity_threshold=0.0, jitter=False
        )

        output2 = sample_points_in_volume(
            xyz=pos, scale=scale, rotation=rotation, opacity=opacity, opacity_threshold=0.35, jitter=False
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
        pos = gaussians['position'].to(device=device, dtype=dtype)                  # model.get_xyz
        scale = gaussians['scale'].to(device=device, dtype=dtype)                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].to(device=device, dtype=dtype)             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).to(device=device, dtype=dtype)    # model.get_opacity

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
        output = sample_points_in_volume(xyz=pos, scale=scale, rotation=rotation, opacity=opacity, viewpoints=anchors)

        assert output.ndim == 2
        assert output.shape[1] == 3
        check_tensor(output, dtype=dtype, device=device)
        validate_samples_inside_shell(output, pos)

    def test_sample_points_in_volume_no_clipping(self, gaussians, device, dtype):
        pos = gaussians['position'].to(device=device, dtype=dtype)                  # model.get_xyz
        scale = gaussians['scale'].to(device=device, dtype=dtype)                   # post activation, i.e. model.get_scaling
        rotation = gaussians['rotation'].to(device=device, dtype=dtype)             # post activation, i.e. model.get_rotation
        opacity = gaussians['opacity'].squeeze(1).to(device=device, dtype=dtype)    # model.get_opacity

        output = sample_points_in_volume(xyz=pos, scale=scale, rotation=rotation, opacity=opacity,
                                         clip_samples_to_input_bbox=False)

        assert output.ndim == 2
        assert output.shape[1] == 3
        check_tensor(output, dtype=dtype, device=device)
