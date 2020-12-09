# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

import random
import pytest

import numpy as np
import torch

from kaolin.render.camera import perspective_camera, generate_perspective_projection, \
                                 rotate_translate_points, generate_rotate_translate_matrices


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("batch_size", [5])
@pytest.mark.parametrize("height", [256])
@pytest.mark.parametrize("width", [512])
class TestCamera:
    @pytest.fixture(autouse=True)
    def camera_pos(self, batch_size, device):
        # shape: (batch_size, 3)
        return torch.tensor([[0, 0, 4]], dtype=torch.float,
                            device=device).repeat(batch_size, 1)

    @pytest.fixture(autouse=True)
    def object_pos(self, batch_size, device):
        # shape: (batch_size, 3)
        return torch.tensor([[0, 0, 0]], dtype=torch.float,
                            device=device).repeat(batch_size, 1)

    @pytest.fixture(autouse=True)
    def camera_up(self, batch_size, device):
        # shape: (batch_size, 3)
        return torch.tensor([[0, 1, 0]], dtype=torch.float,
                            device=device).repeat(batch_size, 1)

    @pytest.fixture(autouse=True)
    def camera_fovy(self):
        # 2.5 means tan(fov angle)
        # tan(fov_y/2) = 2.5, fovy_y is around 45 deg
        angle = np.arctan(1.0 / 2.5) * 2
        return angle

    def test_camera_rot(self, batch_size, device, width, height, camera_pos,
                        object_pos, camera_up):
        # shape: (batch_size, 3, 3)
        mtx_rot, _ = generate_rotate_translate_matrices(
            camera_pos, object_pos, camera_up)
        mtx_rot2 = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                                dtype=torch.float,
                                device=device).repeat(batch_size, 1, 1)

        nRet = torch.allclose(mtx_rot,
                              mtx_rot2,
                              rtol=1e-05,
                              atol=1e-08,
                              equal_nan=False)
        assert nRet

    def test_camera_trans(self, batch_size, device, width, height, camera_pos,
                          object_pos, camera_up):
        # shape: (batch_size, 3, 1)
        _, mtx_trans = generate_rotate_translate_matrices(
            camera_pos, object_pos, camera_up)
        mtx_trans2 = torch.tensor([[0, 0, 4]],
                                  dtype=torch.float,
                                  device=device).repeat(batch_size, 1)
        nRet = torch.allclose(mtx_trans,
                              mtx_trans2,
                              rtol=1e-05,
                              atol=1e-08,
                              equal_nan=False)
        assert nRet

    def test_camera_proj(self, batch_size, height, width, device, camera_fovy):
        # shape: (3, 1)
        # we support arbitrary height and width
        mtx_proj = generate_perspective_projection(camera_fovy,
                                                   ratio=width / height)
        mtx_proj = mtx_proj.to(device)

        mtx_proj2 = torch.tensor([[2.5 / (width / height)], [2.5], [-1]],
                                 dtype=torch.float,
                                 device=device)
        nRet = torch.allclose(mtx_proj,
                              mtx_proj2,
                              rtol=1e-05,
                              atol=1e-08,
                              equal_nan=False)
        assert nRet
