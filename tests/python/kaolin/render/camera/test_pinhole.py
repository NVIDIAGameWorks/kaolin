# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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
import copy
import itertools
import numpy as np
import torch
import kaolin
from kaolin.render.camera import Camera
from kaolin.utils.testing import FLOAT_TYPES

_CAM_DATA_IDX = (0, 1, 2, 3, 4)

@pytest.fixture(params=itertools.product(_CAM_DATA_IDX, FLOAT_TYPES))
def camera_data(request):
    data_idx = request.param[0]
    device, dtype = request.param[1]
    camera = None
    if data_idx == 0:
        camera = Camera.from_args(view_matrix=torch.tensor(
            [[[-5.5742e-01, 1.3878e-17, -8.3023e-01, 0.0000e+00],
              [1.4097e-01, 9.8548e-01, -9.4651e-02, 0.0000e+00],
              [8.1817e-01, -1.6980e-01, -5.4933e-01, -2.0000e+00],
              [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]],

             [[9.6585e-01, 0.0000e+00, 2.5910e-01, 0.0000e+00],
              [1.8479e-01, 7.0098e-01, -6.8883e-01, 0.0000e+00],
              [-1.8163e-01, 7.1318e-01, 6.7704e-01, -2.0000e+00],
              [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]],

             [[-5.3161e-01, -3.4694e-18, 8.4699e-01, 0.0000e+00],
              [-5.7488e-02, 9.9769e-01, -3.6082e-02, 0.0000e+00],
              [-8.4504e-01, -6.7873e-02, -5.3038e-01, -2.0000e+00],
              [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]],
            device=device),
            width=256, height=256,
            fov=0.8232465982437134, dtype=dtype, device=device)
    elif data_idx == 1:
        camera = Camera.from_args(view_matrix=torch.tensor(
        [[[-5.5742e-01, 1.3878e-17, -8.3023e-01, 0.0000e+00],
          [1.4097e-01, 9.8548e-01, -9.4651e-02, 0.0000e+00],
          [8.1817e-01, -1.6980e-01, -5.4933e-01, -2.0000e+00],
          [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]],
        device=device),
        width=256, height=256,
        fov=0.8232465982437134, dtype=dtype, device=device)
    elif data_idx == 2:
        camera = Camera.from_args(
        eye=torch.tensor([4.0, 4.0, 4.0]),
        at=torch.tensor([0.0, 0.0, 0.0]),
        up=torch.tensor([0.0, 1.0, 0.0]),
        fov=30 * np.pi / 180,  # In radians
        width=800, height=800,
        dtype=dtype,
        device=device
    )
    elif data_idx == 3:
        camera = Camera.from_args(
            eye=torch.tensor([[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]),
            at=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            up=torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            fov=30 * np.pi / 180,  # In radians
            width=800, height=800,
            dtype=dtype,
            device=device
        )
    elif data_idx == 4:
        camera = Camera.from_args(
            eye=torch.tensor([[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]),
            at=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            up=torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            fov=30 * np.pi / 180,  # In radians
            x0=12,
            y0=23,
            width=800, height=800,
            dtype=dtype,
            device=device
        )
    return dict(camera=camera)


class TestPinhole:

    def test_project(self, camera_data):
        cam = camera_data['camera']
        # 2 input types supported by cameras
        vertices_b_3 = torch.rand((5, 3), device=cam.device, dtype=cam.dtype)
        vertices_c_b_3 = vertices_b_3.unsqueeze(0).expand(len(cam), 5, 3)
        project_result_b_4 = cam.project(cam.extrinsics.transform(vertices_b_3))
        project_result_c_b_4 = cam.project(cam.extrinsics.transform(vertices_c_b_3))

        # project() should give the same result regardless of input shape
        if len(cam) > 1:
            # for multiple cameras in batch, shape output always broadcasts to (C, B, 3)
            assert torch.allclose(project_result_c_b_4, project_result_b_4)
        else:
            # for single camera in batch, shape output depends on input (C, B, 3) or (B, 3)
            assert torch.allclose(project_result_c_b_4, project_result_b_4[None])

        # validate transform through direct view_projection_matrix:
        vertices_b_4 = kaolin.render.camera.intrinsics.up_to_homogeneous(vertices_b_3)
        view_projection = cam.view_projection_matrix()
        per_cam_mat_result = []
        # carefully perform test: for batched cameras multiply vectors per single camera matrix
        for cam_idx in range(len(cam)):
            mat_result = view_projection[cam_idx] @ vertices_b_4[:, :, None]  # 4x4 mat @ Bx4x1 vec = Bx4x1 vec
            mat_result = mat_result.squeeze(-1)
            per_cam_mat_result.append(mat_result)
        if len(cam) == 1:  # Single camera, result should be of shape (B,3)
            per_cam_mat_result = per_cam_mat_result[0]
        else:
            per_cam_mat_result = torch.stack(per_cam_mat_result)

        # check that transform and matrix multiplication yield the same result
        assert torch.allclose(per_cam_mat_result, project_result_b_4, rtol=1e-3, atol=1e-3)

        # intrinsics also accept (B,4) and (C,B,4) shapes, validate as well against (B,3) and (C,B,3)
        ext_transformed_vertices_b_3 = cam.extrinsics.transform(vertices_b_3)
        ext_transformed_vertices_b_4 = kaolin.render.camera.intrinsics.up_to_homogeneous(
            ext_transformed_vertices_b_3)
        ext_transformed_vertices_c_b_3 = cam.extrinsics.transform(vertices_c_b_3)
        ext_transformed_vertices_c_b_4 = kaolin.render.camera.intrinsics.up_to_homogeneous(
            ext_transformed_vertices_c_b_3)
        int_projected_b_3 = cam.intrinsics.project(ext_transformed_vertices_b_3)
        int_projected_b_4 = cam.intrinsics.project(ext_transformed_vertices_b_4)
        int_projected_c_b_3 = cam.intrinsics.project(ext_transformed_vertices_c_b_3)
        int_projected_c_b_4 = cam.intrinsics.project(ext_transformed_vertices_c_b_4)
        assert torch.allclose(int_projected_b_3, int_projected_b_4)
        assert torch.allclose(int_projected_b_3, int_projected_c_b_3)
        assert torch.allclose(int_projected_b_3, int_projected_c_b_4)

    def test_get_principal_point_properties(self, camera_data):
        cam = camera_data['camera']
        half_w, half_h = cam.width / 2, cam.height / 2
        assert (cam.cx == (half_w + cam.x0)).all()
        assert (cam.cy == (half_h + cam.y0)).all()

    def test_set_principal_point_properties(self, camera_data):
        cam = camera_data['camera']
        half_w, half_h = cam.width / 2, cam.height / 2
        cam.x0 += 67.0
        cam.y0 -= 45.0
        assert (cam.cx == (half_w + cam.x0)).all()
        assert (cam.cy == (half_h + cam.y0)).all()

    def test_set_width(self, camera_data):
        cam = camera_data['camera']

        focal_x = copy.deepcopy(cam.focal_x)
        focal_y = copy.deepcopy(cam.focal_y)
        fov_x = copy.deepcopy(cam.fov_x)
        fov_y = copy.deepcopy(cam.fov_y)
        width = cam.width
        height = cam.height
        cam.width *= 0.5
        assert (torch.allclose(cam.fov_x, fov_x, rtol=1e-3))
        assert (torch.allclose(cam.fov_y, fov_y, rtol=1e-3))
        assert (torch.allclose(cam.focal_x, (focal_x / 2), rtol=1e-3))
        assert (torch.allclose(cam.focal_y, focal_y, rtol=1e-5))
        assert (cam.width == (width * 0.5))
        assert (cam.height == height)

    def test_set_height(self, camera_data):
        cam = camera_data['camera']

        focal_x = copy.deepcopy(cam.focal_x)
        focal_y = copy.deepcopy(cam.focal_y)
        fov_x = copy.deepcopy(cam.fov_x)
        fov_y = copy.deepcopy(cam.fov_y)
        width = cam.width
        height = cam.height
        cam.height *= 0.5
        assert (torch.allclose(cam.fov_x, fov_x, rtol=1e-3))
        assert (torch.allclose(cam.fov_y, fov_y, rtol=1e-3))
        assert (torch.allclose(cam.focal_x, focal_x, rtol=1e-5))
        assert (torch.allclose(cam.focal_y, (focal_y / 2), rtol=1e-3))
        assert (cam.width == width)
        assert (cam.height == (height * 0.5))
