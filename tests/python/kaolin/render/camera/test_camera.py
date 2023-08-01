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
import itertools
import numpy as np
import torch
import kaolin
from kaolin.render.camera import Camera
from kaolin.utils.testing import FLOAT_TYPES


_CAM_DATA_IDX = (0, 1, 2, 3, 4, 5, 6)

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
    elif data_idx == 5:
        camera = Camera.from_args(
            eye=torch.tensor([4.0, 4.0, 4.0]),
            at=torch.tensor([0.0, 0.0, 0.0]),
            up=torch.tensor([0.0, 1.0, 0.0]),
            width=800, height=800,
            dtype=dtype,
            device=device
        )
    elif data_idx == 6:
        camera = Camera.from_args(
            eye=torch.tensor([[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]),
            at=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            up=torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            width=800, height=800,
            dtype=dtype,
            device=device
        )
    return dict(camera=camera)


class TestCameraTransforms:

    def test_transform(self, camera_data):
        cam = camera_data['camera']

        # check various camera types
        # 2 input types supported by cameras
        vertices_b_3 = torch.rand((5, 3), device=cam.device, dtype=cam.dtype)
        vertices_c_b_3 = vertices_b_3.unsqueeze(0).expand(len(cam), 5, 3)
        transform_result_b_3 = cam.transform(vertices_b_3)
        transform_result_c_b_3 = cam.transform(vertices_c_b_3)

        # transform() should give the same result regardless of input shape
        if len(cam) > 1:
            # for multiple cameras in batch, shape output always broadcasts to (C, B, 3)
            assert torch.allclose(transform_result_c_b_3, transform_result_b_3)
        else:
            # for single camera in batch, shape output depends on input (C, B, 3) or (B, 3)
            assert torch.allclose(transform_result_c_b_3, transform_result_b_3[None])

        # validate transform through direct view_projection_matrix:
        vertices_b_4 = kaolin.render.camera.intrinsics.up_to_homogeneous(vertices_b_3)
        vertices_c_b_4 = kaolin.render.camera.intrinsics.up_to_homogeneous(vertices_c_b_3)
        view_projection = cam.view_projection_matrix()
        per_cam_mat_result = []
        # carefully perform test: for batched cameras multiply vectors per single camera matrix
        for cam_idx in range(len(cam)):
            mat_result = view_projection[cam_idx] @ vertices_b_4[:, :, None]  # 4x4 mat @ Bx4x1 vec = Bx4x1 vec
            mat_result = mat_result.squeeze(-1)
            mat_result = kaolin.render.camera.intrinsics.down_from_homogeneous(mat_result)
            per_cam_mat_result.append(mat_result)
        if len(cam) == 1:  # Single camera, result should be of shape (B,3)
            per_cam_mat_result = per_cam_mat_result[0]
        else:
            per_cam_mat_result = torch.stack(per_cam_mat_result)

        # check that transform and matrix multiplication yield the same result
        assert torch.allclose(per_cam_mat_result, transform_result_b_3, rtol=1e-3, atol=1e-2)

        # intrinsics also accept (B,4) and (C,B,4) shapes, validate as well against (B,3) and (C,B,3)
        ext_transformed_vertices_b_3 = cam.extrinsics.transform(vertices_b_3)
        ext_transformed_vertices_b_4 = kaolin.render.camera.intrinsics.up_to_homogeneous(
            ext_transformed_vertices_b_3)
        ext_transformed_vertices_c_b_3 = cam.extrinsics.transform(vertices_c_b_3)
        ext_transformed_vertices_c_b_4 = kaolin.render.camera.intrinsics.up_to_homogeneous(
            ext_transformed_vertices_c_b_3)
        int_transformed_b_3 = cam.intrinsics.transform(ext_transformed_vertices_b_3)
        int_transformed_b_4 = cam.intrinsics.transform(ext_transformed_vertices_b_4)
        int_transformed_c_b_3 = cam.intrinsics.transform(ext_transformed_vertices_c_b_3)
        int_transformed_c_b_4 = cam.intrinsics.transform(ext_transformed_vertices_c_b_4)
        assert torch.allclose(int_transformed_b_3, int_transformed_b_4)
        assert torch.allclose(int_transformed_b_3, int_transformed_c_b_3)
        assert torch.allclose(int_transformed_b_3, int_transformed_c_b_4)


class TestCameraProperties:

    def test_set_width(self, camera_data):
        cam = camera_data['camera']

        width = cam.width
        height = cam.height
        cam.width *= 0.5
        assert (width / 2 == cam.width)
        assert (height == cam.height)

    def test_set_height(self, camera_data):
        cam = camera_data['camera']

        width = cam.width
        height = cam.height
        cam.height *= 0.5
        assert (height / 2 == cam.height)
        assert (width == cam.width)


class TestViewportMatrix:

    def test_viewport(self, camera_data):
        cam = camera_data['camera']

        C, B = len(cam), 100

        # vertices to (C, B, 4, 1)
        vertices = torch.rand((C, B, 1, 3), device=cam.device, dtype=cam.dtype)
        vertices = kaolin.render.camera.intrinsics.up_to_homogeneous(vertices)
        vertices = vertices.transpose(-1, -2)

        vp_matrix = cam.view_projection_matrix()[:,None].expand(C, B ,4, 4)
        viewport_matrix = cam.viewport_matrix()[:,None].expand(C, B ,4, 4)

        clip_coordinates = vp_matrix @ vertices
        ndc_coordinates = clip_coordinates / clip_coordinates[:,:, -1:]

        screen_space_coords = viewport_matrix @ ndc_coordinates
        ndc_coordinates = ndc_coordinates.squeeze(-1)
        x_clip = (ndc_coordinates[:, :, 0] >= -1) & (ndc_coordinates[:, :, 0] <= 1)
        y_clip = (ndc_coordinates[:, :,  1] >= -1) & (ndc_coordinates[:, :, 1] <= 1)
        z_clip = (ndc_coordinates[:, :, 2] >= cam.ndc_min) & (ndc_coordinates[:, :, 2] <= cam.ndc_max)

        expected_screen_coords_x = ((ndc_coordinates[:, :, 0] + 1) * (cam.width / 2)).unsqueeze(-1)
        expected_screen_coords_y = ((ndc_coordinates[:, :, 1] + 1) * (cam.height / 2)).unsqueeze(-1)
        assert torch.allclose(screen_space_coords[:, :, 0], expected_screen_coords_x, rtol=1e-3, atol=1e-4)
        assert torch.allclose(screen_space_coords[:, :, 1], expected_screen_coords_y, rtol=1e-3, atol=1e-4)

        verts_in_frustum = screen_space_coords[x_clip & y_clip & z_clip].squeeze(-1)
        assert (verts_in_frustum[:, 0:2] >= 0).all()
        assert (verts_in_frustum[:, 0] <= cam.width).all()
        assert (verts_in_frustum[:, 1] <= cam.height).all()
        assert (verts_in_frustum[:, 2] >= 0).all()
        assert (verts_in_frustum[:, 2] <= 1).all()
