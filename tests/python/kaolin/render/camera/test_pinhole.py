# Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
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
from kaolin.render.camera import Camera, CameraExtrinsics
from kaolin.render.camera.intrinsics_pinhole import PinholeIntrinsics
from kaolin.utils.testing import FLOAT_TYPES, contained_torch_equal

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


class TestToFromDict:
    def test_from_dict(self):
        in_dict = {
                'width': 100,
                'height': 200,
                'focal_x': 80.0,
                'focal_y': 80.0,
                'x0': 5.0,
                'y0': 10.0,
                'near': 0.1,
                'far': 500.0,
            }
        expected_intrinsics = PinholeIntrinsics.from_focal(**in_dict)

        intrinsics = PinholeIntrinsics.from_dict(in_dict)
        in_dict['classname'] = 'pinhole'
        intrinsics2 = PinholeIntrinsics.from_dict(in_dict)

        assert contained_torch_equal(intrinsics, expected_intrinsics)
        assert contained_torch_equal(intrinsics2, expected_intrinsics)


    def test_to_dict(self):
        intrinsics = PinholeIntrinsics.from_focal(
            width=400,
            height=300,
            focal_x=400.0,
            focal_y=400.0,
            x0=10.0,
            y0=20.0,
            near=0.5,
            far=200.0,
            device='cpu',
            dtype=torch.float32,
        )
        intrinsics_dict = intrinsics.as_dict()
        assert intrinsics_dict['classname'] == 'pinhole'
        assert intrinsics_dict['width'] == 400
        assert intrinsics_dict['height'] == 300
        assert intrinsics_dict['x0'] == 10.0
        assert intrinsics_dict['y0'] == 20.0
        assert intrinsics_dict['near'] == 0.5
        assert intrinsics_dict['far'] == 200.0
        assert 'focal_x' in intrinsics_dict and 'focal_y' in intrinsics_dict

        # Camera with default properties; dict should still be valid and round-trippable.
        intrinsics_default = PinholeIntrinsics.from_fov(
            width=256,
            height=256,
            fov=np.pi / 4,
            device='cpu',
            dtype=torch.float32,
        )
        intrinsics_default_dict = intrinsics_default.as_dict()
        assert intrinsics_default_dict['classname'] == 'pinhole'
        assert intrinsics_default_dict['width'] == intrinsics_default.width
        assert intrinsics_default_dict['height'] == intrinsics_default.height
        assert intrinsics_default_dict['focal_x'] == intrinsics_default.focal_x
        assert intrinsics_default_dict['focal_y'] == intrinsics_default.focal_y
        assert intrinsics_default_dict['x0'] == intrinsics_default.x0
        assert intrinsics_default_dict['y0'] == intrinsics_default.y0
        assert intrinsics_default_dict['near'] == intrinsics_default.near
        assert intrinsics_default_dict['far'] == intrinsics_default.far

    def test_round_trip(self):
        # Round-trip 3 single cameras through as_dict / from_dict; properties must match.
        # Config 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        intrinsics_all = [PinholeIntrinsics.from_fov(
            width=320, height=240, fov=30 * np.pi / 180,
            device=device, dtype=torch.float32,  # we'll use mixed devices
        ),
        # Config 2
        PinholeIntrinsics.from_fov(
            width=800, height=600, fov=np.pi / 6,
            x0=50.0, y0=25.0,
            device='cpu'
        ),
        # Config 3: from_focal (focal-length-based) for round-trip coverage
        PinholeIntrinsics.from_focal(
            width=640, height=480,
            focal_x=520.0, focal_y=510.0,
            x0=15.0, y0=20.0,
            near=0.01, far=2000.0,
            device='cpu'
        )]

        for intrinsics in intrinsics_all:
            param_dict = intrinsics.as_dict()
            reconstructed = PinholeIntrinsics.from_dict(param_dict).to(intrinsics.device)
            param_dict2 = reconstructed.as_dict()

            assert contained_torch_equal(intrinsics, reconstructed)
            assert contained_torch_equal(param_dict, param_dict2)

            assert torch.allclose(reconstructed.params.to(device), intrinsics.params.to(device))
            assert reconstructed.width == intrinsics.width
            assert reconstructed.height == intrinsics.height
            assert reconstructed.near == intrinsics.near
            assert reconstructed.far == intrinsics.far
            assert reconstructed.x0 == intrinsics.x0
            assert reconstructed.y0 == intrinsics.y0
            assert reconstructed.focal_x == intrinsics.focal_x
            assert reconstructed.focal_y == intrinsics.focal_y