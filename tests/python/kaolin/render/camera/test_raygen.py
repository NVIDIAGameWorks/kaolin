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
import copy
import itertools
import numpy as np
import torch
import kaolin
from kaolin.render.camera import \
    (
        Camera, \
        generate_default_grid, \
        generate_centered_pixel_coords, \
        generate_centered_custom_resolution_pixel_coords, \
        generate_pinhole_rays, \
        generate_ortho_rays, \
        generate_rays
    )
from kaolin.utils.testing import FLOAT_TYPES, check_tensor

_BATCHED_PINHOLE_CAM_DATA_IDX = (0, 1, 2)
_PINHOLE_CAM_DATA_IDX = (3, 4)
_BATCHED_ORTHO_CAM_DATA_IDX = (5, 6, 7)
_ORTHO_CAM_DATA_IDX = (8, 9)


@pytest.fixture(params=itertools.product(_BATCHED_PINHOLE_CAM_DATA_IDX, FLOAT_TYPES))
def batched_pinhole_camera_data(request):
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
            eye=torch.tensor([[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]),
            at=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            up=torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            fov=30 * np.pi / 180,  # In radians
            width=800, height=800,
            dtype=dtype,
            device=device
        )
    return dict(camera=camera)

@pytest.fixture(params=itertools.product(_PINHOLE_CAM_DATA_IDX, FLOAT_TYPES))
def pinhole_camera_data(request):
    data_idx = request.param[0]
    device, dtype = request.param[1]
    camera = None
    ray_indices = [1, 7, 38, 123, 1042, 10101, 15739]
    ray_origins = None
    ray_dirs = None
    if data_idx == 3:
        camera = Camera.from_args(
            eye=torch.tensor([4.0, 4.0, 4.0]),
            at=torch.tensor([0.0, 0.0, 0.0]),
            up=torch.tensor([0.0, 1.0, 0.0]),
            fov=30 * np.pi / 180,  # In radians
            width=800, height=800,
            dtype=dtype,
            device=device
        )
        ray_origins = torch.tensor(
            [[4.0000, 4.0000, 4.0000],
            [4.0000, 4.0000, 4.0000],
            [4.0000, 4.0000, 4.0000],
            [4.0000, 4.0000, 4.0000],
            [4.0000, 4.0000, 4.0000],
            [4.0000, 4.0000, 4.0000],
            [4.0000, 4.0000, 4.0000]],
            dtype = dtype,
            device = device
        )
        ray_dirs = torch.tensor(
            [[-0.8188, -0.3357, -0.4657],
            [-0.8169, -0.3360, -0.4688],
            [-0.8069, -0.3375, -0.4848],
            [-0.7774, -0.3412, -0.5284],
            [-0.7314, -0.3454, -0.5880],
            [-0.6135, -0.3529, -0.7064],
            [-0.5938, -0.3563, -0.7214]],
            dtype=dtype,
            device=device
        )
    elif data_idx == 4:
        camera = Camera.from_args(
            eye=torch.tensor([4.0, 4.0, 4.0]),
            at=torch.tensor([0.0, 0.0, 0.0]),
            up=torch.tensor([0.0, 1.0, 0.0]),
            fov=30 * np.pi / 180,  # In radians
            width=300, height=800,
            dtype=dtype,
            device=device
        )
        ray_origins = torch.tensor(
            [[4.0000, 4.0000, 4.0000],
            [4.0000, 4.0000, 4.0000],
            [4.0000, 4.0000, 4.0000],
            [4.0000, 4.0000, 4.0000],
            [4.0000, 4.0000, 4.0000],
            [4.0000, 4.0000, 4.0000],
            [4.0000, 4.0000, 4.0000]],
            dtype = dtype,
            device = device
        )
        ray_dirs = torch.tensor(
            [[-0.7279, -0.3451, -0.5926],
             [-0.7254, -0.3452, -0.5955],
             [-0.7124, -0.3457, -0.6107],
             [-0.6753, -0.3466, -0.6510],
             [-0.6662, -0.3484, -0.6594],
             [-0.6340, -0.3658, -0.6813],
             [-0.6597, -0.3772, -0.6500]],
            dtype=dtype,
            device=device
        )
    return dict(camera=camera, ray_indices=ray_indices, ray_origins=ray_origins, ray_dirs=ray_dirs)

@pytest.fixture(params=itertools.product(_BATCHED_ORTHO_CAM_DATA_IDX, FLOAT_TYPES))
def batched_ortho_camera_data(request):
    data_idx = request.param[0]
    device, dtype = request.param[1]
    camera = None
    if data_idx == 5:
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
            fov_distance=0.8232465982437134, dtype=dtype, device=device)
    elif data_idx == 6:
        camera = Camera.from_args(view_matrix=torch.tensor(
        [[[-5.5742e-01, 1.3878e-17, -8.3023e-01, 0.0000e+00],
          [1.4097e-01, 9.8548e-01, -9.4651e-02, 0.0000e+00],
          [8.1817e-01, -1.6980e-01, -5.4933e-01, -2.0000e+00],
          [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]]],
        device=device),
        width=256, height=256,
        fov_distance=0.8232465982437134, dtype=dtype, device=device)
    elif data_idx == 7:
        camera = Camera.from_args(
            eye=torch.tensor([[4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]),
            at=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            up=torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]),
            fov_distance=1.0,  # In radians
            width=800, height=800,
            dtype=dtype,
            device=device
        )
    return dict(camera=camera)

@pytest.fixture(params=itertools.product(_ORTHO_CAM_DATA_IDX, FLOAT_TYPES))
def ortho_camera_data(request):
    data_idx = request.param[0]
    device, dtype = request.param[1]
    camera = None
    ray_indices = [1, 7, 38, 123, 1042, 10101, 15739]
    ray_origins = None
    ray_dirs = None
    if data_idx == 8:
        camera = Camera.from_args(
            eye=torch.tensor([4.0, 4.0, 4.0]),
            at=torch.tensor([0.0, 0.0, 0.0]),
            up=torch.tensor([0.0, 1.0, 0.0]),
            fov_distance=1.0,
            width=800, height=800,
            dtype=dtype,
            device=device
        )
        ray_origins = torch.tensor(
            [[2.8878, 4.8155, 4.2967],
            [2.8984, 4.8155, 4.2861],
            [2.9532, 4.8155, 4.2313],
            [3.1035, 4.8155, 4.0810],
            [3.3149, 4.8134, 3.8717],
            [3.7839, 4.7910, 3.4251],
            [3.8583, 4.7767, 3.3651]],
            dtype = dtype,
            device = device
        )
        ray_dirs = torch.tensor(
            [[-0.5774, -0.5774, -0.5774],
            [-0.5774, -0.5774, -0.5774],
            [-0.5774, -0.5774, -0.5774],
            [-0.5774, -0.5774, -0.5774],
            [-0.5774, -0.5774, -0.5774],
            [-0.5774, -0.5774, -0.5774],
            [-0.5774, -0.5774, -0.5774]],
            dtype=dtype,
            device=device
        )
    elif data_idx == 9:
        camera = Camera.from_args(
            eye=torch.tensor([4.0, 4.0, 4.0]),
            at=torch.tensor([0.0, 0.0, 0.0]),
            up=torch.tensor([0.0, 1.0, 0.0]),
            fov_distance=0.9,
            width=300, height=800,
            dtype=dtype,
            device=device
        )
        ray_origins = torch.tensor(
            [[3.3968, 4.7339, 3.8693],
            [3.4063, 4.7339, 3.8598],
            [3.4556, 4.7339, 3.8104],
            [3.5909, 4.7339, 3.6752],
            [3.6239, 4.7284, 3.6477],
            [3.7453, 4.6733, 3.5814],
            [3.6641, 4.6384, 3.6975]],
            dtype = dtype,
            device = device
        )
        ray_dirs = torch.tensor(
            [[-0.5774, -0.5774, -0.5774],
            [-0.5774, -0.5774, -0.5774],
            [-0.5774, -0.5774, -0.5774],
            [-0.5774, -0.5774, -0.5774],
            [-0.5774, -0.5774, -0.5774],
            [-0.5774, -0.5774, -0.5774],
            [-0.5774, -0.5774, -0.5774]],
            dtype=dtype,
            device=device
        )
    return dict(camera=camera, ray_indices=ray_indices, ray_origins=ray_origins, ray_dirs=ray_dirs)


class TestPinholeRaygen:

    def test_generate_pinhole_rays(self, pinhole_camera_data):
        cam = pinhole_camera_data['camera']
        ray_origins, ray_dirs = generate_pinhole_rays(cam)
        check_tensor(ray_origins, shape=(cam.height * cam.width, 3), dtype=cam.dtype, device=cam.device.type)
        check_tensor(ray_dirs, shape=(cam.height * cam.width, 3), dtype=cam.dtype, device=cam.device.type)

    def test_generate_pinhole_rays_coords(self, pinhole_camera_data):
        cam = pinhole_camera_data['camera']
        coords_grid = (
            torch.tensor([[0.5, 0.5], [cam.height - 0.5, cam.height - 0.5]], dtype=cam.dtype, device=cam.device),
            torch.tensor([[0.5, cam.width - 0.5], [0.5, cam.width - 0.5]], dtype=cam.dtype, device=cam.device)
        )
        ray_origins, ray_dirs = generate_pinhole_rays(cam, coords_grid)
        check_tensor(ray_origins, shape=(4, 3), dtype=cam.dtype, device=cam.device.type)
        check_tensor(ray_dirs, shape=(4, 3), dtype=cam.dtype, device=cam.device.type)

        # Check the values are correct
        all_ray_origins, all_ray_dirs = generate_pinhole_rays(cam)

        assert torch.isclose(ray_dirs[0], all_ray_dirs[0]).all()
        assert torch.isclose(ray_origins[0], all_ray_origins[0]).all()
        assert torch.isclose(ray_dirs[1], all_ray_dirs[cam.width-1]).all()
        assert torch.isclose(ray_origins[1], all_ray_origins[cam.width-1]).all()
        assert torch.isclose(ray_dirs[2], all_ray_dirs[-cam.width]).all()
        assert torch.isclose(ray_origins[2], all_ray_origins[-cam.width]).all()
        assert torch.isclose(ray_dirs[3], all_ray_dirs[-1]).all()
        assert torch.isclose(ray_origins[3], all_ray_origins[-1]).all()

    def test_generate_pinhole_rays_batched(self, batched_pinhole_camera_data):
        cam = batched_pinhole_camera_data['camera']
        try:
            _, _ = generate_pinhole_rays(cam)
            assert False
        except AssertionError:
            pass    # If triggers an assert - test passes


class TestOrthoRaygen:

    def test_generate_ortho_rays(self, ortho_camera_data):
        cam = ortho_camera_data['camera']
        ray_origins, ray_dirs = generate_ortho_rays(cam)
        check_tensor(ray_origins, shape=(cam.height * cam.width, 3), dtype=cam.dtype, device=cam.device.type)
        check_tensor(ray_dirs, shape=(cam.height * cam.width, 3), dtype=cam.dtype, device=cam.device.type)

    def test_generate_pinhole_rays_coords(self, ortho_camera_data):
        cam = ortho_camera_data['camera']
        coords_grid = (
            torch.tensor([[0.5, 0.5], [cam.height - 0.5, cam.height - 0.5]], dtype=cam.dtype, device=cam.device),
            torch.tensor([[0.5, cam.width - 0.5], [0.5, cam.width - 0.5]], dtype=cam.dtype, device=cam.device)
        )
        ray_origins, ray_dirs = generate_ortho_rays(cam, coords_grid)
        check_tensor(ray_origins, shape=(4, 3), dtype=cam.dtype, device=cam.device.type)
        check_tensor(ray_dirs, shape=(4, 3), dtype=cam.dtype, device=cam.device.type)

        # Check the values are correct
        all_ray_origins, all_ray_dirs = generate_ortho_rays(cam)

        assert torch.isclose(ray_dirs[0], all_ray_dirs[0]).all()
        assert torch.isclose(ray_origins[0], all_ray_origins[0]).all()
        assert torch.isclose(ray_dirs[1], all_ray_dirs[cam.width-1]).all()
        assert torch.isclose(ray_origins[1], all_ray_origins[cam.width-1]).all()
        assert torch.isclose(ray_dirs[2], all_ray_dirs[-cam.width]).all()
        assert torch.isclose(ray_origins[2], all_ray_origins[-cam.width]).all()
        assert torch.isclose(ray_dirs[3], all_ray_dirs[-1]).all()
        assert torch.isclose(ray_origins[3], all_ray_origins[-1]).all()

    def test_generate_pinhole_rays_batched(self, batched_ortho_camera_data):
        cam = batched_ortho_camera_data['camera']
        try:
            _, _ = generate_ortho_rays(cam)
            assert False
        except AssertionError:
            pass    # If triggers an assert - test passes


class TestPixelGrid:

    def test_generate_centered_pixel_coords(self, pinhole_camera_data):
        cam = pinhole_camera_data['camera']
        pixel_y, pixel_x = generate_centered_pixel_coords(cam.width, cam.height, device=cam.device)
        coords_grid = (
            torch.tensor([[0.5, 0.5], [cam.height - 0.5, cam.height - 0.5]], dtype=pixel_y.dtype, device=pixel_y.device),
            torch.tensor([[0.5, cam.width - 0.5], [0.5, cam.width - 0.5]], dtype=pixel_x.dtype, device=pixel_x.device)
        )

        assert torch.isclose(pixel_y[0][0], coords_grid[0][0][0]).all()
        assert torch.isclose(pixel_x[0][0], coords_grid[1][0][0]).all()
        assert torch.isclose(pixel_y[0][cam.width-1], coords_grid[0][0][1]).all()
        assert torch.isclose(pixel_x[0][cam.width-1], coords_grid[1][0][1]).all()
        assert torch.isclose(pixel_y[cam.height-1][0], coords_grid[0][1][0]).all()
        assert torch.isclose(pixel_x[cam.height-1][0], coords_grid[1][1][0]).all()
        assert torch.isclose(pixel_y[cam.height-1][cam.width-1], coords_grid[0][1][1]).all()
        assert torch.isclose(pixel_x[cam.height-1][cam.width-1], coords_grid[1][1][1]).all()

    def test_generate_centered_custom_resolution_pixel_coords(self, pinhole_camera_data):
        cam = pinhole_camera_data['camera']
        pixel_y, pixel_x = generate_centered_custom_resolution_pixel_coords(cam.width, cam.height, device=cam.device)
        coords_grid = (
            torch.tensor([[0.5, 0.5], [cam.height - 0.5, cam.height - 0.5]], dtype=pixel_y.dtype, device=pixel_y.device),
            torch.tensor([[0.5, cam.width - 0.5], [0.5, cam.width - 0.5]], dtype=pixel_x.dtype, device=pixel_x.device)
        )

        assert torch.isclose(pixel_y[0][0], coords_grid[0][0][0]).all()
        assert torch.isclose(pixel_x[0][0], coords_grid[1][0][0]).all()
        assert torch.isclose(pixel_y[0][cam.width-1], coords_grid[0][0][1]).all()
        assert torch.isclose(pixel_x[0][cam.width-1], coords_grid[1][0][1]).all()
        assert torch.isclose(pixel_y[cam.height-1][0], coords_grid[0][1][0]).all()
        assert torch.isclose(pixel_x[cam.height-1][0], coords_grid[1][1][0]).all()
        assert torch.isclose(pixel_y[cam.height-1][cam.width-1], coords_grid[0][1][1]).all()
        assert torch.isclose(pixel_x[cam.height-1][cam.width-1], coords_grid[1][1][1]).all()

    def test_generate_centered_custom_resolution_pixel_coords_half_res(self, pinhole_camera_data):
        cam = pinhole_camera_data['camera']
        res_x = cam.width//2
        res_y = cam.height // 2
        pixel_y, pixel_x = generate_centered_custom_resolution_pixel_coords(cam.width, cam.height,
                                                                            res_x, res_y,
                                                                            device=cam.device)
        check_tensor(pixel_y, shape=(res_y, res_x), dtype=torch.float32, device=cam.device.type)
        check_tensor(pixel_x, shape=(res_y, res_x), dtype=torch.float32, device=cam.device.type)

        coords_grid = (
            torch.tensor([[1.0, 1.0], [cam.height - 1.0, cam.height - 1.0]], dtype=pixel_y.dtype, device=pixel_y.device),
            torch.tensor([[1.0, cam.width - 1.0], [1.0, cam.width - 1.0]], dtype=pixel_x.dtype, device=pixel_x.device)
        )
        assert torch.isclose(pixel_y[0][0], coords_grid[0][0][0]).all()
        assert torch.isclose(pixel_x[0][0], coords_grid[1][0][0]).all()
        assert torch.isclose(pixel_y[0][res_x-1], coords_grid[0][0][1]).all()
        assert torch.isclose(pixel_x[0][res_x-1], coords_grid[1][0][1]).all()
        assert torch.isclose(pixel_y[res_y-1][0], coords_grid[0][1][0]).all()
        assert torch.isclose(pixel_x[res_y-1][0], coords_grid[1][1][0]).all()
        assert torch.isclose(pixel_y[res_y-1][res_x-1], coords_grid[0][1][1]).all()
        assert torch.isclose(pixel_x[res_y-1][res_x-1], coords_grid[1][1][1]).all()


class TestGenerateRays:

    def test_generate_rays_pinhole(self, pinhole_camera_data):
        cam = pinhole_camera_data['camera']
        ray_o, ray_d = generate_rays(cam)
        equal_ray_o, equal_ray_d = cam.generate_rays()

        assert torch.allclose(ray_o, equal_ray_o)
        assert torch.allclose(ray_d, equal_ray_d)

        expected_ray_indices = pinhole_camera_data['ray_indices']
        expected_ray_origins = pinhole_camera_data['ray_origins']
        expected_ray_dirs = pinhole_camera_data['ray_dirs']

        rtol = 1e-3
        if cam.dtype == torch.float16:
            rtol = 1e-2
        assert torch.allclose(ray_o[expected_ray_indices], expected_ray_origins, rtol=rtol)
        assert torch.allclose(ray_d[expected_ray_indices], expected_ray_dirs, rtol=rtol)

    def test_generate_rays_ortho(self, ortho_camera_data):
        cam = ortho_camera_data['camera']
        ray_o, ray_d = generate_rays(cam)
        equal_ray_o, equal_ray_d = cam.generate_rays()

        assert torch.allclose(ray_o, equal_ray_o)
        assert torch.allclose(ray_d, equal_ray_d)

        expected_ray_indices = ortho_camera_data['ray_indices']
        expected_ray_origins = ortho_camera_data['ray_origins']
        expected_ray_dirs = ortho_camera_data['ray_dirs']
        rtol = 1e-3
        if cam.dtype == torch.float16:
            rtol = 1e-2
        assert torch.allclose(ray_o[expected_ray_indices], expected_ray_origins, rtol=rtol)
        assert torch.allclose(ray_d[expected_ray_indices], expected_ray_dirs, rtol=rtol)