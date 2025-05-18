# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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
import numpy as np
import torch
import kaolin
from kaolin.render.camera import Camera, camera_path_generator, loop_camera_path_generator
from kaolin.utils.testing import FLOAT_TYPES


@pytest.fixture(params=FLOAT_TYPES)
def ext_camera_data(request):
    device, dtype = request.param
    key_camera_1 = Camera.from_args(
        eye=[4.0, 4.0, 4.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
        fov=np.deg2rad(45), width=800, height=800,
        device=device, dtype=dtype
    )
    key_camera_2 = Camera.from_args(
        eye=[8.0, 4.0, 4.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
        fov=np.deg2rad(45), width=800, height=800,
        device=device, dtype=dtype
    )
    key_camera_3 = Camera.from_args(
        eye=[4.0, 8.0, 4.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
        fov=np.deg2rad(45), width=800, height=800,
        device=device, dtype=dtype
    )
    key_camera_4 = Camera.from_args(
        eye=[4.0, 4.0, 8.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
        fov=np.deg2rad(45), width=800, height=800,
        device=device, dtype=dtype
    )
    return [key_camera_1, key_camera_2, key_camera_3, key_camera_4]


@pytest.fixture(params=FLOAT_TYPES)
def int_camera_data(request):
    device, dtype = request.param
    key_camera_1 = Camera.from_args(
        eye=[4.0, 4.0, 4.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
        fov=np.deg2rad(45), width=800, height=800,
        device=device, dtype=dtype
    )
    key_camera_2 = Camera.from_args(
        eye=[4.0, 4.0, 4.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
        fov=np.deg2rad(60), width=800, height=800,
        device=device, dtype=dtype
    )
    key_camera_3 = Camera.from_args(
        eye=[4.0, 4.0, 4.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
        fov=np.deg2rad(85), width=800, height=800,
        device=device, dtype=dtype
    )
    key_camera_4 = Camera.from_args(
        eye=[4.0, 4.0, 4.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
        fov=np.deg2rad(35), width=800, height=800,
        device=device, dtype=dtype
    )
    return [key_camera_1, key_camera_2, key_camera_3, key_camera_4]


@pytest.fixture(params=FLOAT_TYPES)
def ortho_camera_data(request):
    device, dtype = request.param
    key_camera_1 = Camera.from_args(
        eye=[4.0, 4.0, 4.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
        fov_distance=1.0, width=800, height=800,
        device=device, dtype=dtype
    )
    key_camera_2 = Camera.from_args(
        eye=[4.0, 4.0, 4.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
        fov_distance=5.0, width=800, height=800,
        device=device, dtype=dtype
    )
    key_camera_3 = Camera.from_args(
        eye=[4.0, 4.0, 4.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
        fov_distance=10.0, width=800, height=800,
        device=device, dtype=dtype
    )
    key_camera_4 = Camera.from_args(
        eye=[4.0, 4.0, 4.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
        fov_distance=5.0, width=800, height=800,
        device=device, dtype=dtype
    )
    return [key_camera_1, key_camera_2, key_camera_3, key_camera_4]


@pytest.fixture(params=FLOAT_TYPES)
def mixed_camera_data(request):
    device, dtype = request.param
    key_camera_1 = Camera.from_args(
        eye=[4.0, 4.0, 4.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
        fov=np.deg2rad(45), width=800, height=800,
        device=device, dtype=dtype
    )
    key_camera_2 = Camera.from_args(
        eye=[8.0, 4.0, 4.0], at=[0.0, 1.0, 0.0], up=[0.0, 1.0, 0.0],
        fov=np.deg2rad(60), width=800, height=800,
        device=device, dtype=dtype
    )
    key_camera_3 = Camera.from_args(
        eye=[4.0, 8.0, 4.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
        fov=np.deg2rad(45), width=800, height=800,
        device=device, dtype=dtype
    )
    key_camera_4 = Camera.from_args(
        eye=[4.0, 4.0, 8.0], at=[0.0, 1.0, 0.0], up=[0.0, 1.0, 0.0],
        fov=np.deg2rad(60), width=800, height=800,
        device=device, dtype=dtype
    )
    key_camera_5 = Camera.from_args(
        eye=[4.0, 4.0, 8.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
        fov=np.deg2rad(45), width=800, height=800,
        device=device, dtype=dtype
    )
    return [key_camera_1, key_camera_2, key_camera_3, key_camera_4, key_camera_5]


class TestCameraInterpolations:

    def test_valid_inputs(self):
        cam = Camera.from_args(
            eye=[4.0, 4.0, 4.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
            fov=np.deg2rad(45), width=800, height=800,
            device='cuda'
        )

        with pytest.raises(ValueError, match=r"Unknown interpolation function specified."):
            cam_path = camera_path_generator(
                trajectory=[cam, cam, cam, cam],
                frames_between_cameras=60,
                interpolation="pol"
            )
            next(cam_path)

        with pytest.raises(ValueError, match=r"For polynomial interpolation, cameras trajectory must have at least 2 cameras."):
            cam_path = camera_path_generator(
                trajectory=[cam],
                frames_between_cameras=60,
                interpolation="polynomial"
            )
            next(cam_path)

        with pytest.raises(ValueError, match=r"For catmull_rom interpolation, cameras trajectory must have at least 4 cameras."):
            cam_path = camera_path_generator(
                trajectory=[cam, cam, cam],
                frames_between_cameras=60,
                interpolation="catmull_rom"
            )
            next(cam_path)


    def test_incompatible_device(self):

        key_camera_1 = Camera.from_args(
            eye=[4.0, 4.0, 4.0], at=[0.0, 0.0, 0.0], up=[0.0, 1.0, 0.0],
            fov=np.deg2rad(45), width=800, height=800,
            device='cuda'
        )
        key_camera_2 = Camera.from_args(
            eye=[8.0, 4.0, 4.0], at=[0.0, 1.0, 0.0], up=[0.0, 1.0, 0.0],
            fov=np.deg2rad(60), width=800, height=800,
            device='cpu'
        )

        with pytest.raises(RuntimeError, match=r"Expected all tensors to be on the same device"):
            cam_path = camera_path_generator(
                trajectory=[key_camera_1, key_camera_2],
                frames_between_cameras=60,
                interpolation="polynomial"
            )
            next(cam_path)


    def test_count_polynomial(self, mixed_camera_data):
        cam_path = camera_path_generator(
            trajectory=mixed_camera_data,
            frames_between_cameras=1,
            interpolation="polynomial"
        )
        count = 0
        for _ in cam_path:
            count += 1

        assert count == 9   # = (len(traj) - 1) * (1+1) + 1

        cam_path = camera_path_generator(
            trajectory=mixed_camera_data,
            frames_between_cameras=5,
            interpolation="polynomial"
        )
        count = 0
        for _ in cam_path:
            count += 1

        assert count == 25  # = (len(traj) - 1) * (5+1) + 1

    def test_count_catmull_rom(self, mixed_camera_data):
        cam_path = camera_path_generator(
            trajectory=mixed_camera_data,
            frames_between_cameras=1,
            interpolation="catmull_rom"
        )
        count = 0
        for _ in cam_path:
            count += 1
        assert count == 9  # = (len(traj) - 1) * (1+1) + 1

        cam_path = camera_path_generator(
            trajectory=mixed_camera_data,
            frames_between_cameras=5,
            interpolation="catmull_rom"
        )
        count = 0
        for _ in cam_path:
            count += 1

        assert count == 25  # = (len(traj) - 1) * (5+1) + 1

    def test_count_loop_polynomial(self, mixed_camera_data):
        cam_path = loop_camera_path_generator(
            trajectory=mixed_camera_data,
            frames_between_cameras=1,
            interpolation="polynomial",
            repeat=1
        )
        count = 0
        for _ in cam_path:
            count += 1
        assert count == 11  # = repeat * len(traj) * (1+1) + 1

        cam_path = loop_camera_path_generator(
            trajectory=mixed_camera_data,
            frames_between_cameras=1,
            interpolation="polynomial",
            repeat=2
        )
        count = 0
        for _ in cam_path:
            count += 1
        assert count == 21  # = repeat * (len(traj)) * (1+1) + 1

        cam_path = loop_camera_path_generator(
            trajectory=mixed_camera_data,
            frames_between_cameras=1,
            interpolation="polynomial",
            repeat=None
        )
        count = 0
        for _ in cam_path:
            count += 1
            if count == 100: # break infinite loop
                break
        assert count == 100

    def test_count_loop_catmull_rom(self, mixed_camera_data):
        cam_path = loop_camera_path_generator(
            trajectory=mixed_camera_data,
            frames_between_cameras=1,
            interpolation="catmull_rom",
            repeat=1
        )
        count = 0
        for _ in cam_path:
            count += 1
        assert count == 11  # = repeat * len(traj) * (1+1) + 1

        cam_path = loop_camera_path_generator(
            trajectory=mixed_camera_data,
            frames_between_cameras=1,
            interpolation="catmull_rom",
            repeat=2
        )
        count = 0
        for _ in cam_path:
            count += 1
        assert count == 21  # = repeat * (len(traj)) * (1+1) + 1

        cam_path = loop_camera_path_generator(
            trajectory=mixed_camera_data,
            frames_between_cameras=1,
            interpolation="catmull_rom",
            repeat=None
        )
        count = 0
        for _ in cam_path:
            count += 1
            if count == 100: # break infinite loop
                break
        assert count == 100

    def test_keyframes_polynomial_ext(self, ext_camera_data):
        cam_path = camera_path_generator(
            trajectory=ext_camera_data,
            frames_between_cameras=10,
            interpolation="polynomial"
        )
        path = [cam for cam in cam_path]

        if ext_camera_data[0].dtype == torch.float16:
            atol = 1e-2
        else:
            atol = 1e-4
        assert torch.allclose(path[0], ext_camera_data[0], atol=atol)
        assert torch.allclose(path[11], ext_camera_data[1], atol=atol)
        assert torch.allclose(path[22], ext_camera_data[2], atol=atol)
        assert torch.allclose(path[33], ext_camera_data[3], atol=atol)

    def test_keyframes_catmull_rom_ext(self, ext_camera_data):
        cam_path = camera_path_generator(
            trajectory=ext_camera_data,
            frames_between_cameras=10,
            interpolation="catmull_rom"
        )
        path = [cam for cam in cam_path]

        if ext_camera_data[0].dtype == torch.float16:
            atol = 1e-2
        else:
            atol = 1e-4
        assert torch.allclose(path[0], ext_camera_data[0], atol=atol)
        assert torch.allclose(path[11], ext_camera_data[1], atol=atol)
        assert torch.allclose(path[22], ext_camera_data[2], atol=atol)
        assert torch.allclose(path[33], ext_camera_data[3], atol=atol)

    def test_keyframes_polynomial_pinhole_int(self, int_camera_data):
        cam_path = camera_path_generator(
            trajectory=int_camera_data,
            frames_between_cameras=10,
            interpolation="polynomial"
        )
        path = [cam for cam in cam_path]

        if int_camera_data[0].dtype == torch.float16:
            atol = 1e-2
        else:
            atol = 1e-4
        assert torch.allclose(path[0], int_camera_data[0], atol=atol)
        assert torch.allclose(path[11], int_camera_data[1], atol=atol)
        assert torch.allclose(path[22], int_camera_data[2], atol=atol)
        assert torch.allclose(path[33], int_camera_data[3], atol=atol)

    def test_keyframes_catmull_rom_pinhole_int(self, int_camera_data):
        cam_path = camera_path_generator(
            trajectory=int_camera_data,
            frames_between_cameras=10,
            interpolation="catmull_rom"
        )
        path = [cam for cam in cam_path]

        if int_camera_data[0].dtype == torch.float16:
            atol = 1e-2
        else:
            atol = 1e-4
        assert torch.allclose(path[0], int_camera_data[0], atol=atol)
        assert torch.allclose(path[11], int_camera_data[1], atol=atol)
        assert torch.allclose(path[22], int_camera_data[2], atol=atol)
        assert torch.allclose(path[33], int_camera_data[3], atol=atol)

    def test_keyframes_polynomial_ortho_int(self, ortho_camera_data):
        cam_path = camera_path_generator(
            trajectory=ortho_camera_data,
            frames_between_cameras=10,
            interpolation="polynomial"
        )
        path = [cam for cam in cam_path]

        if ortho_camera_data[0].dtype == torch.float16:
            atol = 1e-2
        else:
            atol = 1e-4
        assert torch.allclose(path[0], ortho_camera_data[0], atol=atol)
        assert torch.allclose(path[11], ortho_camera_data[1], atol=atol)
        assert torch.allclose(path[22], ortho_camera_data[2], atol=atol)
        assert torch.allclose(path[33], ortho_camera_data[3], atol=atol)

    def test_keyframes_catmull_rom_ortho_int(self, ortho_camera_data):
        cam_path = camera_path_generator(
            trajectory=ortho_camera_data,
            frames_between_cameras=10,
            interpolation="catmull_rom"
        )
        path = [cam for cam in cam_path]

        if ortho_camera_data[0].dtype == torch.float16:
            atol = 1e-2
        else:
            atol = 1e-4
        assert torch.allclose(path[0], ortho_camera_data[0], atol=atol)
        assert torch.allclose(path[11], ortho_camera_data[1], atol=atol)
        assert torch.allclose(path[22], ortho_camera_data[2], atol=atol)
        assert torch.allclose(path[33], ortho_camera_data[3], atol=atol)

    def test_interpolation_vals(self, mixed_camera_data):
        cam_path = camera_path_generator(
            trajectory=mixed_camera_data,
            frames_between_cameras=10,
            interpolation="catmull_rom"
        )
        path = [cam for cam in cam_path]

        atol = 1e-1
        device = path[3].device
        dtype = path[3].dtype
        assert torch.allclose(path[3].cam_pos().squeeze(),
                              torch.tensor([4.6992, 4.2188, 4.1758], device=device, dtype=dtype), atol=atol)
        assert torch.allclose(path[3].cam_forward().squeeze(),
                              torch.tensor([0.6333, 0.5308, 0.5635], device=device, dtype=dtype), atol=atol)
        assert torch.allclose(path[3].fov().squeeze(),
                              torch.tensor([48.8125], device=device, dtype=dtype), atol=atol)

        assert torch.allclose(path[12].cam_pos().squeeze(),
                              torch.tensor([8.0391, 4.1875, 3.9961], device=device, dtype=dtype), atol=atol)
        assert torch.allclose(path[12].cam_forward().squeeze(),
                              torch.tensor([0.8433, 0.3364, 0.4197], device=device, dtype=dtype), atol=atol)
        assert torch.allclose(path[12].fov().squeeze(),
                              torch.tensor([59.6562], device=device, dtype=dtype), atol=atol)

    def test_interpolation_width_height(self, mixed_camera_data):

        mixed_camera_data[1].width = 400
        mixed_camera_data[1].height = 300

        cam_path = camera_path_generator(
            trajectory=mixed_camera_data,
            frames_between_cameras=10,
            interpolation="catmull_rom"
        )
        path = [cam for cam in cam_path]

        assert path[0].width == mixed_camera_data[0].width
        assert path[0].height == mixed_camera_data[0].height
        for i in range(1, 12):
            assert path[i - 1].width > path[i].width
            assert path[i - 1].height > path[i].height
        assert path[11].width == mixed_camera_data[1].width
        assert path[11].height == mixed_camera_data[1].height
