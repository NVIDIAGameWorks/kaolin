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
import math
import random
import torch
import numpy as np
from kaolin.render.camera import kaolin_camera_to_polyscope, polyscope_camera_to_kaolin, Camera


class TestPolyscope:
    def test_cycle(self):
        kal_cam = Camera.from_args(
            eye=torch.rand((3,)),
            at=torch.rand((3,)),
            up=torch.nn.functional.normalize(torch.rand((3,)), dim=0),
            fov=random.random() * math.pi,
            width=512, height=512,
        )
        ps_cam = kaolin_camera_to_polyscope(kal_cam)
        out_cam = polyscope_camera_to_kaolin(ps_cam, width=512, height=512)
        assert torch.allclose(out_cam, kal_cam)

    def test_cycle_cuda(self):
        kal_cam = Camera.from_args(
            eye=torch.rand((3,)),
            at=torch.rand((3,)),
            up=torch.nn.functional.normalize(torch.rand((3,)), dim=0),
            fov=random.random() * math.pi,
            width=1024, height=512,
            device='cuda'
        )
        ps_cam = kaolin_camera_to_polyscope(kal_cam)
        out_cam = polyscope_camera_to_kaolin(ps_cam, width=1024, height=512, device='cuda')
        assert torch.allclose(out_cam, kal_cam)

    def test_kaolin_to_polyscope_regression(self):
        kal_cam = Camera.from_args(
            eye=torch.tensor([1., 2., 3.]),
            at=torch.tensor([0.3, 0.1, 0.2]),
            up=torch.tensor([0., 1., 0.]),
            fov=math.pi / 4,
            width=512, height=512,
        )
        ps_cam = kaolin_camera_to_polyscope(kal_cam)
        expected_R = np.array([
            [0.9701424837112427, 0.0, -0.24253562092781067],
            [-0.13336041569709778, 0.835257351398468, -0.5334416627883911],
            [0.20257967710494995, 0.5498591065406799, 0.8103187084197998]
        ])
        expected_T = np.array([-0.24253559112548828, 0.0631706714630127, -3.7332539558410645])
        expected_aspect = np.array([1.0])
        expected_fovy = np.array([45.0])
        assert np.allclose(expected_R, ps_cam.get_R())
        assert np.allclose(expected_T, ps_cam.get_T())
        assert np.allclose(expected_aspect, ps_cam.get_aspect())
        assert np.allclose(expected_fovy, ps_cam.get_fov_vertical_deg())
