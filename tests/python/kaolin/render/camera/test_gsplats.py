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


import os
import sys
import subprocess
import pytest
import math
import random
import torch
import shutil
import numpy as np
from git import Repo
from kaolin.render.camera import kaolin_camera_to_gsplats, gsplats_camera_to_kaolin, Camera

# dealing with nvcr
if torch.version.cuda == '12.5':
    pytest.skip("gsplats is not installable with CUDA 12.5", allow_module_level=True)

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'gsplats')
@pytest.fixture(scope="module")
def gs_cam_cls():
    repo = Repo.clone_from(
        url='https://github.com/graphdeco-inria/gaussian-splatting',
        multi_options=['--recursive'],
        to_path=ROOT_DIR
    )
    sys.path.append(ROOT_DIR)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        os.path.join(ROOT_DIR, "submodules", "diff-gaussian-rasterization")
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        os.path.join(ROOT_DIR, "submodules", "simple-knn")
    ])
    from .gsplats.scene.cameras import Camera as GSCamera
    
    yield GSCamera
    sys.path.remove(ROOT_DIR)
    shutil.rmtree(ROOT_DIR)

    

class TestGsplats:
    def test_cycle(self, gs_cam_cls):
        kal_cam = Camera.from_args(
            eye=torch.rand((3,)),
            at=torch.rand((3,)),
            up=torch.nn.functional.normalize(torch.rand((3,)), dim=0),
            fov=random.random() * math.pi,
            width=512, height=512,
        )
        gs_cam = kaolin_camera_to_gsplats(kal_cam, gs_cam_cls)
        out_cam = gsplats_camera_to_kaolin(gs_cam)
        assert torch.allclose(out_cam, kal_cam)
    
    def test_kaolin_to_gsplats_regression(self, gs_cam_cls):
        kal_cam = Camera.from_args(
            eye=torch.tensor([1., 2., 3.]), 
            at=torch.tensor([0.3, 0.1, 0.2]),
            up=torch.tensor([0., 1., 0.]),
            fov=math.pi / 4,
            width=512, height=512,
        )
        gs_cam = kaolin_camera_to_gsplats(kal_cam, gs_cam_cls)
        expected_R = np.array([[ 0.9701425,   0.13336042, -0.20257968],
                               [ 0.,         -0.83525735, -0.5498591 ],
                               [-0.24253562,  0.53344166, -0.8103187 ]])
        expected_T = np.array([-0.24253559, -0.06317067,  3.733254  ])
        expected_fovx = torch.tensor([0.7854])
        expected_fovy = torch.tensor([0.7854])
        assert np.allclose(expected_R, gs_cam.R)
        assert np.allclose(expected_T, gs_cam.T)
        assert torch.allclose(expected_fovx, gs_cam.FoVx)
        assert torch.allclose(expected_fovy, gs_cam.FoVy)
