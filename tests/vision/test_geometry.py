"""
Unittests for projective geometry utility functions
"""

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Kornia components Copyright (c) 2019 Kornia project authors
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

import torch
from torch.testing import assert_allclose

import kaolin as kal
from kaolin.vision import *


def test_project_unproject():
    pts = torch.rand(20, 3)
    # intrinsics = torch.rand(4, 4)
    intrinsics = torch.FloatTensor([[720, 0, 120, 0], 
                                    [0, 720, 90, 0], 
                                    [0, 0, 1, 0], 
                                    [0, 0, 0, 1]])
    img_pts = project_points(pts, intrinsics)
    # depths = torch.rand(pts.shape[0]) + 1.
    depths = pts[..., 2]
    cam_pts = unproject_points(img_pts, depths, intrinsics)
    assert_allclose(pts, cam_pts, atol=1e-3, rtol=1e-3)


def test_unproject_project():
    img_pts = torch.rand(20, 2)
    depths = torch.rand(20) + 1.
    intrinsics = torch.FloatTensor([[720, 0, 120, 0], 
                                    [0, 720, 90, 0], 
                                    [0, 0, 1, 0], 
                                    [0, 0, 0, 1]])
    cam_pts = unproject_points(img_pts, depths, intrinsics)
    img_pts_reproj = project_points(cam_pts, intrinsics)
    assert_allclose(img_pts, img_pts_reproj, atol=1e-3, rtol=1e-3)


def test_project_unproject_batch():
    pts = torch.rand(10, 20, 3)
    intrinsics = torch.FloatTensor([[720, 0, 120, 0], 
                                    [0, 720, 90, 0], 
                                    [0, 0, 1, 0], 
                                    [0, 0, 0, 1]])
    intrinsics = intrinsics.repeat(10, 1, 1)
    img_pts = project_points(pts, intrinsics)
    depths = pts[..., 2]
    cam_pts = unproject_points(img_pts, depths, intrinsics)
    assert_allclose(pts, cam_pts, atol=1e-3, rtol=1e-3)


def test_unproject_project_batch():
    img_pts = torch.rand(10, 20, 2)
    depths = torch.rand(10, 20) + 1.
    intrinsics = torch.FloatTensor([[720, 0, 120, 0], 
                                    [0, 720, 90, 0], 
                                    [0, 0, 1, 0], 
                                    [0, 0, 0, 1]])
    intrinsics = intrinsics.repeat(10, 1, 1)
    cam_pts = unproject_points(img_pts, depths, intrinsics)
    img_pts_reproj = project_points(cam_pts, intrinsics)
    assert_allclose(img_pts, img_pts_reproj, atol=1e-3, rtol=1e-3)
