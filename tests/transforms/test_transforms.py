# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import numpy as np
import torch
from torch.testing import assert_allclose

import kaolin as kal
from kaolin import helpers
from kaolin.rep import TriangleMesh


def test_numpy_to_tensor(device='cpu'):
    nptotensor = kal.transforms.NumpyToTensor()
    arr = np.zeros(3)
    ten = nptotensor(arr)
    assert torch.is_tensor(ten)


def test_shift_pointcloud(device='cpu'):
    unit_shift = kal.transforms.ShiftPointCloud(-1)
    pc = torch.ones(4, 3)
    pc_ = unit_shift(pc)
    assert_allclose(pc_, torch.zeros(4, 3))


def test_scale_pointcloud(device='cpu'):
    twice = kal.transforms.ScalePointCloud(2)
    halve = kal.transforms.ScalePointCloud(0.5)
    pc = torch.ones(4, 3)
    pc_ = halve(twice(pc))
    assert_allclose(pc_, torch.ones(4, 3))


def test_translate_pointcloud(device='cpu'):
    pc = torch.ones(4, 3)
    tmat = torch.tensor([[-1.0,-1.0,-1.0]])
    translate = kal.transforms.TranslatePointCloud(tmat)
    pc_ = translate(pc)
    assert_allclose(pc_, torch.zeros(4, 3))
    pc = torch.ones(4, 2)
    tmat = torch.tensor([[-1.0,-1.0]])
    translate = kal.transforms.TranslatePointCloud(tmat)
    pc_ = translate(pc)
    assert_allclose(pc_, torch.zeros(4, 2))


def test_rotate_pointcloud(device='cpu'):
    pc = torch.ones(4, 3)
    rmat = 2 * torch.eye(3)
    rmatinv = rmat.inverse()
    rotate1 = kal.transforms.RotatePointCloud(rmat)
    rotate2 = kal.transforms.RotatePointCloud(rmatinv)
    iden = kal.transforms.Compose([rotate1, rotate2])
    assert_allclose(iden(pc), torch.ones(4, 3))


def test_realign_pointcloud(device='cpu'):
    torch.manual_seed(1234)
    src = 10 * torch.rand(4, 3).to(device)
    tgt = 10 * torch.rand(4, 3).to(device)
    realign = kal.transforms.RealignPointCloud(tgt)
    src_ = realign(src)
    # After transformation, src_.mean(-2) should be equal to tgt.mean(-2)
    assert_allclose(src_.mean(-2), tgt.mean(-2), atol=1e-2, rtol=1e1)
    # Similarly, the stddevs should be equal along dim -2.
    assert_allclose(src_.std(-2), tgt.std(-2), atol=1e-2, rtol=1e1)


def test_normalize_pointcloud(device='cpu'):
    src = torch.rand(4, 3).to(device)
    normalize = kal.transforms.NormalizePointCloud()
    src = normalize(src)
    assert_allclose(src.mean(-2), torch.zeros_like(src.mean(-2)))
    assert_allclose(src.std(-2), torch.ones_like(src.std(-2)))


def test_downsample_voxelgrid(device='cpu'):
    voxel = torch.ones([32, 32, 32]).to(device)
    down = kal.transforms.DownsampleVoxelGrid([2, 2, 2], inplace=False)
    helpers._assert_shape_eq(down(voxel), (16, 16, 16))
    down = kal.transforms.DownsampleVoxelGrid([3, 3, 3], inplace=False)
    helpers._assert_shape_eq(down(voxel), (10, 10, 10))
    down = kal.transforms.DownsampleVoxelGrid([3, 2, 1], inplace=False)
    helpers._assert_shape_eq(down(voxel), (10, 16, 32))


def test_upsample_voxelgrid(device='cpu'):
    voxel = torch.ones([32, 32, 32]).to(device)
    up = kal.transforms.UpsampleVoxelGrid(64)
    helpers._assert_shape_eq(up(voxel), (64, 64, 64))
    up = kal.transforms.UpsampleVoxelGrid(33)
    helpers._assert_shape_eq(up(voxel), (33, 33, 33))


def test_triangle_mesh_to_pointcloud(device='cpu'):
    mesh = TriangleMesh.from_obj('tests/model.obj')
    mesh.to(device)
    mesh2cloud = kal.transforms.TriangleMeshToPointCloud(10000)
    pts = mesh2cloud(mesh)
    helpers._assert_shape_eq(pts, (10000, 3))


def test_triangle_mesh_to_voxelgrid(device='cpu'):
    mesh = TriangleMesh.from_obj('tests/model.obj')
    mesh.to(device)
    mesh2voxel = kal.transforms.TriangleMeshToVoxelGrid(32)
    helpers._assert_shape_eq(mesh2voxel(mesh), (32, 32, 32))


def test_triangle_mesh_to_sdf(device='cpu'):
    mesh = TriangleMesh.from_obj('tests/model.obj')
    mesh.to(device)
    mesh2sdf = kal.transforms.TriangleMeshToSDF(100)
    helpers._assert_shape_eq(mesh2sdf(mesh), (100,), dim=-1)
