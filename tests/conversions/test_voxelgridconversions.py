# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

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
import torch

import kaolin as kal


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_voxelgrid_to_pointcloud(device):
    voxel = torch.ones([32, 32, 32]).to(device)
    points = kal.conversions.voxelgrid_to_pointcloud(voxel, 10)
    assert (set(points.shape) == set([10, 3]))
    assert points.max() <= .5
    assert points.min() >= -.5

    points = kal.conversions.voxelgrid_to_pointcloud(voxel, 100000)
    assert (set(points.shape) == set([100000, 3]))
    assert points.max() <= .5
    assert points.min() >= -.5

    voxel = torch.rand([64, 64, 64]).to(device)
    points = kal.conversions.voxelgrid_to_pointcloud(voxel, 10000)
    assert (set(points.shape) == set([10000, 3]))
    assert points.max() <= .5
    assert points.min() >= -.5


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_voxelgrid_to_trianglemesh(device):

    voxel = torch.rand([32, 32, 32]).to(device)
    verts, faces = kal.conversions.voxelgrid_to_trianglemesh(voxel, mode='marching_cubes')
    assert verts.shape[0] > 0
    assert faces.shape[0] > 0

    voxel = torch.rand([64, 64, 64]).to(device)
    verts, faces = kal.conversions.voxelgrid_to_trianglemesh(voxel, mode='exact')
    assert verts.shape[0] > 0
    assert faces.shape[0] > 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_voxelgrid_to_quadmesh(device):
    voxel = torch.rand([32, 32, 32]).to(device)
    verts, faces = kal.conversions.voxelgrid_to_quadmesh(voxel, thresh=.1)
    assert verts.shape[0] > 0
    assert faces.shape[0] > 0

    voxel = torch.ones([64, 64, 64]).to(device)
    verts, faces = kal.conversions.voxelgrid_to_quadmesh(voxel, thresh=.1)
    assert verts.shape[0] > 0
    assert faces.shape[0] > 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_voxelgrid_to_sdf(device):

    voxel = torch.rand([10, 10, 10]).to(device)
    sdf = kal.conversions.voxelgrid_to_sdf(voxel, thresh=.5)
    points = torch.rand((200, 3)).to(device) - .5
    distances = sdf(points)
    assert set(distances.shape) == set([200])
    assert distances.max() <= 1
    assert distances.min() >= -1

    voxel = torch.ones([10, 10, 10]).to(device)
    sdf = kal.conversions.voxelgrid_to_sdf(voxel, thresh=.5)
    points = torch.rand((200, 3)).to(device) - .5
    distances = sdf(points)
    assert set(distances.shape) == set([200])
    assert distances.sum() == 0
