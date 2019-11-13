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
import sys

import kaolin as kal
from kaolin.rep import TriangleMesh


def test_pointcloud_to_voxelgrid(device='cpu'): 
    mesh = TriangleMesh.from_obj('tests/model.obj')
    if device == 'cuda':
        mesh.cuda()
    pts, _ = kal.conversions.trianglemesh_to_pointcloud(mesh, 1000)

    voxels = kal.conversions.pointcloud_to_voxelgrid(pts, 32, 0.1)
    assert(voxels.shape == (32, 32, 32))


def test_pointcloud_to_trianglemesh(device='cpu'):
    mesh = TriangleMesh.from_obj('tests/model.obj')
    if device == 'cuda':
        mesh.cuda()
    pts, _ = kal.conversions.trianglemesh_to_pointcloud(mesh, 1000)
    mesh_ = kal.conversions.pointcloud_to_trianglemesh(pts)


def test_pointcloud_to_sdf(device='cpu'):
    mesh = TriangleMesh.from_obj('tests/model.obj')
    if device == 'cuda':
        mesh.cuda()
    pts, _ = kal.conversions.trianglemesh_to_pointcloud(mesh, 1000)
    sdf_ = kal.conversions.pointcloud_to_trianglemesh(pts)    
