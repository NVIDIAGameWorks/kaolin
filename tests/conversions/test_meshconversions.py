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


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_to_trianglemesh_to_pointcloud(device):
	mesh = TriangleMesh.from_obj('tests/model.obj')
	if device == 'cuda':
		mesh.cuda()
	
	points, _ = kal.conversions.trianglemesh_to_pointcloud(mesh, 10)
	assert (set(points.shape) == set([10, 3]))
	points, _ = kal.conversions.trianglemesh_to_pointcloud(mesh, 10000)
	assert (set(points.shape) == set([10000, 3]))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_trianglemesh_to_voxelgrid(device):
	mesh = TriangleMesh.from_obj('tests/model.obj')
	if device == 'cuda':
		mesh.cuda()
	voxels = kal.conversions.trianglemesh_to_voxelgrid(mesh, 32,
		normalize ='unit')
	assert (set(voxels.shape) == set([32, 32, 32]))
	voxels = kal.conversions.trianglemesh_to_voxelgrid(mesh, 64,
		normalize ='unit')
	assert (set(voxels.shape) == set([64, 64, 64]))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_trianglemesh_to_sdf(device):
	mesh = TriangleMesh.from_obj('tests/model.obj')
	if device == 'cuda':
		mesh.cuda()
	print(mesh.device)
	sdf = kal.conversions.trianglemesh_to_sdf(mesh)
	distances = sdf(torch.rand(100,3).to(device) -.5)
	assert (set(distances.shape) == set([100]))
	assert ((distances >0).sum()) > 0
	assert ((distances <0).sum()) > 0
