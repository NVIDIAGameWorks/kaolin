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


def test_sdf_to_pointcloud():
	
	sdf = kal.rep.SDF.sphere()
	points = kal.conversions.sdf_to_pointcloud(sdf, bbox_center=0.,
		resolution=10, bbox_dim=1,  num_points = 10000)
	
	assert (set(points.shape) == set([10000, 3]))
	assert kal.rep.SDF._length(points).mean() <=.6
	assert kal.rep.SDF._length(points).mean() >=.4

	points = kal.conversions.sdf_to_pointcloud(sdf, bbox_center=0.,
		resolution=32, bbox_dim=1,  num_points = 10000)
	
	assert (set(points.shape) == set([10000, 3]))
	assert kal.rep.SDF._length(points).mean() <=.55
	assert kal.rep.SDF._length(points).mean() >=.45

	sdf = kal.rep.SDF.box(h =.2, w = .4, l = .5)
	points = kal.conversions.sdf_to_pointcloud(sdf, bbox_center=0.,
		resolution=10, bbox_dim=1,  num_points = 10000)
	
	assert (torch.abs(points[:,0])>.22).sum() == 0
	assert (torch.abs(points[:,1])>.44).sum() == 0
	assert (torch.abs(points[:,2])>.55).sum() == 0


def test_sdf_to_trianglemesh():

	sdf = kal.rep.SDF.sphere()
	verts, faces = kal.conversions.sdf_to_trianglemesh(sdf, bbox_center=0.,
		resolution=10, bbox_dim=1)

	assert kal.rep.SDF._length(verts).mean() <=.6
	assert kal.rep.SDF._length(verts).mean() >=.4

	verts, faces = kal.conversions.sdf_to_trianglemesh(sdf, bbox_center=0.,
		resolution=32, bbox_dim=1)
	
	assert kal.rep.SDF._length(verts).mean() <=.55
	assert kal.rep.SDF._length(verts).mean() >=.45

	sdf = kal.rep.SDF.box(h =.2, w = .4, l = .5)
	verts, faces = kal.conversions.sdf_to_trianglemesh(sdf, bbox_center=0.,
		resolution=10, bbox_dim=1)
	
	assert (torch.abs(verts[:,0])>.22).sum() == 0
	assert (torch.abs(verts[:,1])>.44).sum() == 0
	assert (torch.abs(verts[:,2])>.55).sum() == 0


def test_sdf_to_voxelgrid():

	sdf = kal.rep.SDF.sphere()
	voxels = kal.conversions.sdf_to_voxelgrid(sdf, bbox_center=0.,
		resolution=10, bbox_dim=1)

	assert (voxels).sum() >= 1
	assert (voxels).sum() <= (41**3) /2

	voxels = kal.conversions.sdf_to_voxelgrid(sdf, bbox_center=0.,
		resolution=32, bbox_dim=1)

	assert (voxels).sum() >= 1
	assert (voxels).sum() <= (129**3) /2

