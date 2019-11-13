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


def test_pointcloud_creation(device='cpu'):
	pts = torch.ones(100, 3)
	normals = torch.ones(100, 3)
	pcd = kal.rep.PointCloud(points=pts)
	pcd = kal.rep.PointCloud(points=pts, normals=normals)


# def test_points(device = 'cpu'): 
# 	points1 = torch.ones((100,3)).to(device)
# 	points2 = kal.rep.point.scale(points1, torch.FloatTensor([.5]).to(device))
# 	assert (points1 - 2*points2).sum() == 0 
# 	points1 = torch.rand((100,3)).to(device)
# 	points2 = kal.rep.point.scale(points1, torch.FloatTensor([2]).to(device))
# 	assert (2*points1 - points2).sum() == 0 

# def test_rotate(device = 'cpu'): 
# 	points1 = torch.rand((100,3)).to(device)
# 	rot_mat = [[1,.3,.4], [-.1,.2,.5], [-.2,-.3,-.4]]
# 	points2 = kal.rep.point.rotate(points1, torch.FloatTensor(rot_mat).to(device))
# 	assert (points1 - 2*points2).sum() > 0 

# def test_re_align(device = 'cpu'): 

# 	points1 = torch.rand((100,3)).to(device)
# 	points2 = points1 *3
# 	points2 = kal.rep.point.re_align(points1, points2)
# 	assert (points1 - points2).sum()  < .001 

# 	points1 = torch.rand((1000,3)).to(device)
# 	points2 = 5*torch.rand((1000,3)).to(device) - .3
# 	points2 = kal.rep.point.re_align(points1, points2)

# 	assert torch.abs(torch.max(points1) - torch.max(points2)) < .00001
# 	assert torch.abs(torch.min(points1) - torch.min(points2)) < .00001

# def test_bounding_points(device = 'cpu'): 

# 	points = torch.rand((10000,3)).to(device)
# 	bbox = [.75, .25, .75, .25, .75, .25]
# 	new_points_idx = kal.rep.point.bounding_points(points, bbox, padding = 0)
# 	new_points = points[new_points_idx]
# 	assert (new_points > .75 ).sum() == 0 
# 	assert (new_points < .25 ).sum() == 0 

# def test_points_cpu(): 
# 	test_points("cuda")
# def test_rotate_cpu(): 
# 	test_rotate("cuda")
# def test_re_align_cpu(): 
# 	test_re_align("cuda")
# def test_bounding_points_cpu(): 
	# test_bounding_points("cuda")
