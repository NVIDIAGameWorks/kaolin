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


def test_chamfer_distance(device = 'cpu'):
	mesh1 = TriangleMesh.from_obj('tests/model.obj') 
	mesh2 = TriangleMesh.from_obj('tests/model.obj') 
	if device == 'cuda': 
		mesh1.cuda()
		mesh2.cuda()

	mesh2.vertices = mesh2.vertices * 1.5
	distance = kal.metrics.mesh.chamfer_distance(mesh1, mesh2, num_points = 100)
	distance = kal.metrics.mesh.chamfer_distance(mesh1, mesh2, num_points = 200)
	assert kal.metrics.mesh.chamfer_distance(mesh1, mesh1, num_points = 500) <= 0.1

def test_edge_length(device = 'cpu'):
	mesh = TriangleMesh.from_obj('tests/model.obj') 
	if device == 'cuda': 
		mesh.cuda()
	length1 = kal.metrics.mesh.edge_length(mesh)
	mesh.vertices = mesh.vertices * 2 
	length2 = kal.metrics.mesh.edge_length(mesh)
	assert (length1 < length2)
	mesh.vertices = mesh.vertices * 0
	assert kal.metrics.mesh.edge_length(mesh) == 0
	
def test_laplacian_loss(device = 'cpu'):
	mesh1 = TriangleMesh.from_obj('tests/model.obj') 
	mesh2 = TriangleMesh.from_obj('tests/model.obj') 
	if device == 'cuda': 
		mesh1.cuda()
		mesh2.cuda() 
	mesh2.vertices = mesh2.vertices *1.5
	assert kal.metrics.mesh.laplacian_loss(mesh1, mesh2) > 0 
	assert kal.metrics.mesh.laplacian_loss(mesh1, mesh1) == 0 
	


def test_point_to_surface(device = 'cpu'):
	torch.manual_seed(1)
	torch.cuda.manual_seed(1)
	mesh = TriangleMesh.from_obj('tests/model.obj') 
	points = torch.rand(500,3) -.5
	if device == 'cuda': 
		mesh.cuda()
		points = points.cuda()
	
	distance = kal.metrics.mesh.point_to_surface(points, mesh)
	
	assert (distance > 1).sum() == 0 
	assert ( distance <= 0 ).sum() == 0 
	assert (distance.sum() <= .2) 

def test_chamfer_distance_cpu(): 
	test_chamfer_distance("cuda")
def test_edge_length_cpu(): 
	test_edge_length("cuda")
def test_laplacian_loss_cpu(): 
	test_laplacian_loss("cuda")
def test_point_to_surface_cpu(): 
	test_point_to_surface("cuda")
