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
import os

import kaolin as kal
from kaolin.rep import TriangleMesh
from kaolin.rep import QuadMesh


def test_load_obj(device = 'cpu'): 
	mesh = TriangleMesh.from_obj(('tests/model.obj') )
	if device == 'cuda': 
		mesh.cuda()
	assert mesh.vertices.shape[0] > 0 
	assert mesh.vertices.shape[1] == 3 
	assert mesh.faces.shape[0] > 0 
	assert mesh.faces.shape[1] == 3 

	mesh = TriangleMesh.from_obj('tests/model.obj', with_vt=True, texture_res = 4) 
	if device == 'cuda': 
		mesh.cuda()
	assert mesh.textures.shape[0] > 0 

	mesh = TriangleMesh.from_obj('tests/model.obj', with_vt=True, texture_res = 4, enable_adjacency = True)
	assert mesh.vv.shape[0] > 0 
	assert mesh.edges.shape[0] > 0
	assert mesh.vv_count.shape[0] > 0
	assert mesh.ve.shape[0] > 0
	assert mesh.ve_count.shape[0] > 0
	assert mesh.ff.shape[0] > 0
	assert mesh.ff_count.shape[0] > 0
	assert mesh.ef.shape[0] > 0
	assert mesh.ef_count.shape[0] > 0
	assert mesh.ee.shape[0] > 0
	assert mesh.ee_count.shape[0] > 0
	if device == 'cuda': 
		mesh.cuda()

def test_from_tensors(device='cpu'): 
	mesh = TriangleMesh.from_obj('tests/model.obj', with_vt=True, texture_res = 4)
	if device == 'cuda': 
		mesh.cuda()

	verts = mesh.vertices.clone()
	faces = mesh.faces.clone() 
	uvs = mesh.uvs.clone()
	face_textures = mesh.face_textures.clone()
	textures = mesh.textures.clone()
	
	mesh = TriangleMesh.from_tensors(verts, faces, uvs = uvs, face_textures = face_textures,
			 textures=textures)

	
def test_sample_mesh( device = 'cpu'):
	mesh = TriangleMesh.from_obj('tests/model.obj')
	if device == 'cuda': 
		mesh.cuda() 
	
	points, choices = mesh.sample(100)
	assert (set(points.shape) == set([100, 3])) 
	points, choices = mesh.sample(10000)
	assert (set(points.shape) == set([10000, 3])) 

def test_laplacian_smoothing(device = 'cpu'): 
	mesh = TriangleMesh.from_obj(('tests/model.obj'))
	if device == 'cuda': 
		mesh.cuda()
	v1 = mesh.vertices.clone()
	mesh.laplacian_smoothing(iterations = 3)
	v2 = mesh.vertices.clone()
	assert (torch.abs(v1 - v2)).sum() >0 

def test_compute_laplacian(device = 'cpu'): 
	mesh = TriangleMesh.from_obj(('tests/model.obj'))
	if device == 'cuda': 
		mesh.cuda()
	lap = mesh.compute_laplacian()
	assert ((lap**2).sum(dim=1) > .1).sum() == 0  # asserting laplacian of sphere is small

def test_load_and_save_Tensors(device = 'cpu'): 
	mesh1 = TriangleMesh.from_obj(('tests/model.obj'))
	if device == 'cuda': 
		mesh1.cuda()
	mesh1.save_tensors('copy.npz')
	assert os.path.isfile('copy.npz')
	mesh2 = TriangleMesh.load_tensors ('copy.npz')
	if device == 'cuda': 
		mesh2.cuda()
	assert (torch.abs(mesh1.vertices - mesh2.vertices)).sum() ==0
	assert (torch.abs(mesh1.faces - mesh2.faces)).sum() ==0
	os.remove("copy.npz")

def test_adj_computations(device = 'cpu'): 
	mesh = TriangleMesh.from_obj(('tests/model.obj'))
	if device == 'cuda': 
		mesh.cuda()

	adj_full = mesh.compute_adjacency_matrix_full()
	adj_sparse = mesh.compute_adjacency_matrix_sparse().coalesce()

	assert adj_full.shape[0] == mesh.vertices.shape[0]
	assert ((adj_full-adj_sparse.to_dense()) != 0).sum() == 0 



def test_load_obj_gpu(): 
	test_load_obj("cuda")
def test_from_tensors_gpu(): 
	test_from_tensors("cuda")
def test_sample_mesh_gpu(): 
	test_sample_mesh("cuda")
def test_laplacian_smoothing_gpu(): 
	test_laplacian_smoothing("cuda")
def test_compute_laplacian_gpu(): 
	test_compute_laplacian("cuda")
def test_load_and_save_Tensors_gpu(): 
	test_load_and_save_Tensors("cuda")
def test_adj_computations_gpu(): 
	test_adj_computations("cuda")