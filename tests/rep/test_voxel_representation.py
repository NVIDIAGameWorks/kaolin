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


def test_voxelgrid(device='cpu'):
	voxels = torch.ones([32, 32, 32]).to(device)
	voxgrid = kal.rep.VoxelGrid(voxels=voxels, copy=False)

# #### test voxel operation #####
def test_scale_down(device='cpu'): 
	voxel = torch.ones([32, 32, 32]).to(device)
	down = kal.rep.voxel.scale_down(voxel, [2,2,2])
	assert (set(down.shape) == set([16,16,16]))
	down = kal.rep.voxel.scale_down(voxel, [3,3,3])
	assert (set(down.shape) == set([10,10,10]))
	down = kal.rep.voxel.scale_down(voxel, [3,2,1])
	assert (set(down.shape) == set([10,16,32]))


	voxel = torch.zeros([128, 128, 128]).to(device)
	down = kal.rep.voxel.scale_down(voxel, [2,2,2])
	assert (set(down.shape) == set([64,64,64]))
	down = kal.rep.voxel.scale_down(voxel, [3,3,3])
	assert (set(down.shape) == set([42,42,42]))
	down = kal.rep.voxel.scale_down(voxel, [3,2,1])
	assert (set(down.shape) == set([42,64,128]))

def test_scale_up(device='cpu'): 
	voxel = torch.ones([32, 32, 32]).to(device)
	down = kal.rep.voxel.scale_up(voxel, 64)
	assert (set(down.shape) == set([64,64,64]))
	down = kal.rep.voxel.scale_up(voxel,33)
	assert (set(down.shape) == set([33,33,33]))
	down = kal.rep.voxel.scale_up(voxel,256)
	assert (set(down.shape) == set([256,256,256]))


	voxel = torch.zeros([128, 128, 128]).to(device)
	down = kal.rep.voxel.scale_up(voxel, 128)
	assert (set(down.shape) == set([128,128,128]))

	down = kal.rep.voxel.scale_up(voxel, 150)
	assert (set(down.shape) == set([150,150,150]))

	down = kal.rep.voxel.scale_up(voxel, 256 )
	assert (set(down.shape) == set([256,256,256]))


def test_threshold(device='cpu'): 
	voxel = torch.ones([32, 32, 32]).to(device)
	binary = kal.rep.voxel.threshold(voxel, .5)
	assert binary.sum() == 32*32*32
	assert (set(binary.shape) == set([32,32,32]))

	voxel = torch.ones([32, 32, 32]).to(device) * .3
	binary = kal.rep.voxel.threshold(voxel, .5)
	assert binary.sum() == 0
	assert (set(binary.shape) == set([32,32,32]))

	voxel = torch.ones([64, 64, 64]).to(device) * .7
	binary = kal.rep.voxel.threshold(voxel, .5)
	assert binary.sum() == 64*64*64
	assert (set(binary.shape) == set([64,64,64]))

def test_fill(device='cpu'): 
	voxel = torch.ones([32, 32, 32]).to(device)
	voxel[16, 16, 16] = 0 
	filled_voxel = kal.rep.voxel.fill(voxel)
	assert voxel.sum() < filled_voxel.sum()
	assert filled_voxel.sum() == 32*32*32

	voxel = torch.ones([32, 32, 32]).to(device)
	filled_voxel = kal.rep.voxel.fill(voxel)
	assert voxel.sum()  == filled_voxel.sum()
	assert filled_voxel.sum() == 32*32*32

	voxel = torch.zeros([64, 64, 64]).to(device)
	filled_voxel = kal.rep.voxel.fill(voxel)
	assert voxel.sum()  == filled_voxel.sum()
	assert filled_voxel.sum() == 0


def test_extract_surface(device='cpu'): 
	voxel = torch.ones([32, 32, 32]).to(device)
	surface_voxel = kal.rep.voxel.extract_surface(voxel)
	assert voxel.sum() > surface_voxel.sum()
	assert surface_voxel.sum() == 2*(32*32 + 32*30 + 30*30)
	assert kal.rep.voxel.extract_surface(surface_voxel).sum() == surface_voxel.sum()

	voxel = torch.zeros([32, 32, 32]).to(device)
	surface_voxel = kal.rep.voxel.extract_surface(voxel)
	assert voxel.sum() == surface_voxel.sum()
	assert surface_voxel.sum() == 0
	assert kal.rep.voxel.extract_surface(surface_voxel).sum() == surface_voxel.sum()	

	voxel = torch.ones([64, 64, 64]).to(device)
	surface_voxel = kal.rep.voxel.extract_surface(voxel)
	assert voxel.sum() > surface_voxel.sum()
	assert surface_voxel.sum() == 2*(64*64 + 64*62 + 62*62)
	assert kal.rep.voxel.extract_surface(surface_voxel).sum() == surface_voxel.sum()


def test_extract_odms(device='cpu'): 
	voxel = torch.rand([32, 32, 32]).to(device)
	voxel = kal.rep.voxel.threshold(voxel, .9)
	odms = kal.rep.voxel.extract_odms(voxel)
	assert (set(odms.shape) == set([6,32,32]))
	assert (odms.max() == 32)
	assert (odms.min() == 0)

	voxel = torch.ones([32, 32, 32]).to(device)
	voxel = kal.rep.voxel.threshold(voxel, .5)
	odms = kal.rep.voxel.extract_odms(voxel)
	assert (set(odms.shape) == set([6,32,32]))
	assert (odms.max() == 0)

	voxel = torch.zeros([128, 128, 128]).to(device)
	voxel = kal.rep.voxel.threshold(voxel, .5)
	odms = kal.rep.voxel.extract_odms(voxel)
	assert (set(odms.shape) == set([6,128,128]))
	assert (odms.max() == 128)



def test_project_odms(device='cpu'): 

	odms = torch.rand([32, 32, 32]).to(device) *32
	odms = odms.int()
	voxel = kal.rep.project_odms(odms)
	assert (set(voxel.shape) == set([32,32,32]))

	odms = torch.rand([128, 128, 128]).to(device) *128
	odms = odms.int()
	voxel = kal.rep.project_odms(odms)
	assert (set(voxel.shape) == set([128,128,128]))

	voxel = torch.ones([32, 32, 32]).to(device)
	voxel = kal.rep.voxel.threshold(voxel, .9)
	odms = kal.rep.voxel.extract_odms(voxel)
	new_voxel = kal.rep.voxel.project_odms(odms)
	assert (set(new_voxel.shape) == set([32,32,32])) 
	assert (torch.abs(voxel - new_voxel).sum() == 0 )


def test_scale_down_gpu(): 
	test_scale_down("cuda")
def test_scale_up_gpu(): 
	test_scale_up("cuda")
def test_threshold_gpu(): 
	test_threshold("cuda")
def test_fill_gpu(): 
	test_fill("cuda")
def test_extract_surface_gpu(): 
	test_extract_surface("cuda")
def test_extract_odms_gpu(): 
	test_extract_odms("cuda")
def test_project_odms_gpu(): 
	test_project_odms("cuda")
