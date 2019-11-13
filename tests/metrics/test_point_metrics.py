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
from kaolin.metrics.point import chamfer_distance

def test_chamfer_distance(device = 'cpu'): 	

	A = torch.rand(300,3).to(device)
	B = torch.rand(200,3).to(device)
	distance = kal.metrics.point.chamfer_distance(A,B)	
	assert distance >= 0 
	assert distance <= 2.

	
	B = A.clone()
	distance = kal.metrics.point.chamfer_distance(A,B)	 
	assert distance == 0


def test_directed_distance(device = 'cpu'): 		
	A = torch.rand(300,3).to(device)
	B = torch.rand(200,3).to(device)
	distance = kal.metrics.point.directed_distance(A,B, mean=True)	

	assert distance >= 0 
	assert distance <= .5

	distances = kal.metrics.point.directed_distance(A,B, mean=False)	
	assert ((distances <0).sum() == 1) == 0 
	assert ((distances >1).sum() == 1) == 0
	assert (set(distances.shape)  == set([300]))


	B = A.clone()
	distance = kal.metrics.point.directed_distance(A,B, mean=True)	 
	assert distance == 0

	distances = kal.metrics.point.directed_distance(A,B, mean=False)	 
	assert distances.sum() == 0
	


def test_iou(device = 'cpu'): 
	points = torch.rand(300,3).to(device)
	points1 = points *2. //1 
	points2 = points *1.5 //1

	iou1 = kal.metrics.point.iou(points1,points2)	
	assert iou1 >= 0 
	assert iou1 <= 1 

	points3 = points*1.1 //1

	iou2 = kal.metrics.point.iou(points1,points3)	
	assert iou2 >= 0 
	assert iou2 <= 1 
	assert iou1 >= iou2


def test_f_score(device = 'cpu'): 

	points1 = torch.rand(1,3).to(device) 
	points2 = points1.clone()

	f = kal.metrics.point.f_score(points1,points2, radius = 0.01)	
	assert (f == 1) 

	points2 = points1 * 1.01

	f1 = kal.metrics.point.f_score(points1,points2, radius = 0.01)	
	
	points2 = points2 * 1.011

	f2 = kal.metrics.point.f_score(points1,points2, radius = 0.01)
	assert (f1>= f2)

	f3 = kal.metrics.point.f_score(points1,points2, radius = 0.015)

	assert (f3>= f2)

def test_chamfer_distance_gpu(): 
	test_chamfer_distance("cuda")
def test_directed_distance_gpu(): 
	test_directed_distance("cuda")
def test_iou_gpu(): 
	test_iou("cuda")
def test_f_score_gpu(): 
	test_f_score("cuda")