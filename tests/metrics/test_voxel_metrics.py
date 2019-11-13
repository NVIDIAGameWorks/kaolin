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
from kaolin.metrics.voxel import iou

def test_iou(device = 'cpu'): 	
	
	
	A = torch.rand(2,32,32,32).to(device)
	B = torch.ones(2,32,32,32).to(device)
	distance = iou(A,B)	 
	assert (distance >=0 and distance <=1.)

	A = torch.ones(2,32,32,32).to(device)
	distance = iou(A,B)	 
	assert (distance==1)


	A = torch.zeros(2,32,32,32).to(device)
	B = torch.ones(2,32,32,32).to(device)
	distance = iou(A,B)	 
	assert (distance ==0)

	A = torch.zeros(2,32,32,32).to(device)
	B = torch.zeros(2,32,32,32).to(device)
	distance = iou(A,B)	 
	assert (distance != distance) # should be NaN

def test_iou_gpu(): 
	test_iou("cuda")