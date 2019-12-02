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
import shutil

import kaolin as kal
from torch.utils.data import DataLoader


# Tests below can only be run is a ShapeNet dataset is available

# def test_ModelNet(device = 'cpu'): 
	
# 	models = kal.dataloader.ModelNet(root = 'datasets', categories = ['chair'], train = True)
	
# 	assert len(models) == 889
# 	for obj in models: 
# 		assert obj['class'] == 'chair'
# 		assert set(obj['data'].shape) == set([30,30,30]) 
# 	shutil.rmtree('datasets/')



