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

"""
Network architecture definitions
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F





class upscale(nn.Module):
	def __init__(self, high, low):
		super(upscale, self).__init__()
		self.ratio = high // low 
		self.layer1 = nn.Sequential(
			nn.Conv2d(6, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128))

		self.inner_convs_1 = nn.ModuleList([nn.Conv2d(128, 128, kernel_size=3, padding=1) for i in range(16)])
		self.inner_bns_1 = nn.ModuleList([nn.BatchNorm2d(128) for i in range(16)])
		self.inner_convs_2 = nn.ModuleList([nn.Conv2d(128, 128, kernel_size=3, padding=1) for i in range(16)])
		self.inner_bns_2 = nn.ModuleList([nn.BatchNorm2d(128) for i in range(16)])
		
		self.layer2 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
		)

		sub_list = [nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.PixelShuffle(2)]
		i = 0 
		for i in range(int(math.log(self.ratio,2))-1):
			sub_list.append(nn.Conv2d(32, 128,kernel_size=3, padding=1))
			sub_list.append(nn.PixelShuffle(2))
		self.sub_list = nn.ModuleList(sub_list)

		self.layer3 = nn.Sequential(
			nn.Conv2d( 32, 6, kernel_size=1, padding=0),
			
		)
		

	def forward(self, x):
		x = self.layer1(x)
		temp = x.clone()
		for i in range(16): 
			recall = self.inner_convs_1[i](x.clone())
			recall = self.inner_bns_1[i](recall)
			recall = F.relu(recall)
			recall = self.inner_convs_2[i](recall)
			recall = self.inner_bns_2[i](recall)
			recall = recall + temp 
			temp = recall.clone()
		recall = self.layer2(recall)
		x = x + recall 
	
		for i in range(int(math.log(self.ratio,2))):
			x = self.sub_list[2*i](x)
			x = self.sub_list[2*i + 1](x)
		
		x = self.layer3(x)
		x = torch.sigmoid(x)
		return x
