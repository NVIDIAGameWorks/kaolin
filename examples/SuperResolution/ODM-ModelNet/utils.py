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

import kaolin as kal 
import torch

def down_sample(tgt): 
	inp = []
	for t in tgt : 
		low_res_inp = kal.rep.voxel.scale_down(t, scale = [2, 2, 2])
		low_res_inp = kal.rep.voxel.threshold(low_res_inp, .1)
		inp.append(low_res_inp.unsqueeze(0))
	inp = torch.cat(inp, dim = 0 )
	return inp

def up_sample(inp): 
	NN_pred = []
	for voxel in inp: 
		NN_pred.append(kal.rep.voxel.scale_up(voxel, dim = 30))
	NN_pred = torch.stack(NN_pred)
	return NN_pred

def to_occpumancy_map(inp, threshold = None):
	if threshold is None: 
		threshold = inp.shape[-1]
	zeros = inp< threshold
	ones = inp >= threshold
	inp = inp.clone()
	inp[ones] = 1 
	inp[zeros] = 0 
	return inp


def upsample_omd(inp): 
	scaling = torch.nn.Upsample(scale_factor=2, mode='nearest')
	inp = scaling(inp)
	return inp

