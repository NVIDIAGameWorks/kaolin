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

import argparse
import json
import numpy as np
import os
import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm



from architectures import Generator

import kaolin as kal
"""
Commandline arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument('-expid', type=str, default='GAN', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('-batchsize', type=int, default=50, help='Batch size.')
args = parser.parse_args()


gen = Generator().to(args.device)
gen.load_state_dict(torch.load('log/{0}/gen.pth'.format(args.expid)))
gen.eval()

z = torch.normal(torch.zeros(args.batchsize, 200), torch.ones(args.batchsize, 200)*.33).to(args.device)

fake_voxels = gen(z)[:,0]
for i,model in enumerate(fake_voxels): 
	model = model[:-2,:-2,:-2]
	model = kal.rep.voxel.max_connected(model, .5)
	verts, faces = kal.conversion.voxel.to_mesh_quad(model)
	mesh = kal.rep.QuadMesh.from_tensors( verts, faces)
	mesh.laplacian_smoothing(iterations = 3)
	mesh.show()