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

import torch 
import numpy as np
from torchvision import transforms
from torchvision.transforms import Normalize as norm 

import kaolin as kal



preprocess = transforms.Compose([
   transforms.CenterCrop(137),
   transforms.ToTensor()
])


def loss_lap(mesh): 

	new_lap = torch.matmul(mesh.adj, mesh.vertices)
	loss = 0.01 * torch.mean((new_lap - mesh.vertices) ** 2) * mesh.vertices.shape[0] * 3
	return loss 

def loss_flat(mesh, norms): 
	loss  = 0.
	for i in range(3): 

		norm1 = norms
		norm2 = norms[mesh.ff[:, i]]
		cos = torch.sum(norm1 * norm2, dim=1)
		loss += torch.mean((cos - 1) ** 2) 
	loss *= (mesh.faces.shape[0]/2.)
	return loss

def collate_fn(data): 
	new_data = {}
	for k in data[0]['data'].keys():
		print(k)
		
		if k in ['points','normals','images', 'cam_mat', 'cam_pos', 'sdf_points']:
			new_info = tuple(d['data'][k] for d in data)
			new_info = torch.stack(new_info, 0)
			new_data[k] = new_info
		elif k in ['adj']: 
			
			adj_values = tuple(d[k].coalesce().values() for d in data)
			adj_indices = tuple(d[k].coalesce().indices() for d in data)
			new_data['adj_values'] = adj_values
			new_data['adj_indices'] = adj_indices
			new_data[k] = new_info
		elif k in ['params']:
			for j in data[0]['data']['params'].keys():
				if j in ['cam_mat', 'cam_pos']:

					new_info = tuple([d['data']['params'][j] for d in data])
					new_info = torch.stack(new_info,0)
					new_data[j] = new_info	
		else: 
			new_info = tuple(d['data'][k] for d in data)

			new_data[k] = new_info
	return new_data

def normalize_adj(mesh): 
	adj = mesh.compute_adjacency_matrix_full()
	eye = torch.FloatTensor(np.eye(adj.shape[0])).to(adj.device)
	adj = adj - eye
	nei_count = torch.sum(adj, dim=1)
	adj /= nei_count
	mesh.adj = adj 

