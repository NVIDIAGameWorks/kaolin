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
from torchvision import transforms
from PIL import Image
import numpy as np
import kaolin as kal
import trimesh
import time 
from kaolin import mcubes




preprocess = transforms.Compose([
   transforms.CenterCrop(224),
   transforms.ToTensor()
])


def get_prior_z(cfg, device, **kwargs):
    ''' Returns prior distribution for latent code z.
    Args:
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    z_dim = 0
    p0_z = dist.Normal(
        torch.zeros(z_dim, device=device),
        torch.ones(z_dim, device=device)
    )

    return p0_z


def occ_function(model,code):
	z = torch.zeros(1, 0)

	def eval_query(query):
		pred_occ = model.decode(query.unsqueeze(0), z, code.unsqueeze(0) ).logits[0]
         # values less then .2 are sent to 1 and above(occupied ) are set to 0 -> part fo surface 
		values = pred_occ < .2
		

		return values

		
	return eval_query 


def collate_fn(data): 
	new_data = {}
	for k in data[0].keys():
		
		if k in ['occ_points','occ_values', 'imgs', 'points']:
			new_info = tuple(d[k] for d in data)
			new_info = torch.stack(new_info, 0)
		else: 
			new_info = tuple(d[k] for d in data)

		new_data[k] = new_info
	return new_data


def extract_mesh(occ_hat, model, c=None, stats_dict=dict()):
    
    n_x, n_y, n_z = occ_hat.shape
    box_size = 1 + .05
    threshold = .2
    # Make sure that mesh is watertight
    t0 = time.time()
    occ_hat_padded = np.pad(
        occ_hat, 1, 'constant', constant_values=-1e6)
    vertices, triangles = mcubes.marching_cubes(
        occ_hat_padded, threshold)
    return torch.FloatTensor(vertices.astype(float)).cuda(), torch.LongTensor(triangles.astype(int)).cuda()

