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

from utils import occ_function, collate_fn, extract_mesh
from architectures import OccupancyNetwork
from PIL import Image
import kaolin as kal 

parser = argparse.ArgumentParser()
parser.add_argument('-expid', type=str, default='Direct', help='Unique experiment identifier.')
parser.add_argument('-device', type=str, default='cuda', help='Device to use')
parser.add_argument('-categories', type=str,nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('-vis', action='store_true', help='Visualize each model while evaluating')
parser.add_argument('-f_score', action='store_true', help='compute F-score')
parser.add_argument('-batch_size', type=int, default=3, help='Batch size.')

args = parser.parse_args()


# Data
images_set = kal.dataloader.ShapeNet.Images(root ='../../datasets/',categories =args.categories , \
	download = True, train = False,  split = .7, views=1)
points_set_valid = kal.dataloader.ShapeNet.Points(root ='../../datasets/',categories =args.categories , \
	download = True, train = False, split = .7, num_points=5000 )
sdf_set = kal.dataloader.ShapeNet.SDF_Points(root= '../../datasets/', categories=args.categories, \
	download=True, train = False, split = .7, num_points = 100000, occ = True)
data_set_mesh = kal.dataloader.ShapeNet.Meshes(root= '../../datasets/', \
	categories=args.categories, download=True, train = False, split = .7)


valid_set = kal.dataloader.ShapeNet.Combination([sdf_set, images_set, data_set_mesh, points_set_valid], root='../../datasets/')

dataloader_val = DataLoader(valid_set, batch_size=5, shuffle=False, num_workers=8, collate_fn=collate_fn)


# Model
model = OccupancyNetwork(args.device)
# Load saved weights

model.encoder.load_state_dict(torch.load('log/{}/best_encoder.pth'.format(args.expid)))

model.decoder.load_state_dict(torch.load('log/{}/best_decoder.pth'.format(args.expid)))

iou_epoch = 0.
f_epoch = 0.
num_batches = 0
num_items = 0



with torch.no_grad():
	model.encoder.eval()
	model.decoder.eval() 
	for data in tqdm(dataloader_val):
		imgs = data['imgs'][:,:3].to(args.device)
		sdf_points = data['occ_points'].to(args.device)
		surface_points = data['points'].to(args.device)
		gt_occ = data['occ_values'].to(args.device)


		
		encoding = model.encode_inputs(imgs)
		pred_occ = model.decode(sdf_points, torch.zeros(args.batch_size, 0), encoding ).logits

		i = 0 
		for sdf_point, gt_oc, pred_oc, gt_surf, code in zip(sdf_points, gt_occ, pred_occ, surface_points, encoding):
			#### compute iou ####
			iou_epoch += float((kal.metrics.point.iou(gt_oc, pred_oc, thresh=.2) / \
				float(gt_occ.shape[0])).item())
			
			if args.f_score or args.vis:
				
				# extract mesh from sdf 
				sdf = kal.rep.SDF(occ_function(model, code))
				voxelization = kal.conversion.SDF.to_voxel(sdf)
				verts, faces = extract_mesh( voxelization, model)\
				# algin new mesh to positions and scale of predicted occupancy
				occ_points = sdf_point[pred_oc >= .2]
				verts = kal.rep.point.re_align(occ_points, verts.clone())
				mesh = kal.rep.TriangleMesh.from_tensors(verts, faces)
				if verts.shape[0] == 0: # if mesh is empty count as 0 f-score
					continue 

				if args.vis: 

					tgt_verts = data['verts'][i]
					tgt_faces = data['faces'][i]
					tgt_mesh = kal.rep.TriangleMesh.from_tensors(tgt_verts, tgt_faces)

					print ('Displaying input image')
					img = imgs[i].data.cpu().numpy().transpose((2, 1, 0)) * 255
					img = (img).astype(np.uint8)
					Image.fromarray(img).show()
					print ('Rendering Target Mesh')
					kal.visualize.show_mesh(tgt_mesh)
					print ('Rendering Predicted Mesh')
					mesh.show()
					print('----------------------')
					num_items += 1

				if args.f_score:
					#### compute f score #### 
					pred_surf,_ = mesh.sample(5000)
					f_score = kal.metrics.point.f_score(gt_surf, pred_surf, extend = False)
					f_epoch += (f_score  / float(gt_occ.shape[0])).item()
			i+= 1		

		
		num_batches += 1.

out_iou = iou_epoch / float(num_batches)
print ('IoU over validation set is {0}'.format(out_iou))
if args.f_score: 
	out_f = f_epoch / float(num_batches)
	print ('F-score over validation set is {0}'.format(out_f))



 


	

	