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
import os
import torch
import sys
from tqdm import tqdm
from PIL import Image
import numpy as np

from torch.utils.data import DataLoader

from utils import preprocess
from architectures import Encoder
import kaolin as kal 

parser = argparse.ArgumentParser()
parser.add_argument('-expid', type=str, default='Direct', help='Unique experiment identifier.')
parser.add_argument('-device', type=str, default='cuda', help='Device to use')
parser.add_argument('-categories', type=str,nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('-vis', action='store_true', help='Visualize each model while evaluating')
parser.add_argument('-batchsize', type=int, default=16, help='Batch size.')
parser.add_argument('-f_score', action='store_true', help='compute F-score')
args = parser.parse_args()


# Data
points_set_valid = kal.dataloader.ShapeNet.Points(root ='../../datasets/',categories =args.categories , \
	download = True, train = False, split = .7, num_points=5000 )
images_set_valid = kal.dataloader.ShapeNet.Images(root ='../../datasets/',categories =args.categories , \
	download = True, train = False,  split = .7, views=1, transform= preprocess )
meshes_set_valid = kal.dataloader.ShapeNet.Meshes(root ='../../datasets/', categories =args.categories , \
	download = True, train = False,  split = .7)

valid_set = kal.dataloader.ShapeNet.Combination([points_set_valid, images_set_valid], root='../../datasets/')
dataloader_val = DataLoader(valid_set, batch_size=args.batchsize, shuffle=False, 
	num_workers=8)
# Model
mesh = kal.rep.TriangleMesh.from_obj('386.obj')
if args.device == "cuda": 
	mesh.cuda()
initial_verts = mesh.vertices.clone()


model = Encoder(4, 5, args.batchsize, 137, mesh.vertices.shape[0] ).to(args.device)
# Load saved weights
model.load_state_dict(torch.load('log/{0}/best.pth'.format(args.expid)))

loss_epoch = 0.
f_epoch = 0.
num_batches = 0
num_items = 0
loss_fn = kal.metrics.point.chamfer_distance

model.eval()
with torch.no_grad():
	for data in tqdm(dataloader_val): 
		# data creation
		tgt_points = data['points'].to(args.device)
		inp_images = data['imgs'].to(args.device)

		# inference 
		delta_verts = model(inp_images)

		# eval 
		
		loss = 0. 
		for deltas, tgt, img in zip(delta_verts, tgt_points, inp_images): 	
			mesh.vertices = deltas + initial_verts
			pred_points, _ = mesh.sample(3000)
			loss += 3000* loss_fn(pred_points, tgt) / float(args.batchsize)

			if args.f_score: 
			
				f_score = kal.metrics.point.f_score(tgt, pred_points, extend = False)
				f_epoch += (f_score  / float(args.batchsize)).item()

			if args.vis: 
				tgt_mesh = meshes_set_valid[num_items]
				tgt_verts = tgt_mesh['verts']
				tgt_faces = tgt_mesh['faces']
				tgt_mesh = kal.rep.TriangleMesh.from_tensors(tgt_verts, tgt_faces)

				print ('Displaying input image')
				img = img.data.cpu().numpy().transpose((1, 2, 0))
				img = (img*255.).astype(np.uint8)
				Image.fromarray(img).show()
				print ('Rendering Target Mesh')
				kal.visualize.show_mesh(tgt_mesh)
				print ('Rendering Predicted Mesh')
				kal.visualize.show_mesh(mesh)
				print('----------------------')
				num_items += 1


		loss_epoch += loss.item()
		
		
		num_batches += 1.

out_loss = loss_epoch / float(num_batches)
print ('Loss over validation set is {0}'.format(out_loss))
if args.f_score: 
	out_f = f_epoch / float(num_batches)
	print ('F-score over validation set is {0}'.format(out_f))
