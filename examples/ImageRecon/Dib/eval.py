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
from graphics.render.base import Render as Dib_Renderer
from graphics.utils.utils_perspective import  perspectiveprojectionnp

from utils import preprocess, collate_fn, normalize_adj
from architectures import Encoder
import kaolin as kal 

parser = argparse.ArgumentParser()
parser.add_argument('-expid', type=str, default='Direct', help='Unique experiment identifier.')
parser.add_argument('-categories', type=str,nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('-vis', action='store_true', help='Visualize each model while evaluating')
parser.add_argument('-batchsize', type=int, default=16, help='Batch size.')
parser.add_argument('-f_score', action='store_true', help='compute F-score')
args = parser.parse_args()


# Data
points_set_valid = kal.datasets.ShapeNet.Points(root ='../../datasets/',categories =args.categories , \
	download = True, train = False, split = .7, num_points=5000 )
images_set_valid = kal.datasets.ShapeNet.Images(root ='../../datasets/',categories =args.categories , \
	download = True, train = False,  split = .7, views=1, transform= preprocess )
meshes_set_valid = kal.datasets.ShapeNet.Meshes(root ='../../datasets/', categories =args.categories , \
	download = True, train = False,  split = .7)

valid_set = kal.datasets.ShapeNet.Combination([points_set_valid, images_set_valid, meshes_set_valid], root='../../datasets/')
dataloader_val = DataLoader(valid_set, batch_size=args.batchsize, shuffle=False, collate_fn = collate_fn,
	num_workers=8)

# Model
mesh = kal.rep.TriangleMesh.from_obj('386.obj', enable_adjacency= True)
mesh.cuda()
normalize_adj(mesh)
    

initial_verts = mesh.vertices.clone()
camera_fov_y = 49.13434207744484 * np.pi/ 180.0 
cam_proj = perspectiveprojectionnp(camera_fov_y, 1.0 )
cam_proj =  torch.FloatTensor(cam_proj).cuda()
model = Encoder(4, 5, args.batchsize, 137, mesh.vertices.shape[0] ).cuda()
renderer = Dib_Renderer(137, 137, mode = 'VertexColor')

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
		tgt_points = data['points'].cuda()
		inp_images = data['imgs'].cuda()
		image_gt = inp_images.permute(0,2,3,1)[:,:,:,:3]
		alhpa_gt = inp_images.permute(0,2,3,1)[:,:,:,3:]
		cam_mat = data['cam_mat'].cuda()
		cam_pos = data['cam_pos'].cuda()
		gt_verts = data['verts']
		gt_faces = data['faces']

		# inference 
		delta_verts = model(inp_images)

		# set viewing parameters 
		renderer.camera_params = [cam_mat, cam_pos, cam_proj]
	
		# predict mesh properties
		delta_verts, colours = model(inp_images)
		pred_verts = initial_verts + delta_verts
	
		# render image
		image_pred, _, _ = renderer.forward(points=[(pred_verts*.57 ), mesh.faces], colors=[colours])
		
		# mesh loss
		
		for verts, tgt, inp_img, pred_img, gt_v, gt_f in zip(pred_verts, tgt_points, inp_images, image_pred, gt_verts, gt_faces): 	
			mesh.vertices = verts
			pred_points, _ = mesh.sample(3000)	
			loss_epoch += 3000 * loss_fn(pred_points, tgt).item() / float(args.batchsize)	

			if args.f_score: 
			
				f_score = kal.metrics.point.f_score(tgt, pred_points, extend = False)
				f_epoch += (f_score  / float(args.batchsize)).item()

			if args.vis: 
				tgt_mesh = meshes_set_valid[num_items]
				tgt_mesh = kal.rep.TriangleMesh.from_tensors(gt_v, gt_f)

				print ('Displaying input image')
				img = inp_img.data.cpu().numpy().transpose((1, 2, 0))
				img = (img*255.).astype(np.uint8)
				Image.fromarray(img).show()
				input()
				print ('Displaying predicted image')
				img = pred_img.data.cpu().numpy()
				img = (img*255.).astype(np.uint8)
				Image.fromarray(img).show()
				input()

				print ('Rendering Target Mesh')
				kal.visualize.show_mesh(tgt_mesh)
				print ('Rendering Predicted Mesh')
				mesh.show()
				print('----------------------')
				num_items += 1


		
		num_batches += 1.

out_loss = loss_epoch / float(num_batches)
print ('Loss over validation set is {0}'.format(out_loss))
if args.f_score: 
	out_f = f_epoch / float(num_batches)
	print ('F-score over validation set is {0}'.format(out_f))
