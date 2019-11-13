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

from utils import preprocess, loss_lap, collate_fn, normalize_adj, loss_flat
from graphics.render.base import Render as Dib_Renderer
from graphics.utils.utils_perspective import  perspectiveprojectionnp
from architectures import Encoder

import kaolin as kal 


"""
Commandline arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument('-expid', type=str, default='Direct', help='Unique experiment identifier.')
parser.add_argument('-categories', type=str,nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('-epochs', type=int, default=500, help='Number of train epochs.')
parser.add_argument('-batchsize', type=int, default=16, help='Batch size.')
parser.add_argument('-lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('-val-every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('-print-every', type=int, default=20, help='Print frequency (batches).')
parser.add_argument('-logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('-save-model', action='store_true', help='Saves the model and a snapshot \
	of the optimizer state.')
args = parser.parse_args()



"""
Dataset
"""
sdf_set = kal.dataloader.ShapeNet.SDF_Points(root ='../../datasets/',categories =args.categories , \
	download = True, train = True, split = .7, num_points=3000 )
point_set = kal.dataloader.ShapeNet.Points(root ='../../datasets/',categories =args.categories , \
	download = True, train = True, split = .7, num_points=3000 )
images_set = kal.dataloader.ShapeNet.Images(root ='../../datasets/',categories =args.categories , \
	download = True, train = True,  split = .7, views=23, transform= preprocess )
train_set = kal.dataloader.ShapeNet.Combination([sdf_set, images_set, point_set], root='../../kaolin/datasets/')

dataloader_train = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, num_workers=8)




images_set_valid = kal.dataloader.ShapeNet.Images(root ='../../datasets/',categories =args.categories , \
	download = True, train = False,  split = .7, views=1, transform= preprocess )
dataloader_val = DataLoader(images_set_valid, batch_size=args.batchsize, shuffle=False, 
	num_workers=8)



"""
Model settings 
"""
mesh = kal.rep.TriangleMesh.from_obj('386.obj', enable_adjacency= True)
mesh.cuda()
normalize_adj(mesh)
    

initial_verts = mesh.vertices.clone()
camera_fov_y = 49.13434207744484 * np.pi/ 180.0 
cam_proj = perspectiveprojectionnp(camera_fov_y, 1.0 )
cam_proj =  torch.FloatTensor(cam_proj).cuda()

model = Encoder(4, 5, args.batchsize, 137, mesh.vertices.shape[0] ).cuda()


optimizer = optim.Adam(model.parameters(), lr=args.lr)
renderer = Dib_Renderer(137, 137, mode = 'VertexColor')



# Create log directory, if it doesn't already exist
args.logdir = os.path.join(args.logdir, args.expid)
if not os.path.isdir(args.logdir):
	os.makedirs(args.logdir)
	print('Created dir:', args.logdir)

# Log all commandline args
with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
	json.dump(args.__dict__, f, indent=2)

class Engine(object):


	def __init__(self,  cur_epoch=0, print_every=1, validate_every=1):
		self.cur_epoch = cur_epoch
		self.train_loss = []
		self.val_loss = []
		self.bestval = 1000.

	def train(self):
		loss_epoch = 0.
		num_batches = 0

		model.train()
		# Train loop
		for i, data in enumerate(tqdm(dataloader_train), 0):
			optimizer.zero_grad()
			
			# data creation
			tgt_points = data['points'].cuda()
			inp_images = data['imgs'].cuda()
			image_gt = inp_images.permute(0,2,3,1)[:,:,:,:3]
			alhpa_gt = inp_images.permute(0,2,3,1)[:,:,:,3:]
			cam_mat = data['cam_mat'].cuda()
			cam_pos = data['cam_pos'].cuda()

			# set viewing parameters 
			renderer.camera_params = [cam_mat, cam_pos, cam_proj]
		
			# predict mesh properties
			delta_verts, colours = model(inp_images)
			pred_verts = initial_verts + delta_verts
		
			# render image
			image_pred, alpha_pred, face_norms = renderer.forward(points=[(pred_verts*.57), mesh.faces], colors=[colours])
			
			# colour loss
			img_loss = ((image_pred - image_gt)**2).mean()

			# alpha loss 
			alpha_loss = ((alpha_pred - alhpa_gt)**2).mean()

			# mesh loss
			lap_loss = 0.
			flat_loss = 0.
			for verts, tgt, norms in zip(pred_verts, tgt_points, face_norms): 	
				lap_loss += .1*loss_lap(mesh) / float(args.batchsize)
				flat_loss += .0001 *loss_flat(mesh, norms) / float(args.batchsize)



			loss =  img_loss + alpha_loss + lap_loss + flat_loss 
			loss.backward()
			loss_epoch += float(loss.item())

			# logging
			num_batches += 1
			if i % args.print_every == 0:
				message = f'[TRAIN] Epoch {self.cur_epoch:03d}, Batch {i:03d}:, Img: {(img_loss.item()):4.3f}, '
				message = message + f' Alpha: {(alpha_loss.item()):3.3f}'
				message = message + f' Flat: {(flat_loss.item()):3.3f}, Lap: {(lap_loss.item()):3.3f} '
				
				tqdm.write(message)
			optimizer.step()
		
		
		loss_epoch = loss_epoch / num_batches
		self.train_loss.append(loss_epoch)
		self.cur_epoch += 1

		
		
	def validate(self):
		model.eval()
		with torch.no_grad():	
			num_batches = 0
			loss_epoch = 0.

			# Validation loop
			for i, data in enumerate(tqdm(dataloader_val), 0):

				# data creation
				inp_images = data['imgs'].cuda()
				image_gt = inp_images.permute(0,2,3,1)
				cam_mat = data['cam_mat'].cuda()
				cam_pos = data['cam_pos'].cuda()


				# set viewing parameters 
				renderer.camera_params = [cam_mat, cam_pos, cam_proj]
			
				# predict mesh properties
				delta_verts, colours = model(inp_images)
				pred_verts = initial_verts + delta_verts
			
				# render image
				image_pred, alpha_pred, _ = renderer.forward(points=[(pred_verts*.57 ), mesh.faces], colors=[colours])
				
				full_pred = torch.cat((image_pred, alpha_pred), dim = -1)
	
				# colour loss
				img_loss = ((full_pred - image_gt)**2).mean()
				loss_epoch += float(img_loss.item())

				# logging
				num_batches += 1
				if i % args.print_every == 0:
					out_loss = loss_epoch / float(num_batches)
					message = f'[VAL] Epoch {self.cur_epoch:03d}, Batch {i:03d}:, loss: {(out_loss):4.3f}'
					tqdm.write(message)
						
			out_loss = loss_epoch / float(num_batches)
			tqdm.write(f'[VAL Total] Epoch {self.cur_epoch:03d}, Batch {i:03d}: loss: {out_loss:4.5f}')

			self.val_loss.append(out_loss)

	def save(self):

		save_best = False
		if self.val_loss[-1] <= self.bestval:
			self.bestval = self.val_loss[-1]
			save_best = True
		
		# Create a dictionary of all data to save
		log_table = {
			'epoch': self.cur_epoch,
			'bestval': self.bestval,
			'train_loss': self.train_loss,
			'val_loss': self.val_loss
		}

		# Save the recent model/optimizer states
		torch.save(model.state_dict(), os.path.join(args.logdir, 'recent.pth'))
		torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))
		# Log other data corresponding to the recent model
		with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
			f.write(json.dumps(log_table))

		tqdm.write('====== Saved recent model ======>')
		
		if save_best:
			torch.save(model.state_dict(), os.path.join(args.logdir, 'best.pth'))
			torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
			tqdm.write('====== Overwrote best model ======>')
			
	
trainer = Engine()

for epoch in range(args.epochs): 
	trainer.train()
	if epoch %4 == 0:
		trainer.validate()
		trainer.save()