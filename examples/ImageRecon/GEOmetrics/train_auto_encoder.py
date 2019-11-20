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
import random


from architectures import MeshEncoder, VoxelDecoder

import kaolin as kal 
"""
Commandline arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument('-expid', type=str, default='Direct', help='Unique experiment identifier.')
parser.add_argument('-device', type=str, default='cuda', help='Device to use')
parser.add_argument('-categories', type=str,nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('-epochs', type=int, default=50, help='Number of train epochs.')
parser.add_argument('-lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('-val-every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('-batch_size', type=int, default=25, help='batch size')
parser.add_argument('-print-every', type=int, default=20, help='Print frequency (batches).')
parser.add_argument('-logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('-save-model', action='store_true', help='Saves the model and a snapshot \
	of the optimizer state.')
args = parser.parse_args()



"""
Dataset
"""
mesh_set = kal.datasets.ShapeNet.Surface_Meshes(root ='../../datasets/',categories =args.categories , \
	resolution = 32, download = True, train = True, split = .7, mode = 'Tri' )
voxel_set = kal.datasets.ShapeNet.Voxels(root ='../../datasets/',categories =args.categories , \
	download = True, train = True, resolutions=[32], split = .7 )

train_set = kal.datasets.ShapeNet.Combination([mesh_set, voxel_set], root='../../datasets/')




mesh_set = kal.datasets.ShapeNet.Surface_Meshes(root ='../../datasets/',categories =args.categories , \
	resolution = 32, download = True, train = False, split = .7, mode = 'Tri' )
voxel_set = kal.datasets.ShapeNet.Voxels(root ='../../datasets/',categories =args.categories , \
	download = True, train = False, resolutions=[32], split = .7 )
valid_set = kal.datasets.ShapeNet.Combination([mesh_set, voxel_set], root='../../datasets/')




"""
Model settings 
"""
encoder = MeshEncoder(30).to(args.device)
decoder = VoxelDecoder(30).to(args.device)




parameters =  list(encoder.parameters()) +  list(decoder.parameters()) 
optimizer = optim.Adam(parameters, lr=args.lr)

loss_fn = torch.nn.MSELoss()


# Create log directory, if it doesn't already exist
args.logdir = os.path.join(args.logdir, args.expid)
if not os.path.isdir(args.logdir):
	os.makedirs(args.logdir)
	print('Created dir:', args.logdir)

# Log all commandline args
with open(os.path.join(args.logdir, 'args.txt'), 'w') as f:
	json.dump(args.__dict__, f, indent=2)
 

class Engine(object):
	"""Engine that runs training and inference.
	Args
		- cur_epoch (int): Current epoch.
		- print_every (int): How frequently (# batches) to print loss.
		- validate_every (int): How frequently (# epochs) to run validation.
		
	"""

	def __init__(self,  cur_epoch=0, print_every=1, validate_every=1):
		self.cur_epoch = cur_epoch
		self.train_loss = []
		self.val_loss = []
		self.bestval = 0.

	def train(self):
		loss_epoch = 0.
		num_batches = 0
		encoder.train(), decoder.train()

		# Train loop
		for i in tqdm(range(len(train_set)//args.batch_size)): 
			from time import time 
			
			tgt_voxels = []
			latent_encodings = []
			for j in range(args.batch_size):
				optimizer.zero_grad()
				
			###############################
			####### data creation #########
			###############################
				selection = random.randint(0,len(train_set)-1)
				tgt_voxels.append(train_set[selection]['32'].to(args.device).unsqueeze(0))
				inp_verts = train_set[selection]['verts'].to(args.device)
				inp_adj = train_set[selection]['adj'].to(args.device)
				
			###############################
			########## inference ##########
			###############################
				latent_encodings.append(encoder(inp_verts, inp_adj).unsqueeze(0))
			
			tgt_voxels = torch.cat(tgt_voxels)
			latent_encodings = torch.cat(latent_encodings)
			pred_voxels = decoder(latent_encodings)

			###############################
			########## losses #############
			###############################
			loss = loss_fn(pred_voxels, tgt_voxels)
			loss.backward()
			loss_epoch += float(loss.item())

			# logging
			iou = kal.metrics.voxel.iou(pred_voxels.contiguous(), tgt_voxels)
			num_batches += 1
			if i % args.print_every == 0:
				tqdm.write(f'[TRAIN] Epoch {self.cur_epoch:03d}, Batch {i:03d}: Loss: {float(loss.item())}')
				tqdm.write('Metric iou: {0}'.format(iou))
			optimizer.step()
		loss_epoch = loss_epoch / num_batches
		self.train_loss.append(loss_epoch)
		self.cur_epoch += 1

		
		
	def validate(self):
		encoder.eval(), decoder.eval()
		with torch.no_grad():	
			num_batches = 0
			iou_epoch = 0.

			# Validation loop
			for i in tqdm(range(len(valid_set)//args.batch_size)): 
				tgt_voxels = []
				latent_encodings = []
				for j in range(args.batch_size):
					optimizer.zero_grad()
					
				###############################
				####### data creation #########
				###############################
					tgt_voxels.append(valid_set[i*args.batch_size + j]['32'].to(args.device).unsqueeze(0))
					inp_verts = valid_set[i*args.batch_size + j]['verts'].to(args.device)
					inp_adj = valid_set[i*args.batch_size + j]['adj'].to(args.device)
					
				###############################
				########## inference ##########
				###############################
					latent_encodings.append(encoder(inp_verts, inp_adj).unsqueeze(0))

				tgt_voxels = torch.cat(tgt_voxels)
				latent_encodings = torch.cat(latent_encodings)
				pred_voxels = decoder(latent_encodings)

				###############################
				########## losses #############
				###############################
			
				iou = kal.metrics.voxel.iou(pred_voxels.contiguous(), tgt_voxels)
				iou_epoch += iou

					# logging
				num_batches += 1
				if i % args.print_every == 0:
						out_iou = iou_epoch.item() / float(num_batches)
						tqdm.write(f'[VAL] Epoch {self.cur_epoch:03d}, Batch {i:03d}: IoU: {out_iou}')
						
			out_iou = iou_epoch.item() / float(num_batches)
			tqdm.write(f'[VAL Total] Epoch {self.cur_epoch:03d}, Batch {i:03d}: IoU: {out_iou}')
			self.val_loss.append(out_iou)

	def save(self):

		save_best = False
		if self.val_loss[-1] >= self.bestval:
			self.bestval = self.val_loss[-1]
			save_best = True
		
		# Create a dictionary of all data to save
		log_table = {
			'epoch': self.cur_epoch,
			'bestval':self.bestval,
			'train_loss': self.train_loss,
			'val_loss': self.val_loss,
			'train_metrics': ['Chamfer'],
			'val_metrics': ['Chamfer'],
		}

		# Save the recent model/optimizer states
		
		torch.save(encoder.state_dict(), os.path.join(args.logdir, 'auto_encoder.pth'))
		torch.save(decoder.state_dict(), os.path.join(args.logdir, 'auto_decoder.pth'))
		torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))
		# Log other data corresponding to the recent model
		with open(os.path.join(args.logdir, 'auto_recent.log'), 'w') as f:
			f.write(json.dumps(log_table))

		tqdm.write('====== Saved recent model ======>')
		
		if save_best:
			
			torch.save(encoder.state_dict(), os.path.join(args.logdir, 'auto_best_encoder.pth'))
			torch.save(decoder.state_dict(), os.path.join(args.logdir, 'auto_best_decoder.pth'))
			torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
			tqdm.write('====== Overwrote best model ======>')
			
	
trainer = Engine()

for epoch in range(args.epochs): 
	trainer.train()
	if epoch % 1 == 0: 
		trainer.validate()
		trainer.save()
		
		
