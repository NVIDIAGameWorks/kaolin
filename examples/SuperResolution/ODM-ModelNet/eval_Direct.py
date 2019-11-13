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

from torch.utils.data import DataLoader

from architectures import upscale
from utils import down_sample, up_sample
from dataloaders import ModelNet_ODMS
import kaolin as kal 


parser = argparse.ArgumentParser()
parser.add_argument('-expid', type=str, default='Direct', help='Unique experiment identifier.')
parser.add_argument('-device', type=str, default='cuda', help='Device to use')
parser.add_argument('-vis', action='store_true', help='Visualize each model while evaluating')
parser.add_argument('-categories', type=str,nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('-batchsize', type=int, default=16, help='Batch size.')
args = parser.parse_args()



# Data
valid_set = ModelNet_ODMS(root ='../../datasets/',categories = args.categories, \
	download = True, train = False)
dataloader_val = DataLoader(valid_set, batch_size=args.batchsize, shuffle=False, \
	num_workers=8)

# Model
model = upscale(30,15)
model = model.to(args.device)
# Load saved weights
model.load_state_dict(torch.load('log/{0}/best.pth'.format(args.expid)))


iou_epoch = 0.
iou_NN_epoch = 0.
num_batches = 0

model.eval()
with torch.no_grad():
	for data in tqdm(dataloader_val): 
		
		tgt_odms = data['odms'].to(args.device)
		tgt_voxels = data['voxels'].to(args.device)
		inp_voxels = down_sample(tgt_voxels)
		inp_odms = []
		for voxel in inp_voxels: 
			inp_odms.append(kal.rep.voxel.extract_odms(voxel).unsqueeze(0)) 
		inp_odms = torch.cat(inp_odms)

		pred_odms = model(inp_odms)*30

		NN_pred = up_sample(inp_voxels)
		iou_NN = kal.metrics.voxel.iou(NN_pred.contiguous(), tgt_voxels)
		iou_NN_epoch += iou_NN

		pred_odms = pred_odms.int()
		pred_voxels = []
		for odms, NN_voxel in zip(pred_odms, NN_pred): 
			pred_voxels.append(kal.rep.voxel.project_odms(odms, voxel= NN_voxel, votes = 2).unsqueeze(0))
		pred_voxels = torch.cat(pred_voxels)
		iou = kal.metrics.voxel.iou(pred_voxels.contiguous(), tgt_voxels)
		iou_epoch += iou

		

		num_batches += 1
		if args.vis: 
			for i in range(inp_voxels.shape[0]):	
				print ('Rendering low resolution input')
				kal.visualize.show_voxel(inp_voxels[i], mode = 'exact', thresh = .5)
				print ('Rendering high resolution target')
				kal.visualize.show_voxel(tgt_voxels[i], mode = 'exact', thresh = .5)
				print ('Rendering high resolution prediction')
				kal.visualize.show_voxel(pred_voxels[i], mode = 'exact', thresh = .5)
				print('----------------------')
iou_NN_epoch = iou_NN_epoch.item() / float(num_batches)
print ('IoU for Nearest Neighbor baseline over validation set is {0}'.format(iou_NN_epoch))
out_iou = iou_epoch.item() / float(num_batches)
print ('IoU over validation set is {0}'.format(out_iou))