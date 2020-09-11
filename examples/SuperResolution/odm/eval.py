# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os

import argparse
from tqdm import tqdm 
import torch
from torch.utils.data import DataLoader
import kaolin as kal
from kaolin.datasets import shapenet
from kaolin.models.VoxelSuperresODM import SuperresNetwork
from kaolin.conversions.voxelgridconversions import project_odms

from utils import up_sample, upsample_odm, to_occupancy_map, collate_fn
from preprocessor import VoxelODMs


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['Direct', 'MVD'], default='MVD')
parser.add_argument('--shapenet-root', type=str, help='Path to shapenet data directory.')
parser.add_argument('--cache-dir', type=str, help='Path to data directory.')
parser.add_argument('--expid', type=str, default='ODM', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--categories', type=str, nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('--vis', action='store_true', help='Visualize each model while evaluating')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size.')
args = parser.parse_args()

# Data
preprocessing_params = {'cache_dir': args.cache_dir}
val_set = shapenet.ShapeNet(root=args.shapenet_root, categories=args.categories, train=False, split=0.7,
                            preprocessing_params=preprocessing_params, preprocessing_transform=VoxelODMs())
dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=8, collate_fn=collate_fn)

# Log Directory
logdir = os.path.join(args.logdir, f'{args.expid}_{args.mode.lower()}')

# Model
model = SuperresNetwork(128, 32).to(args.device)
model.load_state_dict(torch.load(os.path.join(logdir, 'best.pth')))

iou_epoch = 0.
iou_NN_epoch = 0.
num_batches = 0

model.eval()
with torch.no_grad():
    for sample in tqdm(dataloader_val):
        data = sample['data']
        tgt_odms = data[128]['odms'].to(args.device)
        tgt_voxels = data[128]['voxels'].to(args.device)
        tgt_odms_occ = to_occupancy_map(tgt_odms)
        inp_odms = data[32]['odms'].to(args.device)
        inp_voxels = data[32]['voxels'].to(args.device)

        # Inference
        pred_odms = model(inp_odms)
        if args.mode == 'MVD':
            initial_odms = upsample_odm(inp_odms) * 4
            distance = 128 - initial_odms
            pred_odms_update = pred_odms * distance
            pred_odms = initial_odms + pred_odms_update

        # Calculate IoU
        if args.mode == 'Direct':
            pred_odms = to_occupancy_map(pred_odms, threshold=0.5)
            pred_odms = pred_odms * pred_odms.shape[-1]
        elif args.mode == 'MVD':
            pred_odms = pred_odms.int()

        NN_pred = up_sample(inp_voxels)
        iou_NN = kal.metrics.voxel.iou(NN_pred.contiguous(), tgt_voxels)
        iou_NN_epoch += iou_NN

        pred_voxels = []
        for odms, voxel_NN in zip(pred_odms, NN_pred): 
            pred_voxels.append(project_odms(odms, voxel_NN, votes=2).unsqueeze(0))
        pred_voxels = torch.cat(pred_voxels)
        iou = kal.metrics.voxel.iou(pred_voxels.contiguous(), tgt_voxels)
        iou_epoch += iou

        if args.vis: 
            for i in range(inp_voxels.shape[0]):	
                print('Rendering low resolution input')
                kal.visualize.show_voxelgrid(inp_voxels[i], mode='exact', thresh=.5)
                print('Rendering high resolution target')
                kal.visualize.show_voxelgrid(tgt_voxels[i], mode='exact', thresh=.5)
                print('Rendering high resolution prediction')
                kal.visualize.show_voxelgrid(pred_voxels[i], mode='exact', thresh=.5)
                print('----------------------')
        num_batches += 1  
iou_NN_epoch = iou_NN_epoch.item() / num_batches
print(f'IoU for Nearest Neighbor baseline over validation set is {iou_NN_epoch}')		
out_iou = iou_epoch.item() / num_batches
print(f'IoU over validation set is {out_iou}')
