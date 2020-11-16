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
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

import kaolin as kal
from kaolin.datasets import shapenet
from kaolin.models.OccupancyNetwork import OccupancyNetwork

from utils import occ_function, collate_fn, extract_mesh, preprocess

parser = argparse.ArgumentParser()
parser.add_argument('--shapenet-root', type=str, help='Root directory of the ShapeNet dataset.')
parser.add_argument('--shapenet-images-root', type=str, help='Root directory of the ShapeNet Rendering dataset.')
parser.add_argument('--cache-dir', type=str, default='cache', help='Path to write intermediate representation to.')
parser.add_argument('--expid', type=str, default='OccNet_1', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--categories', type=str, nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('--no-vis', action='store_true', help='Visualize each model while evaluating')
parser.add_argument('--f-score', action='store_true', help='compute F-score')
parser.add_argument('--batch-size', type=int, default=3, help='Batch size.')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')

args = parser.parse_args()


# Common Params
common_params = {
    'root': args.shapenet_root,
    'categories': args.categories,
    'cache_dir': args.cache_dir,
    'train': False,
    'split': 0.7,
}

img_params = {
    'root': args.shapenet_images_root,
    'categories': args.categories,
    'train': False,
    'split': 0.7,
}

# Data
sdf_set = shapenet.ShapeNet_SDF_Points(**common_params, resolution=150, num_points=100000, occ=True)
point_set = shapenet.ShapeNet_Points(**common_params, resolution=150, num_points=15000)
common_params.pop('cache_dir')
mesh_set = shapenet.ShapeNet_Meshes(**common_params)
images_set = shapenet.ShapeNet_Images(**img_params, views=1, transform=preprocess)
valid_set = shapenet.ShapeNet_Combination([sdf_set, point_set, mesh_set, images_set])


dataloader_val = DataLoader(valid_set, batch_size=5, shuffle=False, num_workers=8, collate_fn=collate_fn)


# Model
model = OccupancyNetwork(args.device)

# Load saved weights
logdir = os.path.join(args.logdir, args.expid)
checkpoint = torch.load(os.path.join(logdir, 'best.ckpt'))
model.load_state_dict(checkpoint['model'])

iou_epoch = 0.
f_epoch = 0.
num_batches = 0

THRESHOLD = 0.2
PADDING = 0.1
UPSAMPLING_STEPS = 2
RESOLUTION_0 = 64

box_size = 1 + PADDING

with torch.no_grad():
    model.encoder.eval()
    model.decoder.eval() 
    for sample in tqdm(dataloader_val):
        data = sample['data']
        imgs = data['images'][:, :3].to(args.device)
        sdf_points = data['occ_points'].to(args.device)
        surface_points = data['points'].to(args.device)
        gt_occ = data['occ_values'].to(args.device)

        encoding = model.encode_inputs(imgs)
        z = model.get_z_from_prior((1,), sample=True).to(args.device)
        pred_occ = model.decode(sdf_points, torch.zeros(args.batch_size, 0), encoding).probs

        i = 0
        for sdf_point, gt_oc, pred_oc, gt_surf, code in zip(sdf_points, gt_occ, pred_occ, surface_points, encoding):
            # Compute IoU
            iou_epoch += kal.metrics.point.iou(gt_oc, pred_oc, thresh=THRESHOLD) / gt_occ.shape[0]
            if args.f_score or not args.no_vis:
                # Extract mesh from sdf
                sdf = occ_function(model, code.unsqueeze(0))
                voxelgrid = kal.conversions.sdf_to_voxelgrid(
                    sdf, resolution=RESOLUTION_0, upsampling_steps=UPSAMPLING_STEPS,
                    threshold=THRESHOLD, bbox_dim=box_size)
                verts, faces = extract_mesh(voxelgrid, model, z, encoding)
                verts, faces = verts.to(args.device), faces.to(args.device)
                mesh = kal.rep.TriangleMesh.from_tensors(verts, faces)
                if verts.shape[0] == 0:     # if mesh is empty count as 0 f-score
                    continue 

                if not args.no_vis: 
                    tgt_verts = data['vertices'][i]
                    tgt_faces = data['faces'][i]
                    tgt_mesh = kal.rep.TriangleMesh.from_tensors(tgt_verts, tgt_faces)

                    print('Displaying input image')
                    img = imgs[i].data.cpu().numpy().transpose((1, 2, 0)) * 255
                    img = (img).astype(np.uint8)
                    Image.fromarray(img).show()
                    print('Rendering Target Mesh')
                    kal.visualize.show_mesh(tgt_mesh)
                    print('Rendering Predicted Mesh')
                    mesh.show()
                    print('----------------------')

                if args.f_score:
                    # Compute F-score
                    pred_surf, _ = mesh.sample(15000)
                    f_score = kal.metrics.point.f_score(gt_surf, pred_surf, extend=False)
                    f_epoch += (f_score / gt_occ.shape[0])
            i += 1		

        num_batches += 1.

out_iou = iou_epoch / float(num_batches)
print(f'IoU over validation set is {out_iou}')
if args.f_score: 
    out_f = f_epoch / float(num_batches)
    print(f'F-score over validation set is {out_f}')
