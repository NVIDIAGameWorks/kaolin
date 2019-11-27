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

import os
import argparse
import torch
from tqdm import tqdm

import kaolin as kal
from kaolin.models.GEOMetrics import VoxelDecoder
from kaolin.models.MeshEncoder import MeshEncoder
from kaolin.datasets import shapenet


parser = argparse.ArgumentParser()
parser.add_argument('--shapenet-root', type=str, help='Root directory of the ShapeNet dataset.')
parser.add_argument('--cache-dir', type=str, default='cache', help='Path to write intermediate representation to.')
parser.add_argument('--expid', type=str, default='GEOmetrics_1', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use.')
parser.add_argument('--categories', type=str, nargs='+', default=['chair'], help='list of object classes to use.')
parser.add_argument('--no-vis', action='store_true', help='Disable visualization of each model.')
args = parser.parse_args()


# Data
mesh_set = shapenet.ShapeNet_Surface_Meshes(root=args.shapenet_root, cache_dir=args.cache_dir, categories=args.categories,
                                            resolution=32, train=False, split=.7, mode='Tri')
voxel_set = shapenet.ShapeNet_Voxels(root=args.shapenet_root, cache_dir=args.cache_dir, categories=args.categories,
                                     train=False, resolutions=[32], split=.7)
valid_set = shapenet.ShapeNet_Combination([mesh_set, voxel_set])


encoder = MeshEncoder(30).to(args.device)
decoder = VoxelDecoder(30).to(args.device)

logdir = f'log/{args.expid}/AutoEncoder'
checkpoint = torch.load(os.path.join(logdir, 'best.ckpt'))
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])

loss_epoch = 0.
num_batches = 0
num_items = 0

encoder.eval(), decoder.eval()
with torch.no_grad():
    for sample in tqdm(valid_set):
        data = sample['data']

        tgt_voxels = data['32'].to(args.device)
        inp_verts = data['vertices'].to(args.device)
        inp_faces = data['faces'].to(args.device)
        inp_adj = data['adj'].to(args.device)

        # Inference
        latent_encoding = encoder(inp_verts, inp_adj).unsqueeze(0)
        pred_voxels = decoder(latent_encoding)[0]

        # Losses
        iou = kal.metrics.voxel.iou(pred_voxels.contiguous(), tgt_voxels.contiguous())

        if not args.no_vis: 
            tgt_mesh = kal.rep.TriangleMesh.from_tensors(inp_verts, inp_faces)
            print('Rendering Input Mesh')
            tgt_mesh.show()
            print('Rendering Target Voxels')
            kal.visualize.show_voxelgrid(tgt_voxels, mode='exact', thresh=.5)
            print('Rendering Predicted Voxels')
            kal.visualize.show_voxelgrid(pred_voxels, mode='exact', thresh=.5)
            print('----------------------')
            num_items += 1

        loss_epoch += iou.item()
        num_batches += 1.

out_loss = loss_epoch / float(num_batches)
print(f'IoU over validation set is {out_loss}')
