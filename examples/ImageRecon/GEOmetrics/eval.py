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
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import preprocess, pooling, get_pooling_index
from utils import setup_meshes, split_meshes, reset_meshes
from utils import collate_fn
from architectures import VGG as Encoder, G_Res_Net
from PIL import Image

import kaolin as kal 
from kaolin.datasets import shapenet

parser = argparse.ArgumentParser()
parser.add_argument('--shapenet-root', type=str, help='Root directory of the ShapeNet dataset.')
parser.add_argument('--shapenet-rendering-root', type=str, help='Root directory of the ShapeNet Rendering dataset.')
parser.add_argument('--cache-dir', type=str, default=None, help='Path to write intermediate representation to.')
parser.add_argument('--expid', type=str, default='Direct', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--categories', type=str, nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('--no-vis', action='store_true', help='Turn off visualization of each model while evaluating')
parser.add_argument('--f_score', action='store_true', help='compute F-score')
parser.add_argument('--batch-size', type=int, default=1, help='Batch size.')
parser.add_argument('--logdir', type=str, default='log', help='Directory where log data was saved to.')
args = parser.parse_args()


# Data
points_set_valid = shapenet.ShapeNet_Points(root=args.shapenet_root, cache_dir=args.cache_dir, categories=args.categories,
                                            train=False, split=.7, num_points=5000)
images_set_valid = shapenet.ShapeNet_Images(root=args.shapenet_rendering_root, categories=args.categories,
                                            train=False, split=.7, views=1, transform=preprocess)
meshes_set_valid = shapenet.ShapeNet_Meshes(root=args.shapenet_root, categories=args.categories,
                                            train=False, split=.7)
valid_set = shapenet.ShapeNet_Combination([points_set_valid, images_set_valid, meshes_set_valid])

dataloader_val = DataLoader(valid_set, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, 
                            num_workers=8)

# Model
meshes = setup_meshes(filename='meshes/386.obj', device=args.device)

encoders = [Encoder().to(args.device) for i in range(3)]
mesh_update_kernels = [963, 1091, 1091] 
mesh_updates = [G_Res_Net(mesh_update_kernels[i], hidden=128, output_features=3).to(args.device) for i in range(3)]

logdir = os.path.join(args.logdir, args.expid)

# Load saved weights
for i, e in enumerate(encoders):
    e.load_state_dict(torch.load(os.path.join(logdir, f'best_encoder_{i}.pth')))
    e.eval()
for i, m in enumerate(mesh_updates):
    m.load_state_dict(torch.load(os.path.join(logdir, f'best_mesh_update_{i}.pth')))
    m.eval()
encoding_dims = [56, 28, 14, 7]

loss_epoch = 0.
f_epoch = 0.
num_batches = 0
num_items = 0

with torch.no_grad():
    for sample in tqdm(valid_set):
        data = sample['data']
        # data creation
        tgt_points = data['points'].to(args.device)
        inp_images = data['images'].to(args.device).unsqueeze(0)
        cam_mat = data['params']['cam_mat'].to(args.device)
        cam_pos = data['params']['cam_pos'].to(args.device)
        tgt_verts = data['vertices'].to(args.device)
        tgt_faces = data['faces'].to(args.device)

        # Inference
        img_features = [e(inp_images) for e in encoders]

        reset_meshes(meshes)
        # Layer_1
        pool_indices = get_pooling_index(meshes['init'][0].vertices, cam_mat, cam_pos, encoding_dims)
        projected_image_features = pooling(img_features[0], pool_indices, 0)
        full_vert_features = torch.cat((meshes['init'][0].vertices, projected_image_features), dim=1)

        delta, future_features = mesh_updates[0](full_vert_features, meshes['adjs'][0])
        meshes['update'][0].vertices = (meshes['init'][0].vertices + delta.clone())
        future_features = split_meshes(meshes, future_features, 0)			

        # Layer_2
        pool_indices = get_pooling_index(meshes['init'][1].vertices, cam_mat, cam_pos, encoding_dims)
        projected_image_features = pooling(img_features[1], pool_indices, 0)
        full_vert_features = torch.cat((meshes['init'][1].vertices, projected_image_features, future_features), dim=1)

        delta, future_features = mesh_updates[1](full_vert_features, meshes['adjs'][1])
        meshes['update'][1].vertices = (meshes['init'][1].vertices + delta.clone())
        future_features = split_meshes(meshes, future_features, 1)	

        # Layer_3
        pool_indices = get_pooling_index(meshes['init'][2].vertices, cam_mat, cam_pos, encoding_dims)
        projected_image_features = pooling(img_features[2], pool_indices, 0)
        full_vert_features = torch.cat((meshes['init'][2].vertices, projected_image_features, future_features), dim=1)

        delta, future_features = mesh_updates[2](full_vert_features, meshes['adjs'][2])
        meshes['update'][2].vertices = (meshes['init'][2].vertices + delta.clone())

        pred_points, _ = meshes['update'][2].sample(5000)

        loss = 3000 * kal.metrics.point.chamfer_distance(pred_points, tgt_points)

        if not args.no_vis: 

            tgt_mesh = kal.rep.TriangleMesh.from_tensors(tgt_verts, tgt_faces)

            print('Displaying input image')
            img = inp_images[0].data.cpu().numpy().transpose((1, 2, 0))
            img = (img * 255.).astype(np.uint8)
            Image.fromarray(img).show()
            print('Rendering Target Mesh')
            kal.visualize.show_mesh(tgt_mesh)
            print('Rendering Predicted Mesh')
            kal.visualize.show_mesh(meshes['update'][2])
            print('----------------------')
            num_items += 1

        if args.f_score: 
            # Compute f score
            f_score = kal.metrics.point.f_score(tgt_points, pred_points, extend=False)
            f_epoch += (f_score / float(args.batch_size)).item()

        loss_epoch += loss.item()

        num_batches += 1.

out_loss = loss_epoch / float(num_batches)
print('Loss over validation set is {0}'.format(out_loss))
if args.f_score: 
    out_f = f_epoch / float(num_batches)
    print('F-score over validation set is {0}'.format(out_f))
