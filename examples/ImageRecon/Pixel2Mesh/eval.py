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

from utils import preprocess, pooling, get_pooling_index
from utils import setup_meshes, split
from architectures import VGG as Encoder, G_Res_Net
import kaolin as kal

parser = argparse.ArgumentParser()
parser.add_argument('-expid', type=str, default='Direct', help='Unique experiment identifier.')
parser.add_argument('-device', type=str, default='cuda', help='Device to use')
parser.add_argument('-categories', type=str, nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('-vis', action='store_true', help='Visualize each model while evaluating')
parser.add_argument('-batchsize', type=int, default=1, help='Batch size.')
parser.add_argument('-f_score', action='store_true', help='compute F-score')
args = parser.parse_args()


# Data
points_set_valid = kal.datasets.ShapeNet_Points(root='../../datasets/', cache_dir='cache/', categories=args.categories,
                                                train=False, split=.7, num_points=5000)
images_set_valid = kal.datasets.ShapeNet_Images(root='../../datasets/', categories=args.categories,
                                                train=False, split=.7, views=1, transform=preprocess)
meshes_set_valid = kal.datasets.ShapeNet_Meshes(root='../../datasets/', categories=args.categories,
                                                train=False, split=.7)
valid_set = kal.datasets.ShapeNet_Combination(
    [points_set_valid, images_set_valid, meshes_set_valid], root='../../datasets/')


# Model
meshes = setup_meshes(filename='meshes/156.obj', device=args.device)

encoder = Encoder().to(args.device)
mesh_update_kernels = [963, 1091, 1091]
mesh_updates = [G_Res_Net(mesh_update_kernels[i], hidden=128, output_features=3).to(args.device) for i in range(3)]

# Load saved weights
encoder.load_state_dict(torch.load('log/{0}/best_encoder.pth'.format(args.expid)))
for i, m in enumerate(mesh_updates):
    m.load_state_dict(torch.load('log/{}/best_mesh_update_{}.pth'.format(args.expid, i)))
encoding_dims = [56, 28, 14, 7]

loss_epoch = 0.
f_epoch = 0.
num_batches = 0
num_items = 0
loss_fn = kal.metrics.point.chamfer_distance

encoder.eval(), [m.eval() for m in mesh_updates]
with torch.no_grad():
    for data in tqdm(valid_set):
        # data creation
        tgt_points = data['points'].to(args.device)
        inp_images = data['imgs'].to(args.device).unsqueeze(0)
        cam_mat = data['cam_mat'].to(args.device)
        cam_pos = data['cam_pos'].to(args.device)

        ###############################
        ########## inference ##########
        ###############################
        img_features = encoder(inp_images)

        ##### layer_1 #####
        pool_indices = get_pooling_index(meshes['init'][0].vertices, cam_mat, cam_pos, encoding_dims)
        projected_image_features = pooling(img_features, pool_indices)
        full_vert_features = torch.cat((meshes['init'][0].vertices, projected_image_features), dim=1)

        pred_verts, future_features = mesh_updates[0](full_vert_features, meshes['adjs'][0])
        meshes['update'][0].vertices = pred_verts.clone()

        ##### layer_2 #####
        future_features = split(meshes, future_features, 0)
        pool_indices = get_pooling_index(meshes['init'][1].vertices, cam_mat, cam_pos, encoding_dims)
        projected_image_features = pooling(img_features, pool_indices)
        full_vert_features = torch.cat((meshes['init'][1].vertices, projected_image_features, future_features), dim=1)

        pred_verts, future_features = mesh_updates[1](full_vert_features, meshes['adjs'][1])
        meshes['update'][1].vertices = pred_verts.clone()

        ##### layer_3 #####
        future_features = split(meshes, future_features, 1)
        pool_indices = get_pooling_index(meshes['init'][2].vertices, cam_mat, cam_pos, encoding_dims)
        projected_image_features = pooling(img_features, pool_indices)
        full_vert_features = torch.cat((meshes['init'][2].vertices, projected_image_features, future_features), dim=1)

        pred_verts, future_features = mesh_updates[2](full_vert_features, meshes['adjs'][2])
        meshes['update'][2].vertices = pred_verts.clone()

        meshes['update'][2].vertices = pred_verts.clone()
        pred_points, _ = meshes['update'][2].sample(5000)

        loss = 3000 * kal.metrics.point.chamfer_distance(pred_points, tgt_points)

        if args.vis:
            tgt_mesh = meshes_set_valid[num_items]
            tgt_verts = tgt_mesh['verts']
            tgt_faces = tgt_mesh['faces']
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
            #### compute f score ####
            f_score = kal.metrics.point.f_score(tgt_points, pred_points, extend=False)
            f_epoch += (f_score / float(args.batchsize)).item()

        loss_epoch += loss.item()

        num_batches += 1.

out_loss = loss_epoch / float(num_batches)
print('Loss over validation set is {0}'.format(out_loss))
if args.f_score:
    out_f = f_epoch / float(num_batches)
    print('F-score over validation set is {0}'.format(out_f))
