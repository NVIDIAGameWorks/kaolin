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
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import preprocess, pooling, get_pooling_index
from utils import setup_meshes, split
from utils import loss_surf, loss_edge, loss_lap, loss_norm
from kaolin.models.Pixel2Mesh import VGG as Encoder, G_Res_Net

import kaolin as kal

parser = argparse.ArgumentParser()
parser.add_argument('--shapenet-root', type=str, required=True,
                    help='Root directory of the ShapeNet dataset.')
parser.add_argument('--shapenet-images-root', type=str, required=True,
                    help='Root directory of the ShapeNet images dataset.')
parser.add_argument('--cache-dir', type=str, default=None,
                    help='Path to writer to save intermediate representations.')
parser.add_argument('--expid', type=str, default='Pixel2Mesh',
                    help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda',
                    help='Device to use')
parser.add_argument('--categories', type=str, nargs='+', default=['chair'],
                    help='list of object classes to use')
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of train epochs.')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate.')
parser.add_argument('--val-every', type=int, default=5,
                    help='Validation frequency (epochs).')
parser.add_argument('--print-every', type=int, default=20,
                    help='Print frequency (batches).')
parser.add_argument('--logdir', type=str, default='log',
                    help='Directory to log data to.')
parser.add_argument('--save-model', action='store_true',
                    help='Saves the model and a snapshot of the optimizer state.')
args = parser.parse_args()

# Dataset settings
points_set = kal.datasets.ShapeNet_Points(root=args.shapenet_root, categories=args.categories,
                                          train=True, split=.7, num_points=2466)
images_set = kal.datasets.ShapeNet_Images(root=args.shapenet_images_root, categories=args.categories,
                                          train=True, split=.7, views=24, transform=preprocess)
train_set = kal.datasets.ShapeNet_Combination([points_set, images_set])
dataloader_train = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8)

points_set_valid = kal.datasets.ShapeNet_Points(root=args.shapenet_root, categories=args.categories,
                                                train=False, split=.7, num_points=10000)
images_set_valid = kal.datasets.ShapeNet_Images(root=args.shapenet_images_root, categories=args.categories,
                                                train=False, split=.7, views=1, transform=preprocess)
valid_set = kal.datasets.ShapeNet_Combination([points_set_valid, images_set_valid])
dataloader_val = DataLoader(valid_set, batch_size=1, shuffle=False, num_workers=8)

# Model settings
meshes = setup_meshes(filename='meshes/156.obj', device=args.device)

encoder = Encoder().to(args.device)
mesh_update_kernels = [963, 1091, 1091]
mesh_updates = [G_Res_Net(mesh_update_kernels[i], hidden=128, output_features=3).to(args.device)
                for i in range(3)]

parameters = list(encoder.parameters())
for i in range(3):
    parameters += list(mesh_updates[i].parameters())
optimizer = optim.Adam(parameters, lr=args.lr)

encoding_dims = [56, 28, 14, 7]

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

    def __init__(self, cur_epoch=0, print_every=1, validate_every=1):
        self.cur_epoch = cur_epoch
        self.train_loss = []
        self.val_loss = []
        self.bestval = 1000.

    def train(self):
        loss_epoch = 0.
        num_batches = 0
        encoder.train(), [m.train() for m in mesh_updates]

        # Train loop
        for i, data in enumerate(tqdm(dataloader_train), 0):
            optimizer.zero_grad()
            ##############
            #### DATA ####
            ##############
            data = data['data']
            tgt_points = data['points'].to(args.device)[0]
            tgt_norms = data['normals'].to(args.device)[0]
            inp_images = data['images'].to(args.device)
            cam_mat = data['params']['cam_mat'].to(args.device)[0]
            cam_pos = data['params']['cam_pos'].to(args.device)[0]

            ###################
            #### INFERENCE ####
            ###################
            img_features = encoder(inp_images)

            # layer_1
            pool_indices = get_pooling_index(meshes['init'][0].vertices, cam_mat, cam_pos, encoding_dims)
            projected_image_features = pooling(img_features, pool_indices)
            full_vert_features = torch.cat((meshes['init'][0].vertices, projected_image_features), dim=1)

            pred_verts, future_features = mesh_updates[0](full_vert_features, meshes['adjs'][0])
            meshes['update'][0].vertices = pred_verts.clone()

            # layer_2
            future_features = split(meshes, future_features, 0)
            pool_indices = get_pooling_index(meshes['init'][1].vertices, cam_mat, cam_pos, encoding_dims)
            projected_image_features = pooling(img_features, pool_indices)
            full_vert_features = torch.cat((meshes['init'][1].vertices, projected_image_features, future_features),
                                           dim=1)
            pred_verts, future_features = mesh_updates[1](full_vert_features, meshes['adjs'][1])
            meshes['update'][1].vertices = pred_verts.clone()

            # layer_3
            future_features = split(meshes, future_features, 1)
            pool_indices = get_pooling_index(meshes['init'][2].vertices, cam_mat, cam_pos, encoding_dims)
            projected_image_features = pooling(img_features, pool_indices)
            full_vert_features = torch.cat((meshes['init'][2].vertices, projected_image_features, future_features),
                                           dim=1)

            pred_verts, future_features = mesh_updates[2](full_vert_features, meshes['adjs'][2])
            meshes['update'][2].vertices = pred_verts.clone()

            ################
            #### LOSSES ####
            ################
            surf_loss = 3000 * loss_surf(meshes, tgt_points)
            edge_loss = 300 * loss_edge(meshes)
            lap_loss = 1500 * loss_lap(meshes)
            norm_loss = .5 * loss_norm(meshes, tgt_points, tgt_norms)
            loss = surf_loss + edge_loss + lap_loss + norm_loss
            loss.backward()
            loss_epoch += float(surf_loss.item())

            # logging
            num_batches += 1
            if i % args.print_every == 0:
                f_loss = kal.metrics.point.f_score(meshes['update'][2].sample(2466)[0], tgt_points, extend=False)
                message = f'[TRAIN] Epoch {self.cur_epoch:03d}, Batch {i:03d}:, Loss: {(surf_loss.item()):4.3f}, '
                message = (message + f'Lap: {(lap_loss.item()):3.3f}, Edge: {(edge_loss.item()):3.3f}, ' +
                           f'Norm: {(norm_loss.item()):3.3f}')
                message = message + f' F: {(f_loss.item()):3.3f}'
                tqdm.write(message)
            optimizer.step()

        loss_epoch = loss_epoch / num_batches
        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1

    def validate(self):
        encoder.eval(), [m.eval() for m in mesh_updates]
        with torch.no_grad():
            num_batches = 0
            loss_epoch = 0.
            f_loss = 0.

            # Validation loop
            for i, data in enumerate(tqdm(dataloader_val), 0):
                optimizer.zero_grad()

                ##############
                #### DATA ####
                ##############
                data = data['data']
                tgt_points = data['points'].to(args.device)[0]
                inp_images = data['images'].to(args.device)
                cam_mat = data['params']['cam_mat'].to(args.device)[0]
                cam_pos = data['params']['cam_pos'].to(args.device)[0]

                ###################
                #### INFERENCE ####
                ###################
                img_features = encoder(inp_images)

                # layer_1
                pool_indices = get_pooling_index(meshes['init'][0].vertices, cam_mat, cam_pos, encoding_dims)
                projected_image_features = pooling(img_features, pool_indices)
                full_vert_features = torch.cat((meshes['init'][0].vertices, projected_image_features), dim=1)

                pred_verts, future_features = mesh_updates[0](full_vert_features, meshes['adjs'][0])
                meshes['update'][0].vertices = pred_verts.clone()

                # layer_2
                future_features = split(meshes, future_features, 0)
                pool_indices = get_pooling_index(meshes['init'][1].vertices, cam_mat, cam_pos, encoding_dims)
                projected_image_features = pooling(img_features, pool_indices)
                full_vert_features = torch.cat((meshes['init'][1].vertices, projected_image_features, future_features),
                                               dim=1)

                pred_verts, future_features = mesh_updates[1](full_vert_features, meshes['adjs'][1])
                meshes['update'][1].vertices = pred_verts.clone()

                # layer_3
                future_features = split(meshes, future_features, 1)
                pool_indices = get_pooling_index(meshes['init'][2].vertices, cam_mat, cam_pos, encoding_dims)
                projected_image_features = pooling(img_features, pool_indices)
                full_vert_features = torch.cat((meshes['init'][2].vertices, projected_image_features, future_features),
                                               dim=1)

                pred_verts, future_features = mesh_updates[2](full_vert_features, meshes['adjs'][2])
                meshes['update'][2].vertices = pred_verts.clone()

                ################
                #### LOSSES ####
                ################
                f_loss += kal.metrics.point.f_score(meshes['update'][2].sample(2466)[0], tgt_points, extend=False)
                surf_loss = 3000 * kal.metrics.point.chamfer_distance(pred_verts.clone(), tgt_points)
                loss_epoch += surf_loss.item()

                # logging
                num_batches += 1
                if i % args.print_every == 0:
                    out_loss = loss_epoch / float(num_batches)
                    f_out_loss = f_loss / float(num_batches)
                    tqdm.write(f'[VAL] Epoch {self.cur_epoch:03d}, Batch {i:03d}: loss: {out_loss:3.3f}, '
                               f'F: {(f_out_loss.item()):3.3f}')

            out_loss = loss_epoch / float(num_batches)
            f_out_loss = f_loss / float(num_batches)
            tqdm.write(f'[VAL Total] Epoch {self.cur_epoch:03d}, Batch {i:03d}: loss: {out_loss:3.3f}, '
                       f'F: {(f_out_loss.item()):3.3f}')

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
        torch.save(encoder.state_dict(), os.path.join(args.logdir, 'encoder.pth'))
        for i, m in enumerate(mesh_updates):
            torch.save(m.state_dict(), os.path.join(args.logdir, 'mesh_update_{}.pth'.format(i)))
        torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))
        # Log other data corresponding to the recent model
        with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
            f.write(json.dumps(log_table))

        tqdm.write('====== Saved recent model ======>')

        if save_best:
            torch.save(encoder.state_dict(), os.path.join(args.logdir, 'best_encoder.pth'))
            for i, m in enumerate(mesh_updates):
                torch.save(m.state_dict(), os.path.join(args.logdir, 'best_mesh_update_{}.pth'.format(i)))
            torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
            tqdm.write('====== Overwrote best model ======>')

trainer = Engine()

for epoch in range(args.epochs):
    trainer.train()
    if epoch % 4 == 0:
        trainer.validate()
        trainer.save()
