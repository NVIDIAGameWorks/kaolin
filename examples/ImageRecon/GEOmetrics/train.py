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
from utils import setup_meshes, split_meshes, reset_meshes
from utils import loss_surf, loss_edge, loss_lap , collate_fn
from architectures import VGG as Encoder, G_Res_Net, MeshEncoder

import kaolin as kal
from kaolin.datasets import shapenet


parser = argparse.ArgumentParser()
parser.add_argument('--shapenet-root', type=str, help='Root directory of the ShapeNet dataset.')
parser.add_argument('--shapenet-rendering-root', type=str, help='Root directory of the ShapeNet Rendering dataset.')
parser.add_argument('--cache-dir', type=str, default=None, help='Path to write intermediate representation to.')
parser.add_argument('--expid', type=str, default='Direct', help='Unique experiment identifier. If using latent-loss, '
    'must be the same expid used during auto-encoder training.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--categories', type=str, nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('--epochs', type=int, default=50, help='Number of train epochs.')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--val-every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--batch-size', type=int, default=5, help='batch size')
parser.add_argument('--print-every', type=int, default=20, help='Print frequency (batches).')
parser.add_argument('--latent-loss', action='store_true', help='indicates latent loss should be used')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('--resume', choices=['none', 'best', 'recent'], default='none',
                    help='Choose which weights to resume training from (none to start from random initialization.)')
args = parser.parse_args()


# Setup Dataset
points_set = shapenet.ShapeNet_Points(root=args.shapenet_root, cache_dir=args.cache_dir, categories=args.categories,
                                      train=True, split=.7, num_points=3000)
images_set = shapenet.ShapeNet_Images(root=args.shapenet_rendering_root, categories=args.categories,
                                      train=True, split=.7, views=23, transform=preprocess)
if args.latent_loss:
    mesh_set = shapenet.ShapeNet_Surface_Meshes(root=args.shapenet_root, cache_dir=args.cache_dir, categories=args.categories,
                                                resolution=32, train=True, split=.7, mode='Tri')
    train_set = shapenet.ShapeNet_Combination([points_set, images_set, mesh_set])
    dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn,
                                  num_workers=8)
else: 
    train_set = shapenet.ShapeNet_Combination([points_set, images_set])
    dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                  num_workers=8)


points_set_valid = shapenet.ShapeNet_Points(root=args.shapenet_root, cache_dir=args.cache_dir, categories=args.categories,
                                            train=False, split=.7, num_points=10000)
images_set_valid = shapenet.ShapeNet_Images(root=args.shapenet_rendering_root, categories=args.categories,
                                            train=False, split=.7, views=1, transform=preprocess)
valid_set = shapenet.ShapeNet_Combination([points_set_valid, images_set_valid])

dataloader_val = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=8)


# Setup models
meshes = setup_meshes(filename='meshes/386.obj', device=args.device)

encoders = [Encoder().to(args.device) for i in range(3)]
mesh_update_kernels = [963, 1091, 1091] 
mesh_updates = [G_Res_Net(mesh_update_kernels[i], hidden=128, output_features=3).to(args.device) for i in range(3)]
if args.latent_loss:
    mesh_encoder = MeshEncoder(30).to(args.device)
    mesh_encoder.load_state_dict(torch.load(os.path.join(args.logdir, args.expid, 'AutoEncoder/best_encoder.pth')))

parameters = []

for i in range(3):
    parameters += list(encoders[i].parameters()) 
    parameters += list(mesh_updates[i].parameters())
optimizer = optim.Adam(parameters, lr=args.learning_rate)

encoding_dims = [56, 28, 14, 7]


# Create log directory, if it doesn't already exist
logdir = os.path.join(args.logdir, args.expid)
if not os.path.isdir(logdir):
    os.makedirs(logdir)
    print('Created dir:', logdir)

# Log all commandline args
with open(os.path.join(logdir, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

class Engine(object):
    """Engine that runs training and inference.
    Args
        - print_every (int): How frequently (# batches) to print loss.
        - resume_name (str): Prefix of weights from which to resume training. If 'none',
            no weights are loaded.
    """

    def __init__(self, print_every, resume_name='none'):
        self.cur_epoch = 0
        self.train_loss = []
        self.val_score = []
        self.bestval = 0
        self.print_every = print_every

        if resume_name != 'none':
            self.load(resume_name)

    def train(self):
        loss_epoch = 0.
        num_batches = 0
        [e.train() for e in encoders], [m.train() for m in mesh_updates]

        # Train loop
        for i, sample in enumerate(tqdm(dataloader_train), 0):
            data = sample['data']
            optimizer.zero_grad()

            # Data Creation
            tgt_points = data['points'].to(args.device)
            inp_images = data['images'].to(args.device)
            cam_mat = data['params']['cam_mat'].to(args.device)
            cam_pos = data['params']['cam_pos'].to(args.device)
            if (tgt_points.shape[0] != args.batch_size) and (inp_images.shape[0] != args.batch_size) \
                    and (cam_mat.shape[0] != args.batch_size) and (cam_pos.shape[0] != args.batch_size): 
                continue
            surf_loss, edge_loss, lap_loss, loss, f_score = 0, 0, 0, 0, 0

            # Inference
            img_features = [e(inp_images) for e in encoders]
            for bn in range(args.batch_size):
                reset_meshes(meshes)

                # Layer_1
                pool_indices = get_pooling_index(meshes['init'][0].vertices, cam_mat[bn], cam_pos[bn], encoding_dims)
                projected_image_features = pooling(img_features[0], pool_indices, bn)
                full_vert_features = torch.cat((meshes['init'][0].vertices, projected_image_features), dim=1)

                delta, future_features = mesh_updates[0](full_vert_features, meshes['adjs'][0])
                meshes['update'][0].vertices = (meshes['init'][0].vertices + delta.clone())
                future_features = split_meshes(meshes, future_features, 0)			

                # Layer_2
                pool_indices = get_pooling_index(meshes['init'][1].vertices, cam_mat[bn], cam_pos[bn], encoding_dims)
                projected_image_features = pooling(img_features[1], pool_indices, bn)
                full_vert_features = torch.cat((meshes['init'][1].vertices, projected_image_features, future_features), dim=1)

                delta, future_features = mesh_updates[1](full_vert_features, meshes['adjs'][1])
                meshes['update'][1].vertices = (meshes['init'][1].vertices + delta.clone())
                future_features = split_meshes(meshes, future_features, 1)	

                # Layer_3
                pool_indices = get_pooling_index(meshes['init'][2].vertices, cam_mat[bn], cam_pos[bn], encoding_dims)
                projected_image_features = pooling(img_features[2], pool_indices, bn)
                full_vert_features = torch.cat((meshes['init'][2].vertices, projected_image_features, future_features), dim=1)
                delta, future_features = mesh_updates[2](full_vert_features, meshes['adjs'][2])
                meshes['update'][2].vertices = (meshes['init'][2].vertices + delta.clone())

                if args.latent_loss:
                    inds = data['adj']['indices'][bn]
                    vals = data['adj']['values'][bn]
                    gt_verts = data['vertices'][bn].to(args.device)
                    vert_len = gt_verts.shape[0]
                    gt_adj = torch.sparse.FloatTensor(inds, vals, torch.Size([vert_len, vert_len])).to(args.device)

                    predicted_latent = mesh_encoder(meshes['update'][2].vertices, meshes['adjs'][2])  
                    gt_latent = mesh_encoder(gt_verts, gt_adj)  
                    latent_loss = torch.mean(torch.abs(predicted_latent - gt_latent)) * .2


                # Losses
                surf_loss += (6000 * loss_surf(meshes, tgt_points[bn]) / float(args.batch_size))
                edge_loss += (300 * .6 * loss_edge(meshes) / float(args.batch_size))
                lap_loss += (1500 * loss_lap(meshes) / float(args.batch_size))

                # F-Score
                f_score += kal.metrics.point.f_score(.57 * meshes['update'][2].sample(2466)[0], .57 * tgt_points[bn],
                                                     extend=False) / float(args.batch_size)


                loss = surf_loss + edge_loss + lap_loss
                if args.latent_loss: 
                    loss += latent_loss
            loss.backward()
            loss_epoch += float(loss.item())

            # logging
            num_batches += 1
            if i % args.print_every == 0:
                message = f'[TRAIN] Epoch {self.cur_epoch:03d}, Batch {i:03d}:, Loss: {loss_epoch:4.3f}, '
                message += f'Surf: {(surf_loss.item()):3.3f}, Lap: {(lap_loss.item()):3.3f}, '
                message += f'Edge: {(edge_loss.item()):3.3f}, F-score: {(f_score.item()):3.3f}'
                if args.latent_loss: 
                    message = message + f', Lat: {(latent_loss.item()):3.3f}'
                tqdm.write(message)

            optimizer.step()

        loss_epoch = loss_epoch / num_batches
        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1

    def validate(self):
        [e.eval() for e in encoders], [m.eval() for m in mesh_updates]
        with torch.no_grad():	
            num_batches = 0
            loss_epoch = 0.
            f_score = 0 
            # Validation loop
            for i, sample in enumerate(tqdm(dataloader_val), 0):
                data = sample['data']
                optimizer.zero_grad()

                # Data Creation
                tgt_points = data['points'].to(args.device)
                inp_images = data['images'].to(args.device)
                cam_mat = data['params']['cam_mat'].to(args.device)
                cam_pos = data['params']['cam_pos'].to(args.device)
                if (tgt_points.shape[0] != args.batch_size) and (inp_images.shape[0] != args.batch_size)  \
                        and (cam_mat.shape[0] != args.batch_size) and (cam_pos.shape[0] != args.batch_size): 
                    continue
                surf_loss = 0

                # Inference
                img_features = [e(inp_images) for e in encoders]
                for bn in range(args.batch_size):
                    reset_meshes(meshes)

                    # Layer_1
                    pool_indices = get_pooling_index(meshes['init'][0].vertices, cam_mat[bn], cam_pos[bn], encoding_dims)
                    projected_image_features = pooling(img_features[0], pool_indices, bn)
                    full_vert_features = torch.cat((meshes['init'][0].vertices, projected_image_features), dim=1)

                    delta, future_features = mesh_updates[0](full_vert_features, meshes['adjs'][0])
                    meshes['update'][0].vertices = (meshes['init'][0].vertices + delta.clone())
                    future_features = split_meshes(meshes, future_features, 0)			

                    # Layer_2
                    pool_indices = get_pooling_index(meshes['init'][1].vertices, cam_mat[bn], cam_pos[bn], encoding_dims)
                    projected_image_features = pooling(img_features[1], pool_indices, bn)
                    full_vert_features = torch.cat((meshes['init'][1].vertices, projected_image_features, future_features), dim=1)

                    delta, future_features = mesh_updates[1](full_vert_features, meshes['adjs'][1])
                    meshes['update'][1].vertices = (meshes['init'][1].vertices + delta.clone())
                    future_features = split_meshes(meshes, future_features, 1)	

                    # Layer_3
                    pool_indices = get_pooling_index(meshes['init'][2].vertices, cam_mat[bn], cam_pos[bn], encoding_dims)
                    projected_image_features = pooling(img_features[2], pool_indices, bn)
                    full_vert_features = torch.cat((meshes['init'][2].vertices, projected_image_features, future_features), dim=1)

                    delta, future_features = mesh_updates[2](full_vert_features, meshes['adjs'][2])
                    meshes['update'][2].vertices = (meshes['init'][2].vertices + delta.clone())
                    pred_points, _ = meshes['update'][2].sample(10000)

                    # Losses
                    surf_loss = 3000 * kal.metrics.point.chamfer_distance(pred_points, tgt_points[bn])

                    # F-Score
                    f_score += (kal.metrics.point.f_score(.57 * meshes['update'][2].sample(2466)[0], .57 * tgt_points[bn],
                                                          extend=False).item() / float(args.batch_size))

                    loss_epoch += (surf_loss.item() / float(args.batch_size))

                    # logging
                num_batches += 1
                if i % args.print_every == 0:
                    out_loss = loss_epoch / float(num_batches)
                    out_f_score = f_score / float(num_batches)
                    tqdm.write(f'[VAL] Epoch {self.cur_epoch:03d}, Batch {i:03d}: loss: {out_loss:3.3f}, F-Score: {out_f_score:3.3f}')

            out_f_score = f_score / float(num_batches)
            out_loss = loss_epoch / float(num_batches)
            tqdm.write(f'[VAL Total] Epoch {self.cur_epoch:03d}, Batch {i:03d}: Loss: {out_loss:3.3f},  F-Score: {out_f_score:3.3f}')

            self.val_score.append(out_f_score)

    def load(self, resume_name):
        for i, e in enumerate(encoders):
            e.load_state_dict(torch.load(os.path.join(logdir, f'{resume_name}_encoder_{i}.pth')))
        for i, m in enumerate(mesh_updates):
            m.load_state_dict(torch.load(os.path.join(logdir, f'{resume_name}_mesh_update_{i}.pth')))

        optimizer.load_state_dict(torch.load(os.path.join(logdir, f'{resume_name}_optim.pth')))
        # Read data corresponding to the loaded model
        with open(os.path.join(logdir, f'{resume_name}.log'), 'r') as f:
            run_data = json.load(f)
        self.cur_epoch = run_data['epoch']
        self.bestval = run_data['bestval']

    def save(self):
        save_best = False
        if self.val_score[-1] >= self.bestval:
            self.bestval = self.val_score[-1]
            save_best = True

        # Create a dictionary of all data to save
        log_table = {
            'epoch': self.cur_epoch,
            'bestval': self.bestval,
            'train_loss': self.train_loss,
            'val_score': self.val_score,
        }

        # Save the recent model/optimizer states
        for i, e in enumerate(encoders):
            torch.save(e.state_dict(), os.path.join(logdir, f'recent_encoder_{i}.pth'))
        for i, m in enumerate(mesh_updates):
            torch.save(m.state_dict(), os.path.join(logdir, f'recent_mesh_update_{i}.pth'))
        torch.save(optimizer.state_dict(), os.path.join(logdir, 'recent_optim.pth'))
        # Log other data corresponding to the recent model
        with open(os.path.join(logdir, 'recent.log'), 'w') as f:
            f.write(json.dumps(log_table))

        tqdm.write('====== Saved recent model ======>')

        if save_best:
            for i, e in enumerate(encoders):
                torch.save(e.state_dict(), os.path.join(logdir, f'best_encoder_{i}.pth'))
            for i, m in enumerate(mesh_updates):
                torch.save(m.state_dict(), os.path.join(logdir, f'best_mesh_update_{i}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'best_optim.pth'))
            # Log other data corresponding to the best model
            with open(os.path.join(logdir, 'best.log'), 'w') as f:
                f.write(json.dumps(log_table))
            tqdm.write('====== Overwrote best model ======>')

trainer = Engine(print_every=args.print_every, resume_name=args.resume)

for epoch in range(args.epochs): 
    trainer.train()
    if epoch % args.val_every == 0:
        trainer.validate()
    trainer.save()
