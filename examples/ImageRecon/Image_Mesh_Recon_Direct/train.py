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

from utils import preprocess, loss_lap
from architectures import Encoder

import kaolin as kal
"""
Commandline arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument('-expid', type=str, default='Direct', help='Unique experiment identifier.')
parser.add_argument('-device', type=str, default='cuda', help='Device to use')
parser.add_argument('-categories', type=str, nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('-epochs', type=int, default=50, help='Number of train epochs.')
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
points_set = kal.datasets.ShapeNet_Points(
    root='../../datasets/', cache_dir='cache/', categories=args.categories, train=True, split=.7, num_points=3000)
images_set = kal.datasets.ShapeNet_Images(
    root='../../datasets/', categories=args.categories, train=True, split=.7, views=23, transform=preprocess)
train_set = kal.datasets.ShapeNet_Combination([points_set, images_set], root='../../datasets/')

dataloader_train = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, num_workers=8)


points_set_valid = kal.datasets.ShapeNet_Points(
    root='../../datasets/', cache_dir='cache/', categories=args.categories, train=False, split=.7, num_points=5000)
images_set_valid = kal.datasets.ShapeNet_Images(root='../../datasets/', categories=args.categories,
                                                train=False, split=.7, views=1, transform=preprocess)
dataloader_val = DataLoader(valid_set, batch_size=args.batchsize, shuffle=False, num_workers=8)


"""
Model settings
"""
mesh = kal.rep.TriangleMesh.from_obj('386.obj')
if args.device == "cuda":
    mesh.cuda()
initial_verts = mesh.vertices.clone()


model = Encoder(4, 5, args.batchsize, 137, mesh.vertices.shape[0]).to(args.device)

loss_fn = kal.metrics.point.chamfer_distance
loss_edge = kal.metrics.mesh.edge_length

optimizer = optim.Adam(model.parameters(), lr=args.lr)


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

    def __init__(self, cur_epoch=0, print_every=1, validate_every=1):
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
            tgt_points = data['points'].to(args.device)
            inp_images = data['imgs'].to(args.device)

            # inference
            delta_verts = model(inp_images)

            # losses
            surf_loss = 0.
            edge_loss = 0.
            lap_loss = 0.
            for deltas, tgt in zip(delta_verts, tgt_points):
                mesh.vertices = deltas + initial_verts
                pred_points, _ = mesh.sample(3000)
                surf_loss += 3000 * loss_fn(pred_points, tgt) / float(args.batchsize)
                edge_loss += 300 * loss_edge(mesh) / float(args.batchsize)
                lap_loss += 150 * loss_lap(mesh, deltas)

            loss = surf_loss + edge_loss + lap_loss
            loss.backward()
            loss_epoch += float(surf_loss.item())

            # logging
            num_batches += 1
            if i % args.print_every == 0:
                tqdm.write(
                    f'[TRAIN] Epoch {self.cur_epoch:03d}, Batch {i:03d}: Loss: {float(surf_loss.item()):3.3f}, Edge: {float(edge_loss.item()):3.3f}, lap: {float(lap_loss.item()):3.3f}')
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
                tgt_points = data['points'].to(args.device)
                inp_images = data['imgs'].to(args.device)

                # inference
                delta_verts = model(inp_images)

                # losses
                loss = 0.
                for deltas, tgt in zip(delta_verts, tgt_points):
                    mesh.vertices = deltas + initial_verts
                    pred_points, _ = mesh.sample(3000)

                    loss += 3000 * loss_fn(pred_points, tgt) / float(args.batchsize)
                loss_epoch += loss.item()

                # logging
                num_batches += 1
                if i % args.print_every == 0:
                    out_loss = loss_epoch / float(num_batches)
                    tqdm.write(f'[VAL] Epoch {self.cur_epoch:03d}, Batch {i:03d}: loss: {out_loss}')

            out_loss = loss_epoch / float(num_batches)
            tqdm.write(f'[VAL Total] Epoch {self.cur_epoch:03d}, Batch {i:03d}: loss: {out_loss}')

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
    if epoch % 4 == 0:
        trainer.validate()
        trainer.save()
