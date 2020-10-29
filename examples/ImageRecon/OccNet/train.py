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
from utils import preprocess
from tqdm import tqdm
import torch.nn.functional as F

import torch.distributions as dist
from architectures import OccupancyNetwork

import kaolin as kal
"""
Commandline arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument('-expid', type=str, default='Direct', help='Unique experiment identifier.')
parser.add_argument('-device', type=str, default='cuda', help='Device to use')
parser.add_argument('-categories', type=str, nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('-epochs', type=int, default=500, help='Number of train epochs.')
parser.add_argument('-lr', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('-val-every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('-batch_size', type=int, default=64, help='batch size')
parser.add_argument('-print-every', type=int, default=3, help='Print frequency (batches).')
parser.add_argument('-logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('-save-model', action='store_true', help='Saves the model and a snapshot \
	of the optimizer state.')
args = parser.parse_args()


"""
Dataset
"""
sdf_set = kal.datasets.ShapeNet_SDF_Points(root='../../datasets/', cache_dir='cache/', categories=args.categories,
                                           train=True, split=.7, num_points=2024, occ=True)
images_set = kal.datasets.ShapeNet_Images(root='../../datasets/', categories=args.categories,
                                          train=True, split=.7, views=23, transform=preprocess)
train_set = kal.datasets.ShapeNet_Combination([sdf_set, images_set], root='../../datasets/')
dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)

sdf_set = kal.datasets.ShapeNet_SDF_Points(root='../../datasets/', cache_dir='cache', categories=args.categories,
                                           train=False, split=.95, num_points=100000, occ=True)
images_set = kal.datasets.ShapeNet_Images(root='../../datasets/', categories=args.categories,
                                          train=False, split=.7, views=1, transform=preprocess)
valid_set = kal.datasets.ShapeNet_Combination([sdf_set, images_set], root='../../datasets/')

dataloader_val = DataLoader(valid_set, batch_size=5, shuffle=False, num_workers=8)


"""
Model settings
"""


model = OccupancyNetwork(args.device)
parameters = list(model.encoder.parameters()) + list(model.decoder.parameters())
optimizer = optim.Adam(parameters, lr=args.lr)


"""
Initial settings
"""


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
        self.bestval = 0

    def train(self):
        loss_epoch = 0.
        num_batches = 0
        model.encoder.train()
        model.decoder.train()

        # Train loop
        for i, data in enumerate(tqdm(dataloader_train), 0):
            optimizer.zero_grad()

            ###############################
            ####### data creation #########
            ###############################
            imgs = data['imgs'][:, :3].to(args.device)
            points = data['occ_points'].to(args.device)
            gt_occ = data['occ_values'].to(args.device)

            ###############################
            ########## inference ##########
            ###############################
            encoding = model.encode_inputs(imgs)
            pred_occ = model.decode(points, torch.zeros(args.batch_size, 0), encoding).logits

            ###############################
            ########## losses #############
            ###############################
            loss = F.binary_cross_entropy_with_logits(pred_occ, gt_occ).mean()
            loss.backward()
            loss_epoch += float(loss.item())

            num_batches += 1
            if i % args.print_every == 0:
                message = f'[TRAIN] Epoch {self.cur_epoch:03d}, Batch {i:03d}:, Loss: {(loss.item()):4.3f}'
                tqdm.write(message)
            optimizer.step()

        loss_epoch = loss_epoch / num_batches
        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1

    def validate_iou(self):
        model.encoder.eval()
        model.decoder.eval()
        with torch.no_grad():
            num_batches = 0
            iou_epoch = 0.

            # Validation loop
            for i, data in enumerate(tqdm(dataloader_val), 0):
                optimizer.zero_grad()

                ###############################
                ####### data creation #########
                ###############################
                imgs = data['imgs'][:, :3].to(args.device)
                points = data['occ_points'].to(args.device)
                gt_occ = data['occ_values'].to(args.device)

                ###############################
                ########## inference ##########
                ###############################
                encoding = model.encode_inputs(imgs)
                pred_occ = model.decode(points, torch.zeros(args.batch_size, 0), encoding).logits

                ###############################
                ########## losses #############
                ###############################

                for pt1, pt2 in zip(gt_occ, pred_occ):
                    iou_epoch += float((kal.metrics.point.iou(pt1, pt2, thresh=.2) / float(gt_occ.shape[0])).item())

                num_batches += 1
                if i % args.print_every == 0:
                    out_loss = iou_epoch / float(num_batches)
                    tqdm.write(f'[VAL IoU] Epoch {self.cur_epoch:03d}, Batch {i:03d}: iou: {out_loss:3.3f}')

            out_loss = iou_epoch / float(num_batches)
            tqdm.write(f'[VAL IoU Total] Epoch {self.cur_epoch:03d}, Batch {i:03d}: iou: {out_loss:3.3f}')

            self.val_loss.append(out_loss)

    def save(self):

        save_best = False
        if self.val_loss[-1] >= self.bestval:
            self.bestval = self.val_loss[-1]
            save_best = True

        # Create a dictionary of all data to save
        log_table = {
            'epoch': self.cur_epoch,
            'bestval': self.bestval,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_metrics': ['Chamfer'],
            'val_metrics': ['Chamfer'],
        }

        # Save the recent model/optimizer states

        torch.save(model.encoder.state_dict(), os.path.join(args.logdir, 'encoder.pth'))
        torch.save(model.decoder.state_dict(), os.path.join(args.logdir, 'decoder.pth'))
        torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'recent_optim.pth'))
        # Log other data corresponding to the recent model
        with open(os.path.join(args.logdir, 'recent.log'), 'w') as f:
            f.write(json.dumps(log_table))

        tqdm.write('====== Saved recent model ======>')

        if save_best:
            torch.save(model.encoder.state_dict(), os.path.join(args.logdir, 'best_encoder.pth'))
            torch.save(model.decoder.state_dict(), os.path.join(args.logdir, 'best_decoder.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.logdir, 'best_optim.pth'))
            tqdm.write('====== Overwrote best model ======>')


trainer = Engine()

for epoch in range(args.epochs):
    trainer.train()
    if epoch % 4 == 0:
        trainer.validate_iou()
        trainer.save()
