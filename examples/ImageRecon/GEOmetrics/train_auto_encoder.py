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
from tqdm import tqdm
import random

import kaolin as kal 
from kaolin.datasets import shapenet
from kaolin.models.GEOMetrics import VoxelDecoder
from kaolin.models.MeshEncoder import MeshEncoder


parser = argparse.ArgumentParser()
parser.add_argument('--shapenet-root', type=str, help='Root directory of the ShapeNet dataset.')
parser.add_argument('--cache-dir', type=str, default='cache', help='Path to write intermediate representation to.')
parser.add_argument('--expid', type=str, default='GEOmetrics_1', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--categories', type=str, nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('--epochs', type=int, default=100, help='Number of train epochs.')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--batch-size', type=int, default=25, help='batch size')
parser.add_argument('--val-every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--print-every', type=int, default=20, help='Print frequency (batches).')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('--resume', choices=['best', 'recent'], default=None,
                    help='Choose which weights to resume training from (none to start from random initialization.)')
args = parser.parse_args()


# Setup Datasets - Training
mesh_set = shapenet.ShapeNet_Surface_Meshes(root=args.shapenet_root, cache_dir=args.cache_dir, categories=args.categories,
                                            train=True, split=.7, mode='Tri', resolution=32)
voxel_set = shapenet.ShapeNet_Voxels(root=args.shapenet_root, cache_dir=args.cache_dir, categories=args.categories,
                                     train=True, split=.7, resolutions=[32])

train_set = shapenet.ShapeNet_Combination([mesh_set, voxel_set])


# Setup Datasets - Validation
mesh_set = shapenet.ShapeNet_Surface_Meshes(root=args.shapenet_root, cache_dir=args.cache_dir, categories=args.categories,
                                            train=False, split=.7, mode='Tri', resolution=32)
voxel_set = shapenet.ShapeNet_Voxels(root=args.shapenet_root, cache_dir=args.cache_dir, categories=args.categories,
                                     train=False, split=.7, resolutions=[32])
valid_set = shapenet.ShapeNet_Combination([mesh_set, voxel_set])


# Setup Models
encoder = MeshEncoder(30).to(args.device)
decoder = VoxelDecoder(30).to(args.device)

parameters = list(encoder.parameters()) + list(decoder.parameters()) 
optimizer = optim.Adam(parameters, lr=args.learning_rate)

loss_fn = torch.nn.MSELoss()


# Create log directory, if it doesn't already exist
logdir = os.path.join(args.logdir, args.expid, 'AutoEncoder')
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
        - resume_name (str, optional): Prefix of weights from which to resume training.
            If None, no weights are loaded.
    """

    def __init__(self, print_every=1, resume_name=None):
        self.cur_epoch = 0
        self.train_loss = {}
        self.val_score = {}
        self.bestval = 0.
        self.print_every = print_every

        if resume_name:
            self.load(resume_name)

    def train(self):
        loss_epoch = 0.
        num_batches = 0
        encoder.train(), decoder.train()

        # Train loop
        for i in tqdm(range(len(train_set) // args.batch_size)): 
            tgt_voxels = []
            latent_encodings = []

            # Can't use a dataloader due to the adj matrix
            for j in range(args.batch_size):
                optimizer.zero_grad()

                # Data Creation
                selection = random.randint(0, len(train_set) - 1)
                tgt_voxels.append(train_set[selection]['data']['32'].to(args.device).unsqueeze(0))
                inp_verts = train_set[selection]['data']['vertices'].to(args.device)
                inp_adj = train_set[selection]['data']['adj'].to(args.device)

                # Inverence
                latent_encodings.append(encoder(inp_verts, inp_adj).unsqueeze(0))

            tgt_voxels = torch.cat(tgt_voxels)
            latent_encodings = torch.cat(latent_encodings)
            pred_voxels = decoder(latent_encodings)

            # Loss
            loss = loss_fn(pred_voxels, tgt_voxels)
            loss.backward()
            loss_epoch += float(loss.item())

            # Logging
            iou = kal.metrics.voxel.iou(pred_voxels.contiguous(), tgt_voxels)
            num_batches += 1
            if i % args.print_every == 0:
                tqdm.write(f'[TRAIN] Epoch {self.cur_epoch:03d}, Batch {i:03d}: Loss: {loss.item():.5f} '
                           f'IoU: {iou:.4f}')
            optimizer.step()
        loss_epoch = loss_epoch / num_batches
        self.train_loss[self.cur_epoch] = loss_epoch

    def validate(self):
        encoder.eval(), decoder.eval()
        with torch.no_grad():	
            num_batches = 0
            iou_epoch = 0.

            # Validation loop
            for i in tqdm(range(len(valid_set) // args.batch_size)): 
                tgt_voxels = []
                latent_encodings = []
                for j in range(args.batch_size):
                    optimizer.zero_grad()

                    # Data Creation
                    tgt_voxels.append(valid_set[i * args.batch_size + j]['data']['32'].to(args.device).unsqueeze(0))
                    inp_verts = valid_set[i * args.batch_size + j]['data']['vertices'].to(args.device)
                    inp_adj = valid_set[i * args.batch_size + j]['data']['adj'].to(args.device)

                    # Inference
                    latent_encodings.append(encoder(inp_verts, inp_adj).unsqueeze(0))

                tgt_voxels = torch.cat(tgt_voxels)
                latent_encodings = torch.cat(latent_encodings)
                pred_voxels = decoder(latent_encodings)

                # Loss
                iou = kal.metrics.voxel.iou(pred_voxels.contiguous(), tgt_voxels)
                iou_epoch += iou

                # logging
                num_batches += 1
                if i % args.print_every == 0:
                    out_iou = iou_epoch.item() / float(num_batches)
                    tqdm.write(f'[VAL] Epoch {self.cur_epoch:03d}, Batch {i:03d}: IoU: {out_iou:.4f}')

            out_iou = iou_epoch.item() / float(num_batches)
            tqdm.write(f'[VAL Total] Epoch {self.cur_epoch:03d}, Batch {i:03d}: IoU: {out_iou:.4f}')
            self.val_score[self.cur_epoch] = out_iou

    def step(self):
        self.cur_epoch += 1

    def load(self, resume_name):
        checkpoint = torch.load(os.path.join(logdir, f'{resume_name}.ckpt'))
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        # Read data corresponding to the loaded model
        with open(os.path.join(logdir, f'{resume_name}.log'), 'r') as f:
            run_data = json.load(f)
        self.cur_epoch = run_data['epoch']
        self.bestval = run_data['bestval']
        self.train_loss = run_data['train_loss']
        self.val_score = run_data['val_score']

        # step to next epoch
        self.step()

    def save(self):
        self._save_checkpoint('recent')

        if self.val_score.get(self.cur_epoch, 0) > self.bestval:
            self.bestval = self.val_score.get(self.cur_epoch, 0)
            self._save_checkpoint('best')

    def _save_checkpoint(self, name):
        # Save Checkpoint
        checkpoint = {
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(logdir, f'{name}.ckpt'))

        # Log other data corresponding to the recent model
        log_table = {
            'epoch': self.cur_epoch,
            'bestval': self.bestval,
            'train_loss': self.train_loss,
            'val_score': self.val_score,
        }
        with open(os.path.join(logdir, f'{name}.log'), 'w') as f:
            f.write(json.dumps(log_table, separators=(',', ':'), indent=4))
        tqdm.write(f'====== Saved {name} checkpoint ======>')

trainer = Engine(print_every=args.print_every, resume_name=args.resume)

for epoch in range(trainer.cur_epoch, args.epochs): 
    trainer.train()
    if epoch % args.val_every == 0: 
        trainer.validate()
    trainer.save()
    trainer.step()
