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
import json

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import kaolin as kal
from kaolin.datasets import shapenet
from kaolin.models.OccupancyNetwork import OccupancyNetwork

from utils import preprocess


parser = argparse.ArgumentParser()
parser.add_argument('--shapenet-root', type=str, help='Root directory of the ShapeNet dataset.')
parser.add_argument('--shapenet-images-root', type=str, help='Root directory of the ShapeNet Rendering dataset.')
parser.add_argument('--cache-dir', type=str, default='cache', help='Path to write intermediate representation to.')
parser.add_argument('--expid', type=str, default='OccNet_1', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--categories', type=str, nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('--epochs', type=int, default=50000, help='Number of train epochs.')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--batch-size', type=int, default=32, help='batch size')
parser.add_argument('--val-every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--print-every', type=int, default=5, help='Print frequency (batches).')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('--resume', choices=['best', 'recent'], default=None,
                    help='Choose which weights to resume training from (None to start from random initialization.)')
args = parser.parse_args()



# Common Params
common_params = {
    'root': args.shapenet_root,
    'categories': args.categories,
    'cache_dir': args.cache_dir,
    'split': 0.7,
}

img_params = {
    'root': args.shapenet_images_root,
    'categories': args.categories,
    'split': 0.7,
}

THRESHOLD = 0.2
PADDING = 0.1
UPSAMPLING_STEPS = 2
RESOLUTION_0 = 64


# Dataset
common_params['train'], img_params['train'] = True, True
sdf_set = shapenet.ShapeNet_SDF_Points(**common_params, resolution=150, num_points=15000, occ=True)
images_set = shapenet.ShapeNet_Images(**img_params, views=23, transform=preprocess)
train_set = shapenet.ShapeNet_Combination([sdf_set, images_set])
dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)

common_params['train'], img_params['train'] = False, False
sdf_set = shapenet.ShapeNet_SDF_Points(**common_params, resolution=150, num_points=100000, occ=True)
images_set = shapenet.ShapeNet_Images(**img_params, views=1, transform=preprocess)
valid_set = shapenet.ShapeNet_Combination([sdf_set, images_set])
dataloader_val = DataLoader(valid_set, batch_size=5, shuffle=False, num_workers=8)


# Model
model = OccupancyNetwork(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


# Create log directory, if it doesn't already exist
logdir = os.path.join(args.logdir, args.expid)
if not os.path.isdir(logdir):
    os.makedirs(logdir)
    print('Created dir:', logdir)


# Log all commandline args
with open(os.path.join(logdir, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)


# Tensorboard
train_writer = SummaryWriter(log_dir=os.path.join(logdir, 'train'))
val_writer = SummaryWriter(log_dir=os.path.join(logdir, 'val'))


class Engine(object):
    """ Engine that runs training and inference.
    Args
        - print_every (int): How frequently (# batches) to print loss.
        - resume_name (str): Prefix of weights from which to resume training. If None,
            no weights are loaded.
    """

    def __init__(self, print_every, resume_name=None):
        self.cur_epoch = 0
        self.train_loss = {}
        self.val_score = {}
        self.bestval = 0
        self.print_every = print_every

        if resume_name:
            self.load(resume_name)

    def train(self):
        loss_epoch = 0.
        iou_epoch = 0.
        num_batches = 0
        model.train()

        # Train loop
        for i, sample in enumerate(tqdm(dataloader_train), 0):
            optimizer.zero_grad()

            # Data
            data = sample['data']
            imgs = data['images'][:, :3].to(args.device)
            points = data['occ_points'].to(args.device)
            gt_occ = data['occ_values'].to(args.device)

            # Infer
            encoding = model.encode_inputs(imgs)

            # Cross-entropy Loss
            decoded = model.decode(points, torch.zeros(args.batch_size, 0), encoding)
            logits = decoded.logits

            loss = F.binary_cross_entropy_with_logits(logits, gt_occ.float(), reduction='none').sum(dim=-1).mean()
            loss.backward()
            loss_epoch += loss.item() / points.size(0)

            # IoU
            pred_occ = decoded.probs
            iou_batch = kal.metrics.point.iou(pred_occ, gt_occ, thresh=THRESHOLD)
            iou_epoch += iou_batch.item()

            num_batches += 1
            if i % self.print_every == 0:
                message = f'[TRAIN] Epoch {self.cur_epoch:03d}, Batch {i:03d}:, Loss: {loss.item():4.3f}, IoU: {iou_batch.item():0.3f}'
                tqdm.write(message)
            optimizer.step()

        loss_epoch = loss_epoch / num_batches
        iou_epoch = iou_epoch / num_batches
        train_writer.add_scalar('Loss', loss_epoch, self.cur_epoch)
        train_writer.add_scalar('IoU', iou_epoch, self.cur_epoch)
        self.train_loss[self.cur_epoch] = loss_epoch

    def validate(self):
        model.eval()
        with torch.no_grad():	
            num_batches = 0
            iou_epoch = 0.

            # Validation loop
            for i, sample in enumerate(tqdm(dataloader_val), 0):
                optimizer.zero_grad()

                # Data
                data = sample['data']
                imgs = data['images'][:, :3].to(args.device)
                points = data['occ_points'].to(args.device)
                gt_occ = data['occ_values'].to(args.device)
                val_batch_size = points.size(0)

                # Infer
                encoding = model.encode_inputs(imgs)
                pred_occ = model.decode(points, torch.zeros(args.batch_size, 0), encoding).probs

                # IoU calculation
                iou_batch = kal.metrics.point.iou(pred_occ, gt_occ, thresh=THRESHOLD)
                iou_epoch += iou_batch.item()

                num_batches += 1
                if i % args.print_every == 0:
                    tqdm.write(f'[ VAL ] Epoch {self.cur_epoch:03d}, Batch {i:03d}: IoU: {iou_batch.item():3.3f}')

            iou_epoch = iou_epoch / num_batches
            tqdm.write(f'[VAL Total] Epoch {self.cur_epoch:03d}, Batch {i:03d}: IoU: {iou_epoch:3.3f}')
            val_writer.add_scalar('IoU', iou_epoch, self.cur_epoch)

            self.val_score[self.cur_epoch] = iou_epoch

    def step(self):
        self.cur_epoch += 1

    def load(self, resume_name):
        checkpoint = torch.load(os.path.join(logdir, f'{resume_name}.ckpt'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        # Read data corresponding to the loaded model
        with open(os.path.join(logdir, f'{resume_name}.log'), 'r') as f:
            log = json.load(f)
        self.cur_epoch = log['epoch']
        self.bestval = log['bestval']
        self.train_loss = log['train_loss']
        self.val_score = log['val_score']

        # step to next epoch
        self.step()

    def save(self):
        # Save the recent model/optimizer states
        self._save_checkpoint('recent')

        # Save the current model if it outperforms previous ones
        if self.val_score.get(self.cur_epoch, 0) > self.bestval:
            self.bestval = self.val_score[self.cur_epoch]
            self._save_checkpoint('best')

    def _save_checkpoint(self, name):
        # Save Checkpoint
        checkpoint = {
            'model': model.state_dict(),
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


trainer = Engine(args.print_every, args.resume)

for epoch in range(trainer.cur_epoch, args.epochs):
    trainer.train()
    if epoch % args.val_every == 0:
        trainer.validate()
    trainer.save()
    trainer.step()
