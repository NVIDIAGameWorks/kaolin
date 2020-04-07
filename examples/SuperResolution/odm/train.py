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


import argparse
import json
import os
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader


from kaolin.datasets import shapenet
from kaolin.conversions.voxelgridconversions import project_odms
from utils import up_sample, upsample_odm, to_occupancy_map, collate_fn
import kaolin as kal
from kaolin.models.VoxelSuperresODM import SuperresNetwork

from preprocessor import VoxelODMs

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['Direct', 'MVD'], default='MVD')
parser.add_argument('--shapenet-root', type=str, help='Path to shapenet data directory.')
parser.add_argument('--cache-dir', type=str, help='Path to data directory.')
parser.add_argument('--expid', type=str, default='ODM', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--categories', type=str, nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('--epochs', type=int, default=100, help='Number of train epochs.')
parser.add_argument('--batch-size', type=int, default=16, help='Batch size.')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Learning rate.')
parser.add_argument('--val-every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--print-every', type=int, default=20, help='Print frequency (batches).')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('--resume', choices=['best', 'recent'], default=None,
                    help='Choose which weights to resume training from (None to start from random initialization.)')
args = parser.parse_args()


# Data
preprocessing_params = {'cache_dir': args.cache_dir}
train_set = shapenet.ShapeNet(root=args.shapenet_root, categories=args.categories, train=True, split=0.7,
                              preprocessing_params=preprocessing_params, preprocessing_transform=VoxelODMs())

val_set = shapenet.ShapeNet(root=args.shapenet_root, categories=args.categories, train=False, split=0.7,
                            preprocessing_params=preprocessing_params, preprocessing_transform=VoxelODMs())

dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=8, collate_fn=collate_fn)
dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=8, collate_fn=collate_fn)


# Model
model = SuperresNetwork(128, 32).to(args.device)
loss_fn = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# Create log directory, if it doesn't already exist
logdir = os.path.join(args.logdir, f'{args.expid}_{args.mode.lower()}')
if not os.path.isdir(logdir):
    os.makedirs(logdir)
    print('Created dir:', logdir)

# Log all commandline args
with open(os.path.join(logdir, 'args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)


class Engine(object):
    """Engine that runs training and inference.
    Args
        model_name (str): Name of model being trained. ['Direct', 'MVD']
        cur_epoch (int): Current epoch.
        print_every (int): How frequently (# batches) to print loss.
        validate_every (int): How frequently (# epochs) to run validation.
    """

    def __init__(self, mode, print_every=1, resume_name=None):
        assert mode in ['Direct', 'MVD']
        self.mode = mode
        self.cur_epoch = 0
        self.train_loss = []
        self.val_score = []
        self.bestval = 0

        if resume_name:
            self.load(resume_name)

    def train(self):
        model.train()
        loss_epoch = 0.
        num_batches = 0

        # Train loop
        for i, sample in enumerate(tqdm(dataloader_train), 0):
            data = sample['data']

            pred_odms = self._get_pred(data)
            loss = self._get_loss(pred_odms, data)

            optimizer.zero_grad()
            loss.backward()
            loss_epoch += loss.item()

            # logging
            num_batches += 1
            if i % args.print_every == 0:
                tqdm.write(f'[TRAIN] Epoch {self.cur_epoch:03d}, Batch {i:03d}: Loss: {float(loss.item())}')

            optimizer.step()

        loss_epoch = loss_epoch / num_batches
        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1

    def validate(self):
        model.eval()
        with torch.no_grad():	
            iou_epoch = 0.
            iou_NN_epoch = 0.
            num_batches = 0
            loss_epoch = 0.

            # Validation loop
            for i, sample in enumerate(tqdm(dataloader_val), 0):
                data = sample['data']
                pred_odms = self._get_pred(data)

                loss = self._get_loss(pred_odms, data)

                loss_epoch += float(loss.item())

                iou_NN, iou = self._calculate_iou(pred_odms, data)
                iou_NN_epoch += iou_NN
                iou_epoch += iou

                # logging
                num_batches += 1
                if i % args.print_every == 0:
                    out_iou = iou_epoch.item() / float(num_batches)
                    out_iou_NN = iou_NN_epoch.item() / float(num_batches)
                    tqdm.write(f'[VAL] Epoch {self.cur_epoch:03d}, Batch {i:03d}: IoU: {out_iou}, Iou Base: {out_iou_NN}')

            out_iou = iou_epoch.item() / float(num_batches)
            out_iou_NN = iou_NN_epoch.item() / float(num_batches)
            tqdm.write(f'[VAL Total] Epoch {self.cur_epoch:03d}, Batch {i:03d}: IoU: {out_iou}, Iou Base: {out_iou_NN}')

            loss_epoch = loss_epoch / num_batches
            self.val_score.append(out_iou)

    def load(self, resume_name):
        model.load_state_dict(torch.load(os.path.join(logdir, f'{resume_name}.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(logdir, f'{resume_name}_optim.pth')))
        with open(os.path.join(logdir, 'recent.log'), 'r') as f:
            log = json.load(f)
        self.cur_epoch = log['epoch']
        self.bestval = log['bestval']
        self.train_loss = log['train_loss']
        self.val_score = log['val_score']

    def save(self):
        save_best = False
        if self.val_score[-1] > self.bestval:
            self.bestval = self.val_score[-1]
            save_best = True

        # Create a dictionary of all data to save
        log_table = {
            'epoch': self.cur_epoch,
            'bestval': self.bestval,
            'train_loss': self.train_loss,
            'val_score': self.val_score,
            'train_metrics': ['NLLLoss', 'iou'],
            'val_metrics': ['NLLLoss', 'iou', 'iou_NN'],
        }

        torch.save(model.state_dict(), os.path.join(logdir, 'recent.pth'))
        torch.save(optimizer.state_dict(), os.path.join(logdir, 'recent_optim.pth'))
        # Log other data corresponding to the recent model
        with open(os.path.join(logdir, 'recent.log'), 'w') as f:
            f.write(json.dumps(log_table))
        tqdm.write('====== Saved recent model ======>')

        if save_best:
            torch.save(model.state_dict(), os.path.join(logdir, 'best.pth'))
            torch.save(optimizer.state_dict(), os.path.join(logdir, 'best_optim.pth'))
            # Log other data corresponding to the recent model
            with open(os.path.join(logdir, 'best.log'), 'w') as f:
                f.write(json.dumps(log_table))
            tqdm.write('====== Overwrote best model ======>')

    def _get_pred(self, data):
        pred_fns = {
            'Direct': self._get_pred_direct,
            'MVD': self._get_pred_mvd,
        }
        return pred_fns[self.mode](data)

    @staticmethod
    def _get_pred_direct(data):
        inp_odms = data[32]['odms'].to(args.device)
        return model(inp_odms)

    @staticmethod
    def _get_pred_mvd(data):
        inp_odms = data[32]['odms'].to(args.device)
        initial_odms = upsample_odm(inp_odms) * 4
        distance = 128 - initial_odms
        pred_odms_update = model(inp_odms) * distance
        return initial_odms + pred_odms_update

    def _get_loss(self, pred, data):
        tgt_odms = data[128]['odms'].to(args.device)
        tgt = to_occupancy_map(tgt_odms) if self.mode == 'Direct' else tgt_odms
        return loss_fn(pred, tgt)

    def _calculate_iou(self, pred_odms, data):
        if self.mode == 'Direct':
            pred_odms[pred_odms > .3] = pred_odms.shape[-1]
            pred_odms[pred_odms <= .7] = 0
        elif self.mode == 'MVD':
            pred_odms = pred_odms.int()
        else:
            raise ValueError

        tgt_voxels = data[128]['voxels'].to(args.device)
        inp_voxels = data[32]['voxels'].to(args.device)
        NN_pred = up_sample(inp_voxels)
        iou_NN = kal.metrics.voxel.iou(NN_pred.contiguous(), tgt_voxels)

        pred_voxels = []
        for odms, voxel_NN in zip(pred_odms, NN_pred): 
            pred_voxels.append(project_odms(odms, voxel_NN, votes=2).unsqueeze(0))
        pred_voxels = torch.cat(pred_voxels)
        iou = kal.metrics.voxel.iou(pred_voxels.contiguous(), tgt_voxels)

        return iou_NN, iou

# Train
trainer = Engine(mode=args.mode, resume_name=args.resume)
print(f'Training {args.mode} model...')
for i, epoch in enumerate(range(args.epochs)): 
    trainer.train()
    if i % args.val_every == 0: 
        trainer.validate()
    trainer.save()
