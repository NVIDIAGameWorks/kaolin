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
from easydict import EasyDict


import kaolin as kal
from kaolin.datasets import shapenet

from kaolin.models.AtlasNet import AtlasNet
from kaolin.models.PointNet import PointNetFeatureExtractor as Encoder


parser = argparse.ArgumentParser()
parser.add_argument('--shapenet-root', type=str, help='Root directory of the ShapeNet dataset.')
parser.add_argument('--shapenet-images-root', type=str, help='Root directory of the ShapeNet Rendering dataset.')
parser.add_argument('--cache-dir', type=str, default='cache', help='Path to write intermediate representation to.')
parser.add_argument('--expid', type=str, default='AtlasNet_1', help='Unique experiment identifier. If using latent-loss, '
                    'must be the same expid used during auto-encoder training.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--categories', type=str, nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('--epochs', type=int, default=600, help='Number of train epochs.')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--batch-size', type=int, default=32, help='batch size')
parser.add_argument('--val-every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--print-every', type=int, default=20, help='Print frequency (batches).')
parser.add_argument('--pre_train', action='store_true', help='indicates latent loss should be used')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('--resume', choices=['best', 'recent'], default=None,
                    help='Choose which weights to resume training from (None to start from random initialization.)')
args = parser.parse_args()


# Setup Dataset
train_set = shapenet.ShapeNet_Points(root=args.shapenet_root, cache_dir=args.cache_dir, categories=args.categories,
                                      train=True, split=.7, num_points=3000)
dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=0)


valid_set = shapenet.ShapeNet_Points(root=args.shapenet_root, cache_dir=args.cache_dir, categories=args.categories,
                                            train=False, split=.7, num_points=10000)
dataloader_val = DataLoader(valid_set, batch_size=args.batch_size, shuffle=False, num_workers=8)


# Setup models
opt = EasyDict({
    "device": args.device,
    "number_points": 2048,
    "number_points_eval": 10000,
    "nb_primitives": 25,
    "remove_all_batchNorms": False,
    "template_type": "SQUARE", #Can also be SPHERE
    "bottleneck_size": 1024,
    "dim_template": 2,
    "hidden_neurons": 512,
    "num_layers": 2,
    "activation": "relu", # can be "relu", "sigmoid", "softplus", "logsigmoid", "softsign", "tanh"
})

encoder = Encoder(feat_size = opt.bottleneck_size).to(args.device)
decoder = AtlasNet(opt).to(args.device)

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200,400], gamma=0.1)

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
        num_batches = 0

        encoder.train()
        decoder.train()

        # Train loop
        for i, sample in enumerate(tqdm(dataloader_train), 0):
            data = sample['data']
            optimizer.zero_grad()

            # Data Creation
            tgt_points = data['points'].to(args.device)
            tgt_points = tgt_points - tgt_points.mean(1, keepdim=True)
            tgt_points = tgt_points / torch.sqrt(torch.max((tgt_points**2).sum(2, keepdim=True), 1, keepdim=True)[0])

            # Inference
            pts_features = encoder(tgt_points)
            generated_points = decoder(pts_features)

            # losses
            chamfer_loss = 0
            for index in range(tgt_points.size(0)):
                chamfer_loss = chamfer_loss + kal.metrics.point.chamfer_distance(tgt_points[index], generated_points[index])
            loss = chamfer_loss / tgt_points.size(0)
            loss.backward()
            loss_epoch += float(chamfer_loss.item())

            # logging
            num_batches +=  tgt_points.size(0)
            if i % args.print_every == 0:
                tqdm.write(f'[TRAIN] Epoch {self.cur_epoch:03d}, Batch {i:03d}: Loss: {float(loss.item()):5.5f}')
            optimizer.step()

        loss_epoch = loss_epoch / num_batches
        tqdm.write(f'[TRAIN Total] Epoch {self.cur_epoch:03d}, Batch {i:03d}: Chamfer: {loss_epoch:5.5f}')
        self.train_loss[self.cur_epoch] = float(loss_epoch)


    def validate(self):
        encoder.eval()
        decoder.eval()

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
                tgt_points = tgt_points - tgt_points.mean(1, keepdim=True)
                tgt_points = tgt_points / torch.sqrt(
                    torch.max((tgt_points ** 2).sum(2, keepdim=True), 1, keepdim=True)[0])

                # Inference
                pts_features = encoder(tgt_points)
                generated_points = decoder(pts_features)

                # losses
                chamfer_loss = 0
                f_loss = 0
                for index in range(tgt_points.size(0)):
                    chamfer_loss += kal.metrics.point.chamfer_distance(tgt_points[index], generated_points[index])
                    # F-Score
                    f_loss += kal.metrics.point.f_score(tgt_points[index], generated_points[index], extend=False)

                loss_epoch += chamfer_loss
                f_score += f_loss

                # logging
                num_batches += tgt_points.size(0)
                if i % args.print_every == 0:
                    out_loss = loss_epoch / num_batches
                    out_f_score = f_score / num_batches
                    tqdm.write(f'[VAL]\tEpoch {self.cur_epoch:03d}, Batch {i:03d}: F-Score: {out_f_score:3.3f} Chamfer: {out_loss:5.5f}')

            out_loss = loss_epoch / num_batches
            out_f_score = f_score / num_batches
            tqdm.write(f'[VAL Total] Epoch {self.cur_epoch:03d}, Batch {i:03d}: F-Score: {out_f_score:3.3f} Chamfer: {out_loss:5.5f}')

            self.val_score[self.cur_epoch] = float(out_f_score.item())

    def step(self):
        self.cur_epoch += 1

    def load(self, resume_name):
        checkpoint = torch.load(os.path.join(logdir, f'{resume_name}.ckpt'))
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

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
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
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
    scheduler.step()
