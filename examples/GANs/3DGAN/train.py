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

import kaolin as kal

from architectures import Generator, Discriminator


parser = argparse.ArgumentParser()
parser.add_argument('--modelnet-root', type=str, help='Root directory of the ModelNet dataset.')
parser.add_argument('--cache-dir', type=str, default=None, help='Path to write intermediate representation to.')
parser.add_argument('--expid', type=str, default='GAN', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--categories', type=str, nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('--epochs', type=int, default=50000, help='Number of train epochs.')
parser.add_argument('--batchsize', type=int, default=50, help='Batch size.')
parser.add_argument('--print-every', type=int, default=2, help='Print frequency (batches).')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint.')

args = parser.parse_args()


# Setup Dataloader
train_set = kal.datasets.ModelNetVoxels(basedir=args.modelnet_root, cache_dir=args.cache_dir,
                                        categories=args.categories, resolutions=[30])
dataloader_train = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, num_workers=8)


# Setup Models
gen = Generator().to(args.device)
dis = Discriminator().to(args.device)


optim_g = optim.Adam(gen.parameters(), lr=.0025, betas=(0.5, 0.999))
optim_d = optim.Adam(dis.parameters(), lr=.00005, betas=(0.5, 0.999))

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
        - cur_epoch (int): Current epoch.
        - print_every (int): How frequently (# batches) to print loss.
        - validate_every (int): How frequently (# epochs) to run validation.
    """

    def __init__(self, print_every=1, resume=False):
        self.cur_epoch = 0
        self.train_loss = []
        self.val_loss = []
        self.bestval = 0
        self.print_every = print_every

        if resume:
            self.load()

    def train(self):
        loss_epoch = 0.
        num_batches = 0
        train_dis = True
        gen.train()
        dis.train()
        # Train loop
        for i, sample in enumerate(tqdm(dataloader_train), 0):
            data = sample['data']
            optim_g.zero_grad(), gen.zero_grad()
            optim_d.zero_grad(), dis.zero_grad()

            # data creation
            real_voxels = torch.zeros(data['30'].shape[0], 32, 32, 32).to(args.device)
            real_voxels[:, 1:-1, 1:-1, 1:-1] = data['30'].to(args.device)

            # train discriminator
            z = torch.normal(torch.zeros(data['30'].shape[0], 200), 
                             torch.ones(data['30'].shape[0], 200) * .33).to(args.device)

            fake_voxels = gen(z)
            d_on_fake = dis(fake_voxels)
            d_on_real = dis(real_voxels)

            d_loss = -torch.mean(torch.log(d_on_real) + torch.log(1. - d_on_fake))

            d_accuracy = ((d_on_real >= .5).float().mean() + (d_on_fake < .5).float().mean()) / 2. 
            g_accuracy = (d_on_fake > .5).float().mean()
            train_dis = d_accuracy < .8

            if train_dis:
                dis.zero_grad()
                d_loss.backward()
                optim_d.step()

            # train generator
            z = torch.normal(torch.zeros(data['30'].shape[0], 200),
                             torch.ones(data['30'].shape[0], 200) * .33).to(args.device)
            fake_voxels = gen(z)
            d_on_fake = dis(fake_voxels)
            g_loss = -torch.mean(torch.log(d_on_fake))

            dis.zero_grad()
            gen.zero_grad()
            g_loss.backward()
            optim_g.step()

            # logging
            num_batches += 1
            if i % self.print_every == 0:
                tqdm.write(f'[TRAIN] Epoch {self.cur_epoch:03d}, Batch {i:03d}: '
                           f'gen: {float(g_accuracy.item()):2.3f}, dis = {float(d_accuracy.item()):2.3f}')

        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1

    def load(self):
        gen.load_state_dict(torch.load(os.path.join(logdir, 'gen.pth')))
        dis.load_state_dict(torch.load(os.path.join(logdir, 'dis.pth')))
        optim_g.load_state_dict(torch.load(os.path.join(logdir, 'optim_g.pth')))
        optim_d.load_state_dict(torch.load(os.path.join(logdir, 'optim_d.pth')))
        # Read data corresponding to the loaded model
        with open(os.path.join(logdir, 'recent.log'), 'r') as f:
            run_data = json.load(f)
        self.cur_epoch = run_data['epoch']

    def save(self):
        # Create a dictionary of all data to save
        log_table = {
            'epoch': self.cur_epoch
        }

        # Save the recent model/optimizer states
        torch.save(gen.state_dict(), os.path.join(logdir, 'gen.pth'))
        torch.save(dis.state_dict(), os.path.join(logdir, 'dis.pth'))
        torch.save(optim_g.state_dict(), os.path.join(logdir, 'optim_g.pth'))
        torch.save(optim_d.state_dict(), os.path.join(logdir, 'optim_d.pth'))
        # Log other data corresponding to the recent model
        with open(os.path.join(logdir, 'recent.log'), 'w') as f:
            f.write(json.dumps(log_table))

        tqdm.write('====== Saved recent model ======>')

trainer = Engine(print_every=args.print_every, resume=args.resume)

for epoch in range(args.epochs): 
    trainer.train()
    if epoch % 10 == 9: 
        trainer.save()
