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

from utils import up_sample
from architectures import EncoderDecoder_32_128, EncoderDecoderForNLL_32_128

import kaolin as kal


parser = argparse.ArgumentParser()
parser.add_argument('--shapenet-root', type=str, required=True, help='Root directory of the ShapeNet dataset.')
parser.add_argument('--cache-dir', type=str, default='cache', help='Directory where intermediate representations will be stored.')
parser.add_argument('--expid', type=str, default='superres', help='Unique experiment identifier.')
parser.add_argument('--device', type=str, default='cuda', help='Device to use')
parser.add_argument('--categories', type=str, nargs='+', default=['chair'], help='list of object classes to use.')
parser.add_argument('--epochs', type=int, default=30, help='Number of train epochs.')
parser.add_argument('--batchsize', type=int, default=16, help='Batch size.')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--val-every', type=int, default=5, help='Validation frequency (epochs).')
parser.add_argument('--print-every', type=int, default=20, help='Print frequency (batches).')
parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
parser.add_argument('--resume', choices=['none', 'best', 'recent'], default='none',
                    help='Choose which weights to resume training from (none to start from random initialization.)')
args = parser.parse_args()


device = torch.device(args.device)

# Dataset Setup
train_set = kal.datasets.ShapeNet_Voxels(root=args.shapenet_root, cache_dir=args.cache_dir,
                                         categories=args.categories, train=True, resolutions=[128, 32],
                                         split=.97)
dataloader_train = DataLoader(train_set, batch_size=args.batchsize, shuffle=True, num_workers=8)

valid_set = kal.datasets.ShapeNet_Voxels(root=args.shapenet_root, cache_dir=args.cache_dir,
                                         categories=args.categories, train=False, resolutions=[128, 32],
                                         split=.97)
dataloader_val = DataLoader(valid_set, batch_size=args.batchsize, shuffle=False, num_workers=8)


# Model settings
model = EncoderDecoderForNLL_32_128().to(args.device)
class_weights = torch.tensor([0.0283, 1 - 0.0283], dtype=torch.float, device=device)
loss_fn = torch.nn.NLLLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


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
        print_every (int): How frequently (# batches) to print loss.
        resume_name (str): Prefix of weights from which to resume training. If 'none',
            no weights are loaded.
    """

    def __init__(self, print_every=1, resume_name='none'):
        self.cur_epoch = 0
        self.train_loss = []
        self.val_loss = []
        self.bestval = 0
        self.print_every = print_every

        if resume_name != 'none':
            self.load(resume_name)

    def train(self):
        loss_epoch = 0.
        num_batches = 0
        diff = 0
        model.train()
        # Train loop
        for i, sample in enumerate(tqdm(dataloader_train), 0):
            data = sample['data']
            optimizer.zero_grad()

            tgt = data['128'].to(device)
            inp = data['32'].to(device)

            # inference
            pred = model(inp.unsqueeze(1))

            # losses
            tgt = tgt.long()
            loss = loss_fn(pred, tgt)
            loss.backward()
            loss_epoch += float(loss.item())
            pred = pred[:, 1, :, :]
            iou = kal.metrics.voxel.iou(pred.contiguous(), tgt)

            # logging
            num_batches += 1
            if i % args.print_every == 0:
                tqdm.write(f'[TRAIN] Epoch {self.cur_epoch:03d}, Batch {i:03d}: Loss: {float(loss.item())}')
                tqdm.write('Metric iou: {0}'.format(iou))
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
                # data creation
                tgt = data['128'].to(device)
                inp = data['32'].to(device)

                # inference
                pred = model(inp.unsqueeze(1))

                # losses
                tgt = tgt.long()
                loss = loss_fn(pred, tgt)
                loss_epoch += float(loss.item())

                pred = pred[:, 1, :, :]
                iou = kal.metrics.voxel.iou(pred.contiguous(), tgt)
                iou_epoch += iou

                NN_pred = up_sample(inp)
                iou_NN = kal.metrics.voxel.iou(NN_pred.contiguous(), tgt)
                iou_NN_epoch += iou_NN

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
            self.val_loss.append(out_iou)

    def load(self, resume_name):
        model_path = os.path.join(logdir, f'{resume_name}.pth')
        optim_path = os.path.join(logdir, f'{resume_name}_optim.pth')
        assert os.path.exists(model_path), f'Model weights not found at "{model_path}"'
        assert os.path.exists(optim_path), f'Optim weights not found at "{optim_path}"'
        model.load_state_dict(torch.load(model_path))
        optimizer.load_state_dict(torch.load(optim_path))

        # Read data corresponding to the loaded model
        with open(os.path.join(logdir, f'{resume_name}.log'), 'r') as f:
            run_data = json.load(f)
        self.cur_epoch = run_data['epoch']

        tqdm.write(f'====== Loaded {resume_name} model ======>')

    def save(self):
        save_best = False
        if self.val_loss[-1] >= self.bestval:
            self.bestval = self.val_loss[-1]
            save_best = True

        # Create a dictionary of all data to save
        log_table = {
            'epoch': self.cur_epoch,
            'bestval': np.min(np.asarray(self.val_loss)),
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_metrics': ['NLLLoss', 'iou'],
            'val_metrics': ['NLLLoss', 'iou', 'iou_NN'],
        }

        # Save the recent model/optimizer states
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


trainer = Engine(print_every=args.print_every, resume_name=args.resume)

for epoch in range(args.epochs):
    trainer.train()
    trainer.validate()
    trainer.save()
