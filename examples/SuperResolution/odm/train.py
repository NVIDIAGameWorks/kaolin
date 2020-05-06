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


# Setup load and save functions
def load(path, name):
    model_state_dict = torch.load(os.path.join(path, f'{name}.pth'))
    optimizer_state_dict = torch.load(os.path.join(path, f'{name}_optim.pth'))
    with open(os.path.join(path, 'recent.log'), 'r') as f:
        log = json.load(f)
    return model_state_dict, optimizer_state_dict, log

def save(path, name, model, optimizer, log):
    torch.save(model.state_dict(), os.path.join(path, f'{name}.pth'))
    torch.save(optimizer.state_dict(), os.path.join(path, f'{name}_optim.pth'))
    # Log other data corresponding to the model
    with open(os.path.join(path, f'{name}.log'), 'w') as f:
        f.write(json.dumps(log))
    tqdm.write('====== Saved recent model ======>')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['Direct', 'MVD'], default='MVD')
    parser.add_argument('--shapenet-root', type=str, help='Path to shapenet data directory.')
    parser.add_argument('--categories', type=str, nargs='+', default=['chair'], help='list of object classes to use')
    parser.add_argument('--cache-dir', type=str, help='Path to data directory.')
    parser.add_argument('--expid', type=str, default='ODM', help='Unique experiment identifier.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--epochs', type=int, default=100, help='Number of train epochs.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size.')
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--val-every', type=int, default=5, help='Validation frequency (epochs).')
    parser.add_argument('--save-every', type=int, default=20, help='Save frequency (epochs).')
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


    # Initialize logging variables
    start_epoch = 0
    train_loss = []
    val_score = []
    bestval = 0

    # Load checkpoint if `resume_name` is supplied
    if args.resume:
        model_state_dict, optimizer_state_dict, log = load(logdir, args.resume)
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        start_epoch = log['epoch'] + 1
        bestval = log['bestval']
        train_loss = log['train_loss']
        val_score = log['val_score']

    # Main training loop
    print(f'Training {args.mode} model...')
    for epoch in range(start_epoch, args.epochs): 
        # TRAIN
        model.train()
        loss_epoch = 0.
        for batch_idx, sample in enumerate(tqdm(dataloader_train)):
            data = sample['data']
            inp_odms = data[32]['odms'].to(args.device)
            tgt_odms = data[128]['odms'].to(args.device)

            # Get predictions
            pred_odms = model(inp_odms)
            if args.mode == 'MVD':
                initial_odms = upsample_odm(inp_odms) * 4
                distance = 128 - initial_odms
                pred_odms_update = pred_odms * distance
                pred_odms = initial_odms + pred_odms_update

            # Compute loss
            tgt = to_occupancy_map(tgt_odms) if args.mode == 'Direct' else tgt_odms
            loss = loss_fn(pred_odms, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            loss_epoch += loss.item()
            if batch_idx % args.print_every == 0:
                tqdm.write(f'[TRAIN] Epoch {epoch:03d}, Batch {batch_idx:03d}: Loss: {float(loss.item())}')

            loss_epoch = loss_epoch / len(dataloader_train)
            train_loss.append(loss_epoch)

        if epoch % args.val_every == 0:
            # VALIDATE
            model.eval()
            with torch.no_grad():	
                iou_epoch = 0.
                iou_NN_epoch = 0.
                loss_epoch = 0.
                num_batches = len(dataloader_val)

                for batch_idx, sample in enumerate(tqdm(dataloader_val)):
                    data = sample['data']
                    inp_odms = data[32]['odms'].to(args.device)
                    inp_voxels = data[32]['voxels'].to(args.device)
                    tgt_voxels = data[128]['voxels'].to(args.device)

                    # Get predictions
                    pred_odms = model(inp_odms)
                    if args.mode == 'MVD':
                        initial_odms = upsample_odm(inp_odms) * 4
                        distance = 128 - initial_odms
                        pred_odms_update = pred_odms * distance
                        pred_odms = initial_odms + pred_odms_update

                    # Calculate IoU
                    if args.mode == 'Direct':
                        pred_odms = to_occupancy_map(pred_odms, threshold=0.5)
                        pred_odms = pred_odms * pred_odms.shape[-1]
                    elif args.mode == 'MVD':
                        pred_odms = pred_odms.int()

                    # Compute the Nearest Neighbour (NN) upsampled voxel grid as a baseline
                    NN_pred = up_sample(inp_voxels)
                    iou_NN = kal.metrics.voxel.iou(NN_pred.contiguous(), tgt_voxels)

                    pred_voxels = []
                    for odms, voxel_NN in zip(pred_odms, NN_pred): 
                        pred_voxels.append(project_odms(odms, voxel_NN, votes=2).unsqueeze(0))
                    pred_voxels = torch.cat(pred_voxels)
                    iou = kal.metrics.voxel.iou(pred_voxels.contiguous(), tgt_voxels)

                    # Logging
                    iou_NN_epoch += iou_NN
                    iou_epoch += iou

                    if batch_idx % args.print_every == 0:
                        out_iou = iou_epoch.item() / (batch_idx + 1)
                        out_iou_NN = iou_NN_epoch.item() / (batch_idx + 1)
                        tqdm.write(f'[VAL] Epoch {epoch:03d}, Batch {batch_idx:03d}: IoU: {out_iou}, Iou Baseline: {out_iou_NN}')

                out_iou = iou_epoch.item() / num_batches
                out_iou_NN = iou_NN_epoch.item() / num_batches
                tqdm.write(f'[VAL Total] Epoch {epoch:03d}, Batch {batch_idx:03d}: IoU: {out_iou}, Iou Baseline: {out_iou_NN}')

                val_score.append(out_iou)

        if epoch % args.save_every == 0:
            # Create a dictionary of all data to save
            log = {
                'epoch': epoch,
                'bestval': bestval,
                'train_loss': train_loss,
                'val_score': val_score,
                'train_metrics': ['NLLLoss', 'iou'],
                'val_metrics': ['NLLLoss', 'iou', 'iou_NN'],
            }
            save(logdir, 'recent', model, optimizer, log)
            if val_score[-1] > bestval:
                bestval = val_score[-1]
                save(logdir, 'best', model, optimizer, log)

if __name__ == '__main__':
    main()
