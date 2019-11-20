import argparse
import os
import torch
import sys
from tqdm import tqdm

from torch.utils.data import DataLoader

from architectures import EncoderDecoderForNLL_32_128
from utils import up_sample
import kaolin as kal

parser = argparse.ArgumentParser()
parser.add_argument('-expid', type=str, default='NLLL', help='Unique experiment identifier.')
parser.add_argument('-device', type=str, default='cuda', help='Device to use')
parser.add_argument('-categories', type=str,nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('-vis', action='store_true', help='Visualize each model while evaluating')
parser.add_argument('-batchsize', type=int, default=16, help='Batch size.')
args = parser.parse_args()


# Data
valid_set = kal.datasets.ShapeNet.Voxels(root='../../datasets/',categories=args.categories,
                                         download=True, train=False, resolutions=[128, 32],
                                         split=.97)
dataloader_val = DataLoader(valid_set, batch_size=args.batchsize, shuffle=False, num_workers=8)
# Model
model = EncoderDecoderForNLL_32_128()
model = model.to(args.device)
# Load saved weights
model.load_state_dict(torch.load('log/{0}/best.pth'.format(args.expid)))

iou_epoch = 0.
iou_NN_epoch = 0.
num_batches = 0


model.eval()
with torch.no_grad():
    for data in tqdm(dataloader_val):
        tgt = data['128'].to(args.device)
        inp = data['32'].to(args.device)

        # inference
        pred = model(inp.unsqueeze(1))

        iou = kal.metrics.voxel.iou(pred[:,1,:,:].contiguous(), tgt)
        iou_epoch += iou

        NN_pred = up_sample(inp)
        iou_NN = kal.metrics.voxel.iou(NN_pred.contiguous(), tgt)
        iou_NN_epoch += iou_NN


        if args.vis:
            for i in range(inp.shape[0]):
                print ('Rendering low resolution input')
                kal.visualize.show_voxel(inp[i], mode = 'exact', thresh = .5)
                print ('Rendering high resolution target')
                kal.visualize.show_voxel(tgt[i], mode = 'exact', thresh = .5)
                print ('Rendering high resolution prediction')
                kal.visualize.show_voxel(pred[i,1], mode = 'exact', thresh = .5)
                print('----------------------')
        num_batches += 1.
out_iou_NN = iou_NN_epoch.item() / float(num_batches)
print ('Nearest Neighbor Baseline IoU over validation set is {0}'.format(out_iou_NN))
out_iou = iou_epoch.item() / float(num_batches)
print ('IoU over validation set is {0}'.format(out_iou))
