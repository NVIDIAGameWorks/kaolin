import argparse
import os
import torch
import sys
from tqdm import tqdm 
from PIL import Image 

from torch.utils.data import DataLoader

from architectures import upscale
from utils import down_sample
from dataloaders import ShapeNet_ODMS
from utils import down_sample, up_sample, upsample_omd, to_occpumancy_map
import kaolin as kal 

parser = argparse.ArgumentParser()
parser.add_argument('-expid', type=str, default='MVD', help='Unique experiment identifier.')
parser.add_argument('-device', type=str, default='cuda', help='Device to use')
parser.add_argument('-categories', type=str,nargs='+', default=['chair'], help='list of object classes to use')
parser.add_argument('-vis', action='store_true', help='Visualize each model while evaluating')
parser.add_argument('-batchsize', type=int, default=16, help='Batch size.')
args = parser.parse_args()

# Data
valid_set = ShapeNet_ODMS(root ='../..//datasets',categories = args.categories, \
	download = True, train = False, high = 128, low = 32, split=.97, voxels = True)
dataloader_val = DataLoader(valid_set, batch_size=args.batchsize, shuffle=False, \
	num_workers=8)


# Model
model_res = upscale(128,32)
model_res = model_res.to(args.device)
model_occ = upscale(128,32)
model_occ = model_occ.to(args.device)
# Load saved weights
model_res.load_state_dict(torch.load('log/{0}/resbest.pth'.format(args.expid)))
model_occ.load_state_dict(torch.load('log/{0}/occbest.pth'.format(args.expid)))


iou_epoch = 0.
iou_NN_epoch = 0.
num_batches = 0


model_res.eval()
model_occ.eval()
with torch.no_grad():
	for data in tqdm(dataloader_val): 
		
		tgt_odms = data['odms_128'].to(args.device)
		tgt_voxels = data['voxels_128'].to(args.device)
		tgt_odms_occ = to_occpumancy_map(tgt_odms)
		inp_odms = data['odms_32'].to(args.device)
		inp_voxels = data['voxels_32'].to(args.device)
		
		# inference res
		initial_odms = upsample_omd(inp_odms)*4
		distance = 128 - initial_odms
		pred_odms_update = model_res(inp_odms)
		pred_odms_update = pred_odms_update * distance
		pred_odms_res = initial_odms + pred_odms_update

		# inference occ
		pred_odms_occ = model_occ(inp_odms)

		# combine
		pred_odms_res = pred_odms_res.int()
		ones = pred_odms_occ > .5
		zeros = pred_odms_occ <= .5
		pred_odms_occ[ones] =  pred_odms_occ.shape[-1]
		pred_odms_occ[zeros] = 0  

		NN_pred = up_sample(inp_voxels)
		iou_NN = kal.metrics.voxel.iou(NN_pred.contiguous(), tgt_voxels)
		iou_NN_epoch += iou_NN
		
		pred_voxels = []
		for i in range(inp_voxels.shape[0]):	
			voxel = NN_pred[i]
			voxel = kal.rep.voxel.project_odms(pred_odms_res[i], voxel = voxel, votes = 2)
			voxel = kal.rep.voxel.project_odms(pred_odms_occ[i], voxel = voxel, votes = 2)
			voxel = voxel.unsqueeze(0)
			pred_voxels.append(voxel)
		pred_voxels = torch.cat(pred_voxels)
		iou = kal.metrics.voxel.iou(pred_voxels.contiguous(), tgt_voxels)
		iou_epoch += iou

		
		if args.vis: 
			for i in range(inp_voxels.shape[0]):	
			
				print ('Rendering low resolution input')
				kal.visualize.show_voxel(inp_voxels[i], mode = 'exact', thresh = .5)
				print ('Rendering high resolution target')
				kal.visualize.show_voxel(tgt_voxels[i], mode = 'exact', thresh = .5)
				print ('Rendering high resolution prediction')
				kal.visualize.show_voxel(pred_voxels[i], mode = 'exact', thresh = .5)
				print('----------------------')
		num_batches += 1  
iou_NN_epoch = iou_NN_epoch.item() / float(num_batches)
print ('IoU for Nearest Neighbor baseline over validation set is {0}'.format(iou_NN_epoch))		
out_iou = iou_epoch.item() / float(num_batches)
print ('IoU over validation set is {0}'.format(out_iou))