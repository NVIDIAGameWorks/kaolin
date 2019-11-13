import torch 

import kaolin as kal

def up_sample(inp): 
	NN_pred = []
	for voxel in inp: 
		NN_pred.append(kal.rep.voxel.scale_up(voxel, dim = 128))
	NN_pred = torch.stack(NN_pred)
	return NN_pred
