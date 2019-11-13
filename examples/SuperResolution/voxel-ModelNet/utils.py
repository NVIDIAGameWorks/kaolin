import torch

import kaolin as kal 

def down_sample(tgt): 
	inp = []
	for t in tgt : 
		low_res_inp = kal.rep.voxel.scale_down(t, scale = [2, 2, 2])
		low_res_inp = kal.rep.voxel.threshold(low_res_inp, .1)
		inp.append(low_res_inp.unsqueeze(0))
	inp = torch.cat(inp, dim = 0 )
	inp = inp.unsqueeze(1)
	return inp

def up_sample(inp): 

	inp = inp[:,0]
	NN_pred = []
	for voxel in inp: 
		NN_pred.append(kal.rep.voxel.scale_up(voxel, dim = 30))
	NN_pred = torch.stack(NN_pred)
	return NN_pred