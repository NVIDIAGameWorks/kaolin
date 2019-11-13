import kaolin as kal 
import torch

def down_sample(tgt): 
	inp = []
	for t in tgt : 
		low_res_inp = kal.rep.voxel.scale_down(t, scale = [2, 2, 2])
		low_res_inp = kal.rep.voxel.threshold(low_res_inp, .1)
		inp.append(low_res_inp.unsqueeze(0))
	inp = torch.cat(inp, dim = 0 )
	return inp

def up_sample(inp): 
	scaling = torch.nn.Upsample(scale_factor=4, mode='nearest')
	NN_pred = []
	for voxel in inp: 
		NN_pred.append(scaling(voxel.unsqueeze(0).unsqueeze(0)).squeeze(1))
	NN_pred = torch.stack(NN_pred).squeeze(1)
	return NN_pred

def to_occpumancy_map(inp, threshold = None):
	if threshold is None: 
		threshold = inp.shape[-1]
	zeros = inp< threshold
	ones = inp >= threshold
	inp = inp.clone()
	inp[ones] = 1 
	inp[zeros] = 0 
	return inp


def upsample_omd(inp): 
	scaling = torch.nn.Upsample(scale_factor=4, mode='nearest')
	inp = scaling(inp)
	return inp

