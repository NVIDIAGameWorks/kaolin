import torch

import kaolin.transforms as tfs


def up_sample(inp):
    NN_pred = []
    upsample_128 = tfs.UpsampleVoxelGrid(dim=128)
    for voxel in inp:
        NN_pred.append(upsample_128(voxel))
    NN_pred = torch.stack(NN_pred)
    return NN_pred
