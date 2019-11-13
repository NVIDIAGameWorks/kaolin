import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import soft_renderer.cuda.voxelization as voxelization_cuda


def voxelize_sub1(faces, size, dim):
    bs = faces.size(0)
    nf = faces.size(1)
    if dim == 0:
        faces = faces[:, :, :, [2, 1, 0]].contiguous()
    elif dim == 1:
        faces = faces[:, :, :, [0, 2, 1]].contiguous()
    voxels = torch.zeros(bs, size, size, size).int().cuda()
    return voxelization_cuda.voxelize_sub1(faces, voxels)[0].transpose(dim + 1, -1)

def voxelize_sub2(faces, size):
    bs = faces.size(0)
    nf = faces.size(1)
    voxels = torch.zeros(bs, size, size, size).int().cuda()
    return voxelization_cuda.voxelize_sub2(faces, voxels)[0]

def voxelize_sub3(faces, voxels):
    bs = voxels.size(0)
    vs = voxels.size(1)
    visible = torch.zeros_like(voxels, dtype=torch.int32).cuda()
    voxels, visible = voxelization_cuda.voxelize_sub3(faces, voxels, visible)

    sum_visible = visible.sum()

    while True:
        voxels, visible = voxelization_cuda.voxelize_sub4(faces, voxels, visible)
        if visible.sum() == sum_visible:
            break
        else:
            sum_visible = visible.sum()
    return 1 - visible


def voxelization(faces, size, normalize=False):
    faces = faces.clone()
    if normalize:
        pass
    else:
        faces *= size

    voxels0 = voxelize_sub1(faces, size, 0)
    voxels1 = voxelize_sub1(faces, size, 1)
    voxels2 = voxelize_sub1(faces, size, 2)
    voxels3 = voxelize_sub2(faces, size)

    voxels = voxels0 + voxels1 + voxels2 + voxels3
    voxels = (voxels > 0).int()
    voxels = voxelize_sub3(faces, voxels)

    return voxels



