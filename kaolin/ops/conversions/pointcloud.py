# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

__all__ = ['pointclouds_to_voxelgrids']

def pointclouds_to_voxelgrids(pointclouds, resolution, return_sparse=False):
    r"""Converts pointclouds to voxelgrids. It separates the 3D space into empty
    voxelgrid, and for each boxes, if there is a corresponding point, set that voxelgrid
    to be occupied.

    Args:
        pointclouds (torch.Tensor):
            Exact batched pointclouds with shape
            :math:`(\text{batch_size}, \text{P}, \text{3})`.
        resolution (int):
            Resolution of output voxelgrids
        return_sparse (bool):
            Whether to return a sparse voxelgrids or not.

    Returns:
        (torch.Tensor or torch.FloatTensor):
         Exact batched voxelgrids with shape
         :math:`(\text{batch_size}, \text{resolution}, \text{resolution}, \test{resolution})`.
         If `return_sparse == True`, a sparse FloatTensor is returned.

    Example:
        >>> pointclouds = torch.tensor([[[0, 0, 0],
        ...                              [1, 1, 1],
        ...                              [2, 2, 2]]], dtype=torch.float)
        >>> pointclouds_to_voxelgrids(pointclouds, 3)
        tensor([[[[1., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 0.]],
        <BLANKLINE>
                 [[0., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 0.]],
        <BLANKLINE>
                 [[0., 0., 0.],
                  [0., 0., 0.],
                  [0., 0., 1.]]]])
    """
    batch_size = pointclouds.shape[0]
    num_p = pointclouds.shape[1]

    device = pointclouds.device
    dtype = pointclouds.dtype

    if isinstance(resolution, int):
        vg_size = (batch_size, resolution, resolution, resolution)
    else:
        raise ValueError(f'Resolution `{resolution}` must be an integer.')

    pc_size = pointclouds.max(dim=1)[0] - pointclouds.min(dim=1)[0]  # size of (batch_size, P)
    max_pc_size = pc_size.max(dim=1)[0]  # size of (batch_size)

    mult = torch.ones(batch_size, device=device, dtype=dtype) * (resolution - 1) / max_pc_size   # size of (batch_size)

    prefix_index = torch.arange(start=0, end=batch_size, device=device, dtype=torch.long).repeat(num_p, 1).T.reshape(-1, 1)

    pc_index = torch.unique(torch.round(((pointclouds - pointclouds.min(dim=1)[0].unsqueeze(1)) * mult.view(-1, 1, 1))).long(), dim=1)
    pc_index = pc_index.reshape(-1, 3)
    pc_index = torch.cat((prefix_index, pc_index), dim=1)

    vg = torch.sparse.FloatTensor(pc_index.T, torch.ones(batch_size * num_p, device=pc_index.device), vg_size)

    if not return_sparse:
        vg = vg.to_dense().to(dtype)

    return vg
