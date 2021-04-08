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

def _base_points_to_voxelgrids(points, resolution, return_sparse=False):
    r"""Converts points to voxelgrids. This is the base function for both trianglemeshes_to_voxelgrids
    and pointclouds_to_voxelgrids. Only points within range [0, 1] are used for voxelization. Points outside
    of [0, 1] will be discarded.

    Args:
        points (torch.Tensor):
            Exact batched points with shape
            :math: `(\text{batch_size}, \text{P}, \text{3})
        resolution (int):
            Resolution of output voxelgrids
        return_sparse (bool):
            Whether to return a sparse voxelgrids or not.

    Returns:
        (torch.Tensor or torch.FloatTensor):
        Exact batched voxelgrids with shape
        :math:`(\text{batch_size}, \text{resolution}, \text{resolution}, \test{resolution})`.
        If return_sparse == True, sparse tensor is returned.
    """
    batch_size = points.shape[0]
    num_p = points.shape[1]

    device = points.device
    dtype = points.dtype

    vg_size = (batch_size, resolution, resolution, resolution)

    mult = torch.ones(batch_size, device=device, dtype=dtype) * (resolution - 1)  # size of (batch_size)

    prefix_index = torch.arange(start=0, end=batch_size, device=device, dtype=torch.long).repeat(num_p, 1).T.reshape(-1, 1)

    pc_index = torch.round(((points) * mult.view(-1, 1, 1))).long()
    pc_index = torch.cat((prefix_index, pc_index.reshape(-1, 3)), dim=1)
    pc_index = torch.unique(pc_index, dim=0)

    # filter point that is outside of range 0 and resolution - 1
    condition = pc_index[:, 1:] <= (resolution - 1)
    condition = torch.logical_and(condition, pc_index[:, 1:] >= 0)
    row_cond = condition.all(1)

    pc_index = pc_index[row_cond, :]
    pc_index = pc_index.reshape(-1, 4)

    vg = torch.sparse.FloatTensor(pc_index.T, torch.ones(pc_index.shape[0], device=pc_index.device), vg_size)

    if not return_sparse:
        vg = vg.to_dense().to(dtype)

    return vg

def pointclouds_to_voxelgrids(pointclouds, resolution, origin=None, scale=None, return_sparse=False):
    r"""Converts pointclouds to voxelgrids. It separates the 3D space into empty
    voxelgrid, and for each boxes, if there is a corresponding point, set that voxelgrid
    to be occupied.

    Args:
        pointclouds (torch.Tensor):
            Exact batched pointclouds with shape
            :math:`(\text{batch_size}, \text{P}, \text{3})`.
        resolution (int):
            Resolution of output voxelgrids
        origin (torch.tensor): origin of the voxelgrid in the pointcloud coordinates. 
                               It has shape :math:`(\text{batch_size}, 3)`.
                               Default: origin = torch.min(pointcloud, dim=1)[0]
        scale (torch.tensor): the scale by which we divide the pointclouds' coordinates.
                              It has shape :math:`(\text{batch_size})`.
                              Default: scale = torch.max(torch.max(pointclouds, dim=1)[0] - origin, dim=1)[0]
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
    if not isinstance(resolution, int):
        raise TypeError(f"Expected resolution to be int "
                        f"but got {type(resolution)}.")

    if origin is None:
        min_val = torch.min(pointclouds, dim=1)[0]
        origin = min_val

    if scale is None:
        max_val = torch.max(pointclouds, dim=1)[0]
        scale = torch.max(max_val - origin, dim=1)[0]

    # Normalize pointcloud with origin and scale
    pointclouds = (pointclouds - origin.unsqueeze(1)) / scale.view(-1, 1, 1)

    vg = _base_points_to_voxelgrids(pointclouds, resolution, return_sparse=return_sparse)

    return vg
