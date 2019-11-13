# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#
#
# Multi-View Silhouette and Depth Decomposition for High Resolution 3D Object Representation components
#
# Copyright (c) 2019 Edward Smith
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from kaolin.rep import VoxelGrid
from kaolin.conversions.voxelgridconversions import confirm_def
from kaolin.conversions.voxelgridconversions import threshold
from kaolin.conversions.voxelgridconversions import extract_surface
from kaolin import helpers


# Tiny eps
EPS = 1e-6


# def downsample(voxel: Union[torch.Tensor, VoxelGrid], scale: List[int],
#                inplace: Optional[bool] = True):
#     r"""Downsamples a voxelgrid, given a (down)scaling factor for each
#     dimension.

#     .. Note::
#         The voxel output is not thresholded.

#     Args:
#         voxel (torch.Tensor): Voxel grid to be downsampled (shape: must
#             be a tensor containing exactly 3 dimensions).
#         scale (list): List of tensors to scale each dimension down by
#             (length: 3).
#         inplace (bool, optional): Bool to make the operation in-place.

#     Returns:
#         (torch.Tensor): Downsampled voxelgrid.

#     Example:
#         >>> x = torch.ones([32, 32, 32])
#         >>> print (x.shape)
#         torch.Size([32, 32, 32])
#         >>> x = downsample(x, [2,2,2])
#         >>> print (x.shape)
#         torch.Size([16, 16, 16])
#     """
#     if isinstance(voxel, VoxelGrid):
#         voxel = voxel.voxels
#     voxel = confirm_def(voxel)

#     if not inplace:
#         voxel = voxel.clone()

#     # Verify that all elements of `scale` are greater than or equal to 1 and
#     # less than the voxel shape for the corresponding dimension.
#     scale_filter = [1, 1]
#     scale_factor = 1.
#     for i in range(3):
#         if scale[i] < 1:
#             raise ValueError('Downsample ratio must be at least 1 along every'
#                 ' dimension.')
#         if scale[i] >= voxel.shape[i]:
#             raise ValueError('Downsample ratio must be less than voxel shape'
#                 ' along every dimension.')
#         scale_filter.append(scale[i])
#         scale_factor *= scale[i]
#     conv_filter = torch.ones(scale_filter).to(voxel.device) / scale_factor

#     voxel = F.conv3d(voxel.unsqueeze(0).unsqueeze(
#         0), conv_filter, stride=scale, padding=0)
#     voxel = voxel.squeeze(0).squeeze(0)

#     return voxel


# def upsample(voxel: torch.Tensor, dim: int):
#     r"""Upsamples a voxel grid by a given scaling factor.

#     .. Note::
#         The voxel output is not thresholded.

#     Args:
#         voxel (torch.Tensor): Voxel grid to be upsampled (shape: must
#             be a 3D tensor)
#         dim (int): New dimensionality (number of voxels along each dimension
#             in the resulting voxel grid).

#     Returns:
#         torch.Tensor: Upsampled voxel grid.

#     Example:
#         >>> x = torch.ones([32, 32, 32])
#         >>> print (x.shape)
#         torch.Size([32, 32, 32])
#         >>> x = upsample(x, 64)
#         >>> print (x.shape)
#         torch.Size([64, 64, 64])
#     """
#     if isinstance(voxel, VoxelGrid):
#         voxel = voxel.voxels
#     voxel = confirm_def(voxel)

#     cur_shape = voxel.shape
#     assert (dim >= cur_shape[0]) and ((dim >= cur_shape[1]) and (
#         dim >= cur_shape[2])), 'All dim values must be larger then current dim'

#     new_positions = []
#     old_positions = []

#     # Defining position correspondences
#     for i in range(3):
#         shape_params = [1, 1, 1]
#         shape_params[i] = dim
#         new_pos = np.arange(dim).reshape(shape_params)

#         for j in range(3):
#             if i == j:
#                 continue
#             new_pos = np.repeat(new_pos, dim, axis=j)
#         new_pos = new_pos.reshape(-1)

#         ratio = float(cur_shape[i]) / float(dim)
#         old_pos = (new_pos * ratio).astype(int)

#         new_positions.append(new_pos)
#         old_positions.append(old_pos)

#     scaled_voxel = torch.FloatTensor(np.zeros([dim, dim, dim])).to(
#         voxel.device)
#     if voxel.is_cuda:
#         scaled_voxel = scaled_voxel.cuda()

#     scaled_voxel[tuple(new_positions)] = voxel[tuple(old_positions)]

#     return scaled_voxel


# def threshold(voxel: Union[torch.Tensor, VoxelGrid], thresh: float,
#               inplace: Optional[bool] = True):
#     r"""Binarizes the voxel array using a specified threshold.

#     Args:
#         voxel (torch.Tensor): Voxel array to be binarized.
#         thresh (float): Threshold with which to binarize.
#         inplace (bool, optional): Bool to make the operation in-place.

#     Returns:
#         (torch.Tensor): Thresholded voxel array.

#     """
#     if isinstance(voxel, VoxelGrid):
#         voxel = voxel.voxels
#     if not inplace:
#         voxel = voxel.clone()
#     helpers._assert_tensor(voxel)
#     voxel[voxel <= thresh] = 0
#     voxel[voxel > thresh] = 1
#     return voxel


# def fill(voxel: Union[torch.Tensor, VoxelGrid], thresh: float = .5):
#     r""" Fills the internal structures in a voxel grid. Used to fill holds
#     and 'solidify' objects.

#     Args:
#         voxel (torch.Tensor): Voxel grid to be filled.
#         thresh (float): Threshold to use for binarization of the grid.

#     Returns:
#         torch.Tensor: filled voxel array
#     """

#     if isinstance(voxel, VoxelGrid):
#         voxel = voxel.voxels
#     voxel = confirm_def(voxel)
#     voxel = threshold(voxel, thresh)
#     voxel = voxel.clone()
#     on = ndimage.binary_fill_holes(voxel.data.cpu())
#     voxel[np.where(on)] = 1
#     return voxel


# def extract_surface(voxel: Union[torch.Tensor, VoxelGrid], thresh: float = .5):
#     r"""Removes any inernal structure(s) from a voxel array.

#     Args:
#         voxel (torch.Tensor): voxel array from which to extract surface
#         thresh (float): threshold with which to binarize

#     Returns:
#         torch.Tensor: surface voxel array
#     """

#     if isinstance(voxel, VoxelGrid):
#         voxel = voxel.voxels
#     voxel = confirm_def(voxel)
#     voxel = threshold(voxel, thresh)
#     off_positions = voxel == 0

#     conv_filter = torch.ones((1, 1, 3, 3, 3))
#     surface_voxel = torch.zeros(voxel.shape)
#     if voxel.is_cuda:
#         conv_filter = conv_filter.cuda()
#         surface_voxel = surface_voxel.cuda()

#     local_occupancy = F.conv3d(voxel.unsqueeze(
#         0).unsqueeze(0), conv_filter, padding=1)
#     local_occupancy = local_occupancy.squeeze(0).squeeze(0)
#     # only elements with exposed faces
#     surface_positions = (local_occupancy < 27) * (local_occupancy > 0)
#     surface_voxel[surface_positions] = 1
#     surface_voxel[off_positions] = 0

#     return surface_voxel


# def extract_odms(voxel: Union[torch.Tensor, VoxelGrid]):
#     r"""Extracts an orthographic depth map from a voxel grid.

#     Args:
#         voxel (torch.Tensor): Voxel grid from which odms are extracted.

#     Returns:
#         (torch.Tensor): 6 ODMs from the 6 primary viewing angles.

#     Example:
#         >>> voxel = torch.ones([128,128,128])
#         >>> voxel = extract_odms(voxel)
#         >>> voxel.shape
#         torch.Size([6, 128, 128])
#     """
#     if isinstance(voxel, VoxelGrid):
#         voxel = voxel.voxels
#     voxel = confirm_def(voxel)
#     cuda = voxel.is_cuda
#     voxel = extract_surface(voxel)
#     voxel = voxel.data.cpu().numpy()

#     dim = voxel.shape[-1]
#     a, b, c = np.where(voxel == 1)
#     big_list = [[[[dim, dim]
#                   for j in range(dim)] for i in range(dim)] for k in range(3)]
#     for i, j, k in zip(a, b, c):
#         big_list[0][i][j][0] = (min(dim - k - 1, big_list[0][i][j][0]))
#         big_list[0][i][j][1] = (min(k, big_list[0][i][j][1]))
#         big_list[1][i][k][0] = (min(dim - j - 1, big_list[1][i][k][0]))
#         big_list[1][i][k][1] = (min(j, big_list[1][i][k][1]))
#         big_list[2][j][k][0] = (min(dim - i - 1, big_list[2][j][k][0]))
#         big_list[2][j][k][1] = (min(i, big_list[2][j][k][1]))

#     odms = np.zeros((6, dim, dim))
#     big_list = np.array(big_list)
#     for k in range(6):
#         odms[k] = big_list[k // 2, :, :, k % 2]
#     odms = torch.FloatTensor(np.array(odms))

#     if cuda:
#         odms = odms.cuda()

#     return odms


# def project_odms(odms: torch.Tensor,
#                  voxel: torch.Tensor = None, votes: int = 1):
#     r"""Projects orthographic depth map onto a voxel array.

#     .. Note::
#         If no voxel grid is provided, we poject onto a completely filled grid.

#     Args:
#         odms (torch.Tensor): ODMs which are to be projected.
#         voxel (torch.Tensor): Voxel grid onto which ODMs are projected.

#     Returns:
#         (torch.Tensor): Updated voxel grid.

#     Example:
#         >>> odms = torch.rand([6,128,128])*128
#         >>> odms = voxel.int()
#         >>> voxel = kal.rep.voxel.project_odms(odms)
#         >>> voxel.shape
#         torch.Size([128, 128, 128])
#     """
#     cuda = odms.is_cuda
#     dim = odms.shape[-1]
#     subtractor = 1. / float(votes)

#     if voxel is None:
#         voxel = torch.ones((dim, dim, dim))
#     else:
#         for i in range(3):
#             assert (voxel.shape[i] == odms.shape[-1]
#                     ), 'Voxel and odm dimension size must be the same'
#         if isinstance(voxel, VoxelGrid):
#             voxel = voxel.voxels
#         voxel = confirm_def(voxel)
#         voxel = threshold(voxel, .5)
#     voxel = voxel.data.cpu().numpy()
#     odms = odms.data.cpu().numpy()

#     for i in range(3):
#         odms[2 * i] = dim - odms[2 * i]

#     depths = np.where(odms <= dim)
#     for x, y, z in zip(*depths):
#         pos = int(odms[x, y, z])
#         if x == 0:
#             voxel[y, z, pos:dim] -= subtractor
#         elif x == 1:
#             voxel[y, z, 0:pos] -= subtractor
#         elif x == 2:
#             voxel[y, pos:dim, z] -= subtractor
#         elif x == 3:
#             voxel[y, 0:pos, z] -= subtractor
#         elif x == 4:
#             voxel[pos:dim, y, z] -= subtractor
#         else:
#             voxel[0:pos, y, z] -= subtractor

#     on = np.where(voxel > 0)
#     off = np.where(voxel <= 0)
#     voxel[on] = 1
#     voxel[off] = 0

#     voxel = confirm_def(voxel)
#     if cuda:
#         voxel = voxel.cuda()
#     return voxel


def max_connected(voxel: Union[torch.Tensor, VoxelGrid], thresh: float = .5):
    r"""Removes unconnecred voxels.

    .. Note::
        Largest maximum connected voxel is maintained.

    Args:
        voxel = voxel array
        thresh (float): threshold with which to binarize

    Returns:
        torch.Torch: updated voxel array

    Example:
        >>> voxel = torch.rand 
    """
    
    voxel = voxel.clone()
    voxel = threshold(voxel, thresh)
    max_component = np.zeros(voxel.shape, dtype=bool)
    for startx in range(voxel.shape[0]):
        for starty in range(voxel.shape[1]):
            for startz in range(voxel.shape[2]):
                if not voxel[startx,starty,startz]:
                    continue
                # start a new component
                component = np.zeros(voxel.shape, dtype=bool)
                stack = [[startx,starty,startz]]
                component[startx,starty,startz] = True
                voxel[startx,starty,startz] = False
                while len(stack) > 0:
                    x,y,z = stack.pop()
                    for i in range(x-1, x+1 + 1):
                        for j in range(y-1, y+1 + 1):
                            for k in range(z-1, z+1 + 1):
                                if (i-x)**2+(j-y)**2+(k-z)**2 > 1:
                                    continue
                                if _voxel_exist(voxel, i,j,k):
                                    voxel[i,j,k] = False
                                    component[i,j,k] = True
                                    stack.append([i,j,k])
                if component.sum() > max_component.sum():
                    max_component = component
    
    return torch.FloatTensor(max_component).to(voxel.device)


def _voxel_exist(voxels, x,y,z):
    if x < 0 or y < 0 or z < 0 or x >= voxels.shape[0]\
        or y >= voxels.shape[1] or z >= voxels.shape[2]:
        return False
    else :
        return voxels[x,y,z] == 1 


if __name__ == '__main__':

    device = 'cpu'

    # # Test downsample
    # x = torch.ones([32, 32, 32]).to(device)
    # x_ = downsample(x, [2, 2, 2])
    # print(x.shape)
    # print(x_.shape)

    # # Test upsample
    # x = torch.ones([32, 32, 32]).to(device)
    # x_ = upsample(x, 64)
    # print(x.shape)
    # print(x_.shape)

    # # Test threshold
    # x = torch.rand([2, 2, 2]).to(device)
    # x_ = threshold(x, 0.4)
