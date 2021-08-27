# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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


__all__ = [
    'points_to_morton',
    'morton_to_points',
    'points_to_corners',
    'points_to_coeffs',
    'unbatched_points_to_octree',
    'quantize_points'
]

import torch

from kaolin import _C

def quantize_points(x, level):
    r"""Quantize [-1, 1] float coordinates in to [0, (2^level)-1] integer coords.

    If a point is out of the range [-1, 1] it will be clipped to it.

    Args:
        x (torch.FloatTensor): floating point coordinates,
                               must but of last dimension 3.
        level (int): Level of the grid

    Returns
        (torch.ShortTensor): Quantized 3D points, of same shape than x.
    """
    res = 2 ** level
    qpts = torch.floor(torch.clamp(res * (x + 1.0) / 2.0, 0, res - 1.)).short()
    return qpts

def unbatched_points_to_octree(points, level, sorted=False):
    r"""Convert (quantized) 3D points to an octree.

    This function assumes that the points are all in the same frame of reference
    of [0, 2^level]. Note that SPC.points does not satisfy this constraint.

    Args:
        points (torch.ShortTensor):
            The Quantized 3d points. This is not exactly like SPC points hierarchies
            as this is only the data for a specific level.
        level (int): Max level of octree, and the level of the points.
        sorted (bool): True if the points are unique and sorted in morton order.

    Returns:
        (torch.ByteTensor): the generated octree,
                            of shape :math:`(2^\text{level}, 2^\text{level}, 2^\text{level})`.
    """
    if not sorted:
        unique = torch.unique(points.contiguous(), dim=0).contiguous()
        morton = torch.sort(points_to_morton(unique).contiguous())[0]
        points = morton_to_points(morton.contiguous())
    return _C.ops.spc.points_to_octree(points.contiguous(), level)

def points_to_morton(points):
    r"""Convert (quantized) 3D points to morton codes.

    Args:
        points (torch.ShortTensor):
            Quantized 3D points. This is not exactly like SPC points hierarchies
            as this is only the data for a specific level,
            of shape :math:`(\text{num_points}, 3)`.

    Returns:
        (torch.LongTensor):
            The morton code of the points,
            of shape :math:`(\text{num_points})`

    Examples:
        >>> inputs = torch.tensor([
        ...     [0, 0, 0],
        ...     [0, 0, 1],
        ...     [0, 0, 2],
        ...     [0, 0, 3],
        ...     [0, 1, 0]], device='cuda', dtype=torch.int16)
        >>> points_to_morton(inputs)
        tensor([0, 1, 8, 9, 2], device='cuda:0')
    """
    shape = list(points.shape)[:-1]
    points = points.reshape(-1, 3)
    return _C.ops.spc.spc_point2morton(points.contiguous()).reshape(*shape)

def morton_to_points(morton):
    r"""Convert morton codes to points.

    Args:
        morton (torch.LongTensor): The morton codes of quantized 3D points,
                                   of shape :math:`(\text{num_points})`.

    Returns:
        (torch.ShortInt):
            The points quantized coordinates,
            of shape :math:`(\text{num_points}, 3)`.

    Examples:
        >>> inputs = torch.tensor([0, 1, 8, 9, 2], device='cuda')
        >>> morton_to_points(inputs)
        tensor([[0, 0, 0],
                [0, 0, 1],
                [0, 0, 2],
                [0, 0, 3],
                [0, 1, 0]], device='cuda:0', dtype=torch.int16)
    """
    shape = list(morton.shape)
    shape.append(3)
    morton = morton.reshape(-1)
    return _C.ops.spc.spc_morton2point(morton.contiguous()).reshape(*shape)

def points_to_corners(points):
    r"""Calculates the corners of the points assuming each point is the 0th bit corner.

    Args:
        points (torch.ShortTensor): Quantized 3D points,
                                    of shape :math:`(\text{num_points}, 3)`.

    Returns:
        (torch.ShortTensor): Quantized 3D new points,
                             of shape :math:`(\text{num_points}, 8, 3)`.

    Examples:
        >>> inputs = torch.tensor([
        ...     [0, 0, 0],
        ...     [0, 2, 0]], device='cuda', dtype=torch.int16)
        >>> points_to_corners(inputs)
        tensor([[[0, 0, 0],
                 [0, 0, 1],
                 [0, 1, 0],
                 [0, 1, 1],
                 [1, 0, 0],
                 [1, 0, 1],
                 [1, 1, 0],
                 [1, 1, 1]],
        <BLANKLINE>
                [[0, 2, 0],
                 [0, 2, 1],
                 [0, 3, 0],
                 [0, 3, 1],
                 [1, 2, 0],
                 [1, 2, 1],
                 [1, 3, 0],
                 [1, 3, 1]]], device='cuda:0', dtype=torch.int16)
    """
    shape = list(points.shape)
    shape.insert(-1, 8)
    return _C.ops.spc.spc_point2corners(points.contiguous()).reshape(*shape)

def points_to_coeffs(x, points):
    r"""Calculates the coefficients for trilinear interpolation.

    To interpolate with the coefficients, do:
    ``torch.sum(features * coeffs, dim=-1)``
    with ``features`` of shape :math:`(\text{num_points}, 8)`

    Args:
        x (torch.FloatTensor): Floating point 3D points,
                               of shape :math:`(\text{num_points}, 3)`.
        points (torch.ShortTensor): Quantized 3D points (the 0th bit of the voxel x is in),
                                    of shape :math:`(\text{num_points}, 3)`.

    Returns:
        (torch.FloatTensor): The trilinear interpolation coefficients,
                             of shape :math:`(\text{num_points}, 8)`.
    """
    shape = list(points.shape)
    shape[-1] = 8
    points = points.reshape(-1, 3)
    x = x.reshape(-1, 3)
    return _C.ops.spc.spc_point2coeff(x.contiguous(), points.contiguous()).reshape(*shape)
