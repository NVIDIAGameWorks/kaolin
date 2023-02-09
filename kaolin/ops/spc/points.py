# Copyright (c) 2021,22 NVIDIA CORPORATION & AFFILIATES.
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
    'unbatched_interpolate_trilinear',
    'coords_to_trilinear',
    'coords_to_trilinear_coeffs',
    'unbatched_points_to_octree',
    'quantize_points',
    'create_dense_spc'
]

import warnings
import numpy as np
import torch

from kaolin import _C

def quantize_points(x, level):
    r"""Quantize :math:`[-1, 1]` float coordinates in to
    :math:`[0, (2^{level})-1]` integer coords.

    If a point is out of the range :math:`[-1, 1]` it will be clipped to it.

    Args:
        x (torch.Tensor): Floating point coordinates,
                          must be of last dimension 3.
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
    of :math:`[0, 2^level]`. Note that SPC.points does not satisfy this constraint.

    Args:
        points (torch.ShortTensor):
            Quantized 3d points. This is not exactly like SPC points hierarchies
            as this is only the data for a specific level,
            of shape :math:`(\text{num_points}, 3)`.
        level (int): Max level of octree, and the level of the points.
        sorted (bool): True if the points are unique and sorted in morton order.
                       Default=False.

    Returns:
        (torch.ByteTensor):
            the generated octree,
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
    return _C.ops.spc.points_to_morton_cuda(points.contiguous()).reshape(*shape)

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
    return _C.ops.spc.morton_to_points_cuda(morton.contiguous()).reshape(*shape)

def points_to_corners(points):
    r"""Calculates the corners of the points assuming each point is the 0th bit corner.

    Args:
        points (torch.ShortTensor): Quantized 3D points,
                                    of shape :math:`(\text{num_points}, 3)`.

    Returns:
        (torch.ShortTensor):
            Quantized 3D new points,
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
    return _C.ops.spc.points_to_corners_cuda(points.contiguous()).reshape(*shape)

class InterpolateTrilinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, coords, pidx, point_hierarchy, trinkets, feats, level):

        feats_out = _C.ops.spc.interpolate_trilinear_cuda(coords.contiguous(), pidx.contiguous(),
                                                          point_hierarchy.contiguous(), trinkets.contiguous(),
                                                          feats.contiguous(), level)

        ctx.save_for_backward(coords, pidx, point_hierarchy, trinkets, feats)
        ctx.level = level
        ctx.feats_shape = feats.shape
        ctx.coords_shape = coords.shape
        return feats_out

    @staticmethod
    def backward(ctx, grad_output):
        coords, pidx, point_hierarchy, trinkets, feats = ctx.saved_tensors

        level = ctx.level
        mask = pidx > -1
        selected_points = point_hierarchy.index_select(0, pidx[mask])
        selected_trinkets = trinkets.index_select(0, pidx[mask])

        is_needs_grad_by_coords = ctx.needs_input_grad[0]
        is_needs_grad_by_features = ctx.needs_input_grad[4]

        grad_feats = None
        if is_needs_grad_by_features:
            # TODO(ttakikawa): Write a fused kernel
            grad_feats = torch.zeros(ctx.feats_shape, device=grad_output.device, dtype=grad_output.dtype)
            coeffs = coords_to_trilinear_coeffs(coords[mask], selected_points[:, None].repeat(1, coords.shape[1], 1), level).type(grad_output.dtype)
            grad_per_corner = (coeffs[..., None] * grad_output[mask][..., None, :]).sum(1)
            grad_feats.index_add_(0, selected_trinkets.reshape(-1),
                                  grad_per_corner.reshape(-1, ctx.feats_shape[-1]).to(grad_feats.dtype))

        # TODO (operel): May want to reimplement with CUDA
        grad_coords = None
        if is_needs_grad_by_coords:
            # Let N be the number of intersected cells in a batch (e.g. pidx > -1)
            # Let D be the features dimensionality
            # Shape (N, 3), xyz coords of intersected cells in range [0, 2^lod]
            coords_ = (2 ** level) * (coords[mask].reshape(-1, 3) * 0.5 + 0.5)
            # Shape (N, 3), quantized xyz coords of intersected cells in range [0, 2^lod]
            points_ = selected_points[:, None].repeat(1, coords.shape[1], 1).reshape(-1, 3)
            # Shape (N, 3), local cell coordinates in range [0.0, 1.0]
            x_ = coords_ - points_
            # Shape (N, 3), 1.0 - local cell coordinates in range [0.0, 1.0]
            _x = 1.0 - x_
            # Shape (N, 8 x 3) tensor of @(coeffs)/@(xyz) where
            # coeffs is the tensor of c000, c001, .. c111, the trilinear interp coefficients
            # (see coords_to_trilinear_coeffs), and xyz is the coords
            grad_coeffs_by_xyz = torch.stack([
                -_x[:, 1] * _x[:, 2],      -_x[:, 0] * _x[:, 2],      -_x[:, 0] * _x[:, 1],
                -_x[:, 1] * x_[:, 2],      -_x[:, 0] * x_[:, 2],       _x[:, 0] * _x[:, 1],
                -x_[:, 1] * _x[:, 2],      _x[:, 0] * _x[:, 2],       -_x[:, 0] * x_[:, 1],
                -x_[:, 1] * x_[:, 2],      _x[:, 0] * x_[:, 2],       _x[:, 0] * x_[:, 1],
                _x[:, 1] * _x[:, 2],       -x_[:, 0] * _x[:, 2],      -x_[:, 0] * _x[:, 1],
                _x[:, 1] * x_[:, 2],       -x_[:, 0] * x_[:, 2],       x_[:, 0] * _x[:, 1],
                 x_[:, 1] * _x[:, 2],       x_[:, 0] * _x[:, 2],      -x_[:, 0] * x_[:, 1],
                 x_[:, 1] * x_[:, 2],       x_[:, 0] * x_[:, 2],       x_[:, 0] * x_[:, 1]
            ], dim=1).to(dtype=grad_output.dtype, device=grad_output.device)
            # Shape (N, 8, 3) tensor of @(coeffs)/@(xyz)
            grad_coeffs_by_xyz = grad_coeffs_by_xyz.reshape(-1, 8, 3)
            # Shape (N, D, 8) tensor of @(feats_out)/@(coeffs)
            grad_fout_by_coeffs = feats[selected_trinkets.long()].permute(0, 2, 1)
            # Shape (N, D, 3) tensor of @(feats_out)/@(xyz), after applying chain rule
            _grad_fout_by_xyz = grad_fout_by_coeffs @ grad_coeffs_by_xyz
            # Shape (N, 1, 3) tensor of @(out)/@(xyz) applying chain rule again
            grad_fout_by_xyz = torch.zeros(
                (grad_output.shape[0], *_grad_fout_by_xyz.shape[1:]),
                device='cuda', dtype=_grad_fout_by_xyz.dtype)
            grad_fout_by_xyz[mask] = _grad_fout_by_xyz
            grad_coords = grad_output @ grad_fout_by_xyz
        return grad_coords, None, None, None, grad_feats, None

def unbatched_interpolate_trilinear(coords, pidx, point_hierarchy, trinkets, feats, level):
    r"""Performs trilinear interpolation on a SPC feature grid.

    Args:
        coords (torch.FloatTensor): 3D coordinates of shape
                                    :math:`(\text{num_coords}, \text{num_samples}, 3)`
                                    in normalized space [-1, 1]. ``num_samples`` indicates the number of
                                    coordinates that are grouped inside the same SPC node for performance
                                    optimization purposes. In many cases the ``pidx`` is
                                    generated from :func:`kaolin.ops.spc.unbatched_query`
                                    and so the ``num_samples`` will be 1.

        pidx (torch.IntTensor): Index to the point hierarchy which contains the voxel
                                which the coords exists in. Tensor of shape
                                :math:`(\text{num_coords})`.
                                This can be computed with :func:`kaolin.ops.spc.unbatched_query`.


        point_hierarchy (torch.ShortTensor):
            The point hierarchy of shape :math:`(\text{num_points}, 3)`.
            See :ref:`point_hierarchies <spc_points>` for a detailed description.

        trinkets (torch.IntTensor): An indirection pointer (in practice, an index) to the feature
                                    tensor of shape :math:`(\text{num_points}, 8)`.

        feats (torch.Tensor): Floating point feature vectors to interpolate of shape
                              :math:`(\text{num_feats}, \text{feature_dim})`.

        level (int): The level of SPC to interpolate on.

    Returns:
        (torch.FloatTensor):
            Interpolated feature vectors of shape :math:`(\text{num_voxels}, \text{num_samples}, \text{feature_dim})`.
    """
    return InterpolateTrilinear.apply(coords, pidx, point_hierarchy, trinkets, feats, level)

def coords_to_trilinear(coords, points, level):
    r"""Calculates the coefficients for trilinear interpolation.

    .. deprecated:: 0.11.0
       This function is deprecated. Use :func:`coords_to_trilinear_coeffs`.

    This calculates coefficients with respect to the dual octree, which represent the corners of the octree
    where the features are stored.

    To interpolate with the coefficients, do:
    ``torch.sum(features * coeffs, dim=-1)``
    with ``features`` of shape :math:`(\text{num_points}, 8)`

    Args:
        coords (torch.FloatTensor): 3D coordinates of shape :math:`(\text{num_coords}, 3)`
                                    in normalized space [-1, 1].
        points (torch.ShortTensor): Quantized 3D points (the 0th bit of the voxel x is in),
                                    of shape :math:`(\text{num_points}, 3)`.
        level (int): The level of SPC to interpolate on.

    Returns:
        (torch.FloatTensor):
            The trilinear interpolation coefficients of shape :math:`(\text{num_points}, 8)`.
    """
    warnings.warn("coords_to_trilinear is deprecated, "
                  "please use kaolin.ops.spc.coords_to_trilinear_coeffs instead",
                  DeprecationWarning, stacklevel=2)
    return coords_to_trilinear_coeffs(coords, points, level)

def coords_to_trilinear_coeffs(coords, points, level):
    r"""Calculates the coefficients for trilinear interpolation.

    This calculates coefficients with respect to the dual octree, which represent the corners of the octree
    where the features are stored.

    To interpolate with the coefficients, do:
    ``torch.sum(features * coeffs, dim=-1)``
    with ``features`` of shape :math:`(\text{num_points}, 8)`

    Args:
        coords (torch.FloatTensor): 3D coordinates of shape :math:`(\text{num_coords}, 3)`
                                    in normalized space [-1, 1].
        points (torch.ShortTensor): Quantized 3D points (the 0th bit of the voxel x is in),
                                    of shape :math:`(\text{num_points}, 3)`.
        level (int): The level of SPC to interpolate on.

    Returns:
        (torch.FloatTensor):
            The trilinear interpolation coefficients of shape :math:`(\text{num_coords}, 8)`.
    """
    shape = list(coords.shape)
    shape[-1] = 8
    points = points.reshape(-1, 3)
    coords = coords.reshape(-1, 3)
    coords_ = (2 ** level) * (coords * 0.5 + 0.5)

    return _C.ops.spc.coords_to_trilinear_cuda(
        coords_.contiguous(), points.contiguous()).reshape(*shape)


def create_dense_spc(level, device):
    """Creates a dense SPC model

    Args:
        level (int): The level at which the octree will be initialized to.
        device (torch.device): Torch device to keep the spc octree

    Returns:
        (torch.ByteTensor): the octree tensor
    """
    lengths = torch.tensor([sum(8 ** l for l in range(level))], dtype=torch.int32)
    octree = torch.full((lengths,), 255, device=device, dtype=torch.uint8)
    return octree, lengths
