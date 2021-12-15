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

import warnings
from kaolin import _C
import torch

__all__ = [
    'unbatched_raytrace',
    'mark_pack_boundaries',
    'mark_first_hit',
    'diff',
    'sum_reduce',
    'cumsum',
    'cumprod',
    'exponential_integration'
]

def unbatched_raytrace(octree, point_hierarchy, pyramid, exsum, origin, direction, level,
                       return_depth=True, with_exit=False):
    r"""Apply ray tracing over an unbatched SPC structure.

    The SPC model will be always normalized between -1 and 1 for each axis.

    Args:
        octree (torch.ByteTensor): the octree structure,
                                   of shape :math:`(\text{num_bytes})`.
        point_hierarchy (torch.ShortTensor): the point hierarchy associated to the octree,
                                             of shape :math:`(\text{num_points}, 3)`.
        pyramid (torch.IntTensor): the pyramid associated to the octree,
                                   of shape :math:`(2, \text{max_level} + 2)`.
        exsum (torch.IntTensor): the prefix sum associated to the octree.
                                 of shape :math:`(\text{num_bytes} + \text{batch_size})`.
        origin (torch.FloatTensor): the origins of the rays,
                                    of shape :math:`(\text{num_rays}, 3)`.
        direction (torch.FloatTensor): the directions of the rays,
                                       of shape :math:`(\text{num_rays}, 3)`.
        level (int): level to use from the octree.

        return_depth (bool): return the depth of each voxel intersection. (Default: True)

        with_exit (bool): return also the exit intersection depth. (Default: False)

    Returns:
        (torch.IntTensor, torch.IntTensor, (optional) torch.FloatTensor):

            - Ray index of intersections sorted by depth of shape :math:`(\text{num_intersection})` 
            - Point hierarchy index of intersections sorted by depth of shape :math:`(\text{num_intersection})` 
              These indices will be `IntTensor`s, but they can be used for indexing with `torch.index_select`.
            - If return_depth is true:
              Float tensor of shape :math:`(\text{num_intersection}), 1` of entry
              depths to each AABB intersection. When `with_exit` is set, returns 
              shape :math:`(\text{num_intersection}), 2` of entry and exit depths.
    """
    output = _C.render.spc.raytrace_cuda(
        octree.contiguous(),
        point_hierarchy.contiguous(),
        pyramid.contiguous(),
        exsum.contiguous(),
        origin.contiguous(),
        direction.contiguous(),
        level,
        return_depth,
        with_exit)
    nuggets = output[0]
    ray_index = nuggets[..., 0]
    point_index = nuggets[..., 1]

    if return_depth:
        return ray_index, point_index, output[1]
    else:
        return ray_index, point_index

def mark_pack_boundaries(pack_ids):
    r"""Mark the boundaries of pack IDs.

    Pack IDs are sorted tensors which mark the ID of the pack each element belongs in.

    For example, the SPC ray trace kernel will return the ray index tensor which marks the ID of the ray
    that each intersection belongs in. This kernel will mark the beginning of each of those packs of
    intersections with a boolean mask (true where the beginning is).

    Args:
        pack_ids (torch.Tensor): pack ids of shape :math:`(\text{num_elems})`
                                 This can be any integral (n-bit integer) type.
    Returns:
        first_hits (torch.BoolTensor): the boolean mask marking the boundaries.

    Examples:
        >>> pack_ids = torch.IntTensor([1,1,1,1,2,2,2]).to('cuda:0')
        >>> mark_pack_boundaries(pack_ids)
        tensor([ True, False, False, False,  True, False, False], device='cuda:0')
    """
    return _C.render.spc.mark_pack_boundaries_cuda(pack_ids.contiguous()).bool()

def mark_first_hit(ridx):
    r"""Mark the first hit in the nuggets.

    .. deprecated:: 0.10.0
       This function is deprecated. Use :func:`mark_pack_boundaries`.

    The nuggets are a packed tensor containing correspondences from ray index to point index, sorted
    within each ray pack by depth. This will mark true for each first hit (by depth) for a pack of
    nuggets.

    Returns:
        first_hits (torch.BoolTensor): the boolean mask marking the first hit by depth.
    """
    warnings.warn("mark_first_hit has been deprecated, please use mark_pack_boundaries instead")
    return mark_pack_boundaries(ridx)

def diff(feats, boundaries):
    r"""Find the delta between each of the features in a pack.

    The deltas are given by `out[i] = feats[i+1] - feats[i]`

    The behavior is similar to :func:`torch.diff` for non-packed tensors, but :func:`torch.diff` 
    will reduce the number of features by 1. This function will instead populate the last diff with 0.

    Args:
        feats (torch.FloatTensor): features of shape :math:`(\text{num_rays}, \text{num_feats})`
        boundaries (torch.BoolTensor): bools of shape :math:`(\text{num_rays})`
            Given some index array marking the pack IDs, the boundaries can be calculated with
            :func:`mark_pack_boundaries`
    Returns:
        (torch.FloatTensor): diffed features of shape :math:`(\text{num_rays}, \text{num_feats})`
    """

    feats_shape = feats.shape

    feat_dim = feats.shape[-1]

    pack_idxes = torch.nonzero(boundaries).contiguous()[..., 0]

    return _C.render.spc.diff_cuda(feats.reshape(-1, feat_dim).contiguous(), pack_idxes.contiguous()).reshape(*feats_shape)

class SumReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feats, info):
        inclusive_sum = _C.render.spc.inclusive_sum_cuda(info.int())
        ctx.save_for_backward(inclusive_sum)
        return _C.render.spc.sum_reduce_cuda(feats, inclusive_sum.contiguous())

    @staticmethod 
    def backward(ctx, grad_output):
        inclusive_sum = ctx.saved_tensors[0]
        grad_feats = None
        if ctx.needs_input_grad[0]:
            grad_feats = grad_output[(inclusive_sum - 1).long()]
        return grad_feats, None

class Cumprod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feats, info, exclusive, reverse):
        nonzero = torch.nonzero(info).int().contiguous()[..., 0]
        prod = _C.render.spc.cumprod_cuda(feats, nonzero, exclusive, reverse)
        ctx.save_for_backward(feats, nonzero, prod)
        ctx.flags = (exclusive, reverse)
        return prod

    @staticmethod
    def backward(ctx, grad_output):

        prod = ctx.saved_tensors
        feats, nonzero, prod = ctx.saved_tensors
        exclusive, reverse = ctx.flags
        out = _C.render.spc.cumsum_cuda(prod * grad_output, nonzero, exclusive, not reverse)

        grad_feats = None
        if ctx.needs_input_grad[0]:
            # Approximate gradient (consistent with TensorFlow)
            grad_feats = out / feats
            grad_feats[grad_feats.isnan()] = 0

        return grad_feats, None, None, None

class Cumsum(torch.autograd.Function):

    @staticmethod
    def forward(ctx, feats, info, exclusive, reverse):
        nonzero = torch.nonzero(info).int().contiguous()[..., 0]
        ctx.save_for_backward(nonzero)
        ctx.flags = (exclusive, reverse)
        cumsum = _C.render.spc.cumsum_cuda(feats, nonzero, exclusive, reverse)
        return cumsum

    @staticmethod
    def backward(ctx, grad_output):
        nonzero, = ctx.saved_tensors
        exclusive, reverse = ctx.flags
        cumsum = _C.render.spc.cumsum_cuda(grad_output.contiguous(), nonzero, exclusive, not reverse)
        return cumsum, None, None, None

def sum_reduce(feats, boundaries):
    r"""Sum the features of packs.

    Args:
        feats (torch.FloatTensor): features of shape :math:`(\text{num_rays}, \text{num_feats})`.
        boundaries (torch.BoolTensor): bools to mark pack boundaries of shape :math:`(\text{num_rays})`.
            Given some index array marking the pack IDs, the boundaries can be calculated with
            :func:`mark_pack_boundaries`.
    Returns:
        (torch.FloatTensor): summed features of shape :math:`(\text{num_packs}, \text{num_feats})`.
    """
    return SumReduce.apply(feats.contiguous(), boundaries.contiguous())

def cumsum(feats, boundaries, exclusive=False, reverse=False):
    r"""Cumulative sum across packs of features.

    This function is similar to :func:`tf.math.cumsum` with the same options, but for packed tensors.
    Refer to the TensorFlow docs for numerical examples of the options.

    Args:
        feats (torch.FloatTensor): features of shape :math:`(\text{num_rays}, \text{num_feats})`.
        boundaries (torch.BoolTensor): bools of shape :math:`(\text{num_rays})`.
            Given some index array marking the pack IDs, the boundaries can be calculated with
            :func:`mark_pack_boundaries`.
        exclusive (bool): Compute exclusive cumsum if true. Exclusive means the current index won't be used
                        for the calculation of the cumulative sum. (Default: False)
        reverse (bool): Compute reverse cumsum if true, i.e. the cumulative sum will start from the end of 
                        each pack, not from the beginning. (Default: False)
    Returns:
        (torch.FloatTensor): features of shape :math:`(\text{num_rays}\, \text{num_feats})`.
    """
    return Cumsum.apply(feats.contiguous(), boundaries.contiguous(), exclusive, reverse)

def cumprod(feats, boundaries, exclusive=False, reverse=False):
    r"""Cumulative product across packs of features.

    This function is similar to :func:`tf.math.cumprod` with the same options, but for packed tensors.
    Refer to the TensorFlow docs for numerical examples of the options.

    Note that the backward gradient follows the same behaviour in TensorFlow, which is to
    replace NaNs by zeros, which is different from the behaviour in PyTorch. To be safe,
    add an epsilon to feats which will make the behaviour consistent.

    Args:
        feats (torch.FloatTensor): features of shape :math:`(\text{num_rays}, \text{num_feats})`.
        boundaries (torch.BoolTensor): bools of shape :math:`(\text{num_rays})`.
            Given some index array marking the pack IDs, the boundaries can be calculated with
            :func:`mark_pack_boundaries`.
        exclusive (bool): Compute exclusive cumprod if true. Exclusive means the current index won't be used
                        for the calculation of the cumulative product. (Default: False)
        reverse (bool): Compute reverse cumprod if true, i.e. the cumulative product will start from the end of 
                        each pack, not from the beginning. (Default: False)
    Returns:
        (torch.FloatTensor): features of shape :math:`(\text{num_rays}, \text{num_feats})`.
    """
    return Cumprod.apply(feats.contiguous(), boundaries.contiguous(), exclusive, reverse)

def exponential_integration(feats, tau, boundaries, exclusive=True):
    r"""Exponential transmittance integration across packs using the optical thickness (tau).

    Exponential transmittance is derived from the Beer-Lambert law. Typical implementations of
    exponential transmittance is calculated with :func:`cumprod`, but the exponential allows a reformulation
    as a :func:`cumsum` which its gradient is more stable and faster to compute. We opt to use the :func:`cumsum`
    formulation.

    For more details, we recommend "Monte Carlo Methods for Volumetric Light Transport" by Novak et al.

    Args:
        feats (torch.FloatTensor): features of shape :math:`(\text{num_rays}, \text{num_feats})`.
        tau (torch.FloatTensor): optical thickness of shape :math:`(\text{num_rays}, 1)`.
        boundaries (torch.BoolTensor): bools of shape :math:`(\text{num_rays})`.
            Given some index array marking the pack IDs, the boundaries can be calculated with
            :func:`mark_pack_boundaries`.
        exclusive (bool): Compute exclusive exponential integration if true. (default: True)

    Returns:
        (torch.FloatTensor, torch.FloatTensor)
        - Integrated features of shape :math:`(\text{num_packs}, \text{num_feats})`.
        - Transmittance of shape :math:`(\text{num_rays}, 1)`.

    """
    # TODO(ttakikawa): This should be a fused kernel... we're iterating over packs, so might as well
    #                  also perform the integration in the same manner.
    alpha = 1.0 - torch.exp(-tau.contiguous())
    # Uses the reformulation as a cumsum and not a cumprod (faster and more stable gradients)
    transmittance = torch.exp(-1.0 * cumsum(tau.contiguous(), boundaries.contiguous(), exclusive=exclusive))
    transmittance = transmittance * alpha
    feats_out = sum_reduce(transmittance * feats.contiguous(), boundaries.contiguous())
    return feats_out, transmittance
