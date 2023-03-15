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
    'feature_grids_to_spc',
    'scan_octrees',
    'generate_points',
    'to_dense',
    'unbatched_query',
    'unbatched_make_dual',
    'unbatched_make_trinkets',
    'unbatched_get_level_points'
]

import math
from torch.autograd import Function
import torch

from kaolin import _C

import math
from .uint8 import bits_to_uint8
from kaolin.rep import Spc
from .points import points_to_morton, points_to_corners


def scan_octrees(octrees, lengths):
    r"""Scan batch of octrees tensor.

    Scanning refers to processing the octrees to extract auxiliary information.

    There are two steps. First, a list is formed
    containing the number of set bits in each octree node/byte. Second, the exclusive
    sum of this list is taken.

    Args:
        octrees (torch.ByteTensor):
            Batched :ref:`packed<packed>` collection of octrees of shape :math:`(\text{num_node})`.
        lengths (torch.IntTensor):
            The number of byte per octree. of shape :math:`(\text{batch_size})`.

    Returns:
        (int, torch.IntTensor, torch.IntTensor):

            - max_level, an int containing the depth of the octrees.
            - :ref:`pyramids<spc_pyramids>`, a tensor containing structural information about
              the batch of structured point cloud hierarchies,
              of shape :math:`(\text{batch_size}, 2, \text{max_level + 1})`.
              See :ref:`the documentation <spc_pyramids>` for more details.
            - :ref:`exsum<spc_exsum>`, a 1D tensor containing the exclusive sum of the bit
              counts of each byte of the individual octrees within the batched input ``octrees`` tensor,
              of size :math:(\text{octree_num_bytes} + \text{batch_size})`.
              See :ref:`the documentation <spc_exsum>` for more details.

    .. note::

        The returned tensor of exclusive sums is padded with an extra element for each
        item in the batch.
    """
    return _C.ops.spc.scan_octrees_cuda(octrees.contiguous(), lengths.contiguous())

def generate_points(octrees, pyramids, exsum):
    r"""Generate the point data for a structured point cloud.
    Decode batched octree into batch of structured point hierarchies,
    and batch of book keeping pyramids.

    Args:
        octrees (torch.ByteTensor):
            Batched (packed) collection of octrees of shape :math:`(\text{num_bytes})`.
        pyramids (torch.IntTensor):
            Batched tensor containing point hierarchy structural information
            of shape :math:`(\text{batch_size}, 2, \text{max_level}+2)`
        exsum (torch.IntTensor):
            Batched tensor containing the exclusive sum of the bit
            counts of individual octrees of shape :math:`(k + \text{batch_size})`

    Returns:
        (torch.ShortTensor):
            A tensor containing batched point hierachies derived from a batch of octrees,
            of shape :math:`(\text{num_points_at_all_levels}, 3)`.
            See :ref:`the documentation<spc_points>` for more details
    """
    return _C.ops.spc.generate_points_cuda(octrees.contiguous(),
                                           pyramids.contiguous(),
                                           exsum.contiguous())

class ToDenseFunction(Function):
    @staticmethod
    def forward(ctx, point_hierarchies, level, pyramids, inputs):
        inputs = inputs.contiguous()
        pyramids = pyramids.contiguous()
        point_hierarchies = point_hierarchies.contiguous()

        ctx.save_for_backward(point_hierarchies, pyramids, inputs)
        ctx.level = level

        return _C.ops.spc.to_dense_forward(point_hierarchies, level, pyramids, inputs)

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.contiguous()

        point_hierarchies, pyramids, inputs = ctx.saved_tensors
        d_inputs = _C.ops.spc.to_dense_backward(point_hierarchies, ctx.level, pyramids,
                                                inputs, grad_outputs)

        return None, None, None, d_inputs

def to_dense(point_hierarchies, pyramids, input, level=-1, **kwargs):
    r"""Convert batched structured point cloud to a batched dense feature grids.

    The size of the input should correspond to level :math:`l` within the
    structured point cloud hierarchy. A dense voxel grid of size
    :math:`(\text{batch_size}, 2^l, 2^l, 2^l, \text{input_channels})` is
    returned where (for a particular batch):

    .. math::

        Y_{P_i} = X_i \quad\text{for}\; i \in 0,\ldots,|X|-1,

    where :math:`P_i` is used as a 3D index for dense array :math:`Y`, and :math:`X_i` is the
    input feature corresponding to to point :math:`P_i`. Locations in :math:`Y` without a
    correspondense in :math:`X` are set to zero.

    Args:
        point_hierarchies (torch.ShortTensor):
            :ref:`Packed <packed>` collection of point hierarchies,
            of shape :math:`(\text{num_points})`.
            See :ref:`point_hierarchies <spc_points>` for a detailed description.

        pyramids (torch.IntTensor):
            Batched tensor containing point hierarchy structural information
            of shape :math:`(\text{batch_size}, 2, \text{max_level}+2)`.
            See :ref:`pyramids <spc_pyramids>` for a detailed description.

        input (torch.FloatTensor):
            Batched tensor of input feature data,
            of shape :math:`(\text{num_inputs}, \text{feature_dim})`.
            With :math:`\text{num_inputs}` corresponding to a number of points in the
            batched point hierarchy at ``level``.

        level (int):
            The level at which the octree points are converted to feature grids.


    Returns:
        (torch.FloatTensor):
            The feature grids, of shape
            :math:`(\text{batch_size}, \text{feature_dim}, 8^\text{level}, 8^\text{level}, 8^\text{level})`.
    """
    remaining_kwargs = kwargs.keys() - Spc.KEYS
    if len(remaining_kwargs) > 0:
        raise TypeError("to_dense got an unexpected keyword argument "
                        f"{list(remaining_kwargs)[0]}")
    if level < 0:
        max_level = pyramids.shape[2] - 2
        level = max_level + 1 + level
    return ToDenseFunction.apply(point_hierarchies, level, pyramids, input)

def feature_grids_to_spc(feature_grids, masks=None):
    r"""Convert sparse feature grids to Structured Point Cloud.

    Args:
        feature_grids (torch.Tensor):
            The sparse 3D feature grids, of shape
            :math:`(\text{batch_size}, \text{feature_dim}, X, Y, Z)`
        masks (optional, torch.BoolTensor):
            The masks showing where are the features,
            of shape :math:`(\text{batch_size}, X, Y, Z)`.
            Default: A feature is determined when not full of zeros.

    Returns:
        (torch.ByteTensor, torch.IntTensor, torch.Tensor):
            a tuple containing:

                - The octree, of size :math:`(\text{num_nodes})`

                - The lengths of each octree, of size :math:`(\text{batch_size})`

                - The coalescent features, of same dtype than ``feature_grids``,
                  of shape :math:`(\text{num_features}, \text{feature_dim})`.
    """
    batch_size = feature_grids.shape[0]
    feature_dim = feature_grids.shape[1]
    x_dim = feature_grids.shape[2]
    y_dim = feature_grids.shape[3]
    z_dim = feature_grids.shape[4]
    dtype = feature_grids.dtype
    device = feature_grids.device
    feature_grids = feature_grids.permute(0, 2, 3, 4, 1)
    level = math.ceil(math.log2(max(x_dim, y_dim, z_dim)))
    # We enforce a power of 2 size to make the subdivision easier
    max_dim = 2 ** level
    padded_feature_grids = torch.zeros(
        (batch_size, max_dim, max_dim, max_dim, feature_dim),
        device=device, dtype=dtype)
    padded_feature_grids[:, :x_dim, :y_dim, :z_dim] = feature_grids
    if masks is None:
        masks = torch.any(padded_feature_grids != 0, dim=-1)
    else:
        assert masks.shape == feature_grids.shape[:-1]
        padded_masks = torch.zeros(
            (batch_size, max_dim, max_dim, max_dim),
            device=device, dtype=torch.bool)
        padded_masks[:, :x_dim, :y_dim, :z_dim] = masks
        masks = padded_masks
    bool2uint8_w = 2 ** torch.arange(8, device=device).reshape(1, 8)
    octrees = []
    coalescent_features = []
    lengths = []
    # TODO(cfujitsang): vectorize for speedup
    for bs in range(batch_size):
        octree = []
        cur_mask = masks[bs:bs + 1]
        cur_feature_grid = padded_feature_grids[bs:bs + 1]
        cur_dim = max_dim
        while cur_dim > 1:
            cur_dim = cur_dim // 2
            cur_mask = cur_mask.reshape(-1, 2, cur_dim, 2, cur_dim, 2, cur_dim)
            cur_feature_grid = cur_feature_grid.reshape(
                -1, 2, cur_dim, 2, cur_dim, 2, cur_dim, feature_dim)
            cur_level_mask = torch.sum(cur_mask, dim=(2, 4, 6)) > 0
            # indexing by masking follow naturally the morton order
            cur_feature_grid = cur_feature_grid.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
                -1, 8, cur_dim, cur_dim, cur_dim, feature_dim)[cur_level_mask.reshape(-1, 8)]
            cur_mask = cur_mask.permute(0, 1, 3, 5, 2, 4, 6).reshape(
                -1, 8, cur_dim, cur_dim, cur_dim)[cur_level_mask.reshape(-1, 8)]
            uint8_mask = bits_to_uint8(cur_level_mask.reshape(-1, 8))
            octree.append(uint8_mask)
        octree = torch.cat(octree, dim=0)
        octrees.append(octree)
        lengths.append(octree.shape[0])
        coalescent_features.append(cur_feature_grid.reshape(-1, feature_dim))
    octrees = torch.cat(octrees, dim=0)
    lengths = torch.tensor(lengths, dtype=torch.int)
    coalescent_features = torch.cat(coalescent_features, dim=0)
    return octrees, lengths, coalescent_features

def unbatched_query(octree, exsum, query_coords, level, with_parents=False):
    r"""Query point indices from the octree.

    Given a :ref:`point hierarchy<spc_points>` (implicitly encoded in ``octree``) and some coordinates, 
    this function will efficiently find the indices of the points in :ref:`point hierarchy<spc_points>` 
    corresponding to the coordinates. Returns -1 if the point does not exist.

    Args:
        octree (torch.ByteTensor): The octree, of shape :math:`(\text{num_bytes})`.
        exsum (torch.IntTensor): The exclusive sum of the octree bytes,
                                 of shape :math:`(\text{num_bytes} + 1)`.
                                 See :ref:`spc_exsum` for more details.
        query_coords (torch.FloatTensor or torch.IntTensor): 
            A tensor of locations to sample of shape :math:`(\text{num_query}, 3)`. If the tensor is
            a FloatTensor, assumes the coordinates are normalized in [-1, 1]. Otherwise if the tensor is
            an IntTensor, assumes the coordinates are in the [0, 2^level] space.
        level (int): The level of the octree to query from.
        with_parents (bool): If True, will return an array of indices up to the specified level as opposed
                             to only a single level (default: False).

    Returns:
        pidx (torch.LongTensor):

            The indices into the point hierarchy of shape :math:`(\text{num_query})`.
            If with_parents is True, then the shape will be :math:`(\text{num_query, level+1})`.

    Examples:
        >>> import kaolin
        >>> points = torch.tensor([[3,2,0],[3,1,1],[3,3,3]], device='cuda', dtype=torch.short)
        >>> octree = kaolin.ops.spc.unbatched_points_to_octree(points, 2)
        >>> length = torch.tensor([len(octree)], dtype=torch.int32)
        >>> _, _, prefix = kaolin.ops.spc.scan_octrees(octree, length)
        >>> query_coords = torch.tensor([[3,2,0]], device='cuda', dtype=torch.short)
        >>> kaolin.ops.spc.unbatched_query(octree, prefix, query_coords, 2, with_parents=False)
        tensor([5], device='cuda:0')
        >>> kaolin.ops.spc.unbatched_query(octree, prefix, query_coords, 2, with_parents=True)
        tensor([[0, 2, 5]], device='cuda:0')
    """
    if not query_coords.is_floating_point():
        input_coords = (query_coords.float() / (2**level)) * 2.0 - 1.0
    else:
        input_coords = query_coords

    if with_parents:
        return _C.ops.spc.query_multiscale_cuda(octree.contiguous(), exsum.contiguous(),
                                                input_coords.contiguous(), level).long()
    else:
        return _C.ops.spc.query_cuda(octree.contiguous(), exsum.contiguous(),
                                     input_coords.contiguous(), level).long()

def unbatched_get_level_points(point_hierarchy, pyramid, level):
    r"""Returns the point set for the given level from the point hierarchy.

    Args:
        point_hierarchy (torch.ShortTensor): 
            The point hierarchy of shape :math:`(\text{num_points}, 3)`.
            See :ref:`point_hierarchies <spc_points>` for a detailed description.

        pyramid (torch.IntTensor): 
            The pyramid of shape :math:`(2, \text{max_level}+2)`
            See :ref:`pyramids <spc_pyramids>` for a detailed description.

        level (int): The level of the point hierarchy to retrieve.

    Returns:
        (torch.ShortTensor): The pointset of shape :math:`(\text{num_points_on_level}, 3)`.
    """
    return point_hierarchy[pyramid[1, level]:pyramid[1, level + 1]]


def unbatched_make_dual(point_hierarchy, pyramid):
    r"""Creates the dual of the octree given the point hierarchy and pyramid.

    Each node of the primary octree (represented as the :ref:`point_hierarchies <spc_points>`) 
    can be thought of as voxels with 8 corners. The dual of the octree represents the corners
    of the primary octree nodes as another tree of nodes with a hierarchy of points and a pyramid. 
    The mapping from the primary octree nodes to the nodes in the dual tree can be obtained through
    trinkets which can be created from ``make_trinkets``.

    Args:
        point_hierarchy (torch.ShortTensor): 
            The point hierarchy of shape :math:`(\text{num_points}, 3)`.
            See :ref:`point_hierarchies <spc_points>` for a detailed description.

        pyramid (torch.IntTensor): 
            The pyramid of shape :math:`(2, \text{max_level}+2)`
            See :ref:`pyramids <spc_pyramids>` for a detailed description.

    Returns:
        (torch.ShortTensor, torch.IntTensor):

            - The point hierarchy of the dual octree of shape :math:`(\text{num_dual_points}, 3)`.
            - The dual pyramid of shape :math:`(2, \text{max_level}+2)`

    Examples:
        >>> import kaolin
        >>> points = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]], device='cuda', dtype=torch.int16)
        >>> level = 1
        >>> octree = kaolin.ops.spc.unbatched_points_to_octree(points, level)
        >>> length = torch.tensor([len(octree)], dtype=torch.int32)
        >>> _, pyramid, prefix = kaolin.ops.spc.scan_octrees(octree, length)
        >>> point_hierarchy = kaolin.ops.spc.generate_points(octree, pyramid, prefix)
        >>> point_hierarchy_dual, pyramid_dual = kaolin.ops.spc.unbatched_make_dual(point_hierarchy, pyramid[0])
        >>> kaolin.ops.spc.unbatched_get_level_points(point_hierarchy_dual, pyramid_dual, 0) # the corners of the root
        tensor([[0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1]], device='cuda:0', dtype=torch.int16)
        >>> kaolin.ops.spc.unbatched_get_level_points(point_hierarchy_dual, pyramid_dual, 1) # the corners of the 1st level
        tensor([[0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
                [0, 0, 2],
                [0, 1, 2],
                [1, 0, 2],
                [1, 1, 2],
                [0, 2, 0],
                [0, 2, 1],
                [1, 2, 0],
                [1, 2, 1]], device='cuda:0', dtype=torch.int16)
   """
    pyramid_dual = torch.zeros_like(pyramid)
    point_hierarchy_dual = []
    for i in range(pyramid.shape[1] - 1):
        corners = points_to_corners(unbatched_get_level_points(point_hierarchy, pyramid, i)).reshape(-1, 3)
        points_dual = torch.unique(corners, dim=0)
        sort_idxes = points_to_morton(points_dual).argsort()
        points_dual = points_dual[sort_idxes]
        point_hierarchy_dual.append(points_dual)
        pyramid_dual[0, i] = len(point_hierarchy_dual[i])
        if i > 0:
            pyramid_dual[1, i] += pyramid_dual[:, i - 1].sum()
    pyramid_dual[1, pyramid.shape[1] - 1] += pyramid_dual[:, pyramid.shape[1] - 2].sum()
    point_hierarchy_dual = torch.cat(point_hierarchy_dual, dim=0)
    return point_hierarchy_dual, pyramid_dual


def unbatched_make_trinkets(point_hierarchy, pyramid, point_hierarchy_dual, pyramid_dual):
    r"""Creates the trinkets for the dual octree.

    The trinkets are indirection pointers (in practice, indices) from the nodes of the primary octree
    to the nodes of the dual octree. The nodes of the dual octree represent the corners of the voxels
    defined by the primary octree. The trinkets are useful for accessing values stored on the corners 
    (like for example a signed distance function) and interpolating them from the nodes of the primary 
    octree.

    Args:
        point_hierarchy (torch.ShortTensor): The point hierarchy of shape :math:`(\text{num_points}, 3)`.
        pyramid (torch.IntTensor): The pyramid of shape :math:`(2, \text{max_level}+2)`
        point_hierarchy_dual (torch.ShortTensor): The point hierarchy of the dual octree of shape
                                                  :math:`(\text{num_dual_points}, 3)`.
        pyramid_dual (torch.IntTensor): The dual pyramid of shape :math:`(2, \text{max_level}+2)`

    Returns:
        (torch.IntTensor, torch.IntTensor):

            - The trinkets of shape :math:`(\text{num_points}, 8)`.
            - Indirection pointers to the parents of shape :math:`(\text{num_points})`.
    """
    device = point_hierarchy.device
    trinkets = []
    parents = []

    # At a high level... the goal of this algorithm is to create a table which maps from the primary
    # octree of voxels to the dual octree of corners, while also keeping track of parents. 
    # It does so by constructing a lookup table which maps morton codes of the source octree corners
    # to the index of the destination (dual), then using pandas to do table lookups. It's a silly
    # solution that would be much faster with a GPU but works well enough.
    for lvl in range(pyramid_dual.shape[1] - 1):
        # The source (primary octree) is sorted in morton order by construction
        points = unbatched_get_level_points(point_hierarchy, pyramid, lvl)
        corners = points_to_corners(points)
        mt_src = points_to_morton(corners.reshape(-1, 3))

        # The destination (dual octree) needs to be sorted too
        points_dual = unbatched_get_level_points(point_hierarchy_dual, pyramid_dual, lvl)
        mt_dest = points_to_morton(points_dual)

        # Uses arange to associate from the morton codes to the point index. The point index is indexed from 0.
        lut = {k: i for i, k in enumerate(mt_dest.cpu().numpy())}

        if lvl == 0:
            parents.append(torch.tensor([-1], device='cuda', dtype=torch.int).to(device))
        else:
            # Dividing by 2 will yield the morton code of the parent
            pc = torch.floor(points / 2.0).short()
            # Morton of the parents (point_hierarchy_index -> parent_morton)
            mt_pc_parent = points_to_morton(pc)
            # Morton of the children (point_hierarchy_index -> self_morton)
            mt_pc_child = points_to_morton(points)

            points_parents = unbatched_get_level_points(point_hierarchy, pyramid, lvl - 1)

            # point_hierarchy_index (i-1) -> parent_morton
            mt_parents = points_to_morton(points_parents)

            # parent_morton -> point_hierarchy_index
            plut = {k: i for i, k in enumerate(mt_parents.cpu().numpy())}
            pc_idx = [plut[i] for i in mt_pc_parent.cpu().numpy()]
            parents.append(torch.tensor(pc_idx, device=device, dtype=torch.int) +
                           pyramid[1, lvl - 1])

        idx = [lut[i] for i in mt_src.cpu().numpy()]
        trinkets.extend(idx)

    # Trinkets are relative to the beginning of each pyramid base
    trinkets = torch.tensor(trinkets, device=device, dtype=torch.int).reshape(-1, 8)
    parents = torch.cat(parents, dim=0)
    return trinkets, parents
