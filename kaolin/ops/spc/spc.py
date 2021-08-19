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
    'feature_grids_to_spc',
    'scan_octrees',
    'generate_points',
    'to_dense',
    'unbatched_query',
]

import math
from torch.autograd import Function
import torch

from kaolin import _C
from .uint8 import bits_to_uint8
from kaolin.rep import Spc


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

            - An int containing the depth of the octrees.

            - A tensor containing structural information about the batch of structured point cloud hierarchies,
              see :ref:`pyramids example <spc_pyramids>`.
            - A tensor containing the exclusive sum of the bit
              counts of each byte of the individual octrees within the batched input ``octrees`` tensor,
              see :ref:`exsum <spc_exsum>`.

    .. note::

        The returned tensor of exclusive sums is padded with an extra element for each
        item in the batch.
    """
    return _C.ops.spc.ScanOctrees(octrees.contiguous(), lengths.contiguous())

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
        (torch.Tensor);
            A tensor containing batched point hierachies derived from a batch of octrees
    """
    return _C.ops.spc.GeneratePoints(octrees.contiguous(),
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
            The number of inputs, :math:`\text{num_inputs}`,
            must correspond to number of points in the
            batched point hierarchy at `level`.

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
            The masks showing where are the features.
            Default: A feature is determined when not full or zeros.

    Returns:
        (torch.ByteTensor, torch.IntTensor, torch.Tensor):
            a tuple containing:

                - The octree, of size :math:`(\text{num_nodes})`

                - The lengths of each octree, of size :math:`(\text{batch_size})`

                - The coalescent features, of same dtype than `feature_grids`,
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

def unbatched_query(octree, point_hierarchy, pyramid, exsum, query_points, level):
    r"""Query point indices from the octree.

    Given a point hierarchy, this function will efficiently find the corresponding indices of the
    points in the points tensor. For each input in query_points, returns a index to the points tensor.

    Args:
        octree (torch.ByteTensor): The octree, of shape :math:`(\text{num_bytes})`.
        point_hierarchy (torch.ShortTensor):
            The points hierarchy, of shape :math:`(\text{num_points}, 3)`.
            See :ref:`spc_points` for more details.
        pyramid (torch.IntTensor): The pyramid info of the point hierarchy,
                                   of shape :math:`(2, \text{max_level} + 2)`.
                                   See :ref:`spc_pyramids` for more details.
        exsum (torch.IntTensor): The exclusive sum of the octree bytes,
                                 of shape :math:`(\text{num_bytes} + 1)`.
                                 See :ref:`spc_pyramids` for more details.
        query_points (torch.ShortTensor): A collection of query indices,
                                          of shape :math:`(\text{num_query}, 3)`.
        level (int): The level of the octree to query from.
    """
    return _C.ops.spc.spc_query(octree.contiguous(), point_hierarchy.contiguous(),
                                pyramid.contiguous(), exsum.contiguous(),
                                query_points.contiguous(), level)
