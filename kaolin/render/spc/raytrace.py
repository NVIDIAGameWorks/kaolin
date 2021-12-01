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

def mark_pack_boundary(pack_ids):
    r"""Mark the boundaries of pack IDs.

    Pack IDs are sorted tensors which mark the ID of the pack each element belongs in.

    For example, the SPC ray trace kernel will return the ray index tensor which marks the ID of the ray
    that each intersection belongs in. This kernel will mark the beginning of each of those packs of
    intersections with a boolean mask (`True` where the beginning is).

    Args:
        pack_ids (torch.Tensor): pack ids of shape :math:`(\text{num_elems})`
                                 This can be any integral (n-bit integer) type.
    Returns:
        first_hits (torch.BoolTensor): the boolean mask marking the boundaries.
    """
    return _C.render.spc.mark_pack_boundary_cuda(pack_ids.contiguous()).bool()

def mark_first_hit(ridx):
    r"""Mark the first hit in the nuggets.

    .. deprecated:: 0.10.0
       This function is deprecated. Use `mark_pack_boundary`.

    The nuggets are a packed tensor containing correspondences from ray index to point index, sorted
    within each ray pack by depth. This will mark True for each first hit (by depth) for a pack of
    nuggets.

    Returns:
        first_hits (torch.BoolTensor): the boolean mask marking the first hit by depth.
    """
    warnings.warn("mark_first_hit has been deprecated, please use mark_pack_boundary instead")
    return mark_pack_boundary(ridx)
