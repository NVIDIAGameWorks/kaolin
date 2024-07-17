# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
import logging

import torch
import numpy as np
import kaolin.ops.spc as spc
from kaolin import _C

__all__ = [
    'bf_recon',
    'unbatched_query'
]

logger = logging.getLogger(__name__)

def processFrame(batch, level, sigma):
    im = batch[0].contiguous()
    dm = batch[1].contiguous()
    Cam = batch[2]
    In = batch[3]
    maxdepth = batch[4]
    mip_levels = batch[5]
    true_depth = batch[6]
    start_level = batch[7]
    points = batch[8]
    device = points.device

    # generate depth minmaxmip
    mips = _C.ops.spc.build_mip2d(dm, In, mip_levels, maxdepth, true_depth).contiguous()

    # list for intermediate tensors
    Occ = []
    Sta = []
    Nvs = []

    # number of points tested each level
    vcnt = np.zeros((level+1), dtype=np.uint32)

    # initialize levels above start level as dense
    for l in range(0, start_level):
        num = pow(8,l)
        Occ.append(torch.ones((num), dtype=torch.int32, device=device))
        Sta.append(torch.full((num,), 2, dtype=torch.int32, device=device))
        Nvs.append(torch.arange(num, dtype=torch.int32, device=device))
        vcnt[l] = num

    for l in range(start_level, level):
        vcnt[l] = points.size(0)
        occupancies, empty_state = _C.ops.spc.oracleB(points, l, sigma, Cam, dm, mips, mip_levels)
        insum = _C.ops.spc.inclusive_sum(occupancies)
        if insum[-1].item() == 0:
            logger.debug("recon terminated")
        points, nvsum = _C.ops.spc.subdivide2(points, insum)

        Occ.append(occupancies)
        Sta.append(empty_state)
        Nvs.append(nvsum)

    vcnt[level] = points.size(0)

    occupancies, empty_state, probabilities = _C.ops.spc.oracleB_final(points, level, sigma, Cam, dm)
    insum = _C.ops.spc.inclusive_sum(occupancies)
    if insum[-1].item() == 0:
        logger.debug("recon terminated")
    points, nvsum = _C.ops.spc.compactify2(points, insum)
    probabilities = probabilities[nvsum[:]]

    colors, normals = _C.ops.spc.colorsB_final(points, level, Cam, sigma, im, dm, probabilities)

    Occ.append(occupancies)
    Sta.append(empty_state)

    # process final voxels
    init_size = vcnt[1:].sum().item() >> 3
    octree = torch.zeros((init_size), dtype=torch.uint8, device=device)
    empty = torch.zeros((init_size), dtype=torch.uint8, device=device)
    occupancies = torch.zeros((init_size), dtype=torch.int32, device=device)

    num_voxels = vcnt[level].item()
    num_nodes = num_voxels >> 3
    total_nodes = num_nodes

    for l in range(level, 0, -1):
        octree, empty, occupancies = \
            _C.ops.spc.process_final_voxels(
                num_nodes, total_nodes, Sta[l], Nvs[l-1], occupancies, Sta[l-1], octree, empty
            )

        num_voxels = vcnt[l-1].item()
        num_nodes = num_voxels >> 3
        total_nodes += num_nodes

    insum = _C.ops.spc.inclusive_sum(occupancies)
    octree, empty = _C.ops.spc.compactify_nodes(total_nodes, insum, octree, empty)

    return octree, empty, probabilities, colors, normals


def fuseBF(
    level,
    octree0, octree1,
    empty0, empty1,
    probs0, probs1,
    colors0, colors1,
    normals0, normals1,
    pyramid0, pyramid1,
    exsum0, exsum1
):
    device = octree0.device

    # start at level 
    start_level = 4

    points = spc.morton_to_points(torch.arange(pow(8, start_level)).to(device))

    # list for intermediate tensors
    Sta = []
    Nvs = []

    # number of points tested each level
    vcnt = np.zeros((level+1), dtype=np.uint32)

    # initialize levels above start level as dense
    for l in range(0, start_level):
        num = pow(8,l)
        Sta.append(torch.full((num,), 2, dtype=torch.int32, device=device))
        Nvs.append(torch.arange(num, dtype=torch.int32, device=device))
        vcnt[l] = num

    for l in range(start_level, level):
        vcnt[l] = points.size(0)
        occupancies, empty_state = \
            _C.ops.spc.merge_empty(points, l, octree0, octree1, empty0, empty1, pyramid0, pyramid1, exsum0, exsum1)
        if occupancies.max().item() == 0:
            logger.debug("recon terminated")
        insum = _C.ops.spc.inclusive_sum(occupancies)
        points, nvsum = _C.ops.spc.subdivide2(points, insum)

        Sta.append(empty_state)
        Nvs.append(nvsum)

    vcnt[level] = points.size(0)
    occupancies, empty_state, occ_prob, colors, normals = _C.ops.spc.bq_merge(points, level,
            octree0, octree1, 
            empty0, empty1, 
            probs0, probs1, 
            colors0, colors1, 
            normals0, normals1,
            pyramid0, pyramid1, 
            exsum0, exsum1)

    if occupancies.max().item() == 0:
        logger.debug("recon terminated")
    insum = _C.ops.spc.inclusive_sum(occupancies)
    points, nvsum = _C.ops.spc.compactify2(points, insum)

    occ_prob = occ_prob[nvsum[:]]
    colors = colors[nvsum[:]]
    Sta.append(empty_state)

    # process final voxels
    init_size = vcnt[1:].sum().item() >> 3
    octree = torch.zeros((init_size), dtype=torch.uint8, device=device)
    empty = torch.zeros((init_size), dtype=torch.uint8, device=device)
    occupancies = torch.zeros((init_size), dtype=torch.int32, device=device)

    num_voxels = vcnt[level].item()
    num_nodes = num_voxels >> 3
    total_nodes = num_nodes

    for l in range(level, 0, -1):
        octree, empty, occupancies =\
            _C.ops.spc.process_final_voxels(
                num_nodes, total_nodes, Sta[l], Nvs[l-1], occupancies, Sta[l-1], octree, empty
            )

        insum = _C.ops.spc.inclusive_sum(occupancies)
        if insum[-1].item() == 0:
            logger.debug("recon terminated")

        num_voxels = vcnt[l-1].item()
        num_nodes = num_voxels >> 3
        total_nodes += num_nodes

    insum = _C.ops.spc.inclusive_sum(occupancies)
    if insum[-1].item() == 0:
        logger.debug("recon terminated")
    octree, empty = _C.ops.spc.compactify_nodes(total_nodes, insum, octree, empty)

    return octree, empty, occ_prob, colors, normals


def extractBQ(octree, empty, probs, colors):
    device = octree.device
    lengths = torch.tensor([len(octree)], dtype=torch.int)
    level, pyramid, exsum = spc.scan_octrees(octree, lengths)

    # number of points tested each level
    vcnt = np.zeros((level+1), dtype=np.uint32)

    # list for intermediate tensors
    Sta = [torch.full((1,), 2, dtype=torch.int32, device=device)]
    Nvs = [torch.arange(1, dtype=torch.int32, device=device)]

    points = spc.morton_to_points(torch.arange(pow(8, 1)).to(device))
    vcnt[0] = pyramid[0,0,0] # = 1

    for l in range(1, level):
        occupancies, empty_state = _C.ops.spc.bq_touch(points, l, octree, empty, pyramid)
        if occupancies.max().item() == 0:
            logger.debug("recon terminated")
        insum = _C.ops.spc.inclusive_sum(occupancies)
        points, nvsum = _C.ops.spc.subdivide2(points, insum)

        vcnt[l] = insum.size(0)
        Sta.append(empty_state)
        Nvs.append(nvsum)

    vcnt[level] = points.size(0)

    occupancies, empty_state = _C.ops.spc.bq_touch(points, level, octree, empty, pyramid)
    insum = _C.ops.spc.inclusive_sum(occupancies)
    if insum[-1].item() == 0:
        logger.debug("recon terminated")
    points, nvsum = _C.ops.spc.compactify2(points, insum)

    Sta.append(empty_state)

    newsum = torch.empty_like(nvsum, dtype=torch.int32)
    newsum[:] = nvsum[:]
    Nvs.append(newsum)

    occupancies, empty_state = _C.ops.spc.bq_extract(points, level, octree, empty, probs, pyramid, exsum)
    insum = _C.ops.spc.inclusive_sum(occupancies)
    if insum[-1].item() == 0:
        logger.debug("recon terminated")
    points, nvsum = _C.ops.spc.compactify2(points, insum)
    out_colors = colors[nvsum[:]]
 
    num_voxels = empty_state.size(0)
    _C.ops.spc.bq_touch_extract(num_voxels, empty_state, Nvs[level], Sta[level])

    # process final voxels
    init_size = vcnt[1:].sum().item() >> 3
    octree = torch.zeros((init_size), dtype=torch.uint8, device=device)
    empty = torch.zeros((init_size), dtype=torch.uint8, device=device)
    occupancies = torch.zeros((init_size), dtype=torch.int32, device=device)

    num_voxels = vcnt[level]
    num_nodes = num_voxels >> 3
    total_nodes = num_nodes

    for l in range(level, 0, -1):
        octree, empty, occupancies =\
            _C.ops.spc.process_final_voxels(
                num_nodes, total_nodes, Sta[l], Nvs[l-1], occupancies, Sta[l-1], octree, empty
            )
        insum = _C.ops.spc.inclusive_sum(occupancies)
        if insum[-1].item() == 0:
            logger.debug("recon terminated")

        num_voxels = vcnt[l-1].item()
        num_nodes = num_voxels >> 3
        total_nodes += num_nodes

    insum = _C.ops.spc.inclusive_sum(occupancies)
    if insum[-1].item() == 0:
        logger.debug("recon terminated")
    octree, empty = _C.ops.spc.compactify_nodes(total_nodes, insum, octree, empty)

    return octree, empty, out_colors


def bf_recon(transformed_dataset, final_level, sigma):
    frame_no = 0
    octree0, empty0, probs0, colors0, normals0, pyramid0, exsum0, weights = \
        None, None, None, None, None, None, None, None

    for batch in transformed_dataset:
        if frame_no == 0:
            octree0, empty0, probs0, colors0, normals0 = processFrame(batch, final_level, sigma)
            lengths = torch.tensor([len(octree0)], dtype=torch.int)
            level, pyramid0, exsum0 = spc.scan_octrees(octree0, lengths)
        else :
            octree1, empty1, probs1, colors1, normals1 = processFrame(batch, final_level, sigma)
            lengths = torch.tensor([len(octree1)], dtype=torch.int)
            level, pyramid1, exsum1 = spc.scan_octrees(octree1, lengths)
            octree0, empty0, probs0, colors0, normals0 = fuseBF(level,
                                                                octree0, octree1, 
                                                                empty0, empty1, 
                                                                probs0, probs1, 
                                                                colors0, colors1, 
                                                                normals0, normals1, 
                                                                pyramid0, pyramid1, 
                                                                exsum0, exsum1)
            lengths = torch.tensor([len(octree0)], dtype=torch.int)
            level, pyramid0, exsum0 = spc.scan_octrees(octree0, lengths)

        frame_no += 1
    octree, empty, colors = extractBQ(octree0, empty0, probs0, colors0)
    return octree, empty, colors


def unbatched_query(octree, empty, exsum, query_coords, level):
    r"""Query point indices from the octree and empty.

    Given a :ref:`point hierarchy<spc_points>` (implicitly encoded in ``octree``) and some coordinates, 
    this function will efficiently find the indices of the points in :ref:`point hierarchy<spc_points>` 
    corresponding to the coordinates. Returns index of point in hierarchy if found, -1 if the point does not exist 
    but is in outside the object, and < -1 if inside the object.

    Args:
        octree (torch.ByteTensor): The octree, of shape :math:`(\text{num_bytes})`.
        empty (torch.ByteTensor): The empty octree encoding state of empty space, of shape :math:`(\text{num_bytes})`.
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

    """

    if not query_coords.is_floating_point():
        input_coords = (query_coords.float() / (2**level)) * 2.0 - 1.0
    else:
        input_coords = query_coords

    return _C.ops.spc.query_cuda_empty(octree.contiguous(), empty.contiguous(), exsum.contiguous(),
                                       input_coords.contiguous(), level).long()
