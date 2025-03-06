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

import torch
import numpy as np
import kaolin.ops.spc as spc
from kaolin import _C


class BFReconstructionTerminatedException(Exception):
    """ Exception that internally rises when Bayesian-Fusion fails due to bad inputs. """
    pass

def processFrame(batch, level, sigma):
    image = batch[0].contiguous()
    depth_map = batch[1].contiguous()
    camera = batch[2]
    intrinsics = batch[3]
    max_depth = batch[4]
    mip_levels = batch[5]
    true_depth = batch[6]
    start_level = batch[7]
    points = batch[8]
    device = points.device

    # generate depth minmaxmip
    mips = _C.ops.spc.build_mip2d(depth_map, intrinsics, mip_levels, max_depth, true_depth).contiguous()

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
        occupancies, empty_state = _C.ops.spc.oracleB(points, l, sigma, camera, depth_map, mips, mip_levels)

        s = occupancies.sum()

        insum = _C.ops.spc.inclusive_sum(occupancies)
        if insum[-1].item() == 0:
            raise BFReconstructionTerminatedException()

        points, nvsum = _C.ops.spc.subdivide(points, insum)

        Occ.append(occupancies)
        Sta.append(empty_state)
        Nvs.append(nvsum)

    vcnt[level] = points.size(0)

    occupancies, empty_state, probabilities = _C.ops.spc.oracleB_final(points, level, sigma, camera, depth_map)
    insum = _C.ops.spc.inclusive_sum(occupancies)
    if insum[-1].item() == 0:
        raise BFReconstructionTerminatedException()
    points, nvsum = _C.ops.spc.compactify(points, insum)
    probabilities = probabilities[nvsum[:]]

    colors, normals = _C.ops.spc.colorsB_final(points, level, camera, sigma, image, depth_map, probabilities)   

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
    if insum[-1].item() == 0:
        raise BFReconstructionTerminatedException()

    octree, empty = _C.ops.spc.compactify_nodes(total_nodes, insum, octree, empty)

    length = torch.tensor([len(octree)], dtype=torch.int)
    level, pyramid, exsum = spc.scan_octrees(octree, length)

    return {
        'octree' : octree,
        'empty' : empty,
        'level' : level,
        'pyramid' : pyramid,
        'exsum' : exsum,
        'probabilities' : probabilities,
        'colors' : colors,
        'normals' : normals
    }


def fuseBF(spc0, spc1):
    device = spc0['octree'].device

    # start at level 
    start_level = 4
    points = spc.morton_to_points(torch.arange(pow(8, start_level)).to(device))

    level = spc0['level']

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
            _C.ops.spc.merge_empty(points, l, 
                                   spc0['octree'], spc1['octree'], 
                                   spc0['empty'], spc1['empty'], 
                                   spc0['exsum'], spc1['exsum'])
        if occupancies.max().item() == 0:
            raise BFReconstructionTerminatedException()
        insum = _C.ops.spc.inclusive_sum(occupancies)
        points, nvsum = _C.ops.spc.subdivide(points, insum)

        Sta.append(empty_state)
        Nvs.append(nvsum)

    vcnt[level] = points.size(0)
    occupancies, empty_state, probabilities, colors, normals = _C.ops.spc.bq_merge(
        points, level, 
        spc0['octree'], spc1['octree'], 
        spc0['empty'], spc1['empty'], 
        spc0['pyramid'], spc1['pyramid'], 
        spc0['probabilities'], spc1['probabilities'], 
        spc0['colors'], spc1['colors'], 
        spc0['normals'], spc1['normals'],
        spc0['exsum'], spc1['exsum'])

    if occupancies.max().item() == 0:
        raise BFReconstructionTerminatedException()
    insum = _C.ops.spc.inclusive_sum(occupancies)
    points, nvsum = _C.ops.spc.compactify(points, insum)

    probabilities = probabilities[nvsum[:]]
    colors = colors[nvsum[:]]
    normals = normals[nvsum[:]]

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
            raise BFReconstructionTerminatedException()

        num_voxels = vcnt[l-1].item()
        num_nodes = num_voxels >> 3
        total_nodes += num_nodes

    insum = _C.ops.spc.inclusive_sum(occupancies)
    if insum[-1].item() == 0:
        raise BFReconstructionTerminatedException()
    octree, empty = _C.ops.spc.compactify_nodes(total_nodes, insum, octree, empty)

    length = torch.tensor([len(octree)], dtype=torch.int)
    level, pyramid, exsum = spc.scan_octrees(octree, length)

    return {
        'octree' : octree,
        'empty' : empty,
        'level' : level,
        'pyramid' : pyramid,
        'exsum' : exsum,
        'probabilities' : probabilities,
        'colors' : colors,
        'normals' : normals
    }

def extractBQ(spcd):
    octree = spcd['octree']
    empty = spcd['empty']
    probs = spcd['probabilities']
    colors = spcd['colors']
    normals = spcd['normals']
    level = spcd['level']
    pyramid = spcd['pyramid']
    exsum = spcd['exsum']

    device = octree.device

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
            raise BFReconstructionTerminatedException()
        insum = _C.ops.spc.inclusive_sum(occupancies)
        points, nvsum = _C.ops.spc.subdivide(points, insum)

        vcnt[l] = insum.size(0)
        Sta.append(empty_state)
        Nvs.append(nvsum)

    vcnt[level] = points.size(0)

    occupancies, empty_state = _C.ops.spc.bq_touch(points, level, octree, empty, pyramid)
    insum = _C.ops.spc.inclusive_sum(occupancies)
    if insum[-1].item() == 0:
        raise BFReconstructionTerminatedException()
    points, nvsum = _C.ops.spc.compactify(points, insum)

    Sta.append(empty_state)

    newsum = torch.empty_like(nvsum, dtype=torch.int32)
    newsum[:] = nvsum[:]
    Nvs.append(newsum)

    occupancies, empty_state = _C.ops.spc.bq_extract(points, level, octree, empty, probs, pyramid, exsum)
    insum = _C.ops.spc.inclusive_sum(occupancies)
    if insum[-1].item() == 0:
        raise BFReconstructionTerminatedException()
    points, nvsum = _C.ops.spc.compactify(points, insum)
    out_colors = colors[nvsum[:]]
    out_normals = normals[nvsum[:]]
 
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
            raise BFReconstructionTerminatedException()

        num_voxels = vcnt[l-1].item()
        num_nodes = num_voxels >> 3
        total_nodes += num_nodes

    insum = _C.ops.spc.inclusive_sum(occupancies)
    if insum[-1].item() == 0:
        raise BFReconstructionTerminatedException()
    octree, empty = _C.ops.spc.compactify_nodes(total_nodes, insum, octree, empty)

    return octree, empty, out_colors, out_normals


def bf_recon(input_dataset, final_level, sigma):
    r""" Reconstruct an object from a collection of calibrated RGBD images.

    .. note::
        For more details, see the 3DV 2016 paper
        `A Closed-Form Bayesian Fusion Equation Using Occupancy Probabilities`_.

    The object is represented by an empty space aware Structured Point Cloud. 
    That is, an octree tensor, and a corresponding empty space tensor. Taken together,
    each octree bit and corresponding empty space bit represent the occupancy state
    of the octree cell. These states are: inside, outside, and occupied. This augmented SPC
    can be queried at arbirary points in space to determine their state.

    Args:
        input_dataset (RayTracedSPCDataset): dataset containing calibrated rgbd images
        final_level (int) Desired depth of output SPC
        sigma (float) parameter that rough equates to noise level of input depths

    Return:
        (torch.ByteTensor) octree representing the geometry of reconstructed model
        (torch.ByteTensor) auxilary structure in bit-to-bit correspondence with octree
        (torch.ByteTensor) colors for final level of octree of size (n,4)
        (torch.FloatTensor) normals for final level of octree of size (n,3)

        
    .. _A Closed-Form Bayesian Fusion Equation Using Occupancy Probabilities:
        https://ieeexplore.ieee.org/document/7785112
    """
    try:
        first_frame_processed = False
        spc0 = {'octree' : None}

        # Iterate over calibrated rgbd data
        for batch in input_dataset:
            is_any_ray_hit = batch[9]
            if not is_any_ray_hit:
                continue
            if not first_frame_processed:
                # create SPC model corresponding to a single image
                spc0 = processFrame(batch, final_level, sigma)
            else:
                # create SPC model corresponding to a single image
                spc1 = processFrame(batch, final_level, sigma)

                # fuse new single frame SPC into existing cummulative SPC model
                spc0 = fuseBF(spc0, spc1)

            first_frame_processed = True

        # extract iso surface from over voxelized model
        # octree, empty, colors, normals = extractBQ(spc0)

        octree = spc0['octree']
        empty = spc0['empty']
        colors = spc0['colors']
        normals = spc0['normals']

        # weights = normals[:,3].reshape(-1,1)
        # normals = torch.nn.functional.normalize(normals[:,:3]/weights)
        # colors = (255.0*colors/weights).to(torch.uint8)
        # colors = torch.cat((colors, torch.zeros((colors.size(0),1), dtype=torch.uint8, device='cuda')), dim=1)

        return octree, empty, colors, normals
    except BFReconstructionTerminatedException:
        return None, None, None, None


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
