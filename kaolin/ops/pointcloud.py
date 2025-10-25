# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

from __future__ import annotations
import torch
import warp as wp

__all__ = [
    'center_points',
    'farthest_point_sampling'
]

def center_points(points: torch.FloatTensor, normalize: bool = False, eps=1e-6):
    r"""Returns points centered at the origin for every pointcloud. If `normalize` is
    set, will also normalize each point cloud spearately to the range of [-0.5, 0.5].
    Note that each point cloud is centered individually.

    Args:
        points (torch.FloatTensor): point clouds of shape :math:`(\text{batch_size}, \text{num_points}, 3)`,
         (other channel numbers supported).
        normalize (bool): if true, will also normalize each point cloud to be in the range [-0.5, 0.5]
        eps (float): eps to use to avoid division by zero when normalizing

    Return:
        (torch.FloatTensor) modified points with same shape, device and dtype as input
    """
    assert len(points.shape) == 3, f'Points have unexpected shape {points.shape}'

    vmin = points.min(dim=1, keepdim=True)[0]
    vmax = points.max(dim=1, keepdim=True)[0]
    vmid = (vmin + vmax) / 2
    res = points - vmid
    if normalize:
        den = (vmax - vmin).max(dim=-1, keepdim=True)[0].clip(min=eps)
        res = res / den
    return res



def farthest_point_sampling(points, k):
    r"""Performs farthest point sampling to select a subset of :math:`k` points from a point cloud. 
    The first point returned is the one most distant from the center, and each subsequent point
    is the one most distant from the previously-selected set. 
    
    This operation is useful for generating a nicely-spaced subset of a large point cloud, with
    a blue-noise-like distribution.

    Even if the point cloud countains `inf` or `NaN` coordinates, this function will always return exactly
    :math:`k` distinct indices.

    Args:
        points (torch.FloatTensor): point clouds of shape :math:`(\text{batch_size}, \text{num_points}, 3)`
        k (int): the number of farthest points to sample. Must be :math:`0 <= k <= \text{num_points}`

    Return:
        (torch.LongTensor) indices into the `points` tensor giving each sampled point, shape :math:`(\text{batch_size}, \text{k})`
    """

    assert len(points.shape) == 3, f'Points have unexpected shape {points.shape}'
    assert points.is_cuda, f'Points should be on a CUDA device, only CUDA is supported for farthest point sampling. Device is {points.device}'
    assert points.dtype == torch.float32, f'Points should have dtype float32. dtype is {points.dtype}'

    # TODO check CUDA >= 12 for warp support

    result_indices = torch.zeros((points.shape[0], k), dtype=torch.int64)

    # Process the batches sequentially
    for i_batch in range(points.shape[0]):
        result_indices[i_batch,:] = _farthest_point_sampling_warp_headchunk(points[i_batch,:], k)

    return result_indices


# implementation efficiencty parameter: how many threads per block,
# which is also the size of the head chunk processed each iteration
_M_TOP_PROCESS = 512

# Some constant values which are internally substituted as distances during processing
# The relative values of these constants is important for algorithm logic.
_INVALID_DIST = -1.0 # Distance used for inf or nan points, so they are only sampled after all others
_TAKEN_DIST = -2.0   # Distance for a point which has been sampled in FPS. 
                    # We use this rather than 0., to get logic right when multiple points are on top of each other.
_PADDED_DIST = -3.0  # Distance for point added as padding to hit the tile size. These should never be touched for any purpose.

def _farthest_point_sampling_warp_headchunk(points, k):
    r"""Internal function that runs the farthest point sampling algorithm.

    Logically the algorithm is as-follows:
      1. Find the center (mean) of the point cloud
      2. Accept the 1st point as the one most distant from the center.
      3. Repeatedly accept the point farthest from any already-sampled point, until we have `k` points.
      4. Return an ordered list of indices of the `k` accepted points

    A naive approach is to repeatedly scan the list for the farthest point, for `O(k*N)` total cost. Instead
    this code implements a block-aware strategy on the GPU. The main insight is that if, at any point in the algorithm, 
    we sort the points by their current distance, then likely many of the next sampled points will lie near the head
    of the sorted ordering. This is particularly true as `k` becomes large. Also, as we accept points, the distance
    to the accepted set only ever decreases, it never increases. We can use these properties together to efficiently
    find points only from the top of the array, while still guaranteeing correctness.

    The algorithm implemented here is:
      1. Find the center (mean) of the point cloud
      2. Accept the 1st point as the one most distant from the center.
      3. Radix-sort the points by their distance from the currently accepted set
      4. Consider only the top-M head-chunk of the sorted list, and find as many points as possible within that chunk:
        4.1 Record the distance of the last point in the head-chunk of the array as `d_threshold`. As long as a point 
            has distance >= `d_threshold` we can safely accept it, since all points outside the head-chunk will 
            always have distance <= `d_threshold`
        4.2 Find the most-distant point in the head-chunk
        4.3 If its distance is < `d_threshold`, break and GOTO 5.
        4.4 Accept the new most-distant point, record it in the output list
        4.5 Update the distances of all points in the head-chunk
        4.6 GOTO 4.2 and try to get the next-nearest point fromt he head-chunk
      5. GOTO 3, re-sort the whole array, and process the head-chunk again, until we have `k` accepted points

    We implement this efficiently on the GPU by doing step 4. within a single block, using Warp's Tile API. This function
    is also written to use only Warp operations and no memory allocations after the initial setup, except a few torch ops
    right at the start, so the code would only need to be slightly modified to enable graph capture.
    
    There is also significant logic in the kernels to ensure correctness in the case of `inf` of `NaN` point coordinates.
    Even if such coordinates exist, this code should always return exactly `k` valid, distinct indices into the point cloud.
    This property is important, because otherwise a spurious bad coordinate might lead to arrays with the wrong shape, eventually
    crashing a training loop, etc.

    Args:
        points (torch.FloatTensor): point clouds of shape :math:`(\text{num_points}, 3)`
        k (int): the number of farthest points to sample. Must be :math:`0 <= k <= \text{num_points}`

    Return:
        None. The output is stored in the result_out array which is passed in pre-allocated.
    """
    device = points.device
    wp_device = wp.device_from_torch(device)
    

    # Padding to minimum size as needed
    N = points.shape[0]
    N_PADDED = max(N,_M_TOP_PROCESS)
    PAD_LEN = N_PADDED - N
    if PAD_LEN < 0:
        points = torch.concatenate((points, torch.zeros((PAD_LEN,3), device=device, dtype=torch.float32)), axis=0)
    points = wp.from_torch(points, requires_grad=False, dtype=wp.vec3)
    
    # Allocate the output array on the torch side, then wrap it to Warp
    # In principle we could allocate the array in Warp, but doing it this way works around 
    # a bug observed in CUDA 11.8 where calling `to_torch()` on an empty Warp array fails
    farthest_point_inds_torch32 = torch.zeros((k,), dtype=torch.int32, device=device)
    farthest_point_inds = wp.from_torch(farthest_point_inds_torch32, requires_grad=False, dtype=wp.int32)
    
    if k==0:
        # early-out in this case, to avoid out of bounds issues below
        return farthest_point_inds_torch32.to(dtype=torch.int64)

    # Allocate all working arrays
    center_point = wp.zeros(1, dtype=wp.vec3, device=wp_device)
    center_count = wp.zeros(1, dtype=wp.float32, device=wp_device)
    # NOTE: radix sort requires 2N space in array, so these are allocated bigger
    point_inds = wp.from_torch(torch.concatenate((
        torch.arange(N_PADDED, dtype=torch.int32, device=device), 
        torch.full((N_PADDED,), fill_value=-1, dtype=torch.int32, device=device)
    ))) 
    distancesSq = wp.full(shape=2*N_PADDED, value=_PADDED_DIST, dtype=wp.float32, device=wp_device) # we re-initialize non-padding entries below
    i_round = wp.zeros(shape=1, dtype=wp.int32, device=wp_device)
    i_prev_round = wp.zeros(shape=1, dtype=wp.int32, device=wp_device)
    farthest_point_inds.fill_(value=-1)


    # == Initialize the first point as the most distance point from the center

    # Find the center
    wp.launch(_compute_center_sum_kernel, 
            dim=N,
            inputs=[points], 
            outputs=[center_point, center_count],
            )
    wp.launch(_divide_center_kernel, 
            dim=1,
            inputs=[center_point, center_count],
            )

    # Initialize distancesSq to be distance from center
    wp.launch(_initialize_distances_kernel, 
            dim=N,
            inputs=[
                points,
                distancesSq,
                center_point,
            ], 
            )

    # Find the first farthest point
    wp.launch(_find_farthest_point_ind_kernel,
            dim=N_PADDED,
            inputs=[
                distancesSq,
                farthest_point_inds,
                i_round
            ], 
            )
    # Update the distancesSq from the first point
    wp.launch(_initialize_distances_indexed_kernel, 
            dim=N,
            inputs=[
                points,
                distancesSq,
                farthest_point_inds,
                N,
            ], 
            )

    # Increment the round
    # do this on the GPU to enable graph capture
    wp.launch(_increment_round_kernel, dim=1, inputs=[ i_round ])
    wp.launch(_increment_round_kernel, dim=1, inputs=[ i_prev_round ])

    ## Main loop: process rounds until we have found k points

    # Optimization: Rather than counting exactly how many points we have found, 
    # we instead track a range of how many we might have found, and fetch 
    # the actual value only if there is a chance we might be done.
    #
    # Most of the time, we can safely run the next round without checking, 
    # which removes the only synchronization point per-iteration
    found_estimate_min = i_round.numpy()[0]
    found_estimate_max = found_estimate_min # inclusive
    while found_estimate_min < k:

        # Sort the points by distance
        wp.utils.radix_sort_pairs(distancesSq, point_inds, count=N_PADDED)

        # Accept as many points as we can off the top
        wp.launch(_take_top_m_farthest_kernel, 
                dim=_M_TOP_PROCESS,
                inputs=[
                    points,
                    N_PADDED,
                    distancesSq,
                    point_inds,
                    farthest_point_inds,
                    i_round,
                    i_prev_round
                ], 
                block_dim=_M_TOP_PROCESS, # must be compatible with the hardware value (e.g. <=1024 for most modern GPUs)
            )
        
        # Recompute all distances
        wp.launch(_update_distances_from_round_kernel, 
                dim=N_PADDED,
                inputs=[
                    points,
                    distancesSq,
                    point_inds,
                    farthest_point_inds,
                    i_round,
                    i_prev_round,
                    N,
                ], 
            )

        # Update the estimate
        found_estimate_min += 1 # we always find at least one point
        found_estimate_max += _M_TOP_PROCESS # at most, we could take all M points in a round (but this is extremely optimistic)

        # This conditional checks if there's a chance we might need to exit the processing loop.
        if found_estimate_max >= k:
            curr_found = i_round.numpy()[0]
            found_estimate_min = curr_found
            found_estimate_max = curr_found

    return farthest_point_inds_torch32.to(dtype=torch.int64)
    


# This kernel does the heavy lifting. Given an array sorted, it finds as many farthest points 
# as possible among the top M elements of the array using a single block.
@wp.kernel
def _take_top_m_farthest_kernel(
    points: wp.array(dtype=wp.vec3), 
    N: wp.int32,
    distancesSq: wp.array(dtype=wp.float32), 
    point_inds: wp.array(dtype=wp.int32), 
    farthest_point_inds: wp.array(dtype=wp.int32),
    i_round_arr: wp.array(dtype=wp.int32),
    i_prev_round_arr: wp.array(dtype=wp.int32),
    ):
    """
    Given arrays of points sorted by their current farthest distance from the taken set.

    0                                       N-M              N
    | -------------------------------------- | ------------- |
                                             ^^^^^^^^^^^^^^^^^
                                                 head chunk
                                             ^
                                             dist=head_chunk_threshold

    This kernel loads the head chunk into a single block, finds as many farthest points as possible
    among the head chunk, then returns to re-sort the array and try again. This is necessarily valid
    because accepting farthest points only ever decreases points' distances, so as long as the farthest
    point distance is greater than the initial distance of the last point in the head chunk, it must
    be the true farthest point in the array.
    """

    block_i = wp.tid()
    i_prev_round_arr[0] = i_round_arr[0]
    i_round = i_round_arr[0]
    top_block_offset = N-_M_TOP_PROCESS # pad so this is always >= 0

    # Load the top M into shared memory with a tile'd op
    inds_tile = wp.tile_load(point_inds, _M_TOP_PROCESS, offset=top_block_offset, storage='shared')
    distsSq_tile = wp.tile_load(distancesSq, _M_TOP_PROCESS, offset=top_block_offset, storage='shared')

    # The least-far point in the head chunk limits what we can safely process
    # as soon as the distances are shorter than this, the actual farthest point
    # might lie outside of the head chunk, so we must stop processing to recompute 
    # distance & re-sort the full array
    head_chunk_threshold = distsSq_tile[0] 

    while i_round < farthest_point_inds.shape[0]:
        # Note: we usually exit the loop due to the break() below. The loop condition above only handles 
        # the occasional case where we have found all K points and stop processing.

        # The head chunk is sorted when this loop starts, but we do not maintain the sorting as the 
        # loop executes; we treat it as un-sorted.

        # Find the most distant point in the head chunk
        top_block_max_i = wp.tile_argmax(distsSq_tile)[0]

        # Gather the index and distance of the most distant point in the head
        top_point_ind = inds_tile[top_block_max_i]
        top_point_dist = distsSq_tile[top_block_max_i]
        top_point_p = points[top_point_ind]

        # If top_point_dist < head_chunk_threshold, then it is possible that the actual farthest
        # point lies outside of the head block, so we must finish this kernel and re-sort
        # the array.
        if top_point_dist < head_chunk_threshold or top_point_dist == _TAKEN_DIST:
            break

        # Flag the new farthest point as taken, and update the other points distances
        # in the head chunk if the points distance to the new farthest point is closer
        old_dist = distsSq_tile[block_i]
        dist = old_dist
        if old_dist != _PADDED_DIST:
            p = points[inds_tile[block_i]]
            new_dist = wp.length_sq(p - top_point_p)
            if wp.isfinite(new_dist):
                dist = wp.min(dist, new_dist)
            if block_i == top_block_max_i:
                dist = _TAKEN_DIST
        distsSq_tile[block_i] = dist

        # Write this newly-selected farthest point into the output list
        if block_i == 0:
            # TODO can we delay and vectorize this?
            farthest_point_inds[i_round] = top_point_ind
        i_round += 1 # do another round!

    # Write the incremented round back to the global arrays, so we know
    # how many farthest points we have found.
    if block_i == 0:
        i_round_arr[0] = i_round

@wp.kernel
def _update_distances_from_round_kernel(
    points: wp.array(dtype=wp.vec3), 
    distancesSq: wp.array(dtype=wp.float32), 
    point_inds: wp.array(dtype=wp.int32),
    farthest_point_inds: wp.array(dtype=wp.int32),
    i_round_arr: wp.array(dtype=wp.int32),
    i_prev_round_arr: wp.array(dtype=wp.int32),
    N: wp.int32,
    ):
    r"""
    Update the global arrays of point distances to account for the distances
    from the last head-chunk processing round (the [i_prev_round, i_round) entries).
    """

    i = wp.tid()
    i_round = i_round_arr[0]
    i_prev_round = i_prev_round_arr[0]
    point_ind = point_inds[i]

    if point_ind >= N:
       # padding point, don't do anything
       return 

    p = points[point_ind]
    this_min_distSq = distancesSq[i]

    for i_new_round in range(i_prev_round, i_round):
        new_point_ind = farthest_point_inds[i_new_round]
        new_p = points[new_point_ind]
        dist = wp.length_sq(p - new_p)
        if point_ind == new_point_ind:
            dist = _TAKEN_DIST
        if wp.isfinite(dist):
            this_min_distSq = wp.min(this_min_distSq, dist)

    distancesSq[i] = this_min_distSq

# === Small helper kernels to compute distances over arrays

@wp.kernel
def _compute_center_sum_kernel(
    points: wp.array(dtype=wp.vec3), 
    center_point_sum: wp.array(dtype=wp.vec3),
    center_point_count: wp.array(dtype=wp.float32)
):
    r"""
    Helper kernel for for computing the center of a point cloud.
    Sum up all positions, and a count of the contributing points (skipping NaNs/infs).
    """
    i = wp.tid()
    p = points[i]

    if wp.isfinite(p.x) and wp.isfinite(p.y) and wp.isfinite(p.z):
        center_point_sum[0] += p
        center_point_count[0] += 1.0

@wp.kernel
def _divide_center_kernel(
    center_point_sum: wp.array(dtype=wp.vec3),
    center_point_count: wp.array(dtype=wp.float32)
):
    r"""
    Helper kernel for for computing the center of a point cloud.
    Divide the position by the count to get a mean, handling NaN/inf.
    """
    center = center_point_sum[0] / center_point_count[0]
    if wp.isfinite(center.x) and wp.isfinite(center.y) and wp.isfinite(center.z):
        center_point_sum[0] = center
    else:
        center_point_sum[0] = wp.vec3(0., 0., 0.)

@wp.kernel
def _initialize_distances_kernel(
    points: wp.array(dtype=wp.vec3), 
    distancesSq: wp.array(dtype=wp.float32), 
    first_point: wp.array(dtype=wp.vec3),
    ):
    r"""
    Helper kernel to initialize the distanceSq from a vec3 point.
    """
    i = wp.tid()
    p = points[i]

    dist = wp.length_sq(p - first_point[0])
    if not wp.isfinite(dist):
        dist = _INVALID_DIST

    distancesSq[i] = dist

@wp.kernel
def _initialize_distances_indexed_kernel(
    points: wp.array(dtype=wp.vec3), 
    distancesSq: wp.array(dtype=wp.float32), 
    first_point_ind: wp.array(dtype=wp.int32),
    N: wp.int32,
    ):
    r"""
    Helper kernel to initialize the distanceSq from an index pointing to a vec3 point.
    """
    i = wp.tid()
    p = points[i]

    dist = wp.length_sq(p - points[first_point_ind[0]])
    if not wp.isfinite(dist):
        # Here we intentionally write INVALID_DIST to the array, rather than skipping.
        # it's probably because the target point is invalid. If the very first point
        # we selected was invalid that would also cause this, but in that case there
        # is not much else to do.
        dist = _INVALID_DIST
    if i == first_point_ind[0]:
        dist = _TAKEN_DIST
    if i >= N: # must be a padding point
        dist = _PADDED_DIST 

    distancesSq[i] = dist

@wp.kernel
def _find_farthest_point_ind_kernel(
    distancesSq: wp.array(dtype=wp.float32), 
    farthest_point_inds: wp.array(dtype=wp.int32),
    i_round_arr: wp.array(dtype=wp.int32),
    ):
    r"""
    Helper kernel to do a CAS scan to find the farthest point. We only do this once 
    at the very beginning to find the first point.
    """
    i = wp.tid()
    i_round = i_round_arr[0]
    
    my_dist = distancesSq[i]

    curr_farthest_ind = farthest_point_inds[i_round]
    while(curr_farthest_ind < 0 or my_dist > distancesSq[curr_farthest_ind]):
        # this loop says "if this distance is greater than current farthest distance, swap it in
        # however, if another thread has already changed it in the meantime, fetch the new value 
        # and check again
        wp.atomic_cas(farthest_point_inds, i_round, curr_farthest_ind, i)
        curr_farthest_ind = farthest_point_inds[i_round]

@wp.kernel
def _increment_round_kernel(
    i_round_arr: wp.array(dtype=wp.int32),
    ):
    r"""
    Helper kernel to increment an integer, used for the `k`'th point we
    have thus found. Used just once at the start.
    """
    i_round_arr[0] += 1