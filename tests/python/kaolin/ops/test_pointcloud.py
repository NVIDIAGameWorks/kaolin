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

import os
import pytest
import torch
import numpy as np
import math

from kaolin.utils.testing import FLOAT_TYPES, with_seed, check_allclose
import kaolin.ops.pointcloud

@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
def test_center_points(device, dtype):
    if dtype == torch.half:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-6  # default torch values

    B = 4
    N = 20
    points = torch.rand((B, N, 3), device=device, dtype=dtype)  # 0..1
    points[:, 0, :] = 1.0  # make sure 1 is included
    points[:, 1, :] = 0.0  # make sure 0 is included
    points = points - 0.5  # -0.5...0.5
    points = torch.clamp((torch.sign(points) * 1e-3) + points, -0.5, 0.5)

    factors = 0.2 + 2 * torch.rand((B, 1, 1), device=device, dtype=dtype)
    translations = torch.rand((B, 1, 3), device=device, dtype=dtype) - 0.5

    # Points are already centered
    check_allclose(points, kaolin.ops.pointcloud.center_points(points), atol=atol, rtol=rtol)
    check_allclose(points * factors, kaolin.ops.pointcloud.center_points(points * factors), atol=atol, rtol=rtol)

    # Points translated
    check_allclose(points, kaolin.ops.pointcloud.center_points(points + 0.5), atol=atol, rtol=rtol)

    points_centered = kaolin.ops.pointcloud.center_points(points + translations)
    check_allclose(points, points_centered, atol=atol, rtol=rtol)

    points_centered = kaolin.ops.pointcloud.center_points(points * factors + translations)
    check_allclose(points * factors, points_centered, atol=atol, rtol=rtol)

    # Now let's also try to normalize
    points_centered = kaolin.ops.pointcloud.center_points(points * factors + translations, normalize=True)
    check_allclose(points, points_centered, atol=atol, rtol=rtol)

    # Now let's test normalizing when there is zero range in one of the dimensions
    points[:, :, 1] = 1.0
    points_centered = kaolin.ops.pointcloud.center_points(points * factors + translations, normalize=True)
    points[:, :, 1] = 0.0
    check_allclose(points, points_centered, atol=atol, rtol=rtol)

    # Now let's try normalizing when one element of the batch is degenerate
    points[0, :, :] = torch.tensor([0, 2., 4.], dtype=dtype, device=device).reshape((1, 3))
    points_centered = kaolin.ops.pointcloud.center_points(points * factors + translations, normalize=True)
    points[0, :, :] = 0
    check_allclose(points, points_centered, atol=atol, rtol=rtol)

def validate_farthest_points_np(all_points, all_indices, k, rel_tol=1e-5, abs_tol=1e-5, verbose=False):
    """
    A helper that validates the outputs of point sampling, checking that the indices are 
    the correct length, valid, and distinct.

    Also tests that the first point really is the most distant from the center, and that each 
    subsequent point really is the farthest, up to tolerance.
    """

    if verbose: 
        print(f"\nChecking {k} farthest points")

    B = all_points.shape[0]
    for i_batch in range(B):

        # Iterate over batch dimension and convert to numpy
        points = all_points[i_batch,:,:].detach().cpu().numpy()
        indices = all_indices[i_batch,:].detach().cpu().numpy()

        ## Check that we have the right number of points
        assert len(indices) == k

        ## Check that all indices are valid
        assert np.all(indices >= 0) and np.all(indices < points.shape[0])

        ## Check that all indices are unique
        assert len(np.unique(indices)) == k

        if indices.shape[0] == 0:
            continue

        ## Check that the first point really is the farthest from the center
        first_pos = points[indices[0],:]
        center_pos = np.mean(points, where=np.isfinite(points), axis=0)
        farthest_from_center_dist = np.nanmax(np.linalg.norm(points - center_pos[None,:], axis=-1))
        first_from_center_dist = np.linalg.norm(first_pos - center_pos, axis=-1)
        if np.isfinite(center_pos).all():
            assert math.isclose(first_from_center_dist, farthest_from_center_dist, rel_tol=rel_tol, abs_tol=abs_tol), f"Point {0}={indices[0]} is not the farthest from the center: {first_from_center_dist} != {farthest_from_center_dist}"
        else:
            # if all points are nan/inf, any first point is fine
            pass


        ## Walk the points one at a time, checking that each new point is the farthest from the previous set
        # (or very close to it, allowing a small numerical epsilon for comparison)
        for i in range(1, k): # first point is always valid

            taken_mask = np.zeros(points.shape[0], dtype=bool)
            taken_mask[indices[:i]] = True
            selected_points = points[indices[:i], :]
            dists_to_current_set = np.linalg.norm(points[:,None,:] - selected_points[None,:,:], axis=2)

            dists_to_current_set[~np.isfinite(dists_to_current_set)] = np.nan # mask any inf values to nan, to simplify handling below
            min_dist_to_current_set = np.nanmin(dists_to_current_set, axis=1)
            farthest_untaken_dist = np.nanmax(min_dist_to_current_set[~taken_mask])
            farthest_untaken_ind = np.where(min_dist_to_current_set == farthest_untaken_dist)[0]
            new_point_dist = min_dist_to_current_set[indices[i]]

            if verbose: 
                print(f"  point {i}={indices[i]}  coords <{points[indices[i]]}>  dist={new_point_dist}  farthest_dist={farthest_untaken_dist}")

            if np.isfinite(farthest_untaken_dist):
                assert math.isclose(new_point_dist, farthest_untaken_dist, rel_tol=rel_tol, abs_tol=abs_tol), f"Point {i}={indices[i]} is not the farthest: {new_point_dist} != {farthest_untaken_dist} (at ind={farthest_untaken_ind})"
            else:
                # here we expect that the algorithm outputs something with not-finite distance, might as well confirm
                assert not np.isfinite(new_point_dist), f"Expected non-finite distance for point {i}={indices[i]}, got {new_point_dist}"

@pytest.mark.parametrize('device', ['cuda'])
def test_farthest_point_sampling(device):
    torch.cuda.manual_seed(42)

    # Basic usage, k < N
    # note: it's important to test with k > M_TOP_PROCESS (512 by default), because some
    # logic is only trigged in that situation
    N = 3000
    k = 32
    points = torch.rand(1, N, 3, device=device) 
    indices = kaolin.ops.pointcloud.farthest_point_sampling(points, k)
    validate_farthest_points_np(points, indices, k)

    # Basic usage w/ batching, k < N
    N = 3000
    k = 32
    B = 4
    points = torch.rand(B, N, 3, device=device) 
    indices = kaolin.ops.pointcloud.farthest_point_sampling(points, k)
    validate_farthest_points_np(points, indices, k)

    # k = 0
    N = 3000
    k = 0
    points = torch.rand(1, N, 3, device=device) 
    indices = kaolin.ops.pointcloud.farthest_point_sampling(points, k)
    validate_farthest_points_np(points, indices, k)

    # k = 1
    N = 3000
    k = 1
    points = torch.rand(1, N, 3, device=device) 
    indices = kaolin.ops.pointcloud.farthest_point_sampling(points, k)
    validate_farthest_points_np(points, indices, k)

    # N smaller than M_TOP_PROCESS
    N = 32
    k = 8
    points = torch.rand(1, N, 3, device=device) 
    indices = kaolin.ops.pointcloud.farthest_point_sampling(points, k)
    validate_farthest_points_np(points, indices, k)

    # k = N
    N = 32
    k = N
    points = torch.rand(1, N, 3, device=device) 
    indices = kaolin.ops.pointcloud.farthest_point_sampling(points, k)
    validate_farthest_points_np(points, indices, k)

    # N equal or 1 greater than M_TOP_PROCESS
    N = 512
    k = 8
    points = torch.rand(1, N, 3, device=device) 
    indices = kaolin.ops.pointcloud.farthest_point_sampling(points, k)
    validate_farthest_points_np(points, indices, k)
    N = 512+1
    k = 8
    points = torch.rand(1, N, 3, device=device) 
    indices = kaolin.ops.pointcloud.farthest_point_sampling(points, k)
    validate_farthest_points_np(points, indices, k)

    # k < N, repeated points
    N = 3000
    k = 32
    points = torch.rand(1, N, 3, device=device) 
    points[1:N//2,:] = points[0,:] # make the first 50% points identical
    indices = kaolin.ops.pointcloud.farthest_point_sampling(points, k)
    validate_farthest_points_np(points, indices, k)

    # k == N, repeated points
    N = 20
    k = N
    points = torch.rand(1, N, 3, device=device) 
    points[1:N//2,:] = points[0,:] # make the first 50% points identical
    indices = kaolin.ops.pointcloud.farthest_point_sampling(points, N)
    validate_farthest_points_np(points, indices, k)

    # k < N, NaN points, no NaNs should be taken
    N = 3000
    k = 32
    points = torch.rand(1, N, 3, device=device) 
    points[1:N//2,:] = np.nan
    indices = kaolin.ops.pointcloud.farthest_point_sampling(points, k)
    validate_farthest_points_np(points, indices, k)
    assert np.all(np.isfinite(points[:,indices,:].detach().cpu().numpy()))

    # k < N, some NaNs must be taken
    N = 20
    k = 15
    points = torch.rand(1, N, 3, device=device) 
    points[1:N//2,:] = np.nan
    indices = kaolin.ops.pointcloud.farthest_point_sampling(points, k)
    validate_farthest_points_np(points, indices, k)

    # k < N, inf points, no infs should be taken
    N = 3000
    k = 32
    points = torch.rand(1, N, 3, device=device) 
    points[1:N//2,:] = np.inf
    indices = kaolin.ops.pointcloud.farthest_point_sampling(points, k)
    validate_farthest_points_np(points, indices, k)
    assert np.all(np.isfinite(points[:,indices,:].detach().cpu().numpy()))

    # k < N, some infs must be taken
    N = 20
    k = 15
    points = torch.rand(1, N, 3, device=device) 
    points[1:N//2,:] = np.inf
    indices = kaolin.ops.pointcloud.farthest_point_sampling(points, k)
    validate_farthest_points_np(points, indices, k)