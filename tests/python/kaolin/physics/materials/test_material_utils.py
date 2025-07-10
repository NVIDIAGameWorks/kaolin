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

import os
import pytest
import torch
import warp as wp
import kaolin.physics as physics
from functools import partial
import kaolin.physics.materials.material_utils as material_utils
from kaolin.utils.testing import check_allclose

@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_to_lame(device, dtype):

    N = 20

    yms = 1e6 * torch.ones(N, device=device, dtype=dtype)
    prs = 0.35 * torch.ones(N, device=device, dtype=dtype)

    mus, lams = material_utils.to_lame(yms, prs)

    expected_mus = yms / (2 * (1 + prs))
    expected_lams = yms * prs / ((1 + prs) * (1 - 2 * prs))

    check_allclose(mus, expected_mus)
    check_allclose(lams, expected_lams)


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_wp_get_F(device, dtype):
    N = 20
    # Create N random points in the unit cube in torch
    pts = torch.rand(N, 3, device=device, dtype=dtype)

    # Create a random deformation in the unit cube in torch
    dz = 1e-1 * torch.rand(1, 3, 4, device=device, dtype=dtype)
    z0 = torch.zeros(1, 3, 4, device=device, dtype=dtype)
    z = dz + z0

    # Create fcn, return vector of ones
    def model(x): return torch.ones(
        (x.shape[0], 1), device=device, dtype=dtype)

    partial_weight_fcn_lbs = partial(
        physics.simplicits.skinning.weight_function_lbs, tfms=z.unsqueeze(0), fcn=model)

    expected_F = physics.utils.finite_diff_jac(
        partial_weight_fcn_lbs, pts).squeeze()

    # Warp code to get F using dFdz
    wp_z = wp.from_torch(z.flatten().contiguous())
    wp_dFdz = physics.simplicits.precomputed.sparse_dFdz_matrix_from_dense(
        model, pts)
    wp_dFdz = wp.sparse.bsr_copy(wp_dFdz, block_shape=(9, 4))
    wp_F = physics.materials.material_utils.get_defo_grad(wp_z, wp_dFdz)
    F = wp.to_torch(wp_F)


    check_allclose(expected_F, F, atol=1e-3)
    # Warp code to get F using finite differences
