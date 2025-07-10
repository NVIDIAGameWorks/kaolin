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
from functools import partial

from kaolin.physics.simplicits.precomputed import lumped_mass_matrix, lbs_matrix, jacobian_dF_dz, _jacobian_dF_dz_const_handle, jacobian_dx_dz, sparse_lbs_matrix, sparse_dFdz_matrix_from_dense
from kaolin.physics.simplicits.network import SimplicitsMLP
from kaolin.physics.utils.warp_utilities import _bsr_to_torch

from kaolin.utils.testing import FLOAT_TYPES, with_seed, check_allclose


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_lumped_mass_matrix_3D(device, dtype):

    N = 20
    DIM = 3

    rhos = torch.ones(N, dtype=dtype, device=device)
    total_vol = 10

    mass_matrix, _ = lumped_mass_matrix(rhos, total_volume=total_vol, dim=DIM)

    expected_mass_matrix = (total_vol / N) * \
        torch.eye(DIM * N, dtype=dtype, device=device)

    check_allclose(mass_matrix, expected_mass_matrix)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_lumped_mass_matrix_2D(device, dtype):

    N = 20
    DIM = 2

    rhos = torch.ones(N, dtype=dtype, device=device)
    total_vol = 10

    mass_matrix, _ = lumped_mass_matrix(rhos, total_volume=total_vol, dim=DIM)

    expected_mass_matrix = (total_vol / N) * \
        torch.eye(DIM * N, dtype=dtype, device=device)

    check_allclose(mass_matrix, expected_mass_matrix)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_lbs_matrix(device, dtype):
    N = 20
    H = 5
    B = 1
    points = torch.rand(N, 3, device=device, dtype=dtype)  # n x 3 points
    transforms = torch.rand(B, H, 3, 4, device=device,
                            dtype=dtype)  # lbs transforms
    # random skinning weights over points
    weights = torch.rand(N, H, device=device, dtype=dtype)

    B = lbs_matrix(points, weights)

    x = B @ transforms[0].flatten() + points.flatten()

    transformed_points = x.reshape(N, 3)

    expected_points = torch.zeros_like(points)

    for i in range(N):
        pt_i = points[i].unsqueeze(0)
        pt4_i = torch.cat(
            (pt_i, torch.tensor([[1]], device=device, dtype=dtype)), dim=1)
        expect_pt_i = pt_i.T
        for j in range(H):
            expect_pt_i += weights[i, j] * transforms[0, j] @ pt4_i.T

        expected_points[i, :] = expect_pt_i.T
        # checking 3D point by 3D point, so flattening should be ok

    check_allclose(expected_points, transformed_points)
    

@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_sparse_lbs_matrix(device, dtype):
    N = 20
    H = 5
    B = 1
    points = torch.rand(N, 3, device=device, dtype=dtype)  # n x 3 points
    transforms = torch.rand(B, H, 3, 4, device=device,
                            dtype=dtype)  # lbs transforms
    # random skinning weights over points
    weights = torch.rand(N, H, device=device, dtype=dtype)

    B = _bsr_to_torch(sparse_lbs_matrix(
        wp.array(weights), wp.array(points, dtype=wp.vec3))).to_dense().to(device)

    x = B @ transforms[0].flatten() + points.flatten()

    transformed_points = x.reshape(N, 3)

    expected_points = torch.zeros_like(points)

    for i in range(N):
        pt_i = points[i].unsqueeze(0)
        pt4_i = torch.cat(
            (pt_i, torch.tensor([[1]], device=device, dtype=dtype)), dim=1)
        expect_pt_i = pt_i.T
        for j in range(H):
            expect_pt_i += weights[i, j] * transforms[0, j] @ pt4_i.T

        expected_points[i, :] = expect_pt_i.T
        # checking 3D point by 3D point, so flattening should be ok

    check_allclose(expected_points, transformed_points)

@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_sparse_dFdz_matrix_from_dense(device, dtype):
    N = 20
    H = 5
    B = 1
    points = torch.rand(N, 3, device=device, dtype=dtype)  # n x 3 points
    transforms = torch.rand(B, H + 1, 3, 4, device=device,
                            dtype=dtype)  # lbs transforms

    model = SimplicitsMLP(3, 64, H, 6).to(device)
    if dtype == torch.double:
        model.double()
    
    def model_plus_rigid(pts): return torch.cat(
        (model(pts), torch.ones((pts.shape[0], 1), device=device)), dim=1)

    expected_dF_dz = jacobian_dF_dz(model=model_plus_rigid, x0=points, z=transforms[0])
    
    dF_dz = _bsr_to_torch(sparse_dFdz_matrix_from_dense(enriched_weights_fcn=model_plus_rigid, pts=points)).to_dense().to(device)
    
    check_allclose(dF_dz, expected_dF_dz, rtol=0.01, atol=0.01)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_jacobian_dF_dz(device, dtype):
    # use super simple weights and calculate expected dFdz manually to test
    # do this later: for more complex weights test using finite differences
    N = 20
    H = 5
    B = 1
    points = torch.rand(N, 3, device=device, dtype=dtype)  # n x 3 points

    # H handles + 1 for the constant handle
    transforms = torch.rand(H + 1, 3, 4, device=device,
                            dtype=dtype).unsqueeze(0)
    model = SimplicitsMLP(3, 64, H, 6).to(device)
    if dtype == torch.double:
        model.double()

    def model_plus_rigid(pts): return torch.cat(
        (model(pts), torch.ones((pts.shape[0], 1), device=device)), dim=1)

    dF_dz1 = _jacobian_dF_dz_const_handle(model, points, transforms.flatten())
    # dF_dz2 = jacobian_dF_dz(model, points, transforms.flatten())
    dF_dz2 = jacobian_dF_dz(model=model_plus_rigid, x0=points, z=transforms[0])

    check_allclose(dF_dz1, dF_dz2, rtol=0.01, atol=0.01)




@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_jacobian_dx_dz(device, dtype):
    N = 20
    H = 5
    B = 1
    points = torch.rand(N, 3, device=device, dtype=dtype)  # n x 3 points
    transforms = torch.rand(B, H + 1, 3, 4, device=device,
                            dtype=dtype)  # lbs transforms

    model = SimplicitsMLP(3, 64, H, 6).to(device)
    if dtype == torch.double:
        model.double()

    def model_plus_rigid(pts): return torch.cat(
        (model(pts), torch.ones((pts.shape[0], 1), device=device)), dim=1)
    weights = model_plus_rigid(points)

    B = jacobian_dx_dz(model_plus_rigid, points,
                       transforms.squeeze().flatten())

    x = B @ transforms.squeeze().flatten() + points.flatten()

    transformed_points = x.reshape(N, 3)

    expected_points = torch.zeros_like(points)

    for i in range(N):
        pt_i = points[i].unsqueeze(0)
        pt4_i = torch.cat(
            (pt_i, torch.tensor([[1]], device=device, dtype=dtype)), dim=1)
        expect_pt_i = pt_i.T
        for j in range(H + 1):
            expect_pt_i += weights[i, j] * transforms[0, j] @ pt4_i.T

        expected_points[i, :] = expect_pt_i.T
        # checking 3D point by 3D point, so flattening should be ok

    check_allclose(expected_points, transformed_points)
