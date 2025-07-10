# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

import pytest
import torch
from typing import Any
import warp as wp
import kaolin.physics as physics

import kaolin.physics.materials.neohookean_elastic_material as neohookean_elastic_material
import kaolin.physics.materials.material_utils as material_utils
from kaolin.utils.testing import check_allclose
from kaolin.physics.materials import NeohookeanElasticMaterial

def _unbatched_neohookean_energy(mu, lam, defo_grad):
    r"""Implements an version of neohookean energy. Calculate energy per-integration primitive.

    Args:
        mu (torch.Tensor): Tensor of lame parameter mu, of shape :math:`(\text{batch_dim}, 1)`
        lam (torch.Tensor): Tensor of lame parameter lambda, of shape :math:`(\text{batch_dim}, 1)`
        defo_grad (torch.Tensor): Flattened 3d deformation gradients, of shape :math:`(\text{batch_dim}*3*3, 1)`

    Returns:
        torch.Tensor: Vector of per-primitive energy values, shape :math:`(\text{batch_dim}, 1)`
    """
    F = defo_grad.reshape(-1, 9)
    F_0 = F[:, 0]
    F_1 = F[:, 1]
    F_2 = F[:, 2]
    F_3 = F[:, 3]
    F_4 = F[:, 4]
    F_5 = F[:, 5]
    F_6 = F[:, 6]
    F_7 = F[:, 7]
    F_8 = F[:, 8]

    J = (F_0 * (F_4 * F_8 - F_5 * F_7)
         - F_1 * (F_3 * F_8 - F_5 * F_6)
         + F_2 * (F_3 * F_7 - F_4 * F_6)).unsqueeze(dim=1)

    IC = (F_0 * F_0 + F_1 * F_1 + F_2 * F_2 +
          F_3 * F_3 + F_4 * F_4 + F_5 * F_5 +
          F_6 * F_6 + F_7 * F_7 + F_8 * F_8).unsqueeze(dim=1)

    return 0.5 * mu * (IC - 3.0) + 0.5 * lam * (J - 1.) * (J - 1.) - mu * (J - 1.0)


def _unbatched_neohookean_gradient(mu, lam, defo_grad):
    r"""Implements a version of neohookean gradients w.r.t deformation gradients. Calculate gradient per-integration primitive.

    Args:
        mu (torch.Tensor): Tensor of lame parameter mu, of shape :math:`(\text{batch_dim}, 1)`
        lam (torch.Tensor): Tensor of lame parameter lambda, of shape :math:`(\text{batch_dim}, 1)`
        defo_grad (torch.Tensor): Flattened 3d deformation gradients, of shape :math:`(\text{batch_dim}*3*3, 1)`

    Returns:
        torch.Tensor: Tensor of per-primitive neohookean gradient w.r.t defo_grad, flattened, shape :math:`(\text{batch_dim}, 9)`
    """
    F = defo_grad.reshape(-1, 9)
    F_0 = F[:, 0]
    F_1 = F[:, 1]
    F_2 = F[:, 2]
    F_3 = F[:, 3]
    F_4 = F[:, 4]
    F_5 = F[:, 5]
    F_6 = F[:, 6]
    F_7 = F[:, 7]
    F_8 = F[:, 8]
    # compute batch determinant of all F's
    J = (F_0 * (F_4 * F_8 - F_5 * F_7)
         - F_1 * (F_3 * F_8 - F_5 * F_6)
         + F_2 * (F_3 * F_7 - F_4 * F_6)).unsqueeze(dim=1)

    # a = 1.0 + mu/lam
    FinvT = torch.inverse(defo_grad.reshape(-1, 3, 3)).transpose(1, 2)

    return (mu[:, :, None] * defo_grad.reshape(-1, 3, 3) + (J * (lam * (J + (-1.0)) + (-mu))).unsqueeze(dim=2) * FinvT)


def _unbatched_neohookean_hessian(mu, lam, defo_grad):
    r"""Implements a version of neohookean hessian w.r.t deformation gradients. Calculate per-integration primitive.

    Args:
        mu (torch.Tensor): Tensor of lame parameter mu, of shape :math:`(\text{batch_dim}, 1)`
        lam (torch.Tensor): Tensor of lame parameter lambda, of shape :math:`(\text{batch_dim}, 1)`
        defo_grad (torch.Tensor): Flattened 3d deformation gradients, of shape :math:`(\text{batch_dim}*3*3, 1)`

    Returns:
        torch.Tensor: Tensor of per-primitive neohookean hessian w.r.t defo_grad, flattened, shape :math:`(\text{batch_dim}, 9,9)`
    """
    F = defo_grad.reshape(-1, 9)
    F_0 = F[:, 0]
    F_1 = F[:, 1]
    F_2 = F[:, 2]
    F_3 = F[:, 3]
    F_4 = F[:, 4]
    F_5 = F[:, 5]
    F_6 = F[:, 6]
    F_7 = F[:, 7]
    F_8 = F[:, 8]
    id_mat = torch.eye(9, device=mu.device)

    # can save more time by not recomputing this stuff
    J = (F_0 * (F_4 * F_8 - F_5 * F_7)
         - F_1 * (F_3 * F_8 - F_5 * F_6)
         + F_2 * (F_3 * F_7 - F_4 * F_6)).unsqueeze(dim=1)

    # a = 1.0 + mu/lam
    FinvT = torch.inverse(defo_grad.reshape(-1, 3, 3)).transpose(1, 2)
    gamma = J * (lam * (2.0 * J + (-1.0)) + (-mu))
    dgamma = gamma - lam * J * J

    FFinv = torch.bmm(FinvT.reshape(-1, 9, 1), FinvT.reshape(-1, 1, 9))
    H1 = mu[:, :, None] * id_mat
    H2 = gamma[:, :, None] * FFinv
    H3 = -dgamma[:, :, None] * \
        FFinv.reshape(-1, 3, 3, 3, 3).transpose(2, 4).reshape(-1, 9, 9)

    return H1 + H2 + H3


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_neohookean_gradients_test_helper_fcn(device, dtype):
    N = 20
    B = 1
    eps = 1e-8

    F = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)
    + eps * torch.rand(N, B, 3, 3, device=device, dtype=dtype)

    yms = 1e3 * torch.ones(N, 1, device=device)
    prs = 0.4 * torch.ones(N, 1, device=device)

    mus, lams = material_utils.to_lame(yms, prs)

    neo_grad = _unbatched_neohookean_gradient(
        mus, lams, F)
    assert (neo_grad.shape[0] == N and neo_grad.shape[1]
            == 3 and neo_grad.shape[2] == 3)
    neo_grad = neo_grad.flatten()

    E0 = torch.sum(
        _unbatched_neohookean_energy(mus, lams, F))

    expected_grad = torch.zeros_like(F.flatten(), device=device)
    row = 0

    # Using Finite Diff to compute the expected gradients
    for n in range(F.shape[0]):
        for i in range(F.shape[1]):
            for j in range(F.shape[2]):
                F[n, i, j] += eps
                El = torch.sum(
                    _unbatched_neohookean_energy(mus, lams, F))
                F[n, i, j] -= 2 * eps
                Er = torch.sum(
                    _unbatched_neohookean_energy(mus, lams, F))
                F[n, i, j] += eps
                expected_grad[row] = (El - Er) / (2 * eps)
                row += 1

    check_allclose(neo_grad, expected_grad, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_neohookean_hessian_helper_vs_fd(device, dtype):
    N = 5
    B = 1
    eps = 1e-3

    F = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)

    yms = 1e3 * torch.ones(N, 1, device=device)
    prs = 0.4 * torch.ones(N, 1, device=device)

    mus, lams = material_utils.to_lame(yms, prs)

    neo_hess = _unbatched_neohookean_hessian(
        mus, lams, F)

    expected_hess = torch.zeros(N, 9, 9, device=device, dtype=dtype)
    row = 0

    # Using Finite Diff to compute the expected hessian
    for n in range(F.shape[0]):
        for i in range(F.shape[1]):
            for j in range(F.shape[2]):
                F[n, i, j] += eps
                Gl = _unbatched_neohookean_gradient(
                    mus[n].unsqueeze(0), lams[n].unsqueeze(0), F[n].unsqueeze(0)).flatten()
                F[n, i, j] -= 2 * eps
                Gr = _unbatched_neohookean_gradient(
                    mus[n].unsqueeze(0), lams[n].unsqueeze(0), F[n].unsqueeze(0)).flatten()
                F[n, i, j] += eps
                expected_hess[n, 3 * i + j, :] = (Gl - Gr) / (2 * eps)
                row += 1

    check_allclose(neo_hess, expected_hess, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_neohookean_hessian_helper_vs_autograd(device, dtype):
    N = 5
    B = 1

    F = 2*torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)

    yms = 1e3 * torch.ones(N, 1, device=device)
    prs = 0.4 * torch.ones(N, 1, device=device)

    mus, lams = material_utils.to_lame(yms, prs)

    # N x 9 x 9 (the block diags)
    neo_hess = _unbatched_neohookean_hessian(
        mus, lams, F)

    # 9N x 9N (the full hessian with a bunch of zeros)
    autograd_hessian = torch.autograd.functional.jacobian(
        lambda x: _unbatched_neohookean_gradient(mus, lams, x), F)
    autograd_hessian = autograd_hessian.reshape(9 * N, 9 * N)

    # Make sure the block diags match up
    for n in range(N):
        check_allclose(
            neo_hess[n], autograd_hessian[9 * n:9 * n + 9, 9 * n:9 * n + 9], rtol=1e-1, atol=1e-1)
        # zero out the block
        autograd_hessian[9 * n:9 * n + 9, 9 * n:9 * n + 9] *= 0

    # Make sure the rest of the matrix is zeros
    check_allclose(torch.zeros_like(autograd_hessian),
                   autograd_hessian, rtol=1e-1, atol=1e-1)


##################################################
###### TESTING THE BATCHED TORCH TRAINING CODE HERE ##
######
######
######
##################################################

@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_neohookean_energy(device, dtype):
    N = 20
    B = 1

    F = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)

    yms = 1e3 * torch.ones(N, 1, 1, device=device)
    prs = 0.4 * torch.ones(N, 1, 1, device=device)

    mus, lams = material_utils.to_lame(yms, prs)

    E1 = torch.sum(
        _unbatched_neohookean_energy(mus, lams, F))

    E2 = torch.sum(_unbatched_neohookean_energy(
        mus, lams, F.unsqueeze(1)))
    check_allclose(E1, E2)



@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_neohookean_energy_complex_deformations(device, dtype):
    N = 21
    B = 4
    C = 3
    eps = 1e-8
    F = eps * torch.rand(N, B, C, 3, 3, device=device, dtype=dtype)

    yms = 1e3 * torch.ones(N, B, C, 1, device=device)
    prs = 0.4 * torch.ones(N, B, C, 1, device=device)

    mus, lams = material_utils.to_lame(yms, prs)

    E1 = torch.tensor([0], device=device, dtype=dtype)
    for b in range(B):
        for c in range(C):
            E1 += torch.sum(_unbatched_neohookean_energy(
                mus[:, b, c, :], lams[:, b, c, :], F[:, b, c, :, :]))

    E2 = torch.sum(neohookean_elastic_material._neohookean_energy(mus, lams, F))

    check_allclose(E1, E2)



@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_neohookean_batched_gradients(device, dtype):
    N = 20
    B = 2
    eps = 1e-8

    F = torch.eye(3, device=device, dtype=dtype).expand(N, B, 3, 3)
    + eps * torch.rand(N, B, 3, 3, device=device, dtype=dtype)

    yms = 1e3 * torch.ones(N, B, 1, device=device)
    prs = 0.4 * torch.ones(N, B, 1, device=device)

    mus, lams = material_utils.to_lame(yms, prs)

    neo_grad = neohookean_elastic_material._neohookean_gradient(mus, lams, F)
    assert (neo_grad.shape[0] == N and neo_grad.shape[1]
            == B and neo_grad.shape[-2] == 3 and neo_grad.shape[-1] == 3)
    neo_grad = neo_grad.flatten()

    E0 = torch.sum(neohookean_elastic_material._neohookean_energy(mus, lams, F))

    expected_grad = torch.zeros_like(F.flatten(), device=device)
    row = 0

    # Using Finite Diff to compute the expected gradients
    for n1 in range(F.shape[0]):
        for n2 in range(F.shape[1]):
            for i in range(F.shape[2]):
                for j in range(F.shape[3]):
                    F[n1, n2, i, j] += eps
                    El = torch.sum(
                        neohookean_elastic_material._neohookean_energy(mus, lams, F))
                    F[n1, n2, i, j] -= 2 * eps
                    Er = torch.sum(
                        neohookean_elastic_material._neohookean_energy(mus, lams, F))
                    F[n1, n2, i, j] += eps
                    expected_grad[row] = (El - Er) / (2 * eps)
                    row += 1

    check_allclose(neo_grad, expected_grad, rtol=1e-2, atol=1e-2)


##################################################
###### TESTING THE WARP CODE HERE ######
######
######
######
##################################################

@pytest.fixture
def neohookean_setup(device, dtype):
    """Fixture to set up neohookean material for testing."""
    N = 20
    B = 1
    vol = 1.0

    yms = 1e3 * torch.ones(N, B, device=device)
    prs = 0.4 * torch.ones(N, B, device=device)
    integration_pt_volume = (vol/N)*torch.ones((N, 1),
                                               device=device, dtype=dtype)

    F = 2.0*torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)

    mus, lams = physics.materials.material_utils.to_lame(yms, prs)

    wp_mus = wp.from_torch(mus.flatten().contiguous(),
                           dtype=wp.dtype_from_torch(dtype))
    wp_lams = wp.from_torch(lams.flatten().contiguous(),
                            dtype=wp.dtype_from_torch(dtype))

    wp_vols = wp.from_torch(
        integration_pt_volume.flatten(), dtype=wp.float32)

    wp_neohookean_struct = physics.materials.NeohookeanElasticMaterial(
        wp_mus, wp_lams, wp_vols)

    return wp_neohookean_struct, F, mus, lams, integration_pt_volume


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_neohookean_energy(device, dtype, neohookean_setup):
    wp_neohookean_struct, defo_grads, mus, lams, vols = neohookean_setup

    expected_energy = torch.sum(_unbatched_neohookean_energy(mus, lams, defo_grads))

    wp_defo_grads = wp.from_torch(defo_grads.contiguous(), dtype=wp.mat33)
    energy = wp_neohookean_struct.energy(wp_defo_grads)

    check_allclose(
        expected_energy/defo_grads.shape[0], wp.to_torch(energy))


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_neohookean_gradient(device, dtype, neohookean_setup):
    wp_neohookean_struct, defo_grads, mus, lams, vols = neohookean_setup

    expected_grad = _unbatched_neohookean_gradient(
        mus, lams, defo_grads).reshape(-1, 9) * vols.reshape(-1, 1)

    wp_defo_grads = wp.from_torch(defo_grads.contiguous(), dtype=wp.mat33)
    wp_neohookean_grad = wp_neohookean_struct.gradient(wp_defo_grads)

    check_allclose(expected_grad.reshape(-1, 3, 3), wp.to_torch(
        wp_neohookean_grad))


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_neohookean_hessian(device, dtype, neohookean_setup):
    wp_neohookean_struct, defo_grads, mus, lams, vols = neohookean_setup

    expected_hess =  (_unbatched_neohookean_hessian(
        mus, lams, defo_grads)/defo_grads.shape[0])

    wp_defo_grads = wp.from_torch(defo_grads.contiguous(), dtype=wp.mat33)

    wp_neohookean_struct.hessian(wp_defo_grads)

    check_allclose(expected_hess, wp.to_torch(
        wp_neohookean_struct.hessians_blocks))
    


