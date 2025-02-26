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

import pytest
import torch
from typing import Any
import warp as wp

import kaolin.physics.materials.neohookean_elastic_material as neohookean_elastic_material
import SparseSimplicits.kaolin.kaolin.physics.materials.material_utils as material_utils

wp.init()
##################################################


@wp.kernel
def elastic_kernel(mus: wp.array(dtype=Any, ndim=2),
                   lams: wp.array(dtype=Any, ndim=2),
                   Fs: wp.array(dtype=wp.mat33, ndim=2),
                   wp_e: wp.array(dtype=Any, ndim=1)):
    pt_idx, batch_idx = wp.tid()

    mu_ = mus[pt_idx, batch_idx]
    lam_ = lams[pt_idx, batch_idx]
    F_ = Fs[pt_idx, batch_idx]

    E = neohookean_elastic_material.wp_neohookean_energy(mu_, lam_, F_)
    wp.atomic_add(wp_e, batch_idx, E)
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
        neohookean_elastic_material.unbatched_neohookean_energy(mus, lams, F))

    E2 = torch.sum(neohookean_elastic_material.unbatched_neohookean_energy(
        mus, lams, F.unsqueeze(1)))
    assert torch.allclose(E1, E2)


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_wp_neohookean_energy(device, dtype):
    N = 20
    B = 4

    F = torch.eye(3, device=device, dtype=dtype).expand(N, B, 3, 3)

    yms = 1e3 * torch.ones(N, B, device=device)
    prs = 0.4 * torch.ones(N, B, device=device)

    mus, lams = material_utils.to_lame(yms, prs)

    E1 = torch.tensor(0, device=device, dtype=dtype)

    wp_e = wp.zeros(B, dtype=wp.dtype_from_torch(dtype))
    wp_F = wp.from_torch(F.contiguous(), dtype=wp.mat33)
    wp_mus = wp.from_torch(mus.contiguous(), dtype=wp.dtype_from_torch(dtype))
    wp_lams = wp.from_torch(
        lams.contiguous(), dtype=wp.dtype_from_torch(dtype))

    wp.launch(
        kernel=elastic_kernel,
        dim=(N, B),
        inputs=[
            wp_mus,  # mus: wp.array(dtype=float),   ; shape (N,B,)
            wp_lams,  # lams: wp.array(dtype=float),  ; shape (N,B,)
            wp_F  # defo_grads: wp.array(dtype=wp.mat33),  ; shape (N,B,3,3)
        ],
        outputs=[wp_e],  # out_e: wp.array(dtype=float)  ; shape (B,)
        adjoint=False
    )
    E2 = wp.to_torch(wp_e).sum()
    assert torch.allclose(E1, E2)


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
            E1 += torch.sum(neohookean_elastic_material.unbatched_neohookean_energy(
                mus[:, b, c, :], lams[:, b, c, :], F[:, b, c, :, :]))

    E2 = torch.sum(neohookean_elastic_material.neohookean_energy(mus, lams, F))

    assert torch.allclose(E1, E2)


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

    neo_grad = neohookean_elastic_material.neohookean_gradient(mus, lams, F)
    assert (neo_grad.shape[0] == N and neo_grad.shape[1] ==
            B and neo_grad.shape[-2] == 3 and neo_grad.shape[-1] == 3)
    neo_grad = neo_grad.flatten()

    E0 = torch.sum(neohookean_elastic_material.neohookean_energy(mus, lams, F))

    expected_grad = torch.zeros_like(F.flatten(), device=device)
    row = 0

    # Using Finite Diff to compute the expected gradients
    for n1 in range(F.shape[0]):
        for n2 in range(F.shape[1]):
            for i in range(F.shape[2]):
                for j in range(F.shape[3]):
                    F[n1, n2, i, j] += eps
                    El = torch.sum(
                        neohookean_elastic_material.neohookean_energy(mus, lams, F))
                    F[n1, n2, i, j] -= 2 * eps
                    Er = torch.sum(
                        neohookean_elastic_material.neohookean_energy(mus, lams, F))
                    F[n1, n2, i, j] += eps
                    expected_grad[row] = (El - Er) / (2 * eps)
                    row += 1

    assert torch.allclose(neo_grad, expected_grad, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_neohookean_gradients(device, dtype):
    N = 20
    B = 1
    eps = 1e-8

    F = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)
    + eps * torch.rand(N, B, 3, 3, device=device, dtype=dtype)

    yms = 1e3 * torch.ones(N, 1, device=device)
    prs = 0.4 * torch.ones(N, 1, device=device)

    mus, lams = material_utils.to_lame(yms, prs)

    neo_grad = neohookean_elastic_material.unbatched_neohookean_gradient(
        mus, lams, F)
    assert (neo_grad.shape[0] == N and neo_grad.shape[1]
            == 3 and neo_grad.shape[2] == 3)
    neo_grad = neo_grad.flatten()

    E0 = torch.sum(
        neohookean_elastic_material.unbatched_neohookean_energy(mus, lams, F))

    expected_grad = torch.zeros_like(F.flatten(), device=device)
    row = 0

    # Using Finite Diff to compute the expected gradients
    for n in range(F.shape[0]):
        for i in range(F.shape[1]):
            for j in range(F.shape[2]):
                F[n, i, j] += eps
                El = torch.sum(
                    neohookean_elastic_material.unbatched_neohookean_energy(mus, lams, F))
                F[n, i, j] -= 2 * eps
                Er = torch.sum(
                    neohookean_elastic_material.unbatched_neohookean_energy(mus, lams, F))
                F[n, i, j] += eps
                expected_grad[row] = (El - Er) / (2 * eps)
                row += 1

    assert torch.allclose(neo_grad, expected_grad, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_neohookean_hessian_vs_fd(device, dtype):
    N = 5
    B = 1
    eps = 1e-3

    F = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)

    yms = 1e3 * torch.ones(N, 1, device=device)
    prs = 0.4 * torch.ones(N, 1, device=device)

    mus, lams = material_utils.to_lame(yms, prs)

    neo_hess = neohookean_elastic_material.unbatched_neohookean_hessian(
        mus, lams, F)

    expected_hess = torch.zeros(N, 9, 9, device=device, dtype=dtype)
    row = 0

    # Using Finite Diff to compute the expected hessian
    for n in range(F.shape[0]):
        for i in range(F.shape[1]):
            for j in range(F.shape[2]):
                F[n, i, j] += eps
                Gl = neohookean_elastic_material.unbatched_neohookean_gradient(
                    mus[n].unsqueeze(0), lams[n].unsqueeze(0), F[n].unsqueeze(0)).flatten()
                F[n, i, j] -= 2 * eps
                Gr = neohookean_elastic_material.unbatched_neohookean_gradient(
                    mus[n].unsqueeze(0), lams[n].unsqueeze(0), F[n].unsqueeze(0)).flatten()
                F[n, i, j] += eps
                expected_hess[n, 3 * i + j, :] = (Gl - Gr) / (2 * eps)
                row += 1

    assert torch.allclose(neo_hess, expected_hess, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_neohookean_hessian_vs_autograd(device, dtype):
    N = 5
    B = 1

    F = 2*torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)

    yms = 1e3 * torch.ones(N, 1, device=device)
    prs = 0.4 * torch.ones(N, 1, device=device)

    mus, lams = material_utils.to_lame(yms, prs)

    # N x 9 x 9 (the block diags)
    neo_hess = neohookean_elastic_material.unbatched_neohookean_hessian(
        mus, lams, F)

    # 9N x 9N (the full hessian with a bunch of zeros)
    autograd_hessian = torch.autograd.functional.jacobian(
        lambda x: neohookean_elastic_material.unbatched_neohookean_gradient(mus, lams, x), F)
    autograd_hessian = autograd_hessian.reshape(9 * N, 9 * N)

    # Make sure the block diags match up
    for n in range(N):
        assert torch.allclose(
            neo_hess[n], autograd_hessian[9 * n:9 * n + 9, 9 * n:9 * n + 9], rtol=1e-1, atol=1e-1)
        # zero out the block
        autograd_hessian[9 * n:9 * n + 9, 9 * n:9 * n + 9] *= 0

    # Make sure the rest of the matrix is zeros
    assert torch.allclose(torch.zeros_like(autograd_hessian),
                          autograd_hessian, rtol=1e-1, atol=1e-1)
