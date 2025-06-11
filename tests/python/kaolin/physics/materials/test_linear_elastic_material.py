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
import warp as wp
from typing import Any

import kaolin.physics.materials.linear_elastic_material as linear_elastic_material
import SparseSimplicits.kaolin.kaolin.physics.materials.material_utils as material_utils

wp.init()
###################################


def linear_elastic_gradient_equivalent(mu, lam, defo_grad):
    """Implements a batched version of the jacobian of linear elastic energy. Calculates gradients per-integration primitive. For more background information, refer to `Jernej Barbic's Siggraph Course Notes\
    <https://viterbi-web.usc.edu/~jbarbic/femdefo/sifakis-courseNotes-TheoryAndDiscretization.pdf>`_ section 3.2.

    Args:
        mu (torch.Tensor): Batched lame parameter mu, of shape :math:`(\text{batch_dim}, 1)`
        lam (torch.Tensor): Batched lame parameter lambda, of shape :math:`(\text{batch_dim}, 1)`
        defo_grad (torch.Tensor): Batched deformation gradients (denoted in literature as F) of any dimension where the last 2 dimensions are 3 x 3, of shape :math:`(\text{batch_dim}, 3, 3)`

    Returns:
        torch.Tensor: Vector of per-primitive jacobians of linear elastic energy w.r.t defo_grad values, of shape :math:`(\text{batch_dim}, 9)`
    """
    dimensions = defo_grad.shape
    batched_dims = dimensions[:-2]
    id_mat = torch.eye(3, device=mu.device).expand(batched_dims + (3, 3))

    batched_trace = torch.vmap(torch.trace)

    # Cauchy strain matrix shape (batch_dim, 3, 3)
    Eps = linear_elastic_material.cauchy_strain(defo_grad)

    # Reshape Eps into [-1, 3, 3] tensor
    batchedEps = Eps.reshape(batched_dims.numel(), 3, 3)
    trace_eps = batched_trace(batchedEps).reshape(
        batched_dims).unsqueeze(-1).unsqueeze(-1)
    g = 2.0 * mu.unsqueeze(-1) * Eps + lam.unsqueeze(-1) * trace_eps * id_mat
    return g


@wp.kernel
def elastic_kernel(mus: wp.array(dtype=Any, ndim=2),
                   lams: wp.array(dtype=Any, ndim=2),
                   Fs: wp.array(dtype=wp.mat33, ndim=2),
                   wp_e: wp.array(dtype=Any, ndim=1)):
    pt_idx, batch_idx = wp.tid()

    mu_ = mus[pt_idx, batch_idx]
    lam_ = lams[pt_idx, batch_idx]
    F_ = Fs[pt_idx, batch_idx]

    E = linear_elastic_material.wp_linear_elastic_energy(mu_, lam_, F_)
    wp.atomic_add(wp_e, batch_idx, E)
############################


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_linear_energy(device, dtype):
    N = 20
    B = 4

    F = torch.eye(3, device=device, dtype=dtype).expand(N, B, 3, 3)

    yms = 1e3 * torch.ones(N, B, 1, device=device)
    prs = 0.4 * torch.ones(N, B, 1, device=device)

    mus, lams = material_utils.to_lame(yms, prs)

    E1 = torch.tensor(0, device=device, dtype=dtype)

    E2 = torch.sum(linear_elastic_material.linear_elastic_energy(mus, lams, F))
    assert torch.allclose(E1, E2)


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_wp_linear_energy(device, dtype):
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
def test_linear_gradients(device, dtype):
    N = 20
    B = 1
    eps = 1e-7
    F = torch.eye(3, device=device, dtype=dtype).expand(N, B, 3, 3) + \
        eps * torch.rand(N, B, 3, 3, device=device, dtype=dtype)

    yms = 1e3 * torch.ones(N, B, 1, device=device)
    prs = 0.4 * torch.ones(N, B, 1, device=device)

    mus, lams = material_utils.to_lame(yms, prs)

    neo_grad = linear_elastic_material.linear_elastic_gradient(mus, lams, F)
    neo_grad_equivalent = linear_elastic_gradient_equivalent(mus, lams, F)
    assert torch.allclose(neo_grad_equivalent, neo_grad, rtol=1e-2, atol=1e-2)
    assert (neo_grad.shape[0] == N and neo_grad.shape[1] ==
            B and neo_grad.shape[-2] == 3 and neo_grad.shape[-1] == 3)
    neo_grad = neo_grad.flatten()

    E0 = torch.sum(linear_elastic_material.linear_elastic_energy(mus, lams, F))

    expected_grad = torch.zeros_like(F.flatten(), device=device)
    row = 0

    # Using Finite Diff to compute the expected gradients
    for n in range(F.shape[0]):
        for i in range(F.shape[1]):
            for j in range(F.shape[2]):
                F[n, i, j] += eps
                El = torch.sum(
                    linear_elastic_material.linear_elastic_energy(mus, lams, F))
                F[n, i, j] -= 2 * eps
                Er = torch.sum(
                    linear_elastic_material.linear_elastic_energy(mus, lams, F))
                F[n, i, j] += eps
                expected_grad[row] = (El - Er) / (2 * eps)
                row += 1

    assert torch.allclose(neo_grad, expected_grad, rtol=1e-2, atol=1e-2)


# @pytest.mark.parametrize('device', ['cuda', 'cpu'])
# @pytest.mark.parametrize('dtype', [torch.float, torch.double])
# def test_linear_hessian_vs_fd(device, dtype):
#     N = 5
#     B = 1
#     eps = 1e-3
#     F = torch.eye(3, device=device, dtype=dtype).expand(N,3,3) # + eps*torch.rand(N, B, 3,3, device=device, dtype=dtype)
#     id_mat = torch.eye(9, device=device, dtype=dtype)

#     yms = 1e3*torch.ones(N,1, device=device)
#     prs = 0.4*torch.ones(N,1, device=device)

#     mus, lams = material_utils.to_lame(yms, prs)

#     neo_hess = linear_elasticity.hessian(mus, lams, F, id_mat)

#     expected_hess = torch.zeros(N, 9,9, device=device, dtype=dtype)
#     row = 0

#     #Using Finite Diff to compute the expected hessian
#     for n in range(F.shape[0]):
#         for i in range(F.shape[1]):
#             for j in range(F.shape[2]):
#                 F[n,i,j] += eps
#                 Gl = linear_elasticity.gradient(mus[n].unsqueeze(0), lams[n].unsqueeze(0), F[n].unsqueeze(0)).flatten()
#                 F[n,i,j] -= 2*eps
#                 Gr = linear_elasticity.gradient(mus[n].unsqueeze(0), lams[n].unsqueeze(0), F[n].unsqueeze(0)).flatten()
#                 F[n,i,j] += eps
#                 expected_hess[n, 3*i + j, :] = (Gl - Gr)/(2*eps)
#                 row +=1

#     assert torch.allclose(neo_hess, expected_hess, rtol=1e-1, atol=1e-1)


# @pytest.mark.parametrize('device', ['cuda', 'cpu'])
# @pytest.mark.parametrize('dtype', [torch.float, torch.double])
# def test_linear_hessian_vs_autograd(device, dtype):
#     N = 5
#     B = 1
#     eps = 1e-3
#     F = torch.eye(3, device=device, dtype=dtype).expand(N,3,3) # + eps*torch.rand(N, B, 3,3, device=device, dtype=dtype)
#     id_mat = torch.eye(9, device=device, dtype=dtype)

#     yms = 1e3*torch.ones(N,1, device=device)
#     prs = 0.4*torch.ones(N,1, device=device)

#     mus, lams = material_utils.to_lame(yms, prs)

#     # N x 9 x 9 (the block diags)
#     neo_hess = linear_elasticity.hessian(mus, lams, F, id_mat)

#     # 9N x 9N (the full hessian with a bunch of zeros)
#     autograd_hessian = torch.autograd.functional.jacobian(lambda x: linear_elasticity.gradient(mus, lams, x), F)
#     autograd_hessian = autograd_hessian.reshape(9*N, 9*N)

#     # Make sure the block diags match up
#     for n in range(N):
#         assert torch.allclose(neo_hess[n], autograd_hessian[9*n:9*n+9, 9*n:9*n+9], rtol=1e-1, atol=1e-1)
#         #zero out the block
#         autograd_hessian[9*n:9*n+9, 9*n:9*n+9] *= 0

#     # Make sure the rest of the matrix is zeros
#     assert torch.allclose(torch.zeros_like(autograd_hessian), autograd_hessian, rtol=1e-1, atol=1e-1)
