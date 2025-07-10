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
import warp as wp


@wp.func
def _cauchy_strain_wp_func(F: wp.mat33) -> wp.mat33:
    r"""Warp function to calculate cauchy strain.

    Args:
        defo_grad (torch.Tensor): Batched deformation gradients (denoted in literature as F) of any dimension where the last 2 dimensions are 3 x 3, of shape :math:`(\text{batch_dims}, 3, 3)`

    Returns:
        torch.Tensor: Per-primitive strain tensor, of shape :math:`(\text{batch_dim}, 3, 3)`
    """
    F_transpose = wp.transpose(F)
    eye = wp.identity(n=3, dtype=float)
    return 0.5 * (F_transpose + F) - eye


@wp.func
def _linear_elastic_energy_wp_func(mu: float, lam: float, F: wp.mat33) -> float:
    r"""Implements a batched version of linear elastic energy. Calculate energy per-integration primitive. For more background information, refer to `Jernej Barbic's Siggraph Course Notes\
    <https://viterbi-web.usc.edu/~jbarbic/femdefo/sifakis-courseNotes-TheoryAndDiscretization.pdf>`_ section 3.2.

    Args:
        mu (torch.Tensor): Batched lame parameter mu, of shape :math:`(\text{batch_dim}, 1)`
        lam (torch.Tensor): Batched lame parameter lambda, of shape :math:`(\text{batch_dim}, 1)`
        defo_grad (torch.Tensor): Batched deformation gradients (denoted in literature as F) of any dimension where the last 2 dimensions are :math:`3 \times 3`, of shape :math:`(\text{batch_dim}, 3, 3)`

    Returns:
        torch.Tensor: Vector of per-primitive energy values, of shape :math:`(\text{batch_dim}, 1)`
    """
    epsilon = _cauchy_strain_wp_func(F)               # wp.mat33
    eps_trace = wp.trace(epsilon)            # float
    eps_transpose = wp.transpose(epsilon)    # wp.mat33
    eps_outerprod = eps_transpose @ epsilon  # wp.mat33
    return mu * wp.trace(eps_outerprod) + (lam / 2.0) * eps_trace * eps_trace


def _cauchy_strain(defo_grad):
    r"""Calculates cauchy strain

    Args:
        defo_grad (torch.Tensor): Batched deformation gradients (denoted in literature as F) of any dimension where the last 2 dimensions are :math:`3 \times 3`, of shape :math:`(\text{batch_dims}, 3, 3)`

    Returns:
        torch.Tensor: Per-primitive strain tensor, of shape :math:`(\text{batch_dim}, 3, 3)`
    """
    dimensions = defo_grad.shape
    batched_dims = dimensions[:-2]
    return 0.5 * (defo_grad.transpose(-2, -1) + defo_grad) - torch.eye(3, device=defo_grad.device)[None].expand(dimensions)


def _linear_elastic_energy(mu, lam, defo_grad):
    r"""Implements a batched version of linear elastic energy. Calculate energy per-integration primitive. For more background information, refer to `Jernej Barbic's Siggraph Course Notes\
    <https://viterbi-web.usc.edu/~jbarbic/femdefo/sifakis-courseNotes-TheoryAndDiscretization.pdf>`_ section 3.2.

    Args:
        mu (torch.Tensor): Batched lame parameter mu, of shape :math:`(\text{batch_dim}, 1)`
        lam (torch.Tensor): Batched lame parameter lambda, of shape :math:`(\text{batch_dim}, 1)`
        defo_grad (torch.Tensor): Batched deformation gradients (denoted in literature as F) of any dimension where the last 2 dimensions are :math:`3 \times 3`, of shape :math:`(\text{batch_dim}, 3, 3)`

    Returns:
        torch.Tensor: Vector of per-primitive energy values, of shape :math:`(\text{batch_dim}, 1)`
    """
    dimensions = defo_grad.shape
    batched_dims = dimensions[:-2]

    # Cauchy strain matrix shape (batch_dim, 3, 3)
    Eps = _cauchy_strain(defo_grad)
    batched_trace = torch.vmap(torch.trace)

    # Trace of cauchy strain
    trace_eps = batched_trace(Eps.reshape(batched_dims.numel(), 3, 3)).reshape(batched_dims).unsqueeze(-1)
    Eps_outerprod = torch.matmul(Eps.transpose(-2, -1), Eps)

    return mu * batched_trace(Eps_outerprod.reshape(batched_dims.numel(), 3, 3)).reshape(batched_dims).unsqueeze(-1) + (lam / 2) * trace_eps * trace_eps


def _linear_elastic_gradient(mu, lam, defo_grad):
    """Implements a batched version of the jacobian of linear elastic energy. Calculates gradients per-integration primitive. For more background information, refer to `Jernej Barbic's Siggraph Course Notes\
    <https://viterbi-web.usc.edu/~jbarbic/femdefo/sifakis-courseNotes-TheoryAndDiscretization.pdf>`_ section 3.2.

    Args:
        mu (torch.Tensor): Batched lame parameter mu, of shape :math:`(\text{batch_dim}, 1)`
        lam (torch.Tensor): Batched lame parameter lambda, of shape :math:`(\text{batch_dim}, 1)`
        defo_grad (torch.Tensor): Batched deformation gradients (denoted in literature as F) of any dimension where the last 2 dimensions are :math:`3 \times 3`, of shape :math:`(\text{batch_dim}, 3, 3)`

    Returns:
        torch.Tensor: Vector of per-primitive jacobians of linear elastic energy w.r.t defo_grad values, of shape :math:`(\text{batch_dim}, 9)`
    """
    dimensions = defo_grad.shape
    batched_dims = dimensions[:-2]
    # I = expanded to be defo_grad shape
    id_mat = torch.eye(3, device=mu.device).expand(batched_dims + (3, 3))
    batched_trace = torch.vmap(torch.trace)

    # F - I
    F_m_I = (defo_grad - id_mat).reshape(batched_dims.numel(), 3, 3)
    trace_F_m_I = batched_trace(F_m_I).reshape(batched_dims).unsqueeze(-1).unsqueeze(-1)
    g2 = lam.unsqueeze(-1) * trace_F_m_I * id_mat
    g1 = 2.0 * mu.unsqueeze(-1) * (defo_grad.transpose(-2, -1) + defo_grad - 2 * id_mat)
    return g1 + g2
