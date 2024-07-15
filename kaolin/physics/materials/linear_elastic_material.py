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

__all__ = [
    'cauchy_strain',
    'linear_elastic_energy'
]
def cauchy_strain(defo_grad):
    r"""Calculates cauchy strain

    Args:
        defo_grad (torch.Tensor): Batched deformation gradients (denoted in literature as F) of any dimension where the last 2 dimensions are 3 x 3, of shape :math:`(\text{batch_dims}, 3, 3)`

    Returns:
        torch.Tensor: Per-primitive strain tensor, of shape :math:`(\text{batch_dim}, 3, 3)`
    """
    dimensions = defo_grad.shape
    batched_dims = dimensions[:-2]
    return 0.5 * (defo_grad.transpose(-2, -1) + defo_grad) - torch.eye(3, device=defo_grad.device)[None].expand(dimensions)

def linear_elastic_energy(mu, lam, defo_grad):
    r"""Implements a batched version of linear elastic energy. Calculate energy per-integration primitive.

    Args:
        mu (torch.Tensor): Batched lame parameter mu, of shape :math:`(\text{batch_dim}, 1)`
        lam (torch.Tensor): Batched lame parameter lambda, of shape :math:`(\text{batch_dim}, 1)`
        defo_grad (torch.Tensor): Batched deformation gradients (denoted in literature as F) of any dimension where the last 2 dimensions are 3 x 3, of shape :math:`(\text{batch_dim}, 3, 3)`

    Returns:
        torch.Tensor: Vector of per-primitive energy values, of shape :math:`(\text{batch_dim}, 1)`
    """
    dimensions = defo_grad.shape
    batched_dims = dimensions[:-2]

    # Cauchy strain matrix shape (batch_dim, 3, 3)
    Eps = cauchy_strain(defo_grad)
    batched_trace = torch.vmap(torch.trace)
    
    # Trace of cauchy strain
    trace_eps = batched_trace(Eps.reshape(batched_dims.numel(), 3, 3)).reshape(batched_dims).unsqueeze(-1)
    Eps_outerprod = torch.matmul(Eps.transpose(-2, -1), Eps)

    AA = (lam / 2) * trace_eps * trace_eps
    AB = mu*batched_trace(Eps_outerprod.reshape(batched_dims.numel(), 3, 3)).reshape(batched_dims).unsqueeze(-1) 
    return mu*batched_trace(Eps_outerprod.reshape(batched_dims.numel(), 3, 3)).reshape(batched_dims).unsqueeze(-1) + (lam / 2) * trace_eps * trace_eps
