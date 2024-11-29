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
from functools import partial

from kaolin.physics.utils.finite_diff import finite_diff_jac
from kaolin.physics.simplicits.utils import standard_lbs, weight_function_lbs

__all__ = [
    'lumped_mass_matrix',
    'lbs_matrix',
    'jacobian_dF_dz_const_handle',
    'jacobian_dF_dz',
    'jacobian_dx_dz'
]


def lumped_mass_matrix(rhos, total_volume, dim=3):
    r"""Calculate the lumped mass matrix of an object sampled via points with spatially uniform sampling, and potentially spatially varying density

    Args:
        rhos (torch.Tensor): Point-wise vector of densities, of shape :math:`(\text{num_samples})`
        total_volume (float): Total volume of object in :math:`m^3`
        dim (int, optional): Spatial dimensions. Defaults to 3.

    Returns:
        torch.Tensor: Diagonal mass matrix of size, of shape :math:`(3*num_samples, 3*num_samples)`
        torch.Tensor: Diagonal INVERSE mass matrix of size, of shape :math:`(3*num_samples, 3*num_samples)`
    """
    # Assumes uniform sample pt distributions over the object
    # Does NOT assume uniform density
    vol_per_sample = total_volume / rhos.shape[0]
    pt_wise_mass = rhos * vol_per_sample
    return torch.diag(pt_wise_mass.repeat_interleave(dim)), torch.diag(1.0 / pt_wise_mass.repeat_interleave(dim))


def lbs_matrix(x0, w):
    r"""Encodes the lbs equation :math:`x_i = sum(w(x^0_i)_j * T_j * \begin{pmatrix}x^0_i\\1\end{pmatrix},\; \forall \; j=0...\|handles\|)` for all pts into matrix B such that flatten(x) = B(x0, w(x0))*flatten(T)

    Args:
        x0 (torch.Tensor): Input points, of shape :math:`(\text{num_samples}, 3)`
        w (torch.Tensor): Weights, of shape :math:`(\text{num_samples}, \text{num_handles})`

    Returns:
        torch.Tensor: Matrix that encodes the lbs transformation, given a set of vertices and corresponding weights, shape :math:`(3*\text{num_samples}, 12*\text{num_handles})`
    """
    num_samples = x0.shape[0]  # N
    num_handles = w.shape[1]  # H

    # Shape (N, 1)
    ones_column = torch.ones(x0.shape[0], 1).to(x0.device)

    # Shape (N, 4) ; x0 homogeneous coords
    x03 = torch.cat((x0, ones_column), dim=1)
    # Shape (3N, 12H) ; 3*num_samples x 4*3*num_handles
    # [[ x1, y1, z1, 1, ... x1, y1, z1, 1 ],
    #  [ x1, y1, z1, 1, ... x1, y1, z1, 1 ],
    #  [ x1, y1, z1, 1, ... x1, y1, z1, 1 ],
    #  [       |        ...        |      ],
    #  [ xN, yN, zN, 1, ... xN, yN, zN, 1 ],
    #  [ xN, yN, zN, 1, ... xN, yN, zN, 1 ],
    #  [ xN, yN, zN, 1, ... xN, yN, zN, 1 ]]
    x03reps = x03.repeat_interleave(3, dim=0).repeat((1, 3 * num_handles))
    # Shape (3N, 12H):
    # [[ w11, w11, w11, w11, w11 (12 times) ... w1H, w1H, w1H, w1H, w1H (12 times) ],
    #  [ w11, w11, w11, w11, w11 (12 times) ... w1H, w1H, w1H, w1H, w1H (12 times) ],
    #  [ w11, w11, w11, w11, w11 (12 times) ... w1H, w1H, w1H, w1H, w1H (12 times) ],
    #  [              |                   ...                     |                ],
    #  [ wN1, wN1, wN1, wN1, wN1 (12 times) ... wNH, wNH, wNH, wNH, N1H (12 times) ],
    #  [ wN1, wN1, wN1, wN1, wN1 (12 times) ... wNH, wNH, wNH, wNH, N1H (12 times) ],
    #  [ wN1, wN1, wN1, wN1, wN1 (12 times) ... wNH, wNH, wNH, wNH, N1H (12 times) ]]
    w_reps = w.repeat_interleave(12, dim=1).repeat_interleave(3, dim=0)
    # Shape (3N, 12H): Hadamard product (element-wise multiplication)
    w_x03reps = torch.mul(w_reps, x03reps)
    # Shape (3N, 3H) ; a (N,H) block matrix of 3x3 identity submats
    # [[ 1, 0, 0, 1, 0, 0, ... 1, 0, 0],
    #  [ 0, 1, 0, 0, 1, 0, ... 0, 1, 0],
    #  [ 0, 0, 1, 0, 0, 1, ... 0, 0, 1],
    #  [         |           ...   |  ],
    #  [ 1, 0, 0, 1, 0, 0, ... 1, 0, 0],
    #  [ 0, 1, 0, 0, 1, 0, ... 0, 1, 0],
    #  [ 0, 0, 1, 0, 0, 1, ... 0, 0, 1]]
    B_setup = torch.kron(
        torch.ones([num_samples, 1], device=x0.device),
        torch.eye(3, device=x0.device)
    ).repeat((1, num_handles))
    # Shape (3N, 12H) ; a (N,H) block matrix of 3x12 binary submats
    # [[ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 ... 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [ 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 ... 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    #  [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 ... 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    #  [                |                   ...                  |                ],
    #  [ 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 ... 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    #  [ 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0 ... 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    #  [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 ... 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    B_mask = torch.repeat_interleave(B_setup, 4, dim=1)

    # Shape (3N, 12H) ; a (N,H) block matrix of (3,12) submats
    # [[ w11*x1, w11*y1, w11*z1, w11, 0,      0,      0,      0,  0,      0,      0,      0   .. w1H*x1, w1H*y1, ..]
    #  [ 0,      0,      0,      0,   w11*x1, w11*y1, w11*z1, w11 0,      0,      0,      0   .. 0,      0,      ..]
    #  [ 0,      0,      0,      0,   0,      0,      0,      0,  w11*x1, w11*y1, w11*z1, w11 .. 0,      0,      ..]
    #  [                                        |                                             ..        |          ]
    #  [wN1*xN,  wN1*yN, wN1*zN, wN1, 0,      0,      0,      0,  0,      0,      0,      0,  .. wNH*xN, wNH*yN  ..]
    #  [ 0,      0,      0,      0,   wN1*xN, wN1*yN, wN1*zN, w1N 0,      0,      0,      0   .. 0,      0,      ..]
    #  [ 0,      0,      0,      0,   0,      0,      0,      0,  wN1*xN, wN1*yN, wN1*zN, wN1 .. 0,      0,      ..]]
    B = torch.mul(B_mask, w_x03reps)
    return B


def jacobian_dF_dz_const_handle(model, x0, z):
    r"""Calculates dF/dz for the skinning weights model (network) and an extra dimension for the constant handle

    Args:
        model (nn.module): Simplicits object network without the constant handle
        x0 (torch.Tensor): Matrix of sample points, of shape :math:`(\text{num_samples}, 3)` 
        z (torch.Tensor): Vector of flattened transforms, of shape :math:`(12*\text{num_samples})`

    Returns:
        torch.Tensor: Jacobian matrix, of shape :math:`(9*\text{num_samples}, 12*\text{num_handles})`
    """
    num_samples = x0.shape[0]

    zeros_vec3_column = torch.zeros((num_samples, 1, 3), device=x0.device)
    ones_col = torch.ones((num_samples, 1), device=x0.device)
    weights = model(x0)
    weights = torch.cat((model(x0), ones_col), dim=1)
    grad_weights = finite_diff_jac(model, x=x0)
    grad_weights = torch.cat([grad_weights.squeeze(), zeros_vec3_column], dim=1)
    x0_h = torch.cat((x0, torch.ones(x0.shape[0], 1, device=x0.device, dtype=x0.dtype)), dim=1).detach()

    def _compute_F(Ts):
        # W0_i, dW0_i: lazy, just avoiding neural net evaluations
        def defgrad(x0h, W0_i, dW0_i):
            T_w = torch.sum(W0_i.unsqueeze(dim=1).unsqueeze(dim=2) * Ts, dim=0)
            x_t = torch.tensordot(Ts, x0h.unsqueeze(dim=0), dims=([2], [1]))
            dT = torch.sum(torch.bmm(x_t, dW0_i.unsqueeze(dim=1)), dim=0)
            return dT + T_w.squeeze()[0:3, 0:3] + torch.eye(3, device=x0_h.device)

        # compute F for each sample point
        return torch.vmap(defgrad)(x0_h, weights, grad_weights)

    def _reshape_and_compute_F(z_):
        transforms = z_.reshape(-1, 3, 4)
        deformation_gradient = _compute_F(transforms)
        return deformation_gradient.reshape(9 * num_samples)

    dF_dz = torch.autograd.functional.jacobian(_reshape_and_compute_F, z).squeeze(-1)

    return dF_dz


def jacobian_dF_dz(model, x0, z):
    r"""Calculates jacobian dF/dz

    Args:
        model (nn.Module): Simplicits object network + constant handle
        x0 (torch.Tensor): Matrix of sample points, of shape :math:`(\text{num_samples}, 3)`
        z (torch.Tensor): Vector of flattened transforms, of shape :math:`(12*\text{num_samples})`

    Returns:
        torch.Tensor: Jacobian matrix, of shape :math:`(9*\text{num_samples}, 12*\text{num_handles})`
    """
    num_samples = x0.shape[0]

    def compute_defo_grad1(z):
        partial_weight_fcn_lbs = partial(weight_function_lbs, tfms=z.reshape(-1, 3, 4).unsqueeze(0), fcn=model)
        # N x 3 x 3 Deformation gradients
        defo_grads = finite_diff_jac(partial_weight_fcn_lbs, x0, eps=1e-7)
        return defo_grads

    dF_dz = torch.autograd.functional.jacobian(lambda x: compute_defo_grad1(x).reshape(9 * num_samples), z.flatten())
    return dF_dz


def jacobian_dx_dz(model, x0, z):
    r"""Calculates jacobian dx/dz

    Args:
        model (nn.Module): Simplicits object network + constant handle
        x0 (torch.Tensor): Matrix of sample points, of shape :math:`(\text{num_samples}, 3)`
        z (torch.Tensor): Vector of flattened transforms, of shape :math:`(12*\text{num_samples})`

    Returns:
        torch.Tensor: Jacobian matrix, of shape :math:`(3*\text{num_samples}, 12*\text{num_handles})`
    """
    dx_dz = torch.autograd.functional.jacobian(lambda x: weight_function_lbs(
        x0, tfms=x.reshape(-1, 3, 4).unsqueeze(0), fcn=model).flatten(), z.flatten())
    return dx_dz
