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
import warp as wp


import kaolin.physics.materials.utils as material_utils
import kaolin.physics.materials.linear_elastic_material as linear_elastic_material
import kaolin.physics.materials.neohookean_elastic_material as neohookean_elastic_material
from kaolin.physics.simplicits.losses import loss_ortho

# Type defs
mat34f = wp.types.matrix(shape=(3, 4), dtype=wp.float32)
mat34 = mat34f

__all__ = [
    'loss_elastic_warp',
    'compute_losses_warp'
]


@wp.func
def _elastic_energy(F: wp.mat33f, mu: float, lam: float, interp_val: float) -> float:
    r""" Linearly blended elastic energy term of (linear elastic, neo-hookean) """
    En = interp_val * neohookean_elastic_material.wp_neohookean_energy(mu, lam, F)
    El = (1.0 - interp_val) * linear_elastic_material.wp_linear_elastic_energy(mu, lam, F)
    return En + El


@wp.func
def _map_x(
    x0: wp.vec3,
    Ts: wp.array(dtype=mat34, ndim=2),    # shape (B, H)
    w: wp.array(dtype=float, ndim=3),     # shape (N, 6, H)
    pt_idx: int,
    transform_idx: int,
    fd_idx: int,
    H: int
) -> wp.vec3:
    r"""Get deformed points

    """
    x0_hom = wp.vec4(x0[0], x0[1], x0[2], 1.0)
    x__map_x0 = wp.vec3()
    for handle_idx in range(H):
        Ts_ = Ts[transform_idx, handle_idx]
        w_ = w[pt_idx, fd_idx, handle_idx]
        x__map_x0 += w_ * (Ts_ @ x0_hom)
    return x0 + x__map_x0


@wp.func
def _finite_difference_defo_grad(
    x0: wp.vec3,
    Ts: wp.array(dtype=mat34, ndim=2),    # shape (B, H)
    w: wp.array(dtype=float, ndim=3),     # shape (N, 6, H)
    pt_idx: int,
    transform_idx: int,
    H: int,
    eps: float
) -> wp.mat33:
    """Finite difference deformation gradient
    """
    delta = wp.sqrt(eps)
    x0_plus_dx = _map_x(x0 + wp.vec3(delta, 0.0, 0.0), Ts, w, pt_idx, transform_idx, 0, H)
    x0_minus_dx = _map_x(x0 - wp.vec3(delta, 0.0, 0.0), Ts, w, pt_idx, transform_idx, 3, H)
    x0_plus_dy = _map_x(x0 + wp.vec3(0.0, delta, 0.0), Ts, w, pt_idx, transform_idx, 1, H)
    x0_minus_dy = _map_x(x0 - wp.vec3(0.0, delta, 0.0), Ts, w, pt_idx, transform_idx, 4, H)
    x0_plus_dz = _map_x(x0 + wp.vec3(0.0, 0.0, delta), Ts, w, pt_idx, transform_idx, 2, H)
    x0_minus_dz = _map_x(x0 - wp.vec3(0.0, 0.0, delta), Ts, w, pt_idx, transform_idx, 5, H)

    x0_dx = x0_plus_dx - x0_minus_dx
    x0_dy = x0_plus_dy - x0_minus_dy
    x0_dz = x0_plus_dz - x0_minus_dz

    jacobian = wp.mat33(
        x0_dx[0], x0_dx[1], x0_dx[2],
        x0_dy[0], x0_dy[1], x0_dy[2],
        x0_dz[0], x0_dz[1], x0_dz[2]
    )
    jacobian /= (2.0 * delta)
    jacobian = wp.transpose(jacobian)
    return jacobian


@wp.kernel
def _warp_elastic_loss(
    x0: wp.array(dtype=wp.vec3),          # shape (N,)
    mus: wp.array(dtype=float, ndim=2),           # shape (N,)
    lams: wp.array(dtype=float, ndim=2),          # shape (N,)
    Ts: wp.array(dtype=mat34, ndim=2),    # shape (B, H)
    w: wp.array(dtype=float, ndim=3),     # shape (N, 6, H)
    H: int,
    interp_val: float,
    eps: float,
    out_E: wp.array(dtype=float)  # shape (B,)
):
    r"""Accumulate elastic losses into out_E
    """
    pt_idx, transform_idx = wp.tid()
    x0_ = x0[pt_idx]             # wp.vec3
    mu_ = mus[pt_idx, transform_idx]            # float
    lam_ = lams[pt_idx, transform_idx]          # float

    F = _finite_difference_defo_grad(x0_, Ts, w, pt_idx, transform_idx, H, eps)    # wp.mat33
    E = _elastic_energy(F, mu_, lam_, interp_val)  # float
    wp.atomic_add(out_E, transform_idx, E)


class _EnergyPotential(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x0, mus, lams, Ts, fd_ws, interp_val, eps=1e-7):
        N = x0.shape[0]  # Number of sampled points
        B = Ts.shape[0]  # Number of sample transformations
        H = Ts.shape[1]  # Number of handles

        ctx.x0 = wp.from_torch(x0.contiguous(), dtype=wp.vec3)
        ctx.mus = wp.from_torch(mus.contiguous())
        ctx.lams = wp.from_torch(lams.contiguous())
        ctx.Ts = wp.from_torch(Ts.contiguous(), dtype=mat34)
        ctx.fd_ws = wp.from_torch(fd_ws.contiguous(), requires_grad=True)
        ctx.H = H
        ctx.interp_val = interp_val
        ctx.eps = eps
        ctx.out_e = wp.zeros(B, dtype=float, requires_grad=True)

        wp.launch(
            kernel=_warp_elastic_loss,
            dim=(N, B),
            inputs=[
                ctx.x0,  # x0: wp.array(dtype=wp.vec3),  ; shape (N,)
                ctx.mus,  # mus: wp.array(dtype=float),   ; shape (N,)
                ctx.lams,  # lams: wp.array(dtype=float),  ; shape (N,)
                ctx.Ts,  # Ts: wp.array(dtype=mat34),    ; shape (B, H)
                ctx.fd_ws,  # w: wp.array(dtype=float),     ; shape (N, 6, H)
                ctx.H,  # H: int
                ctx.interp_val,  # interp_val: float
                ctx.eps  # eps: float
            ],
            outputs=[ctx.out_e],  # out_e: wp.array(dtype=float)  ; shape (B,)
        )

        return wp.to_torch(ctx.out_e)

    @staticmethod
    def backward(ctx, adj_e):

        N = ctx.x0.shape[0]  # Number of sampled points
        B = ctx.Ts.shape[0]  # Number of sample transformations
        # map incoming Torch grads to our output variables
        ctx.out_e.grad = wp.from_torch(adj_e.contiguous())

        wp.launch(
            kernel=_warp_elastic_loss,
            dim=(N, B),
            inputs=[
                ctx.x0,  # x0: wp.array(dtype=wp.vec3),  ; shape (N,)
                ctx.mus,  # mus: wp.array(dtype=float),   ; shape (N,)
                ctx.lams,  # lams: wp.array(dtype=float),  ; shape (N,)
                ctx.Ts,  # Ts: wp.array(dtype=mat34),    ; shape (B, H)
                ctx.fd_ws,  # w: wp.array(dtype=float),     ; shape (N, 6, H)
                ctx.H,  # H: int
                ctx.interp_val,  # interp_val: float
                ctx.eps  # eps: float
            ],
            outputs=[ctx.out_e],  # out_e: wp.array(dtype=float)  ; shape (B,)
            adj_inputs=[None, None, None, None, ctx.fd_ws.grad, ctx.H, ctx.interp_val, ctx.eps],
            adj_outputs=[ctx.out_e.grad],
            adjoint=True
        )

        # return adjoint w.r.t. inputs: x0, mus, lams, Ts, fd_ws, interp_val, eps
        return (None, None, None, None, wp.to_torch(ctx.fd_ws.grad), None, None)


def _prepare_finite_diff_w_bounds(x, model):
    r"""Returns batched values for :math:`(model(x+eps), model(x-eps))` to be use in central finite difference jacobian for deformation gradient

    Args:
        x (torch.Tensor): Input points in :math:`(\text{num_samples, dim})`
        model (nn.Module): Network as a function

    Returns:
        torch.Tensor: Jacobian of size :math:`(\text{num_samples, 2*dim, num_handles})` 
    """
    # Controls numerical accuracy of finite-difference
    eps = 1e-7
    delta = torch.sqrt(torch.tensor(eps, device='cuda:0'))
    finite_diff_bounds = x[:, None].repeat(1, 6, 1)
    finite_diff_bounds[:, 0, 0] += delta
    finite_diff_bounds[:, 1, 1] += delta
    finite_diff_bounds[:, 2, 2] += delta
    finite_diff_bounds[:, 3, 0] -= delta
    finite_diff_bounds[:, 4, 1] -= delta
    finite_diff_bounds[:, 5, 2] -= delta

    batch_dims, coord_dims = finite_diff_bounds.shape[:-1], finite_diff_bounds.shape[-1]
    finite_diff_bounds = finite_diff_bounds.reshape(-1, coord_dims)
    fd_ws = model(finite_diff_bounds)
    fd_ws = fd_ws.reshape(*batch_dims, -1)
    return fd_ws


def loss_elastic_warp(model, pts, yms, prs, rhos, transforms, appx_vol, interp_step):
    r"""Calculate a version of simplicits elastic loss for training. This version uses Nvidia's Warp .

    Args:
        model (nn.Module): Simplicits object network
        pts (torch.Tensor): Tensor of sample points in R^dim, for now dim=3, of shape :math:`(\text{num_samples}, \text{dim})`
        yms (torch.Tensor): Length pt-wise youngs modulus, of shape :math:`(\text{num_samples})`
        prs (torch.Tensor): Length pt-wise poisson ratios, of shape :math:`(\text{num_samples})`
        rhos (torch.Tensor): Length pt-wise density, of shape :math:`(\text{num_samples})`
        transforms (torch.Tensor): Batch of sample transformations, of shape :math:`(\text{batch_size}, \text{num_handles}, \text{dim}, \text{dim}+1)`
        appx_vol (float): Approximate (or exact) volume of object (in :math:`m^3`)
        interp_step (float): Length interpolation schedule for neohookean elasticity (0%->100%)

    Returns:
        torch.Tensor: Elastic loss, single value tensor
    """

    mus, lams = material_utils.to_lame(yms, prs)

    # shape (N, B, 3, 3)
    N, B = pts.shape[0], transforms.shape[0]  # pt_wise_Fs.shape[0:2]

    # shape (N, B, 1)
    mus = mus.expand(N, B)  # .unsqueeze(-1)
    lams = lams.expand(N, B)  # .unsqueeze(-1)

    # weighted average (since we uniformly sample, this is uniform for now)
    # Holds the weight evaluations from the MLP, for each of the coordinates we need to evaluate
    # finite-differences.
    # i.e. for center-difference these are the MLP outputs for the following inputs:
    #     x + dx, x + dy, x + dz,
    #     x - dx, x - dy, x - dz
    fd_ws = _prepare_finite_diff_w_bounds(pts, model)
    fd_ws.retain_grad()
    return (appx_vol / pts.shape[0]) * (torch.sum(_EnergyPotential.apply(pts, mus, lams, transforms, fd_ws, interp_step, 1e-7)))


def compute_losses_warp(model, normalized_pts, yms, prs, rhos, en_interp, batch_size, num_handles, appx_vol, num_samples, le_coeff, lo_coeff):
    r""" Perform a step of the simplicits training process

    Args:
        model (nn.module): Simplicits network
        normalized_pts (torch.Tensor): Spatial points in :math:`R^3`, of shape :math:`(\text{num_pts}, 3)`
        yms (torch.Tensor): Point-wise youngs modulus, of shape :math:`(\text{num_pts})` 
        prs (torch.Tensor): Point-wise poissons ratio, of shape :math:`(\text{num_pts})`
        rhos (torch.Tensor): Point-wise density, of shape :math:`(\text{num_pts})`
        en_interp (float): interpolation between energy at current step
        batch_size (int): Number of sample deformations
        num_handles (int): Number of handles
        appx_vol (float): Approximate volume of object
        num_samples (int): Number of sample points. 
        le_coeff (float): floating point coefficient for elastic loss 
        lo_coeff (float): floating point coefficient for orthogonal loss

    Returns:
        torch.Tensor, torch.Tensor: The elastic loss term, the orthogonality losses terms
    """
    batch_transforms = 0.1 * torch.randn(batch_size, num_handles, 3, 4,
                                         dtype=normalized_pts.dtype).cuda()

    # Select num_samples from all points
    sample_indices = torch.randint(low=0, high=normalized_pts.shape[0], size=(
        num_samples,), device=normalized_pts.device)

    sample_pts = normalized_pts[sample_indices]
    sample_yms = yms[sample_indices]
    sample_prs = prs[sample_indices]
    sample_rhos = rhos[sample_indices]

    # Get current skinning weights at sample pts
    weights = model(sample_pts)

    # Calculate elastic energy for the batch of transforms
    # loss_elastic(model, pts, yms, prs,  rhos, transforms, appx_vol, interp_step)
    # le = torch.tensor(0,device=sample_pts.device, dtype=sample_pts.dtype) #
    le = le_coeff * loss_elastic_warp(model, sample_pts, sample_yms, sample_prs,
                                      sample_rhos, batch_transforms, appx_vol, en_interp)

    # Calculate orthogonality of skinning weights
    lo = lo_coeff * loss_ortho(weights)

    return le, lo
