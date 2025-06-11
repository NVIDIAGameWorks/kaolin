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
import torch.nn as nn
from functools import partial
from kaolin.physics.materials import (
    material_utils,
    linear_elastic_material,
    neohookean_elastic_material)
from kaolin.physics.simplicits import weight_function_lbs
from kaolin.physics.utils.finite_diff import finite_diff_jac

__all__ = [
    'loss_ortho',
    'loss_elastic',
    'compute_losses'
]


def loss_ortho(weights):
    r"""Calculate orthogonality of weights

    Args:
        weights (torch.Tensor): Tensor of weights, of shape :math:`(\text{num_samples}, \text{num_handles})`

    Returns:
        torch.Tensor: Orthogonality loss, single value tensor
    """
    return nn.MSELoss()(weights.T @ weights, torch.eye(weights.shape[1], device=weights.device))


def loss_elastic(model, pts, yms, prs, rhos, transforms, appx_vol, interp_step, elasticity_type="neohookean", interp_material=False):
    r"""Calculate a version of simplicits elastic loss for training.

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

    partial_weight_fcn_lbs = partial(
        weight_function_lbs, tfms=transforms, fcn=model)
    pt_wise_Fs = finite_diff_jac(partial_weight_fcn_lbs, pts)
    pt_wise_Fs = pt_wise_Fs[:, :, 0]

    # shape (N, B, 3, 3)
    N, B = pt_wise_Fs.shape[0:2]

    # shape (N, B, 1)
    mus = mus.expand(N, B).unsqueeze(-1)
    lams = lams.expand(N, B).unsqueeze(-1)
    
    if interp_material:
        mus_min = mus.min()
        lams_min = lams.min()
        mus = (1 - interp_step) * mus_min + interp_step * mus
        lams = (1 - interp_step) * lams_min + interp_step * lams

    # ramps up from 100% linear elasticity to 100% neohookean elasticity
    lin_elastic = (1 - interp_step) * \
        linear_elastic_material.linear_elastic_energy(mus, lams, pt_wise_Fs)
    if elasticity_type == "neohookean":
        neo_elastic = (
            interp_step) * neohookean_elastic_material.neohookean_energy(mus, lams, pt_wise_Fs)
    else:
        raise ValueError(f"Elasticity type {elasticity_type} not supported")

    # weighted average (since we uniformly sample, this is uniform for now)
    return (appx_vol / pts.shape[0]) * (torch.sum(lin_elastic + neo_elastic))


def compute_losses(model, normalized_pts, yms, prs, rhos, en_interp, batch_size, num_handles, appx_vol, num_samples, le_coeff, lo_coeff):
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
                                         dtype=normalized_pts.dtype).to(normalized_pts.device)

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
    le = le_coeff * loss_elastic(model, sample_pts, sample_yms, sample_prs,
                                 sample_rhos, batch_transforms, appx_vol, en_interp)

    # Calculate orthogonality of skinning weights
    lo = lo_coeff * loss_ortho(weights)

    return le, lo
