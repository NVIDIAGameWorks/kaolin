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
import kaolin.physics.materials.utils as material_utils
import kaolin.physics.materials.linear_elastic_material as linear_elastic_material
import kaolin.physics.materials.neohookean_elastic_material as neohookean_elastic_material
from kaolin.physics.simplicits.utils import weight_function_lbs
from kaolin.physics.utils.finite_diff import finite_diff_jac

__all__ = [
    'loss_ortho',
    'loss_elastic',
]

def loss_ortho(weights):
    r"""Calculate orthogonality of weights

    Args:
        weights (torch.Tensor): Tensor of weights, of shape :math:`(\text{num_samples}, \text{num_handles})`

    Returns:
        torch.Tensor: Orthogonality loss, single value tensor
    """
    return nn.MSELoss()(weights.T @ weights, torch.eye(weights.shape[1], device=weights.device))

def loss_elastic(model, pts, yms, prs,  rhos, transforms, appx_vol, interp_step):
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
    
    partial_weight_fcn_lbs = partial(weight_function_lbs,  tfms = transforms, fcn = model)
    pt_wise_Fs = finite_diff_jac(partial_weight_fcn_lbs, pts)
    pt_wise_Fs = pt_wise_Fs[:,:,0]
    
    # shape (N, B, 3, 3)
    N,B = pt_wise_Fs.shape[0:2]
    
    # shape (N, B, 1)
    mus = mus.expand(N, B).unsqueeze(-1)
    lams = lams.expand(N, B).unsqueeze(-1)
    
    # ramps up from 100% linear elasticity to 100% neohookean elasticity
    lin_elastic = (1-interp_step) * linear_elastic_material.linear_elastic_energy(mus, lams, pt_wise_Fs)
    neo_elastic = (interp_step) * neohookean_elastic_material.neohookean_energy(mus, lams, pt_wise_Fs)

    # weighted average (since we uniformly sample, this is uniform for now)
    return (appx_vol/pts.shape[0])*(torch.sum(lin_elastic + neo_elastic))
