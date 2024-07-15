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

from kaolin.physics.simplicits.losses import loss_elastic, loss_ortho

__all__ = [
    'train_step'
]

def train_step(model, normalized_pts, yms, prs, rhos, en_interp, batch_size, num_handles, appx_vol, num_samples, le_coeff, lo_coeff):
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
        torch float, torch float: Returns the elastic and orthogonality losses
    """
    batch_transforms = 0.1*torch.randn(batch_size, num_handles, 3, 4, dtype=normalized_pts.dtype, device=normalized_pts.device)
    
    # Select num_samples from all points
    sample_indices = torch.randint(low=0, high=normalized_pts.shape[0], size=(num_samples,), device=normalized_pts.device)
    
    sample_pts = normalized_pts[sample_indices] 
    sample_yms = yms[sample_indices]
    sample_prs = prs[sample_indices]
    sample_rhos = rhos[sample_indices]
    
    #Get current skinning weights at sample pts
    weights = model(sample_pts)
    
    # Calculate elastic energy for the batch of transforms
    # loss_elastic(model, pts, yms, prs,  rhos, transforms, appx_vol, interp_step)
    # le = torch.tensor(0,device=sample_pts.device, dtype=sample_pts.dtype) #
    le = le_coeff * loss_elastic(model, sample_pts, sample_yms, sample_prs, sample_rhos, batch_transforms, appx_vol, en_interp)
    # Calculate orthogonality of skinning weights
    lo = lo_coeff * loss_ortho(weights)
    
    return le, lo
