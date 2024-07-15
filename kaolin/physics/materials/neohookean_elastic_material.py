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
    'neohookean_energy',
    'unbatched_neohookean_energy',
    'unbatched_neohookean_gradient',
    'unbatched_neohookean_hessian'
]

def neohookean_energy(mu, lam, defo_grad):
    r"""Implements a version of neohookean energy. Calculate energy per-integration primitive.

    Args:
        mu (torch.Tensor): Batched lame parameter mu, of shape :math:`(\text{batch_dims}, 1)`
        lam (torch.Tensor): Batched lame parameter lambda, of shape, :math:`(\text{batch_dims}, 1)` 
        defo_grad (torch.Tensor): Batched deformation gradients (denoted in literature as F) of 3 or more dimensions, :math:`(\text{batch_dims}, 3, 3)`

    Returns:
        torch.Tensor: :math:`(\text{batch_dims}, 1)` vector of per defo-grad energy values
    """
    # Shape (batch_dims, 1)
    C1 = mu / 2
    # Shape (batch_dims, 1)
    D1 = lam / 2


    dimensions = defo_grad.shape
    batched_dims = dimensions[:-2]
    batched_trace = torch.vmap(torch.trace)
    
    r"""Calculate the first invariant I1 = trace(C) = trace(F^TF)"""
    
    # Last 2 dimensions of defo_grad (3,3) are the matrix dimensions. 
    # All prev dimensions are batch dimensions
    FtF = torch.matmul(torch.transpose(defo_grad, -2, -1), defo_grad)

    # Flatten to (-1, 3, 3)
    cauchy_green_strains = FtF.reshape(batched_dims.numel(), 3, 3)
    # Calculate batched traces of tensor and reshape to (batch_dim, 1)
    I1 = batched_trace(cauchy_green_strains).reshape(batched_dims).unsqueeze(-1)

    # Calculate batched determinant of tensor and reshape to (batch_dim, 1)
    J = torch.det(defo_grad).unsqueeze(-1)
    W = C1 * (I1 - 3) + D1 * (J - 1) * (J - 1) - mu * (J - 1.0)
    return W

def unbatched_neohookean_energy(mu, lam, defo_grad):
    r"""Implements an version of neohookean energy. Calculate energy per-integration primitive.

    Args:
        mu (torch.Tensor): Tensor of lame parameter mu, of shape :math:`(\text{batch_dim}, 1)`
        lam (torch.Tensor): Tensor of lame parameter lambda, of shape :math:`(\text{batch_dim}, 1)`
        defo_grad (torch.Tensor): Flattened 3d deformation gradients, of shape :math:`(\text{batch_dim}*3*3, 1)`

    Returns:
        torch.Tensor: Vector of per-primitive energy values, shape :math:`(\text{batch_dim}, 1)`
    """
    F = defo_grad.reshape(-1,9)
    F_0 = F[:,0]
    F_1 = F[:,1]
    F_2 = F[:,2]
    F_3 = F[:,3]
    F_4 = F[:,4]
    F_5 = F[:,5]
    F_6 = F[:,6]
    F_7 = F[:,7]
    F_8 = F[:,8]
    
    J =      (F_0 * (F_4 * F_8 - F_5 * F_7) \
            - F_1 * (F_3 * F_8 - F_5 * F_6) \
            + F_2 * (F_3 * F_7 - F_4 * F_6)).unsqueeze(dim=1)
    
    IC = (F_0 * F_0 + F_1 * F_1 + F_2 * F_2 + \
    F_3 * F_3 + F_4 * F_4 + F_5 * F_5 + \
    F_6 * F_6 + F_7 * F_7 + F_8 * F_8).unsqueeze(dim=1)

    return 0.5 * mu * (IC - 3.0) + 0.5 * lam * (J - 1.) * (J - 1.) - mu * (J - 1.0)

def unbatched_neohookean_gradient(mu, lam, defo_grad):
    r"""Implements a version of neohookean gradients w.r.t deformation gradients. Calculate gradient per-integration primitive.

    Args:
        mu (torch.Tensor): Tensor of lame parameter mu, of shape :math:`(\text{batch_dim}, 1)`
        lam (torch.Tensor): Tensor of lame parameter lambda, of shape :math:`(\text{batch_dim}, 1)`
        defo_grad (torch.Tensor): Flattened 3d deformation gradients, of shape :math:`(\text{batch_dim}*3*3, 1)`

    Returns:
        torch.Tensor: Tensor of per-primitive neohookean gradient w.r.t defo_grad, flattened, shape :math:`(\text{batch_dim}, 9)`
    """
    F = defo_grad.reshape(-1,9)
    F_0 = F[:,0]
    F_1 = F[:,1]
    F_2 = F[:,2]
    F_3 = F[:,3]
    F_4 = F[:,4]
    F_5 = F[:,5]
    F_6 = F[:,6]
    F_7 = F[:,7]
    F_8 = F[:,8]
    #compute batch determinant of all F's
    J =      (F_0 * (F_4 * F_8 - F_5 * F_7) \
            - F_1 * (F_3 * F_8 - F_5 * F_6) \
            + F_2 * (F_3 * F_7 - F_4 * F_6)).unsqueeze(dim=1)
    
    #a = 1.0 + mu/lam
    FinvT = torch.inverse(defo_grad.reshape(-1,3,3)).transpose(1,2)
    
    return (mu[:, :, None] * defo_grad.reshape(-1,3,3) + (J * (lam * (J + (-1.0)) + (-mu))).unsqueeze(dim=2)*FinvT)
    
def unbatched_neohookean_hessian(mu, lam, defo_grad):
    r"""Implements a version of neohookean hessian w.r.t deformation gradients. Calculate per-integration primitive.

    Args:
        mu (torch.Tensor): Tensor of lame parameter mu, of shape :math:`(\text{batch_dim}, 1)`
        lam (torch.Tensor): Tensor of lame parameter lambda, of shape :math:`(\text{batch_dim}, 1)`
        defo_grad (torch.Tensor): Flattened 3d deformation gradients, of shape :math:`(\text{batch_dim}*3*3, 1)`

    Returns:
        torch.Tensor: Tensor of per-primitive neohookean hessian w.r.t defo_grad, flattened, shape :math:`(\text{batch_dim}, 9,9)`
    """
    F = defo_grad.reshape(-1,9)
    F_0 = F[:,0]
    F_1 = F[:,1]
    F_2 = F[:,2]
    F_3 = F[:,3]
    F_4 = F[:,4]
    F_5 = F[:,5]
    F_6 = F[:,6]
    F_7 = F[:,7]
    F_8 = F[:,8]
    id_mat = torch.eye(9,device=mu.device)

    #can save more time by not recomputing this stuff
    J =      (F_0 * (F_4 * F_8 - F_5 * F_7) \
            - F_1 * (F_3 * F_8 - F_5 * F_6) \
            + F_2 * (F_3 * F_7 - F_4 * F_6)).unsqueeze(dim=1)
    
    #a = 1.0 + mu/lam
    FinvT = torch.inverse(defo_grad.reshape(-1,3,3)).transpose(1,2)
    gamma = J*(lam*(2.0*J + (-1.0)) + (-mu))
    dgamma = gamma - lam*J*J
    
    #dFinv = torch.bmm(FinvT.reshape(-1,9,1), FinvT.reshape(-1, 1,9)).reshape(-1,3,3,3,3).transpose(1,2).reshape(-1,9,9)
    #dgamma = (lam*J*(2*J-a)).unsqueeze(dim=2)*torch.bmm(FinvT.reshape(-1,9,1), FinvT.reshape(-1, 1,9))
    FFinv = torch.bmm(FinvT.reshape(-1,9,1), FinvT.reshape(-1, 1,9))
    H = mu[:,:,None] * id_mat + gamma[:,:,None] * FFinv -dgamma[:,:,None] * FFinv.reshape(-1,3,3,3,3).transpose(2,4).reshape(-1,9,9)
    
    return H
