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
from abc import ABC, abstractmethod 

__all__ = [
    'ForceWrapper',
]
class ForceWrapper(ABC):
    def __init__(self):
        pass 
    
    def energy(self, primitive, integration_weights=None):
        r"""Unbatched energy function.

        Args:
            primitive (torch.Tensor): Batched tensor of primitives (points or deformation gradients)
            integration_weights (torch.Tensor, optional): Per-primitive weighting for energy, of shape :math:`(\text{batch_dim})` 'None' value assigns weights to '1'.
        Returns:
            torch.Tensor: Weighted sum of per-primitive material energies, single value tensor
        """
        p_wise_en = self._energy(primitive)
        if integration_weights is not None:
            p_wise_en = integration_weights.reshape(-1, 1) * p_wise_en
        return torch.sum(p_wise_en)
    
    def gradient(self, primitive, integration_weights=None):
        r"""Unbatched grad function.

        Args:
            primitive (torch.Tensor): Batched tensor of primitives (points or deformation gradients)
            integration_weights (torch.Tensor, optional): Per-primitive weighting for energy, of shape :math:`(\text{batch_dim})` 'None' value assigns weights to '1'.
        Returns:
            torch.Tensor: Weighted sum of per-primitive  gradients
        """
        p_wise_grad = self._gradient(primitive)
        if integration_weights is not None:
            p_wise_grad = integration_weights.reshape(-1,1)*p_wise_grad
        return p_wise_grad
    
    def hessian(self, primitive, integration_weights=None):
        r"""Unbatched hessian function.

        Args:
            primitive (torch.Tensor): Batched tensor of primitives (points or deformation gradients)
            integration_weights (torch.Tensor, optional): Per-primitive weighting for energy, of shape :math:`(\text{batch_dim})` 'None' value assigns weights to '1'.
        Returns:
            torch.Tensor: Weighted sum of per-primitive hessian
        """
        p_wise_hess = self._hessian(primitive)
        if integration_weights is not None:
            p_wise_hess = integration_weights.reshape(-1,1)*p_wise_hess
        return p_wise_hess

    @abstractmethod
    def _energy(self, primitive):
        pass
    
    @abstractmethod
    def _gradient(self, primitive):
        pass 
    
    @abstractmethod
    def _hessian(self, primitive):
        pass 