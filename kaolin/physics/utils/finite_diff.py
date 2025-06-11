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

import math
import torch

__all__ = [
    'finite_diff_jac',
]

def finite_diff_jac(fcn, x, eps=1e-7):
    r"""Computes the jacobian :math:`\frac{\partial fcn}{\partial x}` using finite diff 

    Args:
        fcn (callable): Function that takes input :math:`(\text{num_pts}*6, \text{dim})` points for central finite differences and outputs a shape :math:`(\text{num_pts}*6, A_1, ..., A_n)`
        x (torch.Tensor): Input values, :math:`(\text{num_pts}, \text{dim})` 
        eps (float, optional): Squared finite difference epsilon. Defaults to 1e-7.

    Returns:
        torch.Tensor: Returns the jacobian of fcn w.r.t x (:math:`\frac{\partial fcn(x)}{\partial x}`), of shape :math:`(\text{num_pts}, A_1, ..., A_n, \text{dim}, \text{dim})`
    """
    delta = math.sqrt(eps)
    h = delta * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)

    finite_diff_bounds = torch.cat([
        x + h[0], x + h[1], x + h[2],
        x - h[0], x - h[1], x - h[2]
    ], dim=0)

    jacobian = fcn(finite_diff_bounds)
    jacobian = jacobian.reshape(2, 3, -1, *jacobian.shape[1:])
    jacobian = (jacobian[0] - jacobian[1]) / (2 * delta)
    # Move dim0 to the end, i.e: permute(1, 2, 3, 4, 0)
    jacobian = jacobian.permute(*range(1, jacobian.dim()), 0)
    return jacobian
