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
    'generate_fcn_simplicits_scene_energy',
    'generate_fcn_simplicits_scene_gradient',
    'generate_fcn_simplicits_scene_hessian',
    'generate_fcn_simplicits_material_energy',
    'generate_fcn_simplicits_material_gradient',
    'generate_fcn_simplicits_material_hessian'
]


def generate_fcn_simplicits_scene_energy(wrapper_obj, B, coeff=1, integration_sampling=None):
    r"""Generates a function that calculates simplicits handle-wise energy

    Args:
        wrapper_obj (kaolin.physics.utils.ForceWrapper): A force wrapper object
        B (torch.Tensor): Simplicits jacobian dz/dx, of shape :math:`(3*\text{num_pts}, 12*\text{num_handles})` 
        coeff (float): Scaling coefficient. Defaults to 1
        integration_sampling (torch.Tensor, optional): Primitive-wise integration scheme. Defaults to None.

    Returns:
        torch.Tensor: Energy, single value tensor
    """
    return lambda x: coeff * wrapper_obj.energy(x, integration_weights=integration_sampling)


def generate_fcn_simplicits_scene_gradient(wrapper_obj, B, coeff=1, integration_sampling=None):
    r"""Generates a function that calculates simplicits handle-wise gradients

    Args:
        wrapper_obj (kaolin.physics.utils.ForceWrapper): A force wrapper object
        B (torch.Tensor): Simplicits jacobian dz/dx of size :math:`(3*\text{num_pts}, 12*\text{num_handles})` 
        coeff (float): Scaling coefficient. Defaults to 1
        integration_sampling (torch.Tensor, optional): Primitive-wise integration scheme. Defaults to None.

    Returns:
        torch.Tensor: Gradient vector, of shape :math:`(12*\text{num_handles})`
    """
    return lambda x: coeff * B.transpose(0, 1) @ wrapper_obj.gradient(x, integration_weights=integration_sampling).flatten().unsqueeze(-1)


def generate_fcn_simplicits_scene_hessian(wrapper_obj, B, coeff=1, integration_sampling=None):
    r"""Generates a function that calculates simplicits handle-wise hessian

    Args:
        wrapper_obj (kaolin.physics.utils.ForceWrapper): A force wrapper object
        B (torch.Tensor): Simplicits jacobian dz/dx of size :math:`(3*\text{num_pts}, 12*\text{num_handles})` 
        coeff (float): Scaling coefficient. Defaults to 1
        integration_sampling (torch.Tensor, optional): Primitive-wise integration scheme. Defaults to None.

    Returns:
        torch.Tensor: Hessian matrix, of shape :math:`(12*num_handles, 12*\text{num_handles})`
    """
    return lambda x: coeff * B.transpose(0, 1) @ wrapper_obj.hessian(x, integration_weights=integration_sampling).transpose(1, 2).reshape(B.shape[0], B.shape[0]) @ B


def generate_fcn_simplicits_material_energy(wrapper_obj, J, coeff=1, integration_sampling=None):
    r"""Generates a function that calculates simplicits material handle-wise energy

    Args:
        wrapper_obj (kaolin.physics.utils.ForceWrapper): A force wrapper object
        J (torch.Tensor): Simplicits jacobian dF/dz, of shape :math:`(9*\text{num_points}, 12*\text{num_handles})` 
        coeff (float): Scaling coefficient. Defaults to 1.0
        integration_sampling (torch.Tensor, optional): Primitive-wise integration scheme. Defaults to None.

    Returns:
        torch.Tensor: Energy, single value tensor
    """
    return lambda x: wrapper_obj.energy(x, integration_weights=integration_sampling)


def generate_fcn_simplicits_material_gradient(wrapper_obj, J, coeff=1, integration_sampling=None):
    r"""Generates a function that calculates simplicits material handle-wise energy

    Args:
        wrapper_obj (kaolin.physics.utils.ForceWrapper): A force wrapper object
        J (torch.Tensor): Simplicits jacobian dF/dz, of shape :math:`(9*\text{num_points}, 12*\text{num_handles})` 
        coeff (float): Scaling coefficient. Defaults to 1
        integration_sampling (torch.Tensor, optional): Primitive-wise integration scheme. Defaults to None.

    Returns:
        torch.Tensor: Gradient vector, of shape :math:`(12*\text{num_handles})`
    """
    return lambda x: torch.mv(J.transpose(0, 1), wrapper_obj.gradient(x, integration_weights=integration_sampling).flatten()).unsqueeze(-1)


def generate_fcn_simplicits_material_hessian(wrapper_obj, J, coeff=1, integration_sampling=None):
    r"""Generates a function that calculates simplicits material handle-wise energy

    Args:
        wrapper_obj (kaolin.physics.utils.ForceWrapper): A force wrapper object
        J (torch.Tensor): Simplicits jacobian dF/dz, of shape :math:`(9*\text{num_points}, 12*\text{num_handles})` 
        coeff (float): Scaling coefficient. Defaults to 1
        integration_sampling (torch.Tensor, optional): Primitive-wise integration scheme. Defaults to None.

    Returns:
        torch.Tensor: Hessian matrix, of shape :math:`(12*\text{num_handles}, 12*\text{num_handles})`
    """
    return lambda x: torch.matmul(J.transpose(0, 1), torch.bmm(wrapper_obj.hessian(x, integration_weights=integration_sampling).squeeze(), J.reshape(-1, 9, J.shape[1])).reshape(-1, J.shape[1]))
