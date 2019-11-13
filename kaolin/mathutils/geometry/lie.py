# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
import torch.nn as nn


__all__ = [
    'SO3Exp'
]


def SO3_hat(omega):
    r"""Takes in exponential coordinates (:math:`\omega` :math:`B x 3`) and converts to
    tangent vector representations (:math:`[\omega]_{\times}` :math:`B x 3 x 3`).

    Args:
        omega (torch.Tensor): Exponential coordinates

    Returns:
        omega_cross (torch.Tensor): Tangent vectors

    Shape:
        input: :math:`(B, 3)` where :math:`B` is the batchsize.
        output: :math:`(B, 3, 3)` where :math:`B` is the batchsize.

    """

    # Note: Since this function is usually called from another layer,
    # it is assumed that assertion checks are performed before calling.

    # Batchsize
    B = omega.shape[0]
    # Tensor type
    dtype = omega.dtype
    # Device ID
    device = omega.device

    omega_cross = torch.zeros((B, 3, 3)).type(dtype).to(device)
    omega_cross[:, 2, 1] = omega[:, 0].clone().type(dtype).to(device)
    omega_cross[:, 1, 2] = -1 * omega[:, 0].clone().type(dtype).to(device)
    omega_cross[:, 0, 2] = omega[:, 1].clone().type(dtype).to(device)
    omega_cross[:, 2, 0] = -1 * omega[:, 1].clone().type(dtype).to(device)
    omega_cross[:, 1, 0] = omega[:, 2].clone().type(dtype).to(device)
    omega_cross[:, 0, 1] = -1 * omega[:, 2].clone().type(dtype).to(device)

    return omega_cross

    # """
    # Another way to do it, but using torch.cross
    # """
    # # Canonical axes
    # e1 = torch.Tensor([[1, 0, 0]]).type(dtype).to(device)
    # e1 = e1.expand((B, 3))
    # e2 = torch.Tensor([[0, 1, 0]]).type(dtype).to(device)
    # e2 = e2.expand((B, 3))
    # e3 = torch.Tensor([[0, 0, 1]]).type(dtype).to(device)
    # e3 = e3.expand((B, 3))

    # # Construct omega_cross, by concatenating the cross products
    # # of each element of the batch with the canonical axes
    # c1 = torch.cross(omega, e1, dim=1).unsqueeze(2)
    # c2 = torch.cross(omega, e2, dim=1).unsqueeze(2)
    # c3 = torch.cross(omega, e3, dim=1).unsqueeze(2)

    # e1.requires_grad = True
    # e2.requires_grad = True
    # e3.requires_grad = True

    # return torch.cat([c1, c2, c3], dim=2)


class SO3Exp(nn.Module):
    r"""Exponential map for SO(3), i.e., for 3 x 3 rotation matrices.

    Map a batch of so(3) vectors (axis-angle) to 3D rotation matrices.

    Args:
        x (torch.Tensor): Input so(3) exponential coordinates
        eps (float, optional): Threshold to determine which angles are deemed 'small'

    Returns:
        x (torch.Tensor): Output SO(3) rotation matrices

    Shape:
        input: :math:`(B, 3)` where :math:`B` is the batchsize.
        output: :math:`(B, 3, 3)` where :math:`B` is the batchsize.

    Example:
        >>> omega = torch.Tensor([[1, 2, 3], [4, 5, 6]])
        >>> so3exp = SO3Exp()
        >>> print(R)
        >>> print(torch.bmm(R, R.transpose(1,2))) # Should be nearly identity matrices

    """

    def __init__(self, eps=1e-8):
        super(SO3Exp, self).__init__()
        self.SO3_hat = SO3_hat
        self.eps = eps

    def forward(self, x):

        if not torch.is_tensor(x):
            raise TypeError('Expected torch.Tensor. Got {} instead.'.format(
                type(omega)))

        assert x.dim() == 2, 'Expected 2-D tensor. Got {}-D.'.format(x.dim())
        assert x.shape[1] == 3, 'Dim 1 of tensor x must be of shape 3. Got {} instead'.format(
            x.shape[1])

        # Compute omega_cross and omega_cross_squared
        omega = x
        omega_cross = SO3_hat(x)
        omega_cross_sq = torch.bmm(omega_cross, omega_cross)

        # Get angles of rotation (theta)
        theta = torch.norm(omega, dim=1)

        # Get a mask (of shape B; 0 for all angles deemed 'small', 1 otherwise)
        mask = theta > self.eps
        mask = mask.type(x.dtype).to(x.device)

        # Compute the coefficients in the Rodrigues formula
        sin_theta_by_theta = mask * (torch.sin(theta) / theta)
        one_minus_cos_theta_by_theta_sq = mask * \
            ((1 - torch.cos(theta)) / (theta**2))

        # A = (sin(theta) / theta) * omega_cross
        # For small angles, A = omega_cross
        A = (sin_theta_by_theta.view(-1, 1, 1) * omega_cross) + \
            ((1 - mask).view(-1, 1, 1) * omega_cross)
        # B = ((1-cos(theta))/(theta**2)) * omega_cross**2
        # For small angles, B = 0 (we do not need to add the term in, as the mask
        # already zeros out all small angle terms)
        B = one_minus_cos_theta_by_theta_sq.view(-1, 1, 1) * omega_cross_sq
        # I (identity matrix, tiled B times)
        I = torch.eye(3).expand_as(omega_cross)

        # Compute the exponential map of x
        x = I + A + B

        return x
