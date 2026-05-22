# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

import warp as wp

__all__ = [
    "to_lame", "get_defo_grad"
]


def to_lame(yms, prs):
    r"""Converts youngs modulus and poissons ratio to lame parameters

    Args:
        yms (torch.Tensor): tensor of youngs modulus (in :math:`kg/m/s^2`)
        prs (torch.Tensor): tensor of poisson ratios

    Returns:
        torch.Tensor, torch.Tensor: lame parameter mu and lamda
    """
    mus = yms / (2 * (1 + prs))
    lams = yms * prs / ((1 + prs) * (1 - 2 * prs))
    return mus, lams


@wp.kernel
def _get_defo_grad_wp_kernel(F: wp.array(dtype=wp.mat33)):  # pragma: no cover
    tid = wp.tid()
    F[tid] += wp.identity(3, dtype=wp.float32)


def get_defo_grad(wp_z, wp_dFdz):
    r"""
    Get the deformation gradient per-sample point.

    Args:
        wp_z (wp.array(dtype=wp.float)): Warp array of the deformation gradient Jacobian. Vector of size :math:`(12\text{handles},)`
        wp_dFdz (wp.sparse.bsr_matrix): Sparse matrix of the deformation gradient Jacobian. Sparse matrix size :math:`(9 \text{num_points}, 12 \text{handles})`. Block shape :math:`(9, 4)`

    Returns:
        wp.array(dtype=wp.mat33): Warp array of wp.mat33 deformation gradients.
    """

    # Get deformation gradient
    dFdz_z = wp_dFdz @ wp_z
    Fs = wp.array(dFdz_z, dtype=wp.mat33)
    wp.launch(kernel=_get_defo_grad_wp_kernel,
              dim=Fs.shape, inputs=[Fs], adjoint=False)
    return Fs
