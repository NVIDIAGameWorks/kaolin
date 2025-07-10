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

import torch

__all__ = ["standard_transform_to_relative", 
           "create_projection_matrix",
           "hess_reduction",
           "torch_bsr_to_torch_triplets"]


def standard_transform_to_relative(transform):
    r"""Converts a standard transform to a relative transform.
    Args:
        transform (torch.Tensor): A :math:`(3 \times 4)` or :math:`(4 \times 4)` torch tensor specifying object's transform.
    Returns:
        torch.Tensor: A :math:`3 \times 4` torch tensor specifying object's relative transform.
    """
    if transform.shape == (4, 4):
        relative_transform = transform[:3, :]
    elif transform.shape == (3, 4):
        relative_transform = transform
    else:
        raise ValueError("standard transform must be a 3x4 or 4x4 tensor")
    
    relative_transform -= torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0],
                                    [0, 0, 1, 0]], device=transform.device, dtype=transform.dtype)
    
    return relative_transform


def create_projection_matrix(num_dofs, list_of_kin_dofs):
    r"""Creates a projection matrix that removes kinematic degrees of freedom.

    This function creates a reduced matrix that removes specific degrees of freedom (DOFs) 
    from a system, typically kinematic/prescribed DOFs that should not be solved for.
    The resulting matrix will have all rows except those corresponding to kinematic DOFs.

    Args:
        num_dofs (int): Total number of degrees of freedom in the system
        list_of_kin_dofs (list or torch.Tensor): Indices of the kinematic DOFs to remove

    Returns:
        torch.Tensor: A projection matrix of size :math:`(\text{num_dofs} - \text{num_kin_dofs}, \text{num_dofs})`, that removes the kinematic DOF rows.
    """
    # Create a mask of the dynamic (non-kinematic) dofs
    mask = torch.ones(num_dofs, dtype=torch.bool)
    mask[list_of_kin_dofs] = False

    # Create projection matrix by keeping only non-kinematic rows
    P = torch.eye(num_dofs, device=list_of_kin_dofs.device)[mask]
    return P


def hess_reduction(dense_Ja, block_wise_H, dense_Jb=None):
    r""" This does :math:`\text{Ja}^T \times \text{H} \times \text{Jb}` for a block-wise diagonal :math:`\text{H}` matrix.
    
    Args:
        dense_Ja (torch.Tensor): The left Jacobian matrix
        block_wise_H (torch.Tensor): 3D tensor of block-wise Hessian matrices
        dense_Jb (torch.Tensor): The right Jacobian matrix. If not provided, will use :math:`\text{Ja}`
    
    Returns:
        torch.Tensor: The reduced Hessian matrix
    """

    if dense_Jb is None:
        dense_Jb = dense_Ja

    # This does J.T @ H @ J
    batch_size = block_wise_H.shape[0]
    block_size = block_wise_H.shape[1]

    # Reshape J to match dimensions for batch matrix multiply
    Jb_reshaped = dense_Jb.reshape(-1, block_size, dense_Jb.shape[1])

    # Batch matrix multiply H and J_reshaped
    HJ = torch.bmm(block_wise_H, Jb_reshaped)

    # Reshape result to 2D and multiply with J.T
    # Final: (num_handles*12, num_handles*12)
    return torch.matmul(dense_Ja.transpose(0, 1), HJ.reshape(-1, dense_Jb.shape[1]))


def torch_bsr_to_torch_triplets(mat):
    r"""Converts a sparse torch BSR matrix (or CSR matrix) to a sparse torch triplets.

    Args:
        mat (torch.sparse_bsr_tensor): A sparse torch matrix of shape :math:`(m, n)` with block shape :math:`(b, b)`.

    Returns:
        tuple: (row_indices, col_indices, values) representing the triplets
    """

    crow_indices = mat.crow_indices()
    col_indices = mat.col_indices()
    values = mat.values()

    # Get the row indices by expanding the crow_indices
    row_indices = torch.repeat_interleave(torch.arange(
        len(crow_indices) - 1, device=mat.device), torch.diff(crow_indices))

    return (row_indices, col_indices, values)
