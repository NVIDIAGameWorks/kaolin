# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
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


def _validate_tetrahedrons(tetrahedrons):
    """Helper method to validate the dimensions of the batched tetrahedrons tensor.
    Args:
    tetrahedrons (torch.Tensor): Batched tetrahedrons of shape :math:`(\text{batch_size}, \text{num_tetrahedrons}, 4,
    3)`.
    """
    assert tetrahedrons.ndim == 4, f"tetrahedrons has {tetrahedrons.ndim} but must have 4 dimensions."
    assert tetrahedrons.shape[2] == 4, f"The third dimension of the tetrahedrons must be 4 \
    but the input has {tetrahedrons.shape[2]}. Each tetrahedron has 4 vertices."
    assert tetrahedrons.shape[3] == 3, f"The fourth dimension of the tetrahedrons must be 3 \
    but the input has {tetrahedrons.shape[3]}. Each vertex must have 3 dimensions."


def inverse_vertices_offset(tetrahedrons):
    r"""Given tetrahedrons with 4 vertices A, B, C, D. Compute the inverse of the offset matrix w.r.t. vertex A for each
    tetrahedron. The offset matrix is obtained by the concatenation of `B - A`, `C - A` and `D - A`. The resulting shape
    of the offset matrix is :math:`(\text{batch_size}, \text{num_tetrahedrons}, 3, 3)`. The inverse of the offset matrix
    is computed by this function.

    Args:
        tetrahedrons (torch.Tensor): Batched tetrahedrons of shape :math:`(\text{batch_size}, \text{num_tetrahedrons}, 4, 3)`.
    Returns:
        (torch.Tensor): Batched inverse offset matrix of shape :math:`(\text{batch_size}, \text{num_tetrahedrons}, 3,
        3)`. Each offset matrix is of shape (3, 3), hence its inverse is also of shape (3, 3).

    Example:
        >>> tetrahedrons = torch.tensor([[[[-0.0500,  0.0000,  0.0500],
        ...                                [-0.0250, -0.0500,  0.0000],
        ...                                [ 0.0000,  0.0000,  0.0500],
        ...                                [0.5000, 0.5000, 0.4500]]]])
        >>> inverse_vertices_offset(tetrahedrons)
        tensor([[[[   0.0000,   20.0000,    0.0000],
                  [  79.9999, -149.9999,   10.0000],
                  [ -99.9999,  159.9998,  -10.0000]]]])
    """
    _validate_tetrahedrons(tetrahedrons)

    # split the tensor
    A, B, C, D = torch.split(tetrahedrons, split_size_or_sections=1, dim=2)

    # compute the offset matrix w.r.t. vertex A
    offset_matrix = torch.cat([B - A, C - A, D - A], dim=2)

    # compute the inverse of the offset matrix
    inverse_offset_matrix = torch.inverse(offset_matrix)

    return inverse_offset_matrix
