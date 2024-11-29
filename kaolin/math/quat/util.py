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
    'vector_normalize',
    'pad_mat33_to_mat44'
]

@torch.jit.script
def vector_normalize(vec: torch.Tensor) -> torch.Tensor:
    """Generate a normalized version of a batch of vectors.

    Input is **NOT** modified in place.

    Args:
        vec (torch.Tensor): Batch of Nd vectors of shape (b, N).

    Returns:
        torch.Tensor: Batch of normalized Nd vectors of shape (b, N).
    """
    return (vec.T / torch.sqrt(torch.sum(vec ** 2, dim=-1))).T


@torch.jit.script
def pad_mat33_to_mat44(mat33: torch.Tensor) -> torch.Tensor:
    """Pad a 3x3 rotation matrix to equivalent 4x4 rotation matrix.

    Given input matrix :math:`R`, the output is:
    :math:`\\left[\\begin{array}{cc}
    R&\\textbf{0}\\\\
    \\textbf{0}&\\textbf{1}\\\\
    \\end{array}\\right]`

    Args:
        mat33 (torch.Tensor): Batch of 3x3 rotation matices of shape (b, 3, 3).

    Returns:
        torch.Tensor: Batch of 4x4 rotation matrices of shape (b, 4, 4).
    """
    mat44 = torch.nn.functional.pad(mat33, (0, 1, 0, 1), value=0.0)
    mat44[..., 3, 3] = 1
    return mat44
