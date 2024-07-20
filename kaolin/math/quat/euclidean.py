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

from typing import Optional

import torch
from torch import Tensor

from .matrix44 import rot44_from_quat
from .rotation33 import is_rot33_valid, rot33_inverse

__all__ = [
    'euclidean_identity',
    'euclidean_from_rotation_translation',
    'euclidean_rotation_matrix',
    'euclidean_translation_vector',
    'is_euclidean_valid',
    'euclidean_inverse'
]

### CONVERSIONS ###

@torch.jit.script
def euclidean_identity(batch_size: int, device: torch.device = "cuda") -> Tensor:
    """Identity Euclidean transformation for given batch size.

    Args:
        batch_size (int): Batch size.
        device (torch.device, optional): Device memory to use. Defaults to "cuda".

    Returns:
        Tensor: Batch of identity Euclidean transforms of shape (b, 4, 4).
    """
    return torch.eye(4, device=device).repeat((batch_size, 1, 1))


@torch.jit.script
def euclidean_from_rotation_translation(r: Optional[Tensor] = None, t: Optional[Tensor] = None) -> Tensor:
    """Construct a Euclidean transformation matrix from a rotation quaternion and 3d translation.

    Only one of rotation and translation can be None.

    Args:
        r (Optional[Tensor], optional): Rotation quaternion of shape (b, 4). Defaults to None.
        t (Optional[Tensor], optional): 3d translation vector of shape (b, 3). Defaults to None.

    Returns:
        Tensor: Euclidean transformation matrix.
    """
    assert r is not None or t is not None, "rotation and translation can't be all None"
    if r is None:
        # have translation
        assert t is not None
        r2 = euclidean_identity(t.shape[0], device=t.device)
        r2[..., :3, 3] = t
    elif t is None:
        # have rotation
        assert r is not None
        r2 = rot44_from_quat(r)
    else:
        r2 = rot44_from_quat(r)
        r2[..., :3, 3] = t
    return r2


### OPERATORS ###


@torch.jit.script
def euclidean_rotation_matrix(x: Tensor) -> Tensor:
    """Retrieve 3d rotation matrix from Euclidean transformation matrix.

    Args:
        x (Tensor): Euclidean transformation matrices of shape (b, 4, 4).

    Returns:
        Tensor: 3d rotation matrices of shape (b, 3, 3)
    """
    return x[..., :3, :3]


@torch.jit.script
def euclidean_translation_vector(x: Tensor) -> Tensor:
    """Retrieve the 3d translation vector from the Euclidean transformation matrix.

    Args:
        x (Tensor): Euclidean transformation matrices of shape (b, 4, 4).

    Returns:
        Tensor: 3d translation vectors of shape (b, 3)
    """
    return x[..., :3, 3]


@torch.jit.script
def is_euclidean_valid(x: Tensor, throw: bool = False) -> bool:
    """Check whether a matrix represents a valid Euclidean transformation matrix.

    Args:
        x (Tensor): Euclidean transformation matrices of shape (b, 4, 4).

    Returns:
        bool: True if all are valid, False otherwise.
    """
    rot33 = euclidean_rotation_matrix(x)  # check 3d-rotation matrix
    rot_valid = is_rot33_valid(rot33)
    row_valid = bool((x[..., 3, :3] == 0).all())
    col_valid = bool((x[..., 3, 3] == 1).all())
    out = rot_valid and row_valid and col_valid
    if throw and not out:
        raise ValueError(f"Matrix {x} is not a valid Euclidean transformation matrix.\n\tRotation valid? {rot_valid}\n\tRows valid? {row_valid}.\n\tColumns valid? {col_valid}")
    return out


@torch.jit.script
def euclidean_inverse(x: Tensor) -> Tensor:
    """Inverse of a Euclidean transformation matrix.

    Rotation is inverted by :math:`R \\rightarrow R^{-1} = R^{T}`.

    Translation is inverted by :math:`T \\rightarrow -R^{-1}T`.

    The resulting matrix is of the form:
    :math:`\\left[\\begin{array}{cccc}
    R^{-1}&T^{-1}\\\\
    \\textbf{0}&\\textbf{1}\\\\
    \\end{array}\\right]`

    Args:
        x (Tensor): Euclidean transformation matrices of shape (b, 4, 4).

    Returns:
        Tensor: Inverted matrices of shape (b, 4, 4).
    """
    inv_rot = rot33_inverse(euclidean_rotation_matrix(x))
    inv_trans = -inv_rot @ euclidean_translation_vector(x).squeeze()
    mat = torch.zeros_like(x, device=x.device)
    mat[..., :3, :3] = inv_rot
    mat[..., :3, 3] = inv_trans
    mat[..., 3, 3] = 1
    return mat
