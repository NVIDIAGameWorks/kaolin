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
from torch import Tensor

from .rotation33 import rot33_from_quat
from .util import pad_mat33_to_mat44

__all__ = [
    'rot44_from_quat',
    'translation_to_mat44',
    'scale_to_mat44'
]

### CONVERSIONS ###

@torch.jit.script
def rot44_from_quat(quat: Tensor) -> Tensor:
    """Convert a rotation quaternion to a 4x4 rotation matrix.

    Args:
        quat (Tensor): Rotation quaternion of shape (b, 4).

    Returns:
        Tensor: 4x4 rotation matrix of shape (b, 4, 4).
    """
    mat33 = rot33_from_quat(quat)
    return pad_mat33_to_mat44(mat33)


@torch.jit.script
def translation_to_mat44(vec: Tensor) -> torch.Tensor:
    """Generate an identity 4x4 matrix with translation set.

    Args:
        vec (Tensor): 3d translation vector of shape (b, 4). Ignores the last dimension (4th) of the vector.

    Returns:
        torch.Tensor: 4x4 identity matrix with provided translation of shape (b, 4, 4).
    """
    if vec.ndim < 2:
        vec = vec.unsqueeze(0)  # pad batch
    mat = torch.eye(4, 4, device=vec.device, dtype=torch.float).repeat((vec.shape[0], 1, 1))
    mat[..., :3, -1] = vec[..., :3]

    return mat


@torch.jit.script
def scale_to_mat44(scale: Tensor) -> torch.Tensor:
    """Generate a 4x4 matrix scaled to the provided scale.

    Args:
        scale (Tensor): 3d scaling vector of shape (b, 3).

    Returns:
        torch.Tensor: 4x4 matrix with provided scaling of shape (b, 4, 4).
    """
    if scale.ndim < 2:
        scale = scale.unsqueeze(0)  # pad batch
    mat = torch.eye(4, 4, device=scale.device, dtype=torch.float).repeat((scale.shape[0], 1, 1))
    mat[..., :3, :3] *= scale[..., :3].unsqueeze(-1)

    return mat
