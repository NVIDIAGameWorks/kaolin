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

from typing import Tuple

import torch
from torch import Tensor

from .quaternion import quat_from_rot33, quat_imaginary, quat_unit_positive, quat_real

### CONVERSIONS ###


@torch.jit.script
def angle_axis_from_quat(quat: Tensor) -> Tuple[Tensor, Tensor]:
    """Convert a rotation quaternion to (angle, axis) representation.

    The axis is normalized to unit length.

    The angle is guaranteed to be between [0, pi], while axis may be positive or negative.

    Args:
        quat (Tensor): Rotation quaternion of shape (b, 4).

    Returns:
        Tuple[Tensor, Tensor]: Angle in radians of shape (b, 1) and axis of shape (b, 3).
    """
    EPS = 1.1920928955078125e-07  # from _get_eps. cannot be compiled

    q = quat_unit_positive(quat)
    q += EPS
    xyz = quat_imaginary(q)
    w = quat_real(q)
    norm = xyz.norm(p=2, dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, w.abs())
    axis = w.sign() * (xyz / norm)
    return angle, axis


@torch.jit.script
def angle_axis_from_rot33(mat: Tensor) -> Tuple[Tensor, Tensor]:
    """Convert a 3x3 rotation matrix to (angle, axis) representation.

    Args:
        mat (Tensor): Rotation matrix of shape (b, 3, 3).

    Returns:
        Tuple[Tensor, Tensor]: Angle in radians of shape (b,1) and axis of shape (b,3).
    """
    quat = quat_from_rot33(mat)
    return angle_axis_from_quat(quat)
