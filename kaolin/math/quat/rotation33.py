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

from typing import List

import torch
from torch import Tensor

from .util import vector_normalize


@torch.jit.script
def is_rot33_valid(rot33: Tensor, atol: float = 1e-6) -> bool:
    """Checks whether a 3x3 rotation matrix is valid.

    Valid rotation matrices have determinant 1 and are orthogonal.

    Args:
        rot33 (Tensor): Rotation matrix of shape (b, 3, 3).
        atol (float, optional): Absolute error tolerance for orthogonality check. Defaults to 1e-6.

    Returns:
        bool: True if the rotation is valid, False otherwise.
    """
    assert rot33.shape[-1] == rot33.shape[-2], f"invalid rotation matrix: expected square, but got {rot33.shape[-2:]}"

    det = torch.linalg.det(rot33)
    is_det_valid = torch.allclose(det, torch.ones(size=[1], device=det.device))

    # Q^T Q = I  for orthogonal matrix
    # create identity for DxD of last dimension only.
    #   works because assuming square rotation matrix
    identity = torch.eye(rot33.shape[-1], device=rot33.device)

    # transpose matrix of last 2 dimensions only
    ndim = rot33.dim()
    perm_idx = list(range(ndim - 2)) + [ndim - 1, ndim - 2]
    permutation = rot33.permute(perm_idx)
    norm = rot33 @ permutation
    is_orthogonal = torch.allclose(norm, identity, atol=atol)
    return is_det_valid and is_orthogonal


@torch.jit.script
def rot33_identity(batch_size: int = 1, device: torch.device = "cuda") -> Tensor:
    """Generate a batch of identity 3x3 rotation matrices.

    Args:
        batch_size (int, optional): Number of rotation matrices in the batch. Defaults to 1.
        device (torch.device, optional): Device memory to use. Defaults to "cuda".

    Returns:
        Tensor: Batch of identity rotation matrices of shape (b, 3, 3).
    """
    return torch.eye(3, 3, device=device, dtype=torch.float).repeat((batch_size, 1, 1))


@torch.jit.script
def translation_identity(batch_size: int = 1, device: torch.device = "cuda") -> Tensor:
    """Generate a batch of identity 3d translation vectors.

    Args:
        batch_size (int, optional): Number of translation vectors in the batch. Defaults to 1.
        device (torch.device, optional): Device memory to use. Defaults to "cuda".

    Returns:
        Tensor: Batch of identity translation vectors of shape (b, 3).
    """
    return torch.zeros((batch_size, 3), dtype=torch.float, device=device)


@torch.jit.script
def rot33_inverse(mat: Tensor) -> Tensor:
    """Invert a 3x3 rotation matrix.

    Args:
        mat (Tensor): Batch of 3x3 rotation matrices of shape (b, 3, 3).

    Returns:
        Tensor: Batch of inverted 3x3 rotation matrices of shape (b, 3, 3).
    """
    return mat.permute((0, 2, 1))


@torch.jit.script
def rot33_rotate(point: Tensor, mat: Tensor) -> Tensor:
    """Rotate a point using a 3x3 rotation matrix.

    Args:
        point (Tensor): Batch of points to rotate of shape (b, 3).
        mat (Tensor): Batch of 3x3 rotation matrices of shape (b, 3, 3).

    Returns:
        Tensor: Batch of rotated points of shape (b, 3).
    """
    return torch.matmul(mat, point.unsqueeze(-1)).squeeze()  # align batch sizes by appending dummy dimension


### CONVERSIONS ###


@torch.jit.script
def rot33_from_quat(quat: Tensor) -> Tensor:
    """Convert a rotation quaternion to 3x3 rotation matrix representation.

    Args:
        quat (Tensor): Rotation quaternion of shape (b, 4).

    Returns:
        Tensor: Batch of 3x3 rotation matrices of shape (b, 3, 3).
    """
    q = vector_normalize(quat)

    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Set individual elements
    tx = 2.0 * x
    ty = 2.0 * y
    tz = 2.0 * z
    twx = tx * w
    twy = ty * w
    twz = tz * w
    txx = tx * x
    txy = ty * x
    txz = tz * x
    tyy = ty * y
    tyz = tz * y
    tzz = tz * z

    r0 = torch.stack([1.0 - (tyy + tzz), txy - twz, txz + twy], dim=-1)
    r1 = torch.stack([txy + twz, 1.0 - (txx + tzz), tyz - twx], dim=-1)
    r2 = torch.stack([txz - twy, tyz + twx, 1.0 - (txx + tyy)], dim=-1)

    matrix = torch.stack([r0, r1, r2], dim=-2)
    return matrix


@torch.jit.script
def rot33_from_angle_axis(angle: Tensor, axis: Tensor) -> Tensor:
    """Convert an (angle, axis) representation to 3x3 rotation matrix representation.

    Args:
        angle (Tensor): Angle in radians of shape (b, 1).
        axis (Tensor): Axis of rotation of shape (b, 3).

    Returns:
        Tensor: Batch of 3x3 rotation matrices of shape (b, 3, 3).
    """
    # pad out to batching dimensions
    while angle.ndim < 2:
        angle = angle.unsqueeze(0)
    while axis.ndim < 2:
        axis = axis.unsqueeze(0)

    sin_axis = angle.sin() * axis
    cos_angle = angle.cos()
    cos1_axis = (1.0 - cos_angle) * axis
    _, axis_y, axis_z = axis.unbind(-1)
    cos1_axis_x, cos1_axis_y, _ = cos1_axis.unbind(-1)
    sin_axis_x, sin_axis_y, sin_axis_z = sin_axis.unbind(-1)
    c1axy = cos1_axis_x * axis_y
    m01 = c1axy - sin_axis_z
    m10 = c1axy + sin_axis_z
    c1axz = cos1_axis_x * axis_z
    m02 = c1axz + sin_axis_y
    m20 = c1axz - sin_axis_y
    c1ayz = cos1_axis_y * axis_z
    m12 = c1ayz - sin_axis_x
    m21 = c1ayz + sin_axis_x
    diag = cos1_axis * axis + cos_angle
    diag_x, diag_y, diag_z = diag.unbind(-1)
    matrix = torch.stack([
        torch.stack([diag_x, m01, m02], dim=-1),
        torch.stack([m10, diag_y, m12], dim=-1),
        torch.stack([m20, m21, diag_z], dim=-1)
    ], dim=-2)
    return matrix
