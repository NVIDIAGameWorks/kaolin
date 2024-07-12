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

### OPERATORS ###


@torch.jit.script
def quat_real(quat: Tensor) -> Tensor:
    """Get the real component (`w`) of the quaternion.

    Args:
        quat (Tensor): Quaternion of shape (b, 4).

    Returns:
        Tensor: Real valued component of the quanternion of shape (b, 1).
    """
    return quat[..., 3:]


@torch.jit.script
def quat_imaginary(quat: Tensor) -> Tensor:
    """Get the imaginary components (`xyz`) of the quaternion.

    Args:
        quat (Tensor): Quaternion of shape (b, 4).

    Returns:
        Tensor: Imaginary components of the quanternion of shape (b, 3).
    """
    return quat[..., :3]


@torch.jit.script
def quat_positive(quat: Tensor) -> Tensor:
    """Generate a quanternion with positive real component.

    Args:
        quat (Tensor): Quaternion of shape (b, 4).

    Returns:
        Tensor: Quaternion with positive real components of shape (b, 4).
    """
    q = quat
    z = (q[..., 3:] < 0).float()
    q = (1 - 2 * z) * q
    return q


@torch.jit.script
def quat_abs(quat: Tensor) -> Tensor:
    """Compute the L2 norm of a quaternion.

    Args:
        quat (Tensor): Quaternion of shape (b, 4).

    Returns:
        Tensor: Quaternion norm value of shape (b).
    """
    return quat.norm(p=2, dim=-1)


@torch.jit.script
def quat_unit(quat: Tensor) -> Tensor:
    """Normalize quaternion to have norm of 1.

    Args:
        quat (Tensor): Quaternion of shape (b, 4).

    Returns:
        Tensor: Normalized quaternion with norm of 1 of shape (b, 4).
    """
    return torch.nn.functional.normalize(quat, p=2., dim=-1)


@torch.jit.script
def quat_unit_positive(quat: Tensor) -> Tensor:
    """Normalize quaternion to be a valid 3d rotation.

    Forces the quaternion to have a norm of 1 and positive real component.

    Args:
        quat (Tensor): Quaternion of shape (b, 4).

    Returns:
        Tensor: Rotation quaternion of shape (b, 4).
    """
    return quat_unit(quat_positive(quat))  # normalized to positive and unit quaternion


@torch.jit.script
def quat_identity(shape: List[int], device: torch.device = "cuda") -> Tensor:
    """Generate a batch of identity quaternions.

    Args:
        shape (List[int]): Batch shape to generate.
        device (torch.device, optional): Device memory to use. Defaults to "cuda".

    Returns:
        Tensor: Identity quaternion of shape (`shape`, 4).
    """
    w = torch.ones(shape + [1], device=device)
    xyz = torch.zeros(shape + [3], device=device)
    q = torch.cat([xyz, w], dim=-1)
    return quat_unit_positive(q)


@torch.jit.script
def quat_conjugate(quat: Tensor) -> Tensor:
    """Generate conjugate quaternion. Imaginary components are negated.

    Args:
        quat (Tensor): Quaternion of shape (b, 4).

    Returns:
        Tensor: Conjugate quaternion of shape (b, 4).
    """
    return torch.cat([-quat_imaginary(quat), quat_real(quat)], dim=-1)


@torch.jit.script
def quat_inverse(quat: Tensor) -> Tensor:
    """Invert a unit rotation quaternion.

    Same as conjugating a quaternion.

    Args:
        quat (Tensor): Quaternion of shape (b, 4).

    Returns:
        Tensor: Inverted quaternion of shape (b, 4).
    """
    return quat_conjugate(quat)


@torch.jit.script
def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    """Multiply two quaternions.

    Args:
        a (Tensor): First quaternion of shape (b, 4).
        b (Tensor): Second quaternion of shape (b, 4).

    Returns:
        Tensor: Multiplication resulting quaternion of shape (b, 4).
    """
    x1, y1, z1, w1 = a.unbind(-1)
    x2, y2, z2, w2 = b.unbind(-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack([x, y, z, w], dim=-1)


@torch.jit.script
def quat_rotate(rotation: Tensor, point: Tensor) -> Tensor:
    """Rotate a 3d point by a rotation quaternion.

    Args:
        rotation (Tensor): Rotation quaternion of shape (b, 4).
        point (Tensor): Point to rotate of shape (b, 3).

    Returns:
        Tensor: Rotated point of shape (b, 3).
    """
    point_quat = torch.nn.functional.pad(point, (0, 1))
    return quat_imaginary(quat_mul(quat_mul(rotation, point_quat), quat_conjugate(rotation)))


### CONVERSIONS ###


@torch.jit.script
def quat_from_angle_axis(angle: Tensor, axis: Tensor, is_degree: bool = False):
    """Convert an (angle, axis) representation to rotation quaternion representation.

    Args:
        angle (Tensor): Angle in radians of shape (b, 1).
        axis (Tensor): Axis of rotation of shape (b, 3).

    Returns:
        Tensor: Rotation quaternion of shape (b, 4).
    """
    radians = torch.deg2rad(angle) if is_degree else angle
    half_angle = 0.5 * radians
    axis_norm = torch.nn.functional.normalize(axis, p=2., dim=-1)
    w = half_angle.cos()
    xyz = half_angle.sin() * axis_norm
    return torch.hstack([xyz, w])


@torch.jit.script
def _safe_division(a, b) -> Tensor:
    """Safe implementation of dividing a by b to mitigate small numerical values.

    Args:
        a (Tensor): Numerator tensor.
        b (Tensor): Denominator tensor.

    Returns:
        Tensor: Result of dividing a by b (= a/b).
    """
    # values taken from `torch.finfo(dtype).eps`
    #   workaround due to torchscript being unable to compile `finfo` method
    EPS = 0.0
    if b.dtype == torch.float16:
        EPS = 0.0009765625
    elif b.dtype == torch.float32:
        EPS = 1.1920928955078125e-07
    elif b.dtype == torch.float64:
        EPS = 2.220446049250313e-16
    return a / (b + EPS)


@torch.jit.script
def _tr_positive(trace: Tensor, entries: List[List[Tensor]]) -> Tensor:
    sq = _safe_division(torch.tensor(0.5), torch.sqrt(trace + 1.0))
    qx = (entries[2][1] - entries[1][2]) * sq
    qy = (entries[0][2] - entries[2][0]) * sq
    qz = (entries[1][0] - entries[0][1]) * sq
    qw = 0.25 / sq
    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def _case1(entries: List[List[Tensor]]) -> Tensor:
    sq = 2.0 * torch.sqrt(1.0 + entries[0][0] - entries[1][1] - entries[2][2])
    qx = 0.25 * sq
    qy = _safe_division(entries[0][1] + entries[1][0], sq)
    qz = _safe_division(entries[0][2] + entries[2][0], sq)
    qw = _safe_division(entries[2][1] - entries[1][2], sq)
    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def _case2(entries: List[List[Tensor]]) -> Tensor:
    sq = 2.0 * torch.sqrt(1.0 + entries[1][1] - entries[0][0] - entries[2][2])
    qx = _safe_division(entries[0][1] + entries[1][0], sq)
    qy = 0.25 * sq
    qz = _safe_division(entries[1][2] + entries[2][1], sq)
    qw = _safe_division(entries[0][2] - entries[2][0], sq)
    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def _case3(entries: List[List[Tensor]]) -> Tensor:
    sq = 2.0 * torch.sqrt(1.0 + entries[2][2] - entries[0][0] - entries[1][1])
    qx = _safe_division(entries[0][2] + entries[2][0], sq)
    qy = _safe_division(entries[1][2] + entries[2][1], sq)
    qz = 0.25 * sq
    qw = _safe_division(entries[1][0] - entries[0][1], sq)
    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def _case_indices(mat: Tensor, case: Tensor) -> Tensor:
    c = case.unsqueeze(-1)
    newdim = [mat.ndim - 2] + [4]  # reshape (a0, a1, ..., 3, 3) -> (a0, a1, ..., 4)
    return c.tile(newdim)


@torch.jit.script
def quat_from_rot33(mat: Tensor) -> Tensor:
    """Convert a 3x3 rotation matrix to rotation quaternion representation.

    Args:
        mat (Tensor): Rotation matrix of shape (b, 3, 3).

    Returns:
        Tensor: Rotation quaternion of shape (b, 4).
    """
    assert mat.shape[-1] == 3
    assert mat.shape[-2] == 3

    trace = torch.diagonal(mat, dim1=-1, dim2=-2).sum(dim=-1)
    rows = torch.unbind(mat, dim=-2)
    entries = [torch.unbind(row, dim=-1) for row in rows]

    res2 = torch.where(_case_indices(mat, entries[1][1] > entries[2][2]), _case2(entries), _case3(entries))
    res1 = torch.where(
        _case_indices(mat, (entries[0][0] > entries[1][1]) & (entries[0][0] > entries[2][2])),
        _case1(entries),
        res2,
    )
    quat = torch.where(_case_indices(mat, trace > 0), _tr_positive(trace, entries), res1)
    return quat
