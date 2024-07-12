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

from typing import List, Optional

import torch
from torch import Tensor

from .euclidean import euclidean_rotation_matrix, euclidean_translation_vector
from .quaternion import (
    quat_from_rot33,
    quat_identity,
    quat_inverse,
    quat_mul,
    quat_unit_positive,
    quat_rotate,
)
from .rotation33 import translation_identity

__all__ = [
    'transform_from_rotation_translation',
    'transform_from_euclidean',
    'transform_identity',
    'transform_rotation',
    'transform_translation',
    'transform_inverse',
    'transform_mul',
    'transform_apply'
]

### CONVERSIONS ###

@torch.jit.script
def transform_from_rotation_translation(
    rotation: Optional[Tensor] = None, translation: Optional[Tensor] = None
) -> Tensor:
    """Generate a position-quaternion transform from a rotation quaternion and 3d translation.

    Only one argument can be None.

    Args:
        r (Optional[Tensor], optional): Rotation quaternion of shape (b, 4). Defaults to None.
        t (Optional[Tensor], optional): 3d translation vector of shape (b, 3). Defaults to None.

    Returns:
        Tensor: Position-quaternion transform of shape (b, 7).
    """
    assert rotation is not None or translation is not None, "rotation and translation can't be all None"
    if rotation is None:
        assert translation is not None
        rotation = quat_identity(list(translation.shape), device=translation.device)
    if translation is None:
        translation = translation_identity(rotation.shape[0], device=rotation.device)
    return torch.cat([rotation, translation], dim=-1)


@torch.jit.script
def transform_from_euclidean(euclidean: Tensor) -> Tensor:
    """Convert a Euclidean transformation matrix to position-quaternion transform representation.

    Args:
        euclidean (Tensor): Euclidean transformation matrices of shape (b, 4, 4).

    Returns:
        Tensor: Position-quaternion transform of shape (b, 7).
    """
    return transform_from_rotation_translation(
        rotation=quat_from_rot33(euclidean_rotation_matrix(euclidean)),
        translation=euclidean_translation_vector(euclidean),
    )


### OPERATORS ###


@torch.jit.script
def transform_identity(shape: List[int], device: torch.device = "cuda") -> Tensor:
    """Generate a batch of identity position-quaternion transforms.

    Args:
        shape (List[int]): Batch shape to generate.
        device (torch.device, optional): Device memory to use. Defaults to "cuda".

    Returns:
        Tensor: Identity position-quaternion transform of shape (`shape`, 7).
    """
    r = quat_identity(shape, device=device)
    t = torch.zeros(shape + [3], device=device)
    return transform_from_rotation_translation(r, t)


@torch.jit.script
def transform_rotation(x: Tensor) -> Tensor:
    """Retrieve the rotation component of a position-rotation transform.

    Args:
        x (Tensor): Position-quaternion transform of shape (b, 7).

    Returns:
        Tensor: Rotation quaternion of shape (b, 4).
    """
    return x[..., :4]


@torch.jit.script
def transform_translation(x: Tensor) -> Tensor:
    """Retrieve the translation component of a position-rotation transform.

    Args:
        x (Tensor): Position-quaternion transform of shape (b, 7).

    Returns:
        Tensor: 3d translation vector of shape (b, 3).
    """
    return x[..., 4:]


@torch.jit.script
def transform_inverse(x: Tensor) -> Tensor:
    """Invert a position-quaternion transform.

    Args:
        x (Tensor): Position-quaternion transform of shape (b, 7).

    Returns:
        Tensor: Inverted position-quaternion transform of shape (b, 7).
    """
    inv_rot = quat_inverse(transform_rotation(x))
    return transform_from_rotation_translation(
        rotation=inv_rot, translation=quat_rotate(inv_rot, -transform_translation(x))
    )


@torch.jit.script
def transform_mul(x: Tensor, y: Tensor) -> Tensor:
    """Combined two position-quaternion transforms.

    Args:
        x (Tensor): First position-quaternion transform of shape (b, 7).
        y (Tensor): Second position-quaternion transform of shape (b, 7).

    Returns:
        Tensor: Combined position-quaternion transform of shape (b, 7).
    """
    r = quat_unit_positive(quat_mul(transform_rotation(x), transform_rotation(y)))
    t = quat_rotate(transform_rotation(x), transform_translation(y)) + transform_translation(x)
    return transform_from_rotation_translation(
        rotation=r,
        translation=t,
    )


@torch.jit.script
def transform_apply(transform: Tensor, point: Tensor) -> Tensor:
    """Apply a position-quaternion transform to a 3d point.

    Args:
        transform (Tensor): Position-quaternion transform of shape (b, 7).
        point (Tensor): 3d point of shape (b, 3).

    Returns:
        Tensor: Transformed 3d point of shape (b, 3).
    """
    return quat_rotate(transform_rotation(transform), point) + transform_translation(transform)
