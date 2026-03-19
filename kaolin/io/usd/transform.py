# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
import numpy as np

try:
    from pxr import Usd, UsdGeom, Gf
    IDENTITY_TRANSFORM = Gf.Matrix4d(1.0, 0.0, 0.0, 0.0,
                                     0.0, 1.0, 0.0, 0.0,
                                     0.0, 0.0, 1.0, 0.0,
                                     0.0, 0.0, 0.0, 1.0)

except ImportError:
    pass

__all__ = [
    'set_local_to_world_transform',
    'get_local_to_world_transform'
]

def set_local_to_world_transform(file_path_or_stage, prim_or_path, local_to_world, time=None):
    """Set a prim's local xform so that ComputeLocalToWorldTransform equals local_to_world.

    Args:
        file_path_or_stage (Usd.Stage or str):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        prim_or_path (Usd.Prim or str): The prim or path to the prim to set the transform on.
        local_to_world (torch.Tensor): :math:`(4, 4)` local-to-world matrix
        time (convertible to float, optional): Optional for writing USD files. Positive integer indicating the
                time at which to set parameters.
    """
    if time is None:
        time = Usd.TimeCode.Default()

    if isinstance(file_path_or_stage, str):
        stage = Usd.Stage.Open(file_path_or_stage)
    else:
        stage = file_path_or_stage
    
    try:
        if isinstance(prim_or_path, str):
            prim = stage.GetPrimAtPath(prim_or_path)
        else:
            prim = prim_or_path

        if not prim.IsA(UsdGeom.Xformable):
            raise RuntimeError(f"Prim {prim_or_path} is not a UsdGeom.Xformable")

        # GetParentToWorldTransform returns the parent's accumulated world transform
        # (Gf.Matrix4d, USD convention: translation in last row), excluding the prim's own local xform.
        xform_cache = UsdGeom.XformCache(time)
        parent_to_world = xform_cache.GetParentToWorldTransform(prim)

        # Convert local_to_world tensor to Gf.Matrix4d (transpose back to USD convention)
        ltw_gf = Gf.Matrix4d(local_to_world.detach().T.cpu().numpy().tolist())

        # local = parent_to_world_inv @ local_to_world
        local_gf = parent_to_world.GetInverse() * ltw_gf

        xform_op = UsdGeom.Xformable(prim).MakeMatrixXform()
        xform_op.Set(local_gf, time)
        if isinstance(file_path_or_stage, str):
            stage.Save()
    finally:
        del stage


def get_local_to_world_transform(file_path_or_stage, prim_or_path, time=None):
    """Get a prim's local xform so that ComputeLocalToWorldTransform equals local_to_world.

    Args:
        file_path_or_stage (Usd.Stage or str):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        prim_or_path (Usd.Prim or str): The prim or path to the prim to get the transform from.
        time: USD time code.
    
    Returns:
        torch.Tensor: The local-to-world transform matrix of shape (4, 4).
    """
    if time is None:
        time = Usd.TimeCode.Default()

    if isinstance(file_path_or_stage, str):
        stage = Usd.Stage.Open(file_path_or_stage)
    else:
        stage = file_path_or_stage

    try:
        if isinstance(prim_or_path, str):
            prim = stage.GetPrimAtPath(prim_or_path)
        else:
            prim = prim_or_path
        xformable = UsdGeom.Xformable(prim)
        if xformable:
            transform = xformable.ComputeLocalToWorldTransform(time)
            if transform != IDENTITY_TRANSFORM:
                return torch.tensor(np.asarray(transform, dtype=np.float64)).permute(1, 0)
        return None
    finally:
        del stage
