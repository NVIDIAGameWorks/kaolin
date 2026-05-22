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

from typing import Optional
import torch
import numpy as np
from .utils import _get_stage_from_maybe_file
from kaolin.physics.simplicits import PhysicsPoints, SkinnedPhysicsPoints, SkinnedPoints

from pxr import Usd, UsdGeom, UsdVol

__all__ = [
    'add_physics_material',
    'add_skinned_physics',
    'get_physics_material',
    'get_skinned_physics',
    'get_physics_materials_instance_names',
    'get_skinned_physics_instance_names',
    'get_all_physics_materials',
    'get_all_skinned_physics',
]

def add_physics_material(file_path_or_stage,
                         path_or_prim,
                         physics_points: PhysicsPoints,
                         material_name: Optional[str]='default',
                         time: Optional[Usd.TimeCode] = None,
                         overwrite: bool = False):
    r"""
    Add physics material to USD Prim.

    Those contains the information necessary to define the physical property of an object,
    typically what output `VoMP <https://research.nvidia.com/labs/sil/projects/vomp/>`_ and
    the parameters that :class:`SimplicitsObject` take as inputs for training.

    see :class:`PhysicsPoints` for more details.


    Args:
        file_path_or_stage (str or Usd.Stage):
            USD.Stage or path to file to stage to be opened.
        path_or_prim (str or Usd.Prim):
            Usd.Prim, or absolute path within the USD file stage,
            must be a valid ``Sdf.Path``.
        physics_points (PhysicsPoints):
            PhysicsPoints object to add to the USD Prim.
        material_name (optional, str):
            Name of the physics material attribute API. Default: 'default'.
        time (optional, Usd.TimeCode):
            TimeCode defining the time at which the supplied parameters correspond to.
            Default: Usd.TimeCode.Default().
        overwrite (optional, bool):
            if True, and material already exist at ``path_or_prim``, overwrite the material
            with new attributes. Otherwise raise FileExistsError.

    Returns:
        (Usd.Prim): The new material's Prim.
    """
    if time is None:
        time = Usd.TimeCode.Default()

    stage = _get_stage_from_maybe_file(file_path_or_stage)
    try:
        if isinstance(path_or_prim, Usd.Prim):
            prim = path_or_prim
        else:
            prim = stage.GetPrimAtPath(path_or_prim)
        if not overwrite and prim.HasAPI("KaolinPhysicsMaterialAPI", material_name):
            raise FileExistsError(f"Prim at path '{prim.GetPath()}' already has a physics material attribute API with namespace '{material_name}'")
        
        prim.ApplyAPI("KaolinPhysicsMaterialAPI", material_name)

        prim.GetAttribute(f'kaolin_physics_material:{material_name}:pts').Set(physics_points.pts.detach().cpu().numpy(), time=time)

        prim.GetAttribute(f'kaolin_physics_material:{material_name}:yms').Set(physics_points.yms.detach().cpu().numpy(), time=time)

        prim.GetAttribute(f'kaolin_physics_material:{material_name}:prs').Set(physics_points.prs.detach().cpu().numpy(), time=time)

        prim.GetAttribute(f'kaolin_physics_material:{material_name}:rhos').Set(physics_points.rhos.detach().cpu().numpy(), time=time)

        prim.GetAttribute(f'kaolin_physics_material:{material_name}:appx_vol').Set(physics_points.appx_vol.detach().item(), time=time)

        if not isinstance(file_path_or_stage, Usd.Stage):
            stage.Save()
    finally:
        del stage, file_path_or_stage

    return prim

def add_skinned_physics(file_path_or_stage,
                        path_or_prim,
                        skinned_physics_points: SkinnedPhysicsPoints,
                        instance_name: Optional[str] = 'default',
                        time: Optional[Usd.TimeCode] = None,
                        overwrite: bool = False):
    r"""
    Add Skinned physics attributes to USD Prim.

    Contains the information necessary to simulate an object within a :class:`SimplicitsScene`,
    see :class:`SkinnedPhysicsPoints` for more details.

    Args:
        file_path_or_stage (str or Usd.Stage):
            USD.Stage or path to file to stage to be opened.
        path_or_prim (str or Usd.Prim):
            Usd.Prim, or absolute path within the USD file stage,
            must be a valid ``Sdf.Path``.
        skinned_physics_points (SkinnedPhysicsPoints):
            SkinnedPhysicsPoints object to add to the USD Prim.
        instance_name (optional, str):
            Name of the skinned physics instance name,
            allowing to store multiple skinned physics behavior per prim. Default: 'default'.
        time (optional, Usd.TimeCode):
            TimeCode defining the time at which the supplied parameters correspond to.
            Default: Usd.TimeCode.Default().
        overwrite (optional, bool):
            if True, and setup already exist at ``path_or_prim``, overwrite the setup
            with new attributes. Otherwise raise FileExistsError.

    Returns:
        (Usd.Prim): The Prim with skinned physics applied.
    """
    if time is None:
        time = Usd.TimeCode.Default()

    stage = _get_stage_from_maybe_file(file_path_or_stage)
    try:
        if isinstance(path_or_prim, Usd.Prim):
            prim = path_or_prim
        else:
            prim = stage.GetPrimAtPath(path_or_prim)
        if not overwrite and prim.HasAPI("KaolinSkinnedPhysicsAPI", instance_name):
            raise FileExistsError(f"Prim at path '{prim.GetPath()}' already has a skinned physics attribute API with namespace '{instance_name}'")
        prim.ApplyAPI("KaolinSkinnedPhysicsAPI", instance_name)

        prim.GetAttribute(f'kaolin_skinned_physics:{instance_name}:pts').Set(skinned_physics_points.pts.detach().cpu().numpy(), time=time)

        prim.GetAttribute(f'kaolin_skinned_physics:{instance_name}:yms').Set(skinned_physics_points.yms.detach().cpu().numpy(), time=time)

        prim.GetAttribute(f'kaolin_skinned_physics:{instance_name}:prs').Set(skinned_physics_points.prs.detach().cpu().numpy(), time=time)

        prim.GetAttribute(f'kaolin_skinned_physics:{instance_name}:rhos').Set(skinned_physics_points.rhos.detach().cpu().numpy(), time=time)

        prim.GetAttribute(f'kaolin_skinned_physics:{instance_name}:appx_vol').Set(float(skinned_physics_points.appx_vol.detach().item()), time=time)

        prim.GetAttribute(f'kaolin_skinned_physics:{instance_name}:skinning_weights').Set(skinned_physics_points.skinning_weights.detach().cpu().numpy(), time=time)

        prim.GetAttribute(f'kaolin_skinned_physics:{instance_name}:dwdx').Set(skinned_physics_points.dwdx.detach().cpu().numpy(), time=time)

        if skinned_physics_points.renderable is not None:
            prim.GetAttribute(f'kaolin_skinned_physics:{instance_name}:renderable_skinning_weights').Set(
                skinned_physics_points.renderable.skinning_weights.detach().cpu().numpy(), time=time)

        if not isinstance(file_path_or_stage, Usd.Stage):
            stage.Save()
    finally:
        del stage, file_path_or_stage

    return prim

def get_physics_material(file_path_or_stage, prim_or_path, material_name: Optional[str] = 'default',
                         time: Optional[Usd.TimeCode] = None):
    r"""
    Get physics material parameters from a prim with a given material name.

    Args:
        file_path_or_stage (str or Usd.Stage):
            USD.Stage or path to file to stage to be opened.
        prim_or_path (str or Usd.Prim):
            Usd.Prim, or absolute path within the USD file stage,
            must be a valid ``Sdf.Path``.
        material_name (optional, str):
            Name of the physics material. Default: 'default'.
        time (optional, Usd.TimeCode):
            TimeCode defining the time at which the supplied parameters correspond to.
            Default: Usd.TimeCode.Default().
    Returns:
        (PhysicsPoints or None): ``PhysicsPoints`` populated from the prim's
        attributes if the API binding for ``material_name`` is found, otherwise
        ``None``.
    """
    if time is None:
        time = Usd.TimeCode.Default()

    stage = _get_stage_from_maybe_file(file_path_or_stage)
    try:
        if isinstance(prim_or_path, Usd.Prim):
            prim = prim_or_path
        else:
            prim = stage.GetPrimAtPath(prim_or_path)

        if not prim.HasAPI("KaolinPhysicsMaterialAPI", material_name):
            return None
        return PhysicsPoints(
            pts=torch.tensor(np.array(prim.GetAttribute(f'kaolin_physics_material:{material_name}:pts').Get(time=time)), dtype=torch.float32),
            yms=torch.tensor(np.array(prim.GetAttribute(f'kaolin_physics_material:{material_name}:yms').Get(time=time)), dtype=torch.float32),
            prs=torch.tensor(np.array(prim.GetAttribute(f'kaolin_physics_material:{material_name}:prs').Get(time=time)), dtype=torch.float32),
            rhos=torch.tensor(np.array(prim.GetAttribute(f'kaolin_physics_material:{material_name}:rhos').Get(time=time)), dtype=torch.float32),
            appx_vol=torch.tensor(prim.GetAttribute(f'kaolin_physics_material:{material_name}:appx_vol').Get(time=time), dtype=torch.float32)
        )
    finally:
        del stage, file_path_or_stage

def _get_renderable_pts(prim, time, attribute=None):
    r"""
    Get the renderable points from a USD prim.

    Args:
        prim (Usd.Prim): USD.Prim to get the renderable points from.
        time (optional, Usd.TimeCode):
            TimeCode defining the time at which the supplied parameters correspond to.
            Default: Usd.TimeCode.Default().
        attribute (str):
            Name of the attribute to get the renderable points from.
            Default: None.

    Returns:
        (torch.Tensor): Renderable points tensor of shape :math:`(N, 3)`.
    """
    if attribute is not None:
        pts = prim.GetAttribute(attribute).Get(time=time)
    elif prim.GetTypeName() == 'Mesh':
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get(time=time)
    elif prim.GetTypeName() == 'Points':
        pts = UsdGeom.Points(prim).GetPointsAttr().Get(time=time)
    elif prim.GetTypeName() == 'PointInstancer':
        pts = UsdGeom.PointInstancer(prim).GetPositionsAttr().Get(time=time)
    elif prim.GetTypeName() == 'ParticleField3DGaussianSplat':
        pts = UsdVol.ParticleField3DGaussianSplat(prim).GetPositionsAttr().Get(time=time)
    else:
        raise ValueError(f"Unsupported prim type: {prim.GetTypeName()}, "
                          "must assign 'attribute' to get the renderable points from the prim")
    return torch.from_numpy(np.array(pts, dtype=np.float32)).reshape(-1, 3)

def get_skinned_physics(file_path_or_stage, prim_or_path, instance_name: Optional[str] = 'default',
                        time: Optional[Usd.TimeCode] = None, attribute: Optional[str] = None):
    r"""
    Get skinned physics parameters from a prim with a given instance name.

    Contains the information necessary to simulate an object within a :class:`SimplicitsScene`,
    see :class:`SkinnedPhysicsPoints` for more details.

    Args:
        file_path_or_stage (str or Usd.Stage):
            USD.Stage or path to file to stage to be opened.
        prim_or_path (str or Usd.Prim):
            Usd.Prim, or absolute path within the USD file stage,
            must be a valid ``Sdf.Path``.
        instance_name (optional, str):
            Name of the skinned physics instance. Default: 'default'.
        time (optional, Usd.TimeCode):
            TimeCode defining the time at which the supplied parameters correspond to.
            Default: Usd.TimeCode.Default().
        attribute (optional, str):
            Name of the attribute to get the renderable points from.
            Default: Attempt to automatically get the renderable points from the prim, works for Mesh, Points, PointInstancer, and ParticleField3DGaussianSplat.
    Returns:
        (SkinnedPhysicsPoints or None): ``SkinnedPhysicsPoints`` populated from
        the prim's attributes if the API binding for ``instance_name`` is
        found, otherwise ``None``. The ``renderable`` field is populated from
        the prim geometry (or ``attribute`` if provided).
    """
    if time is None:
        time = Usd.TimeCode.Default()

    stage = _get_stage_from_maybe_file(file_path_or_stage)
    try:
        if isinstance(prim_or_path, Usd.Prim):
            prim = prim_or_path
        else:
            prim = stage.GetPrimAtPath(prim_or_path)
        if not prim.HasAPI("KaolinSkinnedPhysicsAPI", instance_name):
            return None

        renderable_skinning_weights = prim.GetAttribute(f'kaolin_skinned_physics:{instance_name}:renderable_skinning_weights').Get(time=time)
        if renderable_skinning_weights is None:
            renderable = None
        else:
            renderable_pts = _get_renderable_pts(prim, time, attribute)
            renderable = SkinnedPoints(
                pts=renderable_pts,
                skinning_weights=torch.tensor(np.array(renderable_skinning_weights), dtype=torch.float32).reshape(renderable_pts.shape[0], -1)
            )
        pts = torch.tensor(np.array(prim.GetAttribute(f'kaolin_skinned_physics:{instance_name}:pts').Get(time=time)), dtype=torch.float32)
        return SkinnedPhysicsPoints(
            pts=pts,
            yms=torch.tensor(np.array(prim.GetAttribute(f'kaolin_skinned_physics:{instance_name}:yms').Get(time=time)), dtype=torch.float32),
            prs=torch.tensor(np.array(prim.GetAttribute(f'kaolin_skinned_physics:{instance_name}:prs').Get(time=time)), dtype=torch.float32),
            rhos=torch.tensor(np.array(prim.GetAttribute(f'kaolin_skinned_physics:{instance_name}:rhos').Get(time=time)), dtype=torch.float32),
            appx_vol=torch.tensor(prim.GetAttribute(f'kaolin_skinned_physics:{instance_name}:appx_vol').Get(time=time), dtype=torch.float32),
            skinning_weights=torch.tensor(np.array(prim.GetAttribute(f'kaolin_skinned_physics:{instance_name}:skinning_weights').Get(time=time)), dtype=torch.float32).reshape(pts.shape[0], -1),
            dwdx=torch.tensor(np.array(prim.GetAttribute(f'kaolin_skinned_physics:{instance_name}:dwdx').Get(time=time)), dtype=torch.float32).reshape(pts.shape[0], -1, 3),
            renderable=renderable
        )

    finally:
        del stage, file_path_or_stage

def get_physics_materials_instance_names(file_path_or_stage, prim_or_path):
    r"""
    Get all physics materials namespaces from a prim with a given attribute API namespace.

    Args:
        file_path_or_stage (str or Usd.Stage): USD.Stage or path to file to stage to be opened.
        prim_or_path (str or Usd.Prim): Usd.Prim, or absolute path within the USD file stage,
            must be a valid ``Sdf.Path``.
    Returns:
        (list): List of physics materials namespaces found in the prim.
    """
    stage = _get_stage_from_maybe_file(file_path_or_stage)
    try:
        if isinstance(prim_or_path, Usd.Prim):
            prim = prim_or_path
        else:
            prim = stage.GetPrimAtPath(prim_or_path)
        applied = prim.GetAppliedSchemas()
        return [':'.join(schema.split(':')[1:])
            for schema in applied if schema.startswith("KaolinPhysicsMaterialAPI")]
    finally:
        del stage, file_path_or_stage

def get_skinned_physics_instance_names(file_path_or_stage,
                                        prim_or_path):
    r"""
    Get all skinned physics instances from a prim with a given attribute API namespace.

    Args:
        file_path_or_stage (str or Usd.Stage): USD.Stage or path to file to stage to be opened.
        prim_or_path (str or Usd.Prim): Usd.Prim, or absolute path within the USD file stage,
            must be a valid ``Sdf.Path``.
    Returns:
        (list): List of skinned physics instances found in the prim.
    """
    stage = _get_stage_from_maybe_file(file_path_or_stage)
    try:
        if isinstance(prim_or_path, Usd.Prim):
            prim = prim_or_path
        else:
            prim = stage.GetPrimAtPath(prim_or_path)
        applied = prim.GetAppliedSchemas()
        return [':'.join(schema.split(':')[1:])
            for schema in applied if schema.startswith("KaolinSkinnedPhysicsAPI")]
    finally:
        del stage, file_path_or_stage

def get_all_physics_materials(file_path_or_stage, prim_or_path, time: Optional[Usd.TimeCode] = None):
    r"""
    Get all physics materials from a prim. See :func:`get_physics_material` for more details.

    Args:
        file_path_or_stage (str or Usd.Stage): USD.Stage or path to file to stage to be opened.
        prim_or_path (str or Usd.Prim): Usd.Prim, or absolute path within the USD file stage,
            must be a valid ``Sdf.Path``.
        time (optional, Usd.TimeCode):
            TimeCode defining the time at which the supplied parameters correspond to.
            Default: Usd.TimeCode.Default().
    Returns:
        (dict): Dictionary of physics materials namespaces and their corresponding physics materials.
    """
    if time is None:
        time = Usd.TimeCode.Default()

    stage = _get_stage_from_maybe_file(file_path_or_stage)

    try:
        if isinstance(prim_or_path, Usd.Prim):
            prim = prim_or_path
        else:
            prim = stage.GetPrimAtPath(prim_or_path)
        physics_materials_instance_names = get_physics_materials_instance_names(stage, prim)
        output = {}
        for physics_material_instance_name in physics_materials_instance_names:
            output[physics_material_instance_name] = get_physics_material(
                stage, prim, physics_material_instance_name, time)
    finally:
        del stage, file_path_or_stage
    return output

def get_all_skinned_physics(file_path_or_stage, prim_or_path, time: Optional[Usd.TimeCode] = None, attribute: Optional[str] = None):
    r"""
    Get all skinned physics instances from a prim. see :func:`get_skinned_physics` for more details.

    Args:
        file_path_or_stage (str or Usd.Stage): USD.Stage or path to file to stage to be opened.
        prim_or_path (str or Usd.Prim): Usd.Prim, or absolute path within the USD file stage,
            must be a valid ``Sdf.Path``.
        time (optional, Usd.TimeCode):
            TimeCode defining the time at which the supplied parameters correspond to.
            Default: Usd.TimeCode.Default().
        attribute (optional, str):
            Name of the attribute to get the skinned physics from.
            Default: Attempt to automatically get the renderable points from the prim,
            works for Mesh, Points, PointInstancer, and ParticleField3DGaussianSplat.
    Returns:
        (dict of str to SkinnedPhysicsPoints): Dictionary of skinned physics namespaces and their corresponding skinned physics.
    """
    if time is None:
        time = Usd.TimeCode.Default()

    stage = _get_stage_from_maybe_file(file_path_or_stage)
    try:
        if isinstance(prim_or_path, Usd.Prim):
            prim = prim_or_path
        else:
            prim = stage.GetPrimAtPath(prim_or_path)
        skinned_physics_instance_names = get_skinned_physics_instance_names(stage, prim)
        output = {}
        for skinned_physics_instance_name in skinned_physics_instance_names:
            output[skinned_physics_instance_name] = get_skinned_physics(
                stage, prim, skinned_physics_instance_name, time, attribute)
        return output
    finally:
        del stage, file_path_or_stage
