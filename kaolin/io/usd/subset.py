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
from pxr import Usd, UsdGeom, Vt

__all__ = [
    'add_subset',
    'import_subsets'
]

def add_subset(file_path_or_stage, prim_or_path, name, indices, family_name='part', override=False):
    r"""Add subset to an existing Prim.

    Args:
        file_path_or_stage (str or Usd.Stage):
            USD.Stage or path to USD file to be opened.
            If a USD.Stage is provided, the stage is not saved at the end.
        prim_or_path (str or Usd.Prim):
            Usd.Prim on which to add the subset, or absolute path within
            the USD file scene. Must be a valid ``Sdf.Path``.
        name (str): name of the subset.
        indices (torch.LongTensor): indices of the elements that represent the subset.
        family_name (str):
            The family groups subsets by purpose (e.g. ``'part'`` for segmentation
            parts, ``'materialBind'`` for material assignments).
            Default: ``'part'``.
        override (bool):
            If True, reassign the indices on an existing subset with the same name.
            If False (default), raise a ``ValueError`` if a subset with that name
            already exists.

    Returns:
        (Usd.Prim): The Prim of the generated subset.
    """
    if isinstance(file_path_or_stage, Usd.Stage):
        stage = file_path_or_stage
    else:
        stage = Usd.Stage.Open(file_path_or_stage)
    try:
        if isinstance(prim_or_path, Usd.Prim):
            prim = prim_or_path
        else:
            prim = stage.GetPrimAtPath(prim_or_path)

        if not prim.IsValid():
            raise ValueError(f"No prim found at path '{prim_or_path}'")

        existing_path = prim.GetPath().AppendChild(name)
        if stage.GetPrimAtPath(existing_path).IsValid():
            if not override:
                raise ValueError(
                    f"Subset '{name}' already exists at '{existing_path}'; "
                    f"use override=True to replace it")
            existing_subset = UsdGeom.Subset(stage.GetPrimAtPath(existing_path))
            existing_subset.GetIndicesAttr().Set(Vt.IntArray.FromNumpy(indices.cpu().numpy()))
            existing_subset.GetFamilyNameAttr().Set(family_name)
            if not isinstance(file_path_or_stage, Usd.Stage):
                stage.Save()
            return stage.GetPrimAtPath(existing_path)

        if UsdGeom.Mesh(prim):
            element = UsdGeom.Tokens.face
        elif UsdGeom.Points(prim):
            element = UsdGeom.Tokens.point
        elif UsdGeom.PointInstancer(prim):
            element = UsdGeom.Tokens.point
        elif prim.GetTypeName() == "ParticleField3DGaussianSplat":
            element = UsdGeom.Tokens.point
        else:
            element = ''
        subset = UsdGeom.Subset.CreateGeomSubset(
            UsdGeom.Imageable(prim),
            name,
            element,
            indices=Vt.IntArray.FromNumpy(indices.cpu().numpy()),
            familyName=family_name
        )
        if not isinstance(file_path_or_stage, Usd.Stage):
            stage.Save()
    finally:
        del stage, file_path_or_stage
    return subset.GetPrim()

def _get_subsets_info(stage, prim_or_path, family_name=None):
    try:
        if isinstance(prim_or_path, Usd.Prim):
            prim = prim_or_path
        else:
            prim = stage.GetPrimAtPath(prim_or_path)

        if not prim.IsValid():
            raise ValueError(f"No prim found at path '{prim_or_path}'")

        if family_name is None:
            family_name = ''

        subsets = UsdGeom.Subset.GetGeomSubsets(
            UsdGeom.Imageable(prim), familyName=family_name) or []
        # TODO(cfujitsang): full path or just subset name? should indices / family_name be in a struct.
        output = {
            str(s.GetPath()): {
                'indices': torch.tensor(np.array(s.GetIndicesAttr().Get())),
                'family_name': s.GetFamilyNameAttr().Get()
            }
         for s in subsets}
    finally:
        del stage
    return output

def import_subsets(file_path_or_stage, prim_or_path, family_name=None):
    """Import subset from Prims.

    Args:
        file_path_or_stage (str or Usd.Stage):
            USD.Stage or path to file to stage to be opened.
        prim_or_path (str or Usd.Prim):
            Usd.Prim on which to get the subsets, or absolute path within
            the USD file scene. Must be a valid ``Sdf.Path``.
        family_name (optional, str):
            The family groups subsets by purpose (e.g. ``'part'`` for segmentation
            parts, ``'materialBind'`` for material assignments).
            Default: ``None``, i.e. return all subsets regardless of family.

    Returns:
        (dict of dict):
            dict mapping each subset's USD path to a dict with the following fields:

        * **indices** (torch.Tensor): indices of the subset.
        * **family_name** (str): family name.

    """
    if isinstance(file_path_or_stage, Usd.Stage):
        stage = file_path_or_stage
    else:
        stage = Usd.Stage.Open(file_path_or_stage)
    try:
        output = _get_subsets_info(stage, prim_or_path, family_name=family_name)
    finally:
        del stage, file_path_or_stage
    return output

