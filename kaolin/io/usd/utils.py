# Copyright (c) 2019,20-21-23 NVIDIA CORPORATION & AFFILIATES.
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

import itertools
import os
import re

import torch

try:
    from pxr import Usd, UsdGeom
except ImportError:
    pass

__all__  = [
    'get_scene_paths',
    'get_authored_time_samples',
    'create_stage',
]

def _get_stage_from_maybe_file(file_path_or_stage):
    """Returns a stage from a file or itself if input is a Usd.Stage

    Args:
        file_path_or_stage (str or Usd.Stage):
            Path to usd file (/*.usd, /*.usda) or :class:`Usd.Stage`.

    Returns:
        (Usd.Stage): The output stage.
    """
    if isinstance(file_path_or_stage, Usd.Stage):
        return file_path_or_stage
    else:
        assert os.path.exists(file_path_or_stage)
        return Usd.Stage.Open(file_path_or_stage)

def get_scene_paths(file_path_or_stage, scene_path_regex=None, prim_types=None,
                    conditional=lambda x: True):
    r"""Return all scene paths contained in specified file or stage. Filter paths with regular
    expression in `scene_path_regex` if provided.

    Args:
        file_path_or_stage (str or Usd.Stage):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        scene_path_regex (str, optional):
            Optional regular expression used to select returned scene paths.
        prim_types (list of str, str, optional):
            Optional list of valid USD Prim types used to select scene paths, or a single USD Prim type string.
        conditional (function path: Bool): Custom conditionals to check

    Returns:
        (list of str): List of filtered scene paths.

    Example:
        >>> # Create a stage with some meshes
        >>> vertices_list = [torch.rand(3, 3) for _ in range(3)]
        >>> faces_list = [torch.tensor([[0, 1, 2]]) for _ in range(3)]
        >>> stage = export_meshes('./new_stage.usd', vertices=vertices_list, faces=faces_list)
        >>> # Retrieve scene paths
        >>> get_scene_paths('./new_stage.usd', prim_types=['Mesh'])
        [Sdf.Path('/World/Meshes/mesh_0'), Sdf.Path('/World/Meshes/mesh_1'), Sdf.Path('/World/Meshes/mesh_2')]
        >>> get_scene_paths('./new_stage.usd', scene_path_regex=r'.*_0', prim_types=['Mesh'])
        [Sdf.Path('/World/Meshes/mesh_0')]
    """
    stage = _get_stage_from_maybe_file(file_path_or_stage)
    if scene_path_regex is None:
        scene_path_regex = '.*'
    if prim_types is not None:
        if isinstance(prim_types, str):
            prim_types = [prim_types]
        prim_types = [pt.lower() for pt in prim_types]

    scene_paths = []
    for p in stage.Traverse():
        is_valid_prim_type = prim_types is None or p.GetTypeName().lower() in prim_types
        is_valid_scene_path = re.match(scene_path_regex, str(p.GetPath()))
        passes_conditional = conditional(p)
        if is_valid_prim_type and is_valid_scene_path and passes_conditional:
            scene_paths.append(p.GetPath())
    return scene_paths

def get_authored_time_samples(file_path_or_stage):
    r"""
    Returns *all* authored time samples within the USD, aggregated across all primitives.

    Args:
        file_path_or_stage (str or Usd.Stage):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.

    Returns:
        (list)
    """
    stage = _get_stage_from_maybe_file(file_path_or_stage)
    scene_paths = get_scene_paths(stage)
    res = set()
    for scene_path in scene_paths:
        prim = stage.GetPrimAtPath(scene_path)
        attr = prim.GetAttributes()
        res.update(set(itertools.chain.from_iterable([x.GetTimeSamples() for x in attr])))
    return sorted(res)

def create_stage(file_path, up_axis='Y'):
    r"""Create a new USD file and return an empty stage.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        up_axis (['Y', 'Z']): Specify the stage up axis. Choose from ``['Y', 'Z']``.

    Returns:
        (Usd.Stage)

    Example:
        >>> stage = create_stage('./new_stage.usd', up_axis='Z')
        >>> type(stage)
        <class 'pxr.Usd.Stage'>
    """
    assert os.path.exists(os.path.dirname(file_path)), f'Directory {os.path.dirname(file_path)} not found.'
    stage = Usd.Stage.CreateNew(str(file_path))
    world = stage.DefinePrim('/World', 'Xform')
    stage.SetDefaultPrim(world)
    UsdGeom.SetStageUpAxis(stage, up_axis)
    return stage
