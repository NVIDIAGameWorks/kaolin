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

import torch
import numpy as np

try:
    from pxr import Usd, UsdGeom, Sdf
except ImportError:
    pass

from .utils import _get_stage_from_maybe_file, get_scene_paths, create_stage

__all__ = [
    'import_voxelgrid',
    'import_voxelgrids',
    'add_voxelgrid',
    'export_voxelgrid',
    'export_voxelgrids',
]

def import_voxelgrid(file_path_or_stage, scene_path, time=None):
    r"""Import a single voxelgrid from a USD file or stage.

    Assumes that the USD voxelgrid is defined by a point instancer. Converts the coordinates
    of each point instance to an occupied voxel. The output grid size is determined by the `grid_size`
    primvar. If not specified, grid size will be determined by the axis with the largest number of occupied
    voxels. The output voxelgrid will be of shape ``[grid_size, grid_size, grid_size]``.

    Args:
        file_path_or_stage (str or Usd.Stage):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        scene_path (str): Scene path within the USD file indicating which PointInstancer primitive
            to import as a voxelgrid.
        time (convertible to float, optional): Positive integer indicating the time at which to retrieve parameters.
    Returns:
        torch.BoolTensor

    Example:
        >>> voxelgrid = torch.rand(32, 32, 32) > 0.5
        >>> stage = export_voxelgrid('./new_stage.usd', voxelgrid, scene_path='/World/voxelgrid')
        >>> voxelgrid_imp = import_voxelgrid('./new_stage.usd',
        ...                                  scene_path='/World/voxelgrid')
        >>> voxelgrid_imp.shape
        torch.Size([32, 32, 32])
    """
    if time is None:
        time = Usd.TimeCode.Default()
    voxelgrid_list = import_voxelgrids(file_path_or_stage, [scene_path], times=[time])
    return voxelgrid_list[0]


def import_voxelgrids(file_path_or_stage, scene_paths=None, times=None):
    r"""Import one or more voxelgrids from a USD file.

    Assumes that the USD voxelgrid is defined by a point instancer. Converts the coordinates
    of each point instance to an occupied voxel. The output grid size is determined from the `grid_size`
    primvar. If not specified, grid size will be determined by the axis with the largest number of occupied
    voxels. The output voxelgrid will be of shape ``[grid_size, grid_size, grid_size]``.

    Args:
        file_path_or_stage (str or Usd.Stage):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        scene_paths (list of str, optional): Scene path(s) within the USD file indicating which PointInstancer
            primitive(s) to import. If None, will return all pointclouds found based on PointInstancer
            prims with `kaolin_type` primvar set to `VoxelGrid`.
        times (list of int): Positive integers indicating the time at which to retrieve parameters.
    Returns:
        (list of torch.BoolTensor)

    Example:
        >>> voxelgrid_1 = torch.rand(32, 32, 32) > 0.5
        >>> voxelgrid_2 = torch.rand(32, 32, 32) > 0.5
        >>> stage = export_voxelgrids('./new_stage.usd', [voxelgrid_1, voxelgrid_2])
        >>> voxelgrid_imp = import_voxelgrids('./new_stage.usd')
        >>> len(voxelgrid_imp)
        2
        >>> voxelgrid_imp[0].shape
        torch.Size([32, 32, 32])
    """
    stage = _get_stage_from_maybe_file(file_path_or_stage)

    # If scene path not specified, find all point clouds
    if scene_paths is None:
        scene_paths = []
        for p in stage.Traverse():
            is_point_instancer = UsdGeom.PointInstancer(p)
            if is_point_instancer and p.GetAttribute('primvars:kaolin_type').Get() == 'VoxelGrid':
                scene_paths.append(p.GetPath())
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)

    voxelgrids = []
    for scene_path, time in zip(scene_paths, times):
        prim = stage.GetPrimAtPath(scene_path)
        assert prim, f'The prim at {scene_path} does not exist.'

        instancer = UsdGeom.PointInstancer(prim)
        assert instancer   # Currently only support pointclouds from point instancers

        voxel_indices = torch.from_numpy(np.array(instancer.GetPositionsAttr().Get(time=time), dtype=np.int64))
        bounds = voxel_indices.max(dim=0)[0]
        max_bound = bounds.max()
        grid_size = prim.GetAttribute('primvars:grid_size').Get(time=time)
        if grid_size is not None:
            assert max_bound < grid_size
        else:
            grid_size = max_bound
        voxelgrid = torch.zeros([grid_size, grid_size, grid_size], dtype=torch.bool)
        voxelgrid[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.
        voxelgrids.append(voxelgrid)

    return voxelgrids


def add_voxelgrid(stage, voxelgrid, scene_path, time=None):
    r"""Add a voxelgrid to an existing USD stage.

    Add a voxelgrid where occupied voxels are defined by non-zero values. The voxelgrid is
    represented by point instances of a cube centered at each occupied index coordinate and scaled.
    The stage is modified but not saved to disk.

    Args:
        stage (Usd.Stage): Stage onto which to add the voxelgrid.
        voxelgrid (torch.BoolTensor): Binary voxelgrid of shape ``(N, N, N)``.
        scene_path (str): Absolute path of voxelgrid within the USD file scene. Must be a valid Sdf.Path.
        time (convertible to float, optional): Positive integer defining the time at which the supplied parameters
            correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> stage = create_stage('./new_stage.usd')
        >>> voxelgrid = torch.rand(32, 32, 32) > 0.5
        >>> stage = add_voxelgrid(stage, voxelgrid, '/World/VoxelGrids/voxelgrid_0')
        >>> stage.Save()
    """
    scene_path = Sdf.Path(scene_path)
    if time is None:
        time = Usd.TimeCode.Default()

    if stage.GetPrimAtPath(scene_path):
        instancer_prim = stage.GetPrimAtPath(scene_path)
    else:
        instancer_prim = stage.DefinePrim(scene_path, 'PointInstancer')
    instancer = UsdGeom.PointInstancer(instancer_prim)
    assert instancer
    cube = UsdGeom.Cube.Define(stage, f'{scene_path}/cube')
    cube.GetSizeAttr().Set(1.0)
    instancer.CreatePrototypesRel().SetTargets([cube.GetPath()])

    # Get each occupied voxel's centre coordinate
    points = torch.nonzero(voxelgrid, as_tuple=False).float()

    # Generate instancer parameters
    indices = [0] * points.shape[0]
    positions = points.detach().cpu().tolist()
    scales = [(1.,) * 3] * points.size(0)

    # Populate point instancer
    instancer.GetProtoIndicesAttr().Set(indices, time=time)
    instancer.GetPositionsAttr().Set(positions, time=time)
    instancer.GetScalesAttr().Set(scales, time=time)

    # Centre at (0, 0, 0) and scale points
    adjusted_scale = 1. / (voxelgrid.size(0) - 1)
    scale_op = instancer.GetPrim().GetAttribute('xformOp:scale')
    if not scale_op:
        scale_op = UsdGeom.Xformable(instancer).AddScaleOp()
    scale_op.Set((adjusted_scale,) * 3, time=time)
    translate_op = instancer.GetPrim().GetAttribute('xformOp:translate')
    if not translate_op:
        UsdGeom.Xformable(instancer).AddTranslateOp()
    translate_op.Set((-0.5,) * 3, time=time)

    # Create a primvar to record the voxelgrid size
    prim = stage.GetPrimAtPath(instancer.GetPath())
    pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar('grid_size', Sdf.ValueTypeNames.Int)
    pv.Set(voxelgrid.size(0))

    # Create a primvar to identify the poitn instancer as a Kaolin VoxelGrid
    pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar('kaolin_type', Sdf.ValueTypeNames.String)
    pv.Set('VoxelGrid')

    return stage


def export_voxelgrid(file_path, voxelgrid, scene_path='/World/VoxelGrids/voxelgrid_0', time=None):
    r"""Export a single voxelgrid to a USD scene.

    Export a binary voxelgrid where occupied voxels are defined by non-zero values. The voxelgrid is
    represented by point instances of a cube centered at each occupied index coordinates. The voxelgrid
    will be scaled so that it fits within a unit cube. The stage is then saved to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        voxelgrid (torch.BoolTensor): Binary voxelgrid of shape ``(N, N, N)``.
        scene_path (str, optional): Absolute path of voxelgrid within the USD file scene. Must be a valid Sdf.Path.
            If no path is provided, a default path is used.
        time (convertible to float, optional): Positive integer defining the time at which the supplied parameters
            correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> voxelgrid = torch.rand(32, 32, 32) > 0.5
        >>> stage = export_voxelgrid('./new_stage.usd', voxelgrid)
    """
    if time is None:
        time = Usd.TimeCode.Default()
    stage = export_voxelgrids(file_path, [voxelgrid], [scene_path], times=[time])
    return stage


def export_voxelgrids(file_path, voxelgrids, scene_paths=None, times=None):
    r"""Export one or more voxelgrids to a USD scene.

    Export one or more binary voxelgrids where occupied voxels are defined by non-zero values. The voxelgrids are
    represented by point instances of a cube centered at each occupied index coordinates and scaled. The voxelgrids
    will be scaled so that it fits within a unit cube. The stage is then saved to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        voxelgrids (list of torch.BoolTensor): List of binary voxelgrid(s) of shape ``(N, N, N)``.
        scene_path (list of str, optional): Absolute path(s) of voxelgrid within the USD file scene.
            Must be a valid Sdf.Path. If no path is provided, a default path is used.
        times (list of int): Positive integers defining the time at which the supplied parameters correspond to.
    Returns:
        (Usd.Stage)

    Example:
        >>> voxelgrid_1 = torch.rand(32, 32, 32) > 0.5
        >>> voxelgrid_2 = torch.rand(32, 32, 32) > 0.5
        >>> stage = export_voxelgrids('./new_stage.usd', [voxelgrid_1, voxelgrid_2])
    """
    if scene_paths is None:
        scene_paths = [f'/World/VoxelGrids/voxelgrid_{i}' for i in range(len(voxelgrids))]
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)
    assert len(voxelgrids) == len(scene_paths)
    stage = create_stage(file_path)
    for scene_path, voxelgrid, time in zip(scene_paths, voxelgrids, times):
        add_voxelgrid(stage, voxelgrid, scene_path, time=time)
    stage.Save()

    return stage
