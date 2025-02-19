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

from collections import namedtuple
import numpy as np
import torch

try:
    from pxr import Usd, UsdGeom, Vt, Sdf
except ImportError:
    pass

from .utils import _get_stage_from_maybe_file, get_scene_paths, create_stage

pointcloud_return_type = namedtuple('pointcloud_return_type', ['points', 'colors', 'normals'])

__all__ = [
    'get_pointcloud_scene_paths',
    'import_pointcloud',
    'import_pointclouds',
    'get_pointcloud_bracketing_time_samples',
    'add_pointcloud',
    'export_pointcloud',
    'export_pointclouds'
]

def get_pointcloud_scene_paths(file_path_or_stage):
    r"""Returns all point cloud scene paths contained in specified file. Assumes that point
    clouds are exported using this API.

    Args:
        file_path_or_stage (str or Usd.Stage):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.

    Returns:
        (list of str): List of filtered scene paths.
    """
    # TODO(mshugrina): is passing prim_types='PointInstancer' the same as UsdGeom.PointInstancer(p) ?
    stage = _get_stage_from_maybe_file(file_path_or_stage)
    geom_points_paths = get_scene_paths(stage, prim_types=['Points'])
    point_instancer_paths = get_scene_paths(stage, prim_types=['PointInstancer'])
    return geom_points_paths + point_instancer_paths

def get_pointcloud_bracketing_time_samples(stage, scene_path, target_time):
    """Returns two time samples that bracket ``target_time`` for point cloud
    attributes at a specified scene_path.

    Args:
        stage (Usd.Stage)
        scene_path (str)
        target_time (Number)

    Returns:
        (iterable of 2 numbers)
    """
    # Note: can also get usd_attr.GetTimeSamples()
    prim = stage.GetPrimAtPath(scene_path)

    if UsdGeom.Points(prim):
        geom_points = UsdGeom.Points(prim)
        result = geom_points.GetPointsAttr().GetBracketingTimeSamples(target_time)
    elif UsdGeom.PointInstancer(prim):
        instancer = UsdGeom.PointInstancer(prim)
        result = instancer.GetPositionsAttr().GetBracketingTimeSamples(target_time)
    else:
        raise TypeError('The prim is neither UsdGeomPoints nor UsdGeomPointInstancer.')
    return result

def import_pointcloud(file_path_or_stage, scene_path, time=None):
    r"""Import a single pointcloud from a USD file or stage.

    Assumes that the USD pointcloud is interpreted using a point instancer or UsdGeomPoints. Converts the coordinates
    of each point instance to a point within the output pointcloud.

    Args:
        file_path_or_stage (str or Usd.Stage):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        scene_path (str): Scene path within the USD file indicating which primitive to import.
        time (convertible to float, optional): Positive integer indicating the time at which to retrieve parameters.
    Returns:
        namedtuple of:
            - **points** (torch.FloatTensor): of shape (num_points, 3)
            - **colors** (torch.FloatTensor): of shape (num_points, 3)
            - **normals** (torch.FloatTensor): of shape (num_points, 3) (not yet implemented)

    Example:
        >>> points = torch.rand(100, 3)
        >>> stage = export_pointcloud('./new_stage.usd', points, scene_path='/World/pointcloud')
        >>> points_imp = import_pointcloud(file_path='./new_stage.usd',
        ...                                scene_path='/World/pointcloud')[0]
        >>> points_imp.shape
        torch.Size([100, 3])
    """
    if time is None:
        time = Usd.TimeCode.Default()

    pointcloud_list = import_pointclouds(file_path_or_stage, [scene_path], times=[time])

    return pointcloud_return_type(*pointcloud_list[0])

def import_pointclouds(file_path_or_stage, scene_paths=None, times=None):
    r"""Import one or more pointclouds from a USD file or stage.

    Assumes that pointclouds are interpreted using point instancers or UsdGeomPoints. Converts the coordinates
    of each point instance to a point within the output pointcloud.

    Args:
        file_path_or_stage (str or Usd.Stage):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        scene_paths (list of str, optional): Scene path(s) within the USD file indicating which primitive(s)
            to import. If None, will return all pointclouds found based on PointInstancer or UsdGeomPoints prims with
            `kaolin_type` primvar set to `PointCloud`.
        times (list of int): Positive integers indicating the time at which to retrieve parameters.
    Returns:
        list of namedtuple of:
            - **points** (list of torch.FloatTensor): of shape (num_points, 3)
            - **colors** (list of torch.FloatTensor): of shape (num_points, 3)
            - **normals** (list of torch.FloatTensor): of shape (num_points, 2) (not yet implemented)

    Example:
        >>> points = torch.rand(100, 3)
        >>> stage = export_pointclouds('./new_stage.usd', [points, points, points])
        >>> pointclouds = import_pointclouds(file_path='./new_stage.usd')[0]
        >>> len(pointclouds)
        3
        >>> pointclouds[0].shape
        torch.Size([100, 3])
    """
    stage = _get_stage_from_maybe_file(file_path_or_stage)

    if scene_paths is None:
        scene_paths = get_pointcloud_scene_paths(stage)
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)

    pointclouds = []
    colors = []
    normals = []
    for scene_path, time in zip(scene_paths, times):
        prim = stage.GetPrimAtPath(scene_path)
        assert prim, f'The prim at {scene_path} does not exist.'

        if UsdGeom.Points(prim):
            geom_points = UsdGeom.Points(prim)
            pointclouds.append(torch.tensor(geom_points.GetPointsAttr().Get(time=time)))

            color = geom_points.GetDisplayColorAttr().Get(time=time)

            if color is None:
                colors.append(color)
            else:
                colors.append(torch.tensor(color))
        elif UsdGeom.PointInstancer(prim):
            instancer = UsdGeom.PointInstancer(prim)
            pointclouds.append(torch.tensor(instancer.GetPositionsAttr().Get(time=time)))
            colors.append(None)
        else:
            raise TypeError('The prim is neither UsdGeomPoints nor UsdGeomPointInstancer.')

    # TODO: place holders for normals for now
    normals = [None] * len(colors)

    params = [pointclouds, colors, normals]
    return [pointcloud_return_type(p, c, n) for p, c, n in zip(*params)]

def add_pointcloud(stage, points, scene_path, colors=None, time=None, points_type='point_instancer'):
    r"""Add a pointcloud to an existing USD stage.

    Create a pointcloud represented by point instances of a sphere centered at each point coordinate.
    The stage is modified but not saved to disk.

    Args:
        stage (Usd.Stage): Stage onto which to add the pointcloud.
        points (torch.FloatTensor): Pointcloud tensor containing ``N`` points of shape ``(N, 3)``.
        scene_path (str): Absolute path of pointcloud within the USD file scene. Must be a valid Sdf.Path.
        colors (torch.FloatTensor, optional): Color tensor corresponding each point in the pointcloud
            tensor of shape ``(N, 3)``. colors only works if points_type is 'usd_geom_points'.
        time (convertible to float, optional): Positive integer defining the time at which the supplied parameters
            correspond to.
        points_type (str): String that indicates whether to save pointcloud as UsdGeomPoints or PointInstancer.
            'usd_geom_points' indicates UsdGeomPoints and 'point_instancer' indicates PointInstancer.
            Please refer here for UsdGeomPoints:
            https://graphics.pixar.com/usd/docs/api/class_usd_geom_points.html and here for PointInstancer
            https://graphics.pixar.com/usd/docs/api/class_usd_geom_point_instancer.html. Default: 'point_instancer'.
    Returns:
        (Usd.Stage)

    Example:
        >>> stage = create_stage('./new_stage.usd')
        >>> points = torch.rand(100, 3)
        >>> stage = add_pointcloud(stage, points, '/World/PointClouds/pointcloud_0')
        >>> stage.Save()
    """
    scene_path = Sdf.Path(scene_path)
    if time is None:
        time = Usd.TimeCode.Default()

    if stage.GetPrimAtPath(scene_path):
        points_prim = stage.GetPrimAtPath(scene_path)
    else:
        if points_type == 'point_instancer':
            points_prim = stage.DefinePrim(scene_path, 'PointInstancer')
        elif points_type == 'usd_geom_points':
            points_prim = stage.DefinePrim(scene_path, 'Points')
        else:
            raise ValueError('Expected points_type to be "usd_geom_points" or "point_instancer", '
                             f'but got "{points_type}".')

    if points_type == 'point_instancer':
        geom_points = UsdGeom.PointInstancer(points_prim)
        sphere = UsdGeom.Sphere.Define(stage, f'{scene_path}/sphere')
        sphere.GetRadiusAttr().Set(0.5)
        geom_points.CreatePrototypesRel().SetTargets([sphere.GetPath()])
    elif points_type == 'usd_geom_points':
        geom_points = UsdGeom.Points(points_prim)

    # Calculate default point scale
    bounds = points.max(dim=0)[0] - points.min(dim=0)[0]
    min_bound = min(bounds)
    scale = (min_bound / points.size(0) ** (1 / 3)).item()

    # Generate instancer parameters
    positions = points.detach().cpu().tolist()
    scales = np.asarray([scale, ] * points.size(0))

    if points_type == 'point_instancer':
        indices = [0] * points.size(0)
        # Populate point instancer
        geom_points.GetProtoIndicesAttr().Set(indices, time=time)
        geom_points.GetPositionsAttr().Set(positions, time=time)
        scales = [(scale,) * 3] * points.size(0)
        geom_points.GetScalesAttr().Set(scales, time=time)
    elif points_type == 'usd_geom_points':
        # Populate UsdGeomPoints
        geom_points.GetPointsAttr().Set(points.numpy(), time=time)
        geom_points.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(scales), time=time)

    if colors is not None and points_type == 'usd_geom_points':
        assert colors.shape == points.shape, 'Colors and points must have the same shape.'
        geom_points.GetDisplayColorAttr().Set(colors.numpy(), time=time)

    return stage

def export_pointcloud(file_path, pointcloud, scene_path='/World/PointClouds/pointcloud_0',
                      color=None, time=None, points_type='point_instancer'):
    r"""Export a single pointcloud to a USD scene.

    Export a single pointclouds to USD. The pointcloud will be added to the USD stage and represented
    by point instances of a sphere centered at each point coordinate. The stage is then saved to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        pointcloud (torch.FloatTensor): Pointcloud tensor containing ``N`` points of shape ``(N, 3)``.
        scene_path (str, optional): Absolute path of pointcloud within the USD file scene. Must be a valid Sdf.Path.
            If no path is provided, a default path is used.
        color (torch.FloatTensor, optional): Color tensor corresponding each point in the pointcloud
            tensor of shape ``(N, 3)``. colors only works if points_type is 'usd_geom_points'.
        time (convertible to float): Positive integer defining the time at which the supplied parameters correspond to.
        points_type (str): String that indicates whether to save pointcloud as UsdGeomPoints or PointInstancer.
                'usd_geom_points' indicates UsdGeomPoints and 'point_instancer' indicates PointInstancer.
                Please refer here for UsdGeomPoints:
                https://graphics.pixar.com/usd/docs/api/class_usd_geom_points.html and here for PointInstancer
                https://graphics.pixar.com/usd/docs/api/class_usd_geom_point_instancer.html. Default: 'point_instancer'.
    Returns:
        (Usd.Stage)

    Example:
        >>> points = torch.rand(100, 3)
        >>> stage = export_pointcloud('./new_stage.usd', points)
    """
    stage = export_pointclouds(file_path, [pointcloud], [scene_path], colors=[color], times=[time],
                               points_type=points_type)
    return stage

def export_pointclouds(file_path, pointclouds, scene_paths=None, colors=None, times=None,
                       points_type='point_instancer'):
    r"""Export one or more pointclouds to a USD scene.

    Export one or more pointclouds to USD. The pointclouds will be added to the USD stage and represented
    by point instances of a sphere centered at each point coordinate. The stage is then saved to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        pointclouds (list of torch.FloatTensor): List of pointcloud tensors of length ``N`` defining N pointclouds.
        scene_paths (list of str, optional): Absolute path(s) of pointcloud(s) within the USD file scene.
            Must be a valid Sdf.Path. If no path is provided, a default path is used.
        times (list of int): Positive integers defining the time at which the supplied parameters correspond to.
        colors (list of tensors, optional): Lits of RGB colors of length ``N``, each corresponding to a pointcloud
            in the pointcloud list. colors only works if points_type is 'usd_geom_points'.
        points_type (str): String that indicates whether to save pointcloud as UsdGeomPoints or PointInstancer.
            'usd_geom_points' indicates UsdGeomPoints and 'point_instancer' indicates PointInstancer.
            Please refer here for UsdGeomPoints:
            https://graphics.pixar.com/usd/docs/api/class_usd_geom_points.html and here for PointInstancer
            https://graphics.pixar.com/usd/docs/api/class_usd_geom_point_instancer.html. Default: 'point_instancer'.
    Returns:
        (Usd.Stage)

    Example:
        >>> points = torch.rand(100, 3)
        >>> stage = export_pointcloud('./new_stage.usd', points)
    """
    if scene_paths is None:
        scene_paths = [f'/World/PointClouds/pointcloud_{i}' for i in range(len(pointclouds))]
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)
    if colors is None:
        colors = [None] * len(scene_paths)

    assert len(pointclouds) == len(scene_paths)
    stage = create_stage(file_path)
    for scene_path, points, color, time in zip(scene_paths, pointclouds, colors, times):
        add_pointcloud(stage, points, scene_path, color, time=time, points_type=points_type)
    stage.Save()

    return stage
