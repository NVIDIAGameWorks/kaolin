# Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES.
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
import os
import re
import warnings
import numpy as np
import torch

from pxr import Usd, UsdGeom, Vt, Sdf

from .utils import _get_stage_from_maybe_file, get_scene_paths, create_stage
from .transform import set_local_to_world_transform, get_local_to_world_transform

pointcloud_return_type = namedtuple('pointcloud_return_type', ['points', 'colors', 'normals', 'transform'])

__all__ = [
    'get_pointcloud_scene_paths',
    'import_pointcloud',
    'import_pointclouds',
    'get_pointcloud_bracketing_time_samples',
    'add_pointcloud',
    'export_pointcloud',
    'export_pointclouds'
]


def _apply_local_to_world(points, local_to_world):
    """Apply a local-to-world transform (translation in last column) to a (N, 3) tensor."""
    if local_to_world is None or points is None or points.numel() == 0:
        return points
    transform = local_to_world.to(points.dtype).to(points.device)
    points_homo = torch.nn.functional.pad(points, (0, 1), mode='constant', value=1.0)
    return (transform @ points_homo.T).T[:, :3].contiguous()

def get_pointcloud_scene_paths(file_path_or_stage, scene_path=None):
    r"""Returns all point cloud scene paths contained in specified file. Assumes that point
    clouds are exported using this API.

    Args:
        file_path_or_stage (str or Usd.Stage):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        scene_path (str, optional): If specified, only return paths under this scene path prefix.

    Returns:
        (list of str): List of filtered scene paths.
    """
    # TODO(mshugrina): is passing prim_types='PointInstancer' the same as UsdGeom.PointInstancer(p) ?
    scene_path_regex = f"{re.escape(scene_path)}(/|$)" if scene_path is not None else None
    stage = _get_stage_from_maybe_file(file_path_or_stage)
    try:
        geom_points_paths = get_scene_paths(stage, prim_types=['Points'], scene_path_regex=scene_path_regex)
        point_instancer_paths = get_scene_paths(stage, prim_types=['PointInstancer'],
                                                scene_path_regex=scene_path_regex)
    finally:
        del stage, file_path_or_stage
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

def import_pointcloud(file_path_or_stage, scene_path=None, time=None):
    r"""Import a pointcloud scene from a USD file or Stage,
    all the prims under `scene_path` are merged into a single world-space pointcloud.

    Each prim's ``local_to_world`` transform (if not identity) is applied to the points
    before concatenation. Colors and normals are concatenated only when every contributing
    prim provides them; otherwise the merged field is ``None``.

    Args:
        file_path_or_stage (str or Usd.Stage):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        scene_path (str, optional): If specified, only import pointclouds under this scene path prefix.
            Default: import all pointclouds in the file.
        time (convertible to float, optional): Positive integer indicating the time at which to retrieve parameters.

    Returns:
        (namedtuple or None): namedtuple with fields:

            - **points** (torch.FloatTensor): of shape (num_points, 3)
            - **colors** (torch.FloatTensor): of shape (num_points, 3) or ``None``
            - **normals** (torch.FloatTensor): of shape (num_points, 3) (not yet implemented)
            - **transform** (torch.FloatTensor or ``None``): always ``None``, since the per-prim
              ``local_to_world`` transforms have already been applied to the merged ``points``.

        Returns ``None`` if no pointclouds are found.

    Example:
        >>> points = torch.rand(100, 3)
        >>> export_pointcloud('./new_stage.usd', scene_path='/World/pointcloud', points=points)
        >>> merged = import_pointcloud('./new_stage.usd')
        >>> merged.points.shape
        torch.Size([100, 3])
    """
    scene_paths = get_pointcloud_scene_paths(file_path_or_stage, scene_path=scene_path)
    if not scene_paths:
        return None
    cloud_list = import_pointclouds(file_path_or_stage, scene_paths, times=time, return_list=True)

    points = torch.cat([_apply_local_to_world(c.points, c.transform) for c in cloud_list], dim=0)
    if all(c.colors is not None for c in cloud_list):
        colors = torch.cat([c.colors for c in cloud_list], dim=0)
    else:
        colors = None
    if all(c.normals is not None for c in cloud_list):
        normals = torch.cat([c.normals for c in cloud_list], dim=0)
    else:
        normals = None
    return pointcloud_return_type(points, colors, normals, None)


def import_pointclouds(file_path_or_stage, scene_paths=None, times=None, return_list=True):
    r"""Import one or more pointclouds from a USD file or stage in their local space.

    Assumes that pointclouds are interpreted using point instancers or UsdGeomPoints. Each prim's
    points are returned in the prim's local space, and its ``local_to_world`` transform is returned
    in the ``transform`` field (or ``None`` if the prim has an identity transform) rather than being
    applied to the points.

    Args:
        file_path_or_stage (str or Usd.Stage):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        scene_paths (list of str, optional): Scene paths within the USD file indicating which primitive(s) to import.
            Default: import every pointcloud found in the file.
        times (list of Number or Number, optional): Positive integers indicating the time at which to retrieve
            parameters. A single value is broadcast across all ``scene_paths``.
        return_list (bool, optional): If ``True`` (default), return a ``list`` of ``pointcloud_return_type``
            namedtuples in the order of ``scene_paths``. If ``False``, return a ``dict`` keyed by scene path.

    Returns:
        (list or dict of namedtuple): Either a list of ``pointcloud_return_type`` namedtuples
        (when ``return_list=True``, the default), or a ``dict[str, pointcloud_return_type]``
        keyed by scene path (when ``return_list=False``). Each namedtuple has fields:

            - **points** (torch.FloatTensor): of shape (num_points, 3), in the prim's local space
            - **colors** (torch.FloatTensor): of shape (num_points, 3) or ``None``
            - **normals** (torch.FloatTensor): of shape (num_points, 3) (not yet implemented)
            - **transform** (torch.FloatTensor or ``None``): local-to-world transform of shape (4, 4),
              or ``None`` if the prim has an identity transform

    Example:
        >>> points = torch.rand(100, 3)
        >>> export_pointclouds('./new_stage.usd', points=[points, points, points])
        >>> pointclouds = import_pointclouds('./new_stage.usd', return_list=False)
        >>> len(pointclouds)
        3
        >>> next(iter(pointclouds.values())).points.shape
        torch.Size([100, 3])
    """
    stage = _get_stage_from_maybe_file(file_path_or_stage)
    try:
        scene_paths = get_pointcloud_scene_paths(stage) if scene_paths is None else scene_paths
        if times is None:
            times = [Usd.TimeCode.Default()] * len(scene_paths)
        elif not isinstance(times, list):
            times = [times] * len(scene_paths)
        elif len(times) != len(scene_paths):
            raise ValueError(
                f"Length of times ({len(times)}) must match length of scene_paths ({len(scene_paths)})"
            )

        output = {}
        for scene_path, time in zip(scene_paths, times):
            prim = stage.GetPrimAtPath(scene_path)
            if not prim:
                raise ValueError(f'The prim at {scene_path} does not exist.')

            if UsdGeom.Points(prim):
                geom_points = UsdGeom.Points(prim)
                points = torch.tensor(np.asarray(geom_points.GetPointsAttr().Get(time=time)))
                color = geom_points.GetDisplayColorAttr().Get(time=time)
                colors = torch.tensor(np.asarray(color)) if color is not None else None
            elif UsdGeom.PointInstancer(prim):
                instancer = UsdGeom.PointInstancer(prim)
                points = torch.tensor(np.asarray(instancer.GetPositionsAttr().Get(time=time)))
                colors = None
            else:
                raise TypeError('The prim is neither UsdGeomPoints nor UsdGeomPointInstancer.')

            transform = get_local_to_world_transform(stage, prim, time)
            transform = transform.float() if transform is not None else None

            # TODO: normals are not yet implemented
            output[str(scene_path)] = pointcloud_return_type(points, colors, None, transform)
    finally:
        del stage, file_path_or_stage
    if return_list:
        return list(output.values())
    return output


def add_pointcloud(stage, scene_path, points=None, colors=None, local_to_world=None, time=None,
                   overwrite=False, points_type='point_instancer'):
    r"""Add a pointcloud to an existing USD stage.

    Create a pointcloud represented by point instances of a sphere centered at each point coordinate.
    The stage is modified but not saved to disk.

    Args:
        stage (Usd.Stage): Stage onto which to add the pointcloud.
        scene_path (str): Absolute path of pointcloud within the USD file scene. Must be a valid Sdf.Path.
        points (torch.FloatTensor): Pointcloud tensor containing ``N`` points of shape ``(N, 3)``.
        colors (torch.FloatTensor, optional): Color tensor corresponding each point in the pointcloud
            tensor of shape ``(N, 3)``. colors only works if points_type is 'usd_geom_points'.
        time (convertible to float, optional): Positive integer defining the time at which the supplied parameters
            correspond to.
        local_to_world (torch.Tensor, optional): Local-to-world transform matrix of shape :math:`(4, 4)`.
            If provided, it is written onto the prim via :func:`set_local_to_world_transform`.
        overwrite (bool): If True, allow replacing an existing prim at ``scene_path`` whose type does not
            match ``points_type``. If False (default), raise ``ValueError`` in that case.
        points_type (str): String that indicates whether to save pointcloud as UsdGeomPoints or PointInstancer.
            'usd_geom_points' indicates UsdGeomPoints and 'point_instancer' indicates PointInstancer.
            Please refer here for UsdGeomPoints:
            https://graphics.pixar.com/usd/docs/api/class_usd_geom_points.html and here for PointInstancer
            https://graphics.pixar.com/usd/docs/api/class_usd_geom_point_instancer.html. Default: 'point_instancer'.

    Returns:
        (Usd.Prim): The generated pointcloud Prim.

    Example:
        >>> stage = create_stage('./new_stage.usd')
        >>> points = torch.rand(100, 3)
        >>> prim = add_pointcloud(stage, '/World/PointClouds/pointcloud_0', points)
        >>> stage.Save()

    .. note::

        **Deprecated since v0.19**: the positional order changed from
        ``add_pointcloud(stage, points, scene_path, ...)`` to
        ``add_pointcloud(stage, scene_path, points, ...)``. v0.18-style calls
        (tensor as the 2nd positional argument) still work but emit a
        ``DeprecationWarning``.
    """
    if torch.is_tensor(scene_path):
        warnings.warn(
            "add_pointcloud positional argument order changed in v0.19: "
            "the new order is (stage, scene_path, points, ...). "
            "Calling with (stage, points, scene_path, ...) is deprecated.",
            DeprecationWarning, stacklevel=2,
        )
        scene_path, points = points, scene_path
    if points is None:
        raise ValueError("points must be provided")
    pointcloud = points
    scene_path = Sdf.Path(scene_path)
    if time is None:
        time = Usd.TimeCode.Default()

    expected_type = 'PointInstancer' if points_type == 'point_instancer' else (
        'Points' if points_type == 'usd_geom_points' else None)
    if expected_type is None:
        raise ValueError('Expected points_type to be "usd_geom_points" or "point_instancer", '
                         f'but got "{points_type}".')

    existing = stage.GetPrimAtPath(scene_path)
    if existing:
        if existing.GetTypeName() != expected_type and not overwrite:
            raise ValueError(
                f"Prim already exists at {scene_path} with type "
                f"{existing.GetTypeName()}; use overwrite=True to replace."
            )
        if existing.GetTypeName() != expected_type:
            stage.RemovePrim(scene_path)
            points_prim = stage.DefinePrim(scene_path, expected_type)
        else:
            points_prim = existing
    else:
        points_prim = stage.DefinePrim(scene_path, expected_type)

    if points_type == 'point_instancer':
        geom_points = UsdGeom.PointInstancer(points_prim)
        sphere = UsdGeom.Sphere.Define(stage, f'{scene_path}/sphere')
        sphere.GetRadiusAttr().Set(0.5)
        geom_points.CreatePrototypesRel().SetTargets([sphere.GetPath()])
    elif points_type == 'usd_geom_points':
        geom_points = UsdGeom.Points(points_prim)

    # Calculate default point scale
    bounds = pointcloud.max(dim=0)[0] - pointcloud.min(dim=0)[0]
    min_bound = min(bounds)
    scale = (min_bound / pointcloud.size(0) ** (1 / 3)).item()

    # Generate instancer parameters
    positions = pointcloud.detach().cpu().tolist()
    scales = np.asarray([scale, ] * pointcloud.size(0))

    if points_type == 'point_instancer':
        indices = [0] * pointcloud.size(0)
        # Populate point instancer
        geom_points.GetProtoIndicesAttr().Set(indices, time=time)
        geom_points.GetPositionsAttr().Set(positions, time=time)
        scales = [(scale,) * 3] * pointcloud.size(0)
        geom_points.GetScalesAttr().Set(scales, time=time)
    elif points_type == 'usd_geom_points':
        # Populate UsdGeomPoints
        geom_points.GetPointsAttr().Set(pointcloud.numpy(), time=time)
        geom_points.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(scales), time=time)

    if colors is not None and points_type == 'usd_geom_points':
        if colors.shape != pointcloud.shape:
            raise ValueError(
                f'Colors shape {tuple(colors.shape)} does not match '
                f'pointcloud shape {tuple(pointcloud.shape)}.')
        geom_points.GetDisplayColorAttr().Set(colors.numpy(), time=time)

    if local_to_world is not None:
        set_local_to_world_transform(stage, points_prim, local_to_world, time)

    return points_prim

def export_pointcloud(file_path, scene_path='/World/PointClouds/pointcloud_0',
                      points=None, colors=None, local_to_world=None, up_axis='Y', time=None,
                      points_type='point_instancer', overwrite=False,
                      pointcloud=None, color=None):
    r"""Export a single pointcloud to a USD scene.

    Export a single pointclouds to USD. The pointcloud will be added to the USD stage and represented
    by point instances of a sphere centered at each point coordinate. The stage is then saved to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_path (str, optional): Absolute path of pointcloud within the USD file scene. Must be a valid Sdf.Path.
            If no path is provided, a default path is used.
        points (torch.FloatTensor): Pointcloud tensor containing ``N`` points of shape ``(N, 3)``.
        colors (torch.FloatTensor, optional): Color tensor corresponding each point in the pointcloud
            tensor of shape ``(N, 3)``. colors only works if points_type is 'usd_geom_points'.
        time (convertible to float): Positive integer defining the time at which the supplied parameters correspond to.
        local_to_world (torch.Tensor, optional): Local-to-world transform matrix of shape :math:`(4, 4)`.
        up_axis (str, optional): Axis to use for up direction. Must be one of 'X', 'Y', 'Z'. Defaults to 'Y'.
        points_type (str): String that indicates whether to save pointcloud as UsdGeomPoints or PointInstancer.
                'usd_geom_points' indicates UsdGeomPoints and 'point_instancer' indicates PointInstancer.
                Please refer here for UsdGeomPoints:
                https://graphics.pixar.com/usd/docs/api/class_usd_geom_points.html and here for PointInstancer
                https://graphics.pixar.com/usd/docs/api/class_usd_geom_point_instancer.html. Default: 'point_instancer'.
        overwrite (bool): If True, overwrite existing .usda. If False (default) raise an error if files already exists.

    Example:
        >>> points = torch.rand(100, 3)
        >>> export_pointcloud('./new_stage.usd', points=points)

    .. note::

        **Deprecated since v0.19**: the positional order changed and two kwargs
        were renamed. v0.18-style calls continue to work but emit a
        ``DeprecationWarning``:

        * 2nd positional was ``pointcloud`` (tensor); it is now ``scene_path`` (str).
        * ``pointcloud=`` keyword has been renamed to ``points=``.
        * ``color=`` keyword has been renamed to ``colors=``.
    """
    if torch.is_tensor(scene_path):
        warnings.warn(
            "export_pointcloud's 2nd positional argument is now scene_path (str); "
            "passing a tensor positionally is deprecated since v0.19. "
            "Use export_pointcloud(file_path, scene_path=..., points=...) instead.",
            DeprecationWarning, stacklevel=2,
        )
        if isinstance(points, str):
            scene_path, points = points, scene_path
        else:
            if points is None:
                points = scene_path
            scene_path = '/World/PointClouds/pointcloud_0'
    if pointcloud is not None:
        warnings.warn(
            "export_pointcloud(pointcloud=...) is deprecated since v0.19; "
            "use points=... instead.",
            DeprecationWarning, stacklevel=2,
        )
        if points is None:
            points = pointcloud
    if color is not None:
        warnings.warn(
            "export_pointcloud(color=...) is deprecated since v0.19; "
            "use colors=... instead.",
            DeprecationWarning, stacklevel=2,
        )
        if colors is None:
            colors = color
    if points is None:
        raise ValueError("points must be provided")
    export_pointclouds(file_path, scene_paths=[scene_path], points=[points],
                       colors=[colors], times=[time], local_to_world=local_to_world,
                       up_axis=up_axis, points_type=points_type, overwrite=overwrite)

def export_pointclouds(file_path, scene_paths=None, points=None, colors=None,
                       local_to_world=None, up_axis='Y', times=None,
                       points_type='point_instancer', overwrite=False,
                       pointclouds=None):
    r"""Export one or more pointclouds to a USD scene.

    Export one or more pointclouds to USD. The pointclouds will be added to the USD stage and represented
    by point instances of a sphere centered at each point coordinate. The stage is then saved to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_paths (list of str, optional): Absolute path(s) of pointcloud(s) within the USD file scene.
            Must be a valid Sdf.Path. If no path is provided, a default path is used.
        points (list of torch.FloatTensor): List of pointcloud tensors of length ``N`` defining N pointclouds.
        colors (list of tensors, optional): List of RGB colors of length ``N``, each corresponding to a pointcloud
            in the points list. colors only works if points_type is 'usd_geom_points'.
        times (list of int): Positive integers defining the time at which the supplied parameters correspond to.
        local_to_world (torch.FloatTensor, optional): local-to-world transforms as a single ``(4, 4)`` tensor
            (broadcast to every pointcloud) or a batched ``(N, 4, 4)`` tensor (one transform per pointcloud).
        up_axis (str, optional): Axis to use for up direction. Must be one of 'X', 'Y', 'Z'. Defaults to 'Y'.
        points_type (str): String that indicates whether to save pointcloud as UsdGeomPoints or PointInstancer.
            'usd_geom_points' indicates UsdGeomPoints and 'point_instancer' indicates PointInstancer.
            Please refer here for UsdGeomPoints:
            https://graphics.pixar.com/usd/docs/api/class_usd_geom_points.html and here for PointInstancer
            https://graphics.pixar.com/usd/docs/api/class_usd_geom_point_instancer.html. Default: 'point_instancer'.
        overwrite (bool): If True, overwrite existing .usda. If False (default) raise an error if files already exists.

    Example:
        >>> points = torch.rand(100, 3)
        >>> export_pointclouds('./new_stage.usd', points=[points])

    .. note::

        **Deprecated since v0.19**: the positional order changed and the 2nd
        positional argument was renamed. v0.18-style calls continue to work but
        emit a ``DeprecationWarning``:

        * 2nd positional was ``pointclouds`` (list of tensors); it is now
          ``scene_paths`` (list of str).
        * ``pointclouds=`` keyword has been renamed to ``points=``.
    """
    if scene_paths is not None and len(scene_paths) > 0 and torch.is_tensor(scene_paths[0]):
        warnings.warn(
            "export_pointclouds's 2nd positional argument is now scene_paths (list of str); "
            "passing a list of tensors positionally is deprecated since v0.19. "
            "Use export_pointclouds(file_path, scene_paths=..., points=...) instead.",
            DeprecationWarning, stacklevel=2,
        )
        if points is None:
            points = scene_paths
        scene_paths = None
    if pointclouds is not None:
        warnings.warn(
            "export_pointclouds(pointclouds=...) is deprecated since v0.19; "
            "use points=... instead.",
            DeprecationWarning, stacklevel=2,
        )
        if points is None:
            points = pointclouds
    if points is None:
        raise ValueError("points must be provided")
    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f"{file_path} already exists; to overwrite whole file use 'overwrite' argument;" +
                              " to add pointcloud to existing usd, use 'add_pointcloud' instead.")

    if local_to_world is not None:
        if not torch.is_tensor(local_to_world):
            raise TypeError(f'Unexpected type {type(local_to_world)} for local_to_world (torch.Tensor expected)')
        if local_to_world.ndim == 3:
            if local_to_world.shape != (len(points), 4, 4):
                raise ValueError(
                    f'local_to_world batched tensor must have shape ({len(points)}, 4, 4), '
                    f'got {tuple(local_to_world.shape)}')
        elif local_to_world.ndim == 2:
            if tuple(local_to_world.shape) != (4, 4):
                raise ValueError(
                    f'local_to_world tensor must have shape (4, 4) or (N, 4, 4), '
                    f'got {tuple(local_to_world.shape)}')
        else:
            raise ValueError(
                f'local_to_world tensor must have shape (4, 4) or (N, 4, 4), '
                f'got {tuple(local_to_world.shape)}')

    if scene_paths is None:
        scene_paths = [f'/World/PointClouds/pointcloud_{i}' for i in range(len(points))]
    if times is None:
        times = [Usd.TimeCode.Default()] * len(scene_paths)
    if colors is None:
        colors = [None] * len(scene_paths)

    if len(points) != len(scene_paths):
        raise ValueError(
            f'Number of pointclouds ({len(points)}) must match number of '
            f'scene_paths ({len(scene_paths)}).')
    stage = create_stage(file_path, up_axis)
    for i, (scene_path, pc, color, time) in enumerate(zip(scene_paths, points, colors, times)):
        if local_to_world is None:
            ltw_i = None
        elif local_to_world.ndim == 3:
            ltw_i = local_to_world[i]
        else:
            ltw_i = local_to_world
        add_pointcloud(stage, scene_path=scene_path, points=pc, colors=color,
                       local_to_world=ltw_i, time=time, points_type=points_type)
    stage.Save()
