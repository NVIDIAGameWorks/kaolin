# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import re
import math
import torch
import logging
import numpy as np
from pathlib import Path
import os
from typing import Dict, Iterable, List, Tuple, Union

from pxr import Usd, UsdGeom, UsdLux, Sdf, Gf, Vt

from kaolin.rep.Mesh import Mesh
from kaolin.rep import TriangleMesh
from kaolin.rep import QuadMesh
from kaolin.rep import PointCloud
from kaolin.rep import VoxelGrid


def is_mesh(obj):
    if isinstance(obj, dict) and {'vertices', 'faces'} <= set(obj):
        return len(obj['faces'][0]) in [3, 4]
    return isinstance(obj, TriangleMesh)


class VisUsd:
    r"""Class to visualize using USD (Universal Scene Description).
    """

    STAGE_SIZE = 100

    def __init__(self):
        self.stage = None

    def set_stage(self, filepath: str = './visualizations/visualize.usda', up_axis='Z'):
        r"""Setup stage with basic structure and primitives.

        Args:
            filepath (str): Path to save visualizations to. Must end in either ".usd" or ".usda".
            up_axis (str): Specify up-axis, choose from ["Y", "Z"].
        """
        filepath = Path(filepath)
        if not filepath.parent.exists():
            filepath.parent.mkdir()
        self.stage = Usd.Stage.CreateNew(str(filepath))

        self.up_axis = up_axis.upper()
        self.up_axis_index = {'Y': 1, 'Z': 2}[self.up_axis]
        assert (self.up_axis in ['Y', 'Z'])
        UsdGeom.SetStageUpAxis(self.stage, self.up_axis)

        self._setup_primitives()

        self.save()

    def _setup_primitives(self):
        r""" Setup basic primitives useful for rendering meshes, point clouds and voxels. """
        UsdGeom.Xform.Define(self.stage, '/Root')
        root = self.stage.GetPrimAtPath('/Root')
        self.stage.SetDefaultPrim(root)
        UsdGeom.Xform.Define(self.stage, '/Root/Visualizer')
        instancer = UsdGeom.PointInstancer.Define(self.stage, '/PointInstancer')

        prim_paths = []

        sphere_proto = UsdGeom.Sphere.Define(self.stage, instancer.GetPath().AppendChild('Sphere'))
        cube_proto = UsdGeom.Cube.Define(self.stage, instancer.GetPath().AppendChild('Cube'))
        prim_paths = [sphere_proto.GetPath(), cube_proto.GetPath()]

        instancer.CreatePrototypesRel().SetTargets(prim_paths)
        self.instancer = instancer
        self.save()

    def visualize(self, object_3d: Union[Dict[str, torch.Tensor], Mesh, PointCloud, VoxelGrid],
                  object_path: str = '/Root/Visualizer/object',
                  fit_to_stage: bool = True, meet_ground: bool = True, center_on_stage: bool = True,
                  translation: Tuple[float, float, float] = (0., 0., 0.)):
        r""" Create USD file with object_3d representation.

            Args:
                object_3d (dict or Mesh or PointCloud or VoxelGrid): The object to visualize.
                object_path (str): The object's path in the USD. This argument is only applicable to meshes.
                fit_to_stage (bool): Whether to resize the objec to fit within a 100x100x100 stage.
                meet_ground (bool): Whether to translate the object to be resting on a ground place at 0.
                translation (Tuple[float, float, float]): Translation of object. Applied after meet_ground.
        """

        assert self.stage is not None

        params = {
            'object_path': object_path,
            'translation': translation,
            'fit_to_stage': fit_to_stage,
            'meet_ground': meet_ground,
            'center_on_stage': center_on_stage,
        }

        if is_mesh(object_3d):
            self._visualize_mesh(object_3d, **params)
        elif isinstance(object_3d, PointCloud):
            self._visualize_points(object_3d, **params)
        elif isinstance(object_3d, VoxelGrid):
            self._visualize_voxels(object_3d, **params)
        else:
            raise ValueError(f'Object of type {type(object_3d)} is not supported.')

    def _visualize_voxels(self, voxels: VoxelGrid, translation: Tuple[float, float, float] = (0., 0., 0.),
                          **kwargs):
        r""" Visualize voxels in USD.

        Args:
            points (torch.Tensor): Array of points of size (num_points, 3)

        """

        points = torch.nonzero(voxels.voxels.detach()).float()
        points, scale, _ = self._fit_to_stage(points, **kwargs)
        points, _ = self._set_points_bottom(points, scale)

        points[:, 0] += translation[0]
        points[:, 1] += translation[1]
        points[:, 2] += translation[2]

        indices = [1] * points.shape[0]
        points = points.cpu().numpy().astype(float)
        positions = points.tolist()
        orientations = [Gf.Quath(1.0, 0.0, 0.0, 0.0)] * points.shape[0]
        scales = [Gf.Vec3f(float(scale))] * points.shape[0]

        self.instancer.GetProtoIndicesAttr().Set(indices)
        self.instancer.GetPositionsAttr().Set(positions)
        self.instancer.GetOrientationsAttr().Set(orientations)
        self.instancer.GetScalesAttr().Set(scales)

        self.save()

    def _visualize_points(self, pointcloud: PointCloud, translation: Tuple[float, float, float] = (0., 0., 0.),
                          **kwargs):
        r""" Visualize points in USD.
        """

        points, _, _ = self._fit_to_stage(pointcloud.points.detach())

        points[:, 0] += translation[0]
        points[:, 1] += translation[1]
        points[:, 2] += translation[2]

        indices = [0] * points.shape[0]
        points = points.cpu().numpy().astype(float)
        positions = points.tolist()
        orientations = [Gf.Quath(1.0, 0.0, 0.0, 0.0)] * points.shape[0]
        scales = [Gf.Vec3f(1)] * points.shape[0]

        self.instancer.GetProtoIndicesAttr().Set(indices)
        self.instancer.GetPositionsAttr().Set(positions)
        self.instancer.GetOrientationsAttr().Set(orientations)
        self.instancer.GetScalesAttr().Set(scales)

        self.save()

    def _visualize_mesh(self, mesh: Union[Dict[str, torch.Tensor], Mesh],
                        object_path: str,
                        translation: Tuple[float, float, float] = (0., 0., 0.), **kwargs):
        r""" Visualize mesh in USD.
        """

        if isinstance(mesh, Mesh):
            vertices, faces = mesh.vertices.detach(), mesh.faces.detach()
        else:
            vertices, faces = mesh['vertices'], mesh['faces']

        usd_mesh = UsdGeom.Mesh.Define(self.stage, object_path)

        num_faces = faces.size(0)
        is_tri = (faces.size(1) == 3)
        face_vertex_counts = [faces.size(1)] * num_faces

        vertices, _, _ = self._fit_to_stage(vertices, **kwargs)

        vertices = vertices.detach().cpu().numpy().astype(float)
        points = vertices.tolist()
        faces = faces.detach().cpu().view(-1).numpy().astype(int)

        usd_mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
        usd_mesh.GetPointsAttr().Set(Vt.Vec3fArray(points))
        usd_mesh.GetFaceVertexIndicesAttr().Set(faces)
        if is_tri:
            usd_mesh.GetPrim().GetAttribute('subdivisionScheme').Set('none')
        UsdGeom.XformCommonAPI(usd_mesh.GetPrim()).SetTranslate(translation)

        self.save()

    def _fit_to_stage(self, points: torch.Tensor, center_on_stage: bool = True,
                      meet_ground: bool = True, fit_to_stage: bool = True, **kwargs):
        r""" Scale and translate (in the Y axis) to fit to
        the specified stage size and keep the object above the floor.

        Args:
            points (torch.Tensor): Tensor of points of size (N, 3).

        Returns:
            Tuple[torch.Tensor, float, torch.Tensor]: Tuple containing scaled and
                translated tensor, scale value, and translation tensor.
        """
        scale = 1.
        translation = [0., 0., 0.]
        if center_on_stage:
            # center at 0, 0, 0
            points, translation = self._set_points_center(points)

        if fit_to_stage:
            # scale points to fit within STAGE_SIZE
            points, scale = self._fit_points(points, self.STAGE_SIZE)

        if meet_ground:
            # set bottom as 0.0 on up_axis
            points, up_translation = self._set_points_bottom(points, **kwargs)
            translation[self.up_axis_index] += up_translation
        return points, scale, translation

    def _fit_points(self, points, fit_size):
        r"""Scale points to fit in cube of size fit_size. """
        longest_side = max(torch.abs(torch.max(points, 0)[0] - torch.min(points, 0)[0]))
        scale = fit_size / longest_side
        points *= scale
        return points, scale

    def _set_points_bottom(self, points: np.ndarray, bottom: float = 0.0, **kwargs):
        r"""Move points to be above bottom value on axis"""
        y_translation = torch.min(points[:, self.up_axis_index])
        points[:, self.up_axis_index] -= y_translation - bottom
        return points, y_translation

    def _set_points_center(self, points, center_point: List[float] = [0., 0., 0.]):
        r"""Set center of points to match center_point."""
        center_point = torch.tensor(center_point, device=points.device)
        extents = torch.max(points, 0)[0] - torch.min(points, 0)[0]
        curr_center = torch.max(points, 0)[0] - extents / 2.0
        translation = center_point - curr_center
        points += translation
        return points, translation

    def save(self):
        r""" Save stage. """
        self.stage.Save()
