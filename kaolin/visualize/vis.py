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

from typing import Union

import torch
import trimesh
import numpy as np
import kaolin
import kaolin as kal
import pptk


def show(inp: Union[kaolin.rep.Mesh, kaolin.rep.PointCloud,
         kaolin.rep.VoxelGrid], options: dict = {}, mode='points'):
    r"""Visualizer class, for the representations defined in kaolin.rep.

    Args:
        inp (kaolin.rep.Mesh or kaolin.rep.PointCloud or kaolin.rep.VoxelGrid):
            A kaolin.rep object to visualize
        options (dict): Visualization options

    """
    if isinstance(inp, kaolin.rep.Mesh):
        colors = [.7, .2, .2]
        if 'colors' in options:
            colors = options['colors']
        show_mesh(inp, colors)
    elif isinstance(inp, kaolin.rep.PointCloud):
        colors = [.7, .2, .2]
        if 'colors' in options:
            colors = options['colors']
        show_pointcloud(inp, colors)
    elif isinstance(inp, kaolin.rep.VoxelGrid):
        thresh = 0.5
        mode = 'exact'
        colors = [.7, .2, .2]
        if 'thres' in options:
            thresh = options['thresh']
        if mode in options:
            mode = options['mode']
        if 'colors' in options:
            colors = options['colors']
        show_voxelgrid(inp, thresh, mode, colors)
    elif mode == 'voxels':
        thresh = 0.5
        mode = 'exact'
        colors = [.7, .2, .2]
        if 'thres' in options:
            thresh = options['thresh']
        if mode in options:
            mode = options['mode']
        if 'colors' in options:
            colors = options['colors']
        show_voxelgrid(inp, thresh, mode, colors)
    elif mode == 'points':
        colors = [.7, .2, .2]
        if 'colors' in options:
            colors = options['colors']
        show_pointcloud(inp, colors)


def show_mesh(input_mesh: kaolin.rep.Mesh, colors: list = [.7, .2, .2]):
    r""" Visualizer for meshes

    Args:
            verts (torch.Tensor): vertices of mesh to be visualized
            faces (torch.Tensor): faces of mesh to be visualized
            colors (list): rbg colour values for rendered mesh
    """

    mesh = trimesh.Trimesh(vertices=input_mesh.vertices.data.cpu().numpy(),
                           faces=input_mesh.faces.data.cpu().numpy())
    mesh.visual.vertex_colors = colors
    mesh.show()


def show_sdf(sdf: kaolin.rep.SDF, mode='mesh', bbox_center: float = 0.,
             bbox_dim: float = 1., num_points: int = 100000,
             colors=[.7, .2, .2]):
    r""" Visualizer for voxel array

    Args:
        sdf (kaolin.rep.SDF): sdf class object.
        mode (str): visualization mode, can render as a mesh, a pointcloud,
                or a colourful sdf pointcloud.
        colors (list): RGB colour values for rendered array.
    """
    assert mode in ['mesh', 'pointcloud', 'sdf']

    if mode == 'mesh':
        verts, faces = kal.conversion.SDF.to_mesh(sdf, bbox_center, bbox_dim)
        mesh = trimesh.Trimesh(vertices=verts.data.cpu().numpy(),
                               faces=faces.data.cpu().numpy())
        mesh.visual.vertex_colors = colors
        mesh.show()

    elif mode == 'pointcloud':
        points = torch.rand(num_points, 3)
        points = bbox_dim * (points + (bbox_center - .5))
        distances = sdf(points)
        points = points[distances <= 0]
        kal.visualize.show_point(points)

    elif mode == 'sdf':
        points = torch.rand(num_points, 3)
        points = bbox_dim * (points + (bbox_center - .5))
        distances = sdf(points)
        v = pptk.viewer(points.data.cpu().numpy())
        v.attributes(distances.data.cpu().numpy())
        input()
        v.close()


def show_pointcloud(points, colors=[.7, .2, .2]):
    r"""Visualizer for point clouds.

    Args:
        points (torch.Tensor): point cloud to be visualized
        colors (list): rbg colour values for rendered array

    """
    point_colours = np.zeros(points.shape)
    point_colours[:] = colors

    v = pptk.viewer(points.data.cpu().numpy())
    v.attributes(point_colours)
    input()
    v.close()


def show_voxelgrid(voxel, thresh=.5, mode='exact', colors=[.7, .2, .2]):
    r""" Visualizer for voxel array

    Args:
            voxel (torch.Tensor): voxel array to be visualized
            threshold (float): threshold for turning on voxel
            mode (str): mode for visualizing, either the exact model, or converted to mesh using marching cubes
            colors (list): rbg colour values for rendered array
    """
    assert (mode in ['exact', 'marching_cubes'])
    voxel = kal.conversions.voxelgridconversions.confirm_def(voxel)
    voxel = kal.conversions.voxelgridconversions.threshold(voxel, thresh=thresh)

    verts, faces = kal.conversions.voxelgrid_to_trianglemesh(
        voxel.cpu(), thresh=.5, mode=mode)
    mesh = trimesh.Trimesh(vertices=verts,
                           faces=faces)
    mesh.visual.vertex_colors = colors
    mesh.show()
