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
#
#
# Occupancy Networks
#
# Copyright 2019 Lars Mescheder, Michael Oechsle, Michael Niemeyer, Andreas Geiger, Sebastian Nowozin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import numpy as np

from kaolin.mise import MISE
import kaolin


def sdf_to_voxelgrid(sdf: kaolin.rep.SDF, bbox_center: float = 0.,
                     bbox_dim: float = 1., resolution: int = 32,
                     upsampling_steps: int = 2, threshold: float = 0.0):
    r"""Converts an SDF to a voxel grid.

    Args:
        sdf (kaolin.rep.SDF) : an object with a .eval_occ function that
            returns the signed distance from the surface of a 3D shape.
        bbox_center (float): center of the surface's bounding box.
        bbox_dim (float): largest dimension of the surface's bounding box.
        resolution (int) : the initial resolution of the voxel, should be
            large enough to properly define the surface.
        upsampling_steps (int) : Number of times the initial resolution will
            be doubled.
            The returned resolution will be resolution * (2 ^ upsampling_steps) + 1

    Returns:
        (torch.Tensor): a voxel grid

    Example:
        >>> sdf = kaolin.rep.SDF.sphere()
        >>> voxel = kaolin.conversions.sdf_to_voxelgrid(sdf, bbox_dim = 2)
    """

    mesh_extractor = MISE(
        resolution, upsampling_steps, threshold)

    points = mesh_extractor.query()
    while points.shape[0] != 0:
        # Query points
        pointsf = torch.tensor(points, dtype=torch.float, device='cuda')
        # Normalize to bounding box
        pointsf = pointsf / (mesh_extractor.resolution)
        pointsf = bbox_dim * pointsf + (bbox_center - bbox_dim / 2)
        values = sdf(pointsf)
        values = values.data.cpu().numpy().astype(np.float64)
        mesh_extractor.update(points, values)
        points = mesh_extractor.query()

    voxels = torch.FloatTensor(mesh_extractor.to_dense())

    return voxels


def sdf_to_trianglemesh(sdf: kaolin.rep.SDF, bbox_center: float = 0.,
                        bbox_dim: float = 1., resolution: int = 32,
                        upsampling_steps: int = 2, threshold: float = 0.0):
    r""" Converts an SDF function to a mesh

    Args:
        sdf (kaolin.rep.SDF): an object with a .eval_occ function that
            returns the signed distance from the surface of a 3D shape.
        bbox_center (float): center of the surface's bounding box.
        bbox_dim (float): largest dimension of the surface's bounding box.
        resolution (int) : the initial resolution of the voxel, should be large
            enough to properly define the surface.
        upsampling_steps (int) : Number of times the initial resolution will be
            doubled.
            The returned resolution will be resolution * (2 ^ upsampling_steps)

    Returns:
        (torch.Tensor): computed mesh preperties

    Example:
        >>> sdf = kaolin.rep.SDF.sphere()
        >>> verts, faces = kaolin.conversion.sdf_to_trianglemesh(sdf, bbox_dim=2)
        >>> mesh = kaolin.rep.TriangleMesh.from_tensors(verts, faces)

    """
    voxel = sdf_to_voxelgrid(sdf, bbox_center, bbox_dim,
                             resolution, upsampling_steps, threshold)

    # reverse sign of voxel values to use signed distance with marching cube function
    voxel = -voxel
    verts, faces = kaolin.conversions.voxelgrid_to_trianglemesh(voxel, thresh=threshold)
    return verts, faces


def sdf_to_pointcloud(sdf: kaolin.rep.SDF, bbox_center: float = 0.,
                      bbox_dim: float = 1., resolution: int = 32,
                      upsampling_steps: int = 2, num_points: int = 5000):
    r"""Converts an SDF fucntion to a point cloud.

    Args:
        sdf (kaolin.rep.SDF) : an object with a .eval_occ function that
            returns the signed distance from the surface of a 3D shape.
        bbox_center (float): center of the surface's bounding box.
        bbox_dim (float): largest dimension of the surface's bounding box.
        resolution (int) : the initial resolution of the voxel, should be large
            enough to properly define the surface.
        upsampling_steps (int) : Number of times the initial resolution will be
            doubled.
            The returned resolution will be resolution * (2 ^ upsampling_steps)
        num_points (int): number of points in computed point cloud.

    Returns:
        (torch.FloatTensor): computed point cloud

    Example:
        >>> sdf = kaolin.rep.SDF.sphere()
        >>> points = kaolin.conversion.sdf_to_pointcloud(sdf, bbox_dim=2)

    """
    verts, faces = sdf_to_trianglemesh(sdf, bbox_center, bbox_dim,
                                       resolution, upsampling_steps)
    mesh = kaolin.rep.TriangleMesh.from_tensors(verts, faces)
    return mesh.sample(num_points)[0]
