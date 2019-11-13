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

import torch
import os
import torch.nn.functional as F
import numpy as np
import kaolin

import kaolin as kal
from kaolin.metrics.point import directed_distance as directed_distance


def trianglemesh_to_pointcloud(mesh: kaolin.rep.Mesh, num_points: int):
    r""" Converts  passed mesh to a pointcloud

    Args:
        mesh (kaolin.rep.Mesh): mesh to convert
        num_points (int):number of points in converted point cloud

    Returns:
       (torch.Tensor): converted point cloud

    Example:
        >>> mesh = kal.TriangleMesh.from_obj('object.obj')
        >>> points = kal.conversions.trianglemesh_to_pointcloud(mesh, 10)
        >>> points
        tensor([[ 0.0524,  0.0039, -0.0111],
                [-0.1995,  0.2999,  0.0408],
                [-0.1921, -0.0268,  0.1811],
                [ 0.1292,  0.0039,  0.2030],
                [-0.1859,  0.1764,  0.0168],
                [-0.1749,  0.1515, -0.0925],
                [ 0.1990,  0.0039, -0.0083],
                [ 0.2173, -0.1285, -0.2248],
                [-0.1916, -0.2143,  0.2064],
                [-0.1935,  0.2401,  0.1003]])
        >>> points.shape
        torch.Size([10, 3])
    """

    points, face_choices = mesh.sample(num_points)
    return points, face_choices


def trianglemesh_to_voxelgrid(mesh: kaolin.rep.Mesh, resolution: int,
             normalize: bool = True, vertex_offset: float = 0.):
    r""" Converts mesh to a voxel model of a given resolution

    Args:
        mesh (kaolin.rep.Mesh): mesh to convert
        resolution (int): desired dresolution of generated voxel array
        normalize (bool): Determines whether to normalize vertices
        vertex_offset (float): Offset applied to all vertices after
                               normalizing.

    Returns:
        voxels (torch.Tensor): voxel array of desired resolution

    Example:
        >>> mesh = kal.TriangleMesh.from_obj('model.obj')
        >>> voxel = kal.conversions.trianglemesh_to_voxelgrid(mesh, 32)
        >>> voxel.shape

    """
    mesh = kal.rep.Mesh.from_tensors(mesh.vertices.clone(), mesh.faces.clone())
    if normalize:
        verts_max = mesh.vertices.max()
        verts_min = mesh.vertices.min()
        mesh.vertices = (mesh.vertices - verts_min) / (verts_max - verts_min)

    mesh.vertices = mesh.vertices + vertex_offset

    points = mesh.vertices
    smallest_side = (1. / resolution)**2

    if mesh.faces.shape[-1] == 4:
        tri_faces_1 = torch.cat((mesh.faces[:, :2], mesh.faces[:, 3:]), dim=1)
        tri_faces_2 = torch.cat((mesh.faces[:, :1], mesh.faces[:, 2:]), dim=1)
        faces = torch.cat((tri_faces_1, tri_faces_2))
    else:
        faces = mesh.faces.clone()

    v1 = torch.index_select(mesh.vertices, 0, faces[:, 0])
    v2 = torch.index_select(mesh.vertices, 0, faces[:, 1])
    v3 = torch.index_select(mesh.vertices, 0, faces[:, 2])

    while True:
        side_1 = (torch.abs(v1 - v2)**2).sum(dim=1).unsqueeze(1)
        side_2 = (torch.abs(v2 - v3)**2).sum(dim=1).unsqueeze(1)
        side_3 = (torch.abs(v3 - v1)**2).sum(dim=1).unsqueeze(1)
        sides = torch.cat((side_1, side_2, side_3), dim=1)
        sides = sides.max(dim=1)[0]

        keep = sides > smallest_side
        if keep.sum() == 0:
            break
        v1 = v1[keep]
        v2 = v2[keep]
        v3 = v3[keep]
        del(side_1, side_2, side_3, keep, sides)

        v4 = (v1 + v3) / 2.
        v5 = (v1 + v2) / 2.
        v6 = (v2 + v3) / 2.

        points = torch.cat((points, v4, v5, v6))

        vertex_set = [v1, v2, v3, v4, v5, v6]
        new_traingles = [[0, 3, 4], [4, 1, 5], [4, 3, 5], [3, 2, 5]]
        new_verts = []

        for i in range(4):
            for j in range(3):
                if i == 0:
                    new_verts.append(vertex_set[new_traingles[i][j]])
                else:
                    new_verts[j] = torch.cat(
                        (new_verts[j], vertex_set[new_traingles[i][j]]))
        v1, v2, v3 = new_verts
        del(v4, v5, v6, vertex_set, new_verts)

    del(v1, v2, v3)

    voxel = torch.zeros((resolution, resolution, resolution))
    points = (points * (resolution - 1)).long()
    points = torch.split(points.permute(1, 0), 1, dim=0)
    points = [m.unsqueeze(0) for m in points]
    voxel[points] = 1
    return voxel


def trianglemesh_to_sdf(mesh: kaolin.rep.Mesh, num_points: int = 10000):
    r""" Converts mesh to a SDF function

    Args:
        mesh (kaolin.rep.Mesh): mesh to convert.
        num_points (int): number of points to sample on surface of the mesh.

    Returns:
        sdf: a signed distance function

    Example:
        >>> mesh = kal.TriangleMesh.from_obj('object.obj')
        >>> sdf = kal.conversions.trianglemesh_to_sdf(mesh)
        >>> points = torch.rand(100,3)
        >>> distances = sdf(points)
    """
    surface_points, _ = mesh.sample(num_points)

    def eval_query(query):
        distances = directed_distance(query, surface_points, mean=False)
        occ_points = kal.rep.SDF.check_sign(mesh, query)
        if torch.is_tensor(occ_points):
            occ_points = occ_points.cpu().numpy()[0]
        distances[np.where(occ_points)] *= -1
        return distances

    return eval_query
