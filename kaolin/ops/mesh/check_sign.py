# Copyright (c) 2019 NVIDIA CORPORATION & AFFILIATES.
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

import numpy as np
import torch

from . import triangle_hash
from kaolin import _C

__all__ = ['check_sign', '_unbatched_check_sign_cuda']

def _unbatched_check_sign_cuda(verts, faces, points):
    n, _ = points.size()
    points = points.contiguous()
    v1 = torch.index_select(verts, 0, faces[:, 0]).view(-1, 3).contiguous()
    v2 = torch.index_select(verts, 0, faces[:, 1]).view(-1, 3).contiguous()
    v3 = torch.index_select(verts, 0, faces[:, 2]).view(-1, 3).contiguous()

    ints = _C.ops.mesh.unbatched_mesh_intersection_cuda(points, v1, v2, v3)
    
    contains = ints % 2 == 1.

    return contains


def check_sign(verts, faces, points, hash_resolution=512):
    r"""Checks if a set of points is contained inside a watertight triangle mesh.

    Shoots a ray from each point to be checked
    and calculates the number of intersections 
    between the ray and triangles in the mesh. 
    Uses the parity of the number of intersections
    to determine if the point is inside the mesh.

    Args:
        verts (torch.Tensor):
            Vertices, of shape :math:`(\text{batch_size}, \text{num_vertices}, 3)`.
        faces (torch.Tensor):
            Faces, of shape :math:`(\text{num_faces}, 3)`.
        points (torch.Tensor):
            Points to check, of shape :math:`(\text{batch_size}, \text{num_points}, 3)`.
        hash_resolution (int):
            Resolution used to check the points sign. Only used with CPU.
            Default: 512.
                               

    Returns:
        (torch.BoolTensor): 
            Tensor indicating whether each point is inside the mesh,
            of shape :math:`(\text{batch_size}, \text{num_points})`.

    Example:
        >>> device = 'cuda' if torch.cuda.is_available() else 'cpu'
        >>> verts = torch.tensor([[[0., 0., 0.],
        ...                       [1., 0.5, 1.],
        ...                       [0.5, 1., 1.],
        ...                       [1., 1., 0.5]]], device = device)
        >>> faces = torch.tensor([[0, 3, 1],
        ...                       [0, 1, 2],
        ...                       [0, 2, 3],
        ...                       [3, 2, 1]], device = device)
        >>> axis = torch.linspace(0.1, 0.9, 3, device = device)
        >>> p_x, p_y, p_z = torch.meshgrid(axis + 0.01, axis + 0.02, axis + 0.03)
        >>> points = torch.cat((p_x.unsqueeze(-1), p_y.unsqueeze(-1), p_z.unsqueeze(-1)), dim=3)
        >>> points = points.view(1, -1, 3)
        >>> check_sign(verts, faces, points)
        tensor([[ True, False, False, False, False, False, False, False, False, False,
                 False, False, False,  True, False, False, False,  True, False, False,
                 False, False, False,  True, False,  True, False]], device='cuda:0')
    """
    assert verts.device == points.device
    assert faces.device == points.device
    device = points.device

    if not faces.dtype == torch.int64:
        raise TypeError(f"Expected faces entries to be torch.int64 "
                        f"but got {faces.dtype}.")
    if not isinstance(hash_resolution, int):
        raise TypeError(f"Expected hash_resolution to be int "
                        f"but got {type(hash_resolution)}.")

    if verts.ndim != 3:
        verts_dim = verts.ndim
        raise ValueError(f"Expected verts to have 3 dimensions " 
                         f"but got {verts_dim} dimensions.")
    if faces.ndim != 2:
        faces_dim = faces.ndim
        raise ValueError(f"Expected faces to have 2 dimensions " 
                         f"but got {faces_dim} dimensions.")
    if points.ndim != 3:
        points_dim = points.ndim
        raise ValueError(f"Expected points to have 3 dimensions " 
                         f"but got {points_dim} dimensions.")

    if verts.shape[2] != 3:
        raise ValueError(f"Expected verts to have 3 coordinates "
                         f"but got {verts.shape[2]} coordinates.")
    if faces.shape[1] != 3:
        raise ValueError(f"Expected faces to have 3 vertices "
                         f"but got {faces.shape[1]} vertices.")
    if points.shape[2] != 3:
        raise ValueError(f"Expected points to have 3 coordinates "
                         f"but got {points.shape[2]} coordinates.")

    xlen = verts[..., 0].max(-1)[0] - verts[..., 0].min(-1)[0]
    ylen = verts[..., 1].max(-1)[0] - verts[..., 1].min(-1)[0]
    zlen = verts[..., 2].max(-1)[0] - verts[..., 2].min(-1)[0]
    maxlen = torch.max(torch.stack([xlen, ylen, zlen]), 0)[0]
    verts = verts / maxlen.view(-1, 1, 1)
    points = points / maxlen.view(-1, 1, 1)
    results = []
    if device.type == 'cuda':
        for i_batch in range(verts.shape[0]):
            contains = _unbatched_check_sign_cuda(verts[i_batch], faces, points[i_batch])
            results.append(contains)
    else:
        for i_batch in range(verts.shape[0]):
            intersector = _UnbatchedMeshIntersector(verts[i_batch], faces, hash_resolution)
            contains = intersector.query(points[i_batch].data.cpu().numpy())
            results.append(torch.tensor(contains).to(device))

    return torch.stack(results)


class _UnbatchedMeshIntersector:
    r"""Class to determine if a point in space lies within or outside a mesh.
    """

    def __init__(self, vertices, faces, resolution=512):
        triangles = vertices.data.cpu().numpy(
        )[faces.data.cpu().numpy()].astype(np.float64)
        n_tri = triangles.shape[0]

        self.resolution = resolution
        self.bbox_min = triangles.reshape(3 * n_tri, 3).min(axis=0)
        self.bbox_max = triangles.reshape(3 * n_tri, 3).max(axis=0)
        # Tranlate and scale it to [0.5, self.resolution - 0.5]^3
        self.scale = (resolution - 1) / (self.bbox_max - self.bbox_min)
        self.translate = 0.5 - self.scale * self.bbox_min

        self._triangles = triangles = self.rescale(triangles)

        triangles2d = triangles[:, :, :2]
        self._tri_intersector2d = _TriangleIntersector2d(
            triangles2d, resolution)

    def query(self, points):
        # Rescale points
        points = self.rescale(points)

        # placeholder result with no hits we'll fill in later
        contains = np.zeros(len(points), dtype=bool)

        # cull points outside of the axis aligned bounding box
        # this avoids running ray tests unless points are close
        inside_aabb = np.all(
            (0 <= points) & (points <= self.resolution), axis=1)
        if not inside_aabb.any():
            return contains

        # Only consider points inside bounding box
        mask = inside_aabb
        points = points[mask]

        # Compute intersection depth and check order
        points_indices, tri_indices = self._tri_intersector2d.query(
            points[:, :2])

        triangles_intersect = self._triangles[tri_indices]
        points_intersect = points[points_indices]

        depth_intersect, abs_n_2 = self.compute_intersection_depth(
            points_intersect, triangles_intersect)

        # Count number of intersections in both directions
        smaller_depth = depth_intersect >= points_intersect[:, 2] * abs_n_2
        bigger_depth = depth_intersect < points_intersect[:, 2] * abs_n_2
        points_indices_0 = points_indices[smaller_depth]
        points_indices_1 = points_indices[bigger_depth]

        nintersect0 = np.bincount(points_indices_0, minlength=points.shape[0])
        nintersect1 = np.bincount(points_indices_1, minlength=points.shape[0])

        # Check if point contained in mesh
        contains1 = (np.mod(nintersect0, 2) == 1)
        contains2 = (np.mod(nintersect1, 2) == 1)
        # if (contains1 != contains2).any():
        #     print('Warning: contains1 != contains2 for some points.')
        contains[mask] = (contains1 & contains2)
        return contains

    def compute_intersection_depth(self, points, triangles):
        t1 = triangles[:, 0, :]
        t2 = triangles[:, 1, :]
        t3 = triangles[:, 2, :]

        v1 = t3 - t1
        v2 = t2 - t1
        # v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
        # v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)

        normals = np.cross(v1, v2)
        alpha = np.sum(normals[:, :2] * (t1[:, :2] - points[:, :2]), axis=1)

        n_2 = normals[:, 2]
        t1_2 = t1[:, 2]
        s_n_2 = np.sign(n_2)
        abs_n_2 = np.abs(n_2)

        mask = (abs_n_2 != 0)

        depth_intersect = np.full(points.shape[0], np.nan)
        depth_intersect[mask] = \
            t1_2[mask] * abs_n_2[mask] + alpha[mask] * s_n_2[mask]
        return depth_intersect, abs_n_2

    def rescale(self, array):
        array = self.scale * array + self.translate
        return array


class _TriangleIntersector2d:
    def __init__(self, triangles, resolution=128):
        self.triangles = triangles
        self.tri_hash = triangle_hash.TriangleHash(triangles, resolution)

    def query(self, points):
        point_indices, tri_indices = self.tri_hash.query(points)
        point_indices = np.array(point_indices, dtype=np.int64)
        tri_indices = np.array(tri_indices, dtype=np.int64)
        points = points[point_indices]
        triangles = self.triangles[tri_indices]
        mask = self.check_triangles(points, triangles)
        point_indices = point_indices[mask]
        tri_indices = tri_indices[mask]
        return point_indices, tri_indices

    def check_triangles(self, points, triangles):
        contains = np.zeros(points.shape[0], dtype=bool)
        A = triangles[:, :2] - triangles[:, 2:]
        A = A.transpose([0, 2, 1])
        y = points - triangles[:, 2]

        detA = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]

        mask = (np.abs(detA) != 0.)
        A = A[mask]
        y = y[mask]
        detA = detA[mask]

        s_detA = np.sign(detA)
        abs_detA = np.abs(detA)

        u = (A[:, 1, 1] * y[:, 0] - A[:, 0, 1] * y[:, 1]) * s_detA
        v = (-A[:, 1, 0] * y[:, 0] + A[:, 0, 0] * y[:, 1]) * s_detA

        sum_uv = u + v
        contains[mask] = (
            (0 < u) & (u < abs_detA) & (0 < v) & (v < abs_detA)
            & (0 < sum_uv) & (sum_uv < abs_detA)
        )
        return contains
