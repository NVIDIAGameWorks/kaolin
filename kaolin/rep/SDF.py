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

import numpy as np
import torch

import kaolin as kal
from kaolin.triangle_hash import TriangleHash as _TriangleHash
import kaolin.cuda.mesh_intersection as mint


class MeshIntersectionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, points: torch.Tensor, verts_1: torch.Tensor,
                verts_2: torch.Tensor, verts_3: torch.Tensor):
        batchsize, n, _ = points.size()
        points = points.contiguous()
        verts_1 = verts_1.contiguous()
        verts_2 = verts_2.contiguous()
        verts_3 = verts_3.contiguous()

        ints = torch.zeros(batchsize, n)
        ints = ints.cuda()

        mint.forward_cuda(points, verts_1, verts_2, verts_3, ints)
        ctx.save_for_backward(ints)

        return ints

    @staticmethod
    def backward(ctx, graddist1: torch.Tensor, graddist2: torch.Tensor):
        ints = ctx.saved_tensors
        gradxyz1 = torch.zeros(ints.size())
        return gradxyz1


class MeshIntersection(torch.nn.Module):
    def forward(self, points: torch.Tensor, verts_1: torch.Tensor,
                verts_2: torch.Tensor, verts_3: torch.Tensor):
        return MeshIntersectionFunction.apply(points, verts_1, verts_2,
                                              verts_3)


def check_sign_fast(mesh, points):
    intersector = MeshIntersection()
    v1 = torch.index_select(mesh.vertices, 0, mesh.faces[:, 0]).view(1, -1, 3)
    v2 = torch.index_select(mesh.vertices, 0, mesh.faces[:, 1]).view(1, -1, 3)
    v3 = torch.index_select(mesh.vertices, 0, mesh.faces[:, 2]).view(1, -1, 3)
    contains = intersector(points.view(1, -1, 3), v1, v2, v3)
    contains = contains > 0
    return contains


def check_sign(mesh, points, hash_resolution=512):
    r""" Checks if a set of points is contained within a mesh

    Args:
        mesh (kal.rep.Mesh): mesh to check against
        points (torch.Tensor): points to check
        hash_resolution: resolution used to check the points sign

    Returns:
        bool value for every point inciating if point is inside object

    Example:


    """
    if mesh.device.type == 'cuda':
        return check_sign_fast(mesh, points)
    else:
        intersector = _MeshIntersector(mesh, hash_resolution)
        contains = intersector.query(points.data.cpu().numpy())
        return contains


def _length(points):
    return torch.sqrt(((points**2).sum(dim=1)))


def sphere(r=.5):

    def eval_sdf(points):
        return _length(points) - r
    return eval_sdf


def box(h=.2, w=.4, l=.5):
    def eval_sdf(points):
        d = torch.abs(points)
        d[:, 0] -= h
        d[:, 1] -= w
        d[:, 2] -= l
        positive_len = _length(torch.max(d, torch.zeros(d.shape).to(d.device)))

        negative_res = torch.max(d[:, 1], d[:, 2])
        negative_res = torch.max(d[:, 0], negative_res)
        negative_res = torch.min(negative_res, torch.zeros(
            negative_res.shape).to(d.device))
        positive_len = positive_len + negative_res

        return positive_len
    return eval_sdf


class _MeshIntersector:
    r"""Class to determine if a point in space lies within our outside a mesh.
    """

    def __init__(self, mesh, resolution=512):
        triangles = mesh.vertices.data.cpu().numpy(
        )[mesh.faces.data.cpu().numpy()].astype(np.float64)
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
        contains = np.zeros(len(points), dtype=np.bool)

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

        # Test the depth:
        # TODO: remove and put into tests
        # points_new = np.concatenate([points[:, :2], depth_intersect[:, None]], axis=1)
        # alpha = (normals * t1).sum(-1)
        # mask = (depth_intersect == depth_intersect)
        # assert(np.allclose((points_new[mask] * normals[mask]).sum(-1),
        #                    alpha[mask]))
        return depth_intersect, abs_n_2

    def rescale(self, array):
        array = self.scale * array + self.translate
        return array


class _TriangleIntersector2d:
    def __init__(self, triangles, resolution=128):
        self.triangles = triangles
        self.tri_hash = _TriangleHash(triangles, resolution)

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
        contains = np.zeros(points.shape[0], dtype=np.bool)
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
