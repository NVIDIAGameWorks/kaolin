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
# GEOMetrics
#
# Copyright (c) 2019 Edward Smith
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
import kaolin as kal
from kaolin.rep import Mesh
import kaolin.cuda.tri_distance as td
import numpy as np



class TringleDistanceFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, verts_1, verts_2, verts_3):
        batchsize, n, _ = points.size()
        points = points.contiguous()
        verts_1 = verts_1.contiguous()
        verts_2 = verts_2.contiguous()
        verts_3 = verts_3.contiguous()

        dist1 = torch.zeros(batchsize, n)
        idx1 = torch.zeros(batchsize, n, dtype=torch.int)
        type1 = torch.zeros(batchsize, n, dtype=torch.int)

        dist1 = dist1.cuda()
        idx1 = idx1.cuda()
        type1 = type1.cuda()
        td.forward_cuda(points, verts_1, verts_2, verts_3, dist1, idx1, type1)
        ctx.save_for_backward(idx1)

        return dist1[0].detach(), idx1[0].detach().long(), type1[0].long()


class TriangleDistance(torch.nn.Module):
    def forward(self, points, verts_1, verts_2, verts_3):
        points = points.view(1, -1, 3)
        verts_1 = verts_1.view(1, -1, 3)
        verts_2 = verts_2.view(1, -1, 3)
        verts_3 = verts_3.view(1, -1, 3)
        return TringleDistanceFunction.apply(points, verts_1, verts_2, verts_3)


def chamfer_distance(mesh1: Mesh, mesh2: Mesh,
                     w1: float = 1., w2: float = 1., num_points=3000):
    r""" computes the chamfer distance bewteen two meshes by sampling the two surfaces
    Args:
            mesh1: (Mesh): first mesh
            mesh2: (Mesh): second mesh
            w1: (float): weighting of forward direction
            w2: (float): weighting of backward direction
            num_points: number of points to sample on each mesh

    Returns:
            chamfer_distance (torch.Tensor): chamfer distance

    Example:
            >>> mesh1 = TriangleMesh.from_obj(file1)
            >>> mesh2 = TriangleMesh.from_obj(file2)
            >>> distance = chamfer_distance(mesh1, mesh2, 500)

    """

    set1 = mesh1.sample(num_points)[0]
    set2 = mesh2.sample(num_points)[0]
    return kal.metrics.point.chamfer_distance(set1, set2, w1, w2)


def edge_length(mesh: Mesh):
    r"""Returns the average length of a face in a mesh

    Args:
            mesh (Mesh): mesh over which to calcuale edge length

    Returns:
            edge_length (torch.Tensor): averge lenght of mesh edge

    Example:
            >>> mesh  = TriangleMesh.from_obj(file)
            >>> length = edge_length(mesh)

    """

    p1 = torch.index_select(mesh.vertices, 0, mesh.faces[:, 0])
    p2 = torch.index_select(mesh.vertices, 0, mesh.faces[:, 1])
    p3 = torch.index_select(mesh.vertices, 0, mesh.faces[:, 2])
    # get edge lentgh
    e1 = p2 - p1
    e2 = p3 - p1
    e3 = p2 - p3

    el1 = ((torch.sum(e1**2, 1))).mean()
    el2 = ((torch.sum(e2**2, 1))).mean()
    el3 = ((torch.sum(e3**2, 1))).mean()

    edge_length = (el1 + el2 + el3) / 6.
    return edge_length


def laplacian_loss(mesh1: Mesh, mesh2: Mesh):
    r"""Returns the change in laplacian over two meshes

    Args:
            mesh1 (Mesh): first mesh
            mesh2: (Mesh): second mesh


    Returns:
            lap_loss (torch.Tensor):  laplacian change over the mesh

    Example:
            >>> mesh1 = TriangleMesh.from_obj(file)
            >>> mesh2 = TriangleMesh.from_obj(file)
            >>> mesh2.vertices = mesh2.vertices * 1.05
            >>> lap = laplacian_loss(mesh1, mesh2)

    """

    lap1 = mesh1.compute_laplacian()
    lap2 = mesh2.compute_laplacian()
    lap_loss = torch.mean(torch.sum((lap1 - lap2)**2, 1))
    return lap_loss


def point_to_surface(points: torch.Tensor, mesh: Mesh):
    r"""Computes the minimum distances from a set of points to a mesh

    Args:
            points (torch.Tensor): set of points
            mesh (Mesh): mesh to calculate distance

    Returns:
            distance: mean distance between points and surface

    Example:
            >>> mesh = TriangleMesh.from_obj(file)
            >>> points = torch.rand(1000,3)
            >>> loss = point_to_surface(points, mesh)

    """

    # extract triangle defs from mesh
    v1 = torch.index_select(mesh.vertices.clone(), 0, mesh.faces[:, 0])
    v2 = torch.index_select(mesh.vertices.clone(), 0, mesh.faces[:, 1])
    v3 = torch.index_select(mesh.vertices.clone(), 0, mesh.faces[:, 2])

    # if quad mesh the separate the triangles
    if mesh.faces.shape[-1] == 4:
        v4 = torch.index_select(mesh.vertices.clone(), 0, mesh.faces[:, 3])
        temp1 = v1.clone()
        temp2 = v2.clone()
        temp3 = v3.clone()
        v1 = torch.cat((v1, v1), dim=0)
        v2 = torch.cat((v2, v4), dim=0)
        v3 = torch.cat((v3, v3), dim=0)

    if points.is_cuda:

        tri_minimum_dist = TriangleDistance()
        # pass to cuda
        distance, indx, dist_type = tri_minimum_dist(points, v1, v2, v3)
        indx = indx.data.cpu().numpy()
        dist_type = torch.LongTensor(dist_type.data.cpu().numpy())
        # reconpute distances to define gradient
        grad_dist = _recompute_point_to_surface(
            [v1, v2, v3], points, indx, dist_type)
        # sanity check
        # print(distance.mean(), grad_dist)
    else:
        grad_dist = _point_to_surface_cpu(v1, v2, v3, points)

    return grad_dist


def _recompute_point_to_surface(verts, p, indecies, dist_type):
    # recompute surface based the calcualted correct assignments of points and triangles
    # and the type of distacne, type 1 to 3 idicates which edge to calcualte to,
    # type 4 indicates the distance is from a point on the triangle not an edge
    v1, v2, v3 = verts
    v1 = v1[indecies]
    v2 = v2[indecies]
    v3 = v3[indecies]

    type_1 = (dist_type == 0)
    type_2 = (dist_type == 1)
    type_3 = (dist_type == 2)
    type_4 = (dist_type == 3)

    v21 = v2 - v1
    v32 = v3 - v2
    v13 = v1 - v3

    p1 = p - v1
    p2 = p - v2
    p3 = p - v3

    dists = []
    dists.append(_compute_edge_dist(v21[type_1], p1[type_1]).view(-1))
    dists.append(_compute_edge_dist(v32[type_2], p2[type_2]).view(-1))
    dists.append(_compute_edge_dist(v13[type_3], p3[type_3]).view(-1))

    if len(np.where(type_4)[0]) > 0:
        nor = torch.cross(v21[type_4], v13[type_4])
        dists.append(_compute_planar_dist(nor, p1[type_4]))

    distances = torch.cat(dists)

    return torch.mean((distances))


def _compute_edge_dist(v, p):
    # batched distance between an edge and a point
    if v.shape[0] == 0:
        return v
    dots = _compute_dot(v, p)
    dots_div = _compute_dot(v, v,)
    dots = torch.clamp(dots / dots_div, 0.0, 1.0).view(-1, 1)
    dots = (v * dots) - p
    dots = _compute_dot(dots, dots)
    return dots.view(-1, 1)


def _compute_dot(p1, p2):
    # batched dot product
    return torch.bmm(p1.view(p1.shape[0], 1, 3),
                     p2.view(p2.shape[0], 3, 1)).view(-1)


def _compute_planar_dist(nor, p):
    # batched distance between a point and a tiangle
    if nor.shape[0] == 0:
        return nor
    dot = _compute_dot(nor, p)
    dot_div = _compute_dot(nor, nor)
    return dot * dot / dot_div


def _point_to_surface_cpu(v1, v2, v3, points):

    faces_len = v1.shape[0]
    v21 = v2 - v1
    v32 = v3 - v2
    v13 = v1 - v3
    nor = torch.cross(v21, v13)

    # make more of them, one set for each sampled point
    v1 = v1.view(
        1, -1, 3).expand(points.shape[0], v1.shape[0], 3).contiguous().view(-1, 3)
    v2 = v2.view(
        1, -1, 3).expand(points.shape[0], v2.shape[0], 3).contiguous().view(-1, 3)
    v3 = v3.view(
        1, -1, 3).expand(points.shape[0], v3.shape[0], 3).contiguous().view(-1, 3)
    v21 = v21.view(
        1, -1, 3).expand(points.shape[0], v21.shape[0], 3).contiguous().view(-1, 3)
    v32 = v32.view(
        1, -1, 3).expand(points.shape[0], v32.shape[0], 3).contiguous().view(-1, 3)
    v13 = v13.view(
        1, -1, 3).expand(points.shape[0], v13.shape[0], 3).contiguous().view(-1, 3)
    nor = nor.view(
        1, -1, 3).expand(points.shape[0], nor.shape[0], 3).contiguous().view(-1, 3)
    p = points.view(-1, 1,
                    3).expand(points.shape[0], faces_len, 3).contiguous().view(-1, 3)

    p1 = p - v1
    p2 = p - v2
    p3 = p - v3

    del(v1)
    del(v2)
    del(v3)

    sign1 = _compute_sign(v21, nor, p1)
    sign2 = _compute_sign(v32, nor, p2)
    sign3 = _compute_sign(v13, nor, p3)

    outside_tringle = torch.le(torch.abs(sign1 + sign2 + sign3), 2)
    inside_tringle = torch.gt(torch.abs(sign1 + sign2 + sign3), 2)
    distances = torch.FloatTensor(np.zeros(sign1.shape))

    del (sign1)
    del (sign2)
    del (sign3)

    outside_tringle = np.where(outside_tringle)
    inside_tringle = np.where(inside_tringle)

    try:
        dotter1 = _compute_dotter(v21[outside_tringle], p1[outside_tringle])
        dotter2 = _compute_dotter(v32[outside_tringle], p2[outside_tringle])
        dotter3 = _compute_dotter(v13[outside_tringle], p3[outside_tringle])
        dots = torch.cat((dotter1, dotter2, dotter3), dim=1)
        edge_distance = torch.min(dots, dim=1)[0]
    except BaseException:
        edge_distance = 0

    distances[outside_tringle] = edge_distance
    try:
        face_distance = _compute_planar_dist(
            nor[inside_tringle], p1[inside_tringle])
    except BaseException:
        face_distance = 0
    distances[inside_tringle] = face_distance

    distances = distances.view(points.shape[0], faces_len)

    min_distaces = torch.min(distances, dim=1)[0]

    return torch.mean(min_distaces)


def _compute_sign(v, nor, p):
    sign = torch.cross(v, nor)
    sign = _compute_dot(sign, p)
    sign = sign.sign()
    return sign


def _compute_dotter(v, p):
    if v.shape[0] == 0:
        return v
    dotter = _compute_dot(v, p)
    dotter_div = _compute_dot(v, v,)
    dotter = torch.clamp(dotter / dotter_div, 0.0, 1.0).view(-1, 1)
    dotter = (v * dotter) - p
    dotter = _compute_dot(dotter, dotter)
    return dotter.view(-1, 1)
