# Copyright (c) 2019,20-21 NVIDIA CORPORATION & AFFILIATES.
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
from kaolin import _C
from ..ops.mesh import uniform_laplacian

def point_to_mesh_distance(pointclouds, face_vertices):
    r"""Computes the distances from pointclouds to meshes (represented by vertices and faces).

    For each point in the pointcloud, it finds the nearest triangle
    in the mesh, and calculated its distance to that triangle.

    .. note::

        The calculated distance is the squared euclidean distance.
        

    Type 0 indicates the distance is from a point on the surface of the triangle.

    Type 1 to 3 indicates the distance is from a point to a vertices.

    Type 4 to 6 indicates the distance is from a point to an edge.

    Args:
        pointclouds (torch.Tensor):
            pointclouds, of shape :math:`(\text{batch_size}, \text{num_points}, 3)`.
        face_vertices (torch.Tensor):
            vertices of each face of meshes,
            of shape :math:`(\text{batch_size}, \text{num_faces}, 3, 3)`.

    Returns:
        (torch.Tensor, torch.LongTensor, torch.IntTensor):

            - Distances between pointclouds and meshes,
              of shape :math:`(\text{batch_size}, \text{num_points})`.
            - face indices selected, of shape :math:`(\text{batch_size}, \text{num_points})`.
            - Types of distance of shape :math:`(\text{batch_size}, \text{num_points})`.

    Example:
        >>> from kaolin.ops.mesh import index_vertices_by_faces
        >>> point = torch.tensor([[[0.5, 0.5, 0.5],
        ...                        [3., 4., 5.]]], device='cuda')
        >>> vertices = torch.tensor([[[0., 0., 0.],
        ...                           [0., 1., 0.],
        ...                           [0., 0., 1.]]], device='cuda')
        >>> faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device='cuda')
        >>> face_vertices = index_vertices_by_faces(vertices, faces)
        >>> distance, index, dist_type = point_to_mesh_distance(point, face_vertices)
        >>> distance
        tensor([[ 0.2500, 41.0000]], device='cuda:0')
        >>> index
        tensor([[0, 0]], device='cuda:0')
        >>> dist_type
        tensor([[5, 5]], device='cuda:0', dtype=torch.int32)
    """

    batch_size = pointclouds.shape[0]
    num_points = pointclouds.shape[1]
    device = pointclouds.device
    dtype = pointclouds.dtype

    distance = []
    face_idx = []
    dist_type = []

    for i in range(batch_size):
        if pointclouds.is_cuda:
            cur_dist, cur_face_idx, cur_dist_type = _UnbatchedTriangleDistanceCuda.apply(
                pointclouds[i], face_vertices[i])
        else:
            cur_dist, cur_face_idx, cur_dist_type = _unbatched_naive_point_to_mesh_distance(
                pointclouds[i], face_vertices[i])

        distance.append(cur_dist)
        face_idx.append(cur_face_idx)
        dist_type.append(cur_dist_type)
    return torch.stack(distance, dim=0), torch.stack(face_idx, dim=0), \
        torch.stack(dist_type, dim=0)

def _compute_dot(p1, p2):
    return p1[..., 0] * p2[..., 0] + \
        p1[..., 1] * p2[..., 1] + \
        p1[..., 2] * p2[..., 2]

def _project_edge(vertex, edge, point):
    point_vec = point - vertex
    length = _compute_dot(edge, edge)
    return _compute_dot(point_vec, edge) / length

def _project_plane(vertex, normal, point):
    point_vec = point - vertex
    unit_normal = normal / torch.norm(normal, dim=-1, keepdim=True)
    dist = _compute_dot(point_vec, unit_normal)
    return point - unit_normal * dist.view(-1, 1)

def _is_not_above(vertex, edge, norm, point):
    edge_norm = torch.cross(norm, edge, dim=-1)
    return _compute_dot(edge_norm.view(1, -1, 3),
                        point.view(-1, 1, 3) - vertex.view(1, -1, 3)) <= 0

def _point_at(vertex, edge, proj):
    return vertex + edge * proj.view(-1, 1)

class _UnbatchedTriangleDistanceCuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, face_vertices):
        num_points = points.shape[0]
        num_faces = face_vertices.shape[0]
        min_dist = torch.zeros((num_points), device=points.device, dtype=points.dtype)
        min_dist_idx = torch.zeros((num_points), device=points.device, dtype=torch.long)
        dist_type = torch.zeros((num_points), device=points.device, dtype=torch.int32)
        _C.metrics.unbatched_triangle_distance_forward_cuda(
            points, face_vertices, min_dist, min_dist_idx, dist_type)
        ctx.save_for_backward(points.contiguous(), face_vertices.contiguous(),
                              min_dist_idx, dist_type)
        ctx.mark_non_differentiable(min_dist_idx, dist_type)
        return min_dist, min_dist_idx, dist_type

    @staticmethod
    def backward(ctx, grad_dist, grad_face_idx, grad_dist_type):
        points, face_vertices, face_idx, dist_type = ctx.saved_tensors
        grad_dist = grad_dist.contiguous()
        grad_points = torch.zeros_like(points)
        grad_face_vertices = torch.zeros_like(face_vertices)
        _C.metrics.unbatched_triangle_distance_backward_cuda(
            grad_dist, points, face_vertices, face_idx, dist_type,
            grad_points, grad_face_vertices)
        return grad_points, grad_face_vertices

def _unbatched_naive_point_to_mesh_distance(points, face_vertices):
    """
    description of distance type:
        - 0: distance to face
        - 1: distance to vertice 0
        - 2: distance to vertice 1
        - 3: distance to vertice 2
        - 4: distance to edge 0-1
        - 5: distance to edge 1-2
        - 6: distance to edge 2-0

    Args:
        points (torch.Tensor): of shape (num_points, 3).
        face_vertices (torch.LongTensor): of shape (num_faces, 3, 3).

    Returns:
        (torch.Tensor, torch.LongTensor, torch.IntTensor):

            - distance, of shape (num_points).
            - face_idx, of shape (num_points).
            - distance_type, of shape (num_points).
    """
    num_points = points.shape[0]
    num_faces = face_vertices.shape[0]

    device = points.device
    dtype = points.dtype

    v1 = face_vertices[:, 0]
    v2 = face_vertices[:, 1]
    v3 = face_vertices[:, 2]

    e21 = v2 - v1
    e32 = v3 - v2
    e13 = v1 - v3

    normals = -torch.cross(e21, e13)

    uab = _project_edge(v1.view(1, -1, 3), e21.view(1, -1, 3), points.view(-1, 1, 3))
    ubc = _project_edge(v2.view(1, -1, 3), e32.view(1, -1, 3), points.view(-1, 1, 3))
    uca = _project_edge(v3.view(1, -1, 3), e13.view(1, -1, 3), points.view(-1, 1, 3))

    is_type1 = (uca > 1.) & (uab < 0.)
    is_type2 = (uab > 1.) & (ubc < 0.)
    is_type3 = (ubc > 1.) & (uca < 0.)
    is_type4 = (uab >= 0.) & (uab <= 1.) & _is_not_above(v1, e21, normals, points)
    is_type5 = (ubc >= 0.) & (ubc <= 1.) & _is_not_above(v2, e32, normals, points)
    is_type6 = (uca >= 0.) & (uca <= 1.) & _is_not_above(v3, e13, normals, points)
    is_type0 = ~(is_type1 | is_type2 | is_type3 | is_type4 | is_type5 | is_type6)

    face_idx = torch.zeros(num_points, device=device, dtype=torch.long)
    all_closest_points = torch.zeros((num_points, num_faces, 3), device=device,
                                     dtype=dtype)

    all_type0_idx = torch.where(is_type0)
    all_type1_idx = torch.where(is_type1)
    all_type2_idx = torch.where(is_type2)
    all_type3_idx = torch.where(is_type3)
    all_type4_idx = torch.where(is_type4)
    all_type5_idx = torch.where(is_type5)
    all_type6_idx = torch.where(is_type6)

    all_types = is_type1.int() + is_type2.int() * 2 + is_type3.int() * 3 + \
        is_type4.int() * 4 + is_type5.int() * 5 + is_type6.int() * 6

    all_closest_points[all_type0_idx] = _project_plane(
        v1[all_type0_idx[1]], normals[all_type0_idx[1]], points[all_type0_idx[0]])
    all_closest_points[all_type1_idx] = v1.view(-1, 3)[all_type1_idx[1]]
    all_closest_points[all_type2_idx] = v2.view(-1, 3)[all_type2_idx[1]]
    all_closest_points[all_type3_idx] = v3.view(-1, 3)[all_type3_idx[1]]
    all_closest_points[all_type4_idx] = _point_at(v1[all_type4_idx[1]], e21[all_type4_idx[1]],
                                                  uab[all_type4_idx])
    all_closest_points[all_type5_idx] = _point_at(v2[all_type5_idx[1]], e32[all_type5_idx[1]],
                                                  ubc[all_type5_idx])
    all_closest_points[all_type6_idx] = _point_at(v3[all_type6_idx[1]], e13[all_type6_idx[1]],
                                                  uca[all_type6_idx])
    all_vec = (all_closest_points - points.view(-1, 1, 3))
    all_dist = _compute_dot(all_vec, all_vec)

    _, min_dist_idx = torch.min(all_dist, dim=-1)
    dist_type = all_types[torch.arange(num_points, device=device), min_dist_idx]
    torch.cuda.synchronize()

    # Recompute the shortest distances
    # This reduce the backward pass to the closest faces instead of all faces
    # O(num_points) vs O(num_points * num_faces)
    selected_face_vertices = face_vertices[min_dist_idx]
    v1 = selected_face_vertices[:, 0]
    v2 = selected_face_vertices[:, 1]
    v3 = selected_face_vertices[:, 2]

    e21 = v2 - v1
    e32 = v3 - v2
    e13 = v1 - v3

    normals = -torch.cross(e21, e13)

    uab = _project_edge(v1, e21, points)
    ubc = _project_edge(v2, e32, points)
    uca = _project_edge(v3, e13, points)

    counter_p = torch.zeros((num_points, 3), device=device, dtype=dtype)

    cond = (dist_type == 1)
    counter_p[cond] = v1[cond]

    cond = (dist_type == 2)
    counter_p[cond] = v2[cond]

    cond = (dist_type == 3)
    counter_p[cond] = v3[cond]

    cond = (dist_type == 4)
    counter_p[cond] = _point_at(v1, e21, uab)[cond]

    cond = (dist_type == 5)
    counter_p[cond] = _point_at(v2, e32, ubc)[cond]

    cond = (dist_type == 6)
    counter_p[cond] = _point_at(v3, e13, uca)[cond]

    cond = (dist_type == 0)
    counter_p[cond] = _project_plane(v1, normals, points)[cond]
    min_dist = torch.sum((counter_p - points) ** 2, dim=-1)

    return min_dist, min_dist_idx, dist_type


def average_edge_length(vertices, faces):
    r"""Returns the average length of each faces in a mesh.

    Args:
        vertices (torch.Tensor): Batched vertices, of shape
                                 :math:`(\text{batch_size}, \text{num_vertices}, 3)`.
        faces (torch.LongTensor): Faces, of shape :math:`(\text{num_faces}, 3)`.
    Returns:
        (torch.Tensor):
            average length of each edges in a face, of shape
            :math:`(\text{batch_size}, \text{num_faces})`.

    Example:
        >>> vertices = torch.tensor([[[1, 0, 0],
        ...                           [0, 1, 0],
        ...                           [0, 0, 1]]], dtype=torch.float)
        >>> faces = torch.tensor([[0, 1, 2]])
        >>> average_edge_length(vertices, faces)
        tensor([[1.4142]])
    """
    batch_size = vertices.shape[0]

    p1 = torch.index_select(vertices, 1, faces[:, 0])
    p2 = torch.index_select(vertices, 1, faces[:, 1])
    p3 = torch.index_select(vertices, 1, faces[:, 2])

    # get edge lentgh
    e1 = p2 - p1
    e2 = p3 - p1
    e3 = p2 - p3

    el1 = torch.sqrt((torch.sum(e1**2, dim=2)))
    el2 = torch.sqrt((torch.sum(e2**2, dim=2)))
    el3 = torch.sqrt((torch.sum(e3**2, dim=2)))

    edge_length = (el1 + el2 + el3) / 3.

    return edge_length

def uniform_laplacian_smoothing(vertices, faces):
    r"""Calculates the uniform laplacian smoothing of meshes.
    The position of updated vertices is defined as :math:`V_i = \frac{1}{N} * \sum^{N}_{j=1}V_j`,
    where :math:`N` is the number of neighbours of :math:`V_i`, :math:`V_j` is the position of the
    j-th adjacent vertex.

    Args:
        vertices (torch.Tensor):
            Vertices of the meshes, of shape :math:`(\text{batch_size}, \text{num_vertices}, 3)`.
        faces (torch.LongTensor):
            Faces of the meshes, of shape :math:`(\text{num_faces}, \text{face_size})`.

    Returns:
        (torch.FloatTensor):
            smoothed vertices, of shape :math:`(\text{batch_size}, \text{num_vertices}, 3)`.

    Example:
        >>> vertices = torch.tensor([[[1, 0, 0],
        ...                           [0, 1, 0],
        ...                           [0, 0, 1]]], dtype=torch.float)
        >>> faces = torch.tensor([[0, 1, 2]])
        >>> uniform_laplacian_smoothing(vertices, faces)
        tensor([[[0.0000, 0.5000, 0.5000],
                 [0.5000, 0.0000, 0.5000],
                 [0.5000, 0.5000, 0.0000]]])
    """
    dtype = vertices.dtype
    num_vertices = vertices.shape[1]

    laplacian_matrix = uniform_laplacian(num_vertices, faces).to(dtype)
    smoothed_vertices = torch.matmul(laplacian_matrix, vertices) + vertices

    return smoothed_vertices
