# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
from ..ops.mesh import uniform_laplacian
from kaolin import _C

class _UnbatchedTriangleDistance(torch.autograd.Function):
    """torch.autograd.Function for triangle_distance.

    Refer to :func:`triangle_distance`.
    """

    @staticmethod
    def forward(ctx, pointcloud, v1, v2, v3):
        device = pointcloud.device
        dtype = pointcloud.dtype

        if not pointcloud.is_cuda:
            raise NotImplementedError("Triangle Distance currently does not support CPU.")

        # For now we don't support half because it will cause issue when backward gradients.
        if pointcloud.dtype == torch.half:
            raise NotImplementedError("Currently it doesn't support torch.half.")

        pointcloud = pointcloud.contiguous()
        v1 = v1.contiguous()
        v2 = v2.contiguous()
        v3 = v3.contiguous()

        n, _ = pointcloud.size()

        dist = torch.zeros(n, device=device, dtype=dtype)
        idx = torch.zeros(n, device=device, dtype=torch.long)
        dist_type = torch.zeros(n, device=device, dtype=torch.int)

        _C.metrics.unbatched_triangle_distance_forward_cuda(pointcloud, v1, v2, v3, dist, idx, dist_type)

        ctx.save_for_backward(pointcloud, v1, v2, v3, idx, dist_type)
        ctx.mark_non_differentiable(idx, dist_type)

        return dist, idx, dist_type

    @staticmethod
    def backward(ctx, grad_output_dist, grad_output_idx, grad_output_type):
        pointcloud, v1, v2, v3, idx, dist_type = ctx.saved_tensors

        v1 = v1.contiguous()
        v2 = v2.contiguous()
        v3 = v3.contiguous()

        grad_input_p = torch.zeros_like(pointcloud)
        grad_input_v1 = torch.zeros_like(v1)
        grad_input_v2 = torch.zeros_like(v2)
        grad_input_v3 = torch.zeros_like(v3)

        grad_output_dist = grad_output_dist.contiguous()

        _C.metrics.unbatched_triangle_distance_backward_cuda(
            grad_output_dist, pointcloud, v1, v2, v3, idx, dist_type, grad_input_p,
            grad_input_v1, grad_input_v2, grad_input_v3)

        return grad_input_p, grad_input_v1, grad_input_v2, grad_input_v3

def _point_to_mesh_distance_cuda(pointcloud, vertices, faces):
    r"""Returns the distance between pointclouds and meshes.
    For each point in the pointcloud, it finds the nearest triangle
    in the mesh, and calculated its distance to that triangle. There 
    are three kind of distances. Type 0 to 2 indicates which edge the point is closest to. 
    Type 3 indicates the distance is from a point on the surface of the triangle, not an edge.

    Args:
        pointcloud (torch.Tensor): Batched Pointcloud of shape (B, P, 3)
        vertices (torch.Tensor): Batched Vertices of shape (B, V, 3) of a Mesh
        faces (torch.LongTensor): Faces of shape (F, 3) of a Mesh

    Returns:
        (torch.Tensor, torch.LongTensor, torch.IntTensor): Distance of shape (B, P), corresponding indices of shape (P) in faces,
        and type of shape (B, P) of distances. 
    """
    vertices_dtype = vertices.dtype
    batch_size = pointcloud.shape[0]

    if vertices.shape[0] != batch_size:
        raise ValueError(f"Expect pointcloud and vertices to have same batch size, "
                         f"but got {vertices.shape[0]} for vertices and {batch_size} for pointcloud.")

    # TODO: Move the type casting into cuda backward functions
    # TODO: Right now this function is sensitive to half datatype when doing compare gradients with double's
    # gradient. Currently we fix the seed to pass the random half gradients comparision with double gradients.
    #  need to find a better way in the future.

    all_dists = []
    all_idx = []
    all_dist_type = []
    for i in range(batch_size):
        v1 = torch.index_select(vertices[i], 0, faces[:, 0])
        v2 = torch.index_select(vertices[i], 0, faces[:, 1])
        v3 = torch.index_select(vertices[i], 0, faces[:, 2])

        dist, idx, dist_type = _UnbatchedTriangleDistance.apply(pointcloud[i], v1, v2, v3)

        all_dists.append(dist)
        all_idx.append(idx)
        all_dist_type.append(dist_type)

    return torch.stack(all_dists), torch.stack(all_idx), torch.stack(all_dist_type)

def point_to_mesh_distance(pointclouds, vertices, faces):
    r"""Computes the distances from pointclouds to meshes (represented by vertices and faces.)
    For each point in the pointcloud, it finds the nearest triangle
    in the mesh, and calculated its distance to that triangle. There 
    are three kind of distances. Type 0 to 2 indicates which edge the point is closest to. 
    Type 3 indicates the distance is from a point on the surface of the triangle, not an edge.

    Args:
        pointclouds (torch.Tensor): pointclouds of shape (B, P, 3)
        vertices (torch.Tensor): vertices of meshes of shape (B, V, 3)
        faces (torch.LongTensor): faces of meshes of shape (F, 3)

    Returns:
        (torch.Tensor, torch.LongTensor, torch.IntTensor): distance between pointclouds and meshes of shape (B, P),
                                                           corresponding indices of shape (B, P) in faces,
                                                           and type of shape (B, P) of distances.  

    Example:
        >>> vertices = torch.tensor([[[0, 0, 0],
        ...                           [0, 1, 0],
        ...                           [0, 0, 1]]], device='cuda', dtype=torch.float)
        >>> faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device='cuda')
        >>> point = torch.tensor([[[0.5, 0.5, 0.5],
        ...                        [3, 4, 5]]], device='cuda', dtype=torch.float)
        >>> distance, index, dist_type = point_to_mesh_distance(point, vertices, faces)
        >>> distance
        tensor([[ 0.2500, 41.0000]], device='cuda:0')
        >>> index
        tensor([[0, 0]], device='cuda:0')
        >>> dist_type
        tensor([[1, 1]], device='cuda:0', dtype=torch.int32)
    """
    device = pointclouds.device
    dtype = pointclouds.dtype

    batch_size = pointclouds.shape[0]

    if pointclouds.is_cuda:
        distance, idx, dist_type = _point_to_mesh_distance_cuda(pointclouds, vertices, faces)
    else:
        P = pointclouds.shape[1]
        distance = torch.zeros((batch_size, P), device=device, dtype=dtype)
        idx = torch.zeros((batch_size, P), device=device, dtype=torch.long)
        dist_type = torch.zeros((batch_size, P), device=device, dtype=torch.int)

        for i in range(batch_size):
            curr_dist, curr_index, curr_dist_type = _point_to_mesh_distance_cpu(pointclouds[i], vertices[i], faces)
            distance[i] = curr_dist
            idx[i] = curr_index
            dist_type[i] = curr_dist_type

    return distance, idx, dist_type

def _compute_dot(p1, p2):
    # batched dot product
    return torch.bmm(p1.view(p1.shape[0], 1, 3),
                     p2.view(p2.shape[0], 3, 1)).view(-1)

def _compute_planar_dist(normal, point):
    # batched distance between a point and a tiangle
    if normal.shape[0] == 0:
        return normal
    dot = _compute_dot(normal, point)
    dot_div = _compute_dot(normal, normal)
    return dot * dot / dot_div

def _point_to_mesh_distance_cpu(points, vertices, faces):
    P = points.shape[0]
    V = vertices.shape[0]
    F = faces.shape[0]

    device = points.device
    dtype = points.dtype

    v1 = torch.index_select(vertices, 0, faces[:, 0])
    v2 = torch.index_select(vertices, 0, faces[:, 1])
    v3 = torch.index_select(vertices, 0, faces[:, 2])

    v21 = v2 - v1
    v32 = v3 - v2
    v13 = v1 - v3

    normals = torch.cross(v21, v13)

    # make more of them, one set for each sampled point
    v1 = v1.unsqueeze(0).expand(P, F, 3).contiguous().view(-1, 3)
    v2 = v2.unsqueeze(0).expand(P, F, 3).contiguous().view(-1, 3)
    v3 = v3.unsqueeze(0).expand(P, F, 3).contiguous().view(-1, 3)

    v21 = v21.unsqueeze(0).expand(P, F, 3).contiguous().view(-1, 3)
    v32 = v32.unsqueeze(0).expand(P, F, 3).contiguous().view(-1, 3)
    v13 = v13.unsqueeze(0).expand(P, F, 3).contiguous().view(-1, 3)

    normals = normals.unsqueeze(0).expand(P, F, 3).contiguous().view(-1, 3)

    p = points.unsqueeze(1).expand(P, F, 3).contiguous().view(-1, 3)

    p1 = p - v1  # shape (P * F, 3)
    p2 = p - v2
    p3 = p - v3

    sign1 = _compute_sign(v21, normals, p1)
    sign2 = _compute_sign(v32, normals, p2)
    sign3 = _compute_sign(v13, normals, p3)

    outside_triangle = torch.le(torch.abs(sign1 + sign2 + sign3), 2)
    inside_triangle = torch.logical_not(outside_triangle)

    outside_triangle = torch.where(outside_triangle)
    inside_triangle = torch.where(inside_triangle)

    distances = torch.zeros(P * F, device=device, dtype=dtype)
    distances_type = torch.zeros(P * F, device=device, dtype=torch.int)

    dist1 = _compute_edge_dist(v21[outside_triangle], p1[outside_triangle])
    dist2 = _compute_edge_dist(v32[outside_triangle], p2[outside_triangle])
    dist3 = _compute_edge_dist(v13[outside_triangle], p3[outside_triangle])
    all_distances = torch.cat((dist1, dist2, dist3), dim=1)
    edge_distance, edge_distance_index = torch.min(all_distances, dim=1)

    distances[outside_triangle] = edge_distance
    distances_type[outside_triangle] = edge_distance_index.int()

    face_distance = _compute_planar_dist(normals[inside_triangle], p1[inside_triangle])

    distances[inside_triangle] = face_distance
    distances_type[inside_triangle] = 3

    distances = distances.view(P, F)

    min_distances, distances_index = torch.min(distances, dim=1)

    distances_type = distances_type.view(P, F)
    distances_type = distances_type.gather(dim=1, index=distances_index.view(-1, 1)).squeeze(-1)

    return min_distances, distances_index, distances_type

def _compute_sign(v, nor, p):
    sign = torch.cross(v, nor)
    sign = _compute_dot(sign, p)
    sign = sign.sign()
    return sign

def _compute_edge_dist(v, p):
    if v.shape[0] == 0:
        return v

    dotter = _compute_dot(v, p)
    dotter_div = _compute_dot(v, v)
    dotter = torch.clamp(dotter / dotter_div, 0.0, 1.0).view(-1, 1)
    dotter = (v * dotter) - p
    dist = _compute_dot(dotter, dotter)
    return dist.view(-1, 1)

def average_edge_length(vertices, faces):
    r"""Returns the average length of each faces in a mesh.

    Args:
        vertices (torch.Tensor): Batched vertices of shape (B, V, 3) of a Mesh
        faces (torch.LongTensor): Faces of shape (F, 3) of a Mesh
    Returns:
        (torch.Tensor): average length of each edges in a face of shape (B, F)

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
