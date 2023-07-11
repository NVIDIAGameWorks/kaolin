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

__all__ = [
    'index_vertices_by_faces',
    'adjacency_matrix',
    'compute_vertex_normals',
    'uniform_laplacian',
]

def index_vertices_by_faces(vertices_features, faces):
    r"""Index vertex features to convert per vertex tensor to per vertex per face tensor.

    Args:
        vertices_features (torch.FloatTensor):
            vertices features, of shape
            :math:`(\text{batch_size}, \text{num_points}, \text{knum})`,
            ``knum`` is feature dimension, the features could be xyz position,
            rgb color, or even neural network features.
        faces (torch.LongTensor):
            face index, of shape :math:`(\text{num_faces}, \text{num_vertices})`.
    Returns:
        (torch.FloatTensor):
            the face features, of shape
            :math:`(\text{batch_size}, \text{num_faces}, \text{num_vertices}, \text{knum})`.
    """
    assert vertices_features.ndim == 3, \
        "vertices_features must have 3 dimensions of shape (batch_size, num_points, knum)"
    assert faces.ndim == 2, "faces must have 2 dimensions of shape (num_faces, num_vertices)"
    input = vertices_features.unsqueeze(2).expand(-1, -1, faces.shape[-1], -1)
    indices = faces[None, ..., None].expand(vertices_features.shape[0], -1, -1, vertices_features.shape[-1])
    return torch.gather(input=input, index=indices, dim=1)


def adjacency_matrix(num_vertices, faces, sparse=True):
    r"""Calculates a adjacency matrix of a mesh.

    Args:
        num_vertices (int): Number of vertices of the mesh.
        faces (torch.LongTensor):
            Faces of shape :math:`(\text{num_faces}, \text{face_size})` of the mesh.
        sparse (bool): Whether to return a sparse tensor or not. Default: True.

    Returns:
        (torch.FloatTensor or torch.sparse.FloatTensor): adjacency matrix

    Example:
        >>> faces = torch.tensor([[0, 1, 2]])
        >>> adjacency_matrix(3, faces)
        tensor(indices=tensor([[0, 0, 1, 1, 2, 2],
                               [1, 2, 0, 2, 0, 1]]),
               values=tensor([1., 1., 1., 1., 1., 1.]),
               size=(3, 3), nnz=6, layout=torch.sparse_coo)
    """
    device = faces.device

    forward_i = torch.stack([faces, torch.roll(faces, 1, dims=-1)], dim=-1)
    backward_i = torch.stack([torch.roll(faces, 1, dims=-1), faces], dim=-1)
    indices = torch.cat([forward_i, backward_i], dim=1).reshape(-1, 2)
    indices = indices.unique(dim=0)

    if sparse:
        indices = indices.t()
        # If vertex i and j have an edge connect to it, A[i, j] = 1
        values = torch.ones(indices.shape[1], device=device)
        adjacency = torch.sparse.FloatTensor(indices, values, (num_vertices, num_vertices))
    else:
        adjacency = torch.zeros((num_vertices, num_vertices), device=device, dtype=torch.float)
        adjacency[indices[:, 0], indices[:, 1]] = 1

    return adjacency

def uniform_laplacian(num_vertices, faces):
    r"""Calculates the uniform laplacian of a mesh.
    :math:`L[i, j] = \frac{1}{num\_neighbours(i)}` if i, j are neighbours.
    :math:`L[i, j] = -1` if i == j. 
    :math:`L[i, j] = 0` otherwise.

    Args:
        num_vertices (int): Number of vertices for the mesh.
        faces (torch.LongTensor):
            Faces of shape :math:`(\text{num_faces}, \text{face_size})` of the mesh.

    Returns:
        (torch.Tensor):
            Uniform laplacian of the mesh of size :math:`(\text{num_vertices}, \text{num_vertices})`
    Example:
        >>> faces = torch.tensor([[0, 1, 2]])
        >>> uniform_laplacian(3, faces)
        tensor([[-1.0000,  0.5000,  0.5000],
                [ 0.5000, -1.0000,  0.5000],
                [ 0.5000,  0.5000, -1.0000]])
    """
    batch_size = faces.shape[0]

    dense_adjacency = adjacency_matrix(num_vertices, faces).to_dense()

    # Compute the number of neighbours of each vertex
    num_neighbour = torch.sum(dense_adjacency, dim=1).view(-1, 1)

    L = torch.div(dense_adjacency, num_neighbour)

    torch.diagonal(L)[:] = -1

    # Fill NaN value with 0
    L[torch.isnan(L)] = 0

    return L


def compute_vertex_normals(faces, face_normals, num_vertices=None):
    r"""Computes normals for every vertex by averaging face normals
    assigned to that vertex for every face that has this vertex.

    Args:
       faces (torch.LongTensor): vertex indices of faces of a fixed-topology mesh batch with
            shape :math:`(\text{num_faces}, \text{face_size})`.
       face_normals (torch.FloatTensor): pre-normalized xyz normal values
            for every vertex of every face with shape
            :math:`(\text{batch_size}, \text{num_faces}, \text{face_size}, 3)`.
       num_vertices (int, optional): number of vertices V (set to max index in faces, if not set)

    Return:
        (torch.FloatTensor): of shape (B, V, 3)
    """
    if num_vertices is None:
        num_vertices = int(faces.max()) + 1

    B = face_normals.shape[0]
    V = num_vertices
    F = faces.shape[0]
    FSz = faces.shape[1]
    vertex_normals = torch.zeros((B, V, 3), dtype=face_normals.dtype, device=face_normals.device)
    counts = torch.zeros((B, V), dtype=face_normals.dtype, device=face_normals.device)

    faces = faces.unsqueeze(0).repeat(B, 1, 1)
    fake_counts = torch.ones((B, F), dtype=face_normals.dtype, device=face_normals.device)
    #              B x F          B x F x 3
    # self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
    # self[i][index[i][j][k]][k] += src[i][j][k]  # if dim == 1
    for i in range(FSz):
        vertex_normals.scatter_add_(1, faces[..., i:i + 1].repeat(1, 1, 3), face_normals[..., i, :])
        counts.scatter_add_(1, faces[..., i], fake_counts)

    counts = counts.clip(min=1).unsqueeze(-1)
    vertex_normals = vertex_normals / counts
    return vertex_normals
