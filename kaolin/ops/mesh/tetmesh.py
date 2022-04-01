# Copyright (c) 2021,22 NVIDIA CORPORATION & AFFILIATES.
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
from kaolin.ops.conversions.tetmesh import _sort_edges

base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long)


def _validate_tet_vertices(tet_vertices):
    r"""Helper method to validate the dimensions of the batched tetrahedrons tensor.

    Args:
        tet_vertices (torch.Tensor):
            Batched tetrahedrons, of shape
            :math:`(\text{batch_size}, \text{num_tetrahedrons}, 4, 3)`.
    """
    assert tet_vertices.ndim == 4, \
        f"tetrahedrons has {tetrahedrons.ndim} but must have 4 dimensions."
    assert tet_vertices.shape[2] == 4, \
        f"The third dimension of the tetrahedrons must be 4 " \
        f"but the input has {tetrahedrons.shape[2]}. Each tetrahedron has 4 vertices."
    assert tet_vertices.shape[3] == 3, \
        f"The fourth dimension of the tetrahedrons must be 3 " \
        f"but the input has {tetrahedrons.shape[3]}. Each vertex must have 3 dimensions."


def inverse_vertices_offset(tet_vertices):
    r"""Given tetrahedrons with 4 vertices A, B, C, D.
    Compute the inverse of the offset matrix w.r.t. vertex A for each
    tetrahedron. The offset matrix is obtained by the concatenation of :math:`B - A`,
    :math:`C - A` and :math:`D - A`. The resulting shape
    of the offset matrix is :math:`(\text{batch_size}, \text{num_tetrahedrons}, 3, 3)`.
    The inverse of the offset matrix is computed by this function.

    Args:
        tet_vertices (torch.Tensor):
            Batched tetrahedrons, of shape
            :math:`(\text{batch_size}, \text{num_tetrahedrons}, 4, 3)`.

    Returns:
        (torch.Tensor):
            Batched inverse offset matrix, of shape
            :math:`(\text{batch_size}, \text{num_tetrahedrons}, 3, 3)`.
            Each offset matrix is of shape :math:`(3, 3)`,
            hence its inverse is also of shape :math:`(3, 3)`.

    Example:
        >>> tet_vertices = torch.tensor([[[[-0.0500,  0.0000,  0.0500],
        ...                                [-0.0250, -0.0500,  0.0000],
        ...                                [ 0.0000,  0.0000,  0.0500],
        ...                                [0.5000, 0.5000, 0.4500]]]])
        >>> inverse_vertices_offset(tet_vertices)
        tensor([[[[   0.0000,   20.0000,    0.0000],
                  [  79.9999, -149.9999,   10.0000],
                  [ -99.9999,  159.9998,  -10.0000]]]])
    """
    _validate_tet_vertices(tet_vertices)

    # split the tensor
    A, B, C, D = torch.split(tet_vertices, split_size_or_sections=1, dim=2)

    # compute the offset matrix w.r.t. vertex A
    offset_matrix = torch.cat([B - A, C - A, D - A], dim=2)

    # compute the inverse of the offset matrix
    inverse_offset_matrix = torch.inverse(offset_matrix)

    return inverse_offset_matrix


def subdivide_tetmesh(vertices, tetrahedrons, features=None):
    r"""Subdivide each tetrahedron in tetmesh into 8 smaller tetrahedrons 
    by adding midpoints. If per-vertex features (e.g. SDF value) are given, the features
    of the new vertices are computed by averaging the features of vertices on the edge.
    For more details and example usage in learning, see 
    `Deep Marching Tetrahedra\: a Hybrid Representation for High-Resolution 3D Shape Synthesis`_ NeurIPS 2021.

    Args:
        vertices (torch.Tensor): batched vertices of tetrahedral meshes, of shape
                                 :math:`(\text{batch_size}, \text{num_vertices}, 3)`.
        tetrahedrons (torch.LongTensor): unbatched tetrahedral mesh topology, of shape
                              :math:`(\text{num_tetrahedrons}, 4)`.
        features (optional, torch.Tensor): batched per-vertex feature vectors, of shape
                            :math:`(\text{batch_size}, \text{num_vertices}, \text{feature_dim})`.

    Returns:
        (torch.Tensor, torch.LongTensor, (optional) torch.Tensor): 

        - batched vertices of subdivided tetrahedral meshes, of shape 
          :math:`(\text{batch_size}, \text{new_num_vertices}, 3)`
        - unbatched tetrahedral mesh topology, of shape 
          :math:`(\text{num_tetrahedrons} * 8, 4)`.
        - batched per-vertex feature vectors of subdivided tetrahedral meshes, of shape 
          :math:`(\text{batch_size}, \text{new_num_vertices}, \text{feature_dim})`.

    Example:
        >>> vertices = torch.tensor([[[0, 0, 0],
        ...               [1, 0, 0],
        ...               [0, 1, 0],
        ...               [0, 0, 1]]], dtype=torch.float)
        >>> tetrahedrons = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        >>> sdf = torch.tensor([[[-1.], [-1.], [0.5], [0.5]]], dtype=torch.float)
        >>> new_vertices, new_tetrahedrons, new_sdf = subdivide_tetmesh(vertices, tetrahedrons, sdf)
        >>> new_vertices
        tensor([[[0.0000, 0.0000, 0.0000],
                 [1.0000, 0.0000, 0.0000],
                 [0.0000, 1.0000, 0.0000],
                 [0.0000, 0.0000, 1.0000],
                 [0.5000, 0.0000, 0.0000],
                 [0.0000, 0.5000, 0.0000],
                 [0.0000, 0.0000, 0.5000],
                 [0.5000, 0.5000, 0.0000],
                 [0.5000, 0.0000, 0.5000],
                 [0.0000, 0.5000, 0.5000]]])
        >>> new_tetrahedrons
        tensor([[0, 4, 5, 6],
                [1, 7, 4, 8],
                [2, 5, 7, 9],
                [3, 6, 9, 8],
                [4, 5, 6, 8],
                [4, 5, 8, 7],
                [9, 5, 8, 6],
                [9, 5, 7, 8]])
        >>> new_sdf
        tensor([[[-1.0000],
                 [-1.0000],
                 [ 0.5000],
                 [ 0.5000],
                 [-1.0000],
                 [-0.2500],
                 [-0.2500],
                 [-0.2500],
                 [-0.2500],
                 [ 0.5000]]])

    .. _Deep Marching Tetrahedra\: a Hybrid Representation for High-Resolution 3D Shape Synthesis:
            https://arxiv.org/abs/2111.04276
    """

    device = vertices.device
    all_edges = tetrahedrons[:, base_tet_edges].reshape(-1, 2)
    all_edges = _sort_edges(all_edges)
    unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)
    idx_map = idx_map + vertices.shape[1]

    pos_feature = torch.cat([vertices, features], -1) if (features is not None) else vertices

    mid_pos_feature = pos_feature[:, unique_edges.reshape(-1)].reshape(
        pos_feature.shape[0], -1, 2, pos_feature.shape[-1]).mean(2)
    new_pos_feature = torch.cat([pos_feature, mid_pos_feature], 1)
    new_pos, new_features = new_pos_feature[..., :3], new_pos_feature[..., 3:]

    idx_a, idx_b, idx_c, idx_d = torch.split(tetrahedrons, 1, -1)
    idx_ab, idx_ac, idx_ad, idx_bc, idx_bd, idx_cd = idx_map.reshape(-1, 6).split(1, -1)

    tet_1 = torch.stack([idx_a, idx_ab, idx_ac, idx_ad], dim=1)
    tet_2 = torch.stack([idx_b, idx_bc, idx_ab, idx_bd], dim=1)
    tet_3 = torch.stack([idx_c, idx_ac, idx_bc, idx_cd], dim=1)
    tet_4 = torch.stack([idx_d, idx_ad, idx_cd, idx_bd], dim=1)
    tet_5 = torch.stack([idx_ab, idx_ac, idx_ad, idx_bd], dim=1)
    tet_6 = torch.stack([idx_ab, idx_ac, idx_bd, idx_bc], dim=1)
    tet_7 = torch.stack([idx_cd, idx_ac, idx_bd, idx_ad], dim=1)
    tet_8 = torch.stack([idx_cd, idx_ac, idx_bc, idx_bd], dim=1)

    new_tetrahedrons = torch.cat([tet_1, tet_2, tet_3, tet_4, tet_5, tet_6, tet_7, tet_8], dim=0).squeeze(-1)

    return (new_pos, new_tetrahedrons) if features is None else (new_pos, new_tetrahedrons, new_features)
