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
from ..batch import tile_to_packed, packed_to_padded, get_first_idx

__all__ = [
    'face_areas',
    'packed_face_areas',
    'sample_points',
    'packed_sample_points',
    'face_normals',
]

def _base_face_areas(face_vertices_0, face_vertices_1, face_vertices_2):
    """Base function to compute the face areas."""
    x1, x2, x3 = torch.split(face_vertices_0 - face_vertices_1, 1, dim=-1)
    y1, y2, y3 = torch.split(face_vertices_1 - face_vertices_2, 1, dim=-1)

    a = (x2 * y3 - x3 * y2) ** 2
    b = (x3 * y1 - x1 * y3) ** 2
    c = (x1 * y2 - x2 * y1) ** 2
    areas = torch.sqrt(a + b + c) * 0.5

    return areas

def _base_sample_points_selected_faces(face_vertices_0, face_vertices_1, face_vertices_2):
    """Base function to sample points."""
    sampling_shape = tuple(int(d) for d in face_vertices_0.shape[:-1]) + (1,)
    # u is proximity to middle point between v1 and v2 against v0.
    # v is proximity to v2 against v1.
    #
    # The probability density for u should be f_U(u) = 2u.
    # However, torch.rand use a uniform (f_X(x) = x) distribution,
    # so using torch.sqrt we make a change of variable to have the desired density
    # f_Y(y) = f_X(y ^ 2) * |d(y ^ 2) / dy| = 2y 
    u = torch.sqrt(torch.rand(sampling_shape,
                              device=face_vertices_0.device,
                              dtype=face_vertices_0.dtype))

    v = torch.rand(sampling_shape,
                   device=face_vertices_0.device,
                   dtype=face_vertices_0.dtype)

    points = (1 - u) * face_vertices_0 + \
        (u * (1 - v)) * face_vertices_1 + \
        u * v * face_vertices_2

    return points

def face_areas(vertices, faces):
    """Compute the areas of each face of triangle meshes.

    Args:
        vertices (torch.Tensor):
            The vertices of the meshes,
            of shape :math:`(\\text{batch_size}, \\text{num_vertices}, 3)`.
        faces (torch.LongTensor):
            the faces of the meshes, of shape :math:`(\\text{num_faces}, 3)`.

    Returns:
        (torch.Tensor):
            the face areas of same type as vertices and of shape
            :math:`(\\text{batch_size}, \\text{num_faces})`.
    """
    if faces.shape[-1] != 3:
        raise NotImplementedError("face_areas is only implemented for triangle meshes")
    faces_0, faces_1, faces_2 = torch.split(faces, 1, dim=1)
    face_v_0 = torch.index_select(vertices, 1, faces_0.reshape(-1))
    face_v_1 = torch.index_select(vertices, 1, faces_1.reshape(-1))
    face_v_2 = torch.index_select(vertices, 1, faces_2.reshape(-1))

    areas = _base_face_areas(face_v_0, face_v_1, face_v_2)

    return areas.squeeze(-1)

def packed_face_areas(vertices, first_idx_vertices, faces, num_faces_per_mesh):
    """Compute the areas of each face of triangle meshes.

    Args:
        vertices (torch.Tensor):
            The packed vertices of the meshes, of shape :math:`(\\text{num_vertices}, 3)`.
        first_idx_vertices (torch.Tensor):
            The :ref:`first_idx<packed_first_idx>` associated to vertices,
            of shape :math:`(\\text{batch_size})`.
        faces (torch.LongTensor):
            The packed faces of the meshes, of shape :math:`(\\text{num_faces}, 3)`.
        num_faces_per_mesh:
            The number of faces per mesh, of shape :math:`(\\text{batch_size})`.

    Returns:
        (torch.Tensor):
            The face areas of same type as vertices
            and of shape :math:`(\\text{num_faces})`.
    """
    if faces.shape[-1] != 3:
        raise NotImplementedError("packed_face_areas is only implemented for triangle meshes")
    merged_faces = tile_to_packed(first_idx_vertices[:-1].to(vertices.device),
                                  num_faces_per_mesh) + faces

    faces_0, faces_1, faces_2 = torch.split(merged_faces, 1, dim=1)
    face_v_0 = torch.index_select(vertices, 0, faces_0.reshape(-1))
    face_v_1 = torch.index_select(vertices, 0, faces_1.reshape(-1))
    face_v_2 = torch.index_select(vertices, 0, faces_2.reshape(-1))

    areas = _base_face_areas(face_v_0, face_v_1, face_v_2)

    return areas.view(-1)


def sample_points(vertices, faces, num_samples, areas=None):
    """Uniformly sample points over the surface of triangle meshes.

    First face on which the point is sampled is randomly selected,
    with the probability of selection being proportional to the area of the face.
    then the coordinate on the face is uniformly sampled.

    Args:
        vertices (torch.Tensor):
            The vertices of the meshes, of shape :math:`(\\text{batch_size}, \\text{num_vertices}, 3)`.
        faces (torch.LongTensor):
            The faces of the mesh, of shape :math:`(\\text{num_faces}, 3)`.
        num_samples (int):
            The number of point sampled per mesh.
        areas (torch.Tensor, optional):
            The areas of each face, of shape :math:`(\\text{batch_size}, \\text{num_faces})`,
            can be preprocessed, for fast on-the-fly sampling,
            will be computed if None (default).
    Returns:
        (torch.Tensor, torch.LongTensor):
            the pointclouds of shape :math:`(\\text{batch_size}, \\text{num_points}, 3)`,
            and the indexes of the faces selected, of shape :math:`(\\text{batch_size}, \\text{num_points})`.
    """
    if faces.shape[-1] != 3:
        raise NotImplementedError("sample_points is only implemented for triangle meshes")
    faces_0, faces_1, faces_2 = torch.split(faces, 1, dim=1)
    face_v_0 = torch.index_select(vertices, 1, faces_0.reshape(-1))
    face_v_1 = torch.index_select(vertices, 1, faces_1.reshape(-1))
    face_v_2 = torch.index_select(vertices, 1, faces_2.reshape(-1))

    if areas is None:
        areas = _base_face_areas(face_v_0, face_v_1, face_v_2).squeeze(-1)
    face_dist = torch.distributions.Categorical(areas)
    face_choices = face_dist.sample([num_samples]).transpose(0, 1)
    _face_choices = face_choices.unsqueeze(-1).repeat(1, 1, 3)
    v0 = torch.gather(face_v_0, 1, _face_choices)
    v1 = torch.gather(face_v_1, 1, _face_choices)
    v2 = torch.gather(face_v_2, 1, _face_choices)

    points = _base_sample_points_selected_faces(v0, v1, v2)

    return points, face_choices

# TODO(cfujitsang): packed_sample_points can return a packed if `num_samples` is an an iterable
def packed_sample_points(vertices, first_idx_vertices,
                         faces, num_faces_per_mesh, num_samples, areas=None):
    """Uniformly sample points over the surface of triangle meshes.

    First face on which the point is sampled is randomly selected,
    with the probability of selection being proportional to the area of the face.
    then the coordinate on the face is uniformly sampled.

    The return pointclouds are with fixed batching.

    Args:
        vertices (torch.Tensor):
            The vertices of the meshes, of shape ``(num_vertices, 3)``
        faces (torch.LongTensor):
            The faces of the mesh, of shape ``(num_faces, 3)``

    Returns:
        (torch.Tensor, torch.LongTensor):
            the pointclouds of shape ``(batch_size, num_points, 3)``,
            and the indexes of the faces selected (as merged faces),
            of shape ``(batch_size, num_points)``
    """
    if faces.shape[-1] != 3:
        raise NotImplementedError("packed_sample_points is only implemented for triangle meshes")
    batch_size = num_faces_per_mesh.shape[0]
    merged_faces = tile_to_packed(first_idx_vertices[:-1].to(vertices.device),
                                  num_faces_per_mesh) + faces

    faces_0, faces_1, faces_2 = torch.split(merged_faces, 1, dim=1)
    face_v_0 = torch.index_select(vertices, 0, faces_0.reshape(-1))
    face_v_1 = torch.index_select(vertices, 0, faces_1.reshape(-1))
    face_v_2 = torch.index_select(vertices, 0, faces_2.reshape(-1))

    if areas is None:
        areas = _base_face_areas(face_v_0, face_v_1, face_v_2).squeeze(-1)
    # TODO(cfujitsang): this is kind of cheating, we should try to avoid padding on packed ops
    # But is works well since setting 0. padding leads to 0. probability to be picked,
    first_idx_faces = get_first_idx(num_faces_per_mesh)
    areas = packed_to_padded(areas.reshape(-1, 1),
                             num_faces_per_mesh.reshape(-1, 1),
                             first_idx_faces,
                             0.).squeeze(-1)
    face_dist = torch.distributions.Categorical(areas)
    face_choices = face_dist.sample([num_samples]).transpose(0, 1)
    # since face_v_X are still packed, we need to merged meshes indexes
    merged_face_choices = \
        (face_choices + first_idx_faces[:-1].reshape(-1, 1).to(face_choices.device)).reshape(-1)
    v0 = torch.index_select(face_v_0, 0, merged_face_choices).reshape(batch_size, num_samples, 3)
    v1 = torch.index_select(face_v_1, 0, merged_face_choices).reshape(batch_size, num_samples, 3)
    v2 = torch.index_select(face_v_2, 0, merged_face_choices).reshape(batch_size, num_samples, 3)

    points = _base_sample_points_selected_faces(v0, v1, v2)

    return points, merged_face_choices.reshape(batch_size, num_samples)

def face_normals(face_vertices, unit=False):
    r"""Calculate normals of triangle meshes.

        Args:
            face_vertices (torch.Tensor):
                3D points in camera coordinate,
                of shape :math:`(\text{batch_size}, \text{num_faces}, 3, 3)`,
                9 means 3 triangle vertices and each contains xyz.
            unit (bool):
                if true, return normals as unit vectors, default is False
        Returns:
            (torch.FloatTensor):
                face normals, of shape :math:`(\text{batch_size}, \text{num_faces}, 3)`
        """
    if face_vertices.shape[-2] != 3:
        raise NotImplementedError("face_normals is only implemented for triangle meshes")
    # Note: Here instead of using the normals from vertexlist2facelist we compute it from scratch
    edges_dist0 = face_vertices[:, :, 1] - face_vertices[:, :, 0]
    edges_dist1 = face_vertices[:, :, 2] - face_vertices[:, :, 0]
    face_normals = torch.cross(edges_dist0, edges_dist1, dim=2)

    if unit:
        face_normals_length = face_normals.norm(dim=2, keepdim=True)
        face_normals = face_normals / (face_normals_length + 1e-10)

    return face_normals

def _unbatched_subdivide_vertices(vertices, faces, resolution):
    r"""Subdivide the triangle mesh's vertices so that every existing edge's length is shorter
    or equal to :math:`(\frac{resolution - 1}{(resolution^2)})^2`.

    It creates a new vertex in the middle of an existing edge, 
    if the length of the edge is larger than :math:`(\frac{resolution - 1}{(resolution^2)})^2`.
    Note: it does not add faces between newly added vertices.
    It only addes new vertices. This function is mainly used in 
    :py:meth:`kaolin.ops.conversions.trianglemesh.trianglemesh_to_voxelgrid`.

    Args:
        vertices (torch.tensor): unbatched vertices of shape (V, 3) of mesh.
        faces (torch.LongTensor): unbatched faces of shape (F, 3) of mesh.
        resolution (int): target resolution to upsample to.

    Returns:
        (torch.Tensor): upsampled vertices.

    Example:
        >>> vertices = torch.tensor([[0, 0, 0],
        ...                          [1, 0, 0],
        ...                          [0, 0, 1]], dtype=torch.float)
        >>> faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
        >>> _unbatched_subdivide_vertices(vertices, faces, 2)
        tensor([[0.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, 0.1250],
                [0.0000, 0.0000, 0.2500],
                [0.0000, 0.0000, 0.3750],
                [0.0000, 0.0000, 0.5000],
                [0.0000, 0.0000, 0.6250],
                [0.0000, 0.0000, 0.7500],
                [0.0000, 0.0000, 0.8750],
                [0.0000, 0.0000, 1.0000],
                [0.1250, 0.0000, 0.0000],
                [0.1250, 0.0000, 0.1250],
                [0.1250, 0.0000, 0.2500],
                [0.1250, 0.0000, 0.3750],
                [0.1250, 0.0000, 0.5000],
                [0.1250, 0.0000, 0.6250],
                [0.1250, 0.0000, 0.7500],
                [0.1250, 0.0000, 0.8750],
                [0.2500, 0.0000, 0.0000],
                [0.2500, 0.0000, 0.1250],
                [0.2500, 0.0000, 0.2500],
                [0.2500, 0.0000, 0.3750],
                [0.2500, 0.0000, 0.5000],
                [0.2500, 0.0000, 0.6250],
                [0.2500, 0.0000, 0.7500],
                [0.3750, 0.0000, 0.0000],
                [0.3750, 0.0000, 0.1250],
                [0.3750, 0.0000, 0.2500],
                [0.3750, 0.0000, 0.3750],
                [0.3750, 0.0000, 0.5000],
                [0.3750, 0.0000, 0.6250],
                [0.5000, 0.0000, 0.0000],
                [0.5000, 0.0000, 0.1250],
                [0.5000, 0.0000, 0.2500],
                [0.5000, 0.0000, 0.3750],
                [0.5000, 0.0000, 0.5000],
                [0.6250, 0.0000, 0.0000],
                [0.6250, 0.0000, 0.1250],
                [0.6250, 0.0000, 0.2500],
                [0.6250, 0.0000, 0.3750],
                [0.7500, 0.0000, 0.0000],
                [0.7500, 0.0000, 0.1250],
                [0.7500, 0.0000, 0.2500],
                [0.8750, 0.0000, 0.0000],
                [0.8750, 0.0000, 0.1250],
                [1.0000, 0.0000, 0.0000]])
    """
    device = vertices.device

    assert resolution > 1
    min_edge_length = ((resolution - 1) / (resolution ** 2))**2

    v1 = torch.index_select(vertices, 0, faces[:, 0])  # shape of (B, F, 3)
    v2 = torch.index_select(vertices, 0, faces[:, 1])
    v3 = torch.index_select(vertices, 0, faces[:, 2])

    while True:
        edge1_length = torch.sum((v1 - v2)**2, dim=1).unsqueeze(1)  # shape (B, F, 1)
        edge2_length = torch.sum((v2 - v3)**2, dim=1).unsqueeze(1)
        edge3_length = torch.sum((v3 - v1)**2, dim=1).unsqueeze(1)

        total_edges_length = torch.cat((edge1_length, edge2_length, edge3_length), dim=1)
        max_edges_length = torch.max(total_edges_length, dim=1)[0]

        # Choose the edges that is greater than the min_edge_length
        keep = max_edges_length > min_edge_length

        # if all the edges are smaller than the min_edge_length, stop upsampling
        K = torch.sum(keep)

        if K == 0:
            break

        V = vertices.shape[0]

        v1 = v1[keep]  # shape of (K, 3), where K is number of edges that has been kept
        v2 = v2[keep]
        v3 = v3[keep]

        # New vertices is placed at the middle of the edge
        v4 = (v1 + v3) / 2  # shape of (K, 3), where K is number of edges that has been kept
        v5 = (v1 + v2) / 2
        v6 = (v2 + v3) / 2

        # update vertices
        vertices = torch.cat((vertices, v4, v5, v6))

        # Get rid of repeated vertices
        vertices, unique_indices = torch.unique(vertices, return_inverse=True, dim=0)

        # Update v1, v2, v3
        v1 = torch.cat((v1, v2, v4, v3))
        v2 = torch.cat((v4, v5, v5, v4))
        v3 = torch.cat((v5, v6, v6, v6))

    return vertices
