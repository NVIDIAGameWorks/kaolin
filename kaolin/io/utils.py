# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import warnings

__all__ = ["heterogeneous_mesh_handler_skip",
           "heterogeneous_mesh_handler_naive_homogenize",
           "mesh_handler_naive_triangulate",
           "NonHomogeneousMeshError"]


class NonHomogeneousMeshError(Exception):
    """Raised when expecting a homogeneous mesh but a heterogenous
    mesh is encountered.
    """

    __slots__ = ['message']

    def __init__(self, message):
        self.message = message


# Mesh Functions
def heterogeneous_mesh_handler_skip(*args, **kwargs):
    r"""Skip heterogeneous meshes."""
    return None


def heterogeneous_mesh_handler_naive_homogenize(*args, **kwargs):
    r"""Same as :func:`mesh_handler_naive_triangulate`, see docs.
    .. deprecated:: 0.14.0
    """
    warnings.warn("heterogeneous_mesh_handler_naive_homogenize is deprecated, "
                  "please use kaolin.io.utils.mesh_handler_naive_triangulate instead",
                  DeprecationWarning, stacklevel=2)
    return mesh_handler_naive_triangulate(*args, **kwargs)


def mesh_handler_naive_triangulate(vertices, face_vertex_counts, *features, face_assignments=None):
    r"""Triangulate a list of faces containing polygons of varying number of edges using naive fan
    triangulation.

    Args:
        vertices (torch.FloatTensor): Vertices with shape ``(N, 3)``.
        face_vertex_counts (torch.LongTensor): Number of vertices for each face with shape ``(M)``
            for ``M`` faces.
        features: Variable length features that need to be handled as 1D Tensor ``(num_face_vertices)``,
            with one feature per face vertex. For example, faces as a tensor
            ``[face0_vertex0_id, face0_vertex1_id, face0_vertex2_id, face1_vertex0_id...]`` or as UV indices:
            ``[face0_vertex0_uv_idx, face0_vertex1_uv_idx, ...]``.
        face_assignments (dict): mapping from key to torch.LongTensor, where each value of the tensor corresponds
            to a face index. These indices will be expanded and rewritten to include triangulated face indices.
            Two modes are supported for face_assignments:
            1) if 1D tensor, each face idx will be replaced with indices of faces it was split into
            2) if 2D tensor, expects shape (K, 2), where [x, i] will be replaced with index of the first face
            [x, i] was split into, effectively supporting tensors containing (start,end].
    Returns:
        (tuple):
            Homogeneous list of attributes with exactly same type and number as function inputs.

            - **vertices** (torch.Tensor): unchanged `vertices` of shape ``(N, 3)``
            - **face_vertex_counts** (torch.LongTensor): tensor of length ``new_num_faces`` filled with 3.
            - **features** (torch.Tensor): of same type as input and shape ``(new_num_faces, 3)``
            - **face_assignments** (dict): returned only if face_assignments is set, with each value containing
                    new face indices equivalent to the prior assignments (see two modes for ``face_assignments``)
    """
    def _homogenize(attr, face_vertex_counts):
        if attr is not None:
            attr = attr if isinstance(attr, list) else attr.tolist()
            idx = 0
            new_attr = []
            for face_vertex_count in face_vertex_counts:
                attr_face = attr[idx:(idx + face_vertex_count)]
                idx += face_vertex_count
                while len(attr_face) >= 3:
                    new_attr.append(attr_face[:3])
                    attr_face.pop(1)
            return torch.tensor(new_attr)
        else:
            return None

    def _homogenize_counts(face_vertex_counts, compute_face_id_mappings=False):
        mappings = []  # mappings[i] = [new face ids that i was split into]
        num_faces = 0
        for face_vertex_count in face_vertex_counts:
            attr_face = list(range(0, face_vertex_count))
            new_indices = []
            while len(attr_face) >= 3:
                if compute_face_id_mappings:
                    new_indices.append(num_faces)
                num_faces += 1
                attr_face.pop(1)
            if compute_face_id_mappings:
                mappings.append(new_indices)
        return torch.full((num_faces,), 3, dtype=torch.long), mappings

    new_attrs = [_homogenize(a, face_vertex_counts) for a in features]
    new_counts, face_idx_mappings = _homogenize_counts(face_vertex_counts,
                                                       face_assignments is not None and len(face_assignments) > 0)

    if face_assignments is None:
        # Note: for python > 3.8 can do "return vertices, new_counts, *new_attrs"
        return tuple([vertices, new_counts] + new_attrs)

    # TODO: this is inefficient and could be improved
    new_assignments = {}
    for k, v in face_assignments.items():
        if len(v.shape) == 1:
            new_idx = []
            for old_idx in v:
                new_idx.extend(face_idx_mappings[old_idx])
            new_idx = torch.LongTensor(new_idx)
        else:
            # We support this (start, end] mode for efficiency of OBJ readers
            assert len(v.shape) == 2 and v.shape[1] == 2, 'Expects shape (K,) or (K, 2) for face_assignments'
            new_idx = torch.zeros_like(v)
            for row in range(v.shape[0]):
                old_idx_start = v[row, 0]
                old_idx_end = v[row, 1] - 1
                new_idx[row, 0] = face_idx_mappings[old_idx_start][0]
                new_idx[row, 1] = face_idx_mappings[old_idx_end][-1] + 1
        new_assignments[k] = new_idx

    # Note: for python > 3.8 can do "return vertices, new_counts, *new_attrs, new_assignments"
    return tuple([vertices, new_counts] + new_attrs + [new_assignments])


