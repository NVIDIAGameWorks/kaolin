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

__all__ = ["heterogeneous_mesh_handler_skip",
           "heterogeneous_mesh_handler_empty",
           'heterogeneous_mesh_handler_naive_homogenize',
           "NonHomogeneousMeshError"]

class NonHomogeneousMeshError(Exception):
    """Raised when expecting a homogeneous mesh but a heterogenous
    mesh is encountered.
    """

    __slots__ = ['message']

    def __init__(self, message):
        self.message = message

# Mesh Functions
# TODO(jlafleche) Support mesh subgroups for materials
def heterogeneous_mesh_handler_skip(*args):
    r"""Skip heterogeneous meshes."""
    return None

def heterogeneous_mesh_handler_empty(*args):
    """Return empty tensors for vertices and faces of heterogeneous meshes."""
    return (torch.FloatTensor(size=(0, 3)), torch.LongTensor(size=(0,)),
            torch.LongTensor(size=(0, 3)), torch.FloatTensor(size=(0, 2)),
            torch.LongTensor(size=(0, 3)), torch.FloatTensor(size=(0, 3, 3)),
            torch.LongTensor(size=(0,)))

def heterogeneous_mesh_handler_naive_homogenize(vertices, face_vertex_counts, *features):
    r"""Homogenize list of faces containing polygons of varying number of edges to triangles using fan
    triangulation.

    Args:
        vertices (torch.FloatTensor): Vertices with shape ``(N, 3)``.
        face_vertex_counts (torch.LongTensor): Number of vertices for each face with shape ``(M)``
            for ``M`` faces.
        *features: Variable length features that need to be handled. For example, faces and uvs.

    Returns:
        (list of torch.tensor): Homogeneous list of attributes.
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

    new_attrs = [_homogenize(a, face_vertex_counts) for a in features]
    new_counts = torch.ones(vertices.size(0), dtype=torch.long).fill_(3)
    return (vertices, new_counts, *new_attrs)
