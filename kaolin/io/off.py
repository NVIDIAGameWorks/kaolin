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

from collections import namedtuple

import torch

__all__ = [
    'import_mesh'
]

return_type = namedtuple('return_type',
                         ['vertices', 'faces', 'face_colors'])

def _is_void(splitted_str):
    return len(splitted_str) == 0 or splitted_str[0].startswith('#')

def import_mesh(path, with_face_colors=False):
    r"""Load data from an off file as a single mesh.

    Args:
        path (str): path to the obj file (with extension).
        with_face_colors (bool): if True, load face colors. Default: False.

    Returns:
        (off.return_type):
            nametuple of:

            - **vertices** (torch.FloatTensor): of shape :math:`(\text{num_vertices}, 3)`.
            - **faces** (torch.LongTensor): of shape :math:`(\text{num_faces}, \text{face_size})`.
            - **face_colors** (torch.LongTensor):
              in the range :math:`[0, 255]`, of shape :math:`(\text{num_faces}, 3)`.
    """
    vertices = []
    uvs = []
    f = open(path, 'r', encoding='utf-8')
    # Get metadata (number of vertices / faces (/ edges))
    for line in f:
        data = line.split()
        if _is_void(data):
            continue
        if data[0].startswith('OFF'):
            # ModelNet40 have some OFFnum_vertices num_faces
            if len(data[0][3:]) > 0:
                num_vertices = int(data[0][3:])
                num_faces = int(data[1])
                break
            elif len(data) > 1:
                num_vertices = int(data[1])
                num_faces = int(data[2])
                break
            continue
        num_vertices = int(data[0])
        num_faces = int(data[1])
        break

    # Get vertices
    for line in f:
        data = line.split()
        if _is_void(data):
            continue
        vertices.append([float(d) for d in data[:3]])
        if len(vertices) == num_vertices:
            break
    vertices = torch.FloatTensor(vertices)

    # Get faces
    faces = []
    face_colors = []
    for line in f:
        data = line.split()
        if _is_void(data):
            continue
        face_size = int(data[0])
        faces.append([int(d) for d in data[1:face_size + 1]])
        if with_face_colors:
            face_colors.append([
                int(d) for d in data[face_size + 1:face_size + 4]
            ])
        if len(faces) == num_faces:
            break
    faces = torch.LongTensor(faces)
    if with_face_colors:
        face_colors = torch.LongTensor(face_colors)
    else:
        face_colors = None

    f.close()
    return return_type(vertices, faces, face_colors)
