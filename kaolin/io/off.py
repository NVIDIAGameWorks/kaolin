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

from collections import namedtuple

import numpy as np
import torch


return_type = namedtuple('return_type',
                         ['vertices', 'faces', 'face_colors'])

def _get_faces(f, num_faces):
    faces = []
    for line in f:
        if line.startswith('#'):
            continue
        data = line.split()
        if len(data) > 0:
            face_size = int(data[0])
            faces.append([int(d) for d in data[1:face_size + 1]])
            if len(faces) == num_faces:
                break
    return torch.LongTensor(faces)

def _get_faces_with_colors(f, num_faces):
    faces = []
    face_colors = []
    for line in f:
        if line.startswith('#'):
            continue
        data = line.split()
        if len(data) > 0:
            face_size = int(data[0])
            faces.append([int(d) for d in data[1:face_size + 1]])
            face_colors.append([float(d) for d in data[face_size + 1:]])
            if len(faces) == num_faces:
                break
    faces = torch.LongTensor(faces)
    face_colors = torch.LongTensor(face_colors)
    return faces, face_colors

def import_mesh(path, with_face_colors=False):
    r"""Load data from an off file as a single mesh.

    Args:
        path (str): path to the obj file (with extension).
        with_face_colors (bool): if True, load face colors. Default: False.

    Returns:

    nametuple of:
        - **vertices** (torch.FloatTensor): of shape (num_vertices, 3)
        - **faces** (torch.LongTensor): of shape (num_faces, face_size)
        - **face_colors** (torch.LongTensor): in the range [0, 255], of shape (num_faces, 3).
    """
    vertices = []
    uvs = []
    f = open(path, 'r', encoding='utf-8')
    # Get metadata (number of vertices / faces (/ edges))
    for line in f:
        if line.startswith('#') or line.startswith('OFF'):
            continue
        data = line.split()
        if len(data) > 0:
            num_vertices = int(data[0])
            num_faces = int(data[1])
            break
    # Get vertices
    for line in f:
        if line.startswith('#'):
            continue
        data = line.split()
        if len(data) > 0:
            vertices.append([float(d) for d in data])
            if len(vertices) == num_vertices:
                break
    vertices = torch.FloatTensor(vertices)
    # Get faces
    if with_face_colors:
        faces, face_colors = _get_faces_with_colors(f, num_faces)
    else:
        faces = _get_faces(f, num_faces)
        face_colors = None
    f.close()
    return return_type(vertices, faces, face_colors)
