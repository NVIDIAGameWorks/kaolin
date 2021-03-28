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
            faces.append([int(d) for d in data[1:face_size]])
            if len(faces) == num_faces:
                break
    return torch.LongTensor(faces)

def _get_faces_with_color(f, num_faces):
    faces = []
    face_colors = []
    for line in f:
        if line.startswith('#'):
            continue
        data = line.split()
        if len(data) > 0:
            face_size = int(data[0])
            faces.append([int(d) for d in data[1:face_size]])
            face_colors.append([float(d) for d in data[face_size:]])
            if len(faces) == num_faces:
                break
    faces = torch.LongTensor(faces)
    face_colors = torch.FloatTensor(face_colors)
    return faces, face_colors

def import_mesh(path, with_face_colors=False, error_handler=None):
    r"""Load data from an off file as a single mesh.

    Args:
        path (str): path to the obj file (with extension).
        with_face_colors (bool): if True, load face colors. Default: False.
        error_handler (Callable):
            function that handle errors that may happen during file processing.
            Default: raise all errors.

    Returns:

    nametuple of:
        - **vertices** (torch.FloatTensor): of shape (num_vertices, 3)
        - **faces** (torch.LongTensor): of shape (num_faces, face_size)
        - **face_colors** (torch.FloatTensor): of shape (num_faces, 3)
    """
    if error_handler is None:
        error_handler = default_error_handler
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
            if len(data) == 3:
                num_edges = int(data[2])
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
    if with_faces_colors:
        faces, faces_colors = _get_faces_with_colors(f, num_faces)
    else:
		faces = _get_faces_with_colors(f, num_faces)
        faces_colors = None
    f.close()
	return return_type(vertices, faces, faces_colors)
