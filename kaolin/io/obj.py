# Copyright (c) 2019,20-21-22 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import warnings
from collections import namedtuple

import numpy as np
import torch
from PIL import Image

from kaolin.io.materials import MaterialLoadError, MaterialFileError, MaterialNotFoundError
from kaolin.io import utils

__all__ = [
    'ignore_error_handler',
    'skip_error_handler',
    'default_error_handler',
    'import_mesh'
]

return_type = namedtuple('return_type',
                         ['vertices', 'faces', 'uvs', 'face_uvs_idx', 'materials',
                          'materials_order', 'vertex_normals', 'face_normals'])

def ignore_error_handler(error, **kwargs):
    """Simple error handler to use in :func:`load_obj` that ignore all errors"""
    pass


def skip_error_handler(error, **kwargs):
    """Simple error handler to use in :func:`load_obj` that skips all errors
    and logs them as warnings."""
    warnings.warn(error.args[0], UserWarning)


def default_error_handler(error, **kwargs):
    """Simple error handle to use in :func:`load_obj` that raises all errors."""
    raise error

def flatten_feature(feature):
    """Flatten the nested list of a feature.
    """
    if feature is None or len(feature) == 0:
        return None
    else:
        return [item for sublist in feature for item in sublist]


# TODO(cfujitsang): support https://en.wikipedia.org/wiki/Wavefront_.obj_file#Geometric_vertex ?
def import_mesh(path, with_materials=False, with_normals=False,
                error_handler=None, heterogeneous_mesh_handler=None):
    r"""Load data from an obj file as a single mesh.

    With limited materials support to Kd, Ka, Ks, map_Kd, map_Ka and map_Ks.
    Followed format described in: http://paulbourke.net/dataformats/obj/

    Args:
        path (str): path to the obj file (with extension).
        with_materials (bool): if True, load materials. Default: False.
        with_normals (bool): if True, load vertex normals. Default: False.
        error_handler (Callable, optional):
            function that handle errors that may happen during file processing,
            with following signature:
            ``error_handler(error: Exception, material_name: Optional[str],
            materials: Optional[list[dict]], materials_order: Optional[list])``.
            Default: raise all errors.
        heterogeneous_mesh_handler (Callable, optional):
            function that handles the import of heterogeneous mesh,
            with following signature:
            ``heterogeneous_mesh_handler(vertices, face_vertex_counts, *args)``
        Default: Heterogenenous mesh will raise a NonHomogeneousError.

    Returns:
        (obj.return_type):
            nametuple of:

            - **vertices** (torch.Tensor): of shape :math:`(\text{num_vertices}, 3)`.
            - **faces** (torch.LongTensor):
              of shape :math:`(\text{num_faces}, \text{face_size})`.
            - **uvs** (torch.Tensor): of shape :math:`(\text{num_uvs}, 2)`.
            - **face_uvs_idx** (torch.LongTensor):
              of shape :math:`(\text{num_faces}, \text{face_size})`.
            - **materials** (list of dict):
              a list of materials (see return values of :func:`load_mtl`).
            - **materials_order** (torch.LongTensor):
              of shape :math:`(\text{num_same_material_groups}, 2)`.
              showing the order in which materials are used over **face_uvs_idx**
              and the first indices in which they start to be used.
              A material can be used multiple times.
            - **vertex_normals** (torch.Tensor): of shape :math:`(\text{num_vertices}, 3)`.
            - **face_normals** (torch.LongTensor):
              of shape :math:`(\text{num_faces}, \text{face_size})`.

    Raises:
        MaterialNotFoundError:
            The .obj is using a material that haven't be found in the material files
        MaterialLoadError:
            From :func:`load_mtl`: Failed to load material,
            very often due to path to map_Kd/map_Ka/map_ks being invalid.
    """
    if error_handler is None:
        error_handler = default_error_handler
    vertices = []
    faces = []
    uvs = []
    # 3 values per face
    face_uvs_idx = []
    vertex_normals = []
    # 3 values per face
    face_normals = []
    # textures = []
    mtl_path = None
    materials_order = []
    materials_dict = {}
    materials_idx = {}

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data = line.split()
            if len(data) == 0:
                continue
            if data[0] == 'v':
                vertices.append(data[1:])
            elif with_materials and data[0] == 'vt':
                uvs.append(data[1:3])
            elif with_normals and data[0] == 'vn':
                vertex_normals.append(data[1:])
            elif data[0] == 'f':
                data = [da.split('/') for da in data[1:]]
                faces.append([int(d[0]) for d in data])
                if with_materials:
                    if len(data[1]) > 1 and data[1][1] != '':
                        face_uvs_idx.append([int(d[1]) for d in data])
                    else:
                        face_uvs_idx.append([0] * len(data))
                if with_normals:
                    if len(data[1]) > 2:
                        face_normals.append([int(d[2]) for d in data])
                    else:
                        face_normals.append([0] * len(data))
            elif with_materials and data[0] == 'usemtl':
                material_name = data[1]
                if material_name not in materials_idx:
                    materials_idx[material_name] = len(materials_idx)
                materials_order.append([materials_idx[material_name], len(face_uvs_idx)])
            elif with_materials and data[0] == 'mtllib':
                mtl_path = os.path.join(os.path.dirname(path), data[1])
                materials_dict.update(load_mtl(mtl_path, error_handler))

    # building materials in right order
    materials = [{} for i in materials_idx]
    for material_name, idx in materials_idx.items():
        if material_name not in materials_dict:
            error_handler(
                MaterialNotFoundError(f"'{material_name}' not found."),
                material_name=material_name, idx=idx,
                materials=materials, materials_order=materials_order)
        else:
            materials[idx] = materials_dict[material_name]

    vertices = torch.FloatTensor([float(el) for sublist in vertices
                                  for el in sublist]).view(-1, 3)

    face_vertex_counts = torch.IntTensor([len(f) for f in faces])

    if not torch.all(face_vertex_counts == face_vertex_counts[0]):
        if heterogeneous_mesh_handler is None:
            raise utils.NonHomogeneousMeshError(f'Mesh is non-homogeneous '
                                                f'and cannot be imported from {path}.'
                                                f'User can set heterogeneous_mesh_handler.'
                                                f'See kaolin.io.utils for the available options')
        else:
            all_features = [faces, face_uvs_idx, face_normals]
            # Flatten all features
            all_features = [flatten_feature(f) for f in all_features]

            mesh = heterogeneous_mesh_handler(vertices, face_vertex_counts, *all_features)
        if mesh is not None:
            vertices, face_vertex_counts, faces, face_uvs_idx, face_normals = mesh

    faces = torch.LongTensor(faces) - 1

    if with_materials:
        uvs = torch.FloatTensor([float(el) for sublist in uvs
                                 for el in sublist]).view(-1, 2)
        face_uvs_idx = torch.LongTensor(face_uvs_idx) - 1
        materials_order = torch.LongTensor(materials_order)
    else:
        uvs = None
        face_uvs_idx = None
        materials = None
        materials_order = None

    if with_normals:
        vertex_normals = torch.FloatTensor(
            [float(el) for sublist in vertex_normals
             for el in sublist]).view(-1, 3)
        face_normals = torch.LongTensor(face_normals) - 1
    else:
        vertex_normals = None
        face_normals = None

    return return_type(vertices, faces, uvs, face_uvs_idx, materials,
                       materials_order, vertex_normals, face_normals)


def load_mtl(mtl_path, error_handler):
    """Load and parse a Material file.

    Followed format described in: https://people.sc.fsu.edu/~jburkardt/data/mtl/mtl.html.
    Currently only support diffuse, ambient and specular parameters (Kd, Ka, Ks)
    through single RGB values or texture maps.

    Args:
        mtl_path (str): Path to the mtl file.

    Returns:
        (dict):
            Dictionary of materials, which are dictionary of properties with optional torch.Tensor values:

            - **Kd**: diffuse color of shape (3)
            - **map_Kd**: diffuse texture map of shape (H, W, 3)
            - **Ks**: specular color of shape (3)
            - **map_Ks**: specular texture map of shape (H', W', 3)
            - **Ka**: ambient color of shape (3)
            - **map_Ka**: ambient texture map of shape (H'', W'', 3)

    Raises:
        MaterialLoadError:
            Failed to load material, very often due to path to map_Kd/map_Ka/map_Ks being invalid.
    """
    mtl_data = {}
    root_dir = os.path.dirname(mtl_path)

    try:
        f = open(mtl_path, 'r', encoding='utf-8')
    except Exception as e:
        error_handler(MaterialFileError(
            f"Failed to load material at path '{mtl_path}':\n{e}"),
            mtl_path=mtl_path, mtl_data=mtl_data)
    else:
        for line in f.readlines():
            data = line.split()
            if len(data) == 0:
                continue
            try:
                if data[0] == 'newmtl':
                    material_name = data[1]
                    mtl_data[material_name] = {}
                elif data[0] in {'map_Kd', 'map_Ka', 'map_Ks'}:
                    texture_path = os.path.join(root_dir, data[1])
                    img = Image.open(texture_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    mtl_data[material_name][data[0]] = torch.from_numpy(
                        np.array(img))
                elif data[0] in {'Kd', 'Ka', 'Ks'}:
                    mtl_data[material_name][data[0]] = torch.tensor(
                        [float(val) for val in data[1:]])
            except Exception as e:
                error_handler(MaterialLoadError(
                    f"Failed to load material at path '{mtl_path}':\n{e}"),
                    data=data, mtl_data=mtl_data)
        f.close()
    return mtl_data
