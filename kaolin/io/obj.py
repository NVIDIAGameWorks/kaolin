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

from kaolin.io.materials import MaterialLoadError, MaterialFileError, MaterialNotFoundError, \
    process_materials_and_assignments
from kaolin.io import utils

__all__ = [
    'ignore_error_handler',
    'skip_error_handler',
    'create_missing_materials_error_handler',
    'default_error_handler',
    'import_mesh'
]

return_type = namedtuple('return_type',
                         ['vertices', 'faces', 'uvs', 'face_uvs_idx', 'materials',
                          'material_assignments', 'normals', 'face_normals_idx'])


def ignore_error_handler(error, **kwargs):
    """Simple error handler to use in :func:`load_obj` that ignore all errors"""
    pass


def skip_error_handler(error, **kwargs):
    """Simple error handler to use in :func:`load_obj` that skips all errors
    and logs them as warnings."""
    warnings.warn(error.args[0], UserWarning)


def create_missing_materials_error_handler(error, **kwargs):
    """Error error_handler to be provided to obj.read_mesh that can handle MaterialNotFound error,
    returning a dummy material with a random diffuse color instead. Material will contain
    an additional "error" field. MaterialFileError and MaterialLoadError will print a warning
    and be ignored."""
    if type(error) == MaterialNotFoundError:
        warnings.warn(f'{error.args[0]}, creating dummy material instead', UserWarning)
        return {'Ka': torch.rand((3,)), 'error': f'Dummy material created for missing material: {error}'}
    elif type(error) in [MaterialFileError, MaterialLoadError]:
        warnings.warn(error.args[0], UserWarning)
    else:
        raise error


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
                error_handler=None, heterogeneous_mesh_handler=None,
                triangulate=False):
    r"""Load data from an obj file as a single mesh.

    With limited materials support to Kd, Ka, Ks, map_Kd, map_Ka and map_Ks.
    Followed format described in: http://paulbourke.net/dataformats/obj/

    Args:
        path (str): path to the obj file (with extension).
        with_materials (bool): if True, load materials. Default: False.
        with_normals (bool): if True, load vertex normals. Default: False.
        error_handler (Callable, optional):
            function that handles errors that can be raised (see raised errors, except `NonHomogeneousMeshError`
            handled separately), with the signature ``error_handler(error: Exception, **kwargs)``.
            Handler can provide special treatment of :class:`MaterialNotFoundError`,
            returning a dummy material dictionary instead (if this is not the case, assignments to
            non-existent materials will be lost). For options see:
            :func:`create_missing_materials_error_handler`, :func:`skip_error_handler`, :func:`ignore_error_handler`,
            and :func:`default_error_handler` (**Default** is to raise all errors).
        heterogeneous_mesh_handler (Callable, optional):
            function that handles a heterogeneous mesh, homogenizing, returning None or throwing error,
            with the following signature:
            ``heterogeneous_mesh_handler(vertices, face_vertex_counts, *args, face_assignments)``
            for example, see :func:`mesh_handler_naive_triangulate <kaolin.io.utils.mesh_handler_naive_triangulate>`
            and :func:`heterogeneous_mesh_handler_skip <kaolin.io.utils.heterogeneous_mesh_handler_skip>`.
            Default: will raise a NonHomogeneousMeshError.
        triangulate: if True, will triangulate all non-triangular meshes using same logic as
            :func:`mesh_handler_naive_triangulate <kaolin.io.utils.mesh_handler_naive_triangulate>`.

    Returns:
        (obj.return_type):
            namedtuple of:

            - **vertices** (torch.Tensor): vertex locations of shape :math:`(\text{num_vertices}, 3)`.
            - **faces** (torch.LongTensor): indices into vertex array
              of shape :math:`(\text{num_faces}, \text{face_size})`.
            - **uvs** (torch.Tensor): UV map coordinates of shape :math:`(\text{num_uvs}, 2)`.
            - **face_uvs_idx** (torch.LongTensor): indices into UVmap for every vertex of every face
              of shape :math:`(\text{num_faces}, \text{face_size})`.
            - **materials** (list of dict):
              a list of materials (see return values of :func:`load_mtl`) sorted by their `material_name`.
            - **material_assignments** (dict of torch.LongTensor): (torch.ShortTensor): of shape `(\text{num_faces},)`
                containing index of the material (in the `materials` list) assigned to the corresponding face,
                or `-1` if no material was assigned.
            - **normals** (torch.Tensor): normal values of shape :math:`(\text{num_normals}, 3)`.
            - **face_normals_idx** (torch.LongTensor): indices into the normal array for every vertex
              of every face, of shape :math:`(\text{num_faces}, \text{face_size})`.

    Raises:
        MaterialNotFoundError:
            The .obj is using a material not parsed from material libraries (set `error_handler` to skip).
        MaterialFileError:
            From :func:`load_mtl`: Failed to open material path (set `error_handler` to skip).
        MaterialLoadError:
            From :func:`load_mtl`: Failed to load material, very often due to path to
            map_Kd/map_Ka/map_ks being invalid (set `error_handler` to skip).
        NonHomogeneousMeshError:
            The number of vertices were not equal for all faces (set `heterogeneous_mesh_handler` to handle).
    """
    triangulate_handler = None if not triangulate else utils.mesh_handler_naive_triangulate

    if error_handler is None:
        error_handler = default_error_handler
    vertices = []
    faces = []
    uvs = []
    # 3 values per face
    face_uvs_idx = []
    normals = []
    # 3 values per face
    face_normals_idx = []

    # materials_dict contains:
    #   {material_name: {properties dict}}
    materials_dict = {}

    # material_assignments contain:
    #    {material_name: [(face_idx_start, face_idx_end], (face_idx_start, face_idx_end])
    material_assignments_dict = {}
    material_faceidx_start = None
    active_material_name = None

    def _maybe_complete_material_assignment():
        if active_material_name is not None:
            if material_faceidx_start != len(face_uvs_idx):  # Only add if at least one face is assigned
                material_assignments_dict.setdefault(active_material_name, []).append(
                    torch.LongTensor([material_faceidx_start, len(face_uvs_idx)]))

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
                normals.append(data[1:])
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
                        face_normals_idx.append([int(d[2]) for d in data])
                    else:
                        face_normals_idx.append([0] * len(data))
            elif with_materials and data[0] == 'usemtl':
                _maybe_complete_material_assignment()
                active_material_name = data[1]
                material_faceidx_start = len(face_uvs_idx)
            elif with_materials and data[0] == 'mtllib':
                mtl_path = os.path.join(os.path.dirname(path), data[1])
                materials_dict.update(load_mtl(mtl_path, error_handler))

    _maybe_complete_material_assignment()

    vertices = torch.FloatTensor([float(el) for sublist in vertices for el in sublist]).view(-1, 3)
    face_vertex_counts = torch.IntTensor([len(f) for f in faces])
    # key: (Nx2) tensor of (start, end faceidx]
    material_assignments_dict = {k: torch.stack(v) for k, v in material_assignments_dict.items()}

    def _apply_handler(handler):
        all_features = [faces, face_uvs_idx, face_normals_idx]
        # Flatten all features
        all_features = [flatten_feature(f) for f in all_features]
        return handler(vertices, face_vertex_counts, *all_features, face_assignments=material_assignments_dict)

    # Handle non-homogeneous meshes
    is_heterogeneous = not torch.all(face_vertex_counts == face_vertex_counts[0])
    if is_heterogeneous:
        if heterogeneous_mesh_handler is None:
            raise utils.NonHomogeneousMeshError(f'Mesh is non-homogeneous '
                                                f'and cannot be imported from {path}.'
                                                f'User can set heterogeneous_mesh_handler.'
                                                f'See kaolin.io.utils for the available options')

        mesh = _apply_handler(heterogeneous_mesh_handler)
        if mesh is None:
            warnings.warn(f'Heterogeneous mesh at path {path} not converted by the handler; returning None.')
            return None
        vertices, face_vertex_counts, faces, face_uvs_idx, face_normals_idx, material_assignments_dict = mesh

    if triangulate_handler is not None and not torch.all(face_vertex_counts == 3):
        mesh = _apply_handler(triangulate_handler)
        if mesh is None:
            warnings.warn(f'Non-triangular mesh at path {path} not triangulated; returning None.')
            return None
        vertices, face_vertex_counts, faces, face_uvs_idx, face_normals_idx, material_assignments_dict = mesh

    faces = torch.LongTensor(faces) - 1

    if with_materials:
        uvs = torch.FloatTensor([float(el) for sublist in uvs
                                 for el in sublist]).view(-1, 2)
        face_uvs_idx = torch.LongTensor(face_uvs_idx) - 1
        materials, material_assignments = process_materials_and_assignments(
            materials_dict, material_assignments_dict, error_handler, faces.shape[0], error_context_str=path)
    else:
        uvs = None
        face_uvs_idx = None
        materials = None
        material_assignments = None

    if with_normals:
        normals = torch.FloatTensor(
            [float(el) for sublist in normals
             for el in sublist]).view(-1, 3)
        face_normals_idx = torch.LongTensor(face_normals_idx) - 1
    else:
        normals = None
        face_normals_idx = None

    return return_type(vertices, faces, uvs, face_uvs_idx, materials,
                       material_assignments, normals, face_normals_idx)


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
            - **material_name**: string name of the material

    Raises:
        MaterialFileError:
            Failed to open material path.
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
                    mtl_data[material_name] = {'material_name': material_name}
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
