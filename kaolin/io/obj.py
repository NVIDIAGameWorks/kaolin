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
import logging
import numpy as np
import torch
from PIL import Image

from kaolin.io.materials import MaterialLoadError, MaterialFileError, MaterialNotFoundError, \
    process_materials_and_assignments
from kaolin.io import utils
from kaolin.render.materials import PBRMaterial
from kaolin.rep import SurfaceMesh

__all__ = [
    'ignore_error_handler',
    'skip_error_handler',
    'create_missing_materials_error_handler',
    'default_error_handler',
    'import_mesh',
    'load_mtl'
]


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
                triangulate=False, raw_materials=True):
    r"""Load data from an obj file as a single mesh, and return data as CPU pytorch tensors in an easy-to-manage
    :class:`kaolin.rep.SurfaceMesh` container.

    .. note::
        Currently has limited materials support for Kd, Ka, Ks, map_Kd, map_Ka and map_Ks,
        following the format described in: http://paulbourke.net/dataformats/obj/

    Args:
        path (str): path to the obj file (with extension).
        with_materials (bool): if True, load materials. Default: False.
        with_normals (bool): if True, load normals. Default: False.
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
            If `heterogeneous_mesh_handler` is not set, this flag will cause non-homogeneous meshes to
            be triangulated and loaded without error; otherwise triangulation executes after `heterogeneous_mesh_handler`,
            which may skip or throw an error.
        raw_materials: if True (default) and `with_materials`, will return raw material values as a dictionary
            (see return values of :func:`load_mtl`); if False will instead return materials
            as instances of :class:`kaolin.render.materials.PBRMaterial`.

    Returns:
        (SurfaceMesh):
            an unbatched instance of :class:`kaolin.rep.SurfaceMesh`, where:

            * **normals** and **face_normals_idx** will only be filled if `with_normals=True`
            * **materials** will be a list of materials (see `raw_materials` argument for available return types)
              sorted by their `material_name`; filled only if `with_materials=True`.
            * **material_assignments** will be a tensor
              of shape ``(num_faces,)`` containing the index
              of the material (in the `materials` list) assigned to the corresponding face,
              or `-1` if no material was assigned; filled only if `with_materials=True`.

    Raises:
        kaolin.io.materials.MaterialNotFoundError:
            The .obj is using a material not parsed from material libraries (set `error_handler` to skip).
        kaolin.io.materials.MaterialFileError:
            From :func:`load_mtl`: Failed to open material path (set `error_handler` to skip).
        kaolin.io.materials.MaterialLoadError:
            From :func:`load_mtl`: Failed to load material, very often due to path to
            map_Kd/map_Ka/map_ks being invalid (set `error_handler` to skip).
        NonHomogeneousMeshError:
            The number of vertices were not equal for all faces (set `heterogeneous_mesh_handler` to handle).


    .. rubric:: Examples

    To load a mesh without loading normals, materials or UVs::

        >>> from kaolin.io.obj import import_mesh
        >>> mesh = import_mesh("sample_data/meshes/pizza.obj")
        >>> print(mesh)
        SurfaceMesh object with batching strategy NONE
                    vertices: [482, 3] (torch.float32)[cpu]
                       faces: [960, 3] (torch.int64)[cpu]
               face_vertices: if possible, computed on access from: (faces, vertices)
                face_normals: if possible, computed on access from: (normals, face_normals_idx) or (vertices, faces)
              vertex_normals: if possible, computed on access from: (faces, face_normals)
                    face_uvs: if possible, computed on access from: (uvs, face_uvs_idx)


        >>> mesh.face_normals  # Causes face_normals and any attributes required to compute it to be auto-computed
        >>> mesh.to_batched()  # Apply fixed topology batching, unsqueezing most attributes
        >>> mesh = mesh.cuda(attributes=["vertices"])  # Moves just vertices to GPU
        >>> print(mesh)
        SurfaceMesh object with batching strategy FIXED
                    vertices: [1, 482, 3] (torch.float32)[cuda:0]
               face_vertices: [1, 960, 3, 3] (torch.float32)[cpu]
                face_normals: [1, 960, 3, 3] (torch.float32)[cpu]
                       faces: [960, 3] (torch.int64)[cpu]
              vertex_normals: if possible, computed on access from: (faces, face_normals)
                    face_uvs: if possible, computed on access from: (uvs, face_uvs_idx)


    To load a mesh with normals, materials and UVs, while triangulating and homogenizing if needed::

        >>> from kaolin.io.obj import import_mesh
        >>> from kaolin.io.utils import mesh_handler_naive_triangulate
        >>> mesh = import_mesh("sample_data/meshes/pizza.obj",
                              with_normals=True, with_materials=True,
                              heterogeneous_mesh_handler=mesh_handler_naive_triangulate,
                              triangulate=True)
        >>> print(mesh)
        SurfaceMesh object with batching strategy NONE
                    vertices: [482, 3] (torch.float32)[cpu]
                     normals: [482, 3] (torch.float32)[cpu]
                         uvs: [514, 2] (torch.float32)[cpu]
                       faces: [960, 3] (torch.int64)[cpu]
            face_normals_idx: [960, 3] (torch.int64)[cpu]
                face_uvs_idx: [960, 3] (torch.int64)[cpu]
        material_assignments: [960] (torch.int16)[cpu]
                   materials: list of length 2
               face_vertices: if possible, computed on access from: (faces, vertices)
                face_normals: if possible, computed on access from: (normals, face_normals_idx) or (vertices, faces)
              vertex_normals: if possible, computed on access from: (faces, face_normals)
                    face_uvs: if possible, computed on access from: (uvs, face_uvs_idx)
    """
    triangulate_handler = None if not triangulate else utils.mesh_handler_naive_triangulate
    if heterogeneous_mesh_handler is None:
        heterogeneous_mesh_handler = triangulate_handler

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
                vertices.append(data[1:4])
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
    is_heterogeneous = face_vertex_counts.numel() > 0 and not torch.all(face_vertex_counts == face_vertex_counts[0])
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
        uvs[..., 1] = 1 - uvs[..., 1]
        face_uvs_idx = torch.LongTensor(face_uvs_idx) - 1
        materials, material_assignments = process_materials_and_assignments(
            materials_dict, material_assignments_dict, error_handler, faces.shape[0], error_context_str=path)
        if not raw_materials:
            materials = [raw_material_to_pbr(m) for m in materials]
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

    return SurfaceMesh(vertices=vertices, faces=faces, uvs=uvs, face_uvs_idx=face_uvs_idx, materials=materials,
                       material_assignments=material_assignments, normals=normals, face_normals_idx=face_normals_idx,
                       unset_attributes_return_none=True)   # for greater backward compatibility

# http://www.paulbourke.net/dataformats/mtl/
# https://github.com/tinyobjloader/tinyobjloader/blob/release/tiny_obj_loader.h#L220
# https://archive.blender.org/developer/D11019
# https://projects.blender.org/blender/blender/commit/a99a62231e
# https://scylardor.fr/2021/05/21/coercing-assimp-into-reading-obj-pbr-materials/
def load_mtl(mtl_path, error_handler):
    """Load and parse a Material file and return its raw values.

    Followed format described in: https://people.sc.fsu.edu/~jburkardt/data/mtl/mtl.html.
    Currently only support diffuse, ambient and specular parameters (Kd, Ka, Ks)
    through single RGB values or texture maps.

    Args:
        mtl_path (str): Path to the mtl file.

    Returns:
        (dict):
            Dictionary of materials, each a dictionary of properties, containing the following keys, torch.Tensor
            values if present in the mtl file. Only keys present in mtl will be set, capitalization of keys
            will be consistent with original mtl, but both upper and lowercase strings will be parsed.

            - **Kd**: diffuse color of shape (3)
            - **map_Kd**: diffuse texture map of shape (H, W, 3)
            - **Ks**: specular color of shape (3)
            - **map_Ks**: specular texture map of shape (H1, W1, 3)
            - **Ka**: ambient color of shape (3)
            - **map_Ka**: ambient texture map of shape (H2, W2, 3)
            - **bump** or **map_bump**: normals texture, typically of shape (H3, W3, 3)
            - **disp**: displacement map, typically of shape (H3, W3, 1)
            - **map_d**: opacity map, typically of shape (H4, W4, 1)
            - **map_ns**: roughness map
            - **map_refl**: metallic map
            - **material_name**: string name of the material

    Raises:
        MaterialFileError:
            Failed to open material path.
        MaterialLoadError:
            Failed to load material, very often due to path to map_Kd/map_Ka/map_Ks being invalid.
    """
    mtl_data = {}
    root_dir = os.path.dirname(mtl_path)

    def _read_image_with_options(root_dir, data):
        # TOOD: this assumption may be wrong; see https://github.com/tinyobjloader/tinyobjloader/blob/cab4ad7254cbf7eaaafdb73d272f99e92f166df8/models/texture-options-issue-85.mtl#L22
        fpath = data[-1]
        texture_path = os.path.join(root_dir, fpath)
        img = Image.open(texture_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.array(img)

        options = {}
        option_vals = []
        for i in range(1, len(data) - 1):
            dval = data[i].strip()
            if dval[0] == '-':
                if len(option_vals) > 0:
                    options[option_vals[0]] = option_vals[1:]
                    option_vals = []
            option_vals.append(dval)

        if len(option_vals) > 0:
            options[option_vals[0]] = option_vals[1:]

        for k, v in options.items():
            if k == '-imfchan':  # parse the channel option
                if len(v) > 0 and len(img.shape) > 2 and img.shape[-1] > 1:
                    if v[0] == 'r':
                        img = img[..., :1]
                    elif v[0] == 'g':
                        img = img[..., 1:2]
                    elif v[0] == 'b':
                        img = img[..., 2:3]
                    else:
                        logging.warning(f'Unrecognized value {v[0]} for flag -imfchan; r, g, or b expected')
            else:
                logging.warning(f'Flag option {k} not supported')
        return torch.from_numpy(img)


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
                # TODO: this is not quite right; need to make this agree with standard.
                elif data[0].lower() in {'map_kd', 'map_ka', 'map_ks', 'bump', 'map_bump', 'disp', 'map_d', 'map_ns', 'map_refl'}:
                    mtl_data[material_name][data[0]] = _read_image_with_options(root_dir, data)
                elif data[0].lower() in {'kd', 'ka', 'ks'}:
                    mtl_data[material_name][data[0]] = torch.tensor(
                        [float(val) for val in data[1:]])
            except Exception as e:
                error_handler(MaterialLoadError(
                    f"Failed to load material at path '{mtl_path}':\n{e}"),
                    data=data, mtl_data=mtl_data)
        f.close()
    return mtl_data


def raw_material_to_pbr(material):
    pbr_params = {'is_specular_workflow': False}  # TODO: is this right?

    # TODO: this is not quite right; need to make this agree with standard
    # See also https://github.com/tinyobjloader/tinyobjloader/blob/release/tiny_obj_loader.h#L220
    # https://projects.blender.org/blender/blender/commit/a99a62231e
    # https://scylardor.fr/2021/05/21/coercing-assimp-into-reading-obj-pbr-materials/
    supported_maps = {
        'map_kd': 'diffuse_texture',
        'map_ks': 'specular_texture',
        'bump': 'normals_texture',
        'map_bump': 'normals_texture',
        'disp': 'displacement_texture',
        'map_d': 'opacity_texture',
        'map_refl': 'metallic_texture',
        'map_ns': 'roughness_texture'
    }
    supported_values = {
        'kd': 'diffuse_color',
        'ks': 'specular_color'
    }
    # TODO: looks like we're missing ambient occlusions in PBRMaterial (map_Ka)
    for k, v in material.items():
        if k == 'material_name':
            pbr_params[k] = v
        elif k.lower() in supported_maps.keys():
            pbr_name = supported_maps[k.lower()]
            pbr_params[pbr_name] = v.float() / 255.
            if pbr_name == 'normals_texture':
                pbr_params[pbr_name] = pbr_params[pbr_name] * 2 - 1.
        elif k.lower() in supported_values.keys():
            pbr_name = supported_values[k.lower()]
            pbr_params[pbr_name] = v
        else:
            logging.warning(f'Cannot convert {k} from obj mtl to PBR spec; use raw_materials=True to import raw values')

    return PBRMaterial(**pbr_params)
