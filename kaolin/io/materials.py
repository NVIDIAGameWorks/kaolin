# Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES.
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
from collections.abc import Mapping

import torch
import warnings


class MaterialError(Exception):
    pass


class MaterialNotSupportedError(MaterialError):
    pass


class MaterialLoadError(MaterialError):
    pass


class MaterialWriteError(MaterialError):
    pass


class MaterialFileError(MaterialError):
    pass


class MaterialNotFoundError(MaterialError):
    pass


class PBRMaterial:
    DEPRECATION_MESSAGE = """PBRMaterial class moved and is now consistent across USD, gltf, obj imports.
    Please use kaolin.render.materials.PBRMaterial instead.
    For USD read functions for materials, replace your code as follows:
    
    Before:
    mat = kaolin.io.PBRMaterial().read_from_usd(path, scene_path)
    mat.write_to_usd(other_path, scene_path)
    
    Now:
    mat = kaolin.io.usd.import_material(path, scene_path)  # returns kaolin.render.materials.PBRMaterial
    kaolin.io.usd.export_material(mat, other_path, scene_path)
    """
    def __init__(self, *args, **kwargs):
        raise DeprecationWarning('PBRMaterial class moved. Please use kaolin.render.materials.PBRMaterial instead')

    def write_to_usd(self, file_path, scene_path, bound_prims=None, time=None,
                     texture_dir='', texture_file_prefix='', shader='UsdPreviewSurface'):
        raise DeprecationWarning('PBRMaterial.write_to_usd is deprecated; instead use kaolin.io.usd.export_material')

    def read_from_usd(self, file_path, scene_path, texture_path=None, time=None):
        raise DeprecationWarning('PBRMaterial.read_from_usd is deprecated; instead use kaolin.io.usd.import_material')


def group_materials_by_name(materials_list, material_assignments):
    """Groups materials that have the same name. Does not group materials that do not have any name set.

    Args:
        materials_list (list of objects): each item is expected to have material_name member, or
            'material_name' key if dictionary.
        material_assignments (torch.LongTensor or None): with integer value corresponding to material

    Returns:
        (tuple) of:
            - **materials** (list): list of material parameters, with any grouped materials replaced by single material
            - **material_assignments** (torch.LongTensor): copy of material_assignments, modified according to grouped
                    materials (or None if input material_assignments were None)
    """
    def _try_to_get_name(material):
        name = None
        if isinstance(material, Mapping):
            name = material.get('material_name')
        else:
            try:
                name = material.material_name
            except Exception as e:
                warnings.warn(f'Material {type(material)} had no material_name property')
        if name == '':
            name = None
        return name

    material_indices = {}
    new_materials_list = []
    new_material_assignments = material_assignments.clone() if material_assignments is not None else None
    for current_mat_idx, mat in enumerate(materials_list):
        name = _try_to_get_name(mat)
        if name in material_indices:
            new_mat_idx = material_indices[name]
        else:
            new_mat_idx = len(new_materials_list)
            new_materials_list.append(mat)
            if name is not None:
                material_indices[name] = new_mat_idx

        if material_assignments is not None:
            new_material_assignments[material_assignments == current_mat_idx] = new_mat_idx

    return new_materials_list, new_material_assignments


def process_materials_and_assignments(materials_dict, material_assignments_dict, error_handler, num_faces,
                                      error_context_str=''):
    """Converts dictionary style materials and assignments to final format (see args/return values).

    Args:
        materials_dict (dict of str to dict): mapping from material name to material parameters
        material_assignments_dict (dict of str to torch.LongTensor): mapping from material name to either
           1) a K x 2 tensor with start and end face indices of the face ranges assigned to that material or
           2) a K, tensor with face indices assigned to that material
        error_handler: handler able to handle MaterialNotFound error - error can be thrown, ignored, or the
            handler can return a dummy material for material not found (if this is not the case, assignments to
            non-existent materials will be lost), e.g. obj.create_missing_materials_error_handler.
        num_faces: total number of faces in the model
        error_context_str (str): any extra info to attach to thrown errors

    Returns:
        (tuple) of:

        - **materials** (list): list of material parameters, sorted alphabetically by their name
        - **material_assignments** (torch.ShortTensor): of shape `(\text{num_faces},)` containing index of the
            material (in the above list) assigned to the corresponding face, or `-1` if no material was assigned.
    """
    def _try_to_set_name(generated_material, material_name):
        if isinstance(generated_material, Mapping):
            generated_material['material_name'] = material_name
        else:
            try:
                generated_material.material_name = material_name
            except Exception as e:
                warnings.warn(f'Cannot set dummy material_name: {e}')

    # Check that all assigned materials exist and if they don't we create a dummy material
    missing_materials = []
    for mat_name in material_assignments_dict.keys():
        if mat_name not in materials_dict:
            dummy_material = error_handler(
                MaterialNotFoundError(f"'Material {mat_name}' not found, but referenced. {error_context_str}"))

            # Either create dummy material or remove assignment
            if dummy_material is not None:
                _try_to_set_name(dummy_material, mat_name)
                materials_dict[mat_name] = dummy_material
            else:
                missing_materials.append(mat_name)

    # Ignore assignments to missing materials (unless handler created dummy material)
    for mat_name in missing_materials:
        del material_assignments_dict[mat_name]

    material_names = sorted(materials_dict.keys())
    materials = [materials_dict[name] for name in material_names]  # Alphabetically ordered materials
    material_assignments = torch.zeros((num_faces,), dtype=torch.int16) - 1

    # Process material assignments to use material indices instead
    for name, values in material_assignments_dict.items():
        mat_idx = material_names.index(name)  # Alphabetically sorted material

        if len(values.shape) == 1:
            indices = values
        else:
            assert len(values.shape) == 2 and values.shape[-1] == 2, \
                f'Unxpected shape {values.shape} for material assignments for material {name} ' \
                f'(expected (K,) or (K, 2)). {error_context_str}'
            # Rewrite (K, 2) tensor of (face_idx_start, face_idx_end] to (M,) tensor of face_idx
            indices = torch.cat(
                [torch.arange(values[r, 0], values[r, 1], dtype=torch.long) for r in range(values.shape[0])])

        # Use face indices as index to set material_id in face-aligned material assignments
        material_assignments[indices] = mat_idx

    return materials, material_assignments
