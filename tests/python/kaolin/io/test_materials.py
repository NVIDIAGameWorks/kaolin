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
import os
import shutil

import torch
import pytest

import kaolin.io.materials
from kaolin.io import usd, obj
import kaolin.render.materials
from kaolin.render.materials import random_material_values, random_material_colorspaces, random_material_textures
from kaolin.utils.testing import contained_torch_equal


class TestPBRMaterial:
    def test_raises_exception(self):
        with pytest.raises(DeprecationWarning):
            mat = kaolin.io.materials.PBRMaterial()


class TestUtilities:
    @pytest.mark.parametrize('any_error_handler', [obj.skip_error_handler, obj.ignore_error_handler,
                                                   obj.create_missing_materials_error_handler,
                                                   obj.default_error_handler])
    @pytest.mark.parametrize('material_assignments_shape', [1, 2])  # face indices, or start,end ranges
    def test_process_materials_and_assignments(self, any_error_handler, material_assignments_shape):
        materials_dict = {
            'bricks': {'Ka': torch.rand((3,)), 'Kd': torch.rand((3,)), 'material_name': 'bricks'},
            'grass': {'Ka': torch.rand((3,)), 'Kd': torch.rand((3,)), 'material_name': 'grass'}}
        if material_assignments_shape == 2:
            material_assignments_dict = {  # Using start,end ranges
                'bricks': torch.LongTensor([[0, 10], [15, 20]]),
                'grass': torch.LongTensor([[10, 15], [21, 22], [25, 30]])}
        else:
            material_assignments_dict = {  # Equivalent to above, but using full list of faces
                'bricks': torch.LongTensor(list(range(0, 10)) + list(range(15, 20))),
                'grass': torch.LongTensor(list(range(10, 15)) + list(range(21, 22)) + list(range(25, 30)))}
        path = 'path'
        num_faces = 30
        expected_materials = [materials_dict['bricks'], materials_dict['grass']]
        expected_assignments = torch.ShortTensor(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, -1, 1, -1, -1, -1, 1, 1, 1, 1, 1])

        # This should succeed with any error handler
        materials, material_assignments = kaolin.io.materials.process_materials_and_assignments(
            materials_dict, material_assignments_dict, any_error_handler, num_faces)
        assert contained_torch_equal(materials, expected_materials)
        assert torch.equal(material_assignments, expected_assignments)

        # Now let's add assignment to a non-existent material
        material_assignments_dict['kitties'] = torch.LongTensor([[22, 25]])
        if any_error_handler == obj.default_error_handler:
            with pytest.raises(obj.MaterialNotFoundError):
                materials, material_assignments = kaolin.io.materials.process_materials_and_assignments(
                    materials_dict, material_assignments_dict, any_error_handler, num_faces, error_context_str=path)
        elif any_error_handler in [obj.skip_error_handler, obj.ignore_error_handler]:
            # Ignore extra assignment
            materials, material_assignments = kaolin.io.materials.process_materials_and_assignments(
                materials_dict, material_assignments_dict, any_error_handler, num_faces, error_context_str=path)
            assert contained_torch_equal(materials, expected_materials)
            assert torch.equal(material_assignments, expected_assignments)
        elif any_error_handler == obj.create_missing_materials_error_handler:
            expected_assignments[22:25] = 2
            materials, material_assignments = kaolin.io.materials.process_materials_and_assignments(
                materials_dict, material_assignments_dict, any_error_handler, num_faces)
            assert [m['material_name'] for m in materials] == ['bricks', 'grass', 'kitties']
            assert contained_torch_equal(materials[:2], expected_materials)

    def test_group_materials_by_name(self):
        unnamed_mat1 = kaolin.render.materials.PBRMaterial(**random_material_values())
        unnamed_mat2 = kaolin.render.materials.PBRMaterial(**random_material_textures())
        happy_mat1 = kaolin.render.materials.PBRMaterial(material_name='happy', **random_material_textures())
        happy_mat2 = kaolin.render.materials.PBRMaterial(material_name='happy', **random_material_textures())
        sad_mat1 = kaolin.render.materials.PBRMaterial(material_name='sad', **random_material_textures())
        sad_mat2 = kaolin.render.materials.PBRMaterial(material_name='sad', **random_material_textures())

        materials_input = [unnamed_mat1,  # 0 --> 0
                           happy_mat1,    # 1 --> 1
                           sad_mat1,      # 2 --> 2
                           sad_mat2,      # 3 --> 2
                           unnamed_mat2,  # 4 --> 3
                           happy_mat2]    # 5 --> 1
        material_assignments_input = torch.LongTensor(
            [[0, 1, 2, 3, 4, 5],
             [0, 0, 1, 1, 2, 2],
             [4, 5, 4, 5, 4, 5],
             [5, 3, 3, 2, 4, 1]])

        expected_materials = [unnamed_mat1, happy_mat1, sad_mat1, unnamed_mat2]
        expected_material_assignments = torch.LongTensor(
            [[0, 1, 2, 2, 3, 1],
             [0, 0, 1, 1, 2, 2],
             [3, 1, 3, 1, 3, 1],
             [1, 2, 2, 2, 3, 1]])

        materials, assignments = kaolin.io.materials.group_materials_by_name(
            materials_input, material_assignments_input)
        assert contained_torch_equal(materials, expected_materials, approximate=True)
        assert contained_torch_equal(assignments, expected_material_assignments)
