# Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES.
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

from kaolin.io import materials as kal_materials
from kaolin.io import usd, obj
from kaolin.utils.testing import contained_torch_equal

_misc_attributes = [
    'diffuse_colorspace',
    'roughness_colorspace',
    'metallic_colorspace',
    'clearcoat_colorspace',
    'clearcoat_roughness_colorspace',
    'opacity_colorspace',
    'ior_colorspace',
    'specular_colorspace',
    'normals_colorspace',
    'displacement_colorspace',
    'is_specular_workflow'
]

_value_attributes = [
    'diffuse_color',
    'roughness_value',
    'metallic_value',
    'clearcoat_value',
    'clearcoat_roughness_value',
    'opacity_value',
    'opacity_threshold',
    'ior_value',
    'specular_color',
    'displacement_value',
]

_texture_attributes = [
    'diffuse_texture',
    'roughness_texture',
    'metallic_texture',
    'clearcoat_texture',
    'clearcoat_roughness_texture',
    'opacity_texture',
    'ior_texture',
    'specular_texture',
    'displacement_texture'
]

# Seed for texture sampling
# TODO(cfujitsang): This might fix the seed for the whole pytest.
torch.random.manual_seed(0)


@pytest.fixture(scope='class')
def out_dir():
    # Create temporary output directory
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_out')
    os.makedirs(out_dir, exist_ok=True)
    yield out_dir
    shutil.rmtree(out_dir)


@pytest.fixture(scope='module')
def material_values():
    params = {
        'diffuse_color': (0., 1., 0.),
        'roughness_value': 0.1,
        'metallic_value': 1.,
        'specular_color': (1., 0., 0.),
        'is_specular_workflow': True,
    }
    yield params


@pytest.fixture(scope='module')
def material_textures():
    params = {
        'diffuse_texture': torch.rand((3, 256, 256)),
        'roughness_texture': torch.rand((1, 256, 256)),
        'metallic_texture': torch.rand((1, 256, 256)),
        'normals_texture': torch.rand((1, 256, 256)),
        'specular_texture': torch.rand((3, 256, 256)),
    }
    yield params


@pytest.fixture(scope='module')
def mesh():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    obj_mesh = obj.import_mesh(os.path.join(cur_dir, os.pardir, os.pardir,
                               os.pardir, 'samples/rocket.obj'), with_normals=True,
                               with_materials=True, error_handler=obj.skip_error_handler)
    return obj_mesh


class TestPBRMaterial:
    def test_separate_texture_path(self, out_dir, material_values):
        file_path = os.path.join(out_dir, 'pbr_test.usda')
        mat = kal_materials.PBRMaterial(**material_values)
        mat.write_to_usd(file_path, '/World/Looks/pbr', texture_dir='texture')

        material_in = kal_materials.PBRMaterial().read_from_usd(file_path, '/World/Looks/pbr', texture_path='texture')

        assert mat.diffuse_color == pytest.approx(material_in.diffuse_color, 0.1)
        assert mat.roughness_value == pytest.approx(material_in.roughness_value, 0.1)
        assert mat.metallic_value == pytest.approx(material_in.metallic_value, 0.1)
        assert mat.specular_color == pytest.approx(material_in.specular_color, 0.1)
        assert mat.is_specular_workflow == material_in.is_specular_workflow

    def test_cycle_values(self, out_dir, material_values):
        file_path = os.path.join(out_dir, 'pbr_test.usda')
        mat = kal_materials.PBRMaterial(**material_values)
        mat.write_to_usd(file_path, '/World/Looks/pbr')

        material_in = kal_materials.PBRMaterial().read_from_usd(file_path, '/World/Looks/pbr')

        assert mat.diffuse_color == pytest.approx(material_in.diffuse_color, 0.1)
        assert mat.roughness_value == pytest.approx(material_in.roughness_value, 0.1)
        assert mat.metallic_value == pytest.approx(material_in.metallic_value, 0.1)
        assert mat.specular_color == pytest.approx(material_in.specular_color, 0.1)
        assert mat.is_specular_workflow == material_in.is_specular_workflow

    def test_cycle_textures(self, out_dir, material_textures):
        """Cycle test for textures. This conversion is lossy!"""
        file_path = os.path.join(out_dir, 'pbr_tex_test.usda')
        mat = kal_materials.PBRMaterial(**material_textures)
        mat.write_to_usd(file_path, '/World/Looks/pbr')

        material_in = kal_materials.PBRMaterial().read_from_usd(file_path, '/World/Looks/pbr')
        assert torch.allclose(mat.diffuse_texture, material_in.diffuse_texture, atol=1e-2)
        assert torch.allclose(mat.roughness_texture, material_in.roughness_texture, atol=1e-2)
        assert torch.allclose(mat.metallic_texture, material_in.metallic_texture, atol=1e-2)
        assert torch.allclose(mat.normals_texture, material_in.normals_texture, atol=1e-2)
        assert torch.allclose(mat.specular_texture, material_in.specular_texture, atol=1e-2)
        assert mat.is_specular_workflow == material_in.is_specular_workflow

    def test_material_values(self, out_dir):
        out_path = os.path.join(out_dir, 'pbr_material_values.usda')
        stage = usd.create_stage(out_path)

        tests = {
            'Default': {},
            'Diffuse': {'diffuse_color': (0., 1., 0.)},
            'SpecularRoughness': {
                'diffuse_color': (1., 0., 0.),
                'roughness_value': 0.1,
                'specular_color': (0., 0., 1.),
                'is_specular_workflow': True
            },
            'Metallic': {
                'diffuse_color': (0., 1., 0.),
                'metallic_value': 1.,
                'is_specular_workflow': False
            },
            'Clearcoat': {'clearcoat_value': 1.},
            'ClearcoatRougness': {'clearcoat_roughness_value': 1.},
            'Opacity': {'opacity_value': 0.5},
            'OpacityThreshold': {'opacity_threshold': 0.5},
            'Ior': {'ior_value': 1.},
            'Displacement': {'displacement_value': 0.1},
        }
        for test_name, params in tests.items():
            prim = stage.DefinePrim(f'/World/{test_name}', 'Sphere')
            mat = kal_materials.PBRMaterial(**params)
            mat.write_to_usd(out_path, f'/World/Looks/{test_name}', bound_prims=[prim])
        stage.Save()

        # Confirm exported USD matches golden file
        # TODO(jlafleche) Render the two mesh for visual comparison
        golden = os.path.join(out_dir, os.pardir, os.pardir, os.pardir,
                              os.pardir, 'samples/golden/pbr_material_values.usda')
        assert open(golden).read() == open(out_path).read()

    def test_material_textures(self, out_dir, mesh):
        def _create_checkerboard(val1, val2):
            channels = len(val1)
            checkerboard = torch.ones((channels, 2, 2)) * torch.tensor(val1)[:, None, None]
            checkerboard[:, 0, 0] = torch.tensor(val2)
            checkerboard[:, 1, 1] = torch.tensor(val2)
            checkerboard = torch.nn.functional.interpolate(checkerboard[None, ...], scale_factor=128)[0]
            return checkerboard
        out_path = os.path.join(out_dir, 'pbr_material_textures.usda')
        stage = usd.create_stage(out_path)

        tests = {
            'Default': {},
            'Diffuse': {'diffuse_texture': _create_checkerboard((0., 1., 0.), (0., 0., 1.)),
                        'diffuse_colorspace': 'sRGB'},
            'Roughness': {'roughness_texture': _create_checkerboard((0.1,), (0.9,)), 'roughness_colorspace': 'raw'},
            'Metallic': {'metallic_texture': _create_checkerboard((0.1,), (0.9,)), 'metallic_colorspace': 'raw'},
            'Clearcoat': {'clearcoat_texture': _create_checkerboard((0.01,), (0.9,)), 'metallic_colorspace': 'raw'},
            'ClearcoatRoughness': {'clearcoat_roughness_texture': _create_checkerboard((0.1,), (0.9,)), 'metallic_colorspace': 'raw'},
            'Opacity': {'opacity_texture': _create_checkerboard((0.1,), (0.9,)), 'opacity_threshold': 0.5,
                        'opacity_colorspace': 'raw'},
            'Ior': {'ior_texture': _create_checkerboard((0.1,), (0.9,)), 'ior_colorspace': 'raw'},
            'Normal': {'normals_texture': _create_checkerboard((0., 0., 1.,), (0., 0.5, 0.5)),
                       'normals_colorspace': 'raw'},
            'Specular': {'specular_texture': _create_checkerboard((1., 0., 0.), (0., 0., 1.)),
                         'is_specular_workflow': True, 'specular_colorspace': 'raw'},
            'Displacement': {'displacement_texture': _create_checkerboard((0.1,), (0.9,)),
                             'displacement_colorspace': 'raw'},
        }

        for test_name, params in tests.items():
            mat = kal_materials.PBRMaterial(**params)
            prim = usd.add_mesh(stage, f'/World/{test_name}', mesh.vertices, mesh.faces,
                                uvs=mesh.uvs,
                                face_uvs_idx=mesh.face_uvs_idx,
                                face_normals=mesh.normals[mesh.face_normals_idx].view(-1, 3))
            mat.write_to_usd(out_path, f'/World/Looks/{test_name}', bound_prims=[prim])
        stage.Save()

        # Confirm exported USD matches golden file
        # TODO(jlafleche) Render the two mesh for visual comparison
        golden = os.path.join(out_dir, os.pardir, os.pardir, os.pardir,
                              os.pardir, 'samples/golden/pbr_material_textures.usda')
        assert open(golden).read() == open(out_path).read()

    def test_colorspace(self, out_dir, mesh):
        out_path = os.path.join(out_dir, 'colorspace_auto.usda')
        stage = usd.create_stage(out_path)

        def _create_checkerboard(val1, val2):
            channels = len(val1)
            checkerboard = torch.ones((channels, 2, 2)) * torch.tensor(val1)[:, None, None]
            checkerboard[:, 0, 0] = torch.tensor(val2)
            checkerboard[:, 1, 1] = torch.tensor(val2)
            checkerboard = torch.nn.functional.interpolate(checkerboard[None, ...], scale_factor=128)[0]
            return checkerboard

        single_channel_texture = _create_checkerboard((0.2,), (0.8,))
        rgb_texture = _create_checkerboard((0., 0.4, 0.), (0., 0., 0.4))

        texture = {'metallic_texture': single_channel_texture, 'metallic_colorspace': 'auto',
                   'roughness_texture': single_channel_texture, 'roughness_colorspace': 'raw',
                   'diffuse_texture': rgb_texture, 'diffuse_colorspace': 'sRGB'}
        material = kal_materials.PBRMaterial(**texture)

        prim = usd.add_mesh(stage, '/World/colorspace_test', mesh.vertices, mesh.faces,
                            uvs=mesh.uvs,
                            face_uvs_idx=mesh.face_uvs_idx,
                            face_normals=mesh.normals[mesh.face_normals_idx].view(-1, 3))
        material.write_to_usd(out_path, '/World/Looks/colorspace_test', bound_prims=[prim])

        material_in = kal_materials.PBRMaterial().read_from_usd(out_path, '/World/Looks/colorspace_test')

        assert material_in.diffuse_colorspace == 'sRGB'
        assert material_in.metallic_colorspace == 'auto'
        assert material_in.roughness_colorspace == 'raw'

    @pytest.mark.parametrize('device', [None, 'cuda:0'])
    @pytest.mark.parametrize('non_blocking', [False, True])
    def test_cuda(self, material_values, material_textures, device, non_blocking):
        mat = kal_materials.PBRMaterial(**material_values, **material_textures)
        cuda_mat = mat.cuda(device=device, non_blocking=non_blocking)
        for param_name in _value_attributes + _texture_attributes:
            val = getattr(mat, param_name)
            cuda_val = getattr(cuda_mat, param_name)
            if val is None:
                assert cuda_val is None
            else:
                assert torch.equal(cuda_val, val.cuda())
                assert val.is_cpu

        for param_name in _misc_attributes:
            assert getattr(mat, param_name) == getattr(cuda_mat, param_name)

    def test_cpu(self, material_values, material_textures):
        mat = kal_materials.PBRMaterial(**material_values, **material_textures)
        # see test_cuda for guarantee that this is reliable
        cuda_mat = mat.cuda()
        cpu_mat = cuda_mat.cpu()
        for param_name in _value_attributes + _texture_attributes:
            cpu_val = getattr(cpu_mat, param_name)
            cuda_val = getattr(cuda_mat, param_name)
            if cuda_val is None:
                assert cpu_val is None
            else:
                assert torch.equal(cpu_val, cuda_val.cpu())
                assert cuda_val.is_cuda

        for param_name in _misc_attributes:
            assert getattr(cpu_mat, param_name) == getattr(cuda_mat, param_name)

    def test_contiguous(self, material_values, material_textures):
        strided_material_textures = {
            k: torch.as_strided(v, (v.shape[0], int(v.shape[1] / 2), int(v.shape[2])), (1, 2, 2))
            for k, v in material_textures.items()
        }
        mat = kal_materials.PBRMaterial(**material_values, **strided_material_textures)
        contiguous_mat = mat.contiguous()
        for param_name in _texture_attributes:
            val = getattr(mat, param_name)
            contiguous_val = getattr(contiguous_mat, param_name)
            if contiguous_val is None:
                assert contiguous_val is None
            else:
                assert torch.equal(contiguous_val, val.contiguous())
                assert not val.is_contiguous()

        for param_name in _value_attributes:
            if contiguous_val is None:
                assert contiguous_val is None
            else:
                assert torch.equal(getattr(mat, param_name), getattr(contiguous_mat, param_name))

        for param_name in _misc_attributes:
            assert getattr(mat, param_name) == getattr(contiguous_mat, param_name)

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
        materials, material_assignments = kal_materials.process_materials_and_assignments(
            materials_dict, material_assignments_dict, any_error_handler, num_faces)
        assert contained_torch_equal(materials, expected_materials)
        assert torch.equal(material_assignments, expected_assignments)

        # Now let's add assignment to a non-existent material
        material_assignments_dict['kitties'] = torch.LongTensor([[22, 25]])
        if any_error_handler == obj.default_error_handler:
            with pytest.raises(obj.MaterialNotFoundError):
                materials, material_assignments = kal_materials.process_materials_and_assignments(
                    materials_dict, material_assignments_dict, any_error_handler, num_faces, error_context_str=path)
        elif any_error_handler in [obj.skip_error_handler, obj.ignore_error_handler]:
            # Ignore extra assignment
            materials, material_assignments = kal_materials.process_materials_and_assignments(
                materials_dict, material_assignments_dict, any_error_handler, num_faces, error_context_str=path)
            assert contained_torch_equal(materials, expected_materials)
            assert torch.equal(material_assignments, expected_assignments)
        elif any_error_handler == obj.create_missing_materials_error_handler:
            expected_assignments[22:25] = 2
            materials, material_assignments = kal_materials.process_materials_and_assignments(
                materials_dict, material_assignments_dict, any_error_handler, num_faces)
            assert [m['material_name'] for m in materials] == ['bricks', 'grass', 'kitties']
            assert contained_torch_equal(materials[:2], expected_materials)
