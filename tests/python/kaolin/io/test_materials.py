# Copyright (c) 2019, 20-21 NVIDIA CORPORATION & AFFILIATES.
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

from kaolin.io import materials, usd, obj


# Seed for texture sampling
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
    material = materials.PBRMaterial(**params)
    yield material


@pytest.fixture(scope='module')
def material_textures():
    params = {
        'diffuse_texture': torch.rand((3, 256, 256)),
        'roughness_texture': torch.rand((1, 256, 256)),
        'metallic_texture': torch.rand((1, 256, 256)),
        'normals_texture': torch.rand((1, 256, 256)),
        'specular_texture': torch.rand((3, 256, 256)),
        'is_specular_workflow': True,
    }
    material = materials.PBRMaterial(**params)
    yield material


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
        material_values.write_to_usd(file_path, '/World/Looks/pbr', texture_dir='texture')

        material_in = materials.PBRMaterial().read_from_usd(file_path, '/World/Looks/pbr', texture_path='texture')

        assert material_values.diffuse_color == pytest.approx(material_in.diffuse_color, 0.1)
        assert material_values.roughness_value == pytest.approx(material_in.roughness_value, 0.1)
        assert material_values.metallic_value == pytest.approx(material_in.metallic_value, 0.1)
        assert material_values.specular_color == pytest.approx(material_in.specular_color, 0.1)
        assert material_values.is_specular_workflow == material_in.is_specular_workflow

    def test_cycle_values(self, out_dir, material_values):
        file_path = os.path.join(out_dir, 'pbr_test.usda')
        material_values.write_to_usd(file_path, '/World/Looks/pbr')

        material_in = materials.PBRMaterial().read_from_usd(file_path, '/World/Looks/pbr')

        assert material_values.diffuse_color == pytest.approx(material_in.diffuse_color, 0.1)
        assert material_values.roughness_value == pytest.approx(material_in.roughness_value, 0.1)
        assert material_values.metallic_value == pytest.approx(material_in.metallic_value, 0.1)
        assert material_values.specular_color == pytest.approx(material_in.specular_color, 0.1)
        assert material_values.is_specular_workflow == material_in.is_specular_workflow

    def test_cycle_textures(self, out_dir, material_textures):
        """Cycle test for textures. This conversion is lossy!"""
        file_path = os.path.join(out_dir, 'pbr_tex_test.usda')
        material_textures.write_to_usd(file_path, '/World/Looks/pbr')

        material_in = materials.PBRMaterial().read_from_usd(file_path, '/World/Looks/pbr')
        assert torch.allclose(material_textures.diffuse_texture, material_in.diffuse_texture, atol=1e-2)
        assert torch.allclose(material_textures.roughness_texture, material_in.roughness_texture, atol=1e-2)
        assert torch.allclose(material_textures.metallic_texture, material_in.metallic_texture, atol=1e-2)
        assert torch.allclose(material_textures.normals_texture, material_in.normals_texture, atol=1e-2)
        assert torch.allclose(material_textures.specular_texture, material_in.specular_texture, atol=1e-2)
        assert material_textures.is_specular_workflow == material_in.is_specular_workflow

    def test_material_values(self, out_dir):
        out_path = os.path.join(out_dir, 'pbr_material_values.usda')
        stage = usd.create_stage(out_path)

        tests = {
            'Default': {},
            'Diffuse': {'diffuse_color': (0., 1., 0.)},
            'Roughness': {'roughness_value': 0.1},
            'Metallic': {'metallic_value': 1.},
            'Clearcoat': {'clearcoat_value': 1.},
            'ClearcoatRougness': {'clearcoat_roughness_value': 1.},
            'Opacity': {'opacity_value': 0.5},
            'OpacityThreshold': {'opacity_threshold': 0.5},
            'Ior': {'ior_value': 1.},
            'Specular': {'specular_color': (1., 0., 0.), 'is_specular_workflow': True},
            'Displacement': {'displacement_value': 0.1},
        }
        for test_name, params in tests.items():
            prim = stage.DefinePrim(f'/World/{test_name}', 'Sphere')
            mat = materials.PBRMaterial(**params)
            mat.write_to_usd(out_path, f'/World/Looks/{test_name}', bound_prims=[prim])
        stage.Save()

        # Confirm exported USD matches golden file
        # TODO(jlafleche) Render the two mesh for visual comparison
        golden = os.path.join(out_dir, os.pardir, os.pardir, os.pardir,
                              os.pardir, 'samples/golden/pbr_material_values.usda')
        assert open(golden).read() == open(out_path).read()

    def test_material_textures(self, out_dir, mesh, material_textures):
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
            'Opacity': {'opacity_texture': _create_checkerboard((0.1,), (0.9,)), 'metallic_colorspace': 'raw',
                        'opacity_threshold': 0.5},
            'Ior': {'ior_texture': _create_checkerboard((0.1,), (0.9,)), 'metallic_colorspace': 'raw'},
            'Normal': {'normals_texture': _create_checkerboard((0., 0., 1.,), (0., 0.5, 0.5)),
                       'normals_colorspace': 'raw'},
            'Specular': {'specular_texture': _create_checkerboard((1., 0., 0.), (0., 0., 1.)),
                         'is_specular_workflow': True, 'specular_colorspace': 'raw'},
            'Displacement': {'displacement_texture': _create_checkerboard((0.1,), (0.9,)),
                             'specular_colorspace': 'raw'},
        }

        for test_name, params in tests.items():
            material_textures = materials.PBRMaterial(**params)
            prim = usd.add_mesh(stage, f'/World/{test_name}', mesh.vertices, mesh.faces,
                                uvs=mesh.uvs,
                                face_uvs_idx=mesh.face_uvs_idx,
                                face_normals=mesh.vertex_normals[mesh.face_normals].view(-1, 3))
            material_textures.write_to_usd(out_path, f'/World/Looks/{test_name}', bound_prims=[prim])
        stage.Save()

        # Confirm exported USD matches golden file
        # TODO(jlafleche) Render the two mesh for visual comparison
        golden = os.path.join(out_dir, os.pardir, os.pardir, os.pardir,
                              os.pardir, 'samples/golden/pbr_material_textures.usda')
        assert open(golden).read() == open(out_path).read()

    def test_colorspace(self, out_dir, mesh, material_textures):
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
        material = materials.PBRMaterial(**texture)

        prim = usd.add_mesh(stage, '/World/colorspace_test', mesh.vertices, mesh.faces,
                            uvs=mesh.uvs,
                            face_uvs_idx=mesh.face_uvs_idx,
                            face_normals=mesh.vertex_normals[mesh.face_normals].view(-1, 3))
        material.write_to_usd(out_path, '/World/Looks/colorspace_test', bound_prims=[prim])

        material_in = materials.PBRMaterial().read_from_usd(out_path, '/World/Looks/colorspace_test')

        assert material_in.diffuse_colorspace == 'sRGB'
        assert material_in.metallic_colorspace == 'auto'
        assert material_in.roughness_colorspace == 'raw'
