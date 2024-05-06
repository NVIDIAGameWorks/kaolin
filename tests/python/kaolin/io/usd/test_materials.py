# Copyright (c) 2019-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import random
import os
import shutil

import torch
import pytest

import kaolin.io.utils
from kaolin.io import usd, obj
import kaolin.io.materials
import kaolin.io.usd.materials as usd_materials
import kaolin.render.materials
from kaolin.render.materials import random_material_values, random_material_colorspaces, random_material_textures
from kaolin.utils.testing import contained_torch_equal, check_allclose, file_contents_equal

__test_dir = os.path.dirname(os.path.realpath(__file__))
__samples_path = os.path.join(__test_dir, os.pardir, os.pardir, os.pardir, os.pardir, 'samples')

def samples_data_path(*args):
    return os.path.join(__samples_path, *args)

@pytest.fixture(scope='function')
def out_dir():
    # Create temporary output directory
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_out')
    os.makedirs(out_dir, exist_ok=True)
    yield out_dir
    shutil.rmtree(out_dir)

@pytest.fixture(scope='module')
def material_values():
    params = random_material_values()
    yield params


@pytest.fixture(scope='module')
def material_textures():
    params = random_material_textures()
    yield params


@pytest.fixture(scope='module')
def mesh():
    obj_mesh = obj.import_mesh(samples_data_path('rocket.obj'), with_normals=True,
                               with_materials=True, error_handler=obj.skip_error_handler)
    return obj_mesh


class TestUsdPreviewSurfaceIo:
    def test_export_import_defaults(self, out_dir):
        mat = kaolin.render.materials.PBRMaterial(
            **random_material_textures(), **random_material_values(), **random_material_colorspaces())
        mat2 = kaolin.render.materials.PBRMaterial(
            **random_material_textures(), **random_material_values(), **random_material_colorspaces())
        file_path = os.path.join(out_dir, 'default_export.usda')
        scene_path, _ = usd_materials.export_material(mat, file_path)
        scene_path2, _ = usd_materials.export_material(mat2, file_path)
        assert scene_path != scene_path2

        material_in = usd_materials.import_material(file_path, scene_path)
        material_in2 = usd_materials.import_material(file_path, scene_path2)
        mat.material_name = scene_path
        mat2.material_name = scene_path2
        assert contained_torch_equal(mat, material_in, approximate=True, print_error_context='', rtol=1e-2, atol=1e-2)
        assert contained_torch_equal(mat2, material_in2, approximate=True, print_error_context='', rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("same_filepath", [True, False])
    def test_texture_no_overwrite(self, out_dir, same_filepath):
        # Note: it is important these textures are random every time, or we won't see an error with a constant fixture
        mat1_colorspaces = random_material_colorspaces()
        mat1 = kaolin.render.materials.PBRMaterial(**random_material_textures(), **mat1_colorspaces)
        mat1.material_name = '/World/Looks/pretty_material'
        mat2 = kaolin.render.materials.PBRMaterial(**random_material_textures(), **random_material_colorspaces())
        mat2.material_name = '/World/Looks2/pretty_material'

        file_path1 = os.path.join(out_dir, 'pretty_material.usda')
        file_path2 = os.path.join(out_dir, 'pretty_material2.usda')
        if same_filepath:
            file_path2 = file_path1

        usd_materials.export_material(mat1, file_path1, mat1.material_name)
        usd_materials.export_material(mat2, file_path2, mat2.material_name)

        material_in1 = usd_materials.import_material(file_path1, mat1.material_name)
        material_in2 = usd_materials.import_material(file_path2, mat2.material_name)
        assert contained_torch_equal(mat1, material_in1, approximate=True, print_error_context='', rtol=1e-2, atol=1e-2)
        assert contained_torch_equal(mat2, material_in2, approximate=True, print_error_context='', rtol=1e-2, atol=1e-2)

    def test_texture_overwrite(self, out_dir):
        colorspaces = random_material_colorspaces()
        vals = random_material_values()
        mat = kaolin.render.materials.PBRMaterial(**random_material_textures(), **vals, **colorspaces)
        mat.material_name = '/World/Looks/pretty_material'
        file_path = os.path.join(out_dir, 'birches.usda')

        usd_materials.export_material(mat, file_path, mat.material_name)
        material_in = usd_materials.import_material(file_path, mat.material_name)
        assert contained_torch_equal(mat, material_in, approximate=True, print_error_context='', rtol=1e-2, atol=1e-2)

        # save original file content
        orig_material_file_content = open(file_path).read()

        # create and write a completely different material
        for overwrite_textures in [True, False]:
            mat2 = kaolin.render.materials.PBRMaterial(**random_material_textures(), **vals, **colorspaces)
            mat2.material_name = mat.material_name
            usd_materials.export_material(mat2, file_path, mat2.material_name, overwrite_textures=overwrite_textures)
            material_in = usd_materials.import_material(file_path, mat2.material_name)
            # we read in the same material
            assert contained_torch_equal(mat2, material_in, approximate=True, print_error_context='', rtol=1e-2,
                                         atol=1e-2)
            # and furthermore file content is the same as before, pointing to same image files
            if overwrite_textures:
                assert orig_material_file_content == open(file_path).read()
            else:
                assert orig_material_file_content != open(file_path).read()

    # Note: this is original test written for USD materials import/export
    def test_separate_texture_path(self, out_dir, material_values, material_textures):
        file_path = os.path.join(out_dir, 'pbr_test.usda')
        scene_path = '/World/Looks/pbr'
        texture_path = 'texture'

        colorspaces = random_material_colorspaces()
        mat = kaolin.render.materials.PBRMaterial(**material_values, **material_textures, **colorspaces)
        mat.material_name = scene_path
        usd_materials.export_material(mat, file_path, scene_path, texture_path=texture_path)

        # The texture path is baked into the USD, so we don't need to pass it in
        material_in = usd_materials.import_material(file_path, scene_path)
        assert contained_torch_equal(mat, material_in, approximate=True, print_error_context='', rtol=1e-2, atol=1e-2)

        # But what if we moved the textures elsewhere
        texture_path2 = os.path.join('otherdir', 'texture')
        os.makedirs(os.path.join(out_dir, 'otherdir'))
        shutil.copytree(os.path.join(out_dir, texture_path), os.path.join(out_dir, texture_path2))
        material_in = usd_materials.import_material(file_path, scene_path, texture_path='otherdir')
        assert contained_torch_equal(mat, material_in, approximate=True, print_error_context='', rtol=1e-2, atol=1e-2)

        # Should also work, if we don't account for relative paths
        material_in = usd_materials.import_material(file_path, scene_path, texture_path=texture_path2)
        assert contained_torch_equal(mat, material_in, approximate=True, print_error_context='', rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("input_values", ["values", "textures", "both", "none"])
    def test_cycle_write_read(self, out_dir, input_values, material_values, material_textures):
        file_path = os.path.join(out_dir, 'pbr_test.usda')
        scene_path = '/World/Looks/pbr'

        colorspaces = random_material_colorspaces()
        if input_values == "none":
            mat = kaolin.render.materials.PBRMaterial()
        elif input_values == "values":
            mat = kaolin.render.materials.PBRMaterial(**material_values)
        elif input_values == "textures":
            mat = kaolin.render.materials.PBRMaterial(**material_textures, **colorspaces)
        elif input_values == "both":
            mat = kaolin.render.materials.PBRMaterial(**material_values, **material_textures, **colorspaces)
        else:
            raise RuntimeError(f'Bug in the test')

        mat.material_name = scene_path
        usd_materials.export_material(mat, file_path, scene_path)
        material_in = usd_materials.import_material(file_path, scene_path)
        assert contained_torch_equal(mat, material_in, approximate=True, print_error_context='', rtol=1e-2, atol=1e-2)

        # Just in case we also manually compare, to avoid bugs in contained_torch_equal
        if input_values in ["values", "both"]:
            assert mat.diffuse_color == pytest.approx(material_in.diffuse_color, 0.1)
            assert mat.roughness_value == pytest.approx(material_in.roughness_value, 0.1)
            assert mat.metallic_value == pytest.approx(material_in.metallic_value, 0.1)
            assert mat.specular_color == pytest.approx(material_in.specular_color, 0.1)
            assert mat.is_specular_workflow == material_in.is_specular_workflow
        if input_values in ["textures", "both"]:
            assert torch.allclose(mat.diffuse_texture, material_in.diffuse_texture, atol=1e-2)
            assert torch.allclose(mat.roughness_texture, material_in.roughness_texture, atol=1e-2)
            assert torch.allclose(mat.metallic_texture, material_in.metallic_texture, atol=1e-2)
            assert torch.allclose(mat.normals_texture, material_in.normals_texture, atol=1e-2)
            assert torch.allclose(mat.specular_texture, material_in.specular_texture, atol=1e-2)
            assert mat.is_specular_workflow == material_in.is_specular_workflow

    def test_material_overwrite(self, out_dir):
        """What if we export different materials to same file and scene path? Material should be fully overwritten."""
        file_path = os.path.join(out_dir, 'pbr_test.usda')
        scene_path = '/World/Looks/pbr'

        # Write random material to file
        mat = kaolin.render.materials.PBRMaterial(**random_material_values())
        mat.material_name = scene_path
        usd_materials.export_material(mat, file_path, scene_path)
        material_in = usd_materials.import_material(file_path, scene_path)
        assert contained_torch_equal(mat, material_in, approximate=True, print_error_context='', rtol=1e-2, atol=1e-2)

        # Write different random material, disjoint properties
        mat2 = kaolin.render.materials.PBRMaterial(
            **random_material_values(), **random_material_textures(), **random_material_colorspaces())
        mat2.material_name = scene_path
        usd_materials.export_material(mat2, file_path, scene_path)
        material_in = usd_materials.import_material(file_path, scene_path)
        assert contained_torch_equal(mat2, material_in, approximate=True, print_error_context='', rtol=1e-2, atol=1e-2)

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
            mat = kaolin.render.materials.PBRMaterial(**params)
            usd_materials.export_material(mat, stage, f'/World/Looks/{test_name}', bound_prims=[prim])
        stage.Save()

        # Confirm exported USD matches golden file
        # TODO(jlafleche) Render the two mesh for visual comparison
        golden = samples_data_path('golden', 'pbr_material_values.usda')
        assert open(golden).read() == open(out_path).read()

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_material_textures(self, device, out_dir, mesh):
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
            mat = kaolin.render.materials.PBRMaterial(**params)
            if device == "cuda":
                mat = mat.cuda()
            prim = usd.add_mesh(stage, f'/World/{test_name}', mesh.vertices, mesh.faces,
                                uvs=mesh.uvs,
                                face_uvs_idx=mesh.face_uvs_idx,
                                face_normals=mesh.normals[mesh.face_normals_idx].view(-1, 3))
            usd_materials.export_material(
                mat, stage, f'/World/Looks/{test_name}', texture_path=out_dir, bound_prims=[prim],
                overwrite_textures=True)  # overwrite, so golden files match every time
        stage.Save()

        # Confirm exported USD matches golden file
        # TODO(jlafleche) Render the two mesh for visual comparison
        golden = samples_data_path('golden', 'pbr_material_textures.usda')
        # Note: due to floating differences of the uvs, which go through some arithmetic during import, can't compare
        assert file_contents_equal(golden, out_path, exclude_pattern='primvars:st =')

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
        material = kaolin.render.materials.PBRMaterial(**texture)

        prim = usd.add_mesh(stage, '/World/colorspace_test', mesh.vertices, mesh.faces,
                            uvs=mesh.uvs,
                            face_uvs_idx=mesh.face_uvs_idx,
                            face_normals=mesh.normals[mesh.face_normals_idx].view(-1, 3))
        material_scene_path = '/World/Looks/colorspace_test'
        usd_materials.export_material(material, out_path, material_scene_path, bound_prims=[prim])

        material_in = usd_materials.import_material(out_path, material_scene_path)

        assert material_in.diffuse_colorspace == 'sRGB'
        assert material_in.metallic_colorspace == 'auto'
        assert material_in.roughness_colorspace == 'raw'

    def test_reports_missing_textures(self):
        pass

    def test_absolute_texture_paths(self):
        pass


class TestDiverseUsdPreviewSurfaceIo:
    def test_reads_real_world_textures(self):
        fname = samples_data_path('io', 'armchair.usd')
        scene_paths = ['/_materials/M_Armchair_Cushions', '/_materials/M_Armchair_Legs']

        materials = [usd_materials.import_material(fname, sp) for sp in scene_paths]
        check_allclose(materials[0].diffuse_texture,
                       kaolin.io.utils.read_image(samples_data_path('io', 'textures', 'armchair_cushions.jpg')))
        check_allclose(materials[1].diffuse_texture,
                       kaolin.io.utils.read_image(samples_data_path('io', 'textures', 'armchair_legs.jpg')))

    @pytest.mark.parametrize('bname', ['ico_flat', 'ico_smooth', 'fox', 'pizza', 'amsterdam', 'armchair'])
    def test_read_write_read_complex(self, out_dir, bname):
        fname = samples_data_path('io', f'{bname}.usd')
        out_fname = os.path.join(out_dir, f'exported_materials_{bname}.usd')

        scene_paths = usd.get_scene_paths(fname, prim_types='Material')
        assert len(scene_paths) > 0

        materials = [usd_materials.import_material(fname, sp) for sp in scene_paths]
        for mat in materials:
            usd_materials.export_material(mat, out_fname)

        # Reimport from the export; scene paths should be the same
        materials_in = [usd_materials.import_material(out_fname, sp) for sp in scene_paths]
        assert contained_torch_equal(materials, materials_in, approximate=True, print_error_context='', rtol=1e-2, atol=1e-2)

