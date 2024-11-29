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
import copy
import os
import shutil

import torch
import pytest
import random

from kaolin.io import usd, obj
from kaolin.utils.testing import contained_torch_equal
from kaolin.render.materials import PBRMaterial, random_material_values, \
    random_material_colorspaces, random_material_textures

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
    'transmittance_colorspace',
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
    'transmittance_value'
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
    'normals_texture',
    'displacement_texture',
    'transmittance_texture'
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
def mesh():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    obj_mesh = obj.import_mesh(os.path.join(cur_dir, os.pardir, os.pardir,
                               os.pardir, 'samples/rocket.obj'), with_normals=True,
                               with_materials=True, error_handler=obj.skip_error_handler)
    return obj_mesh


def assert_string_contains(val, substrings):
    for s in substrings:
        assert s in val, f'String missing {s}: \n {val}'


class TestPBRMaterial:
    def test_supported_tensor_attributes(self):
        core_attributes = set(_value_attributes + _texture_attributes)
        missing = core_attributes.difference(PBRMaterial.supported_tensor_attributes())
        assert len(missing) == 0, f'PBRMaterial.supported_tensor_attributes is Missing core attributes: {missing}'

        missing = set(PBRMaterial.supported_tensor_attributes()).difference(core_attributes)
        assert len(missing) == 0, f'Test attributes are out of date, missing: {missing}'

    @pytest.mark.parametrize('device', ["cuda", "cpu"])
    @pytest.mark.parametrize("method_to_test", ["to", "named"])  # if to test mesh.to or mesh.cpu/mesh.cuda
    @pytest.mark.parametrize("to_device", ["cuda", "cpu"])
    def test_device_convert(self, device, to_device, method_to_test):
        input_attr = random_material_values(device)
        input_attr.update(random_material_textures(device))
        input_attr.update(random_material_colorspaces())
        input_attr['material_name'] = 'birthday party'
        # exhaustive material
        mat = PBRMaterial(**input_attr)
        # material with lots of None values
        simple_mat = PBRMaterial(diffuse_texture=input_attr['diffuse_texture'],
                                 diffuse_color=input_attr['diffuse_color'])
        assert_string_contains(simple_mat.to_string(), ['diffuse_texture', 'diffuse_color'])

        if method_to_test == "to":
            mat_converted = mat.to(to_device)
            simple_mat_converted = simple_mat.to(to_device)
        elif to_device == "cuda":
            mat_converted = mat.cuda()
            simple_mat_converted = simple_mat.cuda()
        elif to_device == "cpu":
            mat_converted = mat.cpu()
            simple_mat_converted = simple_mat.cpu()
        else:
            raise RuntimeError(f'Bug; unknown test condition.')

        # spot check: orig mat, orig device
        for orig_mat in [mat, simple_mat]:
            assert orig_mat.diffuse_texture.device.type == device
            assert orig_mat.diffuse_color.device.type == device
        # spot check: new mat, new device
        for converted in [mat_converted, simple_mat_converted]:
            assert converted.diffuse_texture.device.type == to_device
            assert converted.diffuse_color.device.type == to_device

        # exhaustive check
        for att in PBRMaterial.supported_tensor_attributes():
            assert getattr(mat, att).device.type == device
            assert getattr(mat_converted, att).device.type == to_device

        for att in set(input_attr.keys()).difference(PBRMaterial.supported_tensor_attributes()):
            assert getattr(mat, att) == getattr(mat_converted, att)

        # Test device round-trip
        assert contained_torch_equal(mat, mat_converted.to(device), approximate=True, print_error_context='mat')
        assert contained_torch_equal(simple_mat, simple_mat_converted.to(device), approximate=True, print_error_context='mat')

    @pytest.mark.parametrize('device', [None, 'cuda:0'])
    @pytest.mark.parametrize('non_blocking', [False, True])
    def test_cuda(self, device, non_blocking):
        material_values = random_material_values()
        material_textures = random_material_textures()
        mat = PBRMaterial(**material_values, **material_textures)
        cuda_mat = mat.cuda(device=device, non_blocking=non_blocking)
        for param_name in PBRMaterial.supported_tensor_attributes():
            val = getattr(mat, param_name)
            cuda_val = getattr(cuda_mat, param_name)
            if val is None:
                assert cuda_val is None
            else:
                assert torch.equal(cuda_val, val.cuda())
                assert not val.is_cuda
                assert cuda_val.is_cuda

        for param_name in _misc_attributes:
            assert getattr(mat, param_name) == getattr(cuda_mat, param_name)

    def test_contiguous(self):
        material_values = random_material_values()
        material_textures = random_material_textures()
        strided_material_textures = {
            k: torch.as_strided(v, (v.shape[0], int(v.shape[1] / 2), int(v.shape[2])), (1, 2, 2))
            for k, v in material_textures.items()
        }
        mat = PBRMaterial(**material_values, **strided_material_textures)
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

    def test_hwc_chw(self):
        input_attr = random_material_values()
        input_attr.update(random_material_textures())
        input_attr.update(random_material_colorspaces())
        input_attr['material_name'] = 'birthday party'
        mat = PBRMaterial(**input_attr)
        simple_mat = PBRMaterial(diffuse_texture=input_attr['diffuse_texture'],
                                 diffuse_color=input_attr['diffuse_color'])

        # copy to test for in-place modification errors
        mat_orig = copy.deepcopy(mat)
        simple_mat_orig = copy.deepcopy(simple_mat)

        # default test format hwc, so should be identical
        assert contained_torch_equal(mat_orig, mat.hwc(), approximate=True, print_error_context='hwc2hwc')
        assert contained_torch_equal(simple_mat_orig, simple_mat.hwc(), approximate=True, print_error_context='hwc2hwc')

        # test round trip
        assert contained_torch_equal(mat_orig, mat.chw().hwc(), approximate=True, print_error_context='round')
        assert contained_torch_equal(simple_mat_orig, simple_mat.chw().hwc(), approximate=True, print_error_context='round')

        # test chw
        mat_chw = mat.chw()
        simple_mat_chw = simple_mat.chw()

        # Orig unchanged
        assert contained_torch_equal(mat_orig, mat, approximate=True, print_error_context='orig')
        assert contained_torch_equal(simple_mat_orig, simple_mat, approximate=True, print_error_context='orig')

        # Simple check
        assert list(simple_mat_chw.diffuse_texture.shape) == \
               ([input_attr['diffuse_texture'].shape[2]] + list(input_attr['diffuse_texture'].shape[:2]))
        assert torch.allclose(input_attr['diffuse_texture'].permute(2, 0, 1), simple_mat_chw.diffuse_texture)

        # exhaustive check
        for att in _texture_attributes:
            att_chw = getattr(mat_chw, att)
            att_orig = input_attr[att]
            assert list(att_chw.shape) == ([att_orig.shape[2]] + list(att_orig.shape[:2]))
            assert torch.allclose(att_orig.permute(2, 0, 1), att_chw)

        for att in set(input_attr.keys()).difference(_texture_attributes):
            assert contained_torch_equal(getattr(mat, att), getattr(mat_chw, att), approximate=True, print_error_context=att)

    @pytest.mark.parametrize('device', [None, 'cuda:0'])
    @pytest.mark.parametrize('detailed', [True, False])
    @pytest.mark.parametrize('print_stats', [True, False])
    def test_print(self, device, detailed, print_stats):
        input_attr = random_material_values()
        input_attr.update(random_material_textures())
        input_attr.update(random_material_colorspaces())
        input_attr['material_name'] = 'magical unicorn'

        input_attr_list = list(input_attr.items())
        random.shuffle(input_attr_list)
        input_attr_partial = {v[0]: v[1] for v in input_attr_list[:len(input_attr)//2]}
        input_attr_partial['material_name'] = 'princess'

        mat = PBRMaterial(**input_attr).to(device)
        mat_partial = PBRMaterial(**input_attr_partial).to(device)
        print(mat) # check can do this
        print(mat_partial)

        str_mat = mat.to_string(print_stats=print_stats, detailed=detailed)
        str_mat_partial = mat_partial.to_string(print_stats=print_stats, detailed=detailed)
        assert input_attr['material_name'] in str_mat
        assert input_attr_partial['material_name'] in str_mat_partial
        assert len(str_mat) > len(str_mat_partial)
        for att in PBRMaterial.supported_tensor_attributes():
            assert att in str_mat, f'Missing {att} in {str_mat}'

            if att in input_attr_partial:
                assert att in str_mat_partial, f'Missing {att} in {str_mat}'
            else:
                assert f' {att}:' not in str_mat_partial, f'Should not print missing {att} in {str_mat}'