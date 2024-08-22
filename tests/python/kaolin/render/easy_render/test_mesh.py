# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import shutil
import os
import torch
import pytest

import kaolin
from kaolin.io import import_mesh
from kaolin.io.utils import read_image, write_image
import kaolin.render.easy_render as easy_render
from kaolin.utils.testing import assert_images_close

__test_dir = os.path.dirname(os.path.realpath(__file__))
__samples_path = os.path.join(__test_dir, os.pardir, os.pardir, os.pardir, os.pardir, 'samples')


def render_data_path(*args):
    """ Return path relative to tests/samples/io"""
    return os.path.join(__samples_path, 'render', 'easy_render', *args)


def gt_image_float(basename):
    return read_image(render_data_path('mesh', basename))


@pytest.fixture(scope='class')
def out_dir():
    # Create temporary output directory
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_out')
    os.makedirs(out_dir, exist_ok=True)
    yield out_dir
    shutil.rmtree(out_dir)


class TestEasyRender:
    @pytest.mark.parametrize('bname', ['armchair', 'avocado'])
    @pytest.mark.parametrize('backend', ['cuda', None, 'nvdiffrast'])
    def test_obj_usd_gltf_render_consistency(self, bname, backend):
        if os.getenv('KAOLIN_TEST_NVDIFFRAST', '0') == '0' and backend == 'nvdiffrast':
            pytest.skip(f'test is ignored as KAOLIN_TEST_NVDIFFRAST is not set')

        # default settings
        camera = easy_render.default_camera(512).cuda()
        lighting = easy_render.default_lighting().cuda()

        # check full render
        gt_image = gt_image_float(f'{bname}_render.png').cuda()

        for ext in ['obj', 'usd', 'gltf']:
            mesh = import_mesh(render_data_path(f'{bname}.{ext}'), triangulate=True).cuda()
            mesh.vertices = kaolin.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)
            res = easy_render.render_mesh(camera, mesh, lighting=lighting, backend=backend)
            assert_images_close(gt_image, res[easy_render.RenderPass.render.name].squeeze(0))

    @pytest.mark.parametrize('bname', ['ico_flat', 'ico_smooth', 'fox', 'pizza', 'amsterdam'])
    @pytest.mark.parametrize('backend', ['cuda', 'nvdiffrast'])
    @pytest.mark.parametrize('with_features', [True, False])
    def test_render_all(self, out_dir, bname, backend, with_features):
        if os.getenv('KAOLIN_TEST_NVDIFFRAST', '0') == '0' and backend == 'nvdiffrast':
            pytest.skip(f'test is ignored as KAOLIN_TEST_NVDIFFRAST is not set')

        camera = easy_render.default_camera(512).cuda()
        lighting = easy_render.default_lighting().cuda()

        # full render
        mesh = import_mesh(render_data_path(f'{bname}.usd'), triangulate=True).cuda()
        mesh.vertices = kaolin.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)
        if with_features:
            mesh.face_features = torch.rand((mesh.faces.shape[0], mesh.faces.shape[1], 6)).float().cuda()

        res = easy_render.render_mesh(camera, mesh, lighting=lighting, backend=backend)
        for pass_name in [easy_render.RenderPass.render.name]:  # just compare full render for now
            gt_image = gt_image_float(f'{bname}_{pass_name}.png').cuda()
            assert_images_close(gt_image, res[pass_name].squeeze(0))

        if with_features:
            assert easy_render.RenderPass.features.name in res.keys(), f'Features not rendered'

    @pytest.mark.skipif(os.getenv('KAOLIN_TEST_NVDIFFRAST', '0') == '0', reason='test is ignored as KAOLIN_TEST_NVDIFFRAST is not set')
    @pytest.mark.parametrize('bname', ['ico_flat', 'ico_smooth', 'fox', 'pizza', 'amsterdam'])
    @pytest.mark.parametrize('with_features', [True, False])
    def test_render_comp(self, out_dir, bname, with_features):
        camera = easy_render.default_camera(512).cuda()
        lighting = easy_render.default_lighting().cuda()

        # full render
        mesh = import_mesh(render_data_path(f'{bname}.usd'), triangulate=True).cuda()
        mesh.vertices = kaolin.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)
        if with_features:
            mesh.face_features = torch.rand((mesh.faces.shape[0], mesh.faces.shape[1], 6)).float().cuda()

        res_cuda = easy_render.render_mesh(camera, mesh, lighting=lighting, backend='cuda')
        res_nvdif = easy_render.render_mesh(camera, mesh, lighting=lighting, backend='nvdiffrast')

        assert set(res_cuda.keys()) == set(res_nvdif.keys())
        for pass_name in res_cuda.keys():
            if pass_name == easy_render.RenderPass.features.name:
                # TODO: why a bigger difference for features?
                assert_images_close(res_cuda[pass_name].squeeze(0), res_nvdif[pass_name].squeeze(0),
                                    pixel_disagreement_threshold=0.1)
            elif pass_name == easy_render.RenderPass.face_idx:
                # Almost all faces should be the same
                assert_images_close(res_cuda[pass_name].squeeze(0).unsqueeze(-1).double(),
                                    res_nvdif[pass_name].squeeze(0).unsqueeze(-1).double(),
                                    pixel_disagreement_threshold=1, max_percent_disagreeing_pixels=3,
                                    check_range=False)
            elif pass_name not in [easy_render.RenderPass.uvs.name]:
                assert_images_close(res_cuda[pass_name].squeeze(0), res_nvdif[pass_name].squeeze(0))

    @pytest.mark.parametrize('bname', ['armchair'])
    @pytest.mark.parametrize('backend', ['cuda', 'nvdiffrast'])
    @pytest.mark.parametrize('material_variant', ['full', 'default', 'empty'])
    @pytest.mark.parametrize('lighting_variant', ['default', 'none'])  # TODO: add batched lighting
    @pytest.mark.parametrize('normals_variant', ['normals', 'nonormals'])
    @pytest.mark.parametrize('uvs_variant', ['uvs', 'nouvs'])
    def test_missing_attributes(self, out_dir, bname, backend, material_variant, lighting_variant, normals_variant, uvs_variant):
        if os.getenv('KAOLIN_TEST_NVDIFFRAST', '0') == '0' and backend == 'nvdiffrast':
            pytest.skip(f'test is ignored as KAOLIN_TEST_NVDIFFRAST is not set')

        camera = easy_render.default_camera(144).cuda()

        # full render
        mesh = import_mesh(render_data_path(f'{bname}.usd'), triangulate=True).cuda()
        mesh.vertices = kaolin.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)

        if material_variant == 'default':
            mesh.materials = [easy_render.default_material().cuda()]
            mesh.material_assignments[...] = 0
        elif material_variant == 'empty':
            mesh.materials = None
            mesh.material_assignments = None
        elif material_variant != 'full':
            raise RuntimeError(f'Unknown material variant {material_variant}')

        if lighting_variant == 'default':
            lighting = easy_render.default_lighting().cuda()
        elif lighting_variant == 'none':
            lighting = None
        else:
            raise RuntimeError(f'Unknown lighting variant {lighting_variant}')

        if normals_variant == 'nonormals':
            mesh.face_normals = None
            mesh.vertex_normals = None
            mesh.normals = None
        elif normals_variant != 'normals':
            raise RuntimeError(f'Unknown normals variant {normals_variant}')

        if uvs_variant == 'nouvs':
            mesh.face_uvs = None
            mesh.uvs = None
            mesh.face_uvs_idx = None
        elif uvs_variant != 'uvs':
            raise RuntimeError(f'Unknown uvs variant {uvs_variant}')

        # Test that does not crash
        res = easy_render.render_mesh(camera, mesh, lighting=lighting, backend=backend)
        assert res[easy_render.RenderPass.render] is not None

        # TODO: ensure all of these are sane and add ground truth images
        # TODO: e.g. if no uvs, should use a different material that is more visible?
        # bname_out = f'{bname}_{backend}_render_mat{material_variant}_light{lighting_variant}_{normals_variant}_{uvs_variant}.png'
        # write_image(res[easy_render.RenderPass.render.name], os.path.join(out_dir, bname_out))

