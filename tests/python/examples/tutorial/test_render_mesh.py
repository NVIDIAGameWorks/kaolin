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

import os
import sys

import pytest
import shutil
import subprocess
import torch

import kaolin
import kaolin.render.easy_render as easy_render
from kaolin.io.utils import read_image
from kaolin.utils.testing import tensor_info, assert_images_close

__test_dir = os.path.dirname(os.path.realpath(__file__))
__samples_path = os.path.join(__test_dir, os.pardir, os.pardir, os.pardir, 'samples')
_root_dir = os.path.realpath(os.path.join(__test_dir, os.pardir, os.pardir, os.pardir, os.pardir))


def data_path(*args):
    """ Return path relative to tests/samples/io"""
    return os.path.join(__samples_path, 'render', 'easy_render', *args)


def gt_image_float(basename):
    return read_image(data_path('mesh', basename))


@pytest.fixture(scope='function')
def out_dir():
    # Create temporary output directory
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_render_mesh_out')
    os.makedirs(out_dir, exist_ok=True)
    yield out_dir
    shutil.rmtree(out_dir)  # Note: comment to keep output directory


class TestRenderMeshMain:
    @pytest.mark.parametrize('bname', ['armchair', 'avocado'])
    @pytest.mark.parametrize('resolution', [512, 300])
    @pytest.mark.parametrize('default_material', [None, '255,25,25'])
    def test_runs(self, bname, resolution, default_material, out_dir):
        assert len(os.listdir(out_dir)) == 0, f'Configure test to recreate dir every time'

        include_bname = resolution == 512  # no need to test all permutations

        fname = data_path(f'{bname}.gltf')
        args = f'--mesh_filename={fname} --resolution={resolution} --output_dir={out_dir}'
        if default_material is not None:
            args += f' --use_default_material={default_material}'
        if include_bname:
            args += f' --base_name={bname}'

        # Check that the main function runs
        # Note: to run and capture output do:
        # pytest --capture=tee-sys tests/python/examples/
        res = subprocess.run('cd {}; python examples/tutorial/render_mesh.py {}'.format(_root_dir, args),
                             stderr=sys.stderr, stdout=sys.stdout,
                             shell=True, check=True)


        # Dir now has output files
        assert len(os.listdir(out_dir)) > 0

        # Spot check one of the outputs
        checked_passes = [easy_render.RenderPass.render, easy_render.RenderPass.normals, easy_render.RenderPass.albedo]
        for p in checked_passes:
            if include_bname:
                expected_fname = os.path.join(out_dir, f'{bname}_{p}.png')
                assert os.path.exists(expected_fname)
                img = read_image(expected_fname)
                assert img.shape[0] == resolution
                assert img.shape[1] == resolution
                if p == easy_render.RenderPass.render and default_material is None:
                    expected_img = gt_image_float(f'{bname}_render.png')
                    assert_images_close(img, expected_img)
