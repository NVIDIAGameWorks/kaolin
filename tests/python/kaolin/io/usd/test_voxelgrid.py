# Copyright (c) 2019,20-21-23 NVIDIA CORPORATION & AFFILIATES.
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

from pxr import Usd

from kaolin.io import usd, obj
from kaolin.ops.conversions import trianglemeshes_to_voxelgrids


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
    obj_mesh = obj.import_mesh(
        os.path.join(cur_dir, os.pardir, os.pardir,  os.pardir, os.pardir, 'samples/rocket.obj'),
        with_normals=True, with_materials=True, error_handler=obj.skip_error_handler)
    return obj_mesh

class TestVoxelGrid:
    def setup_method(self):
        self.scene_path = '/World/voxelgrid'
        self.num_multiple = 3

    @staticmethod
    def make_voxelgrid(mesh):
        resolution = 64
        voxelgrid = trianglemeshes_to_voxelgrids(mesh.vertices.unsqueeze(0), mesh.faces,
                                                 resolution)
        return voxelgrid[0].bool()

    @pytest.fixture(scope='class')
    def voxelgrid(self, mesh):
        return TestVoxelGrid.make_voxelgrid(mesh)

    def test_export_single(self, out_dir, voxelgrid):
        out_path = os.path.join(out_dir, 'voxelgrid.usda')
        usd.export_voxelgrid(file_path=out_path, voxelgrid=voxelgrid, scene_path=self.scene_path)

        # Confirm exported USD matches golden file
        golden = os.path.join(out_dir, os.pardir, os.pardir,  os.pardir, os.pardir, os.pardir,
                              'samples/golden/voxelgrid.usda')
        assert open(golden).read() == open(out_path).read()

    def test_export_multiple(self, out_dir, voxelgrid):
        out_path = os.path.join(out_dir, 'voxelgrids.usda')

        # Export multiple voxelgrids using default paths
        usd.export_voxelgrids(file_path=out_path, voxelgrids=[voxelgrid for _ in range(self.num_multiple)])

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_single(self, out_dir, voxelgrid, input_stage):
        out_path = os.path.join(out_dir, 'voxelgrid.usda')
        if input_stage:
            path_or_stage = Usd.Stage.Open(out_path)
        else:
            path_or_stage = out_path
        voxelgrid_in = usd.import_voxelgrid(path_or_stage, scene_path=self.scene_path)
        assert torch.equal(voxelgrid, voxelgrid_in)

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_multiple(self, out_dir, voxelgrid, input_stage):
        out_path = os.path.join(out_dir, 'voxelgrids.usda')
        if input_stage:
            path_or_stage = Usd.Stage.Open(out_path)
        else:
            path_or_stage = out_path
        voxelgrid_in_list = usd.import_voxelgrids(path_or_stage)

        # Confirm imported voxelgrid matches original input
        assert len(voxelgrid_in_list) == self.num_multiple
        for voxelgrid_in in voxelgrid_in_list:
            assert torch.equal(voxelgrid, voxelgrid_in)
