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
import filecmp
import random

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
    def setup_method(self, out_dir):
        self.scene_path = '/World/voxelgrid'
        self.num_multiple = 3

    @pytest.fixture(scope='class')
    def golden_path(self, out_dir):
        return os.path.join(
            out_dir, os.pardir, os.pardir,  os.pardir, os.pardir, os.pardir,
            'samples/golden/voxelgrid.usda'
        )

    @pytest.fixture(scope='class')
    def voxelgrid(self, mesh):
        resolution = 64
        voxelgrid = trianglemeshes_to_voxelgrids(mesh.vertices.unsqueeze(0), mesh.faces,
                                                 resolution)
        return voxelgrid[0].bool()

    def test_export_single(self, out_dir, voxelgrid, golden_path):
        out_path = os.path.join(out_dir, 'voxelgrid.usda')
        usd.export_voxelgrid(file_path=out_path, voxelgrid=voxelgrid, scene_path=self.scene_path)

        # Confirm exported USD matches golden file
        assert filecmp.cmp(golden_path, out_path)

    def test_export_multiple(self, out_dir, voxelgrid):
        out_path = os.path.join(out_dir, 'voxelgrids.usda')
        voxelgrids = [voxelgrid for _ in range(self.num_multiple)]

        # Export multiple voxelgrids using default paths
        usd.export_voxelgrids(file_path=out_path, voxelgrids=voxelgrids)

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_single(self, golden_path, voxelgrid, input_stage):
        if input_stage:
            path_or_stage = Usd.Stage.Open(golden_path)
        else:
            path_or_stage = golden_path
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


    def test_fail_export_twice(self, out_dir, voxelgrid):
        out_path = os.path.join(out_dir, 'fail_exported_twice.usda')
        usd.export_voxelgrid(out_path, voxelgrid=voxelgrid)
        with pytest.raises(FileExistsError):
            usd.export_voxelgrid(out_path, voxelgrid=voxelgrid)

    def test_export_twice(self, out_dir, voxelgrid, golden_path):
        out_path = os.path.join(out_dir, 'exported_twice.usda')
        usd.export_voxelgrid(out_path, voxelgrid=(torch.rand(32, 32, 32) > 0.7))
        usd.export_voxelgrid(out_path, voxelgrid=voxelgrid, scene_path=self.scene_path, overwrite=True)
        assert filecmp.cmp(out_path, golden_path)

    def test_export_overwrite_full_file(self, out_dir, voxelgrid, golden_path):
        out_path = os.path.join(out_dir, 'exported_overwrite_full_file.usda')
        usd.export_voxelgrids(out_path, voxelgrids=[voxelgrid for _ in range(self.num_multiple)])
        usd.export_voxelgrid(out_path, voxelgrid=voxelgrid, scene_path=self.scene_path, overwrite=True)
        assert filecmp.cmp(out_path, golden_path)

    def test_export_overwrite_pointcloud(self, out_dir, voxelgrid, golden_path):
        out_path = os.path.join(out_dir, 'exported_overwrite_pointcloud.usda')
        usd.export_pointcloud(out_path, pointcloud=torch.rand((100, 3)))
        usd.export_voxelgrid(out_path, voxelgrid=voxelgrid, scene_path=self.scene_path, overwrite=True)
        assert filecmp.cmp(out_path, golden_path)

    def test_export_overwrite_mesh(self, out_dir, mesh, voxelgrid, golden_path):
        out_path = os.path.join(out_dir, 'exported_overwrite_voxelgrid.usda')
        usd.export_mesh(out_path, vertices=mesh.vertices, faces=mesh.faces)
        usd.export_voxelgrid(out_path, voxelgrid=voxelgrid, scene_path=self.scene_path, overwrite=True)
        assert filecmp.cmp(out_path, golden_path)

