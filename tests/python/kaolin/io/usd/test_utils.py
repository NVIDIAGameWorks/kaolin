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
import pytest
import shutil
from pxr import Usd

from kaolin.io import usd, obj
from kaolin.ops.conversions import trianglemeshes_to_voxelgrids


__test_dir = os.path.dirname(os.path.realpath(__file__))
__samples_path = os.path.join(__test_dir, os.pardir, os.pardir, os.pardir, os.pardir, 'samples')

@pytest.fixture(scope='class')
def out_dir():
    # Create temporary output directory
    out_dir = os.path.join(__test_dir, '_out')
    os.makedirs(out_dir, exist_ok=True)
    yield out_dir
    shutil.rmtree(out_dir)

@pytest.fixture(scope='module')
def mesh():
    obj_mesh = obj.import_mesh(os.path.join(__samples_path, 'rocket.obj'),
        with_normals=True, with_materials=True, error_handler=obj.skip_error_handler)
    return obj_mesh

@pytest.fixture(scope='module')
def mesh_path():
    return os.path.join(__samples_path, 'golden', 'mesh.usda')   # rocket # TODO: rename file

@pytest.fixture(scope='module')
def pointcloud():
    pointcloud, color, normals = usd.import_pointcloud(
        os.path.join(__samples_path, 'rocket_pointcloud_GeomPoints.usda'),
        '/World/pointcloud')
    return pointcloud

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

class TestMisc:
    @pytest.fixture(scope='class')
    def voxelgrid(self, mesh):
        return TestVoxelGrid.make_voxelgrid(mesh)

    def test_get_authored_time_samples_untimed(self, out_dir, mesh, voxelgrid):
        out_path = os.path.join(out_dir, 'untimed.usda')
        usd.export_voxelgrid(file_path=out_path, voxelgrid=voxelgrid, scene_path='/World/voxelgrid')
        usd.export_mesh(out_path, scene_path='/World/meshes', vertices=mesh.vertices, faces=mesh.faces)

        times = usd.get_authored_time_samples(out_path)
        assert times == []

    def test_get_authored_time_samples_timed(self, out_dir, mesh, voxelgrid, pointcloud):
        out_path = os.path.join(out_dir, 'timed.usda')
        usd.export_voxelgrid(file_path=out_path, voxelgrid=voxelgrid, scene_path='/World/voxelgrid')
        times = usd.get_authored_time_samples(out_path)
        assert times == []

        usd.export_voxelgrid(file_path=out_path, voxelgrid=voxelgrid, scene_path='/World/voxelgrid', time=1)
        times = usd.get_authored_time_samples(out_path)
        assert times == [1]

        usd.export_mesh(out_path, scene_path='/World/meshes', vertices=mesh.vertices, faces=mesh.faces, time=20)
        usd.export_mesh(out_path, scene_path='/World/meshes', vertices=mesh.vertices, faces=None, time=250)
        times = usd.get_authored_time_samples(out_path)
        assert times == [1.0, 20.0, 250.0]

        usd.export_pointcloud(out_path, pointcloud)

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_get_scene_paths(self, mesh_path, input_stage):
        paths = usd.get_scene_paths(mesh_path)
        assert len(paths) == 2

        paths = usd.get_scene_paths(mesh_path, prim_types="Mesh")
        assert len(paths) == 1

        paths = usd.get_scene_paths(mesh_path, prim_types=["Mesh"])
        assert len(paths) == 1

        paths = usd.get_scene_paths(mesh_path, scene_path_regex=".*World.*")
        assert len(paths) == 2