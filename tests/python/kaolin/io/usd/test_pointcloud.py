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

from kaolin.io import usd

@pytest.fixture(scope='class')
def out_dir():
    # Create temporary output directory
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_out')
    os.makedirs(out_dir, exist_ok=True)
    yield out_dir
    shutil.rmtree(out_dir)

@pytest.fixture(scope='module')
def pointcloud():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    pointcloud, color, normals = usd.import_pointcloud(
        os.path.join(cur_dir, os.pardir, os.pardir, os.pardir, os.pardir,
                     'samples/rocket_pointcloud_GeomPoints.usda'),
        '/World/pointcloud')
    return pointcloud

@pytest.fixture(scope='module')
def pointcloud_instancer():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    pointcloud, color, normals = usd.import_pointcloud(
        os.path.join(cur_dir, os.pardir, os.pardir, os.pardir, os.pardir,
                     'samples/rocket_pointcloud.v0.9.0.usda'),
        '/World/pointcloud')
    return pointcloud

@pytest.fixture(scope='module')
def pointcloud_with_color():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    pointcloud, color, normals = usd.import_pointcloud(
        os.path.join(cur_dir, os.pardir, os.pardir, os.pardir, os.pardir,
                     'samples/golden/pointcloud_GeomPoints_colors.usda'),
        '/World/pointcloud')
    return (pointcloud, color)

class TestPointCloud:
    def setup_method(self):
        self.scene_path = '/World/pointcloud'
        self.num_multiple = 3

    def test_export_single(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'pointcloud.usda')
        usd.export_pointcloud(pointcloud=pointcloud, file_path=out_path, scene_path=self.scene_path, points_type='usd_geom_points')

        # Confirm exported USD matches golden file
        golden = os.path.join(out_dir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
                              'samples/golden/pointcloud_GeomPoints.usda')
        assert open(golden).read() == open(out_path).read()

    def test_export_single_instancer(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'pointcloud_instancer.usda')
        usd.export_pointcloud(pointcloud=pointcloud, file_path=out_path, scene_path=self.scene_path)

        # Confirm exported USD matches golden file
        golden = os.path.join(out_dir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
                              'samples/golden/pointcloud_PointInstancer.usda')
        assert open(golden).read() == open(out_path).read()

    def test_export_multiple(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'pointclouds.usda')

        # Export some meshes using default scene paths
        usd.export_pointclouds(pointclouds=[pointcloud for _ in range(self.num_multiple)],
                               file_path=out_path, points_type='usd_geom_points')

        # Test that can get their scene paths later
        scene_paths = usd.get_pointcloud_scene_paths(out_path)
        assert len(scene_paths) == self.num_multiple

    def test_export_multiple_instancer(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'pointclouds_instancer.usda')

        usd.export_pointclouds(pointclouds=[pointcloud for _ in range(self.num_multiple)],
                               file_path=out_path)

        # Test that can get their scene paths later
        scene_paths = usd.get_pointcloud_scene_paths(out_path)
        assert len(scene_paths) == self.num_multiple

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_single(self, out_dir, pointcloud, input_stage):
        out_path = os.path.join(out_dir, 'pointcloud.usda')
        if input_stage:
            path_or_stage = Usd.Stage.Open(out_path)
        else:
            path_or_stage = out_path
        pointcloud_in = usd.import_pointcloud(path_or_stage, scene_path=self.scene_path).points

        # Confirm imported pointcloud matches original input
        assert torch.allclose(pointcloud, pointcloud_in)

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_multiple(self, out_dir, pointcloud, input_stage):
        out_path = os.path.join(out_dir, 'pointclouds.usda')
        if input_stage:
            path_or_stage = Usd.Stage.Open(out_path)
        else:
            path_or_stage = out_path
        pointcloud_in_list = usd.import_pointclouds(path_or_stage)

        # Confirm imported pointcloud matches original input
        assert len(pointcloud_in_list) == self.num_multiple
        for pointcloud_in, colors_in, normals_in in pointcloud_in_list:
            assert torch.allclose(pointcloud, pointcloud_in)

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_single_instancer(self, out_dir, pointcloud_instancer, input_stage):
        # Test that the read from UsdPointInstancer is the same as the read from UsdGeomPoints
        out_path = os.path.join(out_dir, 'pointcloud.usda')
        if input_stage:
            path_or_stage = Usd.Stage.Open(out_path)
        else:
            path_or_stage = out_path
        pointcloud_in, colors_in, normals_in = usd.import_pointcloud(
            path_or_stage, scene_path=self.scene_path)

        # Confirm imported pointcloud matches original input
        assert torch.allclose(pointcloud_instancer, pointcloud_in)

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_multiple_instancer(self, out_dir, pointcloud_instancer, input_stage):
        # Test that the read from UsdPointInstancer is the same as the read from UsdGeomPoints
        out_path = os.path.join(out_dir, 'pointclouds.usda')
        if input_stage:
            path_or_stage = Usd.Stage.Open(out_path)
        else:
            path_or_stage = out_path
        pointcloud_in_list = usd.import_pointclouds(path_or_stage)

        # Confirm imported pointcloud matches original input
        assert len(pointcloud_in_list) == self.num_multiple
        for pointcloud_in, colors_in, normals_in in pointcloud_in_list:
            assert torch.allclose(pointcloud_instancer, pointcloud_in)

    def test_export_single_colors(self, out_dir, pointcloud_with_color):
        # Export a single pointcloud with colors
        pointcloud, color = pointcloud_with_color

        out_path = os.path.join(out_dir, 'pointcloud_colors.usda')
        usd.export_pointcloud(pointcloud=pointcloud, file_path=out_path, color=color,
                              scene_path=self.scene_path, points_type='usd_geom_points')

        # Confirm exported USD matches golden file
        golden = os.path.join(out_dir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
                              'samples/golden/pointcloud_GeomPoints_colors.usda')
        assert open(golden).read() == open(out_path).read()

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_single_color(self, out_dir, pointcloud, input_stage):
        out_path = os.path.join(out_dir, 'pointcloud_colors.usda')
        if input_stage:
            path_or_stage = Usd.Stage.Open(out_path)
        else:
            path_or_stage = out_path
        pointcloud_in, color, _ = usd.import_pointcloud(path_or_stage, scene_path=self.scene_path)

        # Confirm imported pointcloud matches original input
        assert torch.allclose(pointcloud, pointcloud_in)

        # Confirm that points have the same shape as color
        assert pointcloud_in.shape == color.shape
