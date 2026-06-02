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
import warnings

import torch
import pytest

from pxr import Usd

from kaolin.io import usd
from kaolin.io.usd.pointcloud import pointcloud_return_type
from kaolin.utils.testing import contained_torch_equal

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
    pointcloud, color, normals, transform = usd.import_pointcloud(
        os.path.join(cur_dir, os.pardir, os.pardir, os.pardir, os.pardir,
                     'samples/rocket_pointcloud_GeomPoints.usda'),
        scene_path='/World/pointcloud')
    return pointcloud

@pytest.fixture(scope='module')
def pointcloud_instancer():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    pointcloud, color, normals, transform = usd.import_pointcloud(
        os.path.join(cur_dir, os.pardir, os.pardir, os.pardir, os.pardir,
                     'samples/rocket_pointcloud.v0.9.0.usda'),
        scene_path='/World/pointcloud')
    return pointcloud

@pytest.fixture(scope='module')
def pointcloud_with_color():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    pointcloud, color, normals, transform = usd.import_pointcloud(
        os.path.join(cur_dir, os.pardir, os.pardir, os.pardir, os.pardir,
                     'samples/golden/pointcloud_GeomPoints_colors.usda'),
        scene_path='/World/pointcloud')
    return (pointcloud, color)


def _expected_world_points(points, local_to_world):
    """Apply a (4, 4) local-to-world transform (translation in last column) to (N, 3) points."""
    homo = torch.cat([points, torch.ones(points.shape[0], 1)], dim=1)
    return (local_to_world @ homo.T).T[:, :3]


class TestPointCloud:
    def setup_method(self):
        self.scene_path = '/World/pointcloud'
        self.num_multiple = 3

    def test_export_single(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'pointcloud.usda')
        usd.export_pointcloud(points=pointcloud, file_path=out_path, scene_path=self.scene_path, points_type='usd_geom_points')

        # Confirm exported USD matches golden file
        golden = os.path.join(out_dir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
                              'samples/golden/pointcloud_GeomPoints.usda')
        assert filecmp.cmp(golden, out_path)

    def test_export_single_instancer(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'pointcloud_instancer.usda')
        usd.export_pointcloud(points=pointcloud, file_path=out_path, scene_path=self.scene_path)

        # Confirm exported USD matches golden file
        golden = os.path.join(out_dir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
                              'samples/golden/pointcloud_PointInstancer.usda')
        assert filecmp.cmp(golden, out_path)

    def test_export_multiple(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'pointclouds.usda')

        # Export some meshes using default scene paths
        usd.export_pointclouds(points=[pointcloud for _ in range(self.num_multiple)],
                               file_path=out_path, points_type='usd_geom_points')

        # Test that can get their scene paths later
        scene_paths = usd.get_pointcloud_scene_paths(out_path)
        assert len(scene_paths) == self.num_multiple

    def test_export_multiple_instancer(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'pointclouds_instancer.usda')

        usd.export_pointclouds(points=[pointcloud for _ in range(self.num_multiple)],
                               file_path=out_path)

        # Test that can get their scene paths later
        scene_paths = usd.get_pointcloud_scene_paths(out_path)
        assert len(scene_paths) == self.num_multiple

    def test_get_pointcloud_scene_paths_scene_path(self, out_dir, pointcloud):
        """get_pointcloud_scene_paths with scene_path returns only paths under that prefix"""
        foo_paths = ['/World/Foo/pc_0', '/World/Foo/pc_1']
        bar_paths = ['/World/Bar/pc_0']
        all_scene_paths = foo_paths + bar_paths
        out_path = os.path.join(out_dir, 'pointclouds_scene_path.usda')
        usd.export_pointclouds(points=[pointcloud for _ in all_scene_paths],
                               scene_paths=all_scene_paths, file_path=out_path,
                               points_type='usd_geom_points')
        actual = usd.get_pointcloud_scene_paths(out_path, scene_path='/World/Foo')
        assert set(str(p) for p in actual) == set(foo_paths)

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_single(self, out_dir, pointcloud, input_stage):
        golden_path = os.path.join(out_dir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
                                   'samples/golden/pointcloud_PointInstancer.usda')
        if input_stage:
            path_or_stage = Usd.Stage.Open(golden_path)
        else:
            path_or_stage = golden_path
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
        scene_paths = usd.get_pointcloud_scene_paths(out_path)
        pointcloud_in_dict = usd.import_pointclouds(path_or_stage, scene_paths, return_list=False)

        # Confirm imported pointcloud matches original input
        assert len(pointcloud_in_dict) == self.num_multiple
        for key, (pointcloud_in, colors_in, normals_in, transform_in) in pointcloud_in_dict.items():
            assert isinstance(key, str)
            assert torch.allclose(pointcloud, pointcloud_in)

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_all(self, out_dir, pointcloud, input_stage):
        out_path = os.path.join(out_dir, 'pointclouds.usda')
        if input_stage:
            path_or_stage = Usd.Stage.Open(out_path)
        else:
            path_or_stage = out_path
        # With no scene_paths, import_pointclouds discovers every pointcloud in the file.
        pointclouds_dict = usd.import_pointclouds(path_or_stage, return_list=False)

        assert len(pointclouds_dict) == self.num_multiple
        for key, (pointcloud_in, colors_in, normals_in, transform_in) in pointclouds_dict.items():
            assert isinstance(key, str)
            assert torch.allclose(pointcloud, pointcloud_in)

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_single_instancer(self, out_dir, pointcloud_instancer, input_stage):
        # Test that the read from UsdPointInstancer is the same as the read from UsdGeomPoints
        out_path = os.path.join(out_dir, 'pointcloud.usda')
        if input_stage:
            path_or_stage = Usd.Stage.Open(out_path)
        else:
            path_or_stage = out_path
        pointcloud_in, colors_in, normals_in, transform_in = usd.import_pointcloud(
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
        scene_paths = usd.get_pointcloud_scene_paths(out_path)
        pointcloud_in_dict = usd.import_pointclouds(path_or_stage, scene_paths, return_list=False)

        # Confirm imported pointcloud matches original input
        assert len(pointcloud_in_dict) == self.num_multiple
        for pointcloud_in, colors_in, normals_in, transform_in in pointcloud_in_dict.values():
            assert torch.allclose(pointcloud_instancer, pointcloud_in)

    def test_export_single_colors(self, out_dir, pointcloud_with_color):
        # Export a single pointcloud with colors
        pointcloud, color = pointcloud_with_color

        out_path = os.path.join(out_dir, 'pointcloud_colors.usda')
        usd.export_pointcloud(points=pointcloud, file_path=out_path, colors=color,
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
        pointcloud_in, color, _, _ = usd.import_pointcloud(path_or_stage, scene_path=self.scene_path)

        # Confirm imported pointcloud matches original input
        assert torch.allclose(pointcloud, pointcloud_in)

        # Confirm that points have the same shape as color
        assert pointcloud_in.shape == color.shape

############################################
    def test_fail_export_twice(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'fail_exported_twice.usda')
        usd.export_pointcloud(points=pointcloud + 0.1, file_path=out_path, scene_path=self.scene_path)
        with pytest.raises(FileExistsError):
            usd.export_pointcloud(points=pointcloud, file_path=out_path, scene_path=self.scene_path)

    def test_export_twice(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'exported_twice.usda')
        usd.export_pointcloud(points=pointcloud + 0.1, file_path=out_path, scene_path=self.scene_path)
        usd.export_pointcloud(points=pointcloud, file_path=out_path, scene_path=self.scene_path, overwrite=True)
        golden_path = os.path.join(out_dir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
                                  'samples/golden/pointcloud_PointInstancer.usda')
        assert filecmp.cmp(out_path, golden_path)

    def test_export_overwrite_full_file(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'exported_overwrite_full_file.usda')
        usd.export_pointclouds(points=[pointcloud + i + 1 for i in range(self.num_multiple)],
                               file_path=out_path)
        usd.export_pointcloud(points=pointcloud, file_path=out_path, scene_path=self.scene_path, overwrite=True)
        golden_path = os.path.join(out_dir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
                                  'samples/golden/pointcloud_PointInstancer.usda')
        assert filecmp.cmp(out_path, golden_path)

    def test_export_overwrite_voxelgrid(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'exported_overwrite_pointcloud.usda')
        usd.export_voxelgrid(out_path, voxelgrid=torch.rand((32, 32, 32)) > 0.7, scene_path=self.scene_path)
        usd.export_pointcloud(points=pointcloud, file_path=out_path, scene_path=self.scene_path, overwrite=True)
        golden_path = os.path.join(out_dir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
                                  'samples/golden/pointcloud_PointInstancer.usda')
        assert filecmp.cmp(out_path, golden_path)

    def test_export_overwrite_mesh(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'exported_overwrite_voxelgrid.usda')
        usd.export_mesh(out_path, vertices=torch.rand((3, 3)), faces=torch.tensor([[0, 1, 2]], dtype=torch.long))
        usd.export_pointcloud(points=pointcloud, file_path=out_path, scene_path=self.scene_path, overwrite=True)
        golden_path = os.path.join(out_dir, os.pardir, os.pardir, os.pardir, os.pardir, os.pardir,
                                  'samples/golden/pointcloud_PointInstancer.usda')
        assert filecmp.cmp(out_path, golden_path)

    def test_local_to_world_roundtrip(self, out_dir, pointcloud):
        """Exporting with a non-identity local_to_world and re-importing returns world-space points."""
        out_path = os.path.join(out_dir, 'pointcloud_with_transform.usda')
        # Translation by (1, 2, 3) and scale by 2 along x. Translation lives in the last column
        # in the torch convention used by set/get_local_to_world_transform.
        local_to_world = torch.tensor([
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        usd.export_pointcloud(points=pointcloud, file_path=out_path, scene_path=self.scene_path,
                              local_to_world=local_to_world, points_type='usd_geom_points', overwrite=True)
        imported = usd.import_pointcloud(out_path, scene_path=self.scene_path)
        expected = pointcloud.clone()
        expected[:, 0] = expected[:, 0] * 2.0 + 1.0
        expected[:, 1] = expected[:, 1] + 2.0
        expected[:, 2] = expected[:, 2] + 3.0
        assert torch.allclose(imported.points, expected, atol=1e-5)
        # The transform is baked into the merged points, so none is exposed.
        assert imported.transform is None

    def test_export_pointclouds_broadcast_transform(self, out_dir, pointcloud):
        """A single (4, 4) local_to_world is broadcast to every exported pointcloud."""
        out_path = os.path.join(out_dir, 'pointclouds_broadcast_transform.usda')
        scene_paths = ['/World/PointClouds/pc_0', '/World/PointClouds/pc_1']
        points_list = [pointcloud, pointcloud + 1.0]
        local_to_world = torch.tensor([
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        usd.export_pointclouds(out_path, scene_paths=scene_paths, points=points_list,
                               local_to_world=local_to_world, points_type='usd_geom_points',
                               overwrite=True)
        for scene_path, pts in zip(scene_paths, points_list):
            imported = usd.import_pointcloud(out_path, scene_path=scene_path)
            assert torch.allclose(imported.points, _expected_world_points(pts, local_to_world), atol=1e-5)

    def test_export_pointclouds_batched_transform(self, out_dir, pointcloud):
        """A batched (N, 4, 4) local_to_world applies one transform per pointcloud."""
        out_path = os.path.join(out_dir, 'pointclouds_batched_transform.usda')
        scene_paths = ['/World/PointClouds/pc_0', '/World/PointClouds/pc_1']
        points_list = [pointcloud, pointcloud + 1.0]
        transform_0 = torch.tensor([
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        transform_1 = torch.tensor([
            [1.0, 0.0, 0.0, -5.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        local_to_world = torch.stack([transform_0, transform_1], dim=0)
        usd.export_pointclouds(out_path, scene_paths=scene_paths, points=points_list,
                               local_to_world=local_to_world, points_type='usd_geom_points',
                               overwrite=True)
        for scene_path, pts, tfm in zip(scene_paths, points_list, [transform_0, transform_1]):
            imported = usd.import_pointcloud(out_path, scene_path=scene_path)
            assert torch.allclose(imported.points, _expected_world_points(pts, tfm), atol=1e-5)

    def test_import_pointclouds_keeps_local_space_and_transform(self, out_dir, pointcloud):
        """import_pointclouds returns local-space points and exposes local_to_world via transform."""
        out_path = os.path.join(out_dir, 'pointclouds_local_transform.usda')
        scene_paths = ['/World/PointClouds/pc_0', '/World/PointClouds/pc_1']
        points_list = [pointcloud, pointcloud + 1.0]
        transform_0 = torch.tensor([
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        transform_1 = torch.tensor([
            [1.0, 0.0, 0.0, -5.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=torch.float32)
        local_to_world = torch.stack([transform_0, transform_1], dim=0)
        usd.export_pointclouds(out_path, scene_paths=scene_paths, points=points_list,
                               local_to_world=local_to_world, points_type='usd_geom_points',
                               overwrite=True)
        imported = usd.import_pointclouds(out_path, scene_paths, return_list=False)
        for scene_path, pts, tfm in zip(scene_paths, points_list, [transform_0, transform_1]):
            entry = imported[scene_path]
            # Points are returned in local space (transform not applied).
            assert torch.allclose(entry.points, pts, atol=1e-5)
            # The local_to_world transform is exposed instead.
            assert entry.transform is not None
            assert torch.allclose(entry.transform, tfm, atol=1e-5)
            # Applying the exposed transform yields the same world-space points that the
            # singular (merging) importer produces, tying the two APIs together.
            merged = usd.import_pointcloud(out_path, scene_path=scene_path)
            assert torch.allclose(_expected_world_points(entry.points, entry.transform),
                                  merged.points, atol=1e-5)
            # The merging importer bakes the transform in, so it exposes none.
            assert merged.transform is None

    def test_import_pointclouds_identity_transform_is_none(self, out_dir, pointcloud):
        """A prim with an identity (or absent) transform yields transform=None."""
        out_path = os.path.join(out_dir, 'pointclouds_identity_transform.usda')
        usd.export_pointclouds(file_path=out_path, points=[pointcloud, pointcloud],
                               points_type='usd_geom_points', overwrite=True)
        imported = usd.import_pointclouds(out_path)
        for entry in imported:
            assert entry.transform is None
            assert torch.allclose(entry.points, pointcloud, atol=1e-5)

    def test_export_pointclouds_transform_mismatch(self, out_dir, pointcloud):
        """A batched local_to_world whose length != number of pointclouds raises ValueError."""
        out_path = os.path.join(out_dir, 'pointclouds_transform_mismatch.usda')
        points_list = [pointcloud, pointcloud + 1.0]
        # 3 transforms for 2 pointclouds
        local_to_world = torch.eye(4, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
        with pytest.raises(ValueError):
            usd.export_pointclouds(out_path, points=points_list, local_to_world=local_to_world,
                                   points_type='usd_geom_points', overwrite=True)


class TestV018Deprecation:
    """Verify v0.18-style call sites still work and emit DeprecationWarning."""

    def setup_method(self):
        self.scene_path = '/World/pointcloud'

    @pytest.fixture
    def golden_pc_path(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        return os.path.join(cur_dir, os.pardir, os.pardir, os.pardir, os.pardir,
                            'samples/golden/pointcloud_PointInstancer.usda')

    def test_import_pointclouds_default_returns_list(self, out_dir, pointcloud):
        # v0.18 default: returns list
        out_path = os.path.join(out_dir, 'legacy_default_list.usda')
        usd.export_pointclouds(file_path=out_path, points=[pointcloud, pointcloud], overwrite=True)
        result = usd.import_pointclouds(out_path)
        expected = [pointcloud_return_type(pointcloud, None, None, None),
                    pointcloud_return_type(pointcloud, None, None, None)]
        assert isinstance(result, list)
        assert contained_torch_equal(result, expected, approximate=True, atol=1e-5)

    def test_add_pointcloud_legacy_positional_order(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'legacy_add_pc.usda')
        stage = usd.create_stage(out_path)
        # v0.18 order: (stage, points, scene_path)
        with pytest.warns(DeprecationWarning, match="positional argument order"):
            usd.add_pointcloud(stage, pointcloud, self.scene_path, points_type='usd_geom_points')
        stage.Save()
        result = usd.import_pointcloud(out_path, scene_path=self.scene_path)
        # usd_geom_points used, no colors supplied → colors must be None.
        expected = pointcloud_return_type(pointcloud, None, None, None)
        assert contained_torch_equal(result, expected, approximate=True, atol=1e-5)

    def test_export_pointcloud_legacy_positional(self, out_dir, pointcloud):
        # v0.18: export_pointcloud(file, pointcloud_tensor)
        out_path = os.path.join(out_dir, 'legacy_export_pc_positional.usda')
        with pytest.warns(DeprecationWarning, match="positional"):
            usd.export_pointcloud(out_path, pointcloud, overwrite=True)
        result = usd.import_pointcloud(out_path)
        expected = pointcloud_return_type(pointcloud, None, None, None)
        assert contained_torch_equal(result, expected, approximate=True, atol=1e-5)

    def test_export_pointcloud_legacy_positional_with_scene_path(self, out_dir, pointcloud):
        # v0.18: export_pointcloud(file, pointcloud_tensor, '/scene')
        out_path = os.path.join(out_dir, 'legacy_export_pc_positional_scene.usda')
        with pytest.warns(DeprecationWarning, match="positional"):
            usd.export_pointcloud(out_path, pointcloud, self.scene_path, overwrite=True)
        result = usd.import_pointcloud(out_path, scene_path=self.scene_path)
        expected = pointcloud_return_type(pointcloud, None, None, None)
        assert contained_torch_equal(result, expected, approximate=True, atol=1e-5)

    def test_export_pointcloud_legacy_pointcloud_kwarg(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'legacy_export_pointcloud_kwarg.usda')
        with pytest.warns(DeprecationWarning, match="pointcloud"):
            usd.export_pointcloud(out_path, pointcloud=pointcloud, scene_path=self.scene_path, overwrite=True)
        result = usd.import_pointcloud(out_path, scene_path=self.scene_path)
        expected = pointcloud_return_type(pointcloud, None, None, None)
        assert contained_torch_equal(result, expected, approximate=True, atol=1e-5)

    def test_export_pointcloud_legacy_color_kwarg(self, out_dir, pointcloud_with_color):
        pc, color = pointcloud_with_color
        out_path = os.path.join(out_dir, 'legacy_export_color_kwarg.usda')
        with pytest.warns(DeprecationWarning, match="color"):
            usd.export_pointcloud(out_path, points=pc, color=color, scene_path=self.scene_path,
                                  points_type='usd_geom_points', overwrite=True)
        result = usd.import_pointcloud(out_path, scene_path=self.scene_path)
        expected = pointcloud_return_type(pc, color, None, None)
        assert contained_torch_equal(result, expected, approximate=True, atol=1e-5)

    def test_export_pointcloud_legacy_pointcloud_and_color_kwargs(self, out_dir, pointcloud_with_color):
        # Combined: both legacy kwargs (pointcloud= and color=) on the same call.
        # Exercises that both shims fire and route to the correct new params.
        pc, color = pointcloud_with_color
        out_path = os.path.join(out_dir, 'legacy_export_pointcloud_and_color.usda')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            usd.export_pointcloud(out_path, pointcloud=pc, color=color, scene_path=self.scene_path,
                                  points_type='usd_geom_points', overwrite=True)
        deprecation_msgs = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
        # Both renames warn — one for pointcloud, one for color.
        assert any("pointcloud" in m for m in deprecation_msgs)
        assert any("color" in m for m in deprecation_msgs)
        result = usd.import_pointcloud(out_path, scene_path=self.scene_path)
        expected = pointcloud_return_type(pc, color, None, None)
        assert contained_torch_equal(result, expected, approximate=True, atol=1e-5)

    def test_export_pointclouds_legacy_positional(self, out_dir, pointcloud):
        # v0.18 form: export_pointclouds(file, [pointcloud_tensor, ...])
        out_path = os.path.join(out_dir, 'legacy_export_pcs_positional.usda')
        with pytest.warns(DeprecationWarning, match="positional"):
            usd.export_pointclouds(out_path, [pointcloud, pointcloud], overwrite=True)
        result = usd.import_pointclouds(out_path, return_list=False)
        expected_entry = pointcloud_return_type(pointcloud, None, None, None)
        assert len(result) == 2
        for entry in result.values():
            assert contained_torch_equal(entry, expected_entry, approximate=True, atol=1e-5)

    def test_export_pointclouds_legacy_pointclouds_kwarg(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'legacy_export_pcs_kwarg.usda')
        with pytest.warns(DeprecationWarning, match="pointclouds"):
            usd.export_pointclouds(out_path, pointclouds=[pointcloud, pointcloud], overwrite=True)
        result = usd.import_pointclouds(out_path, return_list=False)
        expected_entry = pointcloud_return_type(pointcloud, None, None, None)
        assert len(result) == 2
        for entry in result.values():
            assert contained_torch_equal(entry, expected_entry, approximate=True, atol=1e-5)

