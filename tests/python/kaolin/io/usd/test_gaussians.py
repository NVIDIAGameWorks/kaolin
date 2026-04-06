# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
import math
import torch
import pytest
from pxr import Usd

from kaolin.io import usd
from kaolin.utils.bundled_data import SCANNED_TOYS_PATH, SCANNED_TOYS_NAMES, download_scanned_toys_dataset
from kaolin.utils.env_vars import KaolinTestEnvVars
from kaolin.utils.testing import contained_torch_equal
from kaolin.ops.gaussians import transform_gaussians

TEST_SCANNED_TOYS = os.getenv(KaolinTestEnvVars.TEST_SCANNED_TOYS)


def make_synthetic_gaussian_cloud(n=8, seed=0):
    """Create minimal valid gaussian cloud data for testing."""
    torch.manual_seed(seed)
    theta0 = math.acos(1 / math.sqrt(3)) / 2
    c = 1 / math.sqrt(2)
    rots = torch.tensor([
        [-c * math.sin(theta0), math.cos(theta0), 0, c * math.sin(theta0)],
    ], dtype=torch.float32).repeat(n, 1)
    return {
        'positions': torch.rand(n, 3, dtype=torch.float32),
        'orientations': rots,
        'scales': torch.rand(n, 3, dtype=torch.float32) * 0.1 + 0.05,
        'opacities': torch.rand(n, dtype=torch.float32),
        'sh_coeff': torch.randn(n, 16, 3, dtype=torch.float32) * 0.1,
        'local_to_world': None,
    }

@pytest.fixture(scope='class')
def out_dir():
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_out')
    os.makedirs(out_dir, exist_ok=True)
    yield out_dir
    shutil.rmtree(out_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Tests that run without TOYS dataset (synthetic data)
# -----------------------------------------------------------------------------

class TestGaussianExportImport:
    """Tests for export/import using synthetic data (no dataset required)."""

    @pytest.mark.parametrize('with_stage', [True, False])
    def test_export_import_roundtrip(self, out_dir, with_stage):
        """Export synthetic gaussians and re-import; data should match."""
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        out_path = os.path.join(out_dir, 'gaussian_roundtrip.usdc')
        scene_path = '/World/Gaussians/gaussian_0'

        usd.export_gaussiancloud(
            out_path,
            scene_path=scene_path,
            positions=data['positions'],
            orientations=data['orientations'],
            scales=data['scales'],
            opacities=data['opacities'],
            sh_coeff=data['sh_coeff'],
            overwrite=True,
        )

        if with_stage:
            stage = Usd.Stage.Open(out_path)
            imported = usd.import_gaussianclouds(stage, [scene_path])
        else:
            imported = usd.import_gaussianclouds(out_path, [scene_path])

        assert contained_torch_equal(
            {scene_path: data},
            imported,
            approximate=True,
            rtol=1e-5,
            atol=1e-6,
        )

    @pytest.mark.parametrize('with_stage', [True, False])
    def test_export_import_roundtrip_with_transform(self, out_dir, with_stage):
        """Export synthetic gaussians with a local_to_world transform and re-import; transform should be preserved."""
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        transform = torch.tensor([[2, 0, 0, 0.5],
                                  [0, 0, -2, 0],
                                  [0, 2, 0, 0],
                                  [0, 0, 0, 1]], dtype=torch.float32)
        out_path = os.path.join(out_dir, 'gaussian_roundtrip_transform.usdc')
        scene_path = '/World/Gaussians/gaussian_0'

        usd.export_gaussiancloud(
            out_path,
            scene_path=scene_path,
            positions=data['positions'],
            orientations=data['orientations'],
            scales=data['scales'],
            opacities=data['opacities'],
            sh_coeff=data['sh_coeff'],
            local_to_world=transform,
            overwrite=True,
        )

        if with_stage:
            stage = Usd.Stage.Open(out_path)
            imported = usd.import_gaussianclouds(stage, [scene_path])
        else:
            imported = usd.import_gaussianclouds(out_path, [scene_path])

        assert contained_torch_equal(
            {scene_path: {**data, 'local_to_world': transform}},
            imported,
            approximate=True,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_export_overwrite_raises_without_flag(self, out_dir):
        """Export to existing file without overwrite should raise FileExistsError."""
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        out_path = os.path.join(out_dir, 'gaussian_exists.usdc')
        usd.export_gaussiancloud(
            out_path,
            positions=data['positions'],
            orientations=data['orientations'],
            scales=data['scales'],
            opacities=data['opacities'],
            sh_coeff=data['sh_coeff'],
            overwrite=True,
        )

        with pytest.raises(FileExistsError):
            usd.export_gaussiancloud(
                out_path,
                positions=data['positions'],
                orientations=data['orientations'],
                scales=data['scales'],
                opacities=data['opacities'],
                sh_coeff=data['sh_coeff'],
                overwrite=False,
            )

    def test_add_gaussiancloud_overwrite_raises_without_flag(self, out_dir):
        """Adding at existing scene path without overwrite should raise ValueError."""
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        out_path = os.path.join(out_dir, 'gaussian_add_overwrite.usdc')
        stage = usd.create_stage(out_path)

        usd.add_gaussiancloud(
            stage,
            '/World/Gaussians/gaussian_0',
            positions=data['positions'],
            orientations=data['orientations'],
            scales=data['scales'],
            opacities=data['opacities'],
            sh_coeff=data['sh_coeff'],
            overwrite=True,
        )

        with pytest.raises(ValueError, match=r"Prim already exists.*overwrite=True"):
            usd.add_gaussiancloud(
                stage,
                '/World/Gaussians/gaussian_0',
                positions=data['positions'],
                orientations=data['orientations'],
                scales=data['scales'],
                opacities=data['opacities'],
                sh_coeff=data['sh_coeff'],
                overwrite=False,
            )

    @pytest.mark.parametrize('with_stage', [True, False])
    def test_import_gaussianclouds_all_add_gaussiancloud_roundtrip(self, out_dir, with_stage):
        """Roundtrip: add two clouds with add_gaussiancloud, import_gaussianclouds, compare."""
        clouds = {
            '/World/Gaussians/gaussian_0': make_synthetic_gaussian_cloud(n=8, seed=0),
            '/World/Gaussians/gaussian_1': make_synthetic_gaussian_cloud(n=5, seed=42),
        }

        out_path = os.path.join(out_dir, 'gaussian_all_roundtrip.usdc')
        stage = usd.create_stage(out_path)

        for scene_path, cloud in clouds.items():
            usd.add_gaussiancloud(
                stage,
                scene_path,
                positions=cloud['positions'],
                orientations=cloud['orientations'],
                scales=cloud['scales'],
                opacities=cloud['opacities'],
                sh_coeff=cloud['sh_coeff'],
            )

        stage.Save()

        if with_stage:
            reimported = usd.import_gaussianclouds(stage)
        else:
            del stage
            reimported = usd.import_gaussianclouds(out_path)

        assert contained_torch_equal(
            {scene_path: {**cloud, 'local_to_world': None} for scene_path, cloud in clouds.items()},
            reimported,
            approximate=True,
            rtol=1e-5,
            atol=1e-6,
        )

    @pytest.mark.parametrize('with_stage', [True, False])
    def test_import_gaussiancloud_single(self, out_dir, with_stage):
        """import_gaussiancloud with a single identity-transform cloud returns original data."""
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        out_path = os.path.join(out_dir, 'gaussian_merged_single.usdc')
        scene_path = '/World/Gaussians/gaussian_0'

        usd.export_gaussiancloud(
            out_path,
            scene_path=scene_path,
            positions=data['positions'],
            orientations=data['orientations'],
            scales=data['scales'],
            opacities=data['opacities'],
            sh_coeff=data['sh_coeff'],
            overwrite=True,
        )

        if with_stage:
            stage = Usd.Stage.Open(out_path)
            merged = usd.import_gaussiancloud(stage, scene_path)
        else:
            merged = usd.import_gaussiancloud(out_path, scene_path)

        expected = {k: data[k] for k in ('positions', 'orientations', 'scales', 'opacities', 'sh_coeff')}
        assert 'local_to_world' not in merged
        assert contained_torch_equal(expected, merged, approximate=True, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize('with_stage', [True, False])
    def test_import_gaussiancloud_with_transform(self, out_dir, with_stage):
        """import_gaussiancloud applies local_to_world and returns world-space attributes."""
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        transform = torch.tensor([[2, 0, 0, 0.5],
                                   [0, 0, -2, 0],
                                   [0, 2, 0, 0],
                                   [0, 0, 0, 1]], dtype=torch.float32)
        out_path = os.path.join(out_dir, 'gaussian_merged_transform.usdc')
        scene_path = '/World/Gaussians/gaussian_0'

        usd.export_gaussiancloud(
            out_path,
            scene_path=scene_path,
            positions=data['positions'],
            orientations=data['orientations'],
            scales=data['scales'],
            opacities=data['opacities'],
            sh_coeff=data['sh_coeff'],
            local_to_world=transform,
            overwrite=True,
        )

        if with_stage:
            stage = Usd.Stage.Open(out_path)
            merged = usd.import_gaussiancloud(stage, scene_path)
        else:
            merged = usd.import_gaussiancloud(out_path, scene_path)

        new_xyz, new_rot, new_scales, new_sh_rest = transform_gaussians(
            data['positions'], data['orientations'], data['scales'],
            transform, shs_feat=data['sh_coeff'][:, 1:],
        )
        expected = {
            'positions': new_xyz,
            'orientations': new_rot,
            'scales': new_scales,
            'opacities': data['opacities'],
            'sh_coeff': torch.cat([data['sh_coeff'][:, :1], new_sh_rest], dim=1),
        }
        assert 'local_to_world' not in merged
        assert contained_torch_equal(expected, merged, approximate=True, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('with_stage', [True, False])
    def test_import_gaussiancloud_merge(self, out_dir, with_stage):
        """import_gaussiancloud merges multiple transformed clouds into one concatenated world-space cloud."""
        cloud_0 = make_synthetic_gaussian_cloud(n=8, seed=0)
        cloud_1 = make_synthetic_gaussian_cloud(n=5, seed=42)
        transform_0 = torch.tensor([[2, 0, 0, 0.5],
                                     [0, 0, -2, 0],
                                     [0, 2, 0, 0],
                                     [0, 0, 0, 1]], dtype=torch.float32)
        transform_1 = torch.tensor([[0, 1, 0, -1],
                                     [-1, 0, 0, 0],
                                     [0, 0, 1, 2],
                                     [0, 0, 0, 1]], dtype=torch.float32)

        out_path = os.path.join(out_dir, 'gaussian_merged_multi.usdc')
        stage = usd.create_stage(out_path)

        usd.add_gaussiancloud(
            stage, '/World/Gaussians/gaussian_0',
            positions=cloud_0['positions'], orientations=cloud_0['orientations'],
            scales=cloud_0['scales'], opacities=cloud_0['opacities'], sh_coeff=cloud_0['sh_coeff'],
            local_to_world=transform_0,
        )
        usd.add_gaussiancloud(
            stage, '/World/Gaussians/gaussian_1',
            positions=cloud_1['positions'], orientations=cloud_1['orientations'],
            scales=cloud_1['scales'], opacities=cloud_1['opacities'], sh_coeff=cloud_1['sh_coeff'],
            local_to_world=transform_1,
        )
        stage.Save()

        if with_stage:
            merged = usd.import_gaussiancloud(stage)
        else:
            del stage
            merged = usd.import_gaussiancloud(out_path)

        xyz0, rot0, s0, sh_rest0 = transform_gaussians(
            cloud_0['positions'], cloud_0['orientations'], cloud_0['scales'],
            transform_0, shs_feat=cloud_0['sh_coeff'][:, 1:],
        )
        xyz1, rot1, s1, sh_rest1 = transform_gaussians(
            cloud_1['positions'], cloud_1['orientations'], cloud_1['scales'],
            transform_1, shs_feat=cloud_1['sh_coeff'][:, 1:],
        )
        expected = {
            'positions':    torch.cat([xyz0, xyz1], dim=0),
            'orientations': torch.cat([rot0, rot1], dim=0),
            'scales':       torch.cat([s0, s1], dim=0),
            'opacities':    torch.cat([cloud_0['opacities'], cloud_1['opacities']], dim=0),
            'sh_coeff':     torch.cat([
                torch.cat([cloud_0['sh_coeff'][:, :1], sh_rest0], dim=1),
                torch.cat([cloud_1['sh_coeff'][:, :1], sh_rest1], dim=1),
            ], dim=0),
        }
        assert 'local_to_world' not in merged
        assert contained_torch_equal(expected, merged, approximate=True, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('with_stage', [True, False])
    def test_import_gaussiancloud_root_path(self, out_dir, with_stage):
        """import_gaussiancloud with root_path merges only clouds under that prefix."""
        cloud_foo_0 = make_synthetic_gaussian_cloud(n=8, seed=0)
        cloud_foo_1 = make_synthetic_gaussian_cloud(n=5, seed=42)
        cloud_bar_0 = make_synthetic_gaussian_cloud(n=6, seed=7)

        out_path = os.path.join(out_dir, 'gaussian_root_path_import.usdc')
        stage = usd.create_stage(out_path)
        for scene_path, cloud in [
            ('/World/Foo/gaussian_0', cloud_foo_0),
            ('/World/Foo/gaussian_1', cloud_foo_1),
            ('/World/Bar/gaussian_0', cloud_bar_0),
        ]:
            usd.add_gaussiancloud(
                stage, scene_path,
                positions=cloud['positions'], orientations=cloud['orientations'],
                scales=cloud['scales'], opacities=cloud['opacities'], sh_coeff=cloud['sh_coeff'],
            )
        stage.Save()

        if with_stage:
            merged = usd.import_gaussiancloud(stage, root_path='/World/Foo')
        else:
            del stage
            merged = usd.import_gaussiancloud(out_path, root_path='/World/Foo')
        assert merged is not None
        assert merged['positions'].shape[0] == 8 + 5

    @pytest.mark.parametrize('with_stage', [True, False])
    def test_get_gaussiancloud_scene_paths_synthetic(self, out_dir, with_stage):
        """get_gaussiancloud_scene_paths returns expected paths from synthetic stage."""
        scene_paths = ['/World/Gaussians/gaussian_0', '/World/Gaussians/gaussian_1']
        cloud_0 = make_synthetic_gaussian_cloud(n=8, seed=0)
        cloud_1 = make_synthetic_gaussian_cloud(n=5, seed=42)

        out_path = os.path.join(out_dir, 'gaussian_scene_paths.usdc')
        stage = usd.create_stage(out_path)
        for scene_path, cloud in zip(scene_paths, [cloud_0, cloud_1], strict=True):
            usd.add_gaussiancloud(
                stage,
                scene_path,
                positions=cloud['positions'],
                orientations=cloud['orientations'],
                scales=cloud['scales'],
                opacities=cloud['opacities'],
                sh_coeff=cloud['sh_coeff'],
            )
        stage.Save()

        if with_stage:
            actual = usd.get_gaussiancloud_scene_paths(stage)
        else:
            del stage
            actual = usd.get_gaussiancloud_scene_paths(out_path)
        expected_strs = set(scene_paths)
        actual_strs = set(str(p) for p in actual)
        assert actual_strs == expected_strs

    @pytest.mark.parametrize('with_stage', [True, False])
    def test_get_gaussiancloud_scene_paths_root_path(self, out_dir, with_stage):
        """get_gaussiancloud_scene_paths with root_path returns only paths under that prefix"""
        foo_paths = ['/World/Foo/gaussian_0', '/World/Foo/gaussian_1']
        bar_paths = ['/World/Bar/gaussian_0']
        all_scene_paths = foo_paths + bar_paths

        out_path = os.path.join(out_dir, 'gaussian_root_path.usdc')
        stage = usd.create_stage(out_path)
        for scene_path in all_scene_paths:
            cloud = make_synthetic_gaussian_cloud(n=8, seed=0)
            usd.add_gaussiancloud(
                stage,
                scene_path,
                positions=cloud['positions'],
                orientations=cloud['orientations'],
                scales=cloud['scales'],
                opacities=cloud['opacities'],
                sh_coeff=cloud['sh_coeff'],
            )
        stage.Save()

        if with_stage:
            actual = usd.get_gaussiancloud_scene_paths(stage, root_path='/World/Foo')
        else:
            del stage
            actual = usd.get_gaussiancloud_scene_paths(out_path, root_path='/World/Foo')
        assert set(str(p) for p in actual) == set(foo_paths)


# -----------------------------------------------------------------------------
# Tests that require TOYS dataset (skipped when KAOLIN_TEST_SCANNED_TOYS is unset) # TODO: needed?
# Note that default location for toys is either under sample_data or is set
# using KAOLIN_SCANNED_TOYS_PATH env variable.
# -----------------------------------------------------------------------------

@pytest.mark.skipif(
    TEST_SCANNED_TOYS is None,
    reason="'KAOLIN_TEST_SCANNED_TOYS' environment variable is not set (will download files if needed).",
)
class TestGaussianImportFromToys:
    """Tests for importing gaussians from TOYS dataset usdc files."""
    @pytest.fixture(scope='class', autouse=True)
    def download_toys_dataset(self):
        download_scanned_toys_dataset()

    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_import_toy_file(self, toy_name):
        """Import gaussians from each toy usdc file and validate cloud structure."""
        path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.usdc')
        scene_paths = usd.get_gaussiancloud_scene_paths(path)

        gaussianclouds = usd.import_gaussianclouds(path, scene_paths)

        assert set(gaussianclouds.keys()) == set(scene_paths)
        for cloud in gaussianclouds.values():
            assert 'positions' in cloud
            assert 'orientations' in cloud
            assert 'scales' in cloud
            assert 'opacities' in cloud
            assert 'sh_coeff' in cloud
            assert 'local_to_world' in cloud
            assert cloud['local_to_world'] is None or cloud['local_to_world'].shape == (4, 4)
            n = cloud['positions'].shape[0]
            assert cloud['positions'].shape == (n, 3)
            assert cloud['orientations'].shape == (n, 4)
            assert cloud['scales'].shape == (n, 3)
            assert cloud['opacities'].shape == (n,)
            assert cloud['sh_coeff'].shape[0] == n
            assert cloud['sh_coeff'].shape[2] == 3

    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_import_gaussianclouds_all(self, toy_name):
        """import_gaussianclouds with default scene_pathsreturns dict mapping scene path to cloud."""
        path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.usdc')
        all_clouds = usd.import_gaussianclouds(path)

        scene_paths = usd.get_gaussiancloud_scene_paths(path)

        assert len(all_clouds) == len(scene_paths)
        for sp in scene_paths:
            assert str(sp) in all_clouds

    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_import_export_roundtrip_from_toy(self, out_dir, toy_name):
        """Import cloud from toy, export to temp file, re-import and compare."""
        path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.usdc')
        scene_paths = usd.get_gaussiancloud_scene_paths(path)

        imported = usd.import_gaussianclouds(path, scene_paths)
        # Test round-trip for the first gaussian cloud
        cloud = imported['/World/Gaussians/gaussian_0']

        out_path = os.path.join(out_dir, f'gaussian_roundtrip_{toy_name}.usdc')
        usd.export_gaussiancloud(
            out_path,
            scene_path='/World/Gaussians/gaussian_0',
            positions=cloud['positions'],
            orientations=cloud['orientations'],
            scales=cloud['scales'],
            opacities=cloud['opacities'],
            sh_coeff=cloud['sh_coeff'],
            local_to_world=cloud['local_to_world'],
            overwrite=True,
        )

        reimported = usd.import_gaussianclouds(out_path, ['/World/Gaussians/gaussian_0'])

        assert contained_torch_equal(
            {'/World/Gaussians/gaussian_0': cloud},
            reimported,
            approximate=True,
            rtol=1e-4,
            atol=1e-5,
        )

    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_import_from_stage_object(self, toy_name):
        """Import works when passing Usd.Stage instead of file path."""
        path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.usdc')

        stage = Usd.Stage.Open(path)
        scene_paths = usd.get_gaussiancloud_scene_paths(stage)

        gaussianclouds = usd.import_gaussianclouds(stage, scene_paths)
        assert len(gaussianclouds) >= 1

    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_import_gaussianclouds_vs_ground_truth(self, toy_name):
        """Compare import_gaussianclouds output against ground truth .pt file."""
        path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.usdc')
        gt_path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.pt')

        expected = torch.load(gt_path, weights_only=True)
        scene_paths = ['/World/Gaussians/gaussian_0']
        actual = usd.import_gaussianclouds(path, scene_paths)
        assert len(actual) == 1
        assert contained_torch_equal(
            actual,
            {'/World/Gaussians/gaussian_0': expected},
            approximate=True,
            rtol=1e-4,
            atol=1e-5,
        ), f'Mismatch for cloud {toy_name}'

    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_import_gaussianclouds_all_vs_ground_truth(self, toy_name):
        """Compare import_gaussianclouds with default scene_paths output against ground truth .pt file."""
        path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.usdc')
        gt_path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.pt')

        expected = torch.load(gt_path, weights_only=True)
        actual = usd.import_gaussianclouds(path)
        assert len(actual) == 1
        assert contained_torch_equal(
            actual['/World/Gaussians/gaussian_0'], expected,
            approximate=True,
            rtol=1e-4,
            atol=1e-5,
        ), f'Mismatch for {toy_name}'

    def test_get_gaussiancloud_scene_paths_combined(self):
        """get_gaussiancloud_scene_paths returns BluehairRagdoll_0 and BluehairRagdoll_1 paths from BluehairRagdoll_multi.usdc."""
        path = os.path.join(SCANNED_TOYS_PATH, 'BluehairRagdoll_multi.usdc')

        actual = usd.get_gaussiancloud_scene_paths(path)
        expected = {'/World/Gaussians/BluehairRagdoll_0', '/World/Gaussians/BluehairRagdoll_1'}
        actual_strs = set(str(p) for p in actual)
        assert actual_strs == expected

    def test_import_gaussianclouds_partial_combined(self):
        """Import gaussianclouds from two BluehairRagdoll (2nd is transformed) usdc and compare to ground truth."""
        path = os.path.join(SCANNED_TOYS_PATH, 'BluehairRagdoll_multi.usdc')
        gt = torch.load(os.path.join(SCANNED_TOYS_PATH, 'BluehairRagdoll.pt'), weights_only=True)

        scene_paths = ['/World/Gaussians/BluehairRagdoll_0']
        output = usd.import_gaussianclouds(path, scene_paths)

        expected = {'/World/Gaussians/BluehairRagdoll_0': gt}
        
        assert contained_torch_equal(output, expected, approximate=True, rtol=1e-4, atol=1e-5), 'Mismatch for BluehairRagdoll_multi'

    def test_import_gaussianclouds_all_combined(self):
        """Import gaussianclouds with default scene_paths from two BluehairRagdoll (2nd is transformed) usdc and compare to ground truth."""
        path = os.path.join(SCANNED_TOYS_PATH, 'BluehairRagdoll_multi.usdc')
        gt = torch.load(os.path.join(SCANNED_TOYS_PATH, 'BluehairRagdoll.pt'), weights_only=True)

        transform = torch.tensor([[2, 0, 0, 0.5],
                                   [0, 0, -2, 0],
                                   [0, 2, 0, 0],
                                   [0, 0, 0, 1]], dtype=torch.float32)
        output = usd.import_gaussianclouds(path)

        expected = {
            '/World/Gaussians/BluehairRagdoll_0': gt,
            '/World/Gaussians/BluehairRagdoll_1': {**gt, 'local_to_world': transform},
        }
        assert contained_torch_equal(output, expected, approximate=True, rtol=1e-4, atol=1e-5), 'Mismatch for BluehairRagdoll_multi'

    def test_import_gaussianclouds_partial_compressed(self):
        """Import gaussianclouds from two BluehairRagdoll (2nd is transformed) usdc and compare to ground truth."""
        path = os.path.join(SCANNED_TOYS_PATH, 'BluehairRagdoll_compressed.usdc')
        gt = torch.load(os.path.join(SCANNED_TOYS_PATH, 'BluehairRagdoll.pt'), weights_only=True)

        gt['sh_coeff'] = gt['sh_coeff'][:, :1]

        transform = torch.tensor([[2, 0, 0, 0.5],
                                  [0, 0, -2, 0],
                                  [0, 2, 0, 0],
                                  [0, 0, 0, 1]], dtype=torch.float32)
        scene_paths = ['/World/Gaussians/BluehairRagdoll_1']
        output = usd.import_gaussianclouds(path, scene_paths)

        gt_half = {k: v.half() if v is not None else None for k, v in gt.items()}
        expected = {'/World/Gaussians/BluehairRagdoll_1': {**gt_half, 'local_to_world': transform}}
        assert contained_torch_equal(output, expected, approximate=True, rtol=1e-3, atol=1e-3), 'Mismatch for BluehairRagdoll_compressed'

    def test_import_gaussianclouds_all_compressed(self):
        """Import all gaussianclouds from two BluehairRagdoll (2nd is transformed) usdc and compare to ground truth."""
        path = os.path.join(SCANNED_TOYS_PATH, 'BluehairRagdoll_compressed.usdc')
        gt = torch.load(os.path.join(SCANNED_TOYS_PATH, 'BluehairRagdoll.pt'), weights_only=True)

        gt['sh_coeff'] = gt['sh_coeff'][:, :1]

        transform = torch.tensor([[2, 0, 0, 0.5],
                                  [0, 0, -2, 0],
                                  [0, 2, 0, 0],
                                  [0, 0, 0, 1]], dtype=torch.float32)
        output = usd.import_gaussianclouds(path)

        gt_half = {k: v.half() if v is not None else None for k, v in gt.items()}
        expected = {
            '/World/Gaussians/BluehairRagdoll_0': gt_half,
            '/World/Gaussians/BluehairRagdoll_1': {**gt_half, 'local_to_world': transform},
        }
        assert contained_torch_equal(output, expected, approximate=True, rtol=1e-3, atol=1e-3), 'Mismatch for BluehairRagdoll_compressed'
