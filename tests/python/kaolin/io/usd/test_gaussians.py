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
import wget
from pxr import Usd

from kaolin.io import usd
from kaolin.ops.gaussian.transforms import transform_gaussians
from kaolin.utils.testing import contained_torch_equal

TOYS_DATASET_PATH = os.getenv('KAOLIN_TEST_TOYS_DATASET_PATH')
TOYS_NAMES = ['BluehairRagdoll', 'bublik_octopus', 'knit_meow', 'mer_elephant', 'stink_raccoon', 'sunflower_baby']

def make_synthetic_gaussian_cloud(n=8, seed=0):
    """Create minimal valid gaussian cloud data for testing."""
    torch.manual_seed(seed)
    theta0 = math.acos(1 / math.sqrt(3)) / 2
    c = 1 / math.sqrt(2)
    rots = torch.tensor([
        [math.cos(theta0), 0, c * math.sin(theta0), -c * math.sin(theta0)],
    ], dtype=torch.float32).repeat(n, 1)
    return {
        'positions': torch.rand(n, 3, dtype=torch.float32),
        'orientations': rots,
        'scales': torch.rand(n, 3, dtype=torch.float32) * 0.1 + 0.05,
        'opacities': torch.rand(n, dtype=torch.float32),
        'sh_coeff': torch.randn(n, 16, 3, dtype=torch.float32) * 0.1,
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

    def test_export_import_roundtrip(self, out_dir):
        """Export synthetic gaussians and re-import; data should match."""
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        out_path = os.path.join(out_dir, 'gaussian_roundtrip.usda')
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

        imported = usd.import_gaussianclouds(out_path, [scene_path])[0]

        assert contained_torch_equal(
            data,
            imported,
            approximate=True,
            rtol=1e-5,
            atol=1e-6,
        )

    def test_export_overwrite_raises_without_flag(self, out_dir):
        """Export to existing file without overwrite should raise FileExistsError."""
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        out_path = os.path.join(out_dir, 'gaussian_exists.usda')
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
        out_path = os.path.join(out_dir, 'gaussian_add_overwrite.usda')
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

    def test_import_all_gaussianclouds_add_gaussiancloud_roundtrip(self, out_dir):
        """Roundtrip: add two clouds with add_gaussiancloud, import_all_gaussianclouds, compare."""
        scene_paths = ['/World/Gaussians/gaussian_0', '/World/Gaussians/gaussian_1']
        cloud_0 = make_synthetic_gaussian_cloud(n=8, seed=0)
        cloud_1 = make_synthetic_gaussian_cloud(n=5, seed=42)
        clouds = [cloud_0, cloud_1]

        out_path = os.path.join(out_dir, 'gaussian_all_roundtrip.usda')
        stage = usd.create_stage(out_path)

        for scene_path, cloud in zip(scene_paths, clouds, strict=True):
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

        reimported = usd.import_all_gaussianclouds(out_path)

        assert len(reimported) == len(scene_paths)
        for scene_path, cloud in zip(scene_paths, clouds, strict=True):
            assert scene_path in reimported
            assert contained_torch_equal(
                cloud,
                reimported[scene_path],
                approximate=True,
                rtol=1e-5,
                atol=1e-6,
            )

    def test_get_gaussiancloud_scene_paths_synthetic(self, out_dir):
        """get_gaussiancloud_scene_paths returns expected paths from synthetic stage."""
        scene_paths = ['/World/Gaussians/gaussian_0', '/World/Gaussians/gaussian_1']
        cloud_0 = make_synthetic_gaussian_cloud(n=8, seed=0)
        cloud_1 = make_synthetic_gaussian_cloud(n=5, seed=42)

        out_path = os.path.join(out_dir, 'gaussian_scene_paths.usda')
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

        actual = usd.get_gaussiancloud_scene_paths(out_path)
        expected_strs = set(scene_paths)
        actual_strs = set(str(p) for p in actual)
        assert actual_strs == expected_strs


# -----------------------------------------------------------------------------
# Tests that require TOYS dataset (skipped when KAOLIN_TEST_TOYS_DATASET_PATH unset)
# -----------------------------------------------------------------------------

@pytest.mark.skipif(
    TOYS_DATASET_PATH is None,
    reason="'KAOLIN_TEST_TOYS_DATASET_PATH' environment variable is not set. (Will download files if set but has no files)",
)
class TestGaussianImportFromToys:
    """Tests for importing gaussians from TOYS dataset usdc files."""
    @pytest.fixture(scope='class', autouse=True)
    def download_toys_dataset(self):
        """Download toys dataset if it doesn't exist."""
        if not os.path.exists(TOYS_DATASET_PATH):
            os.makedirs(TOYS_DATASET_PATH, exist_ok=True)
            # raise ValueError(f"Directory {TOYS_DATASET_PATH} does not exist, set 'KAOLIN_TEST_TOYS_DATASET_PATH' "
            #                  "environment variable to an existing directory.")
        for toy_name in TOYS_NAMES:
            path = os.path.join(TOYS_DATASET_PATH, toy_name)
            if not os.path.exists(path + '.usdc'):
                wget.download(f'https://nvidia-kaolin.s3.us-east-2.amazonaws.com/data/{toy_name}.usdc', path + '.usdc')
            if not os.path.exists(path + '.pt'):
                wget.download(f'https://nvidia-kaolin.s3.us-east-2.amazonaws.com/data/{toy_name}.pt', path + '.pt')
        if not os.path.exists(os.path.join(TOYS_DATASET_PATH, 'BluehairRagdoll_multi.usdc')):
            wget.download('https://nvidia-kaolin.s3.us-east-2.amazonaws.com/data/BluehairRagdoll_multi.usdc',
                          os.path.join(TOYS_DATASET_PATH, 'BluehairRagdoll_multi.usdc'))
        if not os.path.exists(os.path.join(TOYS_DATASET_PATH, 'BluehairRagdoll_compressed.usdc')):
            wget.download('https://nvidia-kaolin.s3.us-east-2.amazonaws.com/data/BluehairRagdoll_compressed.usdc',
                          os.path.join(TOYS_DATASET_PATH, 'BluehairRagdoll_compressed.usdc'))

    @pytest.mark.parametrize('toy_name', TOYS_NAMES)
    def test_import_toy_file(self, toy_name):
        """Import gaussians from each toy usdc file and validate cloud structure."""
        path = os.path.join(TOYS_DATASET_PATH, f'{toy_name}.usdc')
        scene_paths = usd.get_gaussiancloud_scene_paths(path)

        gaussianclouds = usd.import_gaussianclouds(path, scene_paths)

        assert len(gaussianclouds) == len(scene_paths)
        for cloud in gaussianclouds:
            assert 'positions' in cloud
            assert 'orientations' in cloud
            assert 'scales' in cloud
            assert 'opacities' in cloud
            assert 'sh_coeff' in cloud
            n = cloud['positions'].shape[0]
            assert cloud['positions'].shape == (n, 3)
            assert cloud['orientations'].shape == (n, 4)
            assert cloud['scales'].shape == (n, 3)
            assert cloud['opacities'].shape == (n,)
            assert cloud['sh_coeff'].shape[0] == n
            assert cloud['sh_coeff'].shape[2] == 3

    @pytest.mark.parametrize('toy_name', TOYS_NAMES)
    def test_import_all_gaussianclouds(self, toy_name):
        """import_all_gaussianclouds returns dict mapping scene path to cloud."""
        path = os.path.join(TOYS_DATASET_PATH, f'{toy_name}.usdc')
        all_clouds = usd.import_all_gaussianclouds(path)

        scene_paths = usd.get_gaussiancloud_scene_paths(path)

        assert len(all_clouds) == len(scene_paths)
        for sp in scene_paths:
            assert str(sp) in all_clouds

    @pytest.mark.parametrize('toy_name', TOYS_NAMES)
    def test_import_export_roundtrip_from_toy(self, out_dir, toy_name):
        """Import cloud from toy, export to temp file, re-import and compare."""
        path = os.path.join(TOYS_DATASET_PATH, f'{toy_name}.usdc')
        scene_paths = usd.get_gaussiancloud_scene_paths(path)

        imported = usd.import_gaussianclouds(path, scene_paths)
        # Test round-trip for the first gaussian cloud
        cloud = imported[0]

        out_path = os.path.join(out_dir, f'gaussian_roundtrip_{toy_name}.usda')
        usd.export_gaussiancloud(
            out_path,
            scene_path='/World/Gaussians/gaussian_0',
            positions=cloud['positions'],
            orientations=cloud['orientations'],
            scales=cloud['scales'],
            opacities=cloud['opacities'],
            sh_coeff=cloud['sh_coeff'],
            overwrite=True,
        )

        reimported = usd.import_gaussianclouds(out_path, ['/World/Gaussians/gaussian_0'])[0]

        assert contained_torch_equal(
            cloud,
            reimported,
            approximate=True,
            rtol=1e-4,
            atol=1e-5,
        )

    @pytest.mark.parametrize('toy_name', TOYS_NAMES)
    def test_import_from_stage_object(self, toy_name):
        """Import works when passing Usd.Stage instead of file path."""
        path = os.path.join(TOYS_DATASET_PATH, f'{toy_name}.usdc')

        stage = Usd.Stage.Open(path)
        scene_paths = usd.get_gaussiancloud_scene_paths(stage)

        gaussianclouds = usd.import_gaussianclouds(stage, scene_paths)
        assert len(gaussianclouds) >= 1

    @pytest.mark.parametrize('toy_name', TOYS_NAMES)
    def test_import_gaussianclouds_vs_ground_truth(self, toy_name):
        """Compare import_gaussianclouds output against ground truth .pt file."""
        path = os.path.join(TOYS_DATASET_PATH, f'{toy_name}.usdc')
        gt_path = os.path.join(TOYS_DATASET_PATH, f'{toy_name}.pt')

        expected = torch.load(gt_path)
        scene_paths = ['/World/Gaussians/gaussian_0']
        actual = usd.import_gaussianclouds(path, scene_paths)
        assert len(actual) == 1
        assert contained_torch_equal(
            actual[0], expected,
            approximate=True,
            rtol=1e-4,
            atol=1e-5,
        ), f'Mismatch for cloud {toy_name}'

    @pytest.mark.parametrize('toy_name', TOYS_NAMES)
    def test_import_all_gaussianclouds_vs_ground_truth(self, toy_name):
        """Compare import_all_gaussianclouds output against ground truth .pt file."""
        path = os.path.join(TOYS_DATASET_PATH, f'{toy_name}.usdc')
        gt_path = os.path.join(TOYS_DATASET_PATH, f'{toy_name}.pt')

        expected = torch.load(gt_path)
        actual = usd.import_all_gaussianclouds(path)
        assert len(actual) == 1
        assert contained_torch_equal(
            actual['/World/Gaussians/gaussian_0'], expected,
            approximate=True,
            rtol=1e-4,
            atol=1e-5,
        ), f'Mismatch for {toy_name}'

    def test_get_gaussiancloud_scene_paths_combined(self):
        """get_gaussiancloud_scene_paths returns BluehairRagdoll_0 and BluehairRagdoll_1 paths from BluehairRagdoll_multi.usdc."""
        path = os.path.join(TOYS_DATASET_PATH, 'BluehairRagdoll_multi.usdc')

        actual = usd.get_gaussiancloud_scene_paths(path)
        expected = {'/World/Gaussians/BluehairRagdoll_0', '/World/Gaussians/BluehairRagdoll_1'}
        actual_strs = set(str(p) for p in actual)
        assert actual_strs == expected

    def test_import_gaussianclouds_combined(self):
        """Import gaussianclouds from two BluehairRagdoll (2nd is transformed) usdc and compare to ground truth."""
        path = os.path.join(TOYS_DATASET_PATH, 'BluehairRagdoll_multi.usdc')
        gt = torch.load(os.path.join(TOYS_DATASET_PATH, 'BluehairRagdoll.pt'))

        scene_paths = ['/World/Gaussians/BluehairRagdoll_0', '/World/Gaussians/BluehairRagdoll_1']
        output = usd.import_gaussianclouds(path, scene_paths)

        transformed_positions, transformed_orientations, transformed_scales, transformed_sh_coeff = transform_gaussians(
            gt['positions'], gt['orientations'], gt['scales'],
            torch.tensor([[2, 0, 0, 0.5],
                          [0, 0, -2, 0],  
                          [0, 2, 0, 0],
                          [0, 0, 0, 1]], dtype=torch.float32),
            shs_feat=gt['sh_coeff'][:, 1:]
        )
        expected = [
            gt,
            {
                'positions': transformed_positions,
                'orientations': transformed_orientations,
                'scales': transformed_scales,
                'opacities': gt['opacities'],
                'sh_coeff': torch.cat([gt['sh_coeff'][:, :1], transformed_sh_coeff], dim=1)
            }
        ]
        assert len(output) == len(scene_paths)
        assert contained_torch_equal(output, expected, approximate=True, rtol=1e-4, atol=1e-5), 'Mismatch for BluehairRagdoll_multi'

    def test_import_all_gaussianclouds_combined(self):
        """Import all gaussianclouds from two BluehairRagdoll (2nd is transformed) usdc and compare to ground truth."""
        path = os.path.join(TOYS_DATASET_PATH, 'BluehairRagdoll_multi.usdc')
        gt = torch.load(os.path.join(TOYS_DATASET_PATH, 'BluehairRagdoll.pt'))

        output = usd.import_all_gaussianclouds(path)

        transformed_positions, transformed_orientations, transformed_scales, transformed_sh_coeff = transform_gaussians(
            gt['positions'], gt['orientations'], gt['scales'],
            torch.tensor([[2, 0, 0, 0.5],
                          [0, 0, -2, 0],  
                          [0, 2, 0, 0],
                          [0, 0, 0, 1]], dtype=torch.float32),
            shs_feat=gt['sh_coeff'][:, 1:]
        )
        expected = {
            '/World/Gaussians/BluehairRagdoll_0': gt,
            '/World/Gaussians/BluehairRagdoll_1': {
                'positions': transformed_positions,
                'orientations': transformed_orientations,
                'scales': transformed_scales,
                'opacities': gt['opacities'],
                'sh_coeff': torch.cat([gt['sh_coeff'][:, :1], transformed_sh_coeff], dim=1)
            }
        }
        assert contained_torch_equal(output, expected, approximate=True, rtol=1e-4, atol=1e-5), 'Mismatch for BluehairRagdoll_multi'

    def test_import_gaussianclouds_compressed(self):
        """Import gaussianclouds from two BluehairRagdoll (2nd is transformed) usdc and compare to ground truth."""
        path = os.path.join(TOYS_DATASET_PATH, 'BluehairRagdoll_compressed.usdc')
        gt = torch.load(os.path.join(TOYS_DATASET_PATH, 'BluehairRagdoll.pt'))

        gt['sh_coeff'] = gt['sh_coeff'][:, :1]

        scene_paths = ['/World/Gaussians/BluehairRagdoll_0', '/World/Gaussians/BluehairRagdoll_1']
        output = usd.import_gaussianclouds(path, scene_paths)

        transformed_positions, transformed_orientations, transformed_scales, _ = transform_gaussians(
            gt['positions'], gt['orientations'], gt['scales'],
            torch.tensor([[2, 0, 0, 0.5],
                          [0, 0, -2, 0],  
                          [0, 2, 0, 0],
                          [0, 0, 0, 1]], dtype=torch.float32),
        )
        expected = [
            {k: v.half() for k, v in gt.items()},
            {
                'positions': transformed_positions.half(),
                'orientations': transformed_orientations.half(),
                'scales': transformed_scales.half(),
                'opacities': gt['opacities'].half(),
                'sh_coeff': gt['sh_coeff'].half(),
            }
        ]
        assert contained_torch_equal(output, expected, approximate=True, rtol=1e-3, atol=1e-3)

    def test_import_all_gaussianclouds_compressed(self):
        """Import all gaussianclouds from two BluehairRagdoll (2nd is transformed) usdc and compare to ground truth."""
        path = os.path.join(TOYS_DATASET_PATH, 'BluehairRagdoll_compressed.usdc')
        gt = torch.load(os.path.join(TOYS_DATASET_PATH, 'BluehairRagdoll.pt'))

        gt['sh_coeff'] = gt['sh_coeff'][:, :1]

        output = usd.import_all_gaussianclouds(path)
        transformed_positions, transformed_orientations, transformed_scales, _ = transform_gaussians(
            gt['positions'], gt['orientations'], gt['scales'],
            torch.tensor([[2, 0, 0, 0.5],
                          [0, 0, -2, 0],  
                          [0, 2, 0, 0],
                          [0, 0, 0, 1]], dtype=torch.float32),
        )
        expected = {
            '/World/Gaussians/BluehairRagdoll_0': {k: v.half() for k, v in gt.items()},
            '/World/Gaussians/BluehairRagdoll_1': {
                'positions': transformed_positions.half(),
                'orientations': transformed_orientations.half(),
                'scales': transformed_scales.half(),
                'opacities': gt['opacities'].half(),
                'sh_coeff': gt['sh_coeff'].half(),
            }
        }
        assert contained_torch_equal(output, expected, approximate=True, rtol=1e-3, atol=1e-3)
