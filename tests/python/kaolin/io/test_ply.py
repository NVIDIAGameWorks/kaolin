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

from kaolin.io import ply
from kaolin.rep import GaussianSplatModel
from kaolin.utils.bundled_data import SCANNED_TOYS_PATH, SCANNED_TOYS_NAMES, download_scanned_toys_dataset
from kaolin.utils.env_vars import KaolinTestEnvVars
from kaolin.utils.testing import contained_torch_equal, with_seed

TEST_SCANNED_TOYS = os.getenv(KaolinTestEnvVars.TEST_SCANNED_TOYS)

_GAUSSIAN_KEYS = ('positions', 'orientations', 'scales', 'opacities', 'sh_coeff')

# TODO: sh degree should be randomized
# TODO: remove manual seed
def make_synthetic_gaussian_cloud(n=8, seed=0):
    """Create minimal valid gaussian cloud data for testing."""
    torch.manual_seed(seed)
    theta0 = math.acos(1 / math.sqrt(3)) / 2
    c = 1 / math.sqrt(2)
    rots = torch.tensor([
        [-c * math.sin(theta0), math.cos(theta0), 0, c * math.sin(theta0)],
    ], dtype=torch.float32).repeat(n, 1)
    rots = torch.nn.functional.normalize(rots, dim=-1)
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

class TestGaussianExportImportPly:
    """Round-trip tests using :mod:`kaolin.io.ply` and synthetic data."""

    def test_export_import_roundtrip(self, out_dir):
        """Export synthetic gaussians to PLY and re-import; tensor fields should match."""
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        out_path = os.path.join(out_dir, 'gaussian_roundtrip.ply')

        ply.export_gaussiancloud(
            out_path,
            positions=data['positions'],
            orientations=data['orientations'],
            scales=data['scales'],
            opacities=data['opacities'],
            sh_coeff=data['sh_coeff'],
            overwrite=True,
        )

        imported = ply.import_gaussiancloud(out_path)

        assert contained_torch_equal(data, imported.as_dict(only_tensors=True), approximate=True, print_error_context='')

    def test_export_overwrite_raises_without_flag(self, out_dir):
        """Export to an existing file without overwrite should raise RuntimeError."""
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        out_path = os.path.join(out_dir, 'gaussian_exists.ply')
        ply.export_gaussiancloud(
            out_path,
            positions=data['positions'],
            orientations=data['orientations'],
            scales=data['scales'],
            opacities=data['opacities'],
            sh_coeff=data['sh_coeff']
        )

        with pytest.raises(RuntimeError):
            ply.export_gaussiancloud(
                out_path,
                positions=data['positions'],
                orientations=data['orientations'],
                scales=data['scales'],
                opacities=data['opacities'],
                sh_coeff=data['sh_coeff'],
                overwrite=False,
            )

        ply.export_gaussiancloud(
            out_path,
            positions=data['positions'],
            orientations=data['orientations'],
            scales=data['scales'],
            opacities=data['opacities'],
            sh_coeff=data['sh_coeff'],
            overwrite=True,
        )


@pytest.mark.skipif(
    TEST_SCANNED_TOYS is None,
    reason="'KAOLIN_TEST_SCANNED_TOYS' environment variable is not set (will download files if needed).",
)
class TestGaussianImportFromToysPly:
    """Tests for importing gaussians from TOYS dataset ``.ply`` files."""

    @pytest.fixture(scope='class', autouse=True)
    def download_toys_dataset(self):
        download_scanned_toys_dataset()

    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_import_toy_file(self, toy_name):
        """Import gaussians from each toy PLY file and validate cloud structure."""
        path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.ply')
        cloud = ply.import_gaussiancloud(path)
        assert cloud.check_sanity()

    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_import_gaussiancloud_default(self, toy_name):
        """``import_gaussiancloud`` returns a single cloud dict for the file."""
        path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.ply')
        cloud = ply.import_gaussiancloud(path)
        assert isinstance(cloud, GaussianSplatModel)
        assert cloud.check_sanity()

    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_import_export_roundtrip_from_toy(self, out_dir, toy_name):
        """Import cloud from toy PLY, export to a temp PLY, re-import and compare."""
        path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.ply')
        cloud = ply.import_gaussiancloud(path)

        out_path = os.path.join(out_dir, f'gaussian_roundtrip_{toy_name}.ply')
        ply.export_gaussiancloud(
            out_path,
            **cloud.as_dict(only_tensors=True),
            overwrite=True,
        )

        reimported = ply.import_gaussiancloud(out_path)

        assert contained_torch_equal(cloud.as_dict(), reimported.as_dict(),
                                     approximate=True, print_error_context=''), f'Mismatch for toy {toy_name}'

    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_import_vs_ground_truth_pt(self, toy_name):
        """Compare PLY import against the companion ground-truth ``.pt`` tensor dict."""
        ply_path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.ply')
        gt_path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.pt')

        expected = torch.load(gt_path, weights_only=True)
        expected['orientations'] = torch.nn.functional.normalize(expected['orientations'])
        del expected['local_to_world']
        actual = ply.import_gaussiancloud(ply_path)

        assert contained_torch_equal(actual.as_dict(only_tensors=True), expected,
                                     approximate=True, print_error_context=''), f'Mismatch for toy {toy_name}'