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

from kaolin.io import usd, ply, import_gaussiancloud
from kaolin.utils.bundled_data import SCANNED_TOYS_PATH, SCANNED_TOYS_NAMES, download_scanned_toys_dataset
from kaolin.utils.env_vars import KaolinTestEnvVars
from kaolin.utils.testing import contained_torch_equal, with_seed

TEST_SCANNED_TOYS = os.getenv(KaolinTestEnvVars.TEST_SCANNED_TOYS)

#_GAUSSIAN_CLOUD_TENSOR_KEYS = ('positions', 'orientations', 'scales', 'opacities', 'sh_coeff')

# TODO: this needs to go into testing utils -- already 3 copies in test code
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

_CUDA_MARK = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')


@pytest.fixture(scope='class')
def out_dir():
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_out')
    os.makedirs(out_dir, exist_ok=True)
    yield out_dir
    shutil.rmtree(out_dir, ignore_errors=True)


# -----------------------------------------------------------------------------
# Cross-format tests (synthetic data, no dataset)
# -----------------------------------------------------------------------------

class TestSyntheticGaussianImportExport:
    """USD vs PLY vs :func:`kaolin.io.import_gaussiancloud` on the same synthetic cloud."""

    @pytest.mark.parametrize('device', [
        'cpu',
        pytest.param('cuda', marks=_CUDA_MARK),
    ])
    def test_export_import_usd_ply_consistency(self, out_dir, device):
        """Write the same synthetic cloud to USD and PLY; readers agree with each other and the source."""
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}

        scene_path = '/World/Gaussians/gaussian_0'
        usd_path = os.path.join(out_dir, f'cross_fmt_{device}.usdc')
        ply_path = os.path.join(out_dir, f'cross_fmt_{device}.ply')

        usd.export_gaussiancloud(
            usd_path,
            scene_path=scene_path,
            positions=data['positions'],
            orientations=data['orientations'],
            scales=data['scales'],
            opacities=data['opacities'],
            sh_coeff=data['sh_coeff'],
            overwrite=True,
        )
        ply.export_gaussiancloud(
            ply_path,
            positions=data['positions'],
            orientations=data['orientations'],
            scales=data['scales'],
            opacities=data['opacities'],
            sh_coeff=data['sh_coeff'],
            overwrite=True,
        )

        from_ply = ply.import_gaussiancloud(ply_path)
        from_usd = usd.import_gaussiancloud(usd_path)

        compare_kwargs = {'approximate': True, 'ignore_device': True, 'print_error_context': 'Fail'}
        assert contained_torch_equal(data, from_ply.as_dict(only_tensors=True), **compare_kwargs)
        assert contained_torch_equal(data, from_usd.as_dict(only_tensors=True), **compare_kwargs)
        assert contained_torch_equal(from_ply.as_dict(), from_usd.as_dict(), **compare_kwargs)

    def test_generic_import(self, out_dir):
        """Same as format-specific readers, but both loads use :func:`kaolin.io.import_gaussiancloud`."""
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        scene_path = '/World/Gaussians/gaussian_0'
        usd_path = os.path.join(out_dir, 'generic_cross.usdc')
        ply_path = os.path.join(out_dir, 'generic_cross.ply')

        usd.export_gaussiancloud(
            usd_path,
            scene_path=scene_path,
            positions=data['positions'],
            orientations=data['orientations'],
            scales=data['scales'],
            opacities=data['opacities'],
            sh_coeff=data['sh_coeff'],
            overwrite=True,
        )
        ply.export_gaussiancloud(
            ply_path,
            positions=data['positions'],
            orientations=data['orientations'],
            scales=data['scales'],
            opacities=data['opacities'],
            sh_coeff=data['sh_coeff'],
            overwrite=True,
        )

        from_ply = import_gaussiancloud(ply_path)
        from_usd = import_gaussiancloud(usd_path)

        compare_kwargs = {'approximate': True, 'print_error_context': 'Fail'}
        assert contained_torch_equal(data, from_ply.as_dict(only_tensors=True), **compare_kwargs)
        assert contained_torch_equal(data, from_usd.as_dict(only_tensors=True), **compare_kwargs)
        assert contained_torch_equal(from_ply.as_dict(), from_usd.as_dict(), **compare_kwargs)

@pytest.mark.skipif(
    TEST_SCANNED_TOYS is None,
    reason="'KAOLIN_TEST_SCANNED_TOYS' environment variable is not set (will download files if needed).",
)
class TestGaussianImportExportPlyUsd:
    """Toy dataset: PLY and USD imports should describe the same clouds."""

    @pytest.fixture(scope='class', autouse=True)
    def download_toys_dataset(self):
        download_scanned_toys_dataset()

    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_import_ply_usd_consistency(self, toy_name):
        """Test that importing from PLY and USD files gives the same result."""
        ply_path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.ply')
        usd_path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.usdc')

        ply_cloud = ply.import_gaussiancloud(ply_path)
        usd_cloud = usd.import_gaussiancloud(usd_path)

        assert contained_torch_equal(ply_cloud.as_dict(), usd_cloud.as_dict(), approximate=True,
                                     print_error_context=f'Mismatch for toy {toy_name}')

    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_generic_import_consistency(self, toy_name):
        """Same as :meth:`test_import_ply_usd_consistency` via :func:`kaolin.io.import_gaussiancloud`."""
        ply_path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.ply')
        usd_path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.usdc')

        ply_cloud = import_gaussiancloud(ply_path)
        usd_cloud = import_gaussiancloud(usd_path)

        assert contained_torch_equal(ply_cloud.as_dict(), usd_cloud.as_dict(), approximate=True,
                                     print_error_context=f'Mismatch for toy {toy_name} (generic import)')
