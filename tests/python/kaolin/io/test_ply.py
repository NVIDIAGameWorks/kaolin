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
import numpy as np
import torch
import pytest
from plyfile import PlyData, PlyElement

from kaolin.io import ply
from kaolin.rep import GaussianSplatModel
from kaolin.utils.bundled_data import (SCANNED_TOYS_PATH, SCANNED_TOYS_NAMES, download_scanned_toys_dataset,
                                       TENSOR_IR_PATH, TENSOR_IR_NAMES, download_tensor_ir_dataset)
from kaolin.utils.env_vars import KaolinTestEnvVars
from kaolin.utils.testing import contained_torch_equal, with_seed

TEST_SCANNED_TOYS = os.getenv(KaolinTestEnvVars.TEST_SCANNED_TOYS)
TEST_TENSOR_IR = os.getenv(KaolinTestEnvVars.TEST_TENSOR_IR)

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
        """Import gaussians from each toy PLY file and validate structure."""
        path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.ply')
        gaussians = ply.import_gaussiancloud(path)
        assert gaussians.check_sanity()

    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_import_gaussiancloud_default(self, toy_name):
        """``import_gaussiancloud`` returns a single GaussianSplatModel for the file."""
        path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.ply')
        gaussians = ply.import_gaussiancloud(path)
        assert isinstance(gaussians, GaussianSplatModel)
        assert gaussians.check_sanity()

    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_import_export_roundtrip_from_toy(self, out_dir, toy_name):
        """Import gaussians from toy PLY, export to a temp PLY, re-import and compare."""
        path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.ply')
        gaussians = ply.import_gaussiancloud(path)

        out_path = os.path.join(out_dir, f'gaussian_roundtrip_{toy_name}.ply')
        ply.export_gaussiancloud(
            out_path,
            **gaussians.as_dict(only_tensors=True),
            overwrite=True,
        )

        reimported = ply.import_gaussiancloud(out_path)

        assert contained_torch_equal(gaussians.as_dict(), reimported.as_dict(),
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


# -------------------------------------------------------------------------------
# Tests that run with custom feature options, toys and material objects needed
# -------------------------------------------------------------------------------

EXPECTED_FEATURES_PER_SHAPE = {
    'tensorir_ficus': {'albedo': 3, 'normal': 3, 'metallic': 1, 'roughness': 1},
    'tensorir_lego':  {'albedo': 3, 'normal': 3, 'metallic': 1, 'roughness': 1, 'segment': 1},
}


def _tensor_ir_path(name):
    return os.path.join(TENSOR_IR_PATH, f'{name}.ply')


@pytest.mark.skipif(
    TEST_SCANNED_TOYS is None or TEST_TENSOR_IR is None,
    reason=("'KAOLIN_TEST_SCANNED_TOYS' and 'KAOLIN_TEST_TENSOR_IR' environment variables "
            "must both be set (will download toys, and verify presence of tensor-IR samples)."),
)
class TestGaussianFeaturesPly:
    """Tests for the optional ``features`` dict in :mod:`kaolin.io.ply`."""

    @pytest.fixture(scope='class', autouse=True)
    def download_datasets(self):
        download_scanned_toys_dataset()
        download_tensor_ir_dataset()

    @staticmethod
    def _export(out_path, data, features=None):
        ply.export_gaussiancloud(
            out_path,
            positions=data['positions'],
            orientations=data['orientations'],
            scales=data['scales'],
            opacities=data['opacities'],
            sh_coeff=data['sh_coeff'],
            features=features,
            overwrite=True,
        )

    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_import_toy_has_no_features(self, toy_name):
        """Vanilla 3DGS toy PLYs have no extra properties and must import as features=None."""
        path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.ply')
        assert ply.import_gaussiancloud(path).features is None

    @pytest.mark.parametrize('name', TENSOR_IR_NAMES)
    def test_import_features_from_disk(self, name):
        """Real gaussian PLYs with extra per-point channels import with the expected feature schema."""
        gaussians = ply.import_gaussiancloud(_tensor_ir_path(name))
        expected = EXPECTED_FEATURES_PER_SHAPE[name]
        assert gaussians.check_sanity()
        assert isinstance(gaussians.features, dict)
        assert set(gaussians.features.keys()) == set(expected.keys())
        for key, k in expected.items():
            feat = gaussians.features[key]
            assert isinstance(feat, torch.Tensor), \
                f'{name}: features[{key!r}] is {type(feat).__name__}, expected torch.Tensor'
            assert feat.device.type == 'cpu', \
                f'{name}: features[{key!r}] on device {feat.device}, expected cpu'
            assert feat.shape == (len(gaussians), k), \
                f'{name}: features[{key!r}] shape {feat.shape} != (N={len(gaussians)}, K={k})'

    @pytest.mark.parametrize('name', TENSOR_IR_NAMES)
    def test_features_roundtrip_from_disk(self, out_dir, name):
        """Round-trip a real gaussian PLY with extra features (import -> export -> reimport)."""
        gaussians = ply.import_gaussiancloud(_tensor_ir_path(name))
        out_path = os.path.join(out_dir, f'roundtrip_{name}.ply')
        ply.export_gaussiancloud(out_path, **gaussians.as_dict(only_tensors=True), overwrite=True)
        reimported = ply.import_gaussiancloud(out_path)
        assert contained_torch_equal(gaussians.as_dict(), reimported.as_dict(),
                                     approximate=True, print_error_context=''), f'Mismatch for {name}'

    def test_no_features_import_returns_none(self, out_dir):
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        out_path = os.path.join(out_dir, 'gauss_no_features.ply')
        self._export(out_path, data)
        assert ply.import_gaussiancloud(out_path).features is None

    def test_features_roundtrip_single(self, out_dir):
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        features = {'mat': torch.randn(8, 4, dtype=torch.float32)}
        out_path = os.path.join(out_dir, 'gauss_features_single.ply')
        self._export(out_path, data, features=features)

        imported = ply.import_gaussiancloud(out_path)
        assert contained_torch_equal(features, imported.features, approximate=True, print_error_context='')

    def test_features_roundtrip_multiple(self, out_dir):
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        features = {
            'albedo': torch.randn(8, 3, dtype=torch.float32),
            'roughness': torch.randn(8, 1, dtype=torch.float32),
            'lobe': torch.randn(8, 7, dtype=torch.float32),
        }
        out_path = os.path.join(out_dir, 'gauss_features_multi.ply')
        self._export(out_path, data, features=features)

        imported = ply.import_gaussiancloud(out_path)
        assert contained_torch_equal(features, imported.features, approximate=True, print_error_context='')

    def test_features_roundtrip_name_with_underscore(self, out_dir):
        """Feature names containing underscores group on the last underscore."""
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        features = {'my_extra_feat': torch.randn(8, 3, dtype=torch.float32)}
        out_path = os.path.join(out_dir, 'gauss_features_underscore.ply')
        self._export(out_path, data, features=features)

        imported = ply.import_gaussiancloud(out_path)
        assert contained_torch_equal(features, imported.features, approximate=True, print_error_context='')

    def test_features_import_recovers_shuffled_columns(self, out_dir):
        """Importer must restore column order by the integer suffix, regardless of property order."""
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        features = {'foo': torch.randn(8, 3, dtype=torch.float32)}
        out_path = os.path.join(out_dir, 'gauss_features_shuffled.ply')
        self._export(out_path, data, features=features)

        plydata = PlyData.read(out_path)
        elem = plydata.elements[0]
        order = [p.name for p in elem.properties]
        i0, i1, i2 = order.index('foo_0'), order.index('foo_1'), order.index('foo_2')
        order[i0], order[i2] = order[i2], order[i0]
        reordered = np.empty(elem.data.shape, dtype=[(n, elem.data.dtype[n]) for n in order])
        for n in order:
            reordered[n] = elem.data[n]
        shuffled_path = os.path.join(out_dir, 'gauss_features_shuffled_after.ply')
        PlyData([PlyElement.describe(reordered, 'vertex')]).write(shuffled_path)

        imported = ply.import_gaussiancloud(shuffled_path)
        assert contained_torch_equal(features, imported.features, approximate=True, print_error_context='')

    def test_import_ignores_non_indexed_extra_property(self, out_dir):
        """Extra PLY properties without a ``_<int>`` suffix must not crash or appear in features."""
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        out_path = os.path.join(out_dir, 'gauss_features_extra.ply')
        self._export(out_path, data)

        plydata = PlyData.read(out_path)
        elem = plydata.elements[0]
        names = [p.name for p in elem.properties] + ['extra_metadata']
        extended = np.empty(elem.data.shape, dtype=[(n, 'f4') for n in names])
        for p in elem.properties:
            extended[p.name] = elem.data[p.name]
        extended['extra_metadata'] = np.zeros(elem.data.shape, dtype='f4')
        extended_path = os.path.join(out_dir, 'gauss_features_with_metadata.ply')
        PlyData([PlyElement.describe(extended, 'vertex')]).write(extended_path)

        assert ply.import_gaussiancloud(extended_path).features is None

    def test_export_rejects_non_dict_features(self, out_dir):
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        out_path = os.path.join(out_dir, 'gauss_features_bad_type.ply')
        with pytest.raises(ValueError):
            self._export(out_path, data, features=[torch.zeros(8, 3)])

    def test_export_rejects_non_string_key(self, out_dir):
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        out_path = os.path.join(out_dir, 'gauss_features_bad_key.ply')
        with pytest.raises(ValueError):
            self._export(out_path, data, features={1: torch.zeros(8, 3)})

    @pytest.mark.parametrize('name', ['opacity', 'f_dc', 'f_dc_extra', 'f_rest_99', 'scale_x', 'rot_w', 'x', 'nx'])
    def test_export_rejects_reserved_key(self, out_dir, name):
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        out_path = os.path.join(out_dir, f'gauss_features_reserved_{name}.ply')
        with pytest.raises(ValueError):
            self._export(out_path, data, features={name: torch.zeros(8, 3)})

    def test_export_rejects_non_tensor_value(self, out_dir):
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        out_path = os.path.join(out_dir, 'gauss_features_bad_value.ply')
        with pytest.raises(ValueError):
            self._export(out_path, data, features={'mat': np.zeros((8, 3), dtype=np.float32)})

    def test_export_rejects_wrong_rank(self, out_dir):
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        out_path = os.path.join(out_dir, 'gauss_features_bad_rank.ply')
        with pytest.raises(ValueError):
            self._export(out_path, data, features={'mat': torch.zeros(8)})

    def test_export_rejects_wrong_point_count(self, out_dir):
        data = make_synthetic_gaussian_cloud(n=8, seed=0)
        out_path = os.path.join(out_dir, 'gauss_features_bad_n.ply')
        with pytest.raises(ValueError):
            self._export(out_path, data, features={'mat': torch.zeros(7, 3)})

