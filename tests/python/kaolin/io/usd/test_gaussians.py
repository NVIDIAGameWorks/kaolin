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
from pxr import Sdf, Usd, UsdGeom, UsdVol

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
        'sh_coeff': torch.randn(n, 16, 3, dtype=torch.float32) * 0.1
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
            imported = usd.import_gaussianclouds(stage, [scene_path], return_list=False)
        else:
            imported = usd.import_gaussianclouds(out_path, [scene_path], return_list=False)

        assert contained_torch_equal(
            {scene_path: data},
            {k: v.as_dict(only_tensors=True) for k, v in imported.items()},
            approximate=True,
            rtol=1e-5,
            atol=1e-6,
            print_error_context='Roundtrip failed '
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
            imported = usd.import_gaussianclouds(stage, [scene_path], return_list=False)
        else:
            imported = usd.import_gaussianclouds(out_path, [scene_path], return_list=False)

        assert contained_torch_equal(
            {scene_path: {**data, 'transform': transform}},
            {k: v.as_dict(only_tensors=True) for k, v in imported.items()},
            approximate=True,
            rtol=1e-5,
            atol=1e-5,
            print_error_context='Failed roundtrip with transform'
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
            reimported = usd.import_gaussianclouds(stage, return_list=False)
        else:
            del stage
            reimported = usd.import_gaussianclouds(out_path, return_list=False)

        assert contained_torch_equal(
            clouds,
            {k: v.as_dict(only_tensors=True) for k, v in reimported.items()},
            approximate=True,
            rtol=1e-5,
            atol=1e-6,
            print_error_context='Failed add roundtrip '
        )

    @pytest.mark.parametrize('with_stage', [True, False])
    def test_import_gaussianclouds_default_returns_list(self, out_dir, with_stage):
        """import_gaussianclouds returns a list by default (return_list=True), in scene_paths order."""
        clouds = {
            '/World/Gaussians/gaussian_0': make_synthetic_gaussian_cloud(n=8, seed=0),
            '/World/Gaussians/gaussian_1': make_synthetic_gaussian_cloud(n=5, seed=42),
        }

        out_path = os.path.join(out_dir, 'gaussian_default_list.usdc')
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
            result = usd.import_gaussianclouds(stage)
            scene_paths = usd.get_gaussiancloud_scene_paths(stage)
        else:
            del stage
            result = usd.import_gaussianclouds(out_path)
            scene_paths = usd.get_gaussiancloud_scene_paths(out_path)

        assert isinstance(result, list)
        assert len(result) == len(clouds)
        # The list must follow scene_paths order.
        expected = [clouds[str(sp)] for sp in scene_paths]
        assert contained_torch_equal(
            expected,
            [r.as_dict(only_tensors=True) for r in result],
            approximate=True,
            rtol=1e-5,
            atol=1e-6,
            print_error_context='Default list return mismatch '
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
        assert contained_torch_equal(expected, merged.as_dict(only_tensors=True),
                                     approximate=True, rtol=1e-5, atol=1e-6)

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

        new_xyz, new_rot, new_scales, new_sh_coeff = transform_gaussians(
            data['positions'], data['orientations'], data['scales'],
            transform, sh_coeff=data['sh_coeff'], use_log_scales=False
        )
        expected = {
            'positions': new_xyz,
            'orientations': new_rot,
            'scales': new_scales,
            'opacities': data['opacities'],
            'sh_coeff': new_sh_coeff,
        }
        assert contained_torch_equal(expected, merged.as_dict(only_tensors=True),
                                     approximate=True, rtol=1e-5, atol=1e-5, print_error_context='Fail')

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

        xyz0, rot0, s0, sh0 = transform_gaussians(
            cloud_0['positions'], cloud_0['orientations'], cloud_0['scales'],
            transform_0, sh_coeff=cloud_0['sh_coeff'], use_log_scales=False
        )
        xyz1, rot1, s1, sh1 = transform_gaussians(
            cloud_1['positions'], cloud_1['orientations'], cloud_1['scales'],
            transform_1, sh_coeff=cloud_1['sh_coeff'], use_log_scales=False
        )
        expected = {
            'positions':    torch.cat([xyz0, xyz1], dim=0),
            'orientations': torch.cat([rot0, rot1], dim=0),
            'scales':       torch.cat([s0, s1], dim=0),
            'opacities':    torch.cat([cloud_0['opacities'], cloud_1['opacities']], dim=0),
            'sh_coeff':     torch.cat([sh0, sh1], dim=0),
        }
        assert contained_torch_equal(expected, merged.as_dict(only_tensors=True),
                                     approximate=True, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize('with_stage', [True, False])
    def test_import_gaussiancloud_scene_path(self, out_dir, with_stage):
        """import_gaussiancloud with scene_path merges only clouds under that prefix."""
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
            merged = usd.import_gaussiancloud(stage, scene_path='/World/Foo')
        else:
            del stage
            merged = usd.import_gaussiancloud(out_path, scene_path='/World/Foo')
        assert merged is not None
        assert merged.positions.shape[0] == 8 + 5

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
    def test_get_gaussiancloud_scene_paths_scene_path(self, out_dir, with_stage):
        """get_gaussiancloud_scene_paths with scene_path returns only paths under that prefix"""
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
            actual = usd.get_gaussiancloud_scene_paths(stage, scene_path='/World/Foo')
        else:
            del stage
            actual = usd.get_gaussiancloud_scene_paths(out_path, scene_path='/World/Foo')
        assert set(str(p) for p in actual) == set(foo_paths)


class TestGaussianFeatures:
    """Extra channels on standard Gaussian prims, without dataset downloads."""

    @pytest.mark.parametrize('extension', ['usda', 'usdc'])
    @pytest.mark.parametrize('n', [1, 8])
    @pytest.mark.parametrize('dtype', [torch.float16, torch.float32, torch.float64])
    def test_features_roundtrip(self, tmp_path, extension, n, dtype):
        """Features of each float dtype export as float32 vertex primvars and re-import unchanged."""
        data = make_synthetic_gaussian_cloud(n=n)
        features = {
            'my_embedding': torch.arange(n * 64, dtype=dtype).reshape(n, 64)[:, ::2],
            'confidence': torch.linspace(0, 1, n, dtype=dtype).reshape(n, 1),
        }
        expected_features = {k: v.float() for k, v in features.items()}
        path = str(tmp_path / f'features.{extension}')
        usd.export_gaussiancloud(path, scene_path='/Gaussians', **data, features=features)

        stage = Usd.Stage.Open(path)
        prim = stage.GetPrimAtPath('/Gaussians')
        assert prim.IsA(UsdVol.ParticleField3DGaussianSplat)
        for name, expected in expected_features.items():
            pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar('kaolin:features:' + name)
            assert pv.GetTypeName() == Sdf.ValueTypeNames.FloatArray
            assert pv.GetInterpolation() == UsdGeom.Tokens.vertex
            assert pv.GetElementSize() == expected.shape[1]
            assert not pv.IsIndexed()
            assert torch.equal(torch.tensor(pv.Get()).reshape(n, -1), expected)

        imported = usd.import_gaussiancloud(path)
        assert contained_torch_equal(
            {**data, 'features': expected_features},
            imported.as_dict(only_tensors=True), approximate=True)

    def test_old_positional_signatures(self, tmp_path):
        """Positional calls written before the features argument existed still work."""
        data = make_synthetic_gaussian_cloud()
        args = [data[k] for k in ('positions', 'orientations', 'scales', 'opacities', 'sh_coeff')]
        path = str(tmp_path / 'old_api.usdc')
        usd.export_gaussiancloud(path, '/Gaussians', *args, None, 'Z', None, False)
        assert usd.import_gaussiancloud(path).features is None
        stage = Usd.Stage.CreateInMemory()
        usd.add_gaussiancloud(stage, '/Gaussians', *args, None, None, False)
        assert usd.import_gaussiancloud(stage).features is None

    @pytest.mark.parametrize('features', [None, {}])
    def test_missing_features_preserve_existing(self, features):
        """Overwriting with features=None or {} keeps existing features and foreign primvars."""
        data = make_synthetic_gaussian_cloud()
        stage = Usd.Stage.CreateInMemory()
        original = {'weights': torch.ones(8, 3)}
        prim = usd.add_gaussiancloud(stage, '/Gaussians', **data, features=original)
        api = UsdGeom.PrimvarsAPI(prim)
        api.CreatePrimvar('foreign:data', Sdf.ValueTypeNames.IntArray).Set([42])
        usd.add_gaussiancloud(stage, '/Gaussians', **data, overwrite=True, features=features)
        imported = usd.import_gaussiancloud(stage)
        assert contained_torch_equal(original, imported.features)
        assert list(api.GetPrimvar('foreign:data').Get()) == [42]

    @pytest.mark.parametrize('features', [None, {'other': torch.ones(6, 1)}])
    def test_count_change_rejects_retained_features(self, features):
        """Overwriting with a new Gaussian count raises unless every existing feature is passed again."""
        stage = Usd.Stage.CreateInMemory()
        usd.add_gaussiancloud(
            stage, '/Gaussians', **make_synthetic_gaussian_cloud(n=4), features={'weights': torch.ones(4, 2)})
        before = stage.GetRootLayer().ExportToString()
        data = make_synthetic_gaussian_cloud(n=6)
        with pytest.raises(ValueError, match="Existing feature 'weights'"):
            usd.add_gaussiancloud(stage, '/Gaussians', **data, overwrite=True, features=features)
        assert stage.GetRootLayer().ExportToString() == before

        replacement = {'weights': torch.zeros(6, 2)}
        usd.add_gaussiancloud(stage, '/Gaussians', **data, overwrite=True, features=replacement)
        imported = usd.import_gaussiancloud(stage)
        assert imported.positions.shape[0] == 6
        assert contained_torch_equal(replacement, imported.features)

    def test_time_sample_count_change_rejects_held_features(self):
        """Writing a new Gaussian count at another time raises unless a feature held from an earlier sample is passed again."""
        stage = Usd.Stage.CreateInMemory()
        small = make_synthetic_gaussian_cloud(n=4)
        large = make_synthetic_gaussian_cloud(n=6)
        usd.add_gaussiancloud(stage, '/Gaussians', **small, time=1, features={'weights': torch.ones(4, 2)})
        before = stage.GetRootLayer().ExportToString()
        with pytest.raises(ValueError, match="Existing feature 'weights'"):
            usd.add_gaussiancloud(stage, '/Gaussians', **large, time=2, overwrite=True)
        assert stage.GetRootLayer().ExportToString() == before

        # Same count keeps the held feature; a new count works once the feature is passed again.
        usd.add_gaussiancloud(stage, '/Gaussians', **small, time=2, overwrite=True)
        usd.add_gaussiancloud(
            stage, '/Gaussians', **large, time=3, overwrite=True, features={'weights': torch.zeros(6, 2)})
        for time, count in ((1, 4), (2, 4), (3, 6)):
            imported = usd.import_gaussiancloud(stage, time=time)
            assert imported.positions.shape[0] == count
            assert imported.features['weights'].shape == (count, 2)

    def test_indexed_import_and_dense_update_in_stronger_layer(self):
        """Indexed features are resolved on import; a dense update in a stronger layer replaces them."""
        data = make_synthetic_gaussian_cloud(n=3)
        base = Usd.Stage.CreateInMemory()
        prim = usd.add_gaussiancloud(
            base, '/Gaussians', **data, features={'retained': torch.ones(3, 1)})
        pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            'kaolin:features:weights', Sdf.ValueTypeNames.FloatArray,
            UsdGeom.Tokens.vertex, 2)
        pv.Set([1., 2., 3., 4.])
        pv.SetIndices([1, 0, 1])
        imported = usd.import_gaussiancloud(base)
        assert torch.equal(imported.features['weights'], torch.tensor([[3., 4.], [1., 2.], [3., 4.]]))

        stage = Usd.Stage.CreateInMemory()
        stage.GetRootLayer().subLayerPaths.append(base.GetRootLayer().identifier)
        replacement = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        usd.add_gaussiancloud(
            stage, '/Gaussians', **data, overwrite=True, features={'weights': replacement})
        imported = usd.import_gaussiancloud(stage)
        assert torch.equal(imported.features['weights'], replacement)
        assert torch.equal(imported.features['retained'], torch.ones(3, 1))
        pv = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath('/Gaussians')).GetPrimvar('kaolin:features:weights')
        assert not pv.IsIndexed()

    def test_time_samples_and_blocked_features(self):
        """Features are stored per time sample; a blocked feature is skipped on import."""
        data = make_synthetic_gaussian_cloud(n=2)
        stage = Usd.Stage.CreateInMemory()
        for time in (1, 2):
            usd.add_gaussiancloud(
                stage, '/Gaussians', **data, time=time, overwrite=True,
                features={'weights': torch.full((2, 3), float(time))})
        for time in (1, 2):
            imported = usd.import_gaussiancloud(stage, time=time)
            assert torch.equal(imported.features['weights'], torch.full((2, 3), float(time)))
        api = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath('/Gaussians'))
        api.BlockPrimvar('primvars:kaolin:features:weights')
        assert usd.import_gaussiancloud(stage, time=1).features is None

    @pytest.mark.parametrize('sampled', ['values', 'indices', 'update'])
    def test_indexed_animation_update_rejected_without_mutation(self, sampled):
        """Updating an indexed feature that has time samples raises and leaves the stage unchanged."""
        data = make_synthetic_gaussian_cloud(n=3)
        stage = Usd.Stage.CreateInMemory()
        prim = usd.add_gaussiancloud(stage, '/Gaussians', **data)
        pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            'kaolin:features:weights', Sdf.ValueTypeNames.FloatArray,
            UsdGeom.Tokens.vertex, 2)
        pv.Set([1., 2., 3., 4.], 1 if sampled == 'values' else Usd.TimeCode.Default())
        pv.SetIndices([1, 0, 1], 1 if sampled == 'indices' else Usd.TimeCode.Default())
        before = stage.GetRootLayer().ExportToString()
        with pytest.raises(ValueError, match='indexed feature'):
            usd.add_gaussiancloud(
                stage, '/Gaussians', **data, overwrite=True,
                time=2 if sampled == 'update' else None,
                features={'weights': torch.ones(3, 2)})
        assert stage.GetRootLayer().ExportToString() == before

    @pytest.mark.parametrize('features, message', [
        (torch.ones(8, 2), 'features must be a dict'),
        ({1: torch.ones(8, 2)}, 'Invalid feature name'),
        ({'': torch.ones(8, 2)}, 'Invalid feature name'),
        ({'with space': torch.ones(8, 2)}, 'Invalid feature name'),
        ({'nested:name': torch.ones(8, 2)}, 'Invalid feature name'),
        ({'indices': torch.ones(8, 2)}, 'Invalid feature name'),
        ({'weights': [[1., 2.]]}, 'floating tensor'),
        ({'weights': torch.ones(8, 2, dtype=torch.int64)}, 'floating tensor'),
        ({'weights': torch.ones(8, 2, dtype=torch.bool)}, 'floating tensor'),
        ({'weights': torch.ones(8, 2, dtype=torch.complex64)}, 'floating tensor'),
        ({'weights': torch.ones(8)}, 'must have shape'),
        ({'weights': torch.ones(8, 2, 2)}, 'must have shape'),
        ({'weights': torch.ones(7, 2)}, 'must have shape'),
        ({'weights': torch.ones(8, 0)}, 'must have shape'),
    ])
    def test_invalid_features_do_not_modify_stage_or_file(self, tmp_path, features, message):
        """Invalid features raise before the stage or the output file is modified."""
        data = make_synthetic_gaussian_cloud()
        stage = Usd.Stage.CreateInMemory()
        usd.add_gaussiancloud(stage, '/Gaussians', **data)
        before = stage.GetRootLayer().ExportToString()
        with pytest.raises(ValueError, match=message):
            usd.add_gaussiancloud(
                stage, '/Gaussians', **{**data, 'positions': data['positions'] + 1},
                overwrite=True, features=features)
        assert stage.GetRootLayer().ExportToString() == before

        path = tmp_path / 'invalid.usdc'
        with pytest.raises(ValueError, match=message):
            usd.export_gaussiancloud(str(path), **data, features=features)
        assert not path.exists()
        path.write_bytes(b'existing contents')
        with pytest.raises(ValueError, match=message):
            usd.export_gaussiancloud(str(path), **data, features=features, overwrite=True)
        assert path.read_bytes() == b'existing contents'

    @pytest.mark.parametrize('change', ['type', 'interpolation', 'width'])
    def test_incompatible_declaration_rejected_before_update(self, change):
        """Updating a feature whose primvar has another type, interpolation or width raises and leaves the stage unchanged."""
        data = make_synthetic_gaussian_cloud()
        stage = Usd.Stage.CreateInMemory()
        prim = usd.add_gaussiancloud(stage, '/Gaussians', **data)
        UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            'kaolin:features:weights',
            Sdf.ValueTypeNames.IntArray if change == 'type' else Sdf.ValueTypeNames.FloatArray,
            UsdGeom.Tokens.constant if change == 'interpolation' else UsdGeom.Tokens.vertex,
            3 if change == 'width' else 2)
        before = stage.GetRootLayer().ExportToString()
        with pytest.raises(ValueError):
            usd.add_gaussiancloud(
                stage, '/Gaussians', **data, overwrite=True,
                features={'valid_first': torch.ones(8, 1), 'weights': torch.ones(8, 2)})
        assert stage.GetRootLayer().ExportToString() == before

    @pytest.mark.parametrize('problem', ['type', 'interpolation', 'width', 'count', 'indices'])
    def test_invalid_owned_primvar_rejected_on_import(self, problem):
        """Import raises on a malformed kaolin:features primvar."""
        data = make_synthetic_gaussian_cloud(n=3)
        stage = Usd.Stage.CreateInMemory()
        prim = usd.add_gaussiancloud(stage, '/Gaussians', **data)
        pv = UsdGeom.PrimvarsAPI(prim).CreatePrimvar(
            'kaolin:features:weights',
            Sdf.ValueTypeNames.IntArray if problem == 'type' else Sdf.ValueTypeNames.FloatArray,
            UsdGeom.Tokens.constant if problem == 'interpolation' else UsdGeom.Tokens.vertex,
            2)
        pv.Set([1, 2, 3, 4] if problem == 'count' else [1, 2, 3, 4, 5, 6])
        if problem == 'width':
            pv.GetAttr().SetMetadata('elementSize', 0)
        if problem == 'indices':
            pv.SetIndices([0, 1, 99])
        with pytest.raises(ValueError, match='feature'):
            usd.import_gaussiancloud(stage)

    def test_foreign_primvars_are_not_features(self):
        """Primvars outside the kaolin:features namespace are not imported as features."""
        data = make_synthetic_gaussian_cloud()
        stage = Usd.Stage.CreateInMemory()
        prim = usd.add_gaussiancloud(stage, '/Gaussians', **data)
        api = UsdGeom.PrimvarsAPI(prim)
        api.CreatePrimvar('displayColor', Sdf.ValueTypeNames.Color3fArray).Set([(1, 0, 0)])
        api.CreatePrimvar('other:embedding', Sdf.ValueTypeNames.FloatArray).Set([1, 2])
        assert usd.import_gaussiancloud(stage).features is None

    @pytest.mark.parametrize('mismatch', [None, 'missing', 'keys', 'width'])
    def test_features_merge(self, mismatch):
        """Merged import concatenates matching features and raises when feature sets differ between prims."""
        data = make_synthetic_gaussian_cloud(n=2)
        stage = Usd.Stage.CreateInMemory()
        features = {'weights': torch.arange(4, dtype=torch.float32).reshape(2, 2)}
        transform = torch.eye(4)
        transform[:3, 3] = torch.tensor([1., 2., 3.])
        usd.add_gaussiancloud(stage, '/A', **data, features=features, local_to_world=transform)
        other_features = features
        if mismatch == 'missing':
            other_features = None
        elif mismatch == 'keys':
            other_features = {'other': features['weights']}
        elif mismatch == 'width':
            other_features = {'weights': torch.ones(2, 3)}
        usd.add_gaussiancloud(stage, '/B', **data, features=other_features)
        clouds = usd.import_gaussianclouds(stage, return_list=False)
        assert contained_torch_equal(clouds['/A'].features, features)
        assert contained_torch_equal(clouds['/B'].features, other_features)
        if mismatch is not None:
            with pytest.raises(ValueError):
                usd.import_gaussiancloud(stage)
        else:
            merged = usd.import_gaussiancloud(stage)
            assert torch.equal(merged.features['weights'], features['weights'].repeat(2, 1))
            assert torch.allclose(merged.positions[:2], data['positions'] + transform[:3, 3])


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

        gaussianclouds = usd.import_gaussianclouds(path, scene_paths, return_list=False)

        assert set(gaussianclouds.keys()) == set(str(p) for p in scene_paths)
        for cloud in gaussianclouds.values():
            assert cloud.check_sanity(log_error=True)

    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_import_gaussianclouds_all(self, toy_name):
        """import_gaussianclouds with default scene_pathsreturns dict mapping scene path to cloud."""
        path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.usdc')
        all_clouds = usd.import_gaussianclouds(path, return_list=False)

        scene_paths = usd.get_gaussiancloud_scene_paths(path)

        assert len(all_clouds) == len(scene_paths)
        for sp in scene_paths:
            assert str(sp) in all_clouds

    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_import_export_roundtrip_from_toy(self, out_dir, toy_name):
        """Import cloud from toy, export to temp file, re-import and compare."""
        path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.usdc')
        scene_paths = usd.get_gaussiancloud_scene_paths(path)

        imported = usd.import_gaussianclouds(path, scene_paths, return_list=False)
        # Test round-trip for the first gaussian cloud
        cloud = imported['/World/Gaussians/gaussian_0']

        out_path = os.path.join(out_dir, f'gaussian_roundtrip_{toy_name}.usdc')

        cloud_dict = cloud.as_dict(only_tensors=True)
        local_to_world = None
        if 'transform' in cloud_dict:
            local_to_world = cloud_dict['transform']
            del cloud_dict['transform']
        usd.export_gaussiancloud(
            out_path, **cloud_dict, local_to_world=local_to_world,
            overwrite=True,
        )

        reimported = usd.import_gaussianclouds(out_path, ['/World/Gaussians/gaussian_0'], return_list=False)

        assert contained_torch_equal(
            {'/World/Gaussians/gaussian_0': cloud_dict},
            {k: v.as_dict(only_tensors=True) for k, v in reimported.items()},
            approximate=True,
            rtol=1e-4,
            atol=1e-5,
            print_error_context='Failed toy roundtrip '
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
        if expected['local_to_world'] is None:
            del expected['local_to_world']
        expected['orientations'] = torch.nn.functional.normalize(expected['orientations'])

        scene_paths = ['/World/Gaussians/gaussian_0']
        actual = usd.import_gaussianclouds(path, scene_paths, return_list=False)
        assert len(actual) == 1
        assert contained_torch_equal(
            {k: v.as_dict(only_tensors=True) for k, v in actual.items()},
            {'/World/Gaussians/gaussian_0': expected},
            approximate=True,
            rtol=1e-4,
            atol=1e-5,
            print_error_context=f'Mismatch for cloud {toy_name}'
        ), f'Mismatch for cloud {toy_name}'

    @pytest.mark.parametrize('toy_name', SCANNED_TOYS_NAMES)
    def test_import_gaussianclouds_all_vs_ground_truth(self, toy_name):
        """Compare import_gaussianclouds with default scene_paths output against ground truth .pt file."""
        path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.usdc')
        gt_path = os.path.join(SCANNED_TOYS_PATH, f'{toy_name}.pt')

        expected = torch.load(gt_path, weights_only=True)
        if expected['local_to_world'] is None:
            del expected['local_to_world']
        expected['orientations'] = torch.nn.functional.normalize(expected['orientations'])

        actual = usd.import_gaussianclouds(path, return_list=False)
        assert len(actual) == 1
        assert contained_torch_equal(
            actual['/World/Gaussians/gaussian_0'].as_dict(only_tensors=True), expected,
            approximate=True,
            rtol=1e-4,
            atol=1e-5,
            print_error_context=f'Mismatch for {toy_name}'
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
        if gt['local_to_world'] is None:
            del gt['local_to_world']
        gt['orientations'] = torch.nn.functional.normalize(gt['orientations'])

        scene_paths = ['/World/Gaussians/BluehairRagdoll_0']
        output = usd.import_gaussianclouds(path, scene_paths, return_list=False)

        expected = {'/World/Gaussians/BluehairRagdoll_0': gt}
        
        assert contained_torch_equal(
            {k: v.as_dict(only_tensors=True) for k, v in output.items()},
            expected,
            approximate=True, rtol=1e-4, atol=1e-5), 'Mismatch for BluehairRagdoll_multi'

    def test_import_gaussianclouds_all_combined(self):
        """Import gaussianclouds with default scene_paths from two BluehairRagdoll (2nd is transformed) usdc and compare to ground truth."""
        path = os.path.join(SCANNED_TOYS_PATH, 'BluehairRagdoll_multi.usdc')
        gt = torch.load(os.path.join(SCANNED_TOYS_PATH, 'BluehairRagdoll.pt'), weights_only=True)
        if gt['local_to_world'] is not None:
            gt['transform'] = gt['local_to_world']
        del gt['local_to_world']
        gt['orientations'] = torch.nn.functional.normalize(gt['orientations'])

        transform = torch.tensor([[2, 0, 0, 0.5],
                                   [0, 0, -2, 0],
                                   [0, 2, 0, 0],
                                   [0, 0, 0, 1]], dtype=torch.float32)
        output = usd.import_gaussianclouds(path, return_list=False)

        expected = {
            '/World/Gaussians/BluehairRagdoll_0': gt,
            '/World/Gaussians/BluehairRagdoll_1': {**gt, 'transform': transform},
        }
        assert contained_torch_equal(
            {k: v.as_dict(only_tensors=True) for k, v in output.items()},
            expected,
            approximate=True, rtol=1e-4, atol=1e-5), 'Mismatch for BluehairRagdoll_multi'

    def test_import_gaussianclouds_partial_compressed(self):
        """Import gaussianclouds from two BluehairRagdoll (2nd is transformed) usdc and compare to ground truth."""
        path = os.path.join(SCANNED_TOYS_PATH, 'BluehairRagdoll_compressed.usdc')
        gt = torch.load(os.path.join(SCANNED_TOYS_PATH, 'BluehairRagdoll.pt'), weights_only=True)
        if gt['local_to_world'] is not None:
            gt['transform'] = gt['local_to_world']
        del gt['local_to_world']
        gt['orientations'] = torch.nn.functional.normalize(gt['orientations'])
        gt['sh_coeff'] = gt['sh_coeff'][:, :1]

        transform = torch.tensor([[2, 0, 0, 0.5],
                                  [0, 0, -2, 0],
                                  [0, 2, 0, 0],
                                  [0, 0, 0, 1]], dtype=torch.float32)
        scene_paths = ['/World/Gaussians/BluehairRagdoll_1']
        output = usd.import_gaussianclouds(path, scene_paths, return_list=False)

        gt_half = {k: v.half() if v is not None else None for k, v in gt.items()}
        expected = {'/World/Gaussians/BluehairRagdoll_1': {**gt_half, 'transform': transform}}
        assert contained_torch_equal(
            {k: v.as_dict(only_tensors=True) for k, v in output.items()},
            expected,
            approximate=True, rtol=1e-3, atol=1e-3), 'Mismatch for BluehairRagdoll_compressed'

    def test_import_gaussianclouds_all_compressed(self):
        """Import all gaussianclouds from two BluehairRagdoll (2nd is transformed) usdc and compare to ground truth."""
        path = os.path.join(SCANNED_TOYS_PATH, 'BluehairRagdoll_compressed.usdc')
        gt = torch.load(os.path.join(SCANNED_TOYS_PATH, 'BluehairRagdoll.pt'), weights_only=True)
        if gt['local_to_world'] is not None:
            gt['transform'] = gt['local_to_world']
        del gt['local_to_world']
        gt['orientations'] = torch.nn.functional.normalize(gt['orientations'])
        gt['sh_coeff'] = gt['sh_coeff'][:, :1]

        transform = torch.tensor([[2, 0, 0, 0.5],
                                  [0, 0, -2, 0],
                                  [0, 2, 0, 0],
                                  [0, 0, 0, 1]], dtype=torch.float32)
        output = usd.import_gaussianclouds(path, return_list=False)

        gt_half = {k: v.half() if v is not None else None for k, v in gt.items()}
        expected = {
            '/World/Gaussians/BluehairRagdoll_0': gt_half,
            '/World/Gaussians/BluehairRagdoll_1': {**gt_half, 'transform': transform},
        }
        assert contained_torch_equal(
            {k: v.as_dict(only_tensors=True) for k, v in output.items()},
            expected,
            approximate=True, rtol=1e-3, atol=1e-3), 'Mismatch for BluehairRagdoll_compressed'
