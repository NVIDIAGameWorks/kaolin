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

import math
import os
import shutil

import numpy as np
import pytest
import torch
from pxr import Sdf, Usd, UsdGeom

from kaolin.io import usd
from kaolin.physics.simplicits import (
    PhysicsPoints,
    SkinnedPhysicsPoints,
    SkinnedPoints,
)
from kaolin.utils.testing import contained_torch_equal


@pytest.fixture(scope='class')
def out_dir():
    d = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_out')
    os.makedirs(d, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _make_physics_points(n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return PhysicsPoints(
        pts=torch.rand(n, 3, generator=g, dtype=torch.float32),
        yms=torch.rand(n, generator=g, dtype=torch.float32) * 1e5 + 1e3,
        prs=torch.rand(n, generator=g, dtype=torch.float32) * 0.4 + 0.05,
        rhos=torch.rand(n, generator=g, dtype=torch.float32) * 1000 + 100,
        # 0-D scalar tensor so the shape matches what get_physics_material returns.
        appx_vol=torch.tensor(0.42 + 0.01 * seed, dtype=torch.float32),
    )


def _make_skinned_physics_points(renderable_pts, n=4, h=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = renderable_pts.shape[0]
    return SkinnedPhysicsPoints(
        pts=torch.rand(n, 3, generator=g, dtype=torch.float32),
        yms=torch.rand(n, generator=g, dtype=torch.float32) * 1e5 + 1e3,
        prs=torch.rand(n, generator=g, dtype=torch.float32) * 0.4 + 0.05,
        rhos=torch.rand(n, generator=g, dtype=torch.float32) * 1000 + 100,
        appx_vol=torch.tensor(0.42 + 0.01 * seed, dtype=torch.float32),
        skinning_weights=torch.rand(n, h, generator=g, dtype=torch.float32),
        dwdx=torch.randn(n, h, 3, generator=g, dtype=torch.float32),
        renderable=SkinnedPoints(
            pts=renderable_pts.to(dtype=torch.float32).clone(),
            skinning_weights=torch.rand(m, h, generator=g, dtype=torch.float32),
        ),
    )


def _make_synthetic_gaussian_cloud(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    theta0 = math.acos(1 / math.sqrt(3)) / 2
    c = 1 / math.sqrt(2)
    rots = torch.tensor(
        [[-c * math.sin(theta0), math.cos(theta0), 0, c * math.sin(theta0)]],
        dtype=torch.float32,
    ).repeat(n, 1)
    return {
        'positions': torch.rand(n, 3, generator=g, dtype=torch.float32),
        'orientations': rots,
        'scales': torch.rand(n, 3, generator=g, dtype=torch.float32) * 0.1 + 0.05,
        'opacities': torch.rand(n, generator=g, dtype=torch.float32),
        'sh_coeff': torch.randn(n, 16, 3, generator=g, dtype=torch.float32) * 0.1,
    }


def _make_mesh_stage(out_dir, fname):
    path = os.path.join(out_dir, fname)
    vertices = torch.tensor(
        [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        dtype=torch.float32,
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
        dtype=torch.long,
    )
    usd.export_mesh(path, vertices=vertices, faces=faces, scene_path='/World/Mesh')
    return path, '/World/Mesh', vertices


def _make_gaussian_stage(out_dir, fname):
    path = os.path.join(out_dir, fname)
    data = _make_synthetic_gaussian_cloud(n=6, seed=0)
    usd.export_gaussiancloud(
        path,
        scene_path='/World/Gaussians/gaussian_0',
        positions=data['positions'],
        orientations=data['orientations'],
        scales=data['scales'],
        opacities=data['opacities'],
        sh_coeff=data['sh_coeff'],
    )
    return path, '/World/Gaussians/gaussian_0', data['positions']


def _make_points_stage(out_dir, fname):
    path = os.path.join(out_dir, fname)
    stage = usd.create_stage(path)
    positions = torch.tensor(
        [[0., 0., 0.], [0.5, 0.5, 0.5], [1., 1., 1.], [0.2, 0.7, 0.4]],
        dtype=torch.float32,
    )
    UsdGeom.Points.Define(stage, '/World/Points').CreatePointsAttr(positions.numpy())
    stage.Save()
    return path, '/World/Points', positions


def _make_point_instancer_stage(out_dir, fname):
    path = os.path.join(out_dir, fname)
    stage = usd.create_stage(path)
    positions = torch.tensor(
        [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        dtype=torch.float32,
    )
    instancer = UsdGeom.PointInstancer.Define(stage, '/World/Instancer')
    instancer.CreatePositionsAttr(positions.numpy())
    instancer.CreateProtoIndicesAttr(
        torch.zeros(positions.shape[0], dtype=torch.int32).numpy()
    )
    stage.Save()
    return path, '/World/Instancer', positions


PRIM_FACTORIES = {
    'mesh': _make_mesh_stage,
    'gaussians': _make_gaussian_stage,
    'points': _make_points_stage,
    'point_instancer': _make_point_instancer_stage,
}


# Geometry attributes we expect to remain bit-identical after applying the
# physics material / skinned physics multi-apply APIs.
PRIM_INTEGRITY_ATTRS = {
    'mesh': ['points', 'faceVertexIndices', 'faceVertexCounts'],
    'gaussians': ['positions', 'orientations', 'scales', 'opacities',
                  'radiance:sphericalHarmonicsCoefficients',
                  'radiance:sphericalHarmonicsDegree'],
    'points': ['points'],
    'point_instancer': ['positions', 'protoIndices'],
}


def _snapshot_attrs(prim, attr_names):
    return {name: np.array(prim.GetAttribute(name).Get()) for name in attr_names}


def _assert_attrs_unchanged(before, after):
    assert before.keys() == after.keys()
    for name, before_val in before.items():
        assert np.array_equal(before_val, after[name]), f"Attribute '{name}' changed"


def _physics_dict(p):
    return {
        'pts': p.pts,
        'yms': p.yms,
        'prs': p.prs,
        'rhos': p.rhos,
        'appx_vol': p.appx_vol,
    }


def _skinned_dict(p):
    d = _physics_dict(p)
    d.update({
        'skinning_weights': p.skinning_weights,
        'dwdx': p.dwdx,
        'renderable_pts': p.renderable.pts,
        'renderable_skinning_weights': p.renderable.skinning_weights,
    })
    return d


def _assert_physics_equal(a, b):
    assert contained_torch_equal(
        _physics_dict(a), _physics_dict(b),
        approximate=True, rtol=1e-5, atol=1e-6,
        print_error_context='',
    )


def _assert_skinned_equal(a, b):
    assert contained_torch_equal(
        _skinned_dict(a), _skinned_dict(b),
        approximate=True, rtol=1e-5, atol=1e-6,
        print_error_context='',
    )


class TestPhysicsMaterial:
    @pytest.mark.parametrize('kind', ['mesh', 'gaussians'])
    @pytest.mark.parametrize('input_stage', [False, True])
    def test_round_trip(self, out_dir, kind, input_stage):
        fname = f'physmat_rt_{kind}_{int(input_stage)}.usda'
        path, prim_path, _ = PRIM_FACTORIES[kind](out_dir, fname)
        material = _make_physics_points(n=4, seed=0)
        target = Usd.Stage.Open(path) if input_stage else path
        usd.add_physics_material(target, prim_path, material)
        result = usd.get_physics_material(target, prim_path)
        _assert_physics_equal(material, result)

    def test_add_returns_prim(self, out_dir):
        path, prim_path, _ = _make_mesh_stage(out_dir, 'physmat_returns_prim.usda')
        stage = Usd.Stage.Open(path)
        prim = usd.add_physics_material(stage, prim_path, _make_physics_points())
        assert isinstance(prim, Usd.Prim)
        assert prim.IsValid()
        assert prim.HasAPI("KaolinPhysicsMaterialAPI", "default")

    def test_add_with_prim_object(self, out_dir):
        path, prim_path, _ = _make_mesh_stage(out_dir, 'physmat_prim_obj.usda')
        stage = Usd.Stage.Open(path)
        prim = stage.GetPrimAtPath(prim_path)
        material = _make_physics_points()
        usd.add_physics_material(stage, prim, material)
        assert prim.HasAPI("KaolinPhysicsMaterialAPI", "default")
        result = usd.get_physics_material(stage, prim)
        _assert_physics_equal(material, result)

    def test_get_returns_none_when_api_absent(self, out_dir):
        path, prim_path, _ = _make_mesh_stage(out_dir, 'physmat_none.usda')
        assert usd.get_physics_material(path, prim_path) is None

    def test_duplicate_raises(self, out_dir):
        path, prim_path, _ = _make_mesh_stage(out_dir, 'physmat_dup.usda')
        usd.add_physics_material(path, prim_path, _make_physics_points(seed=0))
        with pytest.raises(FileExistsError):
            usd.add_physics_material(path, prim_path, _make_physics_points(seed=1))

    def test_overwrite_replaces_values(self, out_dir):
        path, prim_path, _ = _make_mesh_stage(out_dir, 'physmat_overwrite.usda')
        usd.add_physics_material(path, prim_path, _make_physics_points(seed=0))
        new_material = _make_physics_points(seed=1)
        usd.add_physics_material(path, prim_path, new_material, overwrite=True)
        result = usd.get_physics_material(path, prim_path)
        _assert_physics_equal(new_material, result)

    def test_multiple_material_names(self, out_dir):
        path, prim_path, _ = _make_mesh_stage(out_dir, 'physmat_multi.usda')
        soft = _make_physics_points(seed=0)
        stiff = _make_physics_points(seed=1)
        usd.add_physics_material(path, prim_path, soft, material_name='soft')
        usd.add_physics_material(path, prim_path, stiff, material_name='stiff')

        names = set(usd.get_physics_materials_instance_names(path, prim_path))
        assert names == {'soft', 'stiff'}

        all_materials = usd.get_all_physics_materials(path, prim_path)
        assert set(all_materials.keys()) == {'soft', 'stiff'}
        _assert_physics_equal(soft, all_materials['soft'])
        _assert_physics_equal(stiff, all_materials['stiff'])

    def test_custom_material_name(self, out_dir):
        path, prim_path, _ = _make_mesh_stage(out_dir, 'physmat_custom.usda')
        material = _make_physics_points()
        usd.add_physics_material(path, prim_path, material, material_name='rubber')
        assert usd.get_physics_materials_instance_names(path, prim_path) == ['rubber']
        result = usd.get_physics_material(path, prim_path, material_name='rubber')
        _assert_physics_equal(material, result)


class TestSkinnedPhysics:
    @pytest.mark.parametrize('kind', ['mesh', 'gaussians', 'points', 'point_instancer'])
    @pytest.mark.parametrize('input_stage', [False, True])
    def test_round_trip(self, out_dir, kind, input_stage):
        fname = f'skinphys_rt_{kind}_{int(input_stage)}.usda'
        path, prim_path, render_pts = PRIM_FACTORIES[kind](out_dir, fname)
        sp = _make_skinned_physics_points(render_pts, n=4, h=3, seed=0)
        target = Usd.Stage.Open(path) if input_stage else path
        usd.add_skinned_physics(target, prim_path, sp)
        result = usd.get_skinned_physics(target, prim_path)
        _assert_skinned_equal(sp, result)

    def test_add_with_prim_object(self, out_dir):
        path, prim_path, render_pts = _make_mesh_stage(out_dir, 'skinphys_prim_obj.usda')
        stage = Usd.Stage.Open(path)
        prim = stage.GetPrimAtPath(prim_path)
        sp = _make_skinned_physics_points(render_pts, seed=0)
        usd.add_skinned_physics(stage, prim, sp)
        assert prim.HasAPI("KaolinSkinnedPhysicsAPI", "default")
        result = usd.get_skinned_physics(stage, prim)
        _assert_skinned_equal(sp, result)

    def test_round_trip_renderable_none(self, out_dir):
        # Regression: when add_skinned_physics writes a SkinnedPhysicsPoints with
        # renderable=None, no `renderable_skinning_weights` attribute is authored, and
        # get_skinned_physics must return a SkinnedPhysicsPoints with renderable=None
        # (rather than leaving the local unbound and crashing).
        path, prim_path, _ = _make_mesh_stage(out_dir, 'skinphys_renderable_none.usda')
        g = torch.Generator().manual_seed(0)
        n, h = 4, 3
        sp = SkinnedPhysicsPoints(
            pts=torch.rand(n, 3, generator=g, dtype=torch.float32),
            yms=torch.rand(n, generator=g, dtype=torch.float32) * 1e5 + 1e3,
            prs=torch.rand(n, generator=g, dtype=torch.float32) * 0.4 + 0.05,
            rhos=torch.rand(n, generator=g, dtype=torch.float32) * 1000 + 100,
            appx_vol=torch.tensor(0.42, dtype=torch.float32),
            skinning_weights=torch.rand(n, h, generator=g, dtype=torch.float32),
            dwdx=torch.randn(n, h, 3, generator=g, dtype=torch.float32),
            renderable=None,
        )
        usd.add_skinned_physics(path, prim_path, sp)
        result = usd.get_skinned_physics(path, prim_path)
        assert result is not None
        assert result.renderable is None
        assert contained_torch_equal(
            _physics_dict(sp), _physics_dict(result),
            approximate=True, rtol=1e-5, atol=1e-6,
            print_error_context='',
        )
        torch.testing.assert_close(result.skinning_weights, sp.skinning_weights)
        torch.testing.assert_close(result.dwdx, sp.dwdx)

    def test_renderable_explicit_attribute(self, out_dir):
        path, _, _ = _make_mesh_stage(out_dir, 'skinphys_attr.usda')
        stage = Usd.Stage.Open(path)
        xform_path = '/World/XformExplicit'
        UsdGeom.Xform.Define(stage, xform_path)
        prim = stage.GetPrimAtPath(xform_path)
        custom_pts = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            dtype=torch.float32,
        )
        attr = prim.CreateAttribute('extra_points', Sdf.ValueTypeNames.Point3fArray)
        attr.Set(custom_pts.numpy())

        sp = _make_skinned_physics_points(custom_pts, n=4, h=3, seed=0)
        usd.add_skinned_physics(stage, xform_path, sp)
        result = usd.get_skinned_physics(stage, xform_path, attribute='extra_points')
        _assert_skinned_equal(sp, result)

    def test_unsupported_prim_type_raises(self, out_dir):
        path, _, _ = _make_mesh_stage(out_dir, 'skinphys_unsupported.usda')
        stage = Usd.Stage.Open(path)
        xform_path = '/World/XformBare'
        UsdGeom.Xform.Define(stage, xform_path)
        sp = _make_skinned_physics_points(torch.zeros(2, 3), n=4, h=3, seed=0)
        usd.add_skinned_physics(stage, xform_path, sp)
        with pytest.raises(ValueError, match="Unsupported prim type"):
            usd.get_skinned_physics(stage, xform_path)

    def test_duplicate_raises(self, out_dir):
        path, prim_path, render_pts = _make_mesh_stage(out_dir, 'skinphys_dup.usda')
        sp = _make_skinned_physics_points(render_pts, seed=0)
        usd.add_skinned_physics(path, prim_path, sp)
        with pytest.raises(FileExistsError):
            usd.add_skinned_physics(path, prim_path, sp)

    def test_overwrite_replaces_values(self, out_dir):
        path, prim_path, render_pts = _make_mesh_stage(out_dir, 'skinphys_overwrite.usda')
        usd.add_skinned_physics(path, prim_path, _make_skinned_physics_points(render_pts, seed=0))
        new_sp = _make_skinned_physics_points(render_pts, seed=1)
        usd.add_skinned_physics(path, prim_path, new_sp, overwrite=True)
        result = usd.get_skinned_physics(path, prim_path)
        _assert_skinned_equal(new_sp, result)

    def test_multiple_instance_names(self, out_dir):
        path, prim_path, render_pts = _make_mesh_stage(out_dir, 'skinphys_multi.usda')
        a = _make_skinned_physics_points(render_pts, seed=0)
        b = _make_skinned_physics_points(render_pts, seed=1)
        usd.add_skinned_physics(path, prim_path, a, instance_name='a')
        usd.add_skinned_physics(path, prim_path, b, instance_name='b')

        names = set(usd.get_skinned_physics_instance_names(path, prim_path))
        assert names == {'a', 'b'}

        all_sp = usd.get_all_skinned_physics(path, prim_path)
        assert set(all_sp.keys()) == {'a', 'b'}
        _assert_skinned_equal(a, all_sp['a'])
        _assert_skinned_equal(b, all_sp['b'])

    def test_get_returns_none_when_api_absent(self, out_dir):
        path, prim_path, _ = _make_mesh_stage(out_dir, 'skinphys_none.usda')
        assert usd.get_skinned_physics(path, prim_path) is None

    def test_get_all_returns_empty(self, out_dir):
        path, prim_path, _ = _make_mesh_stage(out_dir, 'skinphys_get_all_empty.usda')
        assert usd.get_all_skinned_physics(path, prim_path) == {}

    def test_get_all_forwards_attribute(self, out_dir):
        # Without `attribute`, the Xform fallback would raise. Passing `attribute=`
        # must be forwarded all the way to `_get_renderable_pts` so the explicit
        # path is used instead.
        path, _, _ = _make_mesh_stage(out_dir, 'skinphys_get_all_attr.usda')
        stage = Usd.Stage.Open(path)
        xform_path = '/World/XformGetAll'
        UsdGeom.Xform.Define(stage, xform_path)
        prim = stage.GetPrimAtPath(xform_path)
        custom_pts = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            dtype=torch.float32,
        )
        attr = prim.CreateAttribute('extra_points', Sdf.ValueTypeNames.Point3fArray)
        attr.Set(custom_pts.numpy())

        sp = _make_skinned_physics_points(custom_pts, n=4, h=3, seed=0)
        usd.add_skinned_physics(stage, xform_path, sp, instance_name='custom')

        with pytest.raises(ValueError, match="Unsupported prim type"):
            usd.get_all_skinned_physics(stage, xform_path)

        all_sp = usd.get_all_skinned_physics(stage, xform_path, attribute='extra_points')
        assert set(all_sp.keys()) == {'custom'}
        _assert_skinned_equal(sp, all_sp['custom'])


class TestInstanceNamesConsistency:
    @pytest.mark.parametrize('fn_name', [
        'get_physics_materials_instance_names',
        'get_skinned_physics_instance_names',
    ])
    def test_nonexistent_path_raises(self, out_dir, fn_name):
        # Both *_instance_names go through `_get_stage_from_maybe_file`, so a
        # missing file must raise the same RuntimeError on either function.
        missing = os.path.join(out_dir, 'does_not_exist.usda')
        fn = getattr(usd, fn_name)
        with pytest.raises(RuntimeError, match="File does not exist"):
            fn(missing, '/World/Mesh')


class TestPrimIntegrity:
    @pytest.mark.parametrize('kind', ['mesh', 'gaussians', 'points', 'point_instancer'])
    def test_apply_physics_preserves_geometry(self, out_dir, kind):
        path, prim_path, render_pts = PRIM_FACTORIES[kind](
            out_dir, f'integrity_{kind}.usda',
        )
        stage = Usd.Stage.Open(path)
        prim = stage.GetPrimAtPath(prim_path)
        before = _snapshot_attrs(prim, PRIM_INTEGRITY_ATTRS[kind])

        usd.add_physics_material(stage, prim_path, _make_physics_points())
        usd.add_skinned_physics(
            stage, prim_path, _make_skinned_physics_points(render_pts),
        )

        after = _snapshot_attrs(prim, PRIM_INTEGRITY_ATTRS[kind])
        _assert_attrs_unchanged(before, after)
