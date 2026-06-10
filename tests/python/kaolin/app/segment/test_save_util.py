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
import torch
import pytest

pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdGeom

from kaolin.rep import GaussianSplatModel
from kaolin.app.segment.input_abstraction import GaussianSplatInput
from kaolin.app.segment.segment import DisjointSegmentation
from kaolin.app.segment.save_util import export_scene_as_usd, load_scene_from_usd


def _toy_cloud(n=10):
    gsm = GaussianSplatModel(
        positions=torch.randn(n, 3),
        orientations=torch.cat([torch.ones(n, 1), torch.zeros(n, 3)], dim=1),
        scales=torch.full((n, 3), 0.01),
        opacities=torch.full((n,), 0.9),
        sh_coeff=torch.rand(n, 1, 3),
        sh_degree=0,
    )
    return GaussianSplatInput(gsm)


def _toy_segmentation(n=10):
    """Background covers points 5-9; 'foreground' covers points 0-4."""
    seg = DisjointSegmentation(n, 'cpu')
    mask = torch.zeros(n, dtype=torch.bool)
    mask[:5] = True
    seg.add_segment('foreground', mask)
    return seg


def test_export_creates_usd_file(tmp_path):
    path = str(tmp_path / 'out.usd')
    export_scene_as_usd(_toy_cloud(), _toy_segmentation(), path)
    assert os.path.exists(path)


def test_export_has_gaussian_prim(tmp_path):
    path = str(tmp_path / 'out.usd')
    export_scene_as_usd(_toy_cloud(), _toy_segmentation(), path)
    stage = Usd.Stage.Open(path)
    prim = stage.GetPrimAtPath('/World/Gaussians/gaussian_0')
    assert prim.IsValid()
    assert prim.GetTypeName() == 'ParticleField3DGaussianSplat'


def test_export_subsets_cover_all_segments(tmp_path):
    path = str(tmp_path / 'out.usd')
    export_scene_as_usd(_toy_cloud(), _toy_segmentation(), path)
    stage = Usd.Stage.Open(path)
    prim = stage.GetPrimAtPath('/World/Gaussians/gaussian_0')
    subsets = UsdGeom.Subset.GetGeomSubsets(UsdGeom.Imageable(prim), familyName='part')
    names = {s.GetPrim().GetName() for s in subsets}
    assert 'foreground' in names
    assert 'background' in names


def test_export_subset_indices_match_mask(tmp_path):
    path = str(tmp_path / 'out.usd')
    export_scene_as_usd(_toy_cloud(), _toy_segmentation(), path)
    stage = Usd.Stage.Open(path)
    prim = stage.GetPrimAtPath('/World/Gaussians/gaussian_0')
    subsets = {
        s.GetPrim().GetName(): s
        for s in UsdGeom.Subset.GetGeomSubsets(UsdGeom.Imageable(prim), familyName='part')
    }
    assert list(subsets['foreground'].GetIndicesAttr().Get()) == list(range(5))
    assert list(subsets['background'].GetIndicesAttr().Get()) == list(range(5, 10))


def _make_usd_two_prims(tmp_path, n_a=6, n_b=4):
    """Write a USD with two separate prim paths and no subsets. Returns (path, n_a, n_b)."""
    from kaolin.io.usd.utils import create_stage
    from kaolin.io.usd.gaussians import add_gaussiancloud
    path = str(tmp_path / 'two_prims.usd')
    cloud_a = _toy_cloud(n_a)
    cloud_b = _toy_cloud(n_b)
    stage = create_stage(path)
    add_gaussiancloud(stage, '/World/Gaussians/gaussian_0',
                      positions=cloud_a.gsmodel.positions,
                      orientations=cloud_a.gsmodel.orientations,
                      scales=cloud_a.gsmodel.scales,
                      opacities=cloud_a._orig_opacities,
                      sh_coeff=cloud_a.gsmodel.sh_coeff)
    add_gaussiancloud(stage, '/World/Gaussians/gaussian_1',
                      positions=cloud_b.gsmodel.positions,
                      orientations=cloud_b.gsmodel.orientations,
                      scales=cloud_b.gsmodel.scales,
                      opacities=cloud_b._orig_opacities,
                      sh_coeff=cloud_b.gsmodel.sh_coeff)
    stage.Save()
    del stage
    return path, n_a, n_b


def test_load_round_trip(tmp_path):
    """Export cloud with 2 named segments + background, load back, verify names and indices."""
    n = 10
    path = str(tmp_path / 'scene.usd')
    export_scene_as_usd(_toy_cloud(n), _toy_segmentation(n), path)

    loaded_cloud, loaded_seg = load_scene_from_usd(path, 'cpu')

    assert len(loaded_cloud) == n
    names = set(loaded_seg.segments.keys())
    assert 'foreground' in names
    assert 'background' in names
    fg_indices = loaded_seg.segments['foreground'].mask.nonzero(as_tuple=False).squeeze(1).tolist()
    bg_indices = loaded_seg.segments['background'].mask.nonzero(as_tuple=False).squeeze(1).tolist()
    assert sorted(fg_indices) == list(range(5))
    assert sorted(bg_indices) == list(range(5, 10))


def test_load_multi_prim_no_subsets(tmp_path):
    """Two prims, no subsets → two segments named by prim leaf component."""
    path, n_a, n_b = _make_usd_two_prims(tmp_path)

    loaded_cloud, loaded_seg = load_scene_from_usd(path, 'cpu')

    assert len(loaded_cloud) == n_a + n_b
    names = set(loaded_seg.segments.keys()) - {'background'}
    assert 'gaussian_0' in names
    assert 'gaussian_1' in names
    g0_indices = loaded_seg.segments['gaussian_0'].mask.nonzero(as_tuple=False).squeeze(1).tolist()
    g1_indices = loaded_seg.segments['gaussian_1'].mask.nonzero(as_tuple=False).squeeze(1).tolist()
    assert sorted(g0_indices) == list(range(n_a))
    assert sorted(g1_indices) == list(range(n_a, n_a + n_b))


def test_load_partial_subset_coverage(tmp_path):
    """One prim + one subset covering half → two segments: subset name + prim leaf for remainder."""
    from kaolin.io.usd.utils import create_stage
    from kaolin.io.usd.gaussians import add_gaussiancloud
    from kaolin.io.usd.subset import add_subset
    n = 10
    path = str(tmp_path / 'partial.usd')
    cloud = _toy_cloud(n)
    stage = create_stage(path)
    add_gaussiancloud(stage, '/World/Gaussians/gaussian_0',
                      positions=cloud.gsmodel.positions,
                      orientations=cloud.gsmodel.orientations,
                      scales=cloud.gsmodel.scales,
                      opacities=cloud._orig_opacities,
                      sh_coeff=cloud.gsmodel.sh_coeff)
    # subset covering only first 5 points
    indices = torch.arange(5).long()
    add_subset(stage, '/World/Gaussians/gaussian_0', 'partial_seg', indices, family_name='part')
    stage.Save()
    del stage

    loaded_cloud, loaded_seg = load_scene_from_usd(path, 'cpu')

    assert len(loaded_cloud) == n
    names = set(loaded_seg.segments.keys()) - {'background'}
    assert 'partial_seg' in names, f"Expected 'partial_seg' in {names}"
    assert 'gaussian_0' in names, f"Expected 'gaussian_0' (remainder) in {names}"
    seg_indices = loaded_seg.segments['partial_seg'].mask.nonzero(as_tuple=False).squeeze(1).tolist()
    rem_indices = loaded_seg.segments['gaussian_0'].mask.nonzero(as_tuple=False).squeeze(1).tolist()
    assert sorted(seg_indices) == list(range(5))
    assert sorted(rem_indices) == list(range(5, 10))


def test_load_segment_name_prefix_stripping(tmp_path):
    """Two subsets under same prim → names are just the leaf components, common prefix stripped."""
    n = 10
    path = str(tmp_path / 'strip.usd')
    # Export via the existing helper (writes /World/Gaussians/gaussian_0 with foreground+background subsets)
    export_scene_as_usd(_toy_cloud(n), _toy_segmentation(n), path)

    loaded_cloud, loaded_seg = load_scene_from_usd(path, 'cpu')

    # The raw subset paths are /World/Gaussians/gaussian_0/foreground and /World/Gaussians/gaussian_0/background.
    # After prefix stripping the common /World/Gaussians/gaussian_0/ prefix, names should be 'foreground'/'background'.
    names = set(loaded_seg.segments.keys())
    assert 'foreground' in names
    assert 'background' in names
    # No name should contain a '/' (would indicate prefix was not stripped)
    for name in names:
        assert '/' not in name, f"Segment name '{name}' still contains '/'; prefix stripping failed"
