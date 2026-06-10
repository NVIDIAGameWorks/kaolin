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
"""Smoke tests for the GaussianSplatSegmenter / SceneSegmenter contract (Option A).

Exercises the segmenter library end-to-end (render / project / subscene / optimize) without
Cutie or SAM, plus a Cutie-gated end-to-end tracking test. Requires CUDA + gsplat.
"""
import math
import pytest
import torch

gsplat = pytest.importorskip("gsplat")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

import kaolin
from kaolin.rep import GaussianSplatModel
from kaolin.app.segment.base import SceneSegmenter
from kaolin.app.segment.segmenter import GaussianSplatSegmenter


def _toy_gsmodel(n=200, device="cuda", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    positions = (torch.rand(n, 3, generator=g) - 0.5) * 0.6           # in a small cube around origin
    orientations = torch.zeros(n, 4)
    orientations[:, 0] = 1.0                                          # identity quat (wxyz)
    scales = torch.full((n, 3), 0.02)                                 # small, positive (post-activation)
    opacities = torch.full((n,), 0.9)                                 # in [0, 1]
    sh_coeff = torch.rand(n, 1, 3)                                    # sh_degree 0 -> DC only
    return GaussianSplatModel(
        positions=positions, orientations=orientations, scales=scales,
        opacities=opacities, sh_coeff=sh_coeff, sh_degree=0,
    ).to(device)


def _camera(device="cuda", w=128, h=128):
    return kaolin.render.camera.Camera.from_args(
        eye=torch.tensor([0.4, -0.6, 0.5]), at=torch.tensor([0., 0., 0.]),
        up=torch.tensor([0., 0., 1.]), fov=math.pi * 50 / 180, width=w, height=h,
    ).to(device)


def test_is_scene_segmenter():
    seg = GaussianSplatSegmenter(_toy_gsmodel())
    assert isinstance(seg, SceneSegmenter)
    assert seg.num_primitives == 200


def test_render_contract():
    seg = GaussianSplatSegmenter(_toy_gsmodel())
    out = seg.render(_camera())
    assert set(['render', 'depth', 'visibility_filter']).issubset(out.keys())
    assert out['render'].shape == (3, 128, 128)
    assert out['depth'].shape == (128, 128)
    assert out['visibility_filter'].shape == (seg.num_primitives,)
    assert out['render'].min() >= 0.0 and out['render'].max() <= 1.0


def test_project_2d_mask_to_3d_and_subscene():
    seg = GaussianSplatSegmenter(_toy_gsmodel())
    cam = _camera()
    mask_2d = torch.zeros(128, 128, dtype=torch.bool, device="cuda")
    mask_2d[32:96, 32:96] = True
    mask_3d = seg.project_2d_mask_to_3d(mask_2d, cam, bbox_padding=4)
    assert mask_3d.dtype == torch.bool and mask_3d.shape == (seg.num_primitives,)

    sub = seg.subscene_from_3d_mask(mask_3d)
    assert isinstance(sub, GaussianSplatSegmenter)
    assert sub.num_primitives == int(mask_3d.sum())
    assert sub.parent_mask is not None


def test_optimize_segmentation_runs_and_backprops():
    seg = GaussianSplatSegmenter(_toy_gsmodel())
    cam = _camera()
    mask_2d = torch.zeros(128, 128, device="cuda")
    mask_2d[40:88, 40:88] = 1.0
    res = seg.optimize_segmentation([mask_2d], [cam], lr=1e-1, progress=False)
    assert res['mask_3d'].dtype == torch.bool
    assert res['mask_3d'].shape == (seg.num_primitives,)
    assert res['logits'].shape == (seg.num_primitives,)
    assert torch.isfinite(res['logits']).all()


def test_tracking_module_imports_without_cutie():
    # Importing the tracking module + constructing the facade must not require Cutie.
    from kaolin.app.segment.tracking import GsplatSegmentationTracker, TrackingSession  # noqa: F401
    tracker = GsplatSegmentationTracker()
    assert tracker.num_reference_masks == 0
    assert tracker.can_track() is False


def _orbit_cameras(azimuths, device="cuda", w=96, h=96):
    cams = []
    for az in azimuths:
        eye = torch.tensor([0.8 * math.sin(az + 0.6), -0.8 * math.cos(az + 0.6), 0.5])
        cams.append(kaolin.render.camera.Camera.from_args(
            eye=eye, at=torch.tensor([0., 0., 0.]), up=torch.tensor([0., 0., 1.]),
            fov=math.pi * 50 / 180, width=w, height=h).to(device))
    return cams


def test_track_and_segment_end_to_end():
    """Full GsplatSegmentationTracker.track_and_segment over a few rendered views (needs Cutie)."""
    pytest.importorskip("cutie")
    from kaolin.app.segment.tracking import GsplatSegmentationTracker

    device = "cuda"
    seg = GaussianSplatSegmenter(_toy_gsmodel(n=300, device=device))

    ref_cam = _camera(w=96, h=96)
    ref_rgb = seg.render(ref_cam)['render']                          # (3, 96, 96) in [0, 1]
    ref_mask = torch.zeros(96, 96, device=device)
    ref_mask[16:80, 16:80] = 1.0                                     # generous central region

    tracking_cams = _orbit_cameras([0.0, math.pi / 4, -math.pi / 4])

    tracker = GsplatSegmentationTracker(mem_every=5, device=device)
    tracker.add_ground_truth_mask(ref_rgb, ref_mask, ref_cam)
    assert tracker.num_reference_masks == 1 and tracker.can_track()

    res = tracker.track_and_segment(seg, tracking_cams, refine=False)

    n = seg.num_primitives
    assert res['mask_3d'].dtype == torch.bool and res['mask_3d'].shape == (n,)
    assert res['3d_mask'] is res['mask_3d']                          # legacy alias
    assert len(res['masks_2d']) == len(tracking_cams)
    assert len(res['outputs']) == len(tracking_cams)

    assert tracker.num_predicted_masks == len(tracking_cams)
    assert tracker.predicted_mask(0).shape == (96, 96)
    assert tracker.target_cam(0) is tracking_cams[0]
    assert tracker.target_view(0).shape == (3, 96, 96)

    pm = tracker.predict_mask(seg.render(tracking_cams[1])['render'])
    assert pm.shape == (96, 96)


def test_tracker_wraps_gaussiansplatinput():
    """Option A: the facade accepts the app's GaussianSplatInput (wrapping its .gsmodel)."""
    from kaolin.app.segment.tracking import _as_segmenter
    from kaolin.app.segment.input_abstraction import GaussianSplatInput
    cloud = GaussianSplatInput(_toy_gsmodel(n=50))
    seg = _as_segmenter(cloud)
    assert isinstance(seg, GaussianSplatSegmenter)
    assert seg.num_primitives == 50
