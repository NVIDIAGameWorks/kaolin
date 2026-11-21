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
"""Python<->three.js camera render parity.

The three.js viewer converts ``defaultCameraParameters`` (graphics/camera.ts)
into a three.js camera and renders the fox mesh; the resulting silhouette is
committed as a reference PNG by the TS test
``tests/ts/kaolin/visualize/dash/graphics/threejs/test_camera.ts``.

Here we render the *same* fox mesh with Kaolin's easy_render, using a Kaolin
``Camera`` built from the *same* shared golden (``Camera.from_dict``), and check
that the Kaolin silhouette matches the three.js reference. This validates that
the camera conventions agree across the Python and TypeScript stacks.

Design mirrors ``tests/python/kaolin/render/camera/test_gsplats_nerfstudio.py``
``TestCameraConversionRenderParity``, except the "other framework" silhouette is
a committed three.js golden image rather than a second in-process renderer.
"""
import json
import os

import pytest
import torch

import kaolin
import kaolin.io
from kaolin.render.camera import Camera
from kaolin.render.easy_render import render_mesh, RenderPass

# This file: tests/python/kaolin/visualize/dash/components/src/ts/graphics/threejs/<this>
# TODO: replace this hardcoded path navigation with a shared helper that maps a
#       components/src/ts/... module path to its tests/samples data directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS_ROOT = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 9)))  # -> tests/
_REPO_ROOT = os.path.dirname(_TESTS_ROOT)

_SAMPLES = os.path.join(_TESTS_ROOT, 'samples', 'visualize', 'dash', 'components',
                        'src', 'ts', 'graphics')
GOLDEN_CAMERA_PATH = os.path.join(_SAMPLES, 'default_camera_parameters.json')
REFERENCE_PNG = os.path.join(_SAMPLES, 'threejs', 'camera_default_fox.png')
FOX_OBJ = os.path.join(_REPO_ROOT, 'sample_data', 'meshes', 'fox.obj')


def _precision_recall(gt_mask, pred_mask):
    """Boolean-silhouette precision/recall (same metric as the gsplat parity test)."""
    recall = pred_mask[gt_mask].sum() / gt_mask.sum().clamp(min=1)
    precision = gt_mask[pred_mask].sum() / pred_mask.sum().clamp(min=1)
    return precision.item(), recall.item()


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA not available for easy_render mesh rasterization.")
class TestDefaultCameraRenderParity:
    def test_fox_silhouette_matches_threejs_reference(self):
        device = 'cuda'

        # Build the Kaolin camera from the SAME shared golden the TS side uses.
        assert os.path.isfile(GOLDEN_CAMERA_PATH), f'Missing golden camera: {GOLDEN_CAMERA_PATH}'
        with open(GOLDEN_CAMERA_PATH) as f:
            cam_dict = json.load(f)
        camera = Camera.from_dict(cam_dict).to(device)

        # Render the fox silhouette with Kaolin.
        mesh = kaolin.io.import_mesh(FOX_OBJ, triangulate=True).to(device)
        rendering = render_mesh(camera, mesh)
        face_idx = rendering[RenderPass.face_idx]  # (1, H, W)
        kaolin_mask = (face_idx >= 0).reshape(camera.height, camera.width)

        # Load the committed three.js reference silhouette.
        assert os.path.isfile(REFERENCE_PNG), (
            f'Missing three.js reference: {REFERENCE_PNG}. '
            f'Generate it via the TS test render-parity block first.')
        ref = kaolin.io.utils.read_image(REFERENCE_PNG).to(device)  # (H, W, C) in 0..1
        ref_mask = ref[..., 0] > 0.5

        assert ref_mask.shape == kaolin_mask.shape, (
            f'Reference {tuple(ref_mask.shape)} and kaolin {tuple(kaolin_mask.shape)} '
            f'silhouettes differ in size')

        # Sanity: both silhouettes should be sensible, non-degenerate.
        assert 0.02 < kaolin_mask.float().mean().item() < 0.9, 'Kaolin silhouette looks wrong'
        assert 0.02 < ref_mask.float().mean().item() < 0.9, 'three.js reference looks wrong'

        precision, recall = _precision_recall(ref_mask, kaolin_mask)
        assert recall > 0.9, (
            f'Kaolin silhouette misses the three.js reference (recall={recall:.4f}); '
            f'camera conventions may have diverged across py/ts.')
        assert precision > 0.9, (
            f'Kaolin silhouette overshoots the three.js reference (precision={precision:.4f}); '
            f'camera conventions may have diverged across py/ts.')
