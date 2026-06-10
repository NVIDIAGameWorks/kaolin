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
"""3D segmentation by Cutie video-mask tracking + per-primitive logit optimization.

Public surface
--------------
- ``TrackingSession`` — holds 2D tracking state (ref images/masks/cameras + Cutie processor)
  and drives a ``SceneSegmenter`` to produce a 3D segmentation.
- ``GsplatSegmentationTracker`` — stateful facade with the interface that
  ``InteractiveCloudSelector`` expects, backed by ``TrackingSession`` + ``GaussianSplatSegmenter``.
- ``get_cutie_processor`` / ``track_mask`` / ``most_similar_view`` / ``base_track_masks`` —
  representation-free Cutie helpers.

Cutie (https://github.com/hkchengrex/Cutie) is imported lazily inside ``get_cutie_processor``
so this module imports without it; you only need it to actually build a processor / run tracking.
"""
import os
import math
import hashlib
import logging

import requests
from tqdm import tqdm
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .base import SceneSegmenter
from ._geometry import get_3d_center_from_mask_and_depth
from .views import rotate_camera_around_at

logger = logging.getLogger(__name__)


_cutie_links = {
    'base': ('https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth',
             'a6071de6136982e396851903ab4c083a'),
    'small': ('https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-small-mega.pth',
              '6abf53b4058228babc3686274bc136cc'),
}
SEGMENT_DIR = os.path.dirname(os.path.realpath(__file__))
WEIGHTS_DIR = os.path.join(SEGMENT_DIR, 'weights')


# ============================================================================
# TrackerState: per-user reference data store (no algorithm)
# ============================================================================

class TrackerState:
    """Stores reference (image, mask, camera) pairs added by the user.

    This is a pure data container — it has no knowledge of the tracking algorithm.
    Pass it to ``TrackingSession.from_tracker_state`` to run Cutie + logit optimization.
    """

    def __init__(self):
        self._ref_images = []
        self._ref_masks = []
        self._ref_cams = []
        self._ref_has_occlusions = []

    def add_ground_truth_mask(self, image, mask, cam, has_occlusions=False):
        self._ref_images.append(image)
        self._ref_masks.append(mask)
        self._ref_cams.append(cam)
        self._ref_has_occlusions.append(has_occlusions)

    def reset_ground_truth_masks(self):
        self._ref_images.clear()
        self._ref_masks.clear()
        self._ref_cams.clear()
        self._ref_has_occlusions.clear()

    def can_track(self):
        return len(self._ref_images) > 0

    @property
    def num_reference_masks(self):
        return len(self._ref_images)

    @property
    def ref_image(self):
        return self._ref_images[0]

    @property
    def ref_mask(self):
        return self._ref_masks[0]

    @property
    def ref_cam(self):
        return self._ref_cams[0]

    @property
    def ref_cameras(self):
        return self._ref_cams


# ============================================================================
# Cutie helpers (representation-free)
# ============================================================================

def download_model_if_needed(model, weights_dir=WEIGHTS_DIR):
    """Download a Cutie checkpoint into ``weights_dir`` if missing or md5 mismatched."""
    os.makedirs(weights_dir, exist_ok=True)
    if model not in _cutie_links:
        raise ValueError(f"'{model}' not recognized model, should be ['base', 'small'].")
    link, md5 = _cutie_links[model]
    filename = link.split('/')[-1]
    full_fname = os.path.join(weights_dir, filename)
    if not os.path.exists(full_fname) or hashlib.md5(open(full_fname, 'rb').read()).hexdigest() != md5:
        print(f'Downloading {filename}...')
        r = requests.get(link, stream=True)
        total_size = int(r.headers.get('content-length', 0))
        block_size = 1024
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(full_fname, 'wb') as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        if total_size != 0 and t.n != total_size:
            raise RuntimeError('Error while downloading %s' % filename)


def _cutie_config_dir():
    """Locate the config dir shipped inside the installed ``cutie`` package.

    Uses ``__path__`` rather than ``__file__`` because Cutie installs as a namespace
    package (``cutie.__file__`` is ``None``).
    """
    import cutie
    return os.path.join(list(cutie.__path__)[0], 'config')


def get_cutie_processor(model='base', device='cuda', mem_every=5, use_long_term=False):
    """Build a Cutie ``InferenceCore``. Downloads the checkpoint if needed.

    Cutie + hydra are imported here (lazily) so importing this module doesn't require them.
    """
    from cutie.model.cutie import CUTIE
    from cutie.inference.inference_core import InferenceCore
    from hydra import compose, initialize_config_dir

    download_model_if_needed(model)
    with initialize_config_dir(version_base=None, config_dir=_cutie_config_dir()):
        weights_path = os.path.join(WEIGHTS_DIR, f'cutie-{model}-mega.pth')
        cfg = compose(config_name='eval_config.yaml', overrides=[
            f'model={model}',
            f'weights={weights_path}',
            'dataset=generic',
            f'mem_every={mem_every}',
            f'use_long_term={str(use_long_term)}',
        ])
    cutie = CUTIE(cfg=cfg).to(device).eval()
    model_weights = torch.load(cfg.weights)
    cutie.load_weights(model_weights)
    return InferenceCore(cutie, cfg=cfg)


@torch.inference_mode()
@torch.amp.autocast('cuda')
def track_mask(processor, images_2d, custom_masks_2d):
    """Track a 2D segmentation across an image sequence.

    Args:
        processor: Cutie ``InferenceCore``.
        images_2d (list[Tensor]): each (3, H, W) cuda float.
        custom_masks_2d (dict[int, Tensor] | list[Tensor]): seed masks; usually ``{0: mask_2d}``.

    Returns:
        list[Tensor]: per-image (H, W) byte masks.
    """
    out_masks_2d = []
    with torch.no_grad():
        for i, img in tqdm(enumerate(images_2d), desc='tracking segmentation over images',
                           total=len(images_2d)):
            if i in custom_masks_2d:
                mask_2d = custom_masks_2d[i]
                valid_labels = [1]
            else:
                mask_2d = None
                valid_labels = None
            prob = processor.step(img, mask_2d, valid_labels, end=(i == (len(images_2d) - 1)))
            out_masks_2d.append(torch.argmax(prob, dim=0).byte())
        processor.clear_memory()
    return out_masks_2d


def most_similar_view(target_visibility, view_visibilities):
    """Index of the view whose visibility filter has the largest Jaccard overlap with ``target_visibility``."""
    best_i, best_val = -1, 0
    for i, visibility in enumerate(view_visibilities):
        num = torch.sum(torch.logical_and(visibility, target_visibility)).float()
        den = torch.sum(torch.logical_or(visibility, target_visibility)).float()
        val = num / den
        if val > best_val:
            best_val, best_i = val, i
    return best_i, best_val


def base_track_masks(processor, input_views, ref_images_2d, ref_masks_2d,
                     most_similar_views=None, return_debug=False):
    """Run Cutie forward then backward over ``input_views`` injecting reference masks at their best-matching views."""
    num_views = len(input_views)
    num_references = len(ref_images_2d)

    # Cutie stores pixel-memory features at the spatial resolution of the first frame it sees.
    # Every frame (refs AND tracking views) must share the same H×W, otherwise sensory/memory
    # tensors from one frame can't be added to feature tensors from another.
    # Also, the image_feature_store caches by curr_ti which resets between the forward and
    # backward passes, so stale cached features from a different image would be reused.
    # Fix: normalise everything to a single target H×W and flush the cache before each pass.
    target_h, target_w = ref_images_2d[0].shape[-2:]

    def _resize(img, mask=None):
        if img.shape[-2:] != (target_h, target_w):
            img = F.interpolate(img.unsqueeze(0), size=(target_h, target_w),
                                mode='bilinear', align_corners=False).squeeze(0)
        if mask is not None and mask.shape[-2:] != (target_h, target_w):
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                                 size=(target_h, target_w),
                                 mode='nearest').squeeze(0).squeeze(0)
            return img, mask
        return img

    input_views = [_resize(v) for v in input_views]
    ref_images_2d = [_resize(r) for r in ref_images_2d]
    ref_masks_2d = [
        F.interpolate(m.unsqueeze(0).unsqueeze(0).float(), size=(target_h, target_w),
                      mode='nearest').squeeze(0).squeeze(0)
        if m.shape[-2:] != (target_h, target_w) else m
        for m in ref_masks_2d
    ]

    out_masks_2d = [None] * num_views

    if most_similar_views is None:
        most_similar_views = [0 for _ in range(num_references)]

    def _flush(proc):
        """Full processor reset: clear memory AND the image-feature cache.

        clear_memory() resets curr_ti to -1, so the next pass reuses index 0, 1, …
        The ImageFeatureStore caches by index and is NOT cleared by clear_memory(),
        so stale features from a previous pass would be returned for those indices.
        """
        proc.clear_memory()
        proc.image_feature_store._store.clear()

    with torch.inference_mode():
        with torch.amp.autocast('cuda'):
            references = list(zip(most_similar_views, [i for i in range(num_references)]))
            references.sort()

            start_vidx = references[0][0]
            ref_idx = 0
            debug_steps = []
            _flush(processor)
            for vidx in range(start_vidx, num_views):
                while ref_idx < num_references and references[ref_idx][0] == vidx:
                    ridx = references[ref_idx][1]
                    _ = processor.step(ref_images_2d[ridx], ref_masks_2d[ridx], [1], end=False, force_permanent=True)
                    ref_idx += 1
                    debug_steps.append(f'Add reference view {ridx}')

                prob = processor.step(input_views[vidx], None, None, end=False)
                out_masks_2d[vidx] = torch.argmax(prob, dim=0).float()
                debug_steps.append(f'Predict mask for view {vidx}')
            _flush(processor)

            ref_idx = 0
            while ref_idx < num_references and references[ref_idx][0] == start_vidx:
                ridx = references[ref_idx][1]
                _ = processor.step(ref_images_2d[ridx], ref_masks_2d[ridx], [1], end=False, force_permanent=True)
                ref_idx += 1
                debug_steps.append(f'Add reference view {ridx}')
            for vidx in range(start_vidx, -1, -1):
                prob = processor.step(input_views[vidx], None, None, end=False)
                out_masks_2d[vidx] = torch.argmax(prob, dim=0).float()
                debug_steps.append(f'Predict mask for view {vidx}')
            _flush(processor)

    if return_debug:
        return out_masks_2d, debug_steps
    return out_masks_2d


def _zero_outside_bbox(image_2d, mask_2d, padding=0):
    """In-place: zero ``image_2d`` (3, H, W) outside the bbox of ``mask_2d`` (H, W)."""
    y, x = torch.where(mask_2d)
    if y.numel() == 0:
        return
    height, width = mask_2d.shape
    x0 = int(max(0, x.min().item() - padding))
    y0 = int(max(0, y.min().item() - padding))
    x1 = int(min(width, x.max().item() + padding))
    y1 = int(min(height, y.max().item() + padding))
    image_2d[:, :y0, :] = 0.
    image_2d[:, y1:, :] = 0.
    image_2d[:, :, :x0] = 0.
    image_2d[:, :, x1:] = 0.


# ============================================================================
# TrackingSession: 2D state + segmenter orchestration
# ============================================================================

class TrackingSession:
    """Holds 2D tracking state (ref images/masks/cameras + Cutie processor), independent of
    the 3D representation.

    Usage::

        session = TrackingSession()
        session.add_reference(image_2d, mask_2d, kal_camera)
        out = session.track_and_segment(segmenter, tracking_kal_cameras)
    """

    def __init__(self, processor=None, mem_every: int = 10, device: str = 'cuda'):
        self.processor = processor if processor is not None else get_cutie_processor(
            'base', device=device, mem_every=mem_every)
        self.device = device
        self.ref_images_2d: list = []
        self.ref_masks_2d: list = []
        self.ref_kal_cameras: list = []

    @classmethod
    def from_tracker_state(cls, tracker_state: TrackerState, device: str = 'cuda') -> 'TrackingSession':
        """Build a session pre-loaded with the references stored in ``tracker_state``."""
        session = cls(device=device)
        for img, mask, cam in zip(tracker_state._ref_images, tracker_state._ref_masks, tracker_state._ref_cams):
            session.add_reference(img, mask, cam)
        return session

    def add_reference(self, image_2d: torch.Tensor, mask_2d: torch.Tensor, kal_camera) -> None:
        self.ref_images_2d.append(image_2d)
        self.ref_masks_2d.append(mask_2d)
        self.ref_kal_cameras.append(kal_camera)

    def reset_references(self) -> None:
        self.ref_images_2d.clear()
        self.ref_masks_2d.clear()
        self.ref_kal_cameras.clear()

    @property
    def num_references(self) -> int:
        return len(self.ref_images_2d)

    def save_references(self, path: str) -> None:
        torch.save({
            'ref_images_2d': self.ref_images_2d,
            'ref_masks_2d': self.ref_masks_2d,
            'ref_kal_cameras': self.ref_kal_cameras,
        }, path)

    def load_references(self, path: str) -> None:
        data = torch.load(path, weights_only=False)
        self.ref_images_2d = [x.to(self.device) for x in data['ref_images_2d']]
        self.ref_masks_2d = [x.to(self.device) for x in data['ref_masks_2d']]
        self.ref_kal_cameras = data['ref_kal_cameras']

    def track_and_segment(
            self,
            segmenter: SceneSegmenter,
            tracking_kal_cameras: Sequence,
            refine: bool = False,
            presegment_bbox: bool = True,
            bbox_padding: int = 20,
    ) -> dict:
        """Run Cutie tracking over ``tracking_kal_cameras``, then optimize per-primitive logits.

        Returns ``{'mask_3d', 'logits', 'masks_2d', 'outputs'}`` with legacy aliases
        ``'3d_mask'`` / ``'3d_logits'`` / ``'2d_masks'``.
        """
        if self.num_references == 0:
            raise RuntimeError("TrackingSession has no references. Call add_reference(...) first.")

        ref_images_2d = [img.clone().detach() for img in self.ref_images_2d]
        ref_masks_2d = list(self.ref_masks_2d)
        ref_kal_cameras = list(self.ref_kal_cameras)

        # ---- step 1: optional bbox presegmentation ----
        active_segmenter = segmenter
        parent_mask = None
        if presegment_bbox:
            with torch.no_grad():
                bbox_3d_mask = torch.ones(
                    segmenter.num_primitives, dtype=torch.bool, device=segmenter.device)
                for i, (ref_mask_2d, ref_cam) in enumerate(zip(ref_masks_2d, ref_kal_cameras)):
                    bbox_3d_mask &= segmenter.project_2d_mask_to_3d(
                        ref_mask_2d, ref_cam, bbox_padding=bbox_padding)
                    _zero_outside_bbox(ref_images_2d[i], ref_mask_2d, padding=bbox_padding)
                active_segmenter = segmenter.subscene_from_3d_mask(bbox_3d_mask)
                parent_mask = bbox_3d_mask

        # ---- step 2: render all tracking views (and refs, for visibility matching) ----
        with torch.inference_mode():
            tracking_outputs = [active_segmenter.render(cam) for cam in tqdm(
                tracking_kal_cameras, desc='render tracking views')]
            tracking_renders = [o['render'] for o in tracking_outputs]
            tracking_visibilities = [o['visibility_filter'] for o in tracking_outputs]

            ref_outputs = [active_segmenter.render(cam) for cam in tqdm(
                ref_kal_cameras, desc='render reference views')]
            ref_visibilities = [o['visibility_filter'] for o in ref_outputs]

            similar_views = [most_similar_view(rv, tracking_visibilities)[0] for rv in ref_visibilities]
            logger.debug(f'Most similar views to references: {similar_views}')

            out_masks_2d, debug_steps = base_track_masks(
                self.processor, tracking_renders, ref_images_2d, ref_masks_2d,
                most_similar_views=similar_views, return_debug=True)
            for s in debug_steps:
                logger.debug(s)

        # ---- step 3: per-primitive logit optimization ----
        opt_result = active_segmenter.optimize_segmentation(
            out_masks_2d, tracking_kal_cameras,
            ref_masks_2d=ref_masks_2d, ref_kal_cameras=ref_kal_cameras,
        )

        # ---- step 4: map back to parent scene if we descended ----
        mask_3d, logits_3d = opt_result['mask_3d'], opt_result['logits']
        if parent_mask is not None:
            full_mask_3d = torch.zeros_like(parent_mask)
            full_mask_3d[parent_mask] = mask_3d
            full_logits_3d = torch.zeros(parent_mask.shape, device=segmenter.device, dtype=torch.float)
            full_logits_3d[parent_mask] = logits_3d
            mask_3d, logits_3d = full_mask_3d, full_logits_3d

        # ---- step 5: optional refine ----
        if refine:
            refine_segmenter = segmenter.subscene_from_3d_mask(mask_3d)
            refine_result = refine_segmenter.optimize_segmentation(
                out_masks_2d, tracking_kal_cameras,
                ref_masks_2d=ref_masks_2d, ref_kal_cameras=ref_kal_cameras,
            )
            full_mask_3d = torch.zeros_like(mask_3d)
            full_mask_3d[mask_3d] = refine_result['mask_3d']
            mask_3d = full_mask_3d
            logits_3d = None  # not meaningfully expandable across refine (matches legacy behavior)

        return {
            'mask_3d': mask_3d,
            'logits': logits_3d,
            'masks_2d': out_masks_2d,
            'outputs': tracking_outputs,
            '3d_mask': mask_3d,
            '3d_logits': logits_3d,
            '2d_masks': out_masks_2d,
        }

    def track_and_segment2(
            self,
            segmenter: SceneSegmenter,
            refine: bool = False,
            presegment_bbox: bool = True,
            bbox_padding: int = 20,
            num_camera_per_arc: int = 4,
    ) -> dict:
        """Adaptive arc-orbit variant of ``track_and_segment`` (single reference only).

        Orbits from the reference camera in two elevation=0 arcs, recomputing the pivot from
        the current rendered depth + Cutie's predicted mask. Same dict shape as ``track_and_segment``.
        """
        if self.num_references != 1:
            raise RuntimeError(
                f"track_and_segment2 requires exactly one reference, got {self.num_references}")

        ref_image_2d = self.ref_images_2d[0].clone().detach()
        ref_mask_2d = self.ref_masks_2d[0]
        ref_camera = self.ref_kal_cameras[0]

        active_segmenter = segmenter
        parent_mask = None
        if presegment_bbox:
            with torch.no_grad():
                bbox_3d_mask = segmenter.project_2d_mask_to_3d(
                    ref_mask_2d, ref_camera, bbox_padding=bbox_padding)
                _zero_outside_bbox(ref_image_2d, ref_mask_2d, padding=bbox_padding)
                active_segmenter = segmenter.subscene_from_3d_mask(bbox_3d_mask)
                parent_mask = bbox_3d_mask

        ref_up_axis = ref_camera.cam_up().reshape(-1)
        tracking_outputs: list = []
        tracking_masks_2d: list = []
        tracking_cameras: list = []

        for direction in (+1, -1):
            self.processor.clear_memory()
            with torch.inference_mode():
                with torch.amp.autocast('cuda'):
                    self.processor.step(ref_image_2d, ref_mask_2d, [1], end=False, force_permanent=True)
                current_cam = ref_camera
                for iteration in tqdm(range(num_camera_per_arc + 1),
                                      desc=f'arc {"+" if direction > 0 else "-"}'):
                    out = active_segmenter.render(current_cam)
                    rgb = out['render']
                    depth = out['depth']

                    with torch.amp.autocast('cuda'):
                        prob = self.processor.step(rgb, None, None, end=(iteration == num_camera_per_arc))
                    mask_2d = torch.argmax(prob, dim=0).float()

                    tracking_outputs.append(out)
                    tracking_masks_2d.append(mask_2d)
                    tracking_cameras.append(current_cam)

                    if iteration < num_camera_per_arc:
                        try:
                            pivot = get_3d_center_from_mask_and_depth(mask_2d, depth, current_cam)
                        except ValueError:
                            pivot = segmenter.get_3d_point_from_mask_2d(ref_mask_2d, ref_camera)
                        azimuth = direction * math.pi * (iteration + 1) / num_camera_per_arc
                        current_cam = rotate_camera_around_at(
                            ref_camera, at=pivot, up_axis=ref_up_axis, elevation=0.0, azimuth=azimuth)

        self.processor.clear_memory()

        opt_result = active_segmenter.optimize_segmentation(
            tracking_masks_2d, tracking_cameras,
            ref_masks_2d=[ref_mask_2d], ref_kal_cameras=[ref_camera],
        )

        mask_3d, logits_3d = opt_result['mask_3d'], opt_result['logits']
        if parent_mask is not None:
            full_mask_3d = torch.zeros_like(parent_mask)
            full_mask_3d[parent_mask] = mask_3d
            full_logits_3d = torch.zeros(parent_mask.shape, device=segmenter.device, dtype=torch.float)
            full_logits_3d[parent_mask] = logits_3d
            mask_3d, logits_3d = full_mask_3d, full_logits_3d

        if refine:
            refine_segmenter = segmenter.subscene_from_3d_mask(mask_3d)
            refine_result = refine_segmenter.optimize_segmentation(
                tracking_masks_2d, tracking_cameras,
                ref_masks_2d=[ref_mask_2d], ref_kal_cameras=[ref_camera],
            )
            full_mask_3d = torch.zeros_like(mask_3d)
            full_mask_3d[mask_3d] = refine_result['mask_3d']
            mask_3d = full_mask_3d
            logits_3d = None

        return {
            'mask_3d': mask_3d,
            'logits': logits_3d,
            'masks_2d': tracking_masks_2d,
            'outputs': tracking_outputs,
            '3d_mask': mask_3d,
            '3d_logits': logits_3d,
            '2d_masks': tracking_masks_2d,
        }


# ============================================================================
# GsplatSegmentationTracker: app-facing facade
# ============================================================================

def _as_segmenter(gaussians):
    """Coerce the various things callers pass into a SceneSegmenter.

    Accepts a ``SceneSegmenter`` (used directly), anything exposing ``.gsmodel`` (e.g. the
    app's ``GaussianSplatInput``), or a bare ``GaussianSplatModel``.
    """
    from .segmenter import GaussianSplatSegmenter
    if isinstance(gaussians, SceneSegmenter):
        return gaussians
    if hasattr(gaussians, 'gsmodel'):
        return GaussianSplatSegmenter(gaussians.gsmodel)
    return GaussianSplatSegmenter(gaussians)


class GsplatSegmentationTracker:
    """Stateful tracker facade backed by ``TrackingSession`` + ``GaussianSplatSegmenter``.

    Keeps the interface ``InteractiveCloudSelector`` calls (``add_ground_truth_mask`` /
    ``reset_ground_truth_masks`` / ``track_and_segment`` / ``predict_mask`` / ``ref_*`` /
    ``predicted_mask`` / ``target_view`` / ``target_cam`` / ``num_*``). The Cutie processor is
    created lazily on first use, so the tracker is constructible without Cutie installed.
    """

    def __init__(self, processor=None, mem_every: int = 10, device: str = 'cuda'):
        self._processor = processor
        self._mem_every = mem_every
        self._device = device
        self._session: Optional[TrackingSession] = None

        self._ref_images = []
        self._ref_masks = []
        self._ref_cams = []
        self._ref_has_occlusions = []

        self._result = None
        self._tracked_cameras = None

    def _ensure_session(self) -> TrackingSession:
        if self._session is None:
            self._session = TrackingSession(
                processor=self._processor, mem_every=self._mem_every, device=self._device)
            for img, m, cam in zip(self._ref_images, self._ref_masks, self._ref_cams):
                self._session.add_reference(img, m, cam)
        return self._session

    def can_track(self):
        return len(self._ref_images) > 0

    def update_rendered_views(self, gaussians):
        # Kept for API compatibility; rendering now happens inside TrackingSession.
        pass

    def reset_ground_truth_masks(self):
        self._ref_images, self._ref_masks, self._ref_cams, self._ref_has_occlusions = [], [], [], []
        if self._session is not None:
            self._session.reset_references()

    def add_ground_truth_mask(self, image, mask, gsplat_cam, has_occlusions=False):
        self._ref_images.append(image)
        self._ref_masks.append(mask)
        self._ref_cams.append(gsplat_cam)
        self._ref_has_occlusions.append(has_occlusions)
        if self._session is not None:
            self._session.add_reference(image, mask, gsplat_cam)

    @property
    def num_reference_masks(self):
        return len(self._ref_images)

    @property
    def ref_image(self):
        return self._ref_images[0]

    @property
    def ref_mask(self):
        return self._ref_masks[0]

    @property
    def ref_cam(self):
        return self._ref_cams[0]

    @property
    def ref_cameras(self):
        return self._ref_cams

    def track_and_segment(self, gaussians, gsplat_cameras, refine=False):
        """Track + segment. ``gaussians`` may be a ``SceneSegmenter``, anything with a
        ``.gsmodel`` (the app's ``GaussianSplatInput``), or a ``GaussianSplatModel``."""
        segmenter = _as_segmenter(gaussians)
        session = self._ensure_session()
        res = session.track_and_segment(segmenter, gsplat_cameras, refine=refine)
        self._result = res
        self._tracked_cameras = gsplat_cameras
        return res

    def predict_mask(self, view):
        """Single-view Cutie prediction seeded with the current references."""
        session = self._ensure_session()
        proc = session.processor
        with torch.amp.autocast('cuda'):
            for ref_img, ref_mask in zip(self._ref_images, self._ref_masks):
                proc.step(ref_img, ref_mask, [1], end=False, force_permanent=True)
            prob = proc.step(view, None, None, end=False)
            pred_mask = torch.argmax(prob, dim=0).float()
            proc.clear_memory()
        return pred_mask

    @property
    def num_predicted_masks(self):
        if self._result is None:
            return 0
        return len(self._result['2d_masks'])

    def predicted_mask(self, idx):
        if self._result is None:
            return None
        return self._result['2d_masks'][idx]

    def target_view(self, idx):
        if self._result is None:
            return None
        return self._result['outputs'][idx]['render']

    def target_cam(self, idx):
        if self._tracked_cameras is None:
            return None
        return self._tracked_cameras[idx]
