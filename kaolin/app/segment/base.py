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
"""Representation-agnostic segmenter ABC used by the SAM/Cutie 3D tracking pipeline.

A ``SceneSegmenter`` owns a scene + its (current) 3D segmentation, renders it from a kaolin
camera with an optional per-primitive mask, and provides:
  - ``project_2d_mask_to_3d`` — naive 2D->3D (frustum bbox), instantiates the segmented scene.
  - ``get_3d_point_from_mask_2d`` — single XYZ point (e.g. for a turnaround ``at``).
  - ``optimize_segmentation`` — per-primitive logit fit, driven by ``render(override_color=...)``.

This is a NEW parallel abstraction; the app's existing ``CloudAbstraction`` /
``GaussianSplatInput`` (``input_abstraction.py``) are left untouched. The concrete
``GaussianSplatSegmenter`` lives in ``segmenter.py``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm


class SceneSegmenter(ABC):
    """Holds a scene and (optionally) a derived 3D segmentation of it.

    Attributes
    ----------
    scene : Any
        The underlying 3D representation (e.g. a kaolin ``GaussianSplatModel``).
    """

    scene: Any

    # ---------------- representation-specific abstracts ----------------

    @property
    @abstractmethod
    def num_primitives(self) -> int:
        ...

    @property
    @abstractmethod
    def device(self) -> torch.device:
        ...

    @abstractmethod
    def render(self, kal_camera, mask_3d: Optional[torch.Tensor] = None,
               override_color: Optional[torch.Tensor] = None) -> dict:
        """Render the scene from ``kal_camera``.

        Parameters
        ----------
        kal_camera : kaolin.render.camera.Camera
        mask_3d : (N,) bool tensor or None
            If None, render the full scene. If provided, render only primitives where True.
        override_color : (K, 3) tensor or None
            Per-primitive RGB override. ``K`` equals ``num_primitives`` if ``mask_3d`` is
            None, else ``mask_3d.sum()``. Used by ``optimize_segmentation`` to propagate
            gradients into per-primitive logits.

        Returns
        -------
        dict with at least ``'render'`` (3, H, W) and ``'depth'`` (H, W). Subclasses may
        include extras (e.g. gsplats: ``'visibility_filter'``).
        """

    @abstractmethod
    def get_3d_point_from_mask_2d(self, mask_2d: torch.Tensor, kal_camera) -> torch.Tensor:
        """Return a single (3,) world-space XYZ for a (mask, camera) pair."""

    @abstractmethod
    def project_2d_mask_to_3d(self, mask_2d: torch.Tensor, kal_camera,
                              bbox_padding: int = 0) -> torch.Tensor:
        """Naive 2D->3D projection. Returns the per-primitive (N,) bool mask."""

    @abstractmethod
    def subscene_from_3d_mask(self, mask_3d: torch.Tensor) -> "SceneSegmenter":
        """Return a NEW segmenter scoped to the masked subset (with a ``parent_mask`` attr)."""

    # ---------------- default (representation-agnostic) implementation ----------------

    def optimize_segmentation(
            self,
            masks_2d: Sequence[torch.Tensor],
            kal_cameras: Sequence,
            ref_masks_2d: Optional[Sequence[torch.Tensor]] = None,
            ref_kal_cameras: Optional[Sequence] = None,
            lr: float = 1e-1,
            threshold: float = 0.5,
            progress: bool = True,
    ) -> dict:
        """Per-primitive logit optimization, driven by ``self.render(override_color=...)``.

        Builds a ``(num_primitives, 1)`` logit tensor at 0.48, renders it as RGB through every
        (camera, mask_2d) pair (refs first, then tracking), backprops L1 to each 2D mask, then
        takes one Adam step. Thresholding ``> threshold`` yields the 3D mask.

        Returns ``{'mask_3d': (N,) bool, 'logits': (N,) float}``.
        """
        n = self.num_primitives
        device = self.device
        logits = torch.full((n, 1), 0.48, device=device, dtype=torch.float, requires_grad=True)
        optimizer = torch.optim.Adam([logits], lr=lr, eps=1e-15)
        optimizer.zero_grad()
        with torch.enable_grad():
            pairs = []
            if ref_kal_cameras is not None:
                pairs.extend(zip(ref_kal_cameras, ref_masks_2d))
            pairs.extend(zip(kal_cameras, masks_2d))
            iterator = tqdm(pairs, desc='optimize segmentation') if progress else pairs
            for cam, gt_mask_2d in iterator:
                out = self.render(cam, override_color=logits.repeat(1, 3))
                rendered = out['render']
                gt = gt_mask_2d.unsqueeze(0)
                if gt.shape[-2:] != rendered.shape[-2:]:
                    gt = F.interpolate(gt.unsqueeze(0), size=rendered.shape[-2:],
                                       mode='bilinear', align_corners=False).squeeze(0)
                loss = torch.mean(torch.abs(rendered - gt))
                loss.backward()
            optimizer.step()

        mask_3d = (logits.squeeze(-1) > threshold).detach()
        return {'mask_3d': mask_3d, 'logits': logits.squeeze(-1).detach()}
