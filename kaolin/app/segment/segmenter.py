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
"""``GaussianSplatSegmenter``: concrete ``SceneSegmenter`` over a kaolin ``GaussianSplatModel``.

A NEW class used by the 3D tracking pipeline (``TrackingSession`` / ``GsplatSegmentationTracker``).
It is parallel to — and does not replace — the app's ``CloudAbstraction`` / ``GaussianSplatInput``
(``input_abstraction.py``), which the live ``InteractiveCloudSelector`` still uses unchanged.
"""
from __future__ import annotations

from typing import Optional

import torch

from kaolin.rep import GaussianSplatModel

from . import _gs_renderer
from ._geometry import get_3d_center_from_mask_and_depth
from .base import SceneSegmenter


class GaussianSplatSegmenter(SceneSegmenter):
    """Segmenter backed by a kaolin ``GaussianSplatModel`` rendered via gsplat.

    Parameters
    ----------
    gsmodel : GaussianSplatModel
        Activated attributes (opacities in [0,1], positive scales, unit orientations).
    background : (3,) float tensor or None -- RGB background, defaults to black.
    """

    def __init__(
        self,
        gsmodel: GaussianSplatModel,
        background: Optional[torch.Tensor] = None,
    ):
        self.scene = gsmodel
        self.gsmodel = gsmodel
        device = gsmodel.positions.device
        self._background = (
            background if background is not None
            else torch.zeros(3, device=device, dtype=torch.float32)
        )

    # ---------------- properties ----------------

    @property
    def num_primitives(self) -> int:
        return len(self.gsmodel)

    @property
    def device(self) -> torch.device:
        return self.gsmodel.positions.device

    # ---------------- rendering ----------------

    def render(
        self,
        kal_camera,
        mask_3d: Optional[torch.Tensor] = None,
        override_color: Optional[torch.Tensor] = None,
    ) -> dict:
        """Render via gsplat. When ``override_color`` is given (``optimize_segmentation``),
        the scene tensors are detached so gradients flow only through ``override_color``."""
        target = self.gsmodel if mask_3d is None else self.gsmodel[mask_3d]
        if override_color is not None:
            target = target.detach()
        return _gs_renderer.render(
            kal_camera, target,
            background=self._background,
            override_color=override_color,
        )

    # ---------------- 3D-point lookup ----------------

    def get_3d_point_from_mask_2d(self, mask_2d: torch.Tensor, kal_camera) -> torch.Tensor:
        with torch.no_grad():
            out = self.render(kal_camera)
        return get_3d_center_from_mask_and_depth(mask_2d, out['depth'], kal_camera)

    # ---------------- naive 2D->3D (frustum bbox) ----------------

    def project_2d_mask_to_3d(self, mask_2d: torch.Tensor, kal_camera, bbox_padding: int = 0) -> torch.Tensor:
        y, x = torch.where(mask_2d)
        height, width = mask_2d.shape
        x0 = x.min().cpu() - bbox_padding
        y0 = y.min().cpu() - bbox_padding
        x1 = x.max().cpu() + bbox_padding
        y1 = y.max().cpu() + bbox_padding
        ndc_x0 = (x0.float() * 2 / width) - 1.
        ndc_y0 = (y0.float() * 2 / height) - 1.
        ndc_x1 = (x1.float() * 2 / width) - 1.
        ndc_y1 = (y1.float() * 2 / height) - 1.
        projected = _gs_renderer.project_gaussian_means_to_2d(self.gsmodel, kal_camera)  # (N, 2)
        mask_3d = (
            (projected[:, 0] >= ndc_x0) & (projected[:, 0] <= ndc_x1) &
            (projected[:, 1] >= ndc_y0) & (projected[:, 1] <= ndc_y1)
        )
        return mask_3d

    # ---------------- segmentation state ----------------

    def subscene_from_3d_mask(self, mask_3d: torch.Tensor) -> "GaussianSplatSegmenter":
        return GaussianSplatSegmenter(
            self.gsmodel[mask_3d],
            background=self._background,
        )
