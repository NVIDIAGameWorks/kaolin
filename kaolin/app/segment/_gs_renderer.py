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
"""Kaolin-camera adapter around the gsplat renderer (``pip install gsplat``).

Returns a dict ``{'render': (3, H, W) float, 'depth': (H, W) float, 'visibility_filter': (N,) bool}``.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def render(
    kal_camera,
    gaussians,                                    # kaolin.rep.GaussianSplatModel
    background: Optional[torch.Tensor] = None,
    override_color: Optional[torch.Tensor] = None,
) -> dict:
    """Render a ``GaussianSplatModel`` with gsplat.

    Parameters
    ----------
    kal_camera : kaolin.render.camera.Camera
    gaussians : kaolin.rep.GaussianSplatModel (activated attributes)
    background : (3,) float tensor or None -- composited as ``rgb + bg * (1 - alpha)``.
    override_color : (N, 3) float tensor or None -- direct per-Gaussian RGB (skips SH);
        must be grad-enabled for ``optimize_segmentation``.

    Returns
    -------
    dict: 'render' (3, H, W) clamped [0,1], 'depth' (H, W), 'visibility_filter' (N,) bool.
    """
    import gsplat.rendering  # lazy -- not in all envs

    import kaolin.render.camera as kal_cam
    cam_params = kal_cam.kaolin_camera_to_gsplat_nerfstudio(kal_camera)

    if override_color is not None:
        colors_input = override_color
        sh_degree_arg = None
    else:
        colors_input = gaussians.sh_coeff
        sh_degree_arg = gaussians.sh_degree

    render_colors, render_alphas, info = gsplat.rendering.rasterization(
        gaussians.positions,
        gaussians.orientations,
        gaussians.scales,
        gaussians.opacities,
        colors_input,
        sh_degree=sh_degree_arg,
        render_mode="RGB+D",
        packed=False,   # dense per-Gaussian radii so visibility_filter is (N,), not a packed subset
        **cam_params,
    )
    device = gaussians.positions.device
    rgb = render_colors[0, :, :, :3]        # (H, W, 3)
    depth = render_colors[0, :, :, 3]       # (H, W), positive (gsplat +Z-forward)

    if background is not None:
        alpha = render_alphas[0, :, :, 0]
        bg = background.to(device=device, dtype=rgb.dtype)
        rgb = rgb + (1.0 - alpha.unsqueeze(-1)) * bg

    rgb = rgb.clamp(0.0, 1.0).permute(2, 0, 1)   # (3, H, W)

    # info['radii'] layout varies across gsplat versions: (C, N, 2), (C, N), (N, 2) or (N,).
    # Collapse the trailing per-axis radius dim (size 2) and any camera dim down to (N,).
    radii = info['radii']
    if radii.shape[-1] == 2:
        radii = radii.amax(dim=-1)
    if radii.dim() == 2:
        radii = radii[0]
    visibility_filter = radii > 0                  # (N,) bool

    return {
        'render': rgb,
        'depth': depth,
        'visibility_filter': visibility_filter,
    }


def project_gaussian_means_to_2d(gaussians, kal_camera) -> torch.Tensor:
    """Project Gaussian centers to image NDC. Returns (N, 2) ``[x_ndc, y_ndc]`` in [-1, 1];
    Gaussians behind the camera get sentinel 2.0 to exclude them from bbox tests."""
    device = gaussians.positions.device
    kc = kal_camera.to(device)

    pos_cam = kc.extrinsics.transform(gaussians.positions).squeeze(0)  # (N, 3)
    depth = -pos_cam[:, 2]
    behind = depth <= 0.0

    fx = float(kc.intrinsics.focal_x)
    fy = float(kc.intrinsics.focal_y)
    x0 = float(kc.intrinsics.x0)
    y0 = float(kc.intrinsics.y0)
    width = float(kc.width)
    height = float(kc.height)

    safe_depth = depth.clamp(min=1e-6)
    pixel_x = fx * pos_cam[:, 0] / safe_depth + (width / 2.0 + x0)
    pixel_y = fy * (-pos_cam[:, 1]) / safe_depth + (height / 2.0 + y0)

    ndc_x = pixel_x * 2.0 / width - 1.0
    ndc_y = pixel_y * 2.0 / height - 1.0

    ndc = torch.stack([ndc_x, ndc_y], dim=-1)
    ndc[behind] = 2.0
    return ndc
