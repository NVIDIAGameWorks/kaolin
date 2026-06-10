# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import logging

import torch

import kaolin.render.camera.gsplats_nerfstudio as gs_cam
import kaolin.rep.gaussians as _grep

logger = logging.getLogger(__name__)

try:
    import gsplat.rendering
    _HAS_GSPLAT = True
except ImportError:
    _HAS_GSPLAT = False
    logger.warning('gsplat not available; rendering will return a blank frame.')


def render_objects(gaussians_list: list, camera) -> torch.Tensor:
    """Render a list of GaussianSplatModel instances as a combined scene.

    Args:
        gaussians_list: list of GaussianSplatModel (already in world space).
        camera: kaolin.render.camera.Camera.

    Returns:
        (H, W, 3) uint8 torch.Tensor on CPU, or a blank black frame on error/empty.
    """
    h, w = int(camera.height), int(camera.width)

    if not gaussians_list:
        return torch.zeros(h, w, 3, dtype=torch.uint8)

    device = gaussians_list[0].positions.device
    cam = camera.to(device)

    if not _HAS_GSPLAT:
        return torch.zeros(h, w, 3, dtype=torch.uint8)

    try:
        combined = _grep.GaussianSplatModel.cat(gaussians_list)
        gsplat_params = gs_cam.kaolin_camera_to_gsplat_nerfstudio(cam)
        render_colors, _render_alphas, _info = gsplat.rendering.rasterization(
            combined.positions,
            combined.orientations,
            combined.scales,
            combined.opacities,
            combined.sh_coeff,
            sh_degree=combined.sh_degree,
            **gsplat_params,
        )
        # render_colors: (1, H, W, 3) float [0, 1]
        return (render_colors[0].clamp(0.0, 1.0) * 255).to(torch.uint8).cpu()
    except Exception:
        logger.exception('gsplat rasterization failed')
        return torch.zeros(h, w, 3, dtype=torch.uint8)
