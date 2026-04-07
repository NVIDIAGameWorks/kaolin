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

from __future__ import annotations

import warnings
import torch
from .camera import Camera
from .intrinsics import CameraFOV

__all__ = [
    'kaolin_camera_to_gsplat_inria',
    'gsplat_inria_camera_to_kaolin',
    'kaolin_camera_to_gsplats',
    'gsplats_camera_to_kaolin',
]

def kaolin_camera_to_gsplats(kal_camera, gs_cam_cls):
    """Deprecated function name for INRIA camera conversion.

    * for INRIA Gaussian Splats codebase, use: :py:func:`~kaolin.render.camera.kaolin_camera_to_gsplat_inria`
    * for NerfStudio gsplat package, use: :py:func:`~kaolin.render.camera.kaolin_camera_to_gsplat_nerfstudio`
    """
    warnings.warn(
        "kaolin_camera_to_gsplats has been renamed kaolin_camera_to_gsplat_inria",
        DeprecationWarning, stacklevel=2)
    return kaolin_camera_to_gsplat_inria(kal_camera, gs_cam_cls)


def gsplats_camera_to_kaolin(gs_camera):
    """Deprecated function name for INRIA camera conversion.

    Use instead :py:func:`~kaolin.render.camera.gsplat_inria_camera_to_kaolin`
    """
    warnings.warn(
        "gsplats_camera_to_kaolin has been renamed gsplat_inria_camera_to_kaolin",
        DeprecationWarning, stacklevel=2)
    return gsplat_inria_camera_to_kaolin(gs_camera)


def kaolin_camera_to_gsplat_inria(kal_camera: Camera, gs_cam_cls):
    """Converts Kaolin :ref:`Camera <kaolin.render.camera.Camera>` to `INRIA gaussian splats`_ camera (``gsplats.scene.cameras.Camera``).

    .. note::
       This has been tested with the version commit `472689c`_

    Args:
        kal_camera (Camera): camera to convert.
        gs_cam_cls (class):
            This is the gsplats ``Camera`` class,
            usually located in gsplats/scene/cameras.py.

    Returns:
        (gsplats.scene.cameras.Camera): converted INRIA gaussian splats camera.

    .. _INRIA gaussian splats:
        https://github.com/graphdeco-inria/gaussian-splatting
    .. _472689c:
        https://github.com/graphdeco-inria/gaussian-splatting/tree/472689c0dc70417448fb451bf529ae532d32c095
    """
    R = kal_camera.extrinsics.R[0].clone()
    R[1:3] = -R[1:3]
    T = kal_camera.extrinsics.t.squeeze()
    T[1:3] = -T[1:3]
    return gs_cam_cls(colmap_id=0,
                      R=R.transpose(1, 0).cpu().numpy(),
                      T=T.cpu().numpy(),
                      FoVx=kal_camera.fov(CameraFOV.HORIZONTAL, in_degrees=False),
                      FoVy=kal_camera.fov(CameraFOV.VERTICAL, in_degrees=False),
                      image=torch.zeros((3, kal_camera.height, kal_camera.width)),  # fake
                      gt_alpha_mask=None,
                      image_name='fake',
                      uid=0)


def gsplat_inria_camera_to_kaolin(gs_camera) -> Camera:
    """Convert `INRIA gaussian splats`_ camera (``gsplats.scene.cameras.Camera``) to Kaolin :ref:`Camera <kaolin.render.camera.Camera>`.

    .. note::
       This has been tested with the version commit `472689c`_

    Args:
        gs_camera (gsplats.scene.cameras.Camera): camera to convert.

    Returns:
        (Camera): converted Kaolin camera.

    .. _INRIA gaussian splats:
        https://github.com/graphdeco-inria/gaussian-splatting
    .. _472689c:
        https://github.com/graphdeco-inria/gaussian-splatting/tree/472689c0dc70417448fb451bf529ae532d32c095
    """
    view_mat = gs_camera.world_view_transform.transpose(1, 0).clone()
    view_mat[1:3] = -view_mat[1:3]
    res = Camera.from_args(
        view_matrix=view_mat,
        width=gs_camera.image_width, height=gs_camera.image_height,
        fov=gs_camera.FoVy, device='cpu')
    return res
