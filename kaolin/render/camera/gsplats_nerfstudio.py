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

from __future__ import annotations

import torch

from .camera import Camera
from .extrinsics import CameraExtrinsics
from .intrinsics_pinhole import PinholeIntrinsics


__all__ = ["kaolin_camera_to_gsplat_nerfstudio", "gsplat_nerfstudio_camera_to_kaolin"]


def kaolin_camera_to_gsplat_nerfstudio(kal_camera: Camera):
    """Convert Kaolin :ref:`Camera <kaolin.render.camera.Camera>` to `nerfstudio gsplat library`_ camera parameters,
    as expected by ``gsplat.rendering.rasterization``. Batched conversion is supported. Only Pinhole camera
    model is covered.

    .. note::
       This has been tested with the version `gsplat==1.4.0`.

    Args:
        kal_camera (Camera): camera to convert.

    Returns:
        (dict):
            A dict with the following keys:

            * ``"Ks"`` (torch.Tensor): intrinsics matrix of shape :math:`(C, 3, 3)`.
            * ``"viewmats"`` (torch.Tensor): view matrix of shape :math:`(C, 4, 4)`.
            * ``"width"`` (int): image width from the source camera.
            * ``"height"`` (int): image height from the source camera.
            * ``"camera_model"`` (str): always ``"pinhole"``.

            ``C`` is the number of cameras.

    Raises:
        RuntimeError: if ``kal_camera`` does not use a pinhole intrinsics model.

    .. _nerfstudio gsplat library:
        https://github.com/nerfstudio-project/gsplat
    """
    if kal_camera.intrinsics.__class__.registered_name() != 'pinhole':
        raise RuntimeError(f'Only pinhole camera is supported, not {kal_camera}')

    device = kal_camera.device
    dtype = kal_camera.dtype

    K = torch.zeros((len(kal_camera), 3, 3), device=device, dtype=dtype)
    K[:, 0, 0] = kal_camera.focal_x
    K[:, 1, 1] = kal_camera.focal_y
    K[:, 2, 2] = 1.0
    K[:, 0, 2] = kal_camera.width / 2.0
    K[:, 1, 2] = kal_camera.height / 2.0
    # K = torch.tensor(
    #     [
    #         [kal_camera.focal_x, 0.0, kal_camera.width / 2.0],
    #         [0.0, kal_camera.focal_y , kal_camera.height / 2.0],
    #         [0.0, 0.0, 1.0],
    #     ], device=device, dtype=dtype).unsqueeze(0).repeat(len(kal_camera), 1, 1)

    viewmat = kal_camera.extrinsics.view_matrix()
    transform_mat = torch.eye(4, device=device, dtype=dtype)
    transform_mat[1, 1] = -1
    transform_mat[2, 2] = -1
    viewmat = transform_mat @ viewmat

    return {"viewmats": viewmat, "Ks": K, "width": kal_camera.width, "height": kal_camera.height,
            "camera_model": "pinhole", "near_plane": kal_camera.near, "far_plane": kal_camera.far}


def gsplat_nerfstudio_camera_to_kaolin(Ks, viewmats,
                                       width = None,
                                       height = None,
                                       camera_model = 'pinhole',
                                       near_plane: float = 1e-2,
                                       far_plane: float = 1e2) -> Camera:
    """Convert `nerfstudio gsplat library`_ camera parameters,
    as expected by ``gsplat.rendering.rasterization``, to Kaolin :ref:`Camera <kaolin.render.camera.Camera>`.
    Batched conversion is supported.

    Args:
        Ks (torch.Tensor): `(C, 3, 3)` matrix
        viewmats (torch.Tensor): `(C, 4, 4)` matrix
        width (optional, int): if not set, will guess value from Ks
        height (optional, int): if not set, will guess value from Ks
        camera_model (optional, str): currently only pinhole is supported
        near_plane (optional, float): near clipping plane, defines the min depth of the view frustum.
        far_plane (optional, float): far clipping plane, define the max depth of the view frustum.

    Returns:
        (Camera): converted Kaolin camera.
    """
    device = viewmats.device
    dtype = viewmats.dtype

    transform_mat = torch.eye(4, device=device, dtype=dtype)
    transform_mat[1, 1] = -1
    transform_mat[2, 2] = -1
    converterd_viewmat = transform_mat @ viewmats

    # batched extrinsics
    extrinsics = CameraExtrinsics.from_view_matrix(converterd_viewmat)

    # batched intrinsics
    widths = Ks[:, 0, 2] * 2
    heights = Ks[:, 1, 2] * 2
    if width is not None:
        widths[:] = width
    if height is not None:
        heights[:] = height
    focals_x = Ks[:, 0, 0]
    focals_y = Ks[:, 1, 1]
    intrinsics = PinholeIntrinsics.cat(
        [PinholeIntrinsics.from_focal(width=int(widths[i].item()), height=int(heights[i].item()),
                                      focal_x=focals_x[i].item(), focal_y=focals_y[i].item(),
                                      near=near_plane, far=far_plane).to(device)
         for i in range(Ks.shape[0])])

    return Camera(extrinsics=extrinsics, intrinsics=intrinsics)
