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
"""Turnaround view generation around a 2D mask. Public API takes/returns kaolin cameras."""
import math

import kaolin
import torch

# Re-exported for backward compatibility (callers use ``views.dimensions_to_max_resolution``).
from kaolin.render.camera import dimensions_to_max_resolution  # noqa: F401

from .base import SceneSegmenter


def turnaround_cameras_from_camera_and_2dmask(
        mask_2d: torch.Tensor,
        kal_camera,
        num_cameras: int,
        segmenter: SceneSegmenter,
        min_alpha: float = 0.3,
):
    """Generate ``num_cameras + 1`` orbit cameras around the masked region.

    The first returned camera matches ``kal_camera``; the rest are spaced evenly around the
    camera's up axis, pivoting on a single 3D point from
    ``segmenter.get_3d_point_from_mask_2d(mask_2d, kal_camera)``. ``min_alpha`` is reserved.
    """
    del min_alpha  # currently unused; kept for API stability
    device = mask_2d.device
    at = segmenter.get_3d_point_from_mask_2d(mask_2d, kal_camera)

    kc = kal_camera.to(device)
    eye = kc.extrinsics.inv_view_matrix()[0, :3, 3]
    up = kc.extrinsics.R[0, 1, :]

    cameras = [kaolin.render.camera.Camera.from_args(
        eye=eye, at=at, up=up,
        focal_x=kc.focal_x, focal_y=kc.focal_y,
        height=kc.height, width=kc.width, device=device,
    )]

    for idx in range(num_cameras):
        angle = math.pi * 2 * idx / num_cameras
        cameras.append(turnaround_camera_from_camera(kc, angle, at=at))
    middle = len(cameras) // 2
    cameras = cameras[middle:] + cameras[:middle]
    return cameras


def turnaround_camera_from_camera(kal_camera, angle, at, eye=None, up=None, angle_axis=None):
    """Rotate ``kal_camera`` by ``angle`` (radians) around ``angle_axis`` (default: camera up),
    pivoting at ``at``."""
    device = kal_camera.extrinsics.R.device

    if eye is None:
        eye = kal_camera.extrinsics.inv_view_matrix()[0, :3, 3]
    if up is None:
        up = kal_camera.extrinsics.R[0, 1, :]
    if angle_axis is None:
        angle_axis = up.to(device).unsqueeze(0) / torch.linalg.norm(up)

    rot = torch.eye(4, device=device)
    rot[:3, :3] = kaolin.math.quat.rot33_from_angle_axis(torch.tensor([[angle, ]], device=device), angle_axis)
    trans = torch.eye(4, device=device)
    trans[:3, 3] = -at.to(device)
    trans_inv = torch.eye(4, device=device)
    trans_inv[:3, 3] = at.to(device)

    eye_hom = torch.ones([4], device=device, dtype=torch.float32)
    eye_hom[:3] = eye
    transform = trans_inv @ rot @ trans
    new_eye = transform @ eye_hom

    return kaolin.render.camera.Camera.from_args(
        eye=new_eye[:3], at=at, up=up,
        focal_x=kal_camera.focal_x, focal_y=kal_camera.focal_y,
        height=kal_camera.height, width=kal_camera.width, device=device,
    )


def rotate_camera_around_at(
        kal_camera,
        at: torch.Tensor,
        up_axis: torch.Tensor,
        elevation: float = 0.0,
        azimuth: float = 0.0,
):
    """Rotate a kaolin camera around ``at`` by (azimuth, elevation), with ``up_axis`` as the
    azimuth pole. Same intrinsics, new pose.

    Order of rotations:
      1. Azimuth -- rotate the eye around ``up_axis`` (positive = right-hand-rule).
      2. Elevation -- rotate around the right axis (orthogonal to ``up_axis`` and the
         after-azimuth view direction).
    """
    device = kal_camera.extrinsics.R.device
    at = at.to(device=device, dtype=torch.float32)
    up_axis = up_axis.to(device=device, dtype=torch.float32)
    up_axis = up_axis / torch.linalg.norm(up_axis)

    eye = kal_camera.extrinsics.inv_view_matrix()[0, :3, 3].to(device=device, dtype=torch.float32)
    delta = eye - at

    rot_az = kaolin.math.quat.rot33_from_angle_axis(
        torch.tensor([[azimuth]], device=device, dtype=torch.float32),
        up_axis.unsqueeze(0),
    )[0]
    delta = rot_az @ delta

    view_dir = delta / torch.linalg.norm(delta)
    right = torch.cross(up_axis, view_dir, dim=0)
    right_norm = torch.linalg.norm(right)
    if right_norm > 1e-6:
        right = right / right_norm
        rot_el = kaolin.math.quat.rot33_from_angle_axis(
            torch.tensor([[elevation]], device=device, dtype=torch.float32),
            right.unsqueeze(0),
        )[0]
        delta = rot_el @ delta

    new_eye = at + delta
    return kaolin.render.camera.Camera.from_args(
        eye=new_eye, at=at, up=up_axis,
        focal_x=kal_camera.focal_x, focal_y=kal_camera.focal_y,
        height=kal_camera.height, width=kal_camera.width, device=device,
    )
