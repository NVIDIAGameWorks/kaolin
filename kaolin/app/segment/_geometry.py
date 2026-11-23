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
"""Pure-geometry helper: pick a 3D point from rendered depth + a 2D mask (turnaround pivot)."""
import torch


def get_3d_center_from_mask_and_depth(
        mask_2d: torch.Tensor,
        depth_map: torch.Tensor,
        kal_camera,
) -> torch.Tensor:
    """Estimate a world-space "object center" for use as a turnaround pivot.

    1. 2D mask center = centroid of the True pixels of ``mask_2d``.
    2. Object depth = median of ``depth_map`` over the True pixels (robust to floaters).
    3. Unproject (centroid_x, centroid_y, median_depth) into world space using
       ``kal_camera`` (kaolin convention: -Z forward, +Y up; pixel y increases downward).

    Raises ``ValueError`` if ``mask_2d`` has no True pixels.
    """
    mask_bool = mask_2d.bool()
    ys, xs = torch.where(mask_bool)
    if xs.numel() == 0:
        raise ValueError("mask_2d has no True pixels")

    cx = xs.float().mean()
    cy = ys.float().mean()

    depth_2d = depth_map
    while depth_2d.dim() > 2:
        depth_2d = depth_2d.squeeze(0)
    masked_depth = depth_2d[mask_bool]
    d = masked_depth.median()

    intr = kal_camera.intrinsics
    fx = float(intr.focal_x)
    fy = float(intr.focal_y)
    x0 = float(intr.x0)
    y0 = float(intr.y0)
    width = float(kal_camera.width)
    height = float(kal_camera.height)

    d = d.to(dtype=torch.float32)
    cam_z = -d
    cam_x = (cx - (width / 2.0 + x0)) / fx * (-cam_z)
    cam_y = -(cy - (height / 2.0 + y0)) / fy * (-cam_z)

    device = depth_map.device
    cam_pt = torch.stack([cam_x.to(device), cam_y.to(device), cam_z.to(device),
                          torch.tensor(1.0, device=device, dtype=torch.float32)])

    inv_view = kal_camera.extrinsics.inv_view_matrix()[0].to(device=device, dtype=torch.float32)
    world_pt = (inv_view @ cam_pt)[:3]
    return world_pt
