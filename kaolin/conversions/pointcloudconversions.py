# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

from typing import Union

import torch
import numpy as np

from kaolin.rep.PointCloud import PointCloud
from kaolin.metrics.point import directed_distance
from kaolin import helpers
from kaolin.conversions.voxelgridconversions import voxelgrid_to_trianglemesh
from kaolin.conversions.voxelgridconversions import voxelgrid_to_sdf


def pointcloud_to_voxelgrid(pts: Union[torch.Tensor, PointCloud, np.ndarray],
                            voxres: int, voxsize: float):
    r"""Converts a pointcloud into a voxel grid.

    Args:
        - pts (torch.Tensor or PointCloud): Pointcloud
            (shape: :math:`N \times 3`, where :math:`N` is the number of points
            in the pointcloud).
        - voxres (int): Resolution of the voxel grid.
        - voxsize (float): size of each voxel grid cell.

    Returns:
        (torch.Tensor): Voxel grid.
    """

    if isinstance(pts, PointCloud):
        pts = pts.points
    helpers._assert_tensor(pts)

    # Create a voxel grid.
    voxels = np.zeros((voxres, voxres, voxres), dtype=np.float32)
    # Enumerate the coordinates of each grid cell
    gridpts = np.where(voxels == 0)
    gridpts = np.asarray([gridpts[0], gridpts[1], gridpts[2]]).T.astype(np.float32)
    gridpts = torch.from_numpy(gridpts)
    # Scale grid coordinates appropriately. We currently have coordinated
    # denoting the corners of a voxel; modify so that we represent the center.
    gridpts = voxsize * (gridpts - (voxres - 1) / 2)
    # Get the distance of the closest point in the pointcloud to each grid
    # point.
    dists = directed_distance(gridpts.cuda().view(-1, 3).contiguous(),
                              pts.cuda().view(-1, 3).contiguous(), mean=False)
    dists = dists.view((voxres, voxres, voxres))
    # If this distance is less than the size of a voxel, treat as occupied,
    # else free.
    on_voxels = np.where(dists.cpu().numpy() <= voxsize)
    voxels[on_voxels] = 1

    return voxels


def pointcloud_to_trianglemesh(points: torch.Tensor):
    device = points.device
    voxels = pointcloud_to_voxelgrid(points, 32, 0.1)
    return voxelgrid_to_trianglemesh(torch.from_numpy(voxels).to(device))


def pointcloud_to_sdf(points: torch.Tensor, num_points=5000):
    device = points.device
    voxels = pointcloud_to_voxelgrid(points, 32, 0.1)
    return voxelgrid_to_sdf(torch.from_numpy(voxels).to(device))
