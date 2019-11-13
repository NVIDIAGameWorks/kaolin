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

from typing import Optional

import numpy as np
import torch

from kaolin import helpers


class PointCloud(object):
    r"""Base class to hold pointcloud objects. """

    def __init__(self, points: Optional[torch.Tensor] = None,
                 normals: torch.Tensor = None, device: Optional[str] = 'cpu',
                 copy: Optional[bool] = False):
        r"""Initialize a PointCloud object, given a tensor of points, and
        optionally, a tensor representing poincloud normals.

        Args:
            pts (torch.Tensor): Points that make up the pointcloud (shape:
                :math:`... \times N \times D`), where :math:`N` denotes the
                number of points in the cloud, and :math:`D` denotes the
                dimensionality of each point.
            normals (torch.Tensor): Normals for each point in the cloud
                (shape: :math:`N \times D`, where `D` = 2 or `D` = 3).
                That is, normals can only be provided for 2D or 3D pointclouds.
            device (str, Optional): Device to store the pointcloud object on
                (default: 'cpu'). Must be a valid `torch.device` type.
            copy (bool, Optional): Whether or not to create a deep copy of the
                Tensor(s) used to initialze class members.

        """
        if points is None:
            self.points = None
        else:
            helpers._assert_tensor(points)
            helpers._assert_dim_ge(points, 2)
            self.points = points.clone() if copy else points
            self.points = self.points.to(device)
        if normals is None:
            self.normals = None
        else:
            helpers._assert_tensor(normals)
            if points.dim() == 2:
                helpers._assert_shape_eq(normals, (points.shape[-2], 3))
            self.normals = normals.clone() if copy else normals
            self.normals = self.normals.to(device)


def bounding_points(points: torch.Tensor, bbox: list, padding: float = .05):
    r"""Returns the indices of a set of points which lies within a supplied
    bounding box.

    Args:
        point (torch.Tensor) : Input pointcloud
        bbox (list) : bouding box values (min_x, max_x, min_y, max_y,
            min_z, max_z)
        padding (float) : padding to add to bounding box

    Returns:
        (list): list of indices which lie within supplied bounding box

    Example:
        >>> points = torch.rand(1000)
        >>> subset_idx = bounding_points(points, [.1, .9, .1, .9, .1, .9])
        >>> subset = points[subset_idx]
    """

    x_vals = points[:, 0] >= (bbox[0] - padding)
    x_vals = x_vals & (points[:, 0] <= (bbox[1] + padding))

    y_vals = points[:, 1] >= (bbox[2] - padding)
    y_vals = y_vals & (points[:, 1] <= (bbox[3] + padding))

    z_vals = points[:, 2] >= (bbox[4] - padding)
    z_vals = z_vals & (points[:, 2] <= (bbox[5] + padding))

    sample_box = x_vals & (y_vals & z_vals)

    return sample_box


def random_input_dropout(points: torch.Tensor,
                         max_dropout_rate: float = 0.95):
    """Returns a copy of the given cloud with points randomly removed
    according to max_dropout_rate.

    For each batch, first select a dropout_rate from the uniform distribution
    [0, max_dropout_rate], then  remove (i.e. set to an existing point)
    with a probability equal to the dropout rate.

    Based on the technique described in PointNet++.

    Args:
        points (torch.Tensor or np.ndarray): Input pointcloud
            shape = (batch_size, num_points, num_dim) or (num_points, num_dim)
        max_dropout_rate (float): See method description above.

    """

    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points)

    if not torch.is_tensor(points):
        raise TypeError('Expected type torch.Tensor. Got {} instead.'.format(
            type(points)))

    assert points.dim() == 2 or points.dim(
    ) == 3, 'Point cloud must contain exactly 2 or 3 dimensions.'

    batched = False
    if points.dim() == 3:
        batched = True

    else:
        points = points.unsqueeze(0)

    batch_size = points.shape[0]
    dropout_rates = torch.FloatTensor(
        batch_size, 1).uniform_(0, max_dropout_rate)

    r = torch.rand(*points.shape[:2])
    points = points.clone()
    points[r < dropout_rates, :] = points[0, 0, :]

    if not batched:
        points = points.squeeze(0)

    return points
