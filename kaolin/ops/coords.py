# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
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

from __future__ import division

import torch

def spherical2cartesian(azimuth, elevation, distance=None):
    """Convert spherical coordinates to cartesian.

    Assuming X toward camera, Z-up and Y-right.

    Args:
        azimuth (torch.Tensor): azimuth in radianss.
        elevation (torch.Tensor): elevation in radians.
        distance (torch.Tensor or float, optional): distance. Default: 1.

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor):
            x, y, z, of same shape and dtype than inputs.
    """
    if distance is None:
        z = torch.sin(elevation)
        temp = torch.cos(elevation)
    else:
        z = torch.sin(elevation) * distance
        temp = torch.cos(elevation) * distance
    x = torch.cos(azimuth) * temp
    y = torch.sin(azimuth) * temp
    return x, y, z

def cartesian2spherical(x, y, z):
    """Convert cartersian coordinates to spherical in radians.

    Assuming X toward camera, Z-up and Y-right.

    Args:
        x (torch.Tensor): X components of the coordinates.
        y (torch.Tensor): Y components of the coordinates.
        z (torch.Tensor): Z components of the coordinates.

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor):
            azimuth, elevation, distance, of same shape and dtype than inputs.
    """
    distance = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    elevation = torch.asin(z / distance)
    azimuth = torch.atan2(y, x)
    return azimuth, elevation, distance
