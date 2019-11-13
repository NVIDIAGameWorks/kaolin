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
#
#
# A PyTorch implementation of Neural 3D Mesh Renderer
#
# Copyright (c) 2017 Hiroharu Kato
# Copyright (c) 2018 Nikos Kolotouros
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional, Union

import torch

import kaolin as kal


def get_eye_from_spherical_coords(distance: torch.Tensor,
                                  elevation: torch.Tensor,
                                  azimuth: torch.Tensor,
                                  degrees: Optional[bool] = True):
    r"""Returns the Cartesian position of the eye of the camera, given
    spherical coordinates (distance, azimuth, and elevation).

    Args:
        distance (torch.Tensor): Distance of the eye from the object.
        elevation (torch.Tensor): Elevation angle.
        azimuth (torch.Tensor): Azimuth angle.
        degrees (bool, optional): Bool to indicate that `azimuth` and
            `elevation` are specified in degrees (default: True).

    Returns:
        (torch.Tensor): Position of the "eye" of the camera.

    """

    if degrees:
        azimuth = kal.mathutils.deg2rad(azimuth)
        elevation = kal.mathutils.deg2rad(elevation)
    return torch.stack([
            distance * torch.cos(elevation) * torch.sin(azimuth),
            distance * torch.sin(elevation),
            -distance * torch.cos(elevation) * torch.cos(azimuth)
        ]).transpose(1, 0)
