# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Kornia components Copyright (c) 2019 Kornia project authors
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

import numpy as np
import torch


# Borrowed from kornia
# https://github.com/arraiyopensource/kornia

# Pytorch does not redefine np.pi, hence
pi = torch.tensor(3.14159265358979323846)


# Borrowed from kornia.
# https://github.com/kornia/kornia/blob/master/kornia/geometry/conversions.py
def rad2deg(tensor):
    r"""Converts a tensor of angles from radians to degrees

    Args:
        tensor (torch.Tensor): Input tensor (no shape restrictions).

    Returns:
        torch.Tensor: Tensor of same shape as input.

    Example:
        >>> deg = kaolin.pi * kaolin.rad2deg(torch.rand(1, 2, 3))

    """

    if not torch.is_tensor(tensor):
        raise TypeError('Expected torch.Tensor. Got {} instead.'.format(
            type(tensor)))

    return 180. * tensor / pi.to(tensor.device).type(tensor.dtype)


# Borrowed from kornia.
# https://github.com/kornia/kornia/blob/master/kornia/geometry/conversions.py
def deg2rad(tensor):
    r"""Converts angles from degrees to radians

    Args:
        tensor (torch.Tensor): Input tensor (no shape restrictions).

    Returns:
        torch.Tensor: Tensor of same shape as input.

    Example:
        >>> rad = kaolin.deg2rad(360. * torch.rand(1, 3, 3))

    """

    if not torch.is_tensor(tensor):
        raise TypeError('Expected torch.Tensor. Got {} instead.'.format(
            type(tensor)))

    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.
