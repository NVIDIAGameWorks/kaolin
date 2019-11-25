# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Kornia components:
# Copyright (C) 2017-2019, Arraiy, Inc., all rights reserved.
# Copyright (C) 2019-    , Open Source Vision Foundation, all rights reserved.
# Copyright (C) 2019-    , Kornia authors, all rights reserved.
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

import pytest

import torch
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kaolin as kal
from kaolin.mathutils import pi
from kaolin.testing import tensor_to_gradcheck_var

# from tests.common import device_type as device


def test_pi():
	assert_allclose(pi, 3.14159265)


@pytest.mark.parametrize('shape', [(1,2,3), (2,1,4), (1,1,1)])
def test_rad2deg(shape, device='cpu'):
	x_rad = pi * torch.rand(shape).to(device)
	x_deg = kal.mathutils.rad2deg(x_rad)
	x_deg_to_rad = kal.mathutils.deg2rad(x_deg)
	assert_allclose(x_rad, x_deg_to_rad, atol=1e-8, rtol=1e-5)

	assert gradcheck(kal.mathutils.rad2deg, (tensor_to_gradcheck_var(x_rad)), 
		raise_exception=True)


@pytest.mark.parametrize('shape', [(2,3,1), (1,3,2), (4,2,18)])
def test_deg2rad(shape, device='cpu'):
	x_deg = 180. * torch.rand(shape)
	x_deg = x_deg.to(torch.device(device))
	x_rad = kal.mathutils.deg2rad(x_deg)
	x_rad_to_deg = kal.mathutils.rad2deg(x_rad)
	assert_allclose(x_deg, x_rad_to_deg, atol=1e-8, rtol=1e-5)

	assert gradcheck(kal.mathutils.deg2rad, (tensor_to_gradcheck_var(x_deg)), 
		raise_exception=True)
