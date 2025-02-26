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

import os
import pytest
import torch
from functools import partial

import kaolin.physics
import kaolin.physics.simplicits
from kaolin.physics.utils.finite_diff import finite_diff_jac
from kaolin.physics.simplicits.skinning import standard_lbs, weight_function_lbs
from kaolin.physics.simplicits.network import SimplicitsMLP

from kaolin.utils.testing import FLOAT_TYPES, with_seed, check_allclose


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_finite_diff_jac_2(device, dtype):
    with_seed(9, 9, 9)
    rtol, atol = torch.tensor([1], dtype=dtype, device=device), torch.tensor(
        [1], dtype=dtype, device=device)

    N = 20
    H = 5
    B = 1
    points = torch.rand(N, 3, device=device, dtype=dtype)  # n x 3 points
    transforms = torch.cat((torch.eye(3, device=device, dtype=dtype), torch.zeros(
        3, 1, device=device, dtype=dtype)), dim=1).repeat(H, 1, 1).unsqueeze(0)
    model = SimplicitsMLP(3, 64, H, 6).to(device)
    if dtype == torch.double:
        model.double()
    weights = model(points)

    partial_weight_fcn_lbs = partial(
        weight_function_lbs, tfms=transforms, fcn=model)

    # N x 3 x 3 Deformation gradients
    transformed_points_jac = finite_diff_jac(
        partial_weight_fcn_lbs, points, eps=1e-7)

    expected = torch.eye(3, device=device, dtype=dtype).repeat(N, 1, 1)

    # weird errors when type is half. rtol and atol does not match the expected device type
    check_allclose(expected, transformed_points_jac,
                          rtol=rtol.item(), atol=atol.item())


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_finite_diff_jac(device, dtype):
    with_seed(9, 9, 9)

    rtol, atol = torch.tensor(
        [1e-3], dtype=dtype, device=device), torch.tensor([1e-3], dtype=dtype, device=device)

    N = 20
    H = 5
    B = 1
    points = torch.rand(N, 3, device=device, dtype=dtype)  # n x 3 points
    transforms = torch.rand(B, H, 3, 4, device=device,
                            dtype=dtype)  # lbs transforms
    # random skinning weights over points
    weights = torch.rand(N, H, device=device, dtype=dtype)

    def two_times(points):
        return 2.0 * points

    transformed_points = two_times(points)

    # N x 3 x 3 Deformation gradients
    transformed_points_jac = finite_diff_jac(two_times, points, eps=1e-8)

    expected = 2.0 * torch.eye(3, device=device, dtype=dtype).repeat(N, 1, 1)

    # weird errors when type is half. rtol and atol does not match the expected device type
    check_allclose(expected, transformed_points_jac,
                          rtol=rtol.item(), atol=atol.item())
