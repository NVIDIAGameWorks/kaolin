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

from kaolin.physics.simplicits.utils import standard_lbs, weight_function_lbs

from kaolin.utils.testing import FLOAT_TYPES, with_seed


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_weight_fcn_lbs(device, dtype):
    with_seed(9, 9, 9)
    if dtype == torch.half:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-8  # default torch values

    N = 20
    H = 5
    B = 1
    points = torch.rand(N, 3, device=device, dtype=dtype)  # n x 3 points
    transforms = torch.rand(B, H, 3, 4, device=device,
                            dtype=dtype)  # lbs transforms

    # random skinning weights over points
    weights = torch.ones(N, H, device=device, dtype=dtype)

    def weight_fcn(x0):
        return torch.ones(x0.shape[0], H, device=device, dtype=dtype)

    transformed_points = weight_function_lbs(points, transforms, weight_fcn)

    for i in range(N):
        pt_i = points[i].unsqueeze(0)
        pt4_i = torch.cat(
            (pt_i, torch.tensor([[1]], device=device, dtype=dtype)), dim=1)
        expected_point = pt_i.T
        for j in range(H):
            expected_point += weights[i, j] * transforms[0, j] @ pt4_i.T
        assert torch.allclose(expected_point.flatten(
        ), transformed_points[i].flatten(), atol=atol, rtol=rtol)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_standard_lbs(device, dtype):
    with_seed(9, 9, 9)
    if dtype == torch.half:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-8  # default torch values

    N = 20
    H = 5
    B = 1
    points = torch.rand(N, 3, device=device, dtype=dtype)  # n x 3 points
    transforms = torch.rand(B, H, 3, 4, device=device,
                            dtype=dtype)  # lbs transforms
    # random skinning weights over points
    weights = torch.rand(N, H, device=device, dtype=dtype)

    transformed_points = standard_lbs(points, transforms, weights)

    for i in range(N):
        pt_i = points[i].unsqueeze(0)
        pt4_i = torch.cat(
            (pt_i, torch.tensor([[1]], device=device, dtype=dtype)), dim=1)
        expected_point = pt_i.T
        for j in range(H):
            expected_point += weights[i, j] * transforms[0, j] @ pt4_i.T

        # checking 3D point by 3D point, so flattening should be ok
        assert torch.allclose(expected_point.flatten(
        ), transformed_points[i].flatten(), atol=atol, rtol=rtol)
