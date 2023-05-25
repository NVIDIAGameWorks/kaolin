# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

from kaolin.utils.testing import FLOAT_TYPES, with_seed
import kaolin.ops.pointcloud


@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
def test_center_points(device, dtype):
    with_seed(9, 9, 9)
    if dtype == torch.half:
        rtol, atol = 1e-3, 1e-3
    else:
        rtol, atol = 1e-5, 1e-8  # default torch values

    B = 4
    N = 20
    points = torch.rand((B, N, 3), device=device, dtype=dtype)  # 0..1
    points[:, 0, :] = 1.0  # make sure 1 is included
    points[:, 1, :] = 0.0  # make sure 0 is included
    points = points - 0.5  # -0.5...0.5

    factors = 0.2 + 2 * torch.rand((B, 1, 1), device=device, dtype=dtype)
    translations = torch.rand((B, 1, 3), device=device, dtype=dtype) - 0.5

    # Points are already centered
    assert torch.allclose(points, kaolin.ops.pointcloud.center_points(points), atol=atol, rtol=rtol)
    assert torch.allclose(points * factors, kaolin.ops.pointcloud.center_points(points * factors), atol=atol, rtol=rtol)

    # Points translated
    assert torch.allclose(points, kaolin.ops.pointcloud.center_points(points + 0.5), atol=atol, rtol=rtol)

    points_centered = kaolin.ops.pointcloud.center_points(points + translations)
    assert torch.allclose(points, points_centered, atol=atol, rtol=rtol)

    points_centered = kaolin.ops.pointcloud.center_points(points * factors + translations)
    assert torch.allclose(points * factors, points_centered, atol=atol, rtol=rtol)

    # Now let's also try to normalize
    points_centered = kaolin.ops.pointcloud.center_points(points * factors + translations, normalize=True)
    assert torch.allclose(points, points_centered, atol=atol, rtol=rtol)

    # Now let's test normalizing when there is zero range in one of the dimensions
    points[:, :, 1] = 1.0
    points_centered = kaolin.ops.pointcloud.center_points(points * factors + translations, normalize=True)
    points[:, :, 1] = 0.0
    assert torch.allclose(points, points_centered, atol=atol, rtol=rtol)

    # Now let's try normalizing when one element of the batch is degenerate
    points[0, :, :] = torch.tensor([0, 2., 4.], dtype=dtype, device=device).reshape((1, 3))
    points_centered = kaolin.ops.pointcloud.center_points(points * factors + translations, normalize=True)
    points[0, :, :] = 0
    assert torch.allclose(points, points_centered, atol=atol, rtol=rtol)