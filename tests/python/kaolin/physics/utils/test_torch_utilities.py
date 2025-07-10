# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

import pytest

import torch
import numpy as np
import warp as wp
import warp.sparse as wps
import kaolin.physics as physics
from kaolin.physics.utils.torch_utilities import standard_transform_to_relative, create_projection_matrix, hess_reduction, torch_bsr_to_torch_triplets
from kaolin.utils.testing import check_allclose


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_standard_transform_to_relative(device, dtype):
    # Test 4x4 transform
    transform = torch.eye(4, device=device, dtype=dtype)
    transform[0, 3] = 1.0
    transform[1, 3] = 2.0
    transform[2, 3] = 3.0
    relative_transform = standard_transform_to_relative(transform)
    
    expected = torch.zeros((3, 4), device=device, dtype=dtype)
    expected[0, 3] = 1.0
    expected[1, 3] = 2.0 
    expected[2, 3] = 3.0
    check_allclose(relative_transform, expected)

    # Test 3x4 transform
    transform_3x4 = torch.zeros((3, 4), device=device, dtype=dtype)
    transform_3x4[:3, :3] = torch.eye(3, device=device, dtype=dtype)
    transform_3x4[0, 3] = 1.0
    transform_3x4[1, 3] = 2.0
    transform_3x4[2, 3] = 3.0
    relative_transform = standard_transform_to_relative(transform_3x4)
    check_allclose(relative_transform, expected)

    # Test invalid input shape raises error
    with pytest.raises(ValueError):
        transform_invalid = torch.eye(3, device=device, dtype=dtype)
        standard_transform_to_relative(transform_invalid)
        
@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_create_projection_matrix(device, dtype):
    # Test basic case - removing single DOF
    num_dofs = 5
    kin_dofs = torch.tensor([2], device=device)
    P = create_projection_matrix(num_dofs, kin_dofs)
    
    expected = torch.tensor([[1., 0., 0., 0., 0.],
                           [0., 1., 0., 0., 0.],
                           [0., 0., 0., 1., 0.],
                           [0., 0., 0., 0., 1.]], device=device)
    check_allclose(P, expected)

    # Test removing multiple DOFs
    num_dofs = 6
    kin_dofs = torch.tensor([1, 3, 5], device=device)
    P = create_projection_matrix(num_dofs, kin_dofs)
    
    expected = torch.tensor([[1., 0., 0., 0., 0., 0.],
                           [0., 0., 1., 0., 0., 0.],
                           [0., 0., 0., 0., 1., 0.]], device=device)
    check_allclose(P, expected)

    # Test removing no DOFs
    num_dofs = 3
    kin_dofs = torch.tensor([], device=device, dtype=torch.int64)
    P = create_projection_matrix(num_dofs, kin_dofs)
    
    expected = torch.eye(3, device=device)
    check_allclose(P, expected)

    # Test removing all DOFs
    num_dofs = 4
    kin_dofs = torch.arange(num_dofs, device=device)
    P = create_projection_matrix(num_dofs, kin_dofs)
    
    expected = torch.empty((0, num_dofs), device=device)
    check_allclose(P, expected)

