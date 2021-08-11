# Copyright (c) 2019,20-21 NVIDIA CORPORATION & AFFILIATES.
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

from kaolin.ops.random import random_shape_per_tensor, random_tensor, \
                              random_spc_octrees, manual_seed
from kaolin.utils.testing import BOOL_TYPES, NUM_TYPES, check_tensor, \
                                 check_spc_octrees


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("min_shape,max_shape",
                         [(None, (3, 3)), ((5, 5), (5, 5))])
def test_random_shape_per_tensor(batch_size, min_shape, max_shape):
    old_seed = torch.initial_seed()
    torch.manual_seed(0)
    shape_per_tensor = random_shape_per_tensor(batch_size, min_shape, max_shape)
    if min_shape is None:
        min_shape = tuple([1] * len(max_shape))
    min_shape = torch.tensor(min_shape).unsqueeze(0)
    max_shape = torch.tensor(max_shape).unsqueeze(0)
    assert shape_per_tensor.shape[0] == batch_size
    assert (min_shape <= shape_per_tensor).all() and (
            shape_per_tensor <= max_shape).all()


@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("min_shape,max_shape", [((5, 5, 5), (30, 30, 30))])
def test_random_shape_per_tensor_seed(batch_size, min_shape, max_shape):
    threshold = batch_size * len(max_shape) * 0.9
    manual_seed(0)
    shape_per_tensor1 = random_shape_per_tensor(batch_size, min_shape,
                                                max_shape)
    shape_per_tensor2 = random_shape_per_tensor(batch_size, min_shape,
                                                max_shape)
    assert torch.sum(shape_per_tensor1 != shape_per_tensor2) > threshold
    manual_seed(0)
    shape_per_tensor3 = random_shape_per_tensor(batch_size, min_shape,
                                                max_shape)
    assert torch.equal(shape_per_tensor1, shape_per_tensor3)
    manual_seed(1)
    shape_per_tensor4 = random_shape_per_tensor(batch_size, min_shape,
                                                max_shape)
    assert torch.sum(shape_per_tensor1 != shape_per_tensor4) > threshold


@pytest.mark.parametrize("device,dtype", NUM_TYPES)
@pytest.mark.parametrize("low,high", [(0, 1), (3, 5), (10, 10)])
@pytest.mark.parametrize("shape", [(1,), (3, 3)])
def test_random_tensor(low, high, shape, dtype, device):
    tensor = random_tensor(low, high, shape, dtype, device)
    check_tensor(tensor, shape, dtype, device)
    assert (low <= tensor).all()
    assert (tensor <= high).all()


@pytest.mark.parametrize("device,dtype", BOOL_TYPES)
@pytest.mark.parametrize("low,high", [(0, 1)])
@pytest.mark.parametrize("shape", [(1,), (3, 3)])
def test_random_tensor(low, high, shape, dtype, device):
    tensor = random_tensor(low, high, shape, dtype, device)
    check_tensor(tensor, shape, dtype, device)
    assert (low <= tensor).all()
    assert (tensor <= high).all()


@pytest.mark.parametrize("low,high", [(0, 1), (5, 10)])
@pytest.mark.parametrize("shape", [(10, 10)])
def test_random_tensor_seed(low, high, shape):
    threshold = shape[0] * shape[1] * 0.9
    manual_seed(0)
    tensor1 = random_tensor(low, high, shape)
    tensor2 = random_tensor(low, high, shape)
    assert torch.sum(tensor1 != tensor2) > threshold
    manual_seed(0)
    tensor3 = random_tensor(low, high, shape)
    assert torch.equal(tensor1, tensor3)
    manual_seed(1)
    tensor4 = random_tensor(low, high, shape)
    assert torch.sum(tensor1 != tensor4) > threshold

@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("level", [1, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_random_spc_octree(batch_size, level, device):
    octrees, lengths = random_spc_octrees(batch_size, level, device)
    check_spc_octrees(octrees, lengths, batch_size, level, device)
