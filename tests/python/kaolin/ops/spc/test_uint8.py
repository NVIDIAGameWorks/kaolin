# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
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
from itertools import product

import torch

from kaolin.ops.spc import uint8_to_bits, uint8_bits_sum, \
                           bits_to_uint8

@pytest.mark.parametrize('test_all', [False, True])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
class TestUint8:
    @pytest.fixture(autouse=True)
    def bits_t(self, test_all, device):
        if test_all:
            bits_t = torch.tensor(list(product([False, True], repeat=8)),
                                  dtype=torch.bool, device=device)
        else:
            bits_t = torch.tensor([
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 0, 0, 0, 1]
            ], dtype=torch.bool, device=device)
        # convert to left-to-right binary
        return torch.flip(bits_t, dims=(-1,)).contiguous()

    @pytest.fixture(autouse=True)
    def uint8_t(self, test_all, device):
        if test_all:
            return torch.arange(256, dtype=torch.uint8, device=device)
        else:
            return torch.tensor([0, 1, 15, 255, 129],
                               dtype=torch.uint8, device=device)

    def test_uint8_to_bits(self, uint8_t, bits_t):
        out = uint8_to_bits(uint8_t)
        assert torch.equal(out, bits_t)

    def test_uint8_bits_sum(self, uint8_t, bits_t):
        out = uint8_bits_sum(uint8_t)
        assert torch.equal(out, torch.sum(bits_t, dim=-1))

    def test_bits_to_uint8(self, uint8_t, bits_t):
        out = bits_to_uint8(bits_t)
        assert torch.equal(out, uint8_t)
