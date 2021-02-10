# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
  
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import random

from kaolin.metrics import render
from kaolin.utils.testing import FLOAT_TYPES

@pytest.mark.parametrize('device,dtype', FLOAT_TYPES)
class TestRender:

    @pytest.fixture(autouse=True)
    def lhs_mask(self, device, dtype):
        return torch.tensor([[[0., 0.2, 0.1, 1.],
                              [0.5, 0.5, 0.9, 0.9],
                              [0., 1., 1., 0.9],
                              [0.8, 0.7, 0.2, 0.1]],
                             [[1., 1., 1., 1.],
                              [1., 1., 1., 1.],
                              [1., 1., 1., 1.],
                              [1., 1., 1., 1.]]],
                            dtype=dtype, device=device)

    @pytest.fixture(autouse=True)
    def rhs_mask(self, device, dtype):
        return torch.tensor([[[0.1, 0.3, 0.3, 0.9],
                              [0.5, 0.5, 1., 0.3],
                              [0., 0.9, 0.9, 0.8],
                              [1., 1., 0., 0.]],
                             [[0.3, 0.6, 0.7, 0.7],
                              [0.8, 0.9, 0.9, 1.],
                              [1., 0.9, 0.9, 0.5],
                              [0.8, 0.7, 0.8, 0.5]]],
                            dtype=dtype, device=device)

    def test_mask_iou(self, lhs_mask, rhs_mask, device, dtype):
        loss = render.mask_iou(lhs_mask, rhs_mask)
        assert torch.allclose(loss, torch.tensor([0.3105],
                              dtype=dtype, device=device))
