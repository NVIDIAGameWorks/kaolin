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

from kaolin.utils.testing import FLOAT_DTYPES, ALL_DEVICES
from kaolin.metrics import voxelgrid as vg_metrics

@pytest.mark.parametrize('dtype', FLOAT_DTYPES)
@pytest.mark.parametrize('device', ALL_DEVICES)
class TestIoU:
    def test_handmade_input(self, device, dtype):
        pred = torch.tensor([[[[0, 1, 1],
                               [1, 0, 0],
                               [1, 0, 0]],

                              [[0, 0, 1],
                               [0, 1, 1],
                               [1, 1, 1]],

                              [[1, 0, 0],
                               [0, 1, 1],
                               [0, 0, 1]]],

                             [[[1, 0, 0],
                               [1, 0, 1],
                               [1, 0, 0]],

                              [[1, 0, 0],
                               [0, 1, 0],
                               [0, 1, 0]],

                              [[0, 1, 0],
                               [0, 1, 1],
                               [0, 0, 1]]]], dtype=dtype, device=device)

        gt = torch.tensor([[[[0, 0, 0],
                             [0, 0, 1],
                             [1, 0, 1]],

                            [[1, 1, 1],
                             [0, 1, 1],
                             [1, 1, 1]],

                            [[1, 0, 0],
                             [1, 1, 0],
                             [0, 1, 0]]],

                           [[[1, 0, 1],
                             [0, 1, 1],
                             [1, 0, 1]],

                            [[0, 1, 0],
                             [1, 1, 1],
                             [0, 0, 1]],

                            [[1, 0, 0],
                             [1, 0, 0],
                             [1, 1, 1]]]], dtype=dtype, device=device)
  
        expected = torch.tensor((0.4500, 0.2273), device=device, dtype=torch.float)
        output = vg_metrics.iou(pred, gt)

        assert torch.allclose(expected, output, atol=1e-4)
