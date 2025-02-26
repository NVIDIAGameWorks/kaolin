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

import SparseSimplicits.kaolin.kaolin.physics.materials.material_utils as material_utils


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_to_lame(device, dtype):

    N = 20

    yms = 1e6 * torch.ones(N, device=device, dtype=dtype)
    prs = 0.35 * torch.ones(N, device=device, dtype=dtype)

    mus, lams = material_utils.to_lame(yms, prs)

    expected_mus = yms / (2 * (1 + prs))
    expected_lams = yms * prs / ((1 + prs) * (1 - 2 * prs))

    assert torch.allclose(mus, expected_mus)
    assert torch.allclose(lams, expected_lams)
