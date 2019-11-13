# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

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

import kaolin as kal
from kaolin import helpers


def test_smoke(device='cpu'):
    net = kal.models.VoxelSuperresODM.SuperresNetwork(30, 15)
    x = torch.randn(1, 6, 128, 128)
    x_ = net(x)
    helpers._assert_shape_eq(x, x_.shape, dim=0)
