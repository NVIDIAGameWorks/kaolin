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
    model = kal.models.PointNet.PointNetClassifier(
        in_channels=3,
        feat_size=1024,
        num_classes=13,
        batchnorm=True,
    )
    x = torch.randn(2, 128, 3)
    out = model(x)
    helpers._assert_shape_eq(out, (2, 13))

    model = kal.models.PointNet.PointNetClassifier(
        in_channels=3,
        feat_size=1024,
        num_classes=13,
        batchnorm=True,
        transposed_input=True,
    )
    x = torch.randn(2, 3, 128)
    out = model(x)
    helpers._assert_shape_eq(out, (2, 13))

    model = kal.models.PointNet.PointNetSegmenter(
        in_channels=3,
        feat_size=1024,
        num_classes=13,
        batchnorm=True,
    )
    x = torch.randn(2, 128, 3)
    out = model(x)
    helpers._assert_shape_eq(out, (2, 128, 13))

    model = kal.models.PointNet.PointNetSegmenter(
        in_channels=3,
        feat_size=1024,
        num_classes=13,
        batchnorm=True,
        transposed_input=True,
    )
    x = torch.randn(2, 3, 128)
    out = model(x)
    helpers._assert_shape_eq(out, (2, 128, 13))
