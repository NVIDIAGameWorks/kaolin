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

import pytest
import torch
from torch import Tensor

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_printoptions(precision=8)

from kaolin.math.quat.matrix44 import scale_to_mat44, translation_to_mat44

trans_to_mat44 = [
    [
        torch.tensor([1.0, 2.0, 3.0, 0.0]),
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
    [
        torch.tensor([-1.0, 0.0, 0.0, 0.0]),
        torch.tensor(
            [
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
    [
        torch.tensor([-1.0, 0.0, 0.0]),
        torch.tensor(
            [
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
    [
        torch.tensor([-1.0, 0.0, 0.0, 5.0]),
        torch.tensor(
            [
                [1.0, 0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
    [
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0],
                [4.0, 5.0, 6.0, 0.0],
            ]
        ),
        torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [1.0, 0.0, 0.0, 4.0],
                    [0.0, 1.0, 0.0, 5.0],
                    [0.0, 0.0, 1.0, 6.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ]
        ),
    ],
]

scales_to_mat44 = [
    [
        torch.tensor([1.0, 1.0, 1.0, 0.0]),
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
    [
        torch.tensor([1.0, 2.0, 3.0, 0.0]),
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
    [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
    [
        torch.tensor([1.0, 2.0, 3.0, 5.0]),
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
    [
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 5.0],
                [4.0, 5.0, 6.0, 10.0],
            ]
        ),
        torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                [
                    [4.0, 0.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0, 0.0],
                    [0.0, 0.0, 6.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ]
        ),
    ],
]


class TestMatrix44:
    @pytest.mark.parametrize("translation,expected", trans_to_mat44)
    def test_translation_to_mat44(self, translation: Tensor, expected: Tensor):
        result = translation_to_mat44(translation)
        assert result.device == translation.device
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize("scales,expected", scales_to_mat44)
    def test_scale_to_mat44(self, scales: Tensor, expected: Tensor):
        result = scale_to_mat44(scales)
        assert result.device == scales.device
        assert torch.allclose(result, expected)
