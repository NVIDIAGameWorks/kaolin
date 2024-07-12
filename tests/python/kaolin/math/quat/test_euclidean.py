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

from typing import Optional

import pytest
import torch
from torch import Tensor

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_printoptions(precision=8)

from kaolin.math.quat.euclidean import (
    euclidean_from_rotation_translation,
    euclidean_inverse,
    is_euclidean_valid,
)

euclidean_is_valid = [
    [
        torch.tensor(
            [
                [
                    [-0.8571429, 0.2857143, 0.4285714, 0],
                    [0.2857143, -0.4285714, 0.8571429, 0],
                    [0.4285714, 0.8571429, 0.2857143, 0],
                    [0, 0, 0, 1],
                ]
            ]
        ),
        True,
    ],
    [
        torch.tensor(
            [
                [
                    [-1.8571429, 0.2857143, 0.4285714, 0],
                    [0.2857143, -0.4285714, 0.8571429, 0],
                    [0.4285714, 0.8571429, 0.2857143, 0],
                    [0, 0, 0, 1],
                ]
            ]
        ),
        False,
    ],
    [
        torch.tensor(
            [
                [
                    [-0.8571429, 0.2857143, 0.4285714, 1],
                    [0.2857143, -0.4285714, 0.8571429, 2],
                    [0.4285714, 0.8571429, 0.2857143, 3],
                    [0, 0, 0, 1],
                ]
            ]
        ),
        True,
    ],
    [
        torch.tensor(
            [
                [
                    [-0.8571429, 0.2857143, 0.4285714, 0],
                    [0.2857143, -0.4285714, 0.8571429, 0],
                    [0.4285714, 0.8571429, 0.2857143, 0],
                    [0, 0, 1, 1],
                ]
            ]
        ),
        False,
    ],
    [
        torch.tensor(
            [
                [
                    [-0.8571429, 0.2857143, 0.4285714, 0],
                    [0.2857143, -0.4285714, 0.8571429, 0],
                    [0.4285714, 0.8571429, 0.2857143, 0],
                    [0, 0, 0, 0],
                ]
            ]
        ),
        False,
    ],
    [
        torch.tensor(
            [
                [
                    [1.0, 1.0, 2.0, 0],
                    [0.0, 2.0, 0.0, 0],
                    [2.0, 0.0, 1.0, 0],
                    [0, 0, 0, 1],
                ]
            ]
        ),
        False,
    ],
    [
        torch.tensor(
            [
                [
                    [1.0, 1.0, 0.0, 0],
                    [0.0, 1.0, 0.0, 0],
                    [0.0, 0.0, 1.0, 0],
                    [0, 0, 0, 1],
                ]
            ]
        ),
        False,
    ],
]


euclidean_inversion = [
    [
        torch.tensor(
            [
                [
                    [-0.8245614, 0.0701754, 0.5614035, 0],
                    [0.4912281, -0.4035088, 0.7719298, 0],
                    [0.2807018, 0.9122807, 0.2982456, 0],
                    [0, 0, 0, 1],
                ]
            ]
        ),
        torch.tensor(
            [
                [
                    [-0.8245614, 0.4912281, 0.2807018, 0],
                    [0.0701754, -0.4035088, 0.9122807, 0],
                    [0.5614035, 0.7719298, 0.2982456, 0],
                    [0, 0, 0, 1],
                ]
            ]
        ),
    ],
    [
        torch.tensor(
            [
                [
                    [-0.8245614, 0.0701754, 0.5614035, 1],
                    [0.4912281, -0.4035088, 0.7719298, 2],
                    [0.2807018, 0.9122807, 0.2982456, 3],
                    [0, 0, 0, 1],
                ]
            ]
        ),
        torch.tensor(
            [
                [
                    [-0.8245614, 0.4912281, 0.2807018, -1],
                    [0.0701754, -0.4035088, 0.9122807, -2],
                    [0.5614035, 0.7719298, 0.2982456, -3],
                    [0, 0, 0, 1],
                ]
            ]
        ),
    ],
    [
        torch.tensor(
            [
                [
                    [-0.8245614, 0.0701754, 0.5614035, 1],
                    [0.4912281, -0.4035088, 0.7719298, 2],
                    [0.2807018, 0.9122807, 0.2982456, 3],
                    [0, 0, 0, 1],
                ]
            ]
        ).cuda(),
        torch.tensor(
            [
                [
                    [-0.8245614, 0.4912281, 0.2807018, -1],
                    [0.0701754, -0.4035088, 0.9122807, -2],
                    [0.5614035, 0.7719298, 0.2982456, -3],
                    [0, 0, 0, 1],
                ]
            ]
        ).cuda(),
    ],
]

euclidean_construct = [
    [
        torch.tensor([1, 2, 3, 4.0]),
        torch.tensor([5, 6, 7.0]),
        torch.tensor(
            [
                [0.1333333, -0.6666667, 0.7333333, 5],
                [0.9333333, 0.3333333, 0.1333333, 6],
                [-0.3333333, 0.6666667, 0.6666667, 7],
                [0, 0, 0, 1],
            ]
        ),
    ],
    [
        torch.tensor([1, 2, 3, 4.0]),
        None,
        torch.tensor(
            [
                [0.1333333, -0.6666667, 0.7333333, 0],
                [0.9333333, 0.3333333, 0.1333333, 0],
                [-0.3333333, 0.6666667, 0.6666667, 0],
                [0, 0, 0, 1],
            ]
        ),
    ],
    [
        None,
        torch.tensor([5, 6, 7.0]),
        torch.tensor(
            [
                [1.0, 0, 0, 5],
                [0, 1, 0, 6],
                [0, 0, 1, 7],
                [0, 0, 0, 1],
            ]
        ),
    ],
    [
        torch.tensor([1, 2, 3, 4.0]).cuda(),
        torch.tensor([5, 6, 7.0]).cuda(),
        torch.tensor(
            [
                [0.1333333, -0.6666667, 0.7333333, 5],
                [0.9333333, 0.3333333, 0.1333333, 6],
                [-0.3333333, 0.6666667, 0.6666667, 7],
                [0, 0, 0, 1],
            ]
        ).cuda(),
    ],
]


class TestEuclidean:
    @pytest.mark.parametrize("euclidean,expected", euclidean_is_valid)
    def test_is_euclidean_valid(self, euclidean: Tensor, expected: Tensor):
        is_valid = is_euclidean_valid(euclidean)
        assert is_valid == expected

    @pytest.mark.parametrize("euclidean,expected", euclidean_inversion)
    def test_euclidean_inverse(self, euclidean: Tensor, expected: Tensor):
        inverted = euclidean_inverse(euclidean)
        assert inverted.device == euclidean.device
        assert torch.allclose(inverted, expected)

        # reinverted = euclidean_inverse(inverted)
        # assert reinverted.device == inverted.device
        # assert torch.allclose(reinverted, euclidean)

    @pytest.mark.parametrize("rotation,translation,expected", euclidean_construct)
    def test_euclidean_inverse(self, rotation: Optional[Tensor], translation: Optional[Tensor], expected: Tensor):
        euclidean = euclidean_from_rotation_translation(rotation, translation)
        if rotation is not None:
            assert euclidean.device == rotation.device
        assert euclidean.device == expected.device
        assert torch.allclose(euclidean, expected)
