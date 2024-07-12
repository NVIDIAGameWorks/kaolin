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

from .test_quaternion import q_rotate

from kaolin.math.quat.transform import (
    transform_apply,
    transform_from_euclidean,
    transform_from_rotation_translation,
    transform_identity,
    transform_inverse,
    transform_mul,
)

shape_to_identity = [
    [
        [3],
        None,
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        ).cuda(),
    ],
    [
        [5],
        "cpu",
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        ),
    ],
]

transform_to_inverse = [
    [
        torch.tensor(
            [
                [-0.8245614, 0.0701754, 0.5614035, 0, 0, 0, 0],
                [0.4912281, -0.4035088, 0.7719298, 0, 0, 0, 0],
                [0.2807018, 0.9122807, 0.2982456, 0, 0, 0, 0],
            ]
        ),
        torch.tensor(
            [
                [0.8245614, -0.0701754, -0.5614035, 0, 0, 0, 0],
                [-0.4912281, 0.4035088, -0.7719298, 0, 0, 0, 0],
                [-0.2807018, -0.9122807, -0.2982456, 0, 0, 0, 0],
            ]
        ),
    ],
    [
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0, 0, 1],
                [0.0, 0.0, 0.0, 1.0, 0, 0, 2],
                [0.0, 0.0, 0.0, 1.0, 0, 0, 3],
            ]
        ),
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0, 0, -1],
                [0.0, 0.0, 0.0, 1.0, 0, 0, -2],
                [0.0, 0.0, 0.0, 1.0, 0, 0, -3],
            ]
        ),
    ],
    [
        torch.tensor(
            [
                [-0.8246, 0.0702, 0.5614, 0, 0, 0, 1],
            ]
        ),
        torch.tensor(
            [
                [0.82459998, -0.07020000, -0.56140000, 0.00000000, 0.92586088, -0.07882056, 0.36972323],
            ]
        ),
    ],
    [
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0, 0, 1],
                [0.0, 0.0, 0.0, 1.0, 0, 0, 2],
                [0.0, 0.0, 0.0, 1.0, 0, 0, 3],
            ]
        ).cuda(),
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0, 0, -1],
                [0.0, 0.0, 0.0, 1.0, 0, 0, -2],
                [0.0, 0.0, 0.0, 1.0, 0, 0, -3],
            ]
        ).cuda(),
    ],
]

transform_multiply = [
    [
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0, 0, 1],
                [0.0, 0.0, 0.0, 1.0, 0, 0, 2],
                [0.0, 0.0, 0.0, 1.0, 0, 0, 3],
            ]
        ),
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0, 0, 4],
                [0.0, 0.0, 0.0, 1.0, 0, 0, 5],
                [0.0, 0.0, 0.0, 1.0, 0, 0, 6],
            ]
        ),
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0, 0, 0, 5],
                [0.0, 0.0, 0.0, 1.0, 0, 0, 7],
                [0.0, 0.0, 0.0, 1.0, 0, 0, 9],
            ]
        ),
    ],
    [
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0, 0, 0, 0],
            ]
        ),
        torch.tensor(
            [
                [3.0, 0.0, 0.0, 1.0, 0, 0, 0],
            ]
        ),
        torch.tensor(
            [
                [-0.89442718, -0.00000000, -0.00000000, 0.44721359, 0, 0, 0],
            ]
        ),
    ],
    [
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0, 0, 0, 1],
            ]
        ),
        torch.tensor(
            [
                [3.0, 0.0, 0.0, 1.0, 0, 0, 1],
            ]
        ),
        torch.tensor(
            [
                [-0.89442718, -0.00000000, -0.00000000, 0.44721359, 0, -2, 1],
            ]
        ),
    ],
    [
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0, 0, 0, 1],
            ]
        ).cuda(),
        torch.tensor(
            [
                [3.0, 0.0, 0.0, 1.0, 0, 0, 1],
            ]
        ).cuda(),
        torch.tensor(
            [
                [-0.89442718, -0.00000000, -0.00000000, 0.44721359, 0, -2, 1],
            ]
        ).cuda(),
    ],
]

euclidean_to_pq = [
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
        torch.tensor([0.2672612, 0.5345225, 0.8017837, 0, 0, 0, 0]),
    ],
    [
        torch.tensor(
            [
                [
                    [0.1333333, -0.6666667, 0.7333333, 5],
                    [0.9333333, 0.3333333, 0.1333333, 6],
                    [-0.3333333, 0.6666667, 0.6666667, 7],
                    [0, 0, 0, 1],
                ]
            ]
        ),
        torch.tensor([0.1825742, 0.3651484, 0.5477226, 0.7302967, 5, 6, 7]),
    ],
    # batched
    [
        torch.tensor(
            [
                [
                    [-0.8571429, 0.2857143, 0.4285714, 0],
                    [0.2857143, -0.4285714, 0.8571429, 0],
                    [0.4285714, 0.8571429, 0.2857143, 0],
                    [0, 0, 0, 1],
                ],
                [
                    [0.1333333, -0.6666667, 0.7333333, 5],
                    [0.9333333, 0.3333333, 0.1333333, 6],
                    [-0.3333333, 0.6666667, 0.6666667, 7],
                    [0, 0, 0, 1],
                ],
            ]
        ),
        torch.tensor(
            [
                [0.2672612, 0.5345225, 0.8017837, 0, 0, 0, 0],
                [0.1825742, 0.3651484, 0.5477226, 0.7302967, 5, 6, 7],
            ]
        ),
    ],
    [
        torch.tensor(
            [
                [
                    [0.1333333, -0.6666667, 0.7333333, 5],
                    [0.9333333, 0.3333333, 0.1333333, 6],
                    [-0.3333333, 0.6666667, 0.6666667, 7],
                    [0, 0, 0, 1],
                ]
            ]
        ).cuda(),
        torch.tensor([0.1825742, 0.3651484, 0.5477226, 0.7302967, 5, 6, 7]).cuda(),
    ],
]


class TestTransform:
    @pytest.mark.parametrize("shape,device,expected", shape_to_identity)
    def test_transform_identity(self, shape: Tensor, device: Optional[torch.device], expected: Tensor):
        if device:
            identity = transform_identity(shape, device=device)
            assert str(identity.device) == device
        else:
            identity = transform_identity(shape)
        assert identity.device == expected.device
        assert torch.allclose(identity, expected)

    @pytest.mark.parametrize("transform,expected", transform_to_inverse)
    def test_transform_inverse(self, transform: Tensor, expected: Tensor):
        inverted = transform_inverse(transform)
        assert inverted.device == transform.device
        assert inverted.requires_grad == transform.requires_grad
        assert torch.allclose(inverted, expected)

    @pytest.mark.parametrize("t1,t2,expected", transform_multiply)
    def test_transform_mul(self, t1: Tensor, t2: Tensor, expected: Tensor):
        result = transform_mul(t1, t2)
        assert t1.device == t2.device
        assert t1.requires_grad == t2.requires_grad
        assert result.device == t1.device
        assert result.requires_grad == t1.requires_grad
        assert result.device == expected.device
        assert result.requires_grad == expected.requires_grad
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize("q,point,expected", q_rotate)
    def test_transform_apply_rotate(self, q: Tensor, point: Tensor, expected: Tensor):
        """Transform pure rotation should be identical to quaternion rotation."""
        transform = transform_from_rotation_translation(rotation=q)
        result = transform_apply(transform, point)
        assert transform.device == point.device
        assert transform.requires_grad == point.requires_grad
        assert result.device == transform.device
        assert result.requires_grad == transform.requires_grad
        assert result.device == expected.device
        assert result.requires_grad == expected.requires_grad
        assert torch.allclose(result, expected)

    @pytest.mark.parametrize("euclidean,expected", euclidean_to_pq)
    def test_from_euclidean(self, euclidean: Tensor, expected: Tensor):
        transform = transform_from_euclidean(euclidean)
        assert transform.device == euclidean.device
        assert torch.allclose(transform, expected)
