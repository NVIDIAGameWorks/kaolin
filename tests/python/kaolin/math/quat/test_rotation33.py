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
from .test_quaternion import q_to_rot33
from torch import Tensor

from kaolin.math.quat.angle_axis import angle_axis_from_rot33
from kaolin.math.quat.quaternion import quat_from_rot33

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_printoptions(precision=8)

from kaolin.math.quat.rotation33 import (
    is_rot33_valid,
    rot33_from_angle_axis,
    rot33_from_quat,
    rot33_rotate,
)

rot33_is_valid = [
    [
        torch.tensor(
            [
                [
                    [-0.8571429, 0.2857143, 0.4285714],
                    [0.2857143, -0.4285714, 0.8571429],
                    [0.4285714, 0.8571429, 0.2857143],
                ]
            ]
        ),
        True,
    ],
    [
        torch.tensor(
            [
                [
                    [-0.7333333, -0.1333333, 0.6666667],
                    [0.6666667, -0.3333333, 0.6666667],
                    [0.1333333, 0.9333333, 0.3333333],
                ]
            ]
        ),
        True,
    ],
    # batched
    [
        torch.tensor(
            [
                [
                    [-0.8571429, 0.2857143, 0.4285714],
                    [0.2857143, -0.4285714, 0.8571429],
                    [0.4285714, 0.8571429, 0.2857143],
                ],
                [
                    [-0.7333333, -0.1333333, 0.6666667],
                    [0.6666667, -0.3333333, 0.6666667],
                    [0.1333333, 0.9333333, 0.3333333],
                ],
            ]
        ),
        True,
    ],
    # batched
    [
        torch.tensor(
            [
                [
                    [1.0, 1.0, 2.0],
                    [0.0, 2.0, 0.0],
                    [2.0, 0.0, 1.0],
                ],
                [
                    [-0.7333333, -0.1333333, 0.6666667],
                    [0.6666667, -0.3333333, 0.6666667],
                    [0.1333333, 0.9333333, 0.3333333],
                ],
            ]
        ),
        False,
    ],
    [
        torch.tensor(
            [
                [
                    [1.0, 1.0, 2.0],
                    [0.0, 2.0, 0.0],
                    [2.0, 0.0, 1.0],
                ]
            ]
        ),
        False,
    ],
    [
        torch.tensor(
            [
                [
                    [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ]
        ),
        False,
    ],
]

# 1-axis examples: https://www.gatevidyalay.com/3d-rotation-in-computer-graphics-definition-examples/
rot33_point_rotation = [
    # x-axis rotate 90 degrees
    [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor(
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ]
            ]
        ),
        torch.tensor([1.0, -3.0, 2.0]),
    ],
    # y-axis rotate 90 degrees
    [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor(
            [
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ]
            ]
        ),
        torch.tensor([3.0, 2.0, -1.0]),
    ],
    # z-axis rotate 90 degrees
    [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor(
            [
                [
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ]
        ),
        torch.tensor([-2.0, 1.0, 3.0]),
    ],
    # all-axis rotate 90 degrees
    [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor(
            [
                [
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ]
        ),
        torch.tensor([3.0, -2.0, 1.0]),
    ],
    # x-axis rotate 45 degrees
    [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor(
            [
                [
                    [1, 0.0, 0.0],
                    [0.0, 0.7071069, -0.7071069],
                    [0.0, 0.7071069, 0.7071069],
                ]
            ]
        ),
        torch.tensor([1.0, -0.7071069, 3.5355]),
    ],
    # batched: all-axis rotate 90 degrees & x-axis rotate 45 degrees
    [
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ]
        ),
        torch.tensor(
            [
                [
                    [0.0, 0.0, 1.0],
                    [0.0, -1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                [
                    [1, 0.0, 0.0],
                    [0.0, 0.7071069, -0.7071069],
                    [0.0, 0.7071069, 0.7071069],
                ],
            ]
        ),
        torch.tensor(
            [
                [3.0, -2.0, 1.0],
                [1.0, -0.7071069, 3.5355],
            ]
        ),
    ],
    # x-axis rotate 45 degrees, y & z 90 degrees
    [
        torch.tensor([1.0, 2.0, 3.0]).cuda(),
        torch.tensor(
            [
                [
                    [0.0, 0.0, 1.0],
                    [0.7071069, -0.7071069, 0.0],
                    [0.7071069, 0.7071069, 0.0],
                ]
            ],
        ).cuda(),
        torch.tensor([3.0000, -0.7071, 2.1213]).cuda(),
    ],
]


axis_angle_to_rot33 = [
    [
        torch.tensor(3.14),
        torch.tensor([0.2672612, 0.5345225, 0.8017837]),
        torch.tensor(
            [
                [-0.8571417, 0.2844371, 0.4294225],
                [0.2869911, -0.4285705, 0.8567166],
                [0.4277199, 0.8575680, 0.2857147],
            ]
        ),
    ],
    [
        torch.tensor(-1.0),
        torch.tensor([0.2672612, 0.5345225, 0.8017837]),
        torch.tensor(
            [
                [
                    [0.5731379, 0.7403488, -0.3512785],
                    [-0.6090066, 0.6716445, 0.4219059],
                    [0.5482918, -0.0278793, 0.8358222],
                ]
            ]
        ),
    ],
    # batched
    [
        torch.tensor(
            [
                [3.14],
                [-1.0],
            ]
        ),
        torch.tensor(
            [
                [0.2672612, 0.5345225, 0.8017837],
                [0.2672612, 0.5345225, 0.8017837],
            ]
        ),
        torch.tensor(
            [
                [
                    [-0.8571417, 0.2844371, 0.4294225],
                    [0.2869911, -0.4285705, 0.8567166],
                    [0.4277199, 0.8575680, 0.2857147],
                ],
                [
                    [0.5731379, 0.7403488, -0.3512785],
                    [-0.6090066, 0.6716445, 0.4219059],
                    [0.5482918, -0.0278793, 0.8358222],
                ],
            ]
        ),
    ],
    [
        torch.tensor(-1.0, requires_grad=True),
        torch.tensor([0.2672612, 0.5345225, 0.8017837], requires_grad=True),
        torch.tensor(
            [
                [
                    [0.5731379, 0.7403488, -0.3512785],
                    [-0.6090066, 0.6716445, 0.4219059],
                    [0.5482918, -0.0278793, 0.8358222],
                ]
            ],
            requires_grad=True,
        ),
    ],
    [
        torch.tensor(-1.0, requires_grad=True).cuda(),
        torch.tensor([0.2672612, 0.5345225, 0.8017837], requires_grad=True).cuda(),
        torch.tensor(
            [
                [
                    [0.5731379, 0.7403488, -0.3512785],
                    [-0.6090066, 0.6716445, 0.4219059],
                    [0.5482918, -0.0278793, 0.8358222],
                ]
            ],
            requires_grad=True,
        ).cuda(),
    ],
]


class TestRotation33:
    @pytest.mark.parametrize("rot33,expected", rot33_is_valid)
    def test_is_rot33_valid(self, rot33: Tensor, expected: Tensor):
        is_valid = is_rot33_valid(rot33)
        assert is_valid == expected

    @pytest.mark.parametrize("expected,rot33", q_to_rot33)
    def test_rot33_to_quaternion(self, expected: Tensor, rot33: Tensor):
        q = quat_from_rot33(rot33)
        assert q.device == rot33.device
        assert torch.allclose(q, expected)

        r2 = rot33_from_quat(q)
        assert r2.device == rot33.device
        assert torch.allclose(r2, rot33)

    @pytest.mark.parametrize("angle,axis,expected", axis_angle_to_rot33)
    def test_angle_axis_to_rot33(self, angle: Tensor, axis: Tensor, expected: Tensor):
        rot33 = rot33_from_angle_axis(angle, axis)
        assert angle.device == axis.device
        assert rot33.device == angle.device
        assert angle.requires_grad == axis.requires_grad
        assert rot33.requires_grad == angle.requires_grad
        assert torch.allclose(rot33, expected, rtol=1e-3, atol=1e-5)

        angle2, axis2 = angle_axis_from_rot33(rot33)
        assert angle2.device == angle.device
        assert angle2.device == angle.device
        assert axis2.device == axis.device
        assert axis2.device == axis.device

        if angle.ndim > 0:
            # TODO: vectorize. dumb fix for now
            for idx in range(angle.shape[0]):
                if angle[idx] > 0:
                    assert torch.allclose(angle2[idx], angle[idx])
                    assert torch.allclose(axis2[idx], axis[idx])
                else:
                    # method guarantees positive angle, so flip both signs
                    assert torch.allclose(-angle2[idx], angle[idx])
                    assert torch.allclose(-axis2[idx], axis[idx])
        else:
            if angle > 0:
                assert torch.allclose(angle2, angle)
                assert torch.allclose(axis2, axis)
            else:
                # method guarantees positive angle, so flip both signs
                assert torch.allclose(-angle2, angle)
                assert torch.allclose(-axis2, axis)

    @pytest.mark.parametrize("point,rot33,expected", rot33_point_rotation)
    def test_rot33_rotate(self, point: Tensor, rot33: Tensor, expected: Tensor):
        rotated = rot33_rotate(point, rot33)
        assert point.device == rot33.device
        assert rotated.device == point.device
        assert rotated.device == expected.device
        assert torch.allclose(rotated, expected)
