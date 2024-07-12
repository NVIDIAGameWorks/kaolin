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

from kaolin.math.quat.angle_axis import angle_axis_from_quat
from kaolin.math.quat.matrix44 import rot44_from_quat
from kaolin.math.quat.quaternion import (
    quat_conjugate,
    quat_from_angle_axis,
    quat_from_rot33,
    quat_identity,
    quat_mul,
    quat_unit_positive,
    quat_rotate,
    quat_unit,
)
from kaolin.math.quat.rotation33 import rot33_from_quat

shape_to_identity = [
    [
        [3],
        None,
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).cuda(),
    ],
    [
        [5],
        "cpu",
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
]

# via: https://www.andre-gaschler.com/rotationconverter/
q_to_unit = [
    [torch.tensor([0.6, 0.4, 0.2, 0]), torch.tensor([0.8017837, 0.5345225, 0.2672612, 0])],
    [torch.tensor([0.6, 0.4, 0.2, 1]), torch.tensor([0.4803845, 0.3202563, 0.1601282, 0.8006408])],
    [torch.tensor([0.6, 0.4, 0.2, -1]), torch.tensor([0.4803845, 0.3202563, 0.1601282, -0.8006408])],
    [torch.tensor([-0.6, 0.4, 0.2, -1]), torch.tensor([-0.4803845, 0.3202563, 0.1601282, -0.8006408])],
    [torch.tensor([-0.6, 0.4, 0.2, -1]).cuda(), torch.tensor([-0.4803845, 0.3202563, 0.1601282, -0.8006408]).cuda()],
    # batched
    [
        torch.tensor(
            [
                [0.6, 0.4, 0.2, -1],
                [-0.6, 0.4, 0.2, -1],
            ]
        ),
        torch.tensor(
            [
                [0.4803845, 0.3202563, 0.1601282, -0.8006408],
                [-0.4803845, 0.3202563, 0.1601282, -0.8006408],
            ]
        ),
    ],
]

q_to_norm = [
    [torch.tensor([0.6, 0.4, 0.2, 0]), torch.tensor([0.8017837, 0.5345225, 0.2672612, 0])],
    [torch.tensor([0.6, 0.4, 0.2, 1]), torch.tensor([0.4803845, 0.3202563, 0.1601282, 0.8006408])],
    [torch.tensor([0.6, 0.4, 0.2, -1]), torch.tensor([-0.4803845, -0.3202563, -0.1601282, 0.8006408])],
    [torch.tensor([-0.6, 0.4, 0.2, -1]), torch.tensor([0.4803845, -0.3202563, -0.1601282, 0.8006408])],
    [torch.tensor([0.6, 0.4, 0.2, -1]).cuda(), torch.tensor([-0.4803845, -0.3202563, -0.1601282, 0.8006408]).cuda()],
    # batched
    [
        torch.tensor(
            [
                [0.6, 0.4, 0.2, 1],
                [0.6, 0.4, 0.2, -1],
            ]
        ),
        torch.tensor(
            [
                [0.4803845, 0.3202563, 0.1601282, 0.8006408],
                [-0.4803845, -0.3202563, -0.1601282, 0.8006408],
            ]
        ),
    ],
]

q_to_aaxis = [
    [
        torch.tensor([0.8017837, 0.5345225, 0.2672612, 0]),
        [torch.tensor(3.1415927), torch.tensor([0.8017837, 0.5345225, 0.2672612])],
    ],
    [
        torch.tensor([0.4803845, 0.3202563, 0.1601282, 0.8006408]),
        [torch.tensor(1.2848648), torch.tensor([0.8017837, 0.5345225, 0.2672612])],
    ],
    [
        torch.tensor([0.4509876, 0.3006584, 0.375823, 0.751646]).cuda(),
        [torch.tensor(1.4404843).cuda(), torch.tensor([0.6837635, 0.4558423, 0.5698029]).cuda()],
    ],
    # batched
    [
        torch.tensor(
            [
                [0.4509876, 0.3006584, 0.375823, 0.751646],
                [0.4803845, 0.3202563, 0.1601282, 0.8006408],
            ]
        ),
        [
            torch.tensor(
                [
                    [1.4404843],
                    [1.2848648],
                ]
            ),
            torch.tensor(
                [
                    [0.6837635, 0.4558423, 0.5698029],
                    [0.8017837, 0.5345225, 0.2672612],
                ]
            ),
        ],
    ],
    [
        torch.tensor([-0.5465269, -0.3643512, -0.1821756, 0.7316889]).cuda(),
        [torch.tensor(1.5).cuda(), torch.tensor([-0.8017837, -0.5345225, -0.2672612]).cuda()],
    ],
]

q_mul = [
    [torch.tensor([1, 1, 1, 1]), torch.tensor([2, 3, 4, 1]), torch.tensor([4, 2, 6, -8])],
    [torch.tensor([1, 1, 1, 1]), torch.tensor([2, 3, 4, -1]), torch.tensor([2, 0, 4, -10])],
    [torch.tensor([1, 1, 1, 1]), torch.tensor([-2, 3, 4, 1]), torch.tensor([0, -2, 10, -4])],
    [torch.tensor([1, 1, 1, 1]), torch.tensor([2, -3, 4, 1]), torch.tensor([10, -4, 0, -2])],
    [torch.tensor([1, 1, 1, 1]), torch.tensor([2, 3, -4, 1]), torch.tensor([-4, 10, -2, 0])],
    [torch.tensor([1, 1, 1, -1]), torch.tensor([2, 3, 4, 1]), torch.tensor([0, -4, -2, -10])],
    # batched
    [
        torch.tensor(
            [
                [1, 1, 1, 1],
                [1, 1, 1, -1],
            ]
        ),
        torch.tensor(
            [
                [2, 3, -4, 1],
                [2, 3, 4, 1],
            ]
        ),
        torch.tensor(
            [
                [-4, 10, -2, 0],
                [0, -4, -2, -10],
            ]
        ),
    ],
    [torch.tensor([1, 1, 1, -1]).cuda(), torch.tensor([2, 3, 4, 1]).cuda(), torch.tensor([0, -4, -2, -10]).cuda()],
]

q_conj = [
    [torch.tensor([2, 3, 4, 1]), torch.tensor([-2, -3, -4, 1])],
    [torch.tensor([2, 3, 4, -1]), torch.tensor([-2, -3, -4, -1])],
    # batched
    [
        torch.tensor(
            [
                [2, 3, 4, 1],
                [2, 3, 4, -1],
            ]
        ),
        torch.tensor(
            [
                [-2, -3, -4, 1],
                [-2, -3, -4, -1],
            ]
        ),
    ],
    [torch.tensor([2, 3, 4, -1]).cuda(), torch.tensor([-2, -3, -4, -1]).cuda()],
]


# for axis + angle -> quaternion: https://www.andre-gaschler.com/rotationconverter/
# for rotation output: https://www.vcalc.com/wiki/vCalc/V3+-+Vector+Rotation
q_rotate = [
    [
        # batched
        # axis: 1,1,1 & 3,2,1
        # angle: 1 (radian)
        torch.tensor(
            [
                [0.2767965, 0.2767965, 0.2767965, 0.8775826],
                [0.3843956, 0.2562637, 0.1281319, 0.8775826],
            ]
        ),
        torch.tensor(
            [
                [1, 2, 3],
                [1, 2, 3],
            ]
        ),
        torch.tensor(
            [
                [1.945521194, 1.028353001, 3.026125805],
                [2.424939115, -0.06182504, 2.848832735],
            ]
        ),
    ],
    [
        # axis: 3,2,1
        # angle: 1 (radian)
        torch.tensor([[0.3843956, 0.2562637, 0.1281319, 0.8775826]]),
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[2.424939115, -0.06182504, 2.848832735]]),
    ],
    [
        # axis: 3,2,1
        # angle: 1.5 (radian)
        torch.tensor([[0.5465269, 0.3643512, 0.1821756, 0.7316889]]),
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[3.128381622, -0.663741305, 1.942337742]]),
    ],
    [
        # axis: 3,2,1
        # angle: -1.5 (radian)
        torch.tensor([[-0.5465269, -0.3643512, -0.1821756, 0.7316889]]),
        torch.tensor([[1, 2, 3]]),
        torch.tensor([[0.995647631, 3.601726678, -0.190396249]]),
    ],
    [
        # axis: 3,2,1
        # angle: -1.5 (radian)
        torch.tensor([[-0.5465269, -0.3643512, -0.1821756, 0.7316889]]).cuda(),
        torch.tensor([[1, 2, 3]]).cuda(),
        torch.tensor([[0.995647631, 3.601726678, -0.190396249]]).cuda(),
    ],
]

q_to_rot33 = [
    [torch.tensor([0.0, 0.0, 0.0, 1.0]), torch.eye(3, dtype=torch.float)],
    [
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        ),
    ],
    [
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        ),
    ],
    [
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
        torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    ],
    [
        torch.tensor([0.57735, 0.57735, 0.57735, 0.0]),
        torch.tensor(
            [
                [-0.333333, 0.666667, 0.666667],
                [0.666667, -0.333333, 0.666667],
                [0.666667, 0.666667, -0.333333],
            ]
        ),
    ],
    [
        torch.tensor([0.8017837, 0.5345225, 0.2672612, 0]),
        torch.tensor(
            [[0.2857143, 0.8571429, 0.4285714], [0.8571429, -0.4285714, 0.2857143], [0.4285714, 0.2857143, -0.8571429]]
        ),
    ],
    [
        torch.tensor([0.4803845, 0.3202563, 0.1601282, 0.8006408]),
        torch.tensor(
            [[0.7435898, 0.0512821, 0.6666667], [0.5641026, 0.4871795, -0.6666667], [-0.3589744, 0.8717949, 0.3333333]]
        ),
    ],
    # batched
    [
        torch.tensor(
            [
                [0.4803845, 0.3202563, 0.1601282, 0.8006408],
                [0.8017837, 0.5345225, 0.2672612, 0],
            ]
        ),
        torch.tensor(
            [
                [
                    [0.7435898, 0.0512821, 0.6666667],
                    [0.5641026, 0.4871795, -0.6666667],
                    [-0.3589744, 0.8717949, 0.3333333],
                ],
                [
                    [0.2857143, 0.8571429, 0.4285714],
                    [0.8571429, -0.4285714, 0.2857143],
                    [0.4285714, 0.2857143, -0.8571429],
                ],
            ]
        ),
    ],
    [
        torch.tensor([0.4803845, 0.3202563, 0.1601282, 0.8006408]).cuda(),
        torch.tensor(
            [[0.7435898, 0.0512821, 0.6666667], [0.5641026, 0.4871795, -0.6666667], [-0.3589744, 0.8717949, 0.3333333]]
        ).cuda(),
    ],
]

q_to_mat44 = [
    [torch.tensor([0.0, 0.0, 0.0, 1.0]), torch.eye(4, dtype=torch.float)],
    [
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
    [
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
    [
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
        torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
    [
        torch.tensor([0.57735, 0.57735, 0.57735, 0.0]),
        torch.tensor(
            [
                [-0.333333, 0.666667, 0.666667, 0.0],
                [0.666667, -0.333333, 0.666667, 0.0],
                [0.666667, 0.666667, -0.333333, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
    [
        torch.tensor([0.8017837, 0.5345225, 0.2672612, 0]),
        torch.tensor(
            [
                [0.2857143, 0.8571429, 0.4285714, 0],
                [0.8571429, -0.4285714, 0.2857143, 0],
                [0.4285714, 0.2857143, -0.8571429, 0],
                [0, 0, 0, 1],
            ]
        ),
    ],
    # batched
    [
        torch.tensor(
            [
                [0.8017837, 0.5345225, 0.2672612, 0],
                [0.4803845, 0.3202563, 0.1601282, 0.8006408],
            ]
        ),
        torch.tensor(
            [
                [
                    [0.2857143, 0.8571429, 0.4285714, 0],
                    [0.8571429, -0.4285714, 0.2857143, 0],
                    [0.4285714, 0.2857143, -0.8571429, 0],
                    [0, 0, 0, 1],
                ],
                [
                    [0.7435898, 0.0512821, 0.6666667, 0],
                    [0.5641026, 0.4871795, -0.6666667, 0],
                    [-0.3589744, 0.8717949, 0.3333333, 0],
                    [0, 0, 0, 1],
                ],
            ]
        ),
    ],
    [
        torch.tensor([0.4803845, 0.3202563, 0.1601282, 0.8006408]),
        torch.tensor(
            [
                [0.7435898, 0.0512821, 0.6666667, 0],
                [0.5641026, 0.4871795, -0.6666667, 0],
                [-0.3589744, 0.8717949, 0.3333333, 0],
                [0, 0, 0, 1],
            ]
        ),
    ],
    # CUDA
    [
        torch.tensor([0.4803845, 0.3202563, 0.1601282, 0.8006408]).cuda(),
        torch.tensor(
            [
                [0.7435898, 0.0512821, 0.6666667, 0],
                [0.5641026, 0.4871795, -0.6666667, 0],
                [-0.3589744, 0.8717949, 0.3333333, 0],
                [0, 0, 0, 1],
            ]
        ).cuda(),
    ],
]


class TestQuaternion:
    @pytest.mark.parametrize("shape,device,expected", shape_to_identity)
    def test_quat_identity(self, shape: Tensor, device: Optional[torch.device], expected: Tensor):
        if device:
            identity = quat_identity(shape, device=device)
            assert str(identity.device) == device
        else:
            identity = quat_identity(shape)
        assert identity.device == expected.device
        assert torch.allclose(identity, expected)

    @pytest.mark.parametrize("q,expected", q_to_norm)
    def test_quat_unit_positive(self, q: Tensor, expected: Tensor):
        qnorm = quat_unit_positive(q)
        assert qnorm.device == q.device
        assert torch.allclose(qnorm, expected)

    @pytest.mark.parametrize("q,expected", q_to_rot33)
    def test_to_rot33(self, q: Tensor, expected: Tensor):
        mat = rot33_from_quat(q)
        assert mat.device == q.device
        assert torch.allclose(mat, expected)

        q2 = quat_from_rot33(mat)
        assert torch.allclose(q, q2)

    @pytest.mark.parametrize("q,expected", q_to_mat44)
    def test_to_rot44(self, q: Tensor, expected: Tensor):
        mat = rot44_from_quat(q)
        assert mat.device == q.device
        assert torch.allclose(mat, expected)

    @pytest.mark.parametrize("q,expected", q_to_aaxis)
    def test_quat_to_angle_axis(self, q: Tensor, expected: Tensor):
        angle, axis = angle_axis_from_quat(q)
        assert angle.device == q.device
        assert axis.device == q.device
        assert torch.allclose(angle, expected[0])
        assert torch.allclose(axis, expected[1])

        # TODO: need to check negative angles & expected behavior
        qnorm = quat_unit_positive(q)
        q2 = quat_from_angle_axis(angle, axis)
        assert q2.device == angle.device
        assert torch.allclose(q2, qnorm, atol=1e-6)

    @pytest.mark.parametrize("q,expected", q_to_unit)
    def test_unit(self, q: Tensor, expected: Tensor):
        qnorm = quat_unit(q)
        assert qnorm.device == q.device
        assert torch.allclose(qnorm, expected)

    @pytest.mark.parametrize("q,expected", q_conj)
    def test_conj(self, q: Tensor, expected: Tensor):
        q2 = quat_conjugate(q)
        assert q.device == q2.device
        assert torch.allclose(q2, expected)

    @pytest.mark.parametrize("q1,q2,expected", q_mul)
    def test_mul(self, q1: Tensor, q2: Tensor, expected: Tensor):
        q3 = quat_mul(q1, q2)
        assert q3.device == q2.device
        assert q3.device == q1.device
        assert torch.allclose(q3, expected)

    @pytest.mark.parametrize("q,point,expected", q_rotate)
    def test_rotate(self, q: Tensor, point: Tensor, expected: Tensor):
        rot = quat_rotate(q, point)
        assert rot.device == q.device
        assert q.device == point.device
        assert torch.allclose(rot, expected, atol=1e-5)
