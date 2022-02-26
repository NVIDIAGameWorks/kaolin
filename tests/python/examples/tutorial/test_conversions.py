import numpy as np
import pytest
import torch

from examples.tutorial.conversions import (
    euler_to_quaternion,
    quaternion_to_matrix33,
    quaternion_to_matrix44,
)

q_to_mat44 = [
    [torch.tensor([0, 0, 0, 1]), torch.eye(4)],
    [
        torch.tensor([1, 0, 0, 0]),
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
        torch.tensor([0, 1, 0, 0]),
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
        torch.tensor([0, 0, 1, 0]),
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
]

q_to_mat33 = [
    [torch.tensor([0, 0, 0, 1]), torch.eye(3)],
    [
        torch.tensor([1, 0, 0, 0]),
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        ),
    ],
    [
        torch.tensor([0, 1, 0, 0]),
        torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        ),
    ],
    [
        torch.tensor([0, 0, 1, 0]),
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
]


class TestQuaternion:
    @pytest.mark.parametrize("q,expected", q_to_mat44)
    def test_to_mat44(self, q, expected):
        mat44 = quaternion_to_matrix44(q)
        assert torch.allclose(mat44, expected)

    @pytest.mark.parametrize("q,expected", q_to_mat33)
    def test_to_mat33(self, q, expected):
        mat33 = quaternion_to_matrix33(q)
        assert torch.allclose(mat33, expected)
