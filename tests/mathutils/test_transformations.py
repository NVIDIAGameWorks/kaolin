# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Kornia components:
# Copyright (C) 2017-2019, Arraiy, Inc., all rights reserved.
# Copyright (C) 2019-    , Open Source Vision Foundation, all rights reserved.
# Copyright (C) 2019-    , Kornia authors, all rights reserved.
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
from torch.autograd import gradcheck
from torch.testing import assert_allclose

import kaolin as kal
from kaolin.mathutils import pi
from kaolin.testing import tensor_to_gradcheck_var


class TestRotx:

    def test_smoke(self):
        theta = torch.Tensor([0.])
        assert kal.mathutils.rotx(theta).shape[0] == 1
        assert_allclose(kal.mathutils.rotx(theta), kal.mathutils.rotx(-theta))

    def test_casual_rotation(self):
        theta = pi * torch.rand(10)
        rx = kal.mathutils.rotx(theta)
        rx_transpose = rx.transpose(1, 2)
        rx_inv = kal.mathutils.rotx(-theta)
        assert_allclose(rx_transpose, rx_inv)

    def test_small_rotation(self):
        theta = 1e-4 * torch.rand(10)
        rx = kal.mathutils.rotx(theta)
        rx_transpose = rx.transpose(1, 2)
        rx_inv = kal.mathutils.rotx(-theta)
        assert_allclose(rx_transpose, rx_inv)

    # def test_gradcheck(self):
    #     theta = tensor_to_gradcheck_var(pi * torch.rand(10))
    #     assert gradcheck(kal.mathutils.rotx, theta, raise_exception=True)


class TestRoty:

    def test_smoke(self):
        theta = torch.Tensor([0.])
        assert kal.mathutils.roty(theta).shape[0] == 1
        assert_allclose(kal.mathutils.roty(theta), kal.mathutils.roty(-theta))

    def test_casual_rotation(self):
        theta = pi * torch.rand(10)
        ry = kal.mathutils.roty(theta)
        ry_transpose = ry.transpose(1, 2)
        ry_inv = kal.mathutils.roty(-theta)
        assert_allclose(ry_transpose, ry_inv)

    def test_small_rotation(self):
        theta = 1e-4 * torch.rand(10)
        ry = kal.mathutils.roty(theta)
        ry_transpose = ry.transpose(1, 2)
        ry_inv = kal.mathutils.roty(-theta)
        assert_allclose(ry_transpose, ry_inv)


class TestRotz:

    def test_smoke(self):
        theta = torch.Tensor([0.])
        assert kal.mathutils.rotz(theta).shape[0] == 1
        assert_allclose(kal.mathutils.rotz(theta), kal.mathutils.rotz(-theta))

    def test_casual_rotation(self):
        theta = pi * torch.rand(10)
        rz = kal.mathutils.rotz(theta)
        rz_transpose = rz.transpose(1, 2)
        rz_inv = kal.mathutils.rotz(-theta)
        assert_allclose(rz_transpose, rz_inv)

    def test_small_rotation(self):
        theta = 1e-4 * torch.rand(10)
        rz = kal.mathutils.rotz(theta)
        rz_transpose = rz.transpose(1, 2)
        rz_inv = kal.mathutils.rotz(-theta)
        assert_allclose(rz_transpose, rz_inv)


def test_homogenize_unhomogenize():
    pts = torch.randn(20, 3)
    pts_homo = kal.mathutils.homogenize_points(pts)
    pts_homo_unhomo = kal.mathutils.unhomogenize_points(pts_homo)
    assert_allclose(pts, pts_homo_unhomo)


def test_homogenize_unhomogenize_batch():
    pts = torch.randn(10, 20, 3)
    pts_homo = kal.mathutils.homogenize_points(pts)
    pts_homo_unhomo = kal.mathutils.unhomogenize_points(pts_homo)
    assert_allclose(pts, pts_homo_unhomo)
