# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

import math

import pytest
import itertools
import numpy as np
import torch

from kaolin.render.camera import Camera
from kaolin.render.camera.intrinsics_ortho import OrthographicIntrinsics
from kaolin.utils.testing import FLOAT_TYPES, ALL_DEVICES, contained_torch_equal


class TestToFromDict:
    def test_from_dict(self):
        in_dict = {
                'width': 100,
                'height': 200,
                'fov_distance': 1.5,
                'near': 0.1,
                'far': 500.0,
            }
        expected_intrinsics = OrthographicIntrinsics.from_frustum(**in_dict)

        intrinsics = OrthographicIntrinsics.from_dict(in_dict)
        in_dict['classname'] = 'orthographic'
        intrinsics2 = OrthographicIntrinsics.from_dict(in_dict)

        assert contained_torch_equal(intrinsics, expected_intrinsics)
        assert contained_torch_equal(intrinsics2, expected_intrinsics)

    def test_to_dict(self):
        intrinsics = OrthographicIntrinsics.from_frustum(
            width=400,
            height=300,
            fov_distance=2.0,
            near=-100.0,
            far=200.0,
            device='cpu',
            dtype=torch.float32,
        )
        intrinsics_dict = intrinsics.as_dict()
        assert intrinsics_dict['classname'] == 'orthographic'
        assert intrinsics_dict['width'] == 400
        assert intrinsics_dict['height'] == 300
        assert intrinsics_dict['fov_distance'] == 2.0
        assert intrinsics_dict['near'] == -100.0
        assert intrinsics_dict['far'] == 200.0

        intrinsics_default = OrthographicIntrinsics.from_frustum(
            width=256,
            height=256,
            fov_distance=1.0,
            device='cpu',
            dtype=torch.float32,
        )
        intrinsics_default_dict = intrinsics_default.as_dict()
        assert intrinsics_default_dict['classname'] == 'orthographic'
        assert intrinsics_default_dict['width'] == intrinsics_default.width
        assert intrinsics_default_dict['height'] == intrinsics_default.height
        assert intrinsics_default_dict['fov_distance'] == intrinsics_default.fov_distance.item()
        assert intrinsics_default_dict['near'] == intrinsics_default.near
        assert intrinsics_default_dict['far'] == intrinsics_default.far

    def test_round_trip(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        intrinsics_all = [
            OrthographicIntrinsics.from_frustum(
                width=320, height=240, fov_distance=1.0,
                device=device, dtype=torch.float32,
            ),
            OrthographicIntrinsics.from_frustum(
                width=800, height=600, fov_distance=0.5,
                near=-50.0, far=100.0,
                device='cpu'
            ),
            OrthographicIntrinsics.from_frustum(
                width=100, height=100, fov_distance=1.2,
                near=0.01, far=2000.0,
                device='cpu', dtype=torch.float32,
            ),
        ]

        for intrinsics in intrinsics_all:
            param_dict = intrinsics.as_dict()
            reconstructed = OrthographicIntrinsics.from_dict(param_dict).to(intrinsics.device)
            param_dict2 = reconstructed.as_dict()

            assert contained_torch_equal(intrinsics, reconstructed)
            assert contained_torch_equal(param_dict, param_dict2)

            assert torch.allclose(reconstructed.params.to(device), intrinsics.params.to(device))
            assert reconstructed.width == intrinsics.width
            assert reconstructed.height == intrinsics.height
            assert reconstructed.near == intrinsics.near
            assert reconstructed.far == intrinsics.far
            assert reconstructed.fov_distance == intrinsics.fov_distance