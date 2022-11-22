# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.
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
import math
import torch

from kaolin.utils.testing import FLOAT_TYPES, check_tensor
from kaolin.ops.coords import cartesian2spherical, spherical2cartesian

@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
class TestCartesian2Spherical:
    @pytest.fixture(autouse=True)
    def coords(self, device, dtype):
        coords = torch.rand((11, 7, 3), device=device, dtype=dtype) * 10. - 5.
        return {
                'x': coords[..., 0],
                'y': coords[..., 1],
                'z': coords[..., 2]
        }

    def test_cartesian2spherical(self, coords, dtype):
        x = coords['x']
        y = coords['y']
        z = coords['z']

        azimuth, elevation, distance = cartesian2spherical(x, y, z)
        # This is pretty much how it is currently implemented in the function
        # but this is very simple
        expected_distance = torch.sqrt(
            x ** 2 + y ** 2 + z ** 2)
        expected_elevation = torch.asin(z / distance)
        expected_azimuth = torch.atan2(y, z)
        assert torch.allclose(azimuth, expected_azimuth)
        assert torch.allclose(elevation, expected_elevation)
        assert torch.allclose(distance, expected_distance)

    def test_cartesian2spherical2cartesian(self, coords):
        x = coords['x']
        y = coords['y']
        z = coords['z']

        azimuth, elevation, distance = cartesian2spherical(x, y, z)
        out_x, out_y, out_z = spherical2cartesian(azimuth, elevation, distance)
        assert torch.allclose(x, out_x)
        assert torch.allclose(y, out_y)
        assert torch.allclose(z, out_z)

@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
class TestCartesian2Spherical:
    @pytest.fixture(autouse=True)
    def coords(self, device, dtype):
        # Not uniform but good enough
        return {
            'azimuth': (torch.rand((11, 7), device=device, dtype=dtype) * 2. - 1.) * math.pi,
            'elevation': (torch.rand((11, 7), device=device, dtype=dtype) - 0.5) * math.pi,
            'distance': torch.rand((11, 7), device=device, dtype=dtype) * 10. + 0.1
        }

    def test_spherical2cartesian(self, coords, dtype):
        azimuth = coords['azimuth']
        elevation = coords['elevation']
        distance = coords['distance']
        
        x, y, z = spherical2cartesian(azimuth, elevation, distance)
        # This is pretty much how it is currently implemented in the function
        # but this is very simple
        expected_z = torch.sin(elevation) * distance
        temp = torch.cos(elevation) * distance
        expected_x = torch.cos(azimuth) * temp
        expected_y = torch.sin(azimuth) * temp
        assert torch.equal(x, expected_x)
        assert torch.equal(y, expected_y)
        assert torch.equal(z, expected_z)

    def test_spherical2cartesian2spherical(self, coords):
        azimuth = coords['azimuth']
        elevation = coords['elevation']
        distance = coords['distance']
        
        x, y, z = spherical2cartesian(azimuth, elevation, distance)
        out_azimuth, out_elevation, out_distance = cartesian2spherical(
            x, y, z)
        assert torch.allclose(azimuth, out_azimuth, rtol=1e-3, atol=1e-3)
        assert torch.allclose(elevation, out_elevation, rtol=1e-1, atol=1e-1)
        assert torch.allclose(distance, out_distance, rtol=1e-3, atol=1e-3)
