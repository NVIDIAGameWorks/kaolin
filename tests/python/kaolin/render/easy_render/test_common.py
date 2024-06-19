# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
import pytest

from kaolin.render.camera import Camera
from kaolin.render.lighting import SgLightingParameters
from kaolin.render.materials import PBRMaterial

import kaolin.render.easy_render.common


class TestCommon:
    def test_default_lighting(self):
        lighting = kaolin.render.easy_render.common.default_lighting()
        assert lighting is not None
        assert type(lighting) is SgLightingParameters

    def test_default_camera(self):
        camera = kaolin.render.easy_render.common.default_camera()
        assert camera is not None
        assert type(camera) is Camera

        for res in [25, 200]:
            camera = kaolin.render.easy_render.common.default_camera(res)
            assert camera is not None
            assert type(camera) is Camera
            assert camera.width == res
            assert camera.height == res

    def test_default_material(self):
        material = kaolin.render.easy_render.default_material()
        assert material is not None
        assert type(material) is PBRMaterial
        assert material.diffuse_color is not None
        for color in [(1., 0, 0), (1, 1, 1), torch.tensor([0, 0.3, 0.3], dtype=torch.float)]:
            material = kaolin.render.easy_render.default_material(color)
            assert material is not None
            assert material.diffuse_color is not None
            assert torch.allclose(torch.tensor(color).float(), material.diffuse_color)