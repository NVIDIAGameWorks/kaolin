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
import math
import torch

import kaolin as kal
from kaolin.render.camera import Camera
from kaolin.render.lighting import SgLightingParameters, sg_direction_from_azimuth_elevation
from kaolin.render.materials import PBRMaterial

__all__ = ['default_lighting', 'default_camera', 'default_material']


def default_lighting():
    """ Returns default lighting, represented as Spherical Gaussians.

    Returns:
        (kaolin.render.lighting.SgLightingParameters) default single SG light
    """
    azimuth = torch.full((1,), 2.3, dtype=torch.float)
    elevation = torch.full((1,), math.pi / 3., dtype=torch.float)
    direction = sg_direction_from_azimuth_elevation(azimuth, elevation)
    return SgLightingParameters(amplitude=3., direction=direction, sharpness=5.)


def default_camera(resolution=512):
    """ Returns default pinhole camera, assuming a scene that's centered and normalized around the origin.

    Args:
        resolution (int): rendering resolution.

    Returns:
        (kaolin.render.camera.Camera)
    """
    return Camera.from_args(
        eye=torch.ones((3,)), at=torch.zeros((3,)), up=torch.tensor([0., 1., 0.]),
        fov=math.pi * 45 / 180, height=resolution, width=resolution
    )


def default_material(diffuse_color=None):
    """ Returns simple default PBR material that is a bit specular.

    Args:
        diffuse_color (tuple or torch.FloatTensor): of 0..1 range RGB color value (green by default)

    Returns:
        (kaolin.render.materials.PBRMaterial)
    """
    if diffuse_color is None:
        diffuse_color = (118.0 / 255.0, 185.0 / 255.0, 0)

    return PBRMaterial(diffuse_color=diffuse_color,
                       specular_color=(0.7, 0.5, 0.7),
                       is_specular_workflow=True,
                       material_name='DefaultMaterial')