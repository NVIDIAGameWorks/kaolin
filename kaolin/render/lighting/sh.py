# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
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

# This code is strongly inspired from https://github.com/TheRealMJP/BakingLab

import math

import torch

__all__ = [
    'project_onto_sh9',
    'sh9_irradiance',
    'sh9_diffuse'
]

def project_onto_sh9(directions):
    r"""Project directions, represented as cartesian coordinates,
    onto the spherical harmonic coefficients of degree 3.

    Args:
        directions (torch.Tensor or list of int):
            The directions as cartesian coordinates,
            of any shape but of last dimension 3.

    Returns:
        (torch.Tensor): The spherical harmonics coefficients,
                        of shape ``direction.shape[:-1]`` and last dimension 9.
    """
    # Band 0
    if isinstance(directions, torch.Tensor):
        assert directions.shape[-1] == 3
        x, y, z = torch.split(directions, 1, dim=-1)
        band0 = torch.full_like(x, 0.28209479177)
    elif isinstance(directions, list):
        assert len(directions) == 3
        x, y, z = directions
        band0 = 0.28209479177
    else:
        raise TypeError(f"direction is a {type(direction)}, "
                        "must be a list or a torch.Tensor")
    # Band 1
    band1_m1 = -0.4886025119 * y
    band1_0 = 0.4886025119 * z
    band1_p1 = -0.4886025119 * x

    # Band 2
    band2_m2 = 1.0925484305920792 * (x * y)
    band2_m1 = -1.0925484305920792 * (y * z)
    band2_0 = 0.94617469575 * (z * z) - 0.31539156525
    band2_p1 = -1.0925484305920792 * x * z
    band2_p2 = 0.5462742152960396 * (x * x - y * y)

    if isinstance(directions, torch.Tensor):
        return torch.cat([
            band0,
            band1_m1, band1_0, band1_p1,
            band2_m2, band2_m1, band2_0, band2_p1, band2_p2
        ], dim=-1)
    else:
        return torch.tensor([
            band0,
            band1_m1, band1_0, band1_p1,
            band2_m2, band2_m1, band2_0, band2_p1, band2_p2
        ])

def sh9_irradiance(lights, normals):
    r"""Compute approximate incident irradiance from a single spherical harmonic lobe of degree 3
    representing incoming radiance.

    The clamped cosine lobe is approximated as spherical harmonics.

    Args:
        lights (torch.Tensor): Light parameters of each spherical harmonic
                               (see: :func:`project_onto_sh9`),
                               of 1D size :math:`(9,)`.
        normals (torch.Tensor): Normal of the points where the irradiance is to be estimated,
                                of shape :math:`(\text{num_points}, 3)`.

    Returns:
        (torch.Tensor): The irradiance values, of 1D shape :math:`(\text{num_points},)`.
    """
    assert lights.shape == (9,)
    assert normals.ndim == 2 and normals.shape[-1] == 3
    bands = project_onto_sh9(normals)

    bands[..., 0] *= math.pi
    bands[..., 1:4] *= 2. * math.pi / 3.
    bands[..., 4:] *= math.pi / 4.

    return torch.sum(bands * lights.unsqueeze(-2), dim=-1).reshape(*normals.shape[:-1])

def sh9_diffuse(directions, normals, albedo):
    r"""Compute the outgoing radiance from a single spherical harmonic lobe of degree 3
    representing incoming radiance, using a Lambertian diffuse BRDF.

    Args:
        directions (torch.Tensor): Light directions, of 1D size :math:`(3,)`.
        normals (torch.Tensor): Normal of the points where the radiance is to be estimated,
                                of shape :math:`(\text{num_points}, 3)`.
        albedo (torch.Tensor): albedo (RGB color) of the points where the radiance is to be estimated,
                               of shape :math:`(\text{num_points}, 3)`.

    Returns:
        (torch.Tensor): The diffuse radiance, of same shape than ``albedo``.
    """
    assert directions.shape == (3,)
    assert normals.ndim == 2 and normals.shape[1] == 3
    assert normals.shape == albedo.shape
    lights = project_onto_sh9(directions)
    irradiance = sh9_irradiance(lights, normals)
    return albedo * irradiance.unsqueeze(-1)
