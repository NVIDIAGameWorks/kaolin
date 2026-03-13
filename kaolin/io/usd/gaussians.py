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

from collections import namedtuple
import os
import math
from tqdm import tqdm

import numpy as np
import torch

try:
    from pxr import Usd, UsdGeom, UsdVol
except ImportError:
    pass

from .utils import _get_stage_from_maybe_file, get_scene_paths, create_stage
from kaolin.utils.testing import check_tensor
from kaolin.ops.gaussians import transform_gaussians


gaussiancloud_return_type = namedtuple('gaussiancloud_return_type', ['positions', 'orientations', 'scales', 'opacities', 'sh_coeff'])

__all__ = [
    'add_gaussiancloud',
    'export_gaussiancloud',
    'get_gaussiancloud_scene_paths',
    'import_gaussianclouds',
    'import_all_gaussianclouds'
]

def _convert_if_half(tensor):
    if tensor.dtype == torch.half:
        return tensor.float()
    else:
        return tensor

def _get_gaussiancloud(prim, time):
    assert isinstance(prim, Usd.Prim)
    assert prim.GetTypeName() == "ParticleField3DGaussianSplat"
    gaussian = UsdVol.ParticleField3DGaussianSplat(prim)
    output = {}
    attr = gaussian.GetPositionsAttr().Get(time=time)
    if attr:
        output['positions'] = torch.tensor(np.asarray(attr))
    else:
        output['positions'] = torch.tensor(np.asarray(gaussian.GetPositionshAttr().Get(time=time)))
    attr = gaussian.GetOrientationsAttr().Get(time=time)
    if attr:
        output['orientations'] = torch.tensor(np.asarray(attr))
    else:
        output['orientations'] = torch.tensor(np.asarray(gaussian.GetOrientationshAttr().Get(time=time)))
    attr = gaussian.GetScalesAttr().Get(time=time)
    if attr:
        output['scales'] = torch.tensor(np.asarray(attr))
    else:
        output['scales'] = torch.tensor(np.asarray(gaussian.GetScaleshAttr().Get(time=time)))
    attr = gaussian.GetOpacitiesAttr().Get(time=time)
    if attr:
        output['opacities'] = torch.tensor(np.asarray(attr))
    else:
        output['opacities'] = torch.tensor(np.asarray(gaussian.GetOpacitieshAttr().Get(time=time)))

    sh_degree = gaussian.GetRadianceSphericalHarmonicsDegreeAttr().Get()
    num_coeffs = (sh_degree + 1) ** 2
    attr = gaussian.GetRadianceSphericalHarmonicsCoefficientsAttr().Get(time=time)
    if attr:
        output['sh_coeff'] = torch.tensor(np.asarray(attr)).reshape(-1, num_coeffs, 3)
    else:
        output['sh_coeff'] = torch.tensor(np.asarray(gaussian.GetRadianceSphericalHarmonicsCoefficientshAttr().Get(time=time))).reshape(-1, num_coeffs, 3)
    
    transforms = torch.tensor(np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(time), dtype=np.float32)).permute(1, 0)
    if sh_degree > 0:
        new_position_xyz, new_orientation_quat, new_scale, new_shs = transform_gaussians(
            _convert_if_half(output['positions']),
            _convert_if_half(output['orientations']),
            _convert_if_half(output['scales']),
            transforms,
            _convert_if_half(output['sh_coeff'][:, 1:])
        )
        output['sh_coeff'][:, 1:] = new_shs.to(output['sh_coeff'][:, 1:].dtype)
    else:
        new_position_xyz, new_orientation_quat, new_scale, new_shs = transform_gaussians(
            _convert_if_half(output['positions']),
            _convert_if_half(output['orientations']),
            _convert_if_half(output['scales']),
            transforms,
        )
    output['positions'] = new_position_xyz.to(output['positions'].dtype)
    output['orientations'] = new_orientation_quat.to(output['orientations'].dtype)
    output['scales'] = new_scale.to(output['scales'].dtype)
    return output    

def import_gaussianclouds(file_path_or_stage, scene_paths, times=None):
    r"""Import gaussians from Usd file.

    Args:
        file_path_or_stage (Usd.Stage or str):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        scene_paths (list of str): Scene paths within the USD file indicating which primitive(s)
            to import. If None, all prims of type gaussianscloud will be imported.
        times (list of Float or Float, optional):
            Positive integer defining the time at which the supplied parameters
            correspond to.

    Returns:
        (A list of dict):
            Each dict represent a gaussian cloud with the following fields:

            * **positions** (torch.Tensor):
                Position of each gaussian, of shape :math:`(\text{num_gaussians}, 3)`.
            * **orientations** (torch.Tensor):
                Orientation of each gaussian as quaternions,
                of shape :math:`(\text{num_gaussians}, 4)`.
            * **scales** (torch.Tensor):
                Scale of each gaussian, of shape :math:`(\text{num_gaussians}, 3)`.
            * **opacities** (torch.Tensor):
                Opacity of each gaussian, of shape :math:`(\text{num_gaussians})`.
            * **sh_coeff** (torch.Tensor):
                Spherical harmonics coefficients of each gaussian,
                of shape :math:`(\text{num_gaussians}, (\text{num_degrees} + 1)^2, 3)`.
    """
    stage = _get_stage_from_maybe_file(file_path_or_stage)
    try:
        if times is None:
            times = [Usd.TimeCode.Default()] * len(scene_paths)
        elif not isinstance(times, list):
            times = [times] * len(scene_paths)

        silence_tqdm = len(scene_paths) < 10
        output = []
        for scene_path, time in zip(tqdm(scene_paths, desc='Importing from USD', unit='gaussiancloud', disable=silence_tqdm), times, strict=True):
            # TODO(cfujitsang): need to include the scene_path somewhere in the outputs
            scene_output = {
                'positions': [],
                'orientations': [],
                'scales': [],
                'opacities': [],
                'sh_coeff': []
            }
            prim = stage.GetPrimAtPath(scene_path)

            def _traverse(cur_prim, time):
                assert cur_prim, f'The prim at {scene_path} does not exist.'
                if cur_prim.GetTypeName() == "ParticleField3DGaussianSplat":
                    prim_output = _get_gaussiancloud(cur_prim, time)
                    for attr_name in ['positions', 'orientations', 'scales', 'opacities', 'sh_coeff']:
                        scene_output[attr_name].append(prim_output[attr_name])
                for child in cur_prim.GetChildren():
                    _traverse(child, time)

            _traverse(prim, time)
            for attr_name in ['positions', 'orientations', 'scales', 'opacities', 'sh_coeff']:
                scene_output[attr_name] = torch.cat(scene_output[attr_name], dim=0)
            output.append(scene_output)
    finally:
        del stage, file_path_or_stage
    return output

def import_all_gaussianclouds(file_path_or_stage, time=None):
    r"""Import all the gaussians from Usd file.

    Args:
        file_path_or_stage (Usd.Stage or str):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        time (convertible to float, optional):
            Positive integer defining the time at which the supplied parameters
            correspond to.

    Returns:
        (A dict of dict):
            Where the key is the path to each gaussian and the value dict represent a gaussian cloud with the following fields:

            * **positions** (torch.Tensor):
                Position of each gaussian, of shape :math:`(\text{num_gaussians}, 3)`.
            * **orientations** (torch.Tensor):
                Orientation of each gaussian as quaternions,
                of shape :math:`(\text{num_gaussians}, 4)`.
            * **scales** (torch.Tensor):
                Scale of each gaussian, of shape :math:`(\text{num_gaussians}, 3)`.
            * **opacities** (torch.Tensor):
                Opacity of each gaussian, of shape :math:`(\text{num_gaussians})`.
            * **sh_coeff** (torch.Tensor):
                Spherical harmonics coefficients of each gaussian,
                of shape :math:`(\text{num_gaussians}, (\text{num_degrees} + 1)^2, 3)`.
    """

    stage = _get_stage_from_maybe_file(file_path_or_stage)
    try:
        scene_paths = get_gaussiancloud_scene_paths(stage)
        gaussianclouds = import_gaussianclouds(stage, scene_paths, time)
        output = {str(scene_path): gaussiancloud for scene_path, gaussiancloud in zip(scene_paths, gaussianclouds, strict=True)}
    finally:
        del stage, file_path_or_stage
    return output

def get_gaussiancloud_scene_paths(file_path_or_stage):
    r"""Returns all gaussian cloud scene paths contained in specified file.
    Assuming ParticleField3DGaussianSplat.

    Args:
        file_path_or_stage (str or Usd.Stage):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.

    Returns:
        (list of str): List of filtered scene paths.
    """
    stage = _get_stage_from_maybe_file(file_path_or_stage)
    try:
        gaussians_paths = get_scene_paths(stage, prim_types=['ParticleField3DGaussianSplat'])
    finally:
        del stage, file_path_or_stage
    return gaussians_paths

def add_gaussiancloud(stage, scene_path, positions, orientations, scales,
                      opacities, sh_coeff, time=None, overwrite=False):
    r"""Add a gaussiancloud as a ParticleField3DGaussianSplat to an existing USD stage.

    Create a pointcloud represented by point instances of a sphere centered
    at each point coordinate.
    The stage is modified but not saved to disk.

    Args:
        stage (Usd.Stage): Stage onto which to add the pointcloud.
        scene_path (str):
            Absolute path of gaussian cloud within the USD file scene.
            Must be a valid ``Sdf.Path``.
        positions (torch.Tensor):
            Position of each gaussian, of shape :math:`(\text{num_gaussians}, 3)`.
        orientations (torch.Tensor):
            Orientation of each gaussian as quaternions,
            of shape :math:`(\text{num_gaussians}, 4)`.
        scales (torch.Tensor):
            Scale of each gaussian, of shape :math:`(\text{num_gaussians}, 3)`.
        opacities (torch.Tensor):
            Opacity of each gaussian, of shape :math:`(\text{num_gaussians})`.
        sh_coeff (torch.Tensor):
            Spherical harmonics coefficients of each gaussian,
            of shape :math:`(\text{num_gaussians}, (\text{num_degrees} + 1)^2, 3)`.
        time (convertible to float, optional):
            Positive integer defining the time at which the supplied parameters
            correspond to.
        overwrite (bool): If True, replace existing prim at scene_path. If False (default),
            raise ValueError when a ParticleField3DGaussianSplat already exists there.

    Returns:
        (Usd.Prim): The generated gaussian Prim.
    """
    try:
        if time is None:
            time = Usd.TimeCode.Default()

        existing = stage.GetPrimAtPath(scene_path)
        if existing:
            if existing.GetTypeName() != "ParticleField3DGaussianSplat" or not overwrite:
                raise ValueError(
                    f"Prim already exists at {scene_path}; use overwrite=True to replace."
                )
        gaussian = UsdVol.ParticleField3DGaussianSplat.Define(stage, scene_path)
        # positions
        assert torch.is_tensor(positions)
        assert torch.is_floating_point(positions)
        check_tensor(positions, shape=(None, 3))
        if positions.dtype == torch.half:
            gaussian.GetPositionshAttr().Set(positions.detach().cpu().numpy(), time=time)
        else:
            gaussian.GetPositionsAttr().Set(positions.detach().float().cpu().numpy(), time=time)
        # orientations
        assert torch.is_tensor(orientations)
        assert torch.is_floating_point(orientations)
        check_tensor(orientations, shape=(None, 4))
        if orientations.dtype == torch.half:
            gaussian.GetOrientationshAttr().Set(orientations.detach().cpu().numpy(), time=time)
        else:
            gaussian.GetOrientationsAttr().Set(orientations.detach().float().cpu().numpy(), time=time)
        # scales
        assert torch.is_tensor(scales)
        assert torch.is_floating_point(scales)
        check_tensor(scales, shape=(None, 3))
        if scales.dtype == torch.half:
            gaussian.GetScaleshAttr().Set(scales.detach().cpu().numpy(), time=time)
        else:
            gaussian.GetScalesAttr().Set(scales.detach().float().cpu().numpy(), time=time)
        # opacities
        assert torch.is_tensor(opacities)
        assert torch.is_floating_point(opacities)
        opacities = opacities.squeeze()
        check_tensor(opacities, shape=(None,))
        if opacities.dtype == torch.half:
            gaussian.GetOpacitieshAttr().Set(opacities.detach().cpu().numpy(), time=time)
        else:
            gaussian.GetOpacitiesAttr().Set(opacities.detach().float().cpu().numpy(), time=time)
        # sh_coeff
        assert torch.is_tensor(sh_coeff)
        assert torch.is_floating_point(sh_coeff)
        check_tensor(sh_coeff, shape=(None, None, 3))
        sh_degree = math.isqrt(sh_coeff.shape[1]) - 1
        assert (sh_degree + 1) ** 2 == sh_coeff.shape[1], \
            f"sh_coeff.shape[1] must be a perfect square (got {sh_coeff.shape[1]})"
        gaussian.GetRadianceSphericalHarmonicsDegreeAttr().Set(sh_degree)
        if sh_coeff.dtype == torch.half:
            gaussian.GetRadianceSphericalHarmonicsCoefficientshAttr().Set(
                sh_coeff.detach().cpu().numpy().reshape(-1, 3), time)
        else:
            gaussian.GetRadianceSphericalHarmonicsCoefficientsAttr().Set(
                sh_coeff.detach().float().cpu().numpy().reshape(-1, 3), time)
    finally:
        del stage
    return gaussian.GetPrim()

def export_gaussiancloud(file_path, scene_path='/World/Gaussians/gaussian_0',
                         positions=None, orientations=None, scales=None, opacities=None,
                         sh_coeff=None, up_axis='Y', time=None, overwrite=False):
    r"""Export a single gaussian cloud to a USD scene.

    The gaussian cloud will be added to the USD stage and represented by a
    ParticleField3DGaussianSplat. The stage is then saved to disk.

    Args:
        file_path (str): Path to usd file (\*.usd, \*.usda).
        scene_path (str, optional):
            Absolute path of gaussian cloud within the USD file scene.
            Must be a valid Sdf.Path. If no path is provided, a default path is used.
        positions (torch.Tensor):
            Position of each gaussian, of shape :math:`(\text{num_gaussians}, 3)`.
        orientations (torch.Tensor):
            Orientation of each gaussian as quaternions,
            of shape :math:`(\text{num_gaussians}, 4)`.
        scales (torch.Tensor):
            Scale of each gaussian, of shape :math:`(\text{num_gaussians}, 3)`.
        opacities (torch.Tensor):
            Opacity of each gaussian, of shape :math:`(\text{num_gaussians})`.
        sh_coeff (torch.Tensor):
            Spherical harmonics coefficients of each gaussian,
            of shape :math:`(\text{num_gaussians}, (\text{num_degrees} + 1)^2, 3)`.
        up_axis (str, optional):
            Axis to use for up direction. Must be one of 'X', 'Y', 'Z'. Defaults to 'Y'.
        time (convertible to float, optional):
            Positive integer defining the time at which the supplied parameters
            correspond to.
        overwrite (bool): If True, overwrite existing .usda. If False (default) raise an error if files already exists.

    """
    assert isinstance(scene_path, str)
    if positions is None:
        raise ValueError("positions must be provided")
    if orientations is None:
        raise ValueError("orientations must be provided")
    if scales is None:
        raise ValueError("scales must be provided")
    if opacities is None:
        raise ValueError("opacities must be provided")
    if sh_coeff is None:
        raise ValueError("sh_coeff must be provided")
    if time is None:
        time = Usd.TimeCode.Default()
    if os.path.exists(file_path) and not overwrite:
        raise FileExistsError(f"{file_path} already exists; to overwrite whole file use 'overwrite' argument;" +
                              " to add gaussiancloud to existing usd, use 'add_gaussiancloud' instead.")
    stage = create_stage(file_path, up_axis)
    try:
        add_gaussiancloud(stage, scene_path, positions, orientations, scales,
                          opacities, sh_coeff, time, overwrite)
        stage.Save()
    finally:
        del stage
