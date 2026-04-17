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

import os
import math
import re
from tqdm import tqdm

import numpy as np
import torch

from kaolin.rep import GaussianSplatModel

from pxr import Usd, UsdGeom, UsdVol, Gf

from .utils import _get_stage_from_maybe_file, get_scene_paths, create_stage
from .transform import set_local_to_world_transform, get_local_to_world_transform
from kaolin.utils.testing import check_tensor


__all__ = [
    'add_gaussiancloud',
    'export_gaussiancloud',
    'get_gaussiancloud_scene_paths',
    'import_gaussianclouds',
    'import_gaussiancloud',
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
    output['orientations'] = torch.cat([output['orientations'][..., -1:], output['orientations'][..., :-1]], dim=1)
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
    return output    

def import_gaussianclouds(file_path_or_stage, scene_paths=None, times=None):
    r"""Import gaussians from Usd file.

    Args:
        file_path_or_stage (Usd.Stage or str):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        scene_paths (list of str):
            Scene paths within the USD file indicating which primitive(s) to import.
        times (list of Float or Float, optional):
            Positive integer defining the time at which the supplied parameters
            correspond to.

    Returns:
        (dict of str to GaussianSplatModel):
            Dictionary mapping each scene path to a :class:`~kaolin.rep.GaussianSplatModel`.
            Each model's ``transform`` is set to the computed local-to-world matrix of shape
            :math:`(4, 4)`, or ``None`` if the prim has an identity transform.
    """
    stage = _get_stage_from_maybe_file(file_path_or_stage)
    try:
        scene_paths = get_gaussiancloud_scene_paths(stage) if scene_paths is None else scene_paths
        if times is None:
            times = [Usd.TimeCode.Default()] * len(scene_paths)
        elif not isinstance(times, list):
            times = [times] * len(scene_paths)

        silence_tqdm = len(scene_paths) < 10
        output = {}
        for scene_path, time in zip(tqdm(scene_paths, desc='Importing from USD', unit='gaussiancloud', disable=silence_tqdm), times, strict=True):
            prim = stage.GetPrimAtPath(scene_path)
            assert prim.GetTypeName() == "ParticleField3DGaussianSplat", f"Prim at {scene_path} is not a ParticleField3DGaussianSplat"
            attrs = _get_gaussiancloud(prim, time)
            tfm = get_local_to_world_transform(stage, prim, time)
            tfm = tfm.float() if tfm is not None else None
            output[scene_path] = GaussianSplatModel(**attrs, transform=tfm, strict_checks=False)
    finally:
        del stage, file_path_or_stage
    return output

def import_gaussiancloud(file_path_or_stage, root_path=None, time=None):
    r"""Import all gaussianclouds from a USD file, apply their transforms, and merge into a single world-space cloud.

    Each prim's ``local_to_world`` transform (if not identity) is applied to positions,
    orientations, scales, and spherical-harmonics coefficients before merging.
    If clouds have different numbers of SH coefficients the higher-degree coefficients
    are truncated to the minimum present across all clouds.

    Args:
        file_path_or_stage (Usd.Stage or str):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        root_path (str, optional):
            If specified, only import gaussians under this scene path prefix.
            Default: import all gaussians in the file.
        time (convertible to float, optional):
            Positive integer defining the time at which the supplied parameters correspond to.

    Returns:
        (GaussianSplatModel):
            A single merged gaussian cloud object, or ``None`` if no gaussian clouds are found.
    """
    scene_paths = get_gaussiancloud_scene_paths(file_path_or_stage, root_path=root_path)
    clouds = import_gaussianclouds(file_path_or_stage, scene_paths, times=time)
    if len(clouds) == 0:
        return None

    # Note, transform is applied by default during GaussianSplatModel.cat
    return GaussianSplatModel.cat(list(clouds.values()))


def get_gaussiancloud_scene_paths(file_path_or_stage, root_path=None):
    r"""Returns all gaussian cloud scene paths contained in specified file.
    Assuming ParticleField3DGaussianSplat.

    Args:
        file_path_or_stage (str or Usd.Stage):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        root_path (str, optional): If specified, only return paths under this scene path prefix.

    Returns:
        (list of str): List of filtered scene paths.
    """
    scene_path_regex = f"{re.escape(root_path)}(/|$)" if root_path is not None else None
    stage = _get_stage_from_maybe_file(file_path_or_stage)
    try:
        gaussians_paths = get_scene_paths(stage, prim_types=['ParticleField3DGaussianSplat'],
                                          scene_path_regex=scene_path_regex)
    finally:
        del stage, file_path_or_stage
    return gaussians_paths

def add_gaussiancloud(stage, scene_path, positions, orientations, scales,
                      opacities, sh_coeff, local_to_world=None, time=None, overwrite=False):
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
            Orientation of each gaussian as quaternions, as :math:`(w, x, y, z)`,
            of shape :math:`(\text{num_gaussians}, 4)`.
        scales (torch.Tensor):
            Scale of each gaussian, of shape :math:`(\text{num_gaussians}, 3)`,
            post activation.
        opacities (torch.Tensor):
            Opacity of each gaussian, of shape :math:`(\text{num_gaussians})`,
            post activation.
        sh_coeff (torch.Tensor):
            Spherical harmonics coefficients of each gaussian,
            of shape :math:`(\text{num_gaussians}, (\text{num_degrees} + 1)^2, 3)`.
        time (convertible to float, optional):
            Positive integer defining the time at which the supplied parameters
            correspond to.
        overwrite (bool): If True, replace existing prim at scene_path. If False (default),
            raise ValueError when a ParticleField3DGaussianSplat already exists there.
        local_to_world (torch.Tensor):
            Local-to-world transform matrix of shape :math:`(4, 4)`,
            or ``None`` if the prim has an identity transform.

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
        orientations = torch.cat([orientations[:, 1:], orientations[:, :1]], dim=1)
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
        if opacities.ndim == 2:
            opacities = opacities.squeeze(-1)
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
        if local_to_world is not None:
            set_local_to_world_transform(stage, gaussian.GetPrim(), local_to_world, time)
    finally:
        del stage
    return gaussian.GetPrim()

def export_gaussiancloud(file_path, scene_path='/World/Gaussians/gaussian_0',
                         positions=None, orientations=None, scales=None, opacities=None,
                         sh_coeff=None, local_to_world=None, up_axis='Y', time=None, overwrite=False):
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
            Orientation of each gaussian as quaternions, as :math:`(w, x, y, z)`,
            of shape :math:`(\text{num_gaussians}, 4)`.
        scales (torch.Tensor):
            Scale of each gaussian, of shape :math:`(\text{num_gaussians}, 3)`.
        opacities (torch.Tensor):
            Opacity of each gaussian, of shape :math:`(\text{num_gaussians})`.
        sh_coeff (torch.Tensor):
            Spherical harmonics coefficients of each gaussian,
            of shape :math:`(\text{num_gaussians}, (\text{num_degrees} + 1)^2, 3)`.
        local_to_world (torch.Tensor):
            Local-to-world transform matrix of shape :math:`(4, 4)`,
            or ``None`` if the prim has an identity transform.
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
                          opacities, sh_coeff, local_to_world, time, overwrite)
        stage.Save()
    finally:
        del stage
