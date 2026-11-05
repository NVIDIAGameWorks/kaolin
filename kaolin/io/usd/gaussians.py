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

from pxr import Usd, UsdGeom, UsdVol, Gf, Sdf, Vt

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

_FEATURE_PREFIX = 'kaolin:features:'


def _validate_feature_name(name):
    if (not isinstance(name, str) or not Sdf.Path.IsValidIdentifier(name)
            or name == 'indices'):
        raise ValueError(
            f'Invalid feature name {name!r}: expected a USD identifier '
            'other than "indices"')


def _get_feature_width(primvar):
    """Validate the storage declaration and return its channel count."""
    width = primvar.GetElementSize()
    if (primvar.GetTypeName() != Sdf.ValueTypeNames.FloatArray
            or primvar.GetInterpolation() != UsdGeom.Tokens.vertex
            or width <= 0):
        raise ValueError(
            f'Invalid feature primvar {primvar.GetAttr().GetPath()}: expected '
            'float[], vertex interpolation, and positive elementSize')
    return width


def _validate_features(features, num_gaussians, prim=None, time=None):
    """Validate all features before authoring any stage data.

    Checks that:

    * ``features`` is a dict of valid names to floating ``(N, K)`` tensors.
    * If ``prim`` exists, a feature being updated keeps its channel count, and
      is not an indexed feature with time samples.
    * If ``prim`` exists, every feature kept on it (not passed in ``features``)
      still has one row per Gaussian at ``time``. This fails when an overwrite
      changes the Gaussian count, or when a value held from another time sample
      has a different count.
    """
    if features is None:
        features = {}
    elif not isinstance(features, dict):
        raise ValueError('features must be a dict of floating (N, K) tensors')
    api = UsdGeom.PrimvarsAPI(prim) if prim else None
    for name, tensor in features.items():
        _validate_feature_name(name)
        if not torch.is_tensor(tensor) or not torch.is_floating_point(tensor):
            raise ValueError(f'features[{name!r}] must be a floating tensor')
        if (tensor.ndim != 2 or tensor.shape[0] != num_gaussians
                or tensor.shape[1] <= 0):
            raise ValueError(
                f'features[{name!r}] must have shape (N={num_gaussians}, K) '
                f'with K > 0, got {tuple(tensor.shape)}')
        if api is not None:
            primvar = api.GetPrimvar(_FEATURE_PREFIX + name)
            if primvar and _get_feature_width(primvar) != tensor.shape[1]:
                raise ValueError(
                    f'Cannot change channel count of feature {name!r}')
            if primvar and primvar.IsIndexed():
                if (not Usd.TimeCode(time).IsDefault()
                        or primvar.GetAttr().GetNumTimeSamples()
                        or primvar.GetIndicesAttr().GetNumTimeSamples()):
                    # Blocking indices affects every sample, not just this time.
                    raise ValueError(
                        f'Cannot update indexed feature {name!r} with time '
                        'samples; flatten the feature across all times first')
    if api is None:
        return
    for primvar in api.GetAuthoredPrimvars():
        name = str(primvar.GetPrimvarName())
        if not name.startswith(_FEATURE_PREFIX):
            continue
        name = name[len(_FEATURE_PREFIX):]
        if name in features:
            continue
        values = primvar.ComputeFlattened(time) if primvar.IsIndexed() else primvar.Get(time)
        width = primvar.GetElementSize()
        if values is None or width <= 0:
            continue
        if len(values) != num_gaussians * width:
            raise ValueError(
                f'Existing feature {name!r} holds {len(values) // width} entries '
                f'at this time but {num_gaussians} Gaussians are being written; '
                'pass it again in features')


def _write_features(prim, features, time):
    """Author validated features as dense, Gaussian-major float32 arrays."""
    api = UsdGeom.PrimvarsAPI(prim)
    for name, tensor in (features or {}).items():
        values = tensor.detach().float().cpu().contiguous().numpy().reshape(-1)
        primvar = api.CreatePrimvar(
            _FEATURE_PREFIX + name, Sdf.ValueTypeNames.FloatArray,
            UsdGeom.Tokens.vertex, tensor.shape[1])
        # Dense updates must also override indices authored in weaker layers.
        primvar.BlockIndices()
        primvar.Set(Vt.FloatArray.FromNumpy(values), time)


def _read_features(prim, num_gaussians, time):
    """Read only Kaolin feature primvars, resolving optional USD indexing."""
    features = {}
    for primvar in UsdGeom.PrimvarsAPI(prim).GetAuthoredPrimvars():
        name = str(primvar.GetPrimvarName())
        if not name.startswith(_FEATURE_PREFIX):
            continue
        raw = primvar.Get(time)
        if raw is None:
            continue
        name = name[len(_FEATURE_PREFIX):]
        _validate_feature_name(name)
        width = _get_feature_width(primvar)
        values = primvar.ComputeFlattened(time) if primvar.IsIndexed() else raw
        if values is None or len(values) != num_gaussians * width:
            raise ValueError(
                f'Invalid values for feature {name!r}: expected '
                f'{num_gaussians * width} scalars after resolving indices')
        features[name] = torch.from_numpy(
            np.array(values, dtype=np.float32, copy=True).reshape(num_gaussians, width))
    return features or None


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
    output['features'] = _read_features(prim, len(output['positions']), time)
    return output

def import_gaussianclouds(file_path_or_stage, scene_paths=None, times=None, return_list=True):
    r"""Import gaussians from Usd file.

    Args:
        file_path_or_stage (Usd.Stage or str):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        scene_paths (list of str):
            Scene paths within the USD file indicating which primitive(s) to import.
        times (list of Float or Float, optional):
            Positive integer defining the time at which the supplied parameters
            correspond to.
        return_list (bool, optional): If ``True`` (default), return a ``list`` of
            :class:`~kaolin.rep.GaussianSplatModel` in the order of ``scene_paths``. If ``False``,
            return a ``dict`` keyed by scene path.

    Returns:
        (list or dict of GaussianSplatModel):
            Either a list of :class:`~kaolin.rep.GaussianSplatModel` (when ``return_list=True``,
            the default), or a ``dict[str, GaussianSplatModel]`` keyed by scene path (when
            ``return_list=False``). Each model's ``transform`` is set to the computed
            local-to-world matrix of shape :math:`(4, 4)`, or ``None`` if the prim has an
            identity transform. Kaolin feature primvars are restored as a dictionary
            of float32 ``(N, K)`` tensors in ``features``, or ``None`` when absent.
    """
    stage = _get_stage_from_maybe_file(file_path_or_stage)
    try:
        scene_paths = get_gaussiancloud_scene_paths(stage) if scene_paths is None else scene_paths
        if times is None:
            times = [Usd.TimeCode.Default()] * len(scene_paths)
        elif not isinstance(times, list):
            times = [times] * len(scene_paths)
        else:
            if len(times) != len(scene_paths):
                raise ValueError(
                    f"Length of times ({len(times)}) must match length of scene_paths ({len(scene_paths)})"
                )

        silence_tqdm = len(scene_paths) < 10
        output = {}
        for scene_path, time in zip(tqdm(scene_paths, desc='Importing from USD', unit='gaussiancloud', disable=silence_tqdm), times):
            prim = stage.GetPrimAtPath(scene_path)
            assert prim.GetTypeName() == "ParticleField3DGaussianSplat", f"Prim at {scene_path} is not a ParticleField3DGaussianSplat"
            attrs = _get_gaussiancloud(prim, time)
            tfm = get_local_to_world_transform(stage, prim, time)
            tfm = tfm.float() if tfm is not None else None
            output[str(scene_path)] = GaussianSplatModel(**attrs, transform=tfm, strict_checks=False)
    finally:
        del stage, file_path_or_stage
    if return_list:
        return list(output.values())
    return output

def import_gaussiancloud(file_path_or_stage, scene_path=None, time=None):
    r"""Import all gaussianclouds from a USD file, apply their transforms, and merge into a single world-space cloud.

    Each prim's ``local_to_world`` transform (if not identity) is applied to positions,
    orientations, scales, and spherical-harmonics coefficients before merging.
    If clouds have different numbers of SH coefficients the higher-degree coefficients
    are truncated to the minimum present across all clouds.
    Features are concatenated without spatial transformation. All clouds must have
    matching feature keys and channel counts, or all have no features. Use
    :func:`import_gaussianclouds` to retain clouds with heterogeneous features.

    Args:
        file_path_or_stage (Usd.Stage or str):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        scene_path (str, optional):
            If specified, only import gaussians under this scene path prefix.
            Default: import all gaussians in the file.
        time (convertible to float, optional):
            Positive integer defining the time at which the supplied parameters correspond to.

    Returns:
        (GaussianSplatModel):
            A single merged gaussian cloud object, or ``None`` if no gaussian clouds are found.
    """
    scene_paths = get_gaussiancloud_scene_paths(file_path_or_stage, scene_path=scene_path)
    clouds = import_gaussianclouds(file_path_or_stage, scene_paths, times=time, return_list=True)
    if len(clouds) == 0:
        return None

    # Note, transform is applied by default during GaussianSplatModel.cat
    return GaussianSplatModel.cat(clouds)


def get_gaussiancloud_scene_paths(file_path_or_stage, scene_path=None):
    r"""Returns all gaussian cloud scene paths contained in specified file.
    Assuming ParticleField3DGaussianSplat.

    Args:
        file_path_or_stage (str or Usd.Stage):
            Path to usd file (\*.usd, \*.usda) or :class:`Usd.Stage`.
        scene_path (str, optional): If specified, only return paths under this scene path prefix.

    Returns:
        (list of str): List of filtered scene paths.
    """
    scene_path_regex = f"{re.escape(scene_path)}(/|$)" if scene_path is not None else None
    stage = _get_stage_from_maybe_file(file_path_or_stage)
    try:
        gaussians_paths = get_scene_paths(stage, prim_types=['ParticleField3DGaussianSplat'],
                                          scene_path_regex=scene_path_regex)
    finally:
        del stage, file_path_or_stage
    return gaussians_paths

def add_gaussiancloud(stage, scene_path, positions, orientations, scales,
                      opacities, sh_coeff, local_to_world=None, time=None, overwrite=False,
                      *, features=None):
    r"""Add a gaussiancloud as a ParticleField3DGaussianSplat to an existing USD stage.

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
        overwrite (bool): If True, update existing prim at scene_path. If False (default),
            raise ValueError when a ParticleField3DGaussianSplat already exists there.
        local_to_world (torch.Tensor):
            Local-to-world transform matrix of shape :math:`(4, 4)`,
            or ``None`` if the prim has an identity transform.
        features (dict, optional): Named floating tensors of shape ``(N, K)``,
            with ``K > 0``. Stored as float32 ``primvars:kaolin:features:<name>``
            arrays with vertex interpolation and ``elementSize=K``. Names must be
            valid USD identifiers other than ``indices``. Updates preserve omitted
            features; ``None`` or an empty dict leaves existing features unchanged.
            Existing feature declarations and channel counts must match. Callers
            must keep retained features aligned with Gaussian count and order.
            Indexed features can only be replaced at default time when neither
            their values nor indices have time samples.
            Features are stored as given and are never transformed. They can hold
            geometric data such as surface normals, but those values stay fixed:
            they are not recomputed when ``local_to_world`` is set or when the
            cloud is transformed on import.

    Raises:
        ValueError: If a feature name, tensor, or existing declaration is invalid.

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
        _validate_features(features, positions.shape[0], existing, time)
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
        _write_features(gaussian.GetPrim(), features, time)
    finally:
        del stage
    return gaussian.GetPrim()

def export_gaussiancloud(file_path, scene_path='/World/Gaussians/gaussian_0',
                         positions=None, orientations=None, scales=None, opacities=None,
                         sh_coeff=None, local_to_world=None, up_axis='Y', time=None, overwrite=False,
                         *, features=None):
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
        features (dict, optional): Named floating ``(N, K)`` tensors, serialized as
            float32 per-Gaussian primvars. See :func:`add_gaussiancloud` for naming
            and shape requirements. Defaults to ``None``.

    Raises:
        ValueError: If a feature name or tensor is invalid.

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
    _validate_features(features, positions.shape[0])
    stage = create_stage(file_path, up_axis)
    try:
        add_gaussiancloud(stage, scene_path, positions, orientations, scales,
                          opacities, sh_coeff, local_to_world, time, overwrite,
                          features=features)
        stage.Save()
    finally:
        del stage
