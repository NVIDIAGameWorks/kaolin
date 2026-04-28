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

import os.path
from collections import namedtuple
import math
import numpy as np
import torch

from plyfile import PlyData, PlyElement

from kaolin.rep import GaussianSplatModel

__all__ = [
    'import_gaussiancloud',
    'export_gaussiancloud',
]

def import_gaussiancloud(filename: str, device='cpu', dtype=torch.float32, #sh_degree=None,
                         apply_activations=True,
                         scale_activation=torch.exp,
                         rotation_activation=torch.nn.functional.normalize,
                         density_activation=torch.sigmoid):
    """
    Import a 3D Gaussian Splat cloud from a PLY file.
    
    Args:
        filename (str): path to the PLY file.
        device (str, optional): device to load the data onto. Defaults to 'cpu'.
        dtype (torch.dtype, optional): data type to load the data as. Defaults to torch.float32.
        apply_activations (bool, optional): whether to apply the activations to the data. Defaults to True.
        scale_activation (callable, optional): activation function to apply to the scale. Defaults to torch.exp.
        rotation_activation (callable, optional): activation function to apply to the rotation. Defaults to torch.nn.functional.normalize.
        density_activation (callable, optional): activation function to apply to the density. Defaults to torch.sigmoid.

    Returns:
        (GaussianSplatModel): A single gaussian cloud object.
    """
    # Source: nvidia 3dgrut
    # https://github.com/nv-tlabs/3dgrut/blob/3f00f3242891907ccba5b3c558ec1e870498f67e/threedgrut/model/model.py#L671
    plydata = PlyData.read(filename)

    mogt_pos = np.stack((np.asarray(plydata.elements[0]["x"]),
                         np.asarray(plydata.elements[0]["y"]),
                         np.asarray(plydata.elements[0]["z"])), axis=1)
    mogt_densities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    num_gaussians = mogt_pos.shape[0]
    mogt_albedo = np.zeros((num_gaussians, 3, 1))
    mogt_albedo[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    mogt_albedo[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    mogt_albedo[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))

    sh_degree = GaussianSplatModel.compute_sh_degree(len(extra_f_names) // 3 + 1)

    num_speculars = (sh_degree + 1) ** 2 - 1
    mogt_specular = np.zeros((num_gaussians, len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        mogt_specular[:, idx] = np.asarray(plydata.elements[0][attr_name])
    mogt_specular = mogt_specular.reshape((num_gaussians, 3, num_speculars))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    mogt_scales = np.zeros((num_gaussians, len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        mogt_scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    mogt_rotation = np.zeros((num_gaussians, len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        mogt_rotation[:, idx] = np.asarray(plydata.elements[0][attr_name])


    tensor_kwargs = {'dtype': dtype, 'device': device}
    densities = torch.tensor(mogt_densities, **tensor_kwargs)
    scales = torch.tensor(mogt_scales, **tensor_kwargs)
    rotations = torch.tensor(mogt_rotation, **tensor_kwargs)
    if apply_activations:
        densities = density_activation(densities)
        scales = scale_activation(scales)
        rotations = rotation_activation(rotations)

    sh_coeff = torch.cat([torch.tensor(mogt_albedo, **tensor_kwargs).transpose(1, 2),
                        torch.tensor(mogt_specular, **tensor_kwargs).transpose(1, 2)], dim=1)

    kwargs = {
        'positions': torch.tensor(mogt_pos, **tensor_kwargs),
        'orientations': rotations,
        'scales': scales,
        'opacities': densities.squeeze(-1),
        'sh_coeff': sh_coeff
    }
    return GaussianSplatModel(**kwargs)


# TODO: should design Gaussian class such that can do the following:
# cloud = GaussianCloud(...)
# export_gaussiancloud('/tmp/cloud.ply', **cloud.as_dict())
def export_gaussiancloud(file_path, positions, orientations, scales, opacities,
                         sh_coeff, sh_degree=None, overwrite=False):
    """Write a 3D Gaussian splatting-style PLY (``f_dc_*``, ``f_rest_*``, ``opacity``, ``scale_*``, ``rot_*``).

    Values are stored in the same raw space expected by :func:`import_gaussiancloud` when
    ``apply_activations=True`` (log-density, log-scale, unnormalized quaternion components).

    Args:
        file_path (str): path to the PLY file.
        positions (torch.Tensor): position of each gaussian, of shape :math:`(\text{num_gaussians}, 3)`.
        orientations (torch.Tensor): orientation of each gaussian as quaternions, as :math:`(w, x, y, z)`,
            of shape :math:`(\text{num_gaussians}, 4)`.
        scales (torch.Tensor): scale of each gaussian, of shape :math:`(\text{num_gaussians}, 3)`.
        opacities (torch.Tensor): opacity of each gaussian, of shape :math:`(\text{num_gaussians})`.
        sh_coeff (torch.Tensor): spherical harmonics coefficients of each gaussian,
            of shape :math:`(\text{num_gaussians}, (\text{num_degrees} + 1)^2, 3)`.
        sh_degree (int): optionally pass for sanity checking, otherwise will guess the value
        overwrite (bool, optional): whether to overwrite the file if it already exists. Defaults to False.

    Raises:
        RuntimeError: if the file already exists and overwrite is False.
    """
    if not overwrite and os.path.exists(file_path):
        raise RuntimeError(f'Cannot overwrite: {file_path}')

    n, num_sh, three = sh_coeff.shape
    if three != 3:
        raise ValueError('sh_coeff last dim must be 3 (RGB)')

    if sh_degree is None:
        sh_degree = GaussianSplatModel.compute_sh_degree(num_sh)
    expected_num_features = GaussianSplatModel.compute_num_sh_coeff(sh_degree)
    if expected_num_features != num_sh:
        raise ValueError(f'sh_coeff.shape[1]={num_sh} must be {expected_num_features} for sh degree {sh_degree}')

    # Inverse of import_gaussiancloud activations (sigmoid, exp, normalize).
    op_t = opacities.detach().float().view(n, -1).squeeze(-1)
    op_t = torch.clamp(op_t, 1e-6, 1.0 - 1e-6)
    raw_opacity = torch.logit(op_t)

    raw_scale = torch.log(scales.detach().float().clamp_min(1e-12))

    # Match import: mogt_specular (n, 3 * num_speculars) column-major for reshape (n, 3, ns).
    rest = sh_coeff[:, 1:, :].detach().float().cpu().numpy()
    rest_n3ns = np.transpose(rest, (0, 2, 1))
    f_rest_flat = rest_n3ns.reshape(n, -1)

    f_dc = sh_coeff[:, 0, :].detach().float().cpu().numpy()
    xyz = positions.detach().float().cpu().numpy()
    normals = np.zeros_like(xyz)
    opacities_np = raw_opacity.detach().cpu().numpy().reshape(n, 1)
    scale_np = raw_scale.detach().cpu().numpy()
    rotation_np = orientations.detach().float().cpu().numpy()

    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(3):
        l.append(f'f_dc_{i}')
    for i in range(f_rest_flat.shape[1]):
        l.append(f'f_rest_{i}')
    l.append('opacity')
    for i in range(scale_np.shape[1]):
        l.append(f'scale_{i}')
    for i in range(rotation_np.shape[1]):
        l.append(f'rot_{i}')

    dtype_full = [(attribute, 'f4') for attribute in l]
    elements = np.empty(n, dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest_flat, opacities_np, scale_np, rotation_np), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(file_path)
