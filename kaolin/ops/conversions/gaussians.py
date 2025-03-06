# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

import torch
from kaolin import _C
import kaolin.render.spc as spc_render

__all__ = [
    'gs_to_voxelgrid'
]

def gs_to_voxelgrid(xyz, scales, rots, opacities, level, iso=11.345, tol=1. / 8., step=10):
    r"""Define a voxelgrid overlapping with a 3D Gaussians Splat. Opacity is integrated over multiple samples.

    .. note::
        This op is not differentiable

    Args:
        xyz (torch.cuda.FloatTensor):
            3D Volumetric Gaussians means, of shape :math:`(\text{num_gaussians, 3})`.
        scales (torch.cuda.FloatTensor):
            3D Volumetric Gaussians scales, of shape :math:`(\text{num_gaussians, 3})`.
        rots (torch.cuda.FloatTensor):
            3D Volumetric Gaussians rotations, of shape :math:`(\text{num_gaussians, 4})`.
        opacities (torch.cuda.FloatTensor):
            3D Volumetric Gaussians opacities, of shape :math:`(\text{num_gaussians})`.
        level (int): level at which to process, resolution will be at :math:`2^level`.
        iso (float):
            The isocontour value used to determine the surface of a Gaussian.
            Default: In the Gaussian Splat paper the 99th percentile value is advised (11.345).
        tol (float):
            Minimum allowable scale in the scale matrix of the :math:'(\Sigma = R S S^T R^T)'
            factorization of a Gaussian covariance matrix. This value is relative to the size
            of a voxel at the deepest level of SPC point hierarchy. This value is needed to
            ensure that :math:`(\Sigma)` is numerically invertable. Note: This issue does not
            arise in Gaussian Splatting since it is only necessary that the 2D projection of a
            Gaussian have an inverse. Default: 0.125.
        step (int): number of samples for opacity integration.

    Returns:
        (torch.cuda.ShortTensor, torch.cuda.LongTensor, torch.cuda.BoolTensor, torch.cuda.FloatTensor):

            - list the voxels coordinates, of shape :math:`(\text{num_voxels}, 3)`
            - the accumulated opacities, of shape :math:`(\text{num_voxels})`
    """
    xyz = xyz.contiguous()
    scales = scales.contiguous()
    rots = rots.contiguous()
    opacities = opacities.contiguous()
    point_per_intersection, gaus_id_per_intersection, pack_boundary, cov3d_invs = _C.ops.conversions.gs_to_spc_cuda(
        xyz,
        scales,
        rots,
        iso,
        tol,
        level
    )
    opacity_integral_per_intersection = _C.ops.conversions.integrate_gs_cuda(
        point_per_intersection.contiguous(),
        gaus_id_per_intersection.contiguous(),
        xyz,
        cov3d_invs.contiguous(),
        opacities,
        level,
        step
    )
    sum_vox_opacities = 1. - spc_render.prod_reduce((1. - opacity_integral_per_intersection).unsqueeze(-1), pack_boundary).squeeze(-1)
    points = point_per_intersection[pack_boundary]
    return points, sum_vox_opacities
