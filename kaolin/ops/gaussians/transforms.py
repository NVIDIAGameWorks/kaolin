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

import logging
import torch
import math
from kaolin.math import quat as quat_ops

__all__ = [
    'transform_gaussians',
    'transform_sh'
]

logger = logging.getLogger(__name__)


def _transform_xyz(xyz: torch.Tensor, transform: torch.Tensor):
    """Apply 4x4 transform to positions. Supports single transform (4x4) or batch (Nx4x4)."""
    if len(transform.shape) == 2:  # single transform for all the gaussians
        transform = transform.unsqueeze(0)
    res = (transform[..., :3, :3] @ xyz[:, :, None] + transform[..., :3, 3:]).squeeze(-1)
    return res

def _transform_rot(rot: torch.Tensor, transform: torch.Tensor):
    """Apply rotation to quaternions. Input rot uses gsplats convention [w,x,y,z]; output same."""
    if len(transform.shape) == 2:  # single transform for all the gaussians
        transform = transform.unsqueeze(0)

    rot_quat = quat_ops.quat_from_rot33(transform[..., :3, :3])
    rot_unit = rot / torch.linalg.norm(rot, dim=-1).unsqueeze(-1)

    # Note: gsplats use Hamiltonion convention [real, imag], whereas Kaolin uses the other convention[imag, real]
    rot_unit = torch.cat([rot_unit[:, 1:], rot_unit[:, :1]], dim=-1)

    result = quat_ops.quat_mul(rot_quat, rot_unit)
    result = torch.cat([result[:, 3:], result[:, :3]], dim=-1)
    return result

def _decompose_4x4_transform(transform):
    """ Decompose 4x4 transform into translation, rotation, scale.
    Returns:
        translation, rotation, scale
    """
    translation = transform[..., :3, 3:]
    scale = torch.linalg.norm(transform[..., :3, :3], dim=-2)
    rotation = transform[..., :3, :3] / scale.unsqueeze(-2)

    return translation, rotation, scale

def transform_gaussians(xyz, rotations, raw_scales, transform, shs_feat=None, use_log_scales=False):
    r"""Apply a 4x4 affine transform to gaussian positions, rotations, and scales.

    Transforms are applied as T @ pt for positions. Rotations use gsplats convention
    [w, x, y, z] (Hamiltonian, real-first). Scale is decomposed from the transform
    and applied to raw_scales.

    Args:
        xyz (torch.Tensor): Positions of shape :math:`(N, 3)`.
        rotations (torch.Tensor): Unit quaternions [w,x,y,z], shape :math:`(N, 4)`.
        raw_scales (torch.Tensor): Scale per gaussian, shape :math:`(N, 3)`.
        transform (torch.Tensor): Affine transform, shape :math:`(4, 4)` or :math:`(N, 4, 4)`.
            Single transform is broadcast to all gaussians.
        shs_feat (torch.Tensor): SH features of shape :math:`(N, \text{num_coeffs}, 3)`. Default: None.
        use_log_scales (bool): If True, use log-based scaling for scales. Default: False.

    Returns:
        tuple: (new_xyz, new_rotations, new_scales, Optional[new_shs_feat]) with same shapes as inputs.
    """
    assert xyz.dtype == rotations.dtype == raw_scales.dtype == transform.dtype, f"xyz, rotations, raw_scales, and transform must have the same dtype, got xyz.dtype: {xyz.dtype}, rotations.dtype: {rotations.dtype}, raw_scales.dtype: {raw_scales.dtype}, transform.dtype: {transform.dtype}"
    assert xyz.device == rotations.device == raw_scales.device == transform.device, f"xyz, rotations, raw_scales, and transform must be on the same device, got xyz.device: {xyz.device}, rotations.device: {rotations.device}, raw_scales.device: {raw_scales.device}, transform.device: {transform.device}"
    if shs_feat is not None:
        assert shs_feat.dtype == xyz.dtype, f"shs_feat must have the same dtype as the other inputs, got shs_feat.dtype: {shs_feat.dtype}, xyz.dtype: {xyz.dtype}"
        assert shs_feat.device == xyz.device, f"shs_feat must be on the same device as the other inputs, got shs_feat.device: {shs_feat.device}, xyz.device: {xyz.device}"

    if len(transform.shape) == 2:  # single transform for all the gaussians
        transform = transform.unsqueeze(0)

    # transforms: n x 4 x 4, where 4 x 4 transform T is applied to pt as T @ pt.
    _, rotation, scale = _decompose_4x4_transform(transform)

    new_xyz = _transform_xyz(xyz, transform)
    new_rotations = _transform_rot(rotations, rotation)

    if raw_scales.dtype != scale.dtype:
        scale = scale.to(raw_scales.dtype)
    if not use_log_scales:
        new_scales = raw_scales * scale
    else:
        scaling_norm_factor = torch.log(scale) / raw_scales + 1
        new_scales = raw_scales * scaling_norm_factor

    if shs_feat is None:
        new_shs_feat = None
    else:
        new_shs_feat = transform_sh(shs_feat, rotation)

    return new_xyz, new_rotations, new_scales, new_shs_feat

# Q R Q^{-1} decomposes into permutation [1,2,0] + this sign pattern
_S_3DGS = [[1, -1, 1], [-1, 1, -1], [1, -1, 1]]

def transform_sh(sh_feat: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    r"""
    Rotate real SH coefficients (bands 1-3) in batch.

    Args:
        sh_feat (torch.Tensor): SH coefficients for bands l=1..3, RGB, of shape :math:`(N, \text{num_coeffs}, 3)`.
        R (torch.Tensor): rotation matrices (SO(3)), of shape :math:`(N, 3, 3)`.

    Returns:
        (torch.Tensor): rotated SH coefficients, of shape :math:`(N, \text{num_coeffs}, 3)`.
    """
    assert sh_feat.dtype == R.dtype, f"sh and R must have the same dtype, got sh.dtype: {sh_feat.dtype}, R.dtype: {R.dtype}"
    assert sh_feat.device == R.device, f"sh and R must be on the same device, got sh.device: {sh_feat.device}, R.device: {R.device}"
    num_coeffs = sh_feat.shape[1]
    degree = math.isqrt(num_coeffs + 1) - 1
    assert ((degree + 1) ** 2) - 1 == num_coeffs
    if degree > 3:
        raise NotImplementedError(f"transform_sh does not support degree > 3, got {degree}")
    assert sh_feat.shape[2] == 3, f"sh must have 3 channels, got sh.shape: {sh_feat.shape}"
    assert R.shape[0] == sh_feat.shape[0] or R.shape[0] == 1, f"R must have the same number of batches as sh, got R.shape: {R.shape}, sh.shape: {sh_feat.shape}"
    assert R.shape[1:] == (3, 3), f"R must be a rotation matrix, got R.shape: {R.shape}"

    out = torch.empty_like(sh_feat)

    # D^1 = Q R Q^{-1} = R[perm][:,perm] * S
    perm = [1, 2, 0]
    S = R.new_tensor(_S_3DGS)
    D1 = R[:, perm][:, :, perm] * S                        # (N, 3, 3)

    out[:, :3] = (D1 @ sh_feat[:, :3])

    if degree == 1:
        return out

    # Feed D^1 into recurrence → produces correct D^l directly
    # D2 = _apply_wigner_sparse(2, D1, D1)                   # (N, 5, 5)
    products, out_idx = _fused_apply_wigner_sparse_band2(D1)
    D2 = torch.zeros(out_idx.shape[0], 5 * 5, device=products.device, dtype=products.dtype)
    D2.scatter_add_(1, out_idx, products)
    D2 = D2.view(-1, 5, 5)
    out[:, 3:8] = (D2 @ sh_feat[:, 3:8])

    if degree == 2:
        return out

    # D3 = _apply_wigner_sparse(3, D1, D2)                   # (N, 7, 7)
    products, out_idx = _fused_apply_wigner_sparse_band3(D1, D2) # (N, 7, 7)
    D3 = torch.zeros(out_idx.shape[0], 7 * 7, device=products.device, dtype=products.dtype)
    D3.scatter_add_(1, out_idx, products)
    D3 = D3.view(-1, 7, 7)
    out[:, 8:15] = (D3 @ sh_feat[:, 8:15])
    if degree == 3:
        return out

# TODO(cfujitsang): torch compile generator for different l values?
# @lru_cache(maxsize=4)
# def _precompute_band(l: int):
#     lm1 = l - 1
#     coeffs, m_indices, n_indices = [], [], []
#     r_rows, r_cols, p_rows, p_cols = [], [], [], []

#     def _add(mi, ni, scalar, p_terms):
#         for (ri, rc, pr, pc, s) in p_terms:
#             coeffs.append(scalar * s)
#             m_indices.append(mi)
#             n_indices.append(ni)
#             r_rows.append(ri)
#             r_cols.append(rc)
#             p_rows.append(pr)
#             p_cols.append(pc)

#     def _p(i, a, b):
#         ri = i + 1
#         if b == l:
#             return [(ri, 2, a + lm1, 2 * lm1,  1.0),
#                     (ri, 0, a + lm1, 0,        -1.0)]
#         elif b == -l:
#             return [(ri, 2, a + lm1, 0,         1.0),
#                     (ri, 0, a + lm1, 2 * lm1,   1.0)]
#         else:
#             return [(ri, 1, a + lm1, b + lm1, 1.0)]

#     for m in range(-l, l + 1):
#         for n in range(-l, l + 1):
#             mi, ni, abs_m = m + l, n + l, abs(m)
#             d = float((l+n)*(l-n)) if abs(n) < l else float(2*l*(2*l-1))
#             if d == 0:
#                 continue

#             # u · U
#             num_u = float((l+m)*(l-m))
#             if num_u > 0:
#                 _add(mi, ni, math.sqrt(num_u / d), _p(0, m, n))

#             # v · V
#             if m == 0:
#                 inner = float(l * (l - 1))
#                 if inner > 0:
#                     _add(mi, ni, -0.5 * math.sqrt(2.0 * inner / d),
#                          _p(1, 1, n) + _p(-1, -1, n))
#             elif m > 0:
#                 num_v = float((l+m-1) * (l+m))
#                 if num_v > 0:
#                     v = 0.5 * math.sqrt(num_v / d)
#                     dm1 = 1.0 if m == 1 else 0.0
#                     s1, s2 = math.sqrt(1 + dm1), -(1 - dm1)
#                     terms = [(ri,rc,pr,pc,s*s1)
#                              for ri,rc,pr,pc,s in _p(1, m-1, n)]
#                     if abs(s2) > 1e-15:
#                         terms += [(ri,rc,pr,pc,s*s2)
#                                   for ri,rc,pr,pc,s in _p(-1, -m+1, n)]
#                     _add(mi, ni, v, terms)
#             else:
#                 num_v = float((l+abs_m-1) * (l+abs_m))
#                 if num_v > 0:
#                     v = 0.5 * math.sqrt(num_v / d)
#                     dm1 = 1.0 if m == -1 else 0.0
#                     s1, s2 = (1 - dm1), math.sqrt(1 + dm1)
#                     terms = []
#                     if abs(s1) > 1e-15:
#                         terms += [(ri,rc,pr,pc,s*s1)
#                                   for ri,rc,pr,pc,s in _p(1, m+1, n)]
#                     terms += [(ri,rc,pr,pc,s*s2)
#                               for ri,rc,pr,pc,s in _p(-1, -m-1, n)]
#                     _add(mi, ni, v, terms)

#             # w · W
#             if abs_m != 0:
#                 iw = float((l-abs_m-1) * (l-abs_m))
#                 if iw > 0:
#                     w = -0.5 * math.sqrt(iw / d)
#                     if m > 0:
#                         _add(mi, ni, w,
#                              _p(1, m+1, n) + _p(-1, -m-1, n))
#                     else:
#                         terms = _p(1, m-1, n)
#                         terms += [(ri,rc,pr,pc,-s)
#                                   for ri,rc,pr,pc,s in _p(-1, -m+1, n)]
#                         _add(mi, ni, w, terms)

#     return tuple(
#         torch.tensor(x, dtype=torch.float64 if i == 0 else torch.long)
#         for i, x in enumerate([coeffs, m_indices, n_indices,
#                                 r_rows, r_cols, p_rows, p_cols])
#     )

# def _apply_wigner_sparse(l: int, M1: torch.Tensor, Mprev: torch.Tensor) -> torch.Tensor:
#     N, dim = M1.shape[0], 2 * l + 1
#     dev, dt = M1.device, M1.dtype

#     coeff, mi, ni, rr, rc, pr, pc = _precompute_band(l)
#     coeff = coeff.to(device=dev, dtype=dt)
#     mi, ni = mi.to(dev), ni.to(dev)
#     rr, rc = rr.to(dev), rc.to(dev)
#     pr, pc = pr.to(dev), pc.to(dev)

#     T = coeff.shape[0]
#     products = coeff[None] * M1[:, rr, rc] * Mprev[:, pr, pc]
#     out_idx = (mi * dim + ni)[None].expand(N, T)

#     M = torch.zeros(N, dim * dim, device=dev, dtype=dt)
#     M.scatter_add_(1, out_idx, products)
#     return M.view(N, dim, dim)



@torch.compile
def _fused_apply_wigner_sparse_band2(M1: torch.Tensor) -> torch.Tensor:
    N = M1.shape[0]
    coeff = torch.tensor([ 0.5000,  0.5000,  0.5000,  0.5000,  1.0000,  1.0000,  0.8660,  0.8660,
         1.0000,  1.0000,  0.5000, -0.5000,  0.5000, -0.5000,  0.5000,  0.5000,
         0.5000,  0.5000,  1.0000,  1.0000,  0.8660,  0.8660,  1.0000,  1.0000,
         0.5000, -0.5000,  0.5000, -0.5000,  0.5774,  0.5774, -0.2887, -0.2887,
        -0.2887, -0.2887,  1.1547, -0.5774, -0.5774,  1.0000, -0.5000, -0.5000,
         1.1547, -0.5774, -0.5774,  0.5774, -0.5774, -0.2887,  0.2887, -0.2887,
         0.2887,  0.5000,  0.5000,  0.5000,  0.5000,  1.0000,  1.0000,  0.8660,
         0.8660,  1.0000,  1.0000,  0.5000, -0.5000,  0.5000, -0.5000,  0.5000,
         0.5000, -0.5000, -0.5000,  1.0000, -1.0000,  0.8660, -0.8660,  1.0000,
        -1.0000,  0.5000, -0.5000, -0.5000,  0.5000], dtype=M1.dtype, device=M1.device)
    rr = torch.tensor([2, 2, 0, 0, 2, 0, 2, 0, 2, 0, 2, 2, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0,
        1, 1, 0, 0, 1, 1, 2, 2, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 1, 2, 2, 0,
        0, 1, 1, 2, 2, 1, 2, 1, 2, 1, 2, 1, 1, 2, 2, 2, 2, 0, 0, 2, 0, 2, 0, 2,
        0, 2, 2, 0, 0], device=M1.device)
    rc = torch.tensor([2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 2, 0, 2, 0, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1,
        2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 2, 0, 2,
        0, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 2, 0, 2, 0, 2, 0, 2, 0, 1, 1, 1, 1, 1,
        1, 2, 0, 2, 0], device=M1.device)
    pr = torch.tensor([0, 0, 2, 2, 0, 2, 0, 2, 0, 2, 0, 0, 2, 2, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1,
        0, 0, 1, 1, 1, 1, 2, 2, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 1, 2, 2, 0,
        0, 2, 2, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 1, 2, 2, 0, 0, 2, 0, 2, 0, 2,
        0, 2, 2, 0, 0], device=M1.device)
    pc = torch.tensor([0, 2, 0, 2, 0, 0, 1, 1, 2, 2, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 1, 1, 2, 2,
        2, 0, 2, 0, 0, 2, 0, 2, 0, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0, 2, 0, 2,
        0, 0, 2, 0, 2, 0, 0, 1, 1, 2, 2, 2, 0, 2, 0, 0, 2, 0, 2, 0, 0, 1, 1, 2,
        2, 2, 0, 2, 0], device=M1.device)
    out_idx = torch.tensor([ 0,  0,  0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  4,  4,  5,  5,  5,  5,
         6,  6,  7,  7,  8,  8,  9,  9,  9,  9, 10, 10, 10, 10, 10, 10, 11, 11,
        11, 12, 12, 12, 13, 13, 13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 16,
        16, 17, 17, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 22, 22, 23,
        23, 24, 24, 24, 24], device=M1.device)
    products = coeff[None] * M1[:, rr, rc] * M1[:, pr, pc]
    return products, out_idx[None].expand(N, 77)


@torch.compile
def _fused_apply_wigner_sparse_band3(M1: torch.Tensor, Mprev: torch.Tensor) -> torch.Tensor:
    N = M1.shape[0]
    coeff = torch.tensor([ 0.5000,  0.5000,  0.5000,  0.5000,  1.2247,  1.2247,  0.9682,  0.9682,
         0.9129,  0.9129,  0.9682,  0.9682,  1.2247,  1.2247,  0.5000, -0.5000,
         0.5000, -0.5000,  0.4082,  0.4082,  0.4082,  0.4082,  0.4082,  0.4082,
         1.0000,  1.0000,  1.0000,  0.7906,  0.7906,  0.7906,  0.7454,  0.7454,
         0.7454,  0.7906,  0.7906,  0.7906,  1.0000,  1.0000,  1.0000,  0.4082,
        -0.4082,  0.4082, -0.4082,  0.4082, -0.4082,  0.5164,  0.5164,  0.4472,
         0.4472, -0.1291, -0.1291,  0.1291,  0.1291,  1.2649,  1.0954, -0.3162,
         0.3162,  1.0000,  0.8660, -0.2500,  0.2500,  0.9428,  0.8165, -0.2357,
         0.2357,  1.0000,  0.8660, -0.2500,  0.2500,  1.2649,  1.0954, -0.3162,
         0.3162,  0.5164, -0.5164,  0.4472, -0.4472, -0.1291,  0.1291,  0.1291,
        -0.1291,  0.5477,  0.5477, -0.3162, -0.3162, -0.3162, -0.3162,  1.3416,
        -0.7746, -0.7746,  1.0607, -0.6124, -0.6124,  1.0000, -0.5774, -0.5774,
         1.0607, -0.6124, -0.6124,  1.3416, -0.7746, -0.7746,  0.5477, -0.5477,
        -0.3162,  0.3162, -0.3162,  0.3162,  0.5164,  0.5164,  0.4472,  0.4472,
        -0.1291, -0.1291, -0.1291, -0.1291,  1.2649,  1.0954, -0.3162, -0.3162,
         1.0000,  0.8660, -0.2500, -0.2500,  0.9428,  0.8165, -0.2357, -0.2357,
         1.0000,  0.8660, -0.2500, -0.2500,  1.2649,  1.0954, -0.3162, -0.3162,
         0.5164, -0.5164,  0.4472, -0.4472, -0.1291,  0.1291, -0.1291,  0.1291,
         0.4082,  0.4082,  0.4082,  0.4082, -0.4082, -0.4082,  1.0000,  1.0000,
        -1.0000,  0.7906,  0.7906, -0.7906,  0.7454,  0.7454, -0.7454,  0.7906,
         0.7906, -0.7906,  1.0000,  1.0000, -1.0000,  0.4082, -0.4082,  0.4082,
        -0.4082, -0.4082,  0.4082,  0.5000,  0.5000, -0.5000, -0.5000,  1.2247,
        -1.2247,  0.9682, -0.9682,  0.9129, -0.9129,  0.9682, -0.9682,  1.2247,
        -1.2247,  0.5000, -0.5000, -0.5000,  0.5000], dtype=M1.dtype, device=M1.device)
    rr = torch.tensor([2, 2, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0,
        1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 1, 2, 2, 0, 0, 1, 1, 0,
        0, 2, 2, 0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2,
        0, 1, 1, 0, 0, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0,
        1, 2, 0, 1, 2, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 2, 2, 0, 0, 1, 2, 2, 0,
        1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 2, 2, 0, 1, 1, 2, 2, 2, 2, 0, 0,
        1, 1, 2, 2, 0, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 1, 2,
        2, 0, 0, 2, 2, 0, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 2, 0, 0], device=M1.device)
    rc = torch.tensor([2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2,
        0, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 2, 0, 2, 0, 2, 0,
        2, 0, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 2,
        0, 2, 0, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 2, 0], device=M1.device)
    pr = torch.tensor([0, 0, 4, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 0, 4, 4, 0, 0, 1, 1, 3, 3,
        0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 1, 3, 0, 0, 1, 1, 3, 3, 1, 1, 2,
        2, 0, 0, 4, 4, 1, 2, 0, 4, 1, 2, 0, 4, 1, 2, 0, 4, 1, 2, 0, 4, 1, 2, 0,
        4, 1, 1, 2, 2, 0, 0, 4, 4, 2, 2, 3, 3, 1, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1,
        2, 3, 1, 2, 3, 1, 2, 2, 3, 3, 1, 1, 3, 3, 2, 2, 4, 4, 0, 0, 3, 2, 4, 0,
        3, 2, 4, 0, 3, 2, 4, 0, 3, 2, 4, 0, 3, 2, 4, 0, 3, 3, 2, 2, 4, 4, 0, 0,
        4, 4, 3, 3, 1, 1, 4, 3, 1, 4, 3, 1, 4, 3, 1, 4, 3, 1, 4, 3, 1, 4, 4, 3,
        3, 1, 1, 4, 4, 0, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 4, 0, 0], device=M1.device)
    pc = torch.tensor([0, 4, 0, 4, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 0, 4, 0, 0, 4, 0, 4, 0, 4,
        0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 0, 4, 0, 4, 0, 0, 4, 0,
        4, 0, 4, 0, 4, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4,
        4, 4, 0, 4, 0, 4, 0, 4, 0, 0, 4, 0, 4, 0, 4, 0, 0, 0, 1, 1, 1, 2, 2, 2,
        3, 3, 3, 4, 4, 4, 4, 0, 4, 0, 4, 0, 0, 4, 0, 4, 0, 4, 0, 4, 0, 0, 0, 0,
        1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 0, 4, 0, 4, 0, 4, 0,
        0, 4, 0, 4, 0, 4, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 0, 4,
        0, 4, 0, 0, 4, 0, 4, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 0, 4, 0], device=M1.device)
    out_idx = torch.tensor([ 0,  0,  0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  6,  6,
         7,  7,  7,  7,  7,  7,  8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 11,
        12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15,
        15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19,
        19, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 22, 22, 22,
        23, 23, 23, 24, 24, 24, 25, 25, 25, 26, 26, 26, 27, 27, 27, 27, 27, 27,
        28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31,
        31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34,
        35, 35, 35, 35, 35, 35, 36, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39,
        40, 40, 40, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 43, 43, 44, 44, 45,
        45, 46, 46, 47, 47, 48, 48, 48, 48], device=M1.device)
    products = coeff[None] * M1[:, rr, rc] * Mprev[:, pr, pc]
    return products, out_idx[None].expand(N, 189)
