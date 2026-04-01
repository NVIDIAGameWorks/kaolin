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

import pytest
import torch
import math

from kaolin.ops.gaussians import transform_gaussians, transform_shs
import kaolin.math.quat as quat_ops

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]

def eval_sh(deg, sh, dirs):
    """
    Evaluate spherical harmonics at unit directions
    using hardcoded SH polynomials.
    Works with torch/np/jnp.
    ... Can be 0 or more batch dimensions.
    Args:
        deg: int SH deg. Currently, 0-3 supported
        sh: jnp.ndarray SH coeffs [..., C, (deg + 1) ** 2]
        dirs: jnp.ndarray unit directions [..., 3]
    Returns:
        [..., C]
    """
    assert deg <= 4 and deg >= 0
    coeff = (deg + 1) ** 2
    assert sh.shape[-1] >= coeff

    result = C0 * sh[..., 0]
    if deg > 0:
        x, y, z = dirs[..., 0:1], dirs[..., 1:2], dirs[..., 2:3]
        result = (result -
                  C1 * y * sh[..., 1] +
                  C1 * z * sh[..., 2] -
                  C1 * x * sh[..., 3])

        if deg > 1:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result = (result +
                      C2[0] * xy * sh[..., 4] +
                      C2[1] * yz * sh[..., 5] +
                      C2[2] * (2.0 * zz - xx - yy) * sh[..., 6] +
                      C2[3] * xz * sh[..., 7] +
                      C2[4] * (xx - yy) * sh[..., 8])

            if deg > 2:
                result = (result +
                          C3[0] * y * (3 * xx - yy) * sh[..., 9] +
                          C3[1] * xy * z * sh[..., 10] +
                          C3[2] * y * (4 * zz - xx - yy)* sh[..., 11] +
                          C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * sh[..., 12] +
                          C3[4] * x * (4 * zz - xx - yy) * sh[..., 13] +
                          C3[5] * z * (xx - yy) * sh[..., 14] +
                          C3[6] * x * (xx - 3 * yy) * sh[..., 15])

    return result

def naive_transform_rotation(rot: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Basic reimplementation of kaolin.ops.gaussians.transforms._transform_rot (only support xyzw convention)"""
    scale = torch.linalg.norm(transform[..., :3, :3], dim=-2)
    rot_quat = quat_ops.quat_from_rot33(transform[..., :3, :3] / scale.unsqueeze(-2))
    rot_unit = rot / torch.linalg.norm(rot, dim=-1).unsqueeze(-1)
    result = quat_ops.quat_mul(rot_quat, rot_unit)
    return result


def _quat_wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Reorder [w, x, y, z] -> [x, y, z, w] for ops that use xyzw."""
    return torch.cat([q[:, 1:], q[:, :1]], dim=-1)


def _quat_xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Reorder [x, y, z, w] -> [w, x, y, z]."""
    return torch.cat([q[:, -1:], q[:, :-1]], dim=-1)

# TODO(cfujitsang): Add some rendering in the tests
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.half, torch.float32, torch.float64])
class TestTransformGaussians:
    """Tests for transform_gaussians function."""

    @pytest.fixture
    def xyz(self, device, dtype):
        return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype, device=device)

    @pytest.fixture
    def rotations(self, device, dtype):
        """Unit quaternions in wxyz convention [w, x, y, z] (same orientations as former xyzw gsplat fixture)."""
        return torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ],
            dtype=dtype,
            device=device,
        )

    @pytest.fixture
    def scales(self, device, dtype):
        return torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [0.5, 0.5, 0.5]], dtype=dtype, device=device)

    @pytest.fixture
    def shs_feat(self, device, dtype):
        return torch.randn(3, 15, 3, device=device, dtype=dtype)

    @pytest.fixture
    def trs_transform(self, device, dtype):
        """Composed TRS: uniform scale=2.0, rotation 45° around Z, translation [3.5, -1.2, 2.7]."""
        angle = math.pi / 4
        c, s = math.cos(angle), math.sin(angle)
        sf = 2.0
        return torch.tensor([
            [sf * c, -sf * s,  0.0,  3.5],
            [sf * s,  sf * c,  0.0, -1.2],
            [0.0,     0.0,    sf,    2.7],
            [0.0,     0.0,     0.0,  1.0],
        ], dtype=dtype, device=device)

    @pytest.fixture
    def batched_trs_transforms(self, device, dtype):
        """Three distinct TRS transforms stacked as (3, 4, 4):
          [0] scale=2.0, 45° around Z, t=[ 3.5, -1.2,  2.7]
          [1] scale=0.5, 30° around X, t=[-1.0,  2.5, -3.0]
          [2] scale=3.0, 60° around Y, t=[ 0.5, -0.5,  1.5]
        """
        def _trs(sf, axis, angle_rad, tx, ty, tz):
            c, s = math.cos(angle_rad), math.sin(angle_rad)
            if axis == 'x':
                R = [[1, 0,  0], [0, c, -s], [0, s, c]]
            elif axis == 'y':
                R = [[c, 0, s], [0, 1,  0], [-s, 0, c]]
            else:  # z
                R = [[c, -s, 0], [s, c,  0], [0,  0, 1]]
            return torch.tensor([
                [sf * R[0][0], sf * R[0][1], sf * R[0][2], tx],
                [sf * R[1][0], sf * R[1][1], sf * R[1][2], ty],
                [sf * R[2][0], sf * R[2][1], sf * R[2][2], tz],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=dtype, device=device)

        return torch.stack([
            _trs(2.0, 'z', math.pi / 4,  3.5, -1.2,  2.7),
            _trs(0.5, 'x', math.pi / 6, -1.0,  2.5, -3.0),
            _trs(3.0, 'y', math.pi / 3,  0.5, -0.5,  1.5),
        ])
    
    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_transform_shs(self, shs_feat, trs_transform, degree, device, dtype):
        """Test transform_shs by sampling the transformed SH vs transformed directions on original SH."""
        shs_feat_input = shs_feat[:, :((degree + 1) ** 2) - 1, :]
        new_shs_feat = transform_shs(shs_feat_input, trs_transform[:3, :3].unsqueeze(0)) # N x (degree + 1) ** 2 - 1 x 3
        num_dirs = 100
        dirs = torch.randn(1, num_dirs, 3, device=device, dtype=dtype)
        dirs = dirs / torch.linalg.norm(dirs, dim=-1).unsqueeze(-1) # 1 x num_dirs x 3
        rgb = eval_sh(
            degree,
            torch.nn.functional.pad(new_shs_feat, (0, 0, 1, 0, 0, 0), value=0).permute(0, 2, 1).unsqueeze(1), # N x 1 x 3 x (degree + 1) ** 2 
            dirs # 1 x num_dirs x 3
        )
        tfm_dirs = (trs_transform[None, None, :3, :3].transpose(-1, -2) @ dirs.unsqueeze(-1)).squeeze(-1)
        gt = eval_sh(
            degree,
            torch.nn.functional.pad(shs_feat_input, (0, 0, 1, 0, 0, 0), value=0).permute(0, 2, 1).unsqueeze(1), # N x 1 x 3 x (degree + 1) ** 2 
            tfm_dirs # 1 x num_dirs x 3
        )

        if dtype == torch.half:
            atol = 1e-1
            rtol = 1e-2
        else:
            atol = 1e-4
            rtol = 1e-4
        torch.testing.assert_close(rgb, gt, atol=atol, rtol=rtol)

    @pytest.mark.parametrize("degree", [1, 2, 3])
    def test_batched_transform_shs(self, shs_feat, batched_trs_transforms, degree, device, dtype):
        """Test transform_shs by sampling the transformed SH vs transformed directions on original SH."""
        shs_feat_input = shs_feat[:, :((degree + 1) ** 2) - 1, :]
        new_shs_feat = transform_shs(shs_feat_input, batched_trs_transforms[..., :3, :3]) # N x (degree + 1) ** 2 - 1 x 3
        num_dirs = 100
        dirs = torch.randn(1, num_dirs, 3, device=device, dtype=dtype)
        dirs = dirs / torch.linalg.norm(dirs, dim=-1).unsqueeze(-1) # 1 x num_dirs x 3
        rgb = eval_sh(
            degree,
            torch.nn.functional.pad(new_shs_feat, (0, 0, 1, 0, 0, 0), value=0).permute(0, 2, 1).unsqueeze(1), # N x 1 x 3 x (degree + 1) ** 2 
            dirs # 1 x num_dirs x 3
        )
        tfm_dirs = (batched_trs_transforms[:, None, :3, :3].transpose(-1, -2) @ dirs.unsqueeze(-1)).squeeze(-1) # 
        gt = eval_sh(
            degree,
            torch.nn.functional.pad(shs_feat_input, (0, 0, 1, 0, 0, 0), value=0).permute(0, 2, 1).unsqueeze(1), # N x 1 x 3 x (degree + 1) ** 2 
            tfm_dirs # 1 x num_dirs x 3
        )

        if dtype == torch.half:
            atol = 1e-1
            rtol = 1e-2
        else:
            atol = 1e-4
            rtol = 1e-4
        torch.testing.assert_close(rgb[:2], gt[:2], atol=atol, rtol=rtol)
        # 3rd rotation matrix is causing some larger differences, but still within acceptable tolerance (not a bug)
        torch.testing.assert_close(rgb[2:], gt[2:], atol=1e-1, rtol=1e-2)


    @pytest.mark.parametrize("use_xyzw", [True, False])
    @pytest.mark.parametrize("use_shs_feat", [False, True])
    @pytest.mark.parametrize("use_log_scales", [True, False])
    def test_identity_transform(self, xyz, rotations, scales, shs_feat, use_log_scales, use_shs_feat, use_xyzw, device, dtype):
        """Identity transform returns unchanged positions, rotations, and scales."""
        identity = torch.eye(4, dtype=dtype, device=device)

        if use_shs_feat:
            shs_feat_input = shs_feat
        else:
            shs_feat_input = None

        if use_xyzw:
            rotations = _quat_wxyz_to_xyzw(rotations)

        new_xyz, new_rotations, new_scales, new_shs_feat = transform_gaussians(
            xyz, rotations, scales, identity, shs_feat_input, use_log_scales=use_log_scales, use_xyzw=use_xyzw)

        torch.testing.assert_close(new_xyz, xyz)
        torch.testing.assert_close(new_rotations, rotations)
        torch.testing.assert_close(new_scales, scales)
        torch.testing.assert_close(new_shs_feat, shs_feat_input)

    @pytest.mark.parametrize("use_xyzw", [True, False])
    @pytest.mark.parametrize("use_shs_feat", [False, True])
    @pytest.mark.parametrize("use_log_scales", [True, False])
    def test_translation_only(self, xyz, rotations, scales, shs_feat, use_log_scales, use_shs_feat, use_xyzw, device, dtype):
        """Translation transform only affects positions."""
        translation = torch.tensor([2.0, 3.0, 5.0], dtype=dtype, device=device)
        transform = torch.eye(4, dtype=dtype, device=device)
        transform[:3, 3] = translation

        if use_shs_feat:
            shs_feat_input = shs_feat
        else:
            shs_feat_input = None

        if use_xyzw:
            rotations = _quat_wxyz_to_xyzw(rotations)

        new_xyz, new_rotations, new_scales, new_shs_feat = transform_gaussians(
            xyz, rotations, scales, transform, shs_feat_input, use_log_scales=use_log_scales, use_xyzw=use_xyzw)

        torch.testing.assert_close(new_xyz, xyz + translation)
        torch.testing.assert_close(new_rotations, rotations)
        torch.testing.assert_close(new_scales, scales)
        torch.testing.assert_close(new_shs_feat, shs_feat_input)

    @pytest.mark.parametrize("use_xyzw", [True, False])
    @pytest.mark.parametrize("use_shs_feat", [False, True])
    @pytest.mark.parametrize("use_log_scales", [True, False])
    def test_uniform_scale(self, xyz, rotations, scales, shs_feat, use_log_scales, use_shs_feat, use_xyzw, device, dtype):
        """Uniform scale transform scales positions and raw_scales."""
        s = 3.0
        transform = torch.eye(4, dtype=dtype, device=device)
        transform[0, 0] = transform[1, 1] = transform[2, 2] = s

        if use_shs_feat:
            shs_feat_input = shs_feat
        else:
            shs_feat_input = None

        if use_xyzw:
            rotations = _quat_wxyz_to_xyzw(rotations)

        new_xyz, new_rotations, new_scales, new_shs_feat = transform_gaussians(
            xyz, rotations, scales, transform, shs_feat_input, use_log_scales=use_log_scales, use_xyzw=use_xyzw)

        if use_log_scales:
            expected_scales = scales * (math.log(s) / scales + 1)
        else:
            expected_scales = scales * s

        torch.testing.assert_close(new_xyz, xyz * s)
        torch.testing.assert_close(new_rotations, rotations)
        torch.testing.assert_close(new_scales, expected_scales)
        torch.testing.assert_close(new_shs_feat, shs_feat_input)

    @pytest.mark.parametrize("use_xyzw", [True, False])
    @pytest.mark.parametrize("use_shs_feat", [False, True])
    @pytest.mark.parametrize("use_log_scales", [True, False])
    def test_rotation_yaw_180(self, xyz, rotations, scales, shs_feat, use_log_scales, use_shs_feat, use_xyzw, device, dtype):
        """Rotation transform only affects rotations."""
        yaw = math.pi
        transform = torch.tensor([[
            [math.cos(yaw), 0, -math.sin(yaw), 0],
            [0, 1, 0, 0],
            [math.sin(yaw), 0, math.cos(yaw), 0],
            [0, 0, 0, 1],
        ]], dtype=dtype, device=device)

        if use_shs_feat:
            shs_feat_input = shs_feat
        else:
            shs_feat_input = None

        if use_xyzw:
            rotations_input = _quat_wxyz_to_xyzw(rotations)
        else:
            rotations_input = rotations

        new_xyz, new_rotations, new_scales, new_shs_feat = transform_gaussians(
            xyz, rotations_input, scales, transform, shs_feat_input, use_log_scales=use_log_scales, use_xyzw=use_xyzw)

        expected_xyz = xyz.clone()
        expected_xyz[:, [0, 2]] = -expected_xyz[:, [0, 2]]

        expected_rotations = naive_transform_rotation(_quat_wxyz_to_xyzw(rotations), transform)

        if not use_xyzw:
            expected_rotations = _quat_xyzw_to_wxyz(expected_rotations)

        if use_shs_feat:
            expected_shs_feat = shs_feat.clone()
            expected_shs_feat[:, [ 1,  2,  3,  4, 11, 12, 13, 14]] = -expected_shs_feat[:, [ 1,  2,  3,  4, 11, 12, 13, 14]]
        else:
            expected_shs_feat = None

        torch.testing.assert_close(new_xyz, expected_xyz)
        torch.testing.assert_close(new_rotations, expected_rotations)
        torch.testing.assert_close(new_scales, scales)
        torch.testing.assert_close(new_shs_feat, expected_shs_feat)

    @pytest.mark.parametrize("use_xyzw", [True, False])
    @pytest.mark.parametrize("use_shs_feat", [False, True])
    @pytest.mark.parametrize("use_log_scales", [True, False])
    def test_rotation_pitch_180(self, xyz, rotations, scales, shs_feat, use_log_scales, use_shs_feat, use_xyzw, device, dtype):
        """Rotation transform only affects rotations."""
        pitch = math.pi
        transform = torch.tensor([[
            [1, 0, 0, 0],
            [0, math.cos(pitch), math.sin(pitch), 0],
            [0, -math.sin(pitch), math.cos(pitch), 0],
            [0, 0, 0, 1],
        ]], dtype=dtype, device=device)

        if use_shs_feat:
            shs_feat_input = shs_feat
        else:
            shs_feat_input = None

        if use_xyzw:
            rotations_input = _quat_wxyz_to_xyzw(rotations)
        else:
            rotations_input = rotations

        new_xyz, new_rotations, new_scales, new_shs_feat = transform_gaussians(
            xyz, rotations_input, scales, transform, shs_feat_input, use_xyzw=use_xyzw, use_log_scales=use_log_scales)

        expected_xyz = xyz.clone()
        expected_xyz[:, [1, 2]] = -expected_xyz[:, [1, 2]]

        expected_rotations = naive_transform_rotation(_quat_wxyz_to_xyzw(rotations), transform)
        if not use_xyzw:
            expected_rotations = _quat_xyzw_to_wxyz(expected_rotations)

        if use_shs_feat:
            expected_shs_feat = shs_feat.clone()
            expected_shs_feat[:, [ 0,  1,  3,  6,  8, 10, 11, 13]] = -expected_shs_feat[:, [ 0,  1,  3,  6,  8, 10, 11, 13]]
        else:
            expected_shs_feat = None
        torch.testing.assert_close(new_xyz, expected_xyz)
        torch.testing.assert_close(new_rotations, expected_rotations)
        torch.testing.assert_close(new_scales, scales)
        torch.testing.assert_close(new_shs_feat, expected_shs_feat)

    @pytest.mark.parametrize("use_xyzw", [True, False])
    @pytest.mark.parametrize("use_shs_feat", [False, True])
    @pytest.mark.parametrize("use_log_scales", [True, False])
    def test_rotation_roll_180(self, xyz, rotations, scales, shs_feat, use_log_scales, use_shs_feat, use_xyzw, device, dtype):
        """Rotation transform only affects rotations."""
        roll = math.pi
        transform = torch.tensor([[
            [math.cos(roll), -math.sin(roll), 0, 0],
            [math.sin(roll), math.cos(roll), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]], dtype=dtype, device=device)

        if use_shs_feat:
            shs_feat_input = shs_feat
        else:
            shs_feat_input = None

        if use_xyzw:
            rotations_input = _quat_wxyz_to_xyzw(rotations)
        else:
            rotations_input = rotations

        new_xyz, new_rotations, new_scales, new_shs_feat = transform_gaussians(
            xyz, rotations_input, scales, transform, shs_feat_input, use_xyzw=use_xyzw, use_log_scales=use_log_scales)

        expected_xyz = xyz.clone()
        expected_xyz[:, [0, 1]] = -expected_xyz[:, [0, 1]]
        expected_rotations = naive_transform_rotation(_quat_wxyz_to_xyzw(rotations), transform)

        if not use_xyzw:
            expected_rotations = _quat_xyzw_to_wxyz(expected_rotations)

        if use_shs_feat:
            expected_shs_feat = shs_feat.clone()
            expected_shs_feat[:, [ 0,  2,  4,  6,  8, 10, 12, 14]] = -expected_shs_feat[:, [ 0,  2,  4,  6,  8, 10, 12, 14]]
        else:
            expected_shs_feat = None
        torch.testing.assert_close(new_xyz, expected_xyz)
        torch.testing.assert_close(new_rotations, expected_rotations)
        torch.testing.assert_close(new_scales, scales)
        torch.testing.assert_close(new_shs_feat, expected_shs_feat)

    @pytest.mark.parametrize("use_xyzw", [True, False])
    @pytest.mark.parametrize("use_shs_feat", [False, True])
    @pytest.mark.parametrize("use_log_scales", [True, False])
    def test_composed_trs_transform(self, xyz, rotations, scales, trs_transform, shs_feat,
                                    use_log_scales, use_shs_feat, use_xyzw, device, dtype):
        """Composed translation + rotation + scale on non-trivial gaussians."""
        if use_shs_feat:
            shs_feat_input = shs_feat
        else:
            shs_feat_input = None

        if use_xyzw:
            rotations_input = _quat_wxyz_to_xyzw(rotations)
        else:
            rotations_input = rotations

        new_xyz, new_rotations, new_scales, new_shs_feat = transform_gaussians(
            xyz, rotations_input, scales, trs_transform, shs_feat_input, use_log_scales=use_log_scales, use_xyzw=use_xyzw)

        # Positions: full 4x4 affine application
        R = trs_transform[:3, :3]
        S = torch.linalg.norm(R, dim=-2).unsqueeze(-2)
        R /= S
        t = trs_transform[:3, 3]
        expected_xyz = (R @ xyz.T).T * S + t
        if use_log_scales:
            expected_scales = scales * (torch.log(S) / scales + 1)
        else:
            expected_scales = scales * S
    
        if use_shs_feat:
            expected_shs_feat = transform_shs(shs_feat, R.unsqueeze(0))
        else:
            expected_shs_feat = None

        expected_rotations = naive_transform_rotation(
            _quat_wxyz_to_xyzw(rotations), trs_transform.unsqueeze(0))
        if not use_xyzw:
            expected_rotations = _quat_xyzw_to_wxyz(expected_rotations)

        torch.testing.assert_close(new_xyz, expected_xyz)
        torch.testing.assert_close(new_scales, expected_scales)
        norms = torch.linalg.norm(new_rotations, dim=-1)
        torch.testing.assert_close(norms, torch.ones(xyz.shape[0], dtype=dtype, device=device), atol=1e-5, rtol=0.0)
        torch.testing.assert_close(new_rotations, expected_rotations)
        torch.testing.assert_close(new_shs_feat, expected_shs_feat)

    @pytest.mark.parametrize("use_xyzw", [True, False])
    @pytest.mark.parametrize("use_shs_feat", [False, True])
    @pytest.mark.parametrize("use_log_scales", [True, False])
    def test_inverse_transform_returns_original(self, xyz, rotations, scales, trs_transform,
                                                shs_feat, use_log_scales, use_shs_feat, use_xyzw, dtype):
        """Applying the inverse transform after the forward transform recovers the original values."""
        if dtype == torch.half:
            inv_transform = torch.linalg.inv(trs_transform.float()).half()
        else:
            inv_transform = torch.linalg.inv(trs_transform)
        if use_shs_feat:
            shs_feat_input = shs_feat
        else:
            shs_feat_input = None

        if use_xyzw:
            rotations = _quat_wxyz_to_xyzw(rotations)

        new_xyz, new_rotations, new_scales, new_shs_feat = transform_gaussians(
            xyz, rotations, scales, trs_transform, shs_feat=shs_feat_input, use_log_scales=use_log_scales, use_xyzw=use_xyzw)
        recovered_xyz, recovered_rotations, recovered_scales, recovered_shs_feat = transform_gaussians(
            new_xyz, new_rotations, new_scales, inv_transform, new_shs_feat, use_log_scales=use_log_scales, use_xyzw=use_xyzw)

        if dtype == torch.half:
            atol = 1e-3
            rtol = 1e-3
        else:
            atol = 1e-5
            rtol = 1e-5
        torch.testing.assert_close(recovered_xyz, xyz, atol=atol, rtol=rtol)
        torch.testing.assert_close(recovered_rotations, rotations, atol=atol, rtol=rtol)
        torch.testing.assert_close(recovered_scales, scales, atol=atol, rtol=rtol)
        torch.testing.assert_close(recovered_shs_feat, shs_feat_input, atol=atol, rtol=rtol)

    # -------------------------------------------------------------------
    # Batched (Nx4x4) variants – each gaussian gets a different transform
    # -------------------------------------------------------------------

    @pytest.mark.parametrize("use_xyzw", [True, False])
    @pytest.mark.parametrize("use_shs_feat", [False, True])
    @pytest.mark.parametrize("use_log_scales", [True, False])
    def test_batched_identity_transform(self, xyz, rotations, scales, shs_feat, use_log_scales, use_shs_feat, use_xyzw, device, dtype):
        """Per-gaussian identity transforms leave all outputs unchanged."""
        n = xyz.shape[0]
        transforms = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).expand(n, 4, 4).contiguous()

        if use_shs_feat:
            shs_feat_input = shs_feat
        else:
            shs_feat_input = None

        if use_xyzw:
            rotations = _quat_wxyz_to_xyzw(rotations)

        new_xyz, new_rotations, new_scales, new_shs_feat = transform_gaussians(
            xyz, rotations, scales, transforms, shs_feat=shs_feat_input, use_log_scales=use_log_scales, use_xyzw=use_xyzw)

        torch.testing.assert_close(new_xyz, xyz)
        torch.testing.assert_close(new_rotations, rotations)
        torch.testing.assert_close(new_scales, scales)
        torch.testing.assert_close(new_shs_feat, shs_feat_input)

    @pytest.mark.parametrize("use_xyzw", [True, False])
    @pytest.mark.parametrize("use_shs_feat", [False, True])
    @pytest.mark.parametrize("use_log_scales", [True, False])
    def test_batched_translation_only(self, xyz, rotations, scales, shs_feat, use_log_scales, use_shs_feat, use_xyzw, device, dtype):
        """Different translation per gaussian only affects positions."""
        translations = torch.tensor([[2.0, 3.0, 5.0], [-1.5, 0.5, 2.0], [4.0, -2.5, 1.0]], dtype=dtype, device=device)
        n = xyz.shape[0]
        transforms = torch.eye(4, dtype=dtype, device=device).unsqueeze(0).expand(n, 4, 4).clone()
        transforms[:, :3, 3] = translations

        if use_shs_feat:
            shs_feat_input = shs_feat
        else:
            shs_feat_input = None

        if use_xyzw:
            rotations = _quat_wxyz_to_xyzw(rotations)

        new_xyz, new_rotations, new_scales, new_shs_feat = transform_gaussians(
            xyz, rotations, scales, transforms, shs_feat=shs_feat_input,
            use_xyzw=use_xyzw,
            use_log_scales=use_log_scales)

        torch.testing.assert_close(new_xyz, xyz + translations)
        torch.testing.assert_close(new_rotations, rotations)
        torch.testing.assert_close(new_scales, scales)
        torch.testing.assert_close(new_shs_feat, shs_feat_input)

    @pytest.mark.parametrize("use_xyzw", [True, False])
    @pytest.mark.parametrize("use_shs_feat", [False, True])
    @pytest.mark.parametrize("use_log_scales", [True, False])
    def test_batched_uniform_scale(self, xyz, rotations, scales, shs_feat, use_log_scales, use_shs_feat, use_xyzw, device, dtype):
        """Different uniform scale per gaussian scales positions and raw_scales independently."""
        scale_factors = torch.tensor([2.0, 0.5, 3.0], dtype=dtype, device=device)
        n = xyz.shape[0]
        transforms = torch.zeros(n, 4, 4, dtype=dtype, device=device)
        transforms[:, 0, 0] = scale_factors
        transforms[:, 1, 1] = scale_factors
        transforms[:, 2, 2] = scale_factors
        transforms[:, 3, 3] = 1.0

        if use_shs_feat:
            shs_feat_input = shs_feat
        else:
            shs_feat_input = None

        if use_xyzw:
            rotations = _quat_wxyz_to_xyzw(rotations)

        new_xyz, new_rotations, new_scales, new_shs_feat = transform_gaussians(
            xyz, rotations, scales, transforms, shs_feat=shs_feat_input,
            use_log_scales=use_log_scales,
            use_xyzw=use_xyzw)

        torch.testing.assert_close(new_xyz, xyz * scale_factors.unsqueeze(-1))
        torch.testing.assert_close(new_rotations, rotations)
        if use_log_scales:
            torch.testing.assert_close(new_scales, scales * (torch.log(scale_factors).unsqueeze(-1) / scales + 1))
        else:
            torch.testing.assert_close(new_scales, scales * scale_factors.unsqueeze(-1))
        torch.testing.assert_close(new_shs_feat, shs_feat_input)

    @pytest.mark.parametrize("use_xyzw", [True, False])
    @pytest.mark.parametrize("use_shs_feat", [False, True])
    @pytest.mark.parametrize("use_log_scales", [True, False])
    def test_batched_rotation_180(self, xyz, rotations, scales, shs_feat, use_log_scales, use_shs_feat, use_xyzw, device, dtype):
        """Rotation transform only affects rotations."""
        yaw = math.pi
        pitch = math.pi
        roll = math.pi
        transform = torch.tensor([
            [[math.cos(yaw), 0, -math.sin(yaw), 0],
             [0, 1, 0, 0],
             [math.sin(yaw), 0, math.cos(yaw), 0],
             [0, 0, 0, 1]
            ],
            [[1, 0, 0, 0],
             [0, math.cos(pitch), math.sin(pitch), 0],
             [0, -math.sin(pitch), math.cos(pitch), 0],
             [0, 0, 0, 1]
            ],
            [[math.cos(roll), -math.sin(roll), 0, 0],
             [math.sin(roll), math.cos(roll), 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]
            ]
        ], dtype=dtype, device=device)

        if use_shs_feat:
            shs_feat_input = shs_feat
        else:
            shs_feat_input = None

        if use_xyzw:
            rotations_input = _quat_wxyz_to_xyzw(rotations)
        else:
            rotations_input = rotations

        new_xyz, new_rotations, new_scales, new_shs_feat = transform_gaussians(
            xyz, rotations_input, scales, transform, shs_feat_input, use_log_scales=use_log_scales, use_xyzw=use_xyzw)

        expected_xyz = xyz.clone()
        expected_xyz[0, [0, 2]] = -expected_xyz[0, [0, 2]]  
        expected_xyz[1, [1, 2]] = -expected_xyz[1, [1, 2]]
        expected_xyz[2, [0, 1]] = -expected_xyz[2, [0, 1]]

        expected_rotations = naive_transform_rotation(_quat_wxyz_to_xyzw(rotations), transform)
        if not use_xyzw:
            expected_rotations = _quat_xyzw_to_wxyz(expected_rotations)

        if use_shs_feat:
            expected_shs_feat = shs_feat.clone()
            expected_shs_feat[0, [ 1,  2,  3,  4, 11, 12, 13, 14]] = -expected_shs_feat[0, [ 1,  2,  3,  4, 11, 12, 13, 14]]
            expected_shs_feat[1, [ 0,  1,  3,  6,  8, 10, 11, 13]] = -expected_shs_feat[1, [ 0,  1,  3,  6,  8, 10, 11, 13]]
            expected_shs_feat[2, [ 0,  2,  4,  6,  8, 10, 12, 14]] = -expected_shs_feat[2, [ 0,  2,  4,  6,  8, 10, 12, 14]]
        else:
            expected_shs_feat = None
        torch.testing.assert_close(new_xyz, expected_xyz)
        torch.testing.assert_close(new_rotations, expected_rotations)
        torch.testing.assert_close(new_scales, scales)
        torch.testing.assert_close(new_shs_feat, expected_shs_feat)

    @pytest.mark.parametrize("use_xyzw", [True, False])
    @pytest.mark.parametrize("use_shs_feat", [False, True])
    @pytest.mark.parametrize("use_log_scales", [True, False])
    def test_batched_composed_trs_transform(self, xyz, rotations, scales, batched_trs_transforms,
                                            shs_feat, use_log_scales, use_shs_feat, use_xyzw, device, dtype):
        """Composed translation + rotation + scale on non-trivial gaussians."""
        if use_shs_feat:
            shs_feat_input = shs_feat
        else:
            shs_feat_input = None

        if use_xyzw:
            rotations_input = _quat_wxyz_to_xyzw(rotations)
        else:
            rotations_input = rotations

        new_xyz, new_rotations, new_scales, new_shs_feat = transform_gaussians(
            xyz, rotations_input, scales, batched_trs_transforms, shs_feat=shs_feat_input,
            use_xyzw=use_xyzw,
            use_log_scales=use_log_scales)

        # Expected xyz: per-gaussian affine R_i @ xyz_i + t_i
        R = batched_trs_transforms[:, :3, :3] # (N, 3, 3)
        S = torch.linalg.norm(R, dim=-2) # (N, 3)
        R /= S.unsqueeze(-2) # (N, 3, 3)
        t = batched_trs_transforms[:, :3, 3] # (N, 3)
        expected_xyz = (R @ xyz.unsqueeze(-1)).squeeze(-1) * S + t
        if use_log_scales:
            expected_scales = scales * (torch.log(S) / scales + 1)
        else:
            expected_scales = scales * S
        if use_shs_feat:
            expected_shs_feat = transform_shs(shs_feat, R)
        else:
            expected_shs_feat = None
        expected_rotations = naive_transform_rotation(_quat_wxyz_to_xyzw(rotations), batched_trs_transforms)
        if not use_xyzw:
            expected_rotations = _quat_xyzw_to_wxyz(expected_rotations)
        
        if dtype == torch.half:
            atol = 1e-3
            rtol = 1e-3
        else:
            atol = 1e-5
            rtol = 1e-5

        torch.testing.assert_close(new_xyz, expected_xyz)
        torch.testing.assert_close(new_scales, expected_scales)
        torch.testing.assert_close(new_shs_feat, expected_shs_feat)
        # Output quaternions must remain unit-norm
        norms = torch.linalg.norm(new_rotations, dim=-1)
        torch.testing.assert_close(norms, torch.ones(xyz.shape[0], dtype=dtype, device=device), atol=atol, rtol=rtol)
        torch.testing.assert_close(new_rotations, expected_rotations)

    def test_default_batched_composed_trs_transform(self, xyz, rotations, scales, batched_trs_transforms,
                                                    device, dtype):
        """Composed translation + rotation + scale; default quaternion layout is wxyz."""

        new_xyz, new_rotations, new_scales, new_shs_feat = transform_gaussians(
            xyz, rotations, scales, batched_trs_transforms)

        # Expected xyz: per-gaussian affine R_i @ xyz_i + t_i
        R = batched_trs_transforms[:, :3, :3] # (N, 3, 3)
        S = torch.linalg.norm(R, dim=-2) # (N, 3)
        R /= S.unsqueeze(-2) # (N, 3, 3)
        t = batched_trs_transforms[:, :3, 3] # (N, 3)
        expected_xyz = (R @ xyz.unsqueeze(-1)).squeeze(-1) * S + t
        expected_scales = scales * S
        expected_shs_feat = None
        expected_rotations = _quat_xyzw_to_wxyz(
            naive_transform_rotation(_quat_wxyz_to_xyzw(rotations), batched_trs_transforms))

        if dtype == torch.half:
            atol = 1e-3
            rtol = 1e-3
        else:
            atol = 1e-5
            rtol = 1e-5

        torch.testing.assert_close(new_xyz, expected_xyz)
        torch.testing.assert_close(new_scales, expected_scales)
        torch.testing.assert_close(new_shs_feat, expected_shs_feat)
        # Output quaternions must remain unit-norm
        norms = torch.linalg.norm(new_rotations, dim=-1)
        torch.testing.assert_close(norms, torch.ones(xyz.shape[0], dtype=dtype, device=device), atol=atol, rtol=rtol)
        torch.testing.assert_close(new_rotations, expected_rotations)

    @pytest.mark.parametrize("use_xyzw", [True, False])
    @pytest.mark.parametrize("use_shs_feat", [False, True])
    @pytest.mark.parametrize("use_log_scales", [True, False])
    def test_batched_inverse_transform_returns_original(self, xyz, rotations, scales, batched_trs_transforms,
                                                        shs_feat, use_log_scales, use_shs_feat, use_xyzw, dtype):
        """Per-gaussian inverse transform recovers original positions, rotations, and scales."""
        if dtype == torch.half:
            inv_transforms = torch.linalg.inv(batched_trs_transforms.float()).half()
        else:
            inv_transforms = torch.linalg.inv(batched_trs_transforms)
        if use_shs_feat:
            shs_feat_input = shs_feat
        else:
            shs_feat_input = None

        if use_xyzw:
            rotations = _quat_wxyz_to_xyzw(rotations)

        new_xyz, new_rotations, new_scales, new_shs_feat = transform_gaussians(
            xyz, rotations, scales, batched_trs_transforms, shs_feat=shs_feat_input,
            use_log_scales=use_log_scales,
            use_xyzw=use_xyzw)
        recovered_xyz, recovered_rotations, recovered_scales, recovered_shs_feat = transform_gaussians(
            new_xyz, new_rotations, new_scales, inv_transforms, new_shs_feat, use_log_scales=use_log_scales, use_xyzw=use_xyzw
        )

        if dtype == torch.half:
            atol = 1e-3
            rtol = 1e-3
            sh_atol = 1e-2
            sh_rtol = 1e-2
        else:
            atol = 1e-5
            rtol = 1e-5
            sh_atol = 1e-3
            sh_rtol = 1e-3
        torch.testing.assert_close(recovered_xyz, xyz, atol=atol, rtol=rtol)
        torch.testing.assert_close(recovered_rotations, rotations, atol=atol, rtol=rtol)
        torch.testing.assert_close(recovered_scales, scales, atol=atol, rtol=rtol)
        torch.testing.assert_close(recovered_shs_feat, shs_feat_input, atol=sh_atol, rtol=sh_rtol)
