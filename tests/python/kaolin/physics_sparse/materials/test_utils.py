import pytest
import warp as wp
import torch
from typing import Any
import kaolin.physics as physics
import kaolin.physics_sparse as physics_sparse
from functools import partial


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_wp_get_F(device, dtype):
    N = 20
    # Create N random points in the unit cube in torch
    pts = torch.rand(N, 3, device=device, dtype=dtype)

    # Create a random deformation in the unit cube in torch
    dz = 1e-1 * torch.rand(1, 3, 4, device=device, dtype=dtype)
    z0 = torch.zeros(1, 3, 4, device=device, dtype=dtype)
    z = dz + z0

    # Create fcn, return vector of ones
    def model(x): return torch.ones(
        (x.shape[0], 1), device=device, dtype=dtype)

    partial_weight_fcn_lbs = partial(
        physics.simplicits.utils.weight_function_lbs, tfms=z.unsqueeze(0), fcn=model)

    expected_F = physics.utils.finite_diff_jac(
        partial_weight_fcn_lbs, pts).squeeze()

    # Warp code to get F using dFdz
    wp_pts = wp.from_torch(pts)
    wp_z = wp.from_torch(z.flatten().contiguous())
    wp_weights = wp.from_torch(model(pts).contiguous())
    wp_dFdz = physics_sparse.simplicits_common.sparse_dFdz_matrix(wp_weights)
    wp_F = physics_sparse.materials.utils.wp_get_F(wp_z, wp_dFdz)
    wp_F = wp.to_torch(wp_F)

    assert (torch.allclose(expected_F, wp_F, atol=1e-3))

    # Warp code to get F using finite differences
