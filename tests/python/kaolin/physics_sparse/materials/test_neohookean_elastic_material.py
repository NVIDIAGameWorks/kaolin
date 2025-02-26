import pytest

import torch
import warp as wp
import warp.sparse as wps
import kaolin.physics as physics
import kaolin.physics_sparse as physics_sparse


def setup_neohookean(device, dtype):
    N = 20
    B = 1
    vol = 1.0

    yms = 1e3 * torch.ones(N, B, device=device)
    prs = 0.4 * torch.ones(N, B, device=device)
    integration_pt_volume = (vol/N)*torch.ones((N, 1),
                                               device=device, dtype=dtype)

    F = 2.0*torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)

    mus, lams = physics.materials.utils.to_lame(yms, prs)

    wp_mus = wp.from_torch(mus.flatten().contiguous(),
                           dtype=wp.dtype_from_torch(dtype))
    wp_lams = wp.from_torch(lams.flatten().contiguous(),
                            dtype=wp.dtype_from_torch(dtype))

    wp_vols = wp.from_torch(
        integration_pt_volume.flatten(), dtype=wp.float32)

    wp_neohookean_struct = physics_sparse.materials.NeohookeanElasticMaterial(
        wp_mus, wp_lams, wp_vols)

    return wp_neohookean_struct, F, mus, lams, integration_pt_volume


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_neohookean_energy(device, dtype):
    wp_neohookean_struct, defo_grads, mus, lams, vols = setup_neohookean(
        device, dtype)

    expected_energy = torch.sum(
        physics.materials.unbatched_neohookean_energy(mus, lams, defo_grads))

    wp_defo_grads = wp.from_torch(defo_grads.contiguous(), dtype=wp.mat33)
    energy = wp_neohookean_struct.energy(wp_defo_grads)

    assert torch.allclose(
        expected_energy/defo_grads.shape[0], wp.to_torch(energy))


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_neohookean_gradient(device, dtype):
    wp_neohookean_struct, defo_grads, mus, lams, vols = setup_neohookean(
        device, dtype)

    expected_grad = physics.materials.unbatched_neohookean_gradient(
        mus, lams, defo_grads).reshape(-1, 9) * vols.reshape(-1, 1)

    wp_defo_grads = wp.from_torch(defo_grads.contiguous(), dtype=wp.mat33)
    wp_neohookean_struct.gradient(wp_defo_grads)

    # print(expected_grad)
    # print("------------")
    # print(wp.to_torch(wp_neohookean_struct.gradients))

    assert torch.allclose(expected_grad, wp.to_torch(
        wp_neohookean_struct.gradients))


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_neohookean_hessian(device, dtype):
    wp_neohookean_struct, defo_grads, mus, lams, vols = setup_neohookean(
        device, dtype)

    expected_hess = (1.0/defo_grads.shape[0]) * physics.materials.unbatched_neohookean_hessian(
        mus, lams, defo_grads)

    wp_defo_grads = wp.from_torch(defo_grads.contiguous(), dtype=wp.mat33)

    wp_neohookean_struct.hessian(wp_defo_grads)
    print(expected_hess.shape, wp.to_torch(
        wp_neohookean_struct.hessians_blocks).shape)
    print(expected_hess[0])
    print("------------")
    print(wp.to_torch(wp_neohookean_struct.hessians_blocks)[0])

    assert torch.allclose(expected_hess, wp.to_torch(
        wp_neohookean_struct.hessians_blocks))
