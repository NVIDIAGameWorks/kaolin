import pytest
import warp as wp
import numpy as np
import torch
from typing import Any
import kaolin.physics as physics
from kaolin.physics_sparse.materials.linear_elastic_material import cauchy_strain, linear_elastic_energy, linear_elastic_gradient

wp.init()


@wp.kernel
def elastic_kernel(mus: wp.array(dtype=Any, ndim=2),
                   lams: wp.array(dtype=Any, ndim=2),
                   Fs: wp.array(dtype=wp.mat33, ndim=2),
                   wp_e: wp.array(dtype=Any, ndim=1)):
    pt_idx, batch_idx = wp.tid()

    mu_ = mus[pt_idx, batch_idx]
    lam_ = lams[pt_idx, batch_idx]
    F_ = Fs[pt_idx, batch_idx]

    E = linear_elastic_energy(mu_, lam_, F_)
    wp.atomic_add(wp_e, batch_idx, E)


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_wp_linear_energy(device, dtype):
    N = 20
    B = 4

    F = torch.eye(3, device=device, dtype=dtype).expand(N, B, 3, 3)

    yms = 1e3 * torch.ones(N, B, device=device)
    prs = 0.4 * torch.ones(N, B, device=device)

    mus, lams = physics.materials.utils.to_lame(yms, prs)

    E1 = torch.tensor(0, device=device, dtype=dtype)

    wp_e = wp.zeros(B, dtype=wp.dtype_from_torch(dtype))
    wp_F = wp.from_torch(F.contiguous(), dtype=wp.mat33)
    wp_mus = wp.from_torch(mus.contiguous(), dtype=wp.dtype_from_torch(dtype))
    wp_lams = wp.from_torch(
        lams.contiguous(), dtype=wp.dtype_from_torch(dtype))

    wp.launch(
        kernel=elastic_kernel,
        dim=(N, B),
        inputs=[
            wp_mus,  # mus: wp.array(dtype=float),   ; shape (N,B,)
            wp_lams,  # lams: wp.array(dtype=float),  ; shape (N,B,)
            wp_F  # defo_grads: wp.array(dtype=wp.mat33),  ; shape (N,B,3,3)
        ],
        outputs=[wp_e],  # out_e: wp.array(dtype=float)  ; shape (B,)
        adjoint=False
    )
    E2 = wp.to_torch(wp_e).sum()
    assert torch.allclose(E1, E2)
