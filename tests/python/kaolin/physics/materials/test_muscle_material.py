# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
import pytest
import torch
from functools import partial

import kaolin.physics.materials.muscle_material as muscle_material
import kaolin.physics.materials as material
import kaolin.physics.materials.utils as material_utils


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_precompute_fiber_matrix(device, dtype):
    N = 20
    fiber_vecs = torch.tensor([0, 1, 0], device=device, dtype=dtype).expand(N, 3)
    B1 = muscle_material.precompute_fiber_matrix(fiber_vecs)

    B2 = torch.zeros(N, 3, 9, device=fiber_vecs.device, dtype=fiber_vecs.dtype)
    for p in range(N):
        B2[p, 0, 0] = fiber_vecs[p, 0]
        B2[p, 0, 1] = fiber_vecs[p, 1]
        B2[p, 0, 2] = fiber_vecs[p, 2]
        B2[p, 1, 3] = fiber_vecs[p, 0]
        B2[p, 1, 4] = fiber_vecs[p, 1]
        B2[p, 1, 5] = fiber_vecs[p, 2]
        B2[p, 2, 6] = fiber_vecs[p, 0]
        B2[p, 2, 7] = fiber_vecs[p, 1]
        B2[p, 2, 8] = fiber_vecs[p, 2]
    assert torch.allclose(B1, B2)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_muscle_energy_unactivated(device, dtype):
    N = 20
    B = 1

    F = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)
    fiber_vecs = torch.tensor([0, 1, 0], device=device, dtype=dtype).expand(N, 3)
    B = muscle_material.precompute_fiber_matrix(fiber_vecs)

    activation = 0
    E1 = torch.tensor(0, device=device, dtype=dtype)
    E2 = torch.sum(muscle_material.unbatched_muscle_energy(activation, B, F))
    assert torch.allclose(E1, E2)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_muscle_gradient(device, dtype):
    N = 40
    B = 1

    F = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)
    fiber_vecs = torch.tensor([0, 1, 0], device=device, dtype=dtype).expand(N, 3)
    B = muscle_material.precompute_fiber_matrix(fiber_vecs)

    activation = 1

    G2 = muscle_material.unbatched_muscle_gradient(activation, B, F)
    G3 = torch.autograd.functional.jacobian(lambda p: torch.sum(
        muscle_material.unbatched_muscle_energy(activation, B, p)), F)

    assert torch.allclose(G3.reshape(-1, 9), G2)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_muscle_hessian(device, dtype):
    N = 20
    B = 1

    F = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)
    fiber_vecs = torch.tensor([0, 1, 0], device=device, dtype=dtype).expand(N, 3)
    B = muscle_material.precompute_fiber_matrix(fiber_vecs)

    activation = 12

    H2 = muscle_material.unbatched_muscle_hessian(activation, B, F)

    H3 = torch.autograd.functional.hessian(lambda p: torch.sum(
        muscle_material.unbatched_muscle_energy(activation, B, p)), F)
    H3 = H3.reshape(9 * N, 9 * N)
    # Make sure the block diags match up
    for n in range(N):
        assert torch.allclose(H2[n], H3[9 * n:9 * n + 9, 9 * n:9 * n + 9], rtol=1e-1, atol=1e-1)
        # zero out the block
        H3[9 * n:9 * n + 9, 9 * n:9 * n + 9] *= 0

    # Make sure the rest of the matrix is zeros
    assert torch.allclose(torch.zeros_like(H3), H3, rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_muscle_material_fiber_matrix(device, dtype):
    N = 20
    fiber_vecs = torch.tensor([0, 1, 0], device=device, dtype=dtype).expand(N, 3)
    muscle_object = material.MuscleMaterial(fiber_vecs=fiber_vecs)

    B1 = muscle_object.fiber_mat_blocks

    B2 = torch.zeros(N, 3, 9, device=fiber_vecs.device, dtype=fiber_vecs.dtype)
    for p in range(N):
        B2[p, 0, 0] = fiber_vecs[p, 0]
        B2[p, 0, 1] = fiber_vecs[p, 1]
        B2[p, 0, 2] = fiber_vecs[p, 2]
        B2[p, 1, 3] = fiber_vecs[p, 0]
        B2[p, 1, 4] = fiber_vecs[p, 1]
        B2[p, 1, 5] = fiber_vecs[p, 2]
        B2[p, 2, 6] = fiber_vecs[p, 0]
        B2[p, 2, 7] = fiber_vecs[p, 1]
        B2[p, 2, 8] = fiber_vecs[p, 2]
    assert torch.allclose(B1, B2)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_muscle_material_energy_unactivated(device, dtype):
    N = 20
    B = 1

    F = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)
    fiber_vecs = torch.tensor([0, 1, 0], device=device, dtype=dtype).expand(N, 3)
    muscle_object = material.MuscleMaterial(fiber_vecs=fiber_vecs)

    activation = 0
    E1 = torch.tensor(0, device=device, dtype=dtype)
    E2 = torch.sum(muscle_object.energy(F.flatten()))
    assert torch.allclose(E1, E2)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_muscle_material_gradient(device, dtype):
    N = 40
    B = 1

    F = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)
    fiber_vecs = torch.tensor([0, 1, 0], device=device, dtype=dtype).expand(N, 3)
    muscle_object = material.MuscleMaterial(fiber_vecs=fiber_vecs)

    activation = 1

    muscle_object.set_activation(activation)

    G2 = muscle_object.gradient(F.flatten())
    G3 = torch.autograd.functional.jacobian(lambda p: torch.sum(
        muscle_object.energy(p)), F.flatten())

    assert torch.allclose(G3.reshape(-1, 9), G2)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_muscle_material_hessian(device, dtype):
    N = 20
    B = 1

    F = torch.eye(3, device=device, dtype=dtype).expand(N, 3, 3)
    fiber_vecs = torch.tensor([0, 1, 0], device=device, dtype=dtype).expand(N, 3)
    muscle_object = material.MuscleMaterial(fiber_vecs=fiber_vecs)

    activation = 12

    muscle_object.set_activation(activation)

    H2 = muscle_object.hessian(F.flatten())

    H3 = torch.autograd.functional.hessian(lambda p: torch.sum(
        muscle_object.energy(p)), F.flatten())
    H3 = H3.reshape(9 * N, 9 * N)
    # Make sure the block diags match up
    for n in range(N):
        assert torch.allclose(H2[n], H3[9 * n:9 * n + 9, 9 * n:9 * n + 9], rtol=1e-1, atol=1e-1)
        # zero out the block
        H3[9 * n:9 * n + 9, 9 * n:9 * n + 9] *= 0

    # Make sure the rest of the matrix is zeros
    assert torch.allclose(torch.zeros_like(H3), H3, rtol=1e-1, atol=1e-1)
