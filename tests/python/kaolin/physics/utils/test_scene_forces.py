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

import kaolin.physics.utils as physics_utils
from kaolin.physics.simplicits.precomputed import lbs_matrix, jacobian_dF_dz,jacobian_dF_dz_const_handle
from kaolin.physics.utils.finite_diff import finite_diff_jac 


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
@pytest.mark.parametrize('dim', [2, 3])
def test_gravity_energy(device, dtype, dim):
    N = 20
    DIM = dim
    
    rhos = torch.ones(N,1, dtype=dtype, device=device)
    total_vol = 10
    
    acc = torch.zeros(DIM*N, device=device, dtype=dtype)
    acc[1::3] = 9.8
    points = torch.rand(N, DIM, device=device, dtype=dtype)  # n x 3 points
    
    gravity_object = physics_utils.Gravity(rhos=rhos, acceleration=torch.tensor([0,9.8,0], device=device, dtype=dtype)[0:DIM])
    
    ge = gravity_object.energy(points, integration_weights=torch.tensor(total_vol/N, device=device, dtype=dtype))
    
    expected_ge = torch.tensor([0], device=device, dtype=dtype)
    integ_weights = torch.tensor(total_vol/N, device=device, dtype=dtype)
    for i in range(0,points.shape[0]):
        expected_ge += rhos[i] * integ_weights * points[i,:] @ torch.tensor([0,9.8, 0], device=device, dtype=dtype)[0:DIM]

    assert torch.allclose(ge, expected_ge)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
@pytest.mark.parametrize('dim', [2, 3])  
def test_gravity_gradient(device, dtype, dim):
    N = 20
    DIM = dim
    
    rhos = torch.ones(N,1, dtype=dtype, device=device)
    total_vol = 10
    
    masses = rhos*(total_vol/rhos.shape[0]) 
    acc = torch.tensor([0,9.8,0], device=device, dtype=dtype)[0:DIM]
    points = torch.rand(N, DIM, device=device, dtype=dtype)  # n x 3 points
    
    gravity_object = physics_utils.Gravity(rhos=rhos, acceleration=torch.tensor([0,9.8,0], device=device, dtype=dtype)[0:DIM])
    
    gg = gravity_object.gradient(points, integration_weights=torch.tensor(total_vol/N, device=device, dtype=dtype))
    
    expected_gg = torch.zeros_like(points, device=device, dtype=dtype)
    
    integ_weights = torch.tensor(total_vol/N, device=device, dtype=dtype)
    for i in range(0,points.shape[0]):
        expected_gg[i,:] = rhos[i] * integ_weights * acc
    
    assert torch.allclose(gg, expected_gg)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])    
@pytest.mark.parametrize('dim', [2, 3])  
def test_gravity_hessian(device, dtype, dim):
    N = 20
    DIM = dim
    
    rhos = torch.ones(N,1, dtype=dtype, device=device)
    total_vol = 10
    
    masses = rhos*(total_vol/rhos.shape[0]) 
    acc = torch.tensor([0,9.8,0], device=device, dtype=dtype)[0:DIM]
    points = torch.rand(N, DIM, device=device, dtype=dtype)  # n x 3 points
    
    gravity_object = physics_utils.Gravity(rhos=rhos, acceleration=torch.tensor([0,9.8,0], device=device, dtype=dtype)[0:DIM])
    
    gh = gravity_object.hessian(points, integration_weights=torch.tensor(total_vol/N, device=device, dtype=dtype))
    
    expected_gh = torch.zeros(N,N, DIM, DIM, device=device, dtype=dtype)

    assert torch.allclose(gh, expected_gh)
    

@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
@pytest.mark.parametrize('dim', [2, 3])  
def test_floor_energy(device, dtype, dim):
    N = 20
    AXIS = 1
    FLOOR_HEIGHT = 0.5
    DIM=dim
    
    rhos = torch.ones(N, dtype=dtype, device=device)
    total_vol = 10

    points = torch.rand(N, DIM, device=device, dtype=dtype)  # n x 3 points
    
    floor_object = physics_utils.Floor(floor_height=FLOOR_HEIGHT, floor_axis=AXIS)
    
    fe = floor_object.energy(points)
    expected_fe = torch.zeros(N, device=device, dtype=dtype)
    
    for i in range(0,points.shape[0]):
        expected_fe[i] = (points[i,AXIS] - FLOOR_HEIGHT)**2 if points[i, AXIS]<FLOOR_HEIGHT else 0

    print(fe, expected_fe, torch.sum(expected_fe))
    assert torch.allclose(fe, torch.sum(expected_fe))
    
@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
@pytest.mark.parametrize('dim', [2, 3])  
def test_floor_gradient(device, dtype, dim):
    N = 20
    AXIS = 1
    FLOOR_HEIGHT = 0.5
    DIM=dim
    
    rhos = torch.ones(N, dtype=dtype, device=device)
    total_vol = 10
    
    points = torch.rand(N, DIM, device=device, dtype=dtype, requires_grad=True)  # n x 3 points
    
    floor_object = physics_utils.Floor(floor_height=FLOOR_HEIGHT, floor_axis=AXIS)
    
    fg = floor_object.gradient(points)
    
    points = points.detach()
    expected_fg = torch.zeros_like(points)
    
    fe0 = floor_object.energy(points)
    eps = 1e-4
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            points[i, j] += eps
            fel =  torch.sum(floor_object.energy(points))
            points[i, j] -= 2*eps
            fer = torch.sum(floor_object.energy(points))
            points[i, j] += eps 
            expected_fg[i, j] += (fel - fer)/(2*eps)

    assert torch.allclose(fg, expected_fg, rtol=1e-3, atol=1e-3)
    
@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
@pytest.mark.parametrize('dim', [2, 3])  
def test_floor_hessian(device, dtype, dim):
    N = 20
    AXIS = 1
    FLOOR_HEIGHT = 0.5
    DIM=dim
    
    points = torch.rand(N, 3, device=device, dtype=dtype, requires_grad=True)  # n x 3 points
    
    floor_object = physics_utils.Floor(floor_height=FLOOR_HEIGHT, floor_axis=AXIS)
    
    fh = floor_object.hessian(points)

    expected_fh = torch.autograd.functional.hessian(floor_object.energy, points).transpose(1,2)
    
    assert torch.allclose(fh, expected_fh)

@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
@pytest.mark.parametrize('dim', [2, 3])  
def test_boundary_energy(device, dtype, dim):
    N = 20
    DIM = dim

    points = torch.rand(N, DIM, device=device, dtype=dtype)  # n x 3 points
    
    bdry_cond = physics_utils.Boundary()
    bdry_indx = torch.nonzero(points[:,0]<0.5, as_tuple=False).squeeze()
    bdry_pos = points[bdry_indx,:]+0.1
    bdry_cond.set_pinned_verts(bdry_indx, bdry_pos)
    
    be = bdry_cond.energy(points)
    expected_be = torch.tensor([0], device=device, dtype=dtype)
    
    for i in range(bdry_indx.shape[0]):
        idx = bdry_indx[i]
        expected_be += torch.sum(torch.square(bdry_pos[i] - points[idx])) #sq norm
    
    assert torch.allclose(torch.sum(be), expected_be)


@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
@pytest.mark.parametrize('dim', [2, 3])  
def test_boundary_gradient(device, dtype, dim):
    N = 20
    DIM = dim

    points = torch.rand(N, DIM, device=device, dtype=dtype, requires_grad=True)  # n x 3 points
 
    bdry_cond = physics_utils.Boundary()
    bdry_indx = torch.nonzero(points[:,0]<0.5, as_tuple=False).squeeze()
    bdry_pos = points[bdry_indx,:]+0.1
    bdry_cond.set_pinned_verts(bdry_indx, bdry_pos)
    
    bg = bdry_cond.gradient(points)


    points = points.clone().detach()
    expected_bg = torch.zeros_like(points)

    be = bdry_cond.energy(points)

    eps = 1e-4
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            points[i, j] += eps
            fel =  torch.sum(bdry_cond.energy(points))
            points[i, j] -= 2*eps
            fer = torch.sum(bdry_cond.energy(points))
            points[i, j] += eps 
            expected_bg[i, j] += (fel - fer)/(2*eps)
    print(bg)
    print(expected_bg)
    assert torch.allclose(bg, expected_bg, rtol=1e-3, atol=1e-3)
    
    
@pytest.mark.parametrize('device', ['cuda', 'cpu'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
@pytest.mark.parametrize('dim', [2, 3]) 
def test_boundary_hessian(device, dtype, dim):
    N = 20
    DIM = dim

    points = torch.rand(N, DIM, device=device, dtype=dtype, requires_grad=True)  # n x 3 points
    bdry_cond = physics_utils.Boundary()
    bdry_indx = torch.nonzero(points[:,0]<0.5, as_tuple=False).squeeze()
    bdry_pos = points[bdry_indx,:]+0.1
    bdry_cond.set_pinned_verts(bdry_indx, bdry_pos)
    
    bh = bdry_cond.hessian(points)

    expected_bh = torch.autograd.functional.hessian(bdry_cond.energy, points).transpose(1,2)
    
    assert torch.allclose(bh, expected_bh)