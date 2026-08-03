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


import os
import warp as wp
import pytest
import torch
from functools import partial

import kaolin.physics.common.collisions as collisions
from kaolin.physics.simplicits.precomputed import lbs_matrix
from kaolin.physics.utils.torch_utilities import hess_reduction
from kaolin.physics.utils.warp_utilities import capture_function_torch
from kaolin.utils.testing import with_seed

# Every fixture and test in this file builds tensors directly on 'cuda' -- the collision
# kernels are Warp CUDA kernels with no CPU path. Without this the whole file reports as
# errors rather than skips on a CPU-only lane.
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="collision kernels require CUDA")


def _collision_contact_energy_analytical(
        offset, nor, rc, rp_ratio, mu, mu_dt, nu):
    d = torch.dot(offset, nor)
    d_hat = d / rc
    active = (d_hat > rp_ratio) & (d_hat <= 1.0)

    dc = d_hat - 1.0
    dp = d_hat - rp_ratio
    barrier = 2.0 * torch.log(dp)

    dE_d_hat = -dc * (barrier + dc / dp)
    energy = -dc * dc * torch.log(dp)

    vt = (offset - d * nor) / mu_dt
    vt_norm = vt.norm()
    mu_fn = -mu * dE_d_hat / rc
    h_vt = (
        0.5 * nu * vt_norm * vt_norm
        + torch.where(
            vt_norm < 1.0,
            vt_norm * vt_norm * (1.0 - vt_norm / 3.0),
            vt_norm - 1.0 / 3.0,
        )
    )
    energy = energy + mu_fn * mu_dt * h_vt
    return torch.where(active, energy, torch.zeros_like(energy))


def _collision_contact_gradient_from_energy(
        dx_cur, dx_start, kinematic_gap, nor, idx_a, idx_b,
        rc, rp_ratio, mu, mu_dt, nu):
    delta_a = dx_cur[idx_a] - dx_start[idx_a]
    delta_b = dx_cur[idx_b] - dx_start[idx_b] if idx_b >= 0 else torch.zeros_like(delta_a)
    offset = (delta_a + kinematic_gap - delta_b).detach().requires_grad_(True)
    energy = _collision_contact_energy_analytical(
        offset, nor, rc, rp_ratio, mu, mu_dt, nu)
    energy.backward()
    return offset.grad


@pytest.fixture(params=['one_object', 'two_objects', "three_objects", "two_objects_one_static"])
@with_seed(2, 2, 2)
def test_scenes(request):
    device = 'cuda'
    if request.param == 'one_object':
        num_points = 20
        dx = torch.zeros(num_points, 3, device=device)
        x0 = torch.rand(num_points, 3, device=device)
        density = torch.ones(num_points, device=device)
        volume = torch.ones(num_points, device=device)/num_points
        return {
            'x0': wp.array(x0, dtype=wp.vec3),
            'dx': wp.array(dx, dtype=wp.vec3),
            'obj_ids': wp.array(torch.zeros(num_points, device=device, dtype=torch.int32), dtype=wp.int32), # all points are in the same object
            'is_static': wp.array(torch.zeros(num_points, device=device, dtype=torch.int32), dtype=wp.int32), # all points are dynamic
            'weights': wp.array(torch.ones((num_points, 1), device=device), dtype=wp.float32)
        }
    elif request.param == 'two_objects' or request.param == 'three_objects':
        num_objects = 2 if request.param == 'two_objects' else 3
        _stacked_x0 = []
        _stacked_dx = []
        _stacked_obj_ids = []
        _stacked_is_static = []
        _stacked_weights = []
        for i in range(num_objects):
            num_points = 20
            dx = torch.zeros(num_points, 3, device=device)
            dx[:, 1] += 0.5*i
            x0 = torch.rand(num_points, 3, device=device)
            _stacked_x0.append(x0)
            _stacked_dx.append(dx)
            _stacked_obj_ids.append(torch.ones(num_points, device=device, dtype=torch.int32)*i)
            _stacked_is_static.append(torch.zeros(num_points, device=device, dtype=torch.int32)) # all points are dynamic
            _stacked_weights.append(torch.ones(num_points, 1, device=device))
            
        return {
            'x0': wp.array(torch.cat(_stacked_x0, dim=0), dtype=wp.vec3),
            'dx': wp.array(torch.cat(_stacked_dx, dim=0), dtype=wp.vec3),
            'obj_ids': wp.array(torch.cat(_stacked_obj_ids, dim=0), dtype=wp.int32),
            'is_static': wp.array(torch.cat(_stacked_is_static, dim=0), dtype=wp.int32),
            'weights': wp.from_torch(torch.block_diag(*_stacked_weights).contiguous())
        }
    elif request.param == 'two_objects_one_static':
        _stacked_x0 = []
        _stacked_dx = []
        _stacked_obj_ids = []
        _stacked_is_static = []
        _stacked_weights = []
        for i in range(2):
            num_points = 20
            dx = torch.zeros(num_points, 3, device=device)
            dx[:, 1] += 0.5*i
            x0 = torch.rand(num_points, 3, device=device)
            _stacked_x0.append(x0)
            _stacked_dx.append(dx)
            _stacked_obj_ids.append(torch.ones(num_points, device=device, dtype=torch.int32)*i)
            _stacked_is_static.append(torch.zeros(num_points, device=device, dtype=torch.int32))  # all points are dynamic
            _stacked_weights.append(torch.ones(num_points, 1, device=device))
        
        # make one object static.
        # dtype=torch.int32 is load-bearing: the other entries are int32, so a float32
        # here promotes the whole torch.cat to float32, and wp.array(<cuda float32>,
        # dtype=wp.int32) takes Warp's __cuda_array_interface__ path, which reinterprets
        # the bits rather than converting. 1.0f becomes 1065353216, so `cp_is_static == 1`
        # is never true and this fixture silently marks nothing static.
        _stacked_is_static[0] = torch.ones(num_points, device=device, dtype=torch.int32)

        return {
            'x0': wp.array(torch.cat(_stacked_x0, dim=0), dtype=wp.vec3),
            'dx': wp.array(torch.cat(_stacked_dx, dim=0), dtype=wp.vec3),
            'obj_ids': wp.array(torch.cat(_stacked_obj_ids, dim=0), dtype=wp.int32),
            'is_static': wp.array(torch.cat(_stacked_is_static, dim=0), dtype=wp.int32),
            'weights': wp.from_torch(torch.block_diag(*_stacked_weights).contiguous())
        }
    else:
        assert False, "Invalid test scene"


# collision Tests
@pytest.mark.parametrize("test_scenes", [
    "one_object", "two_objects", "three_objects", "two_objects_one_static"], indirect=True)
def test_detect_collisions(test_scenes):

    x0 = test_scenes['x0']
    dx = test_scenes['dx']
    obj_ids = test_scenes['obj_ids']
    is_static = test_scenes['is_static']

    # Collision parameters 
    dt = 0.01
    collision_radius = 0.05
    detection_ratio = 1.5 
    impenetrable_barrier_ratio = 0.5 
    friction = 0.2
    collision = collisions.Collision(dt=dt, 
                                     collision_particle_radius=collision_radius, 
                                     detection_ratio=detection_ratio, 
                                     impenetrable_barrier_ratio=impenetrable_barrier_ratio, 
                                     friction=friction) # other parameters are default
    collision.detect_collisions(dx, x0, obj_ids, is_static)
    num_contact = collision.num_contacts 
    if num_contact > 0:
        collision_indices_a = wp.to_torch(collision.collision_indices_a[:num_contact])
        collision_indices_b = wp.to_torch(collision.collision_indices_b[:num_contact])   
        # make pairs from collision_indices_a and collision_indices_b
        sorted_pairs = torch.stack((collision_indices_a, collision_indices_b), dim=1)
        sorted_pairs = sorted_pairs[sorted_pairs[:, 0].argsort()]    
    else:
        sorted_pairs = torch.tensor([], dtype=torch.int32, device=wp.device_to_torch(x0.device))
    

    t_x0 = wp.to_torch(x0)
    t_dx = wp.to_torch(dx)
    t_x = t_x0 + t_dx
    t_obj_ids = wp.to_torch(obj_ids)
    t_is_static = wp.to_torch(is_static)
    
    # Analytically calculate collision energy
    # loop through all pairs of points and calculate the distance between them
    expected_pairs_set = []
    expected_num_contacts = 0
    for i in range(len(t_x)):
        for j in range(i+1, len(t_x)):
            if t_obj_ids[i] == t_obj_ids[j]:
                pass # ignore self collisions
            else:
                # Get distance between points
                dist = (t_x[i] - t_x[j]).norm()
                if dist<=2.0*collision_radius*detection_ratio:
                    if t_is_static[i] == 1:
                        expected_pairs_set.append((-1, j))
                    elif t_is_static[j] == 1:
                        expected_pairs_set.append((i, -1))
                    else:
                        expected_pairs_set.append((i, j))
                    expected_num_contacts += 1
                    
    expected_pairs = torch.tensor(
        list(expected_pairs_set), dtype=torch.int32, device=t_x.device)
    
    
    # Loop through each row of sorted_pairs, check if it is in expected_pairs_set and remove it if it is
    for i in range(len(sorted_pairs)):
        pair = (sorted_pairs[i, 0].item(), sorted_pairs[i, 1].item())
        # Check both orientations since collision detection might return (a,b) or (b,a)
        if pair in expected_pairs_set:
            expected_pairs_set.remove(pair)
        elif (pair[1], pair[0]) in expected_pairs_set:
            expected_pairs_set.remove((pair[1], pair[0]))
        else:
            assert False, f"collision pair {pair} not found in expected pairs"
    
    assert num_contact == expected_num_contacts, "number of contacts don't match analytical calculation"
    assert len(expected_pairs_set)==0, "Not all collision pairs were found"


@pytest.mark.parametrize("test_scenes", [
    "two_objects",
    "three_objects",
    "two_objects_one_static"
], indirect=True)
def test_collision_jacobian(test_scenes):
    
    x0 = test_scenes['x0']
    dx = test_scenes['dx']
    weights = test_scenes['weights']
    obj_ids = test_scenes['obj_ids']
    is_static = test_scenes['is_static']
    
    

    # Collision parameters
    dt = 0.01
    collision_radius = 0.05
    detection_ratio = 1.5
    impenetrable_barrier_ratio = 0.5
    friction = 0.5
    collision = collisions.Collision(dt=dt,
                                     collision_particle_radius=collision_radius,
                                     detection_ratio=detection_ratio,
                                     impenetrable_barrier_ratio=impenetrable_barrier_ratio,
                                     friction=friction)  # other parameters are default
    
    collision.detect_collisions(dx, x0, obj_ids, is_static)
    collision.calculate_jacobian(weights, x0, is_static)
    collision_jacobian = collision.collision_J_dense
    
    t_x0 = wp.to_torch(x0)
    t_dx = wp.to_torch(dx)
    t_x = t_x0 + t_dx
    t_obj_ids = wp.to_torch(obj_ids)
    t_is_static = wp.to_torch(is_static)
    t_weights = wp.to_torch(weights)
    t_B = lbs_matrix(t_x0, t_weights)
    
    ##### Torch Code For Collision Jacobian #####
    # indices of the colliding point pairs
    t_ind_a = wp.to_torch(collision.collision_indices_a[:collision.num_contacts])
    t_ind_b = wp.to_torch(collision.collision_indices_b[:collision.num_contacts])

    # 3n x h jacobian matrix where the flattened x = t_B@z
    # triplicated indices
    # multiply indices by 3 and repeat interleave increasing by 1 each time
    # Create indices for x,y,z components by multiplying by 3 and adding offsets
    t3_ind_a = torch.repeat_interleave(
        3*t_ind_a, 3) + torch.tile(torch.arange(3, device=t_x0.device), (t_ind_a.shape[0],))
    t3_ind_b = torch.repeat_interleave(
        3*t_ind_b, 3) + torch.tile(torch.arange(3, device=t_x0.device), (t_ind_b.shape[0],))
    # Grab the rows of the jacobian that correspond to the indices above
    expected_jacobian = t_B[t3_ind_a, :] - t_B[t3_ind_b, :]

    assert torch.allclose(collision_jacobian, expected_jacobian, rtol=1e-5), \
        "Collision jacobian doesn't match analytical calculation"



@pytest.mark.parametrize("test_scenes", [
    "one_object",
    "two_objects"
], indirect=True)
def test_collision_energy(test_scenes):

    x0 = test_scenes['x0']
    dx = test_scenes['dx']
    obj_ids = test_scenes['obj_ids']
    is_static = test_scenes['is_static']

     # Collision parameters 
    dt = 0.01
    collision_radius = 0.05
    detection_ratio = 1.5 
    impenetrable_barrier_ratio = 0.5 
    friction = 0.5
    collision = collisions.Collision(dt=dt, 
                                     collision_particle_radius=collision_radius, 
                                     detection_ratio=detection_ratio, 
                                     impenetrable_barrier_ratio=impenetrable_barrier_ratio, 
                                     friction=friction) # other parameters are default

    mu = friction
    mu_dt = dt*collision.friction_reg
    nu = collision.friction_fluid*collision.friction_reg

    collision.detect_collisions(dx, x0, obj_ids, is_static)

    energy = collision.energy(dx, x0, coeff=1.0)

    t_x0 = wp.to_torch(x0)
    t_dx = wp.to_torch(dx)
    t_x = t_x0 + t_dx
    t_obj_ids = wp.to_torch(obj_ids)

    rc = 2.0 * collision_radius
    expected_energy = torch.tensor(0.0, device=t_x.device, dtype=t_x.dtype)
    for i in range(len(t_x0)):
        for j in range(i + 1, len(t_x0)):
            if t_obj_ids[i] == t_obj_ids[j]:
                continue
            offset = t_x[i] - t_x[j]
            normal = offset / offset.norm()
            expected_energy += _collision_contact_energy_analytical(
                offset, normal, rc, impenetrable_barrier_ratio, mu, mu_dt, nu)

    assert torch.allclose(wp.to_torch(energy)[0], expected_energy, rtol=1e-5), \
        "Collision energy doesn't match analytical calculation"

        
@pytest.mark.parametrize("test_scenes", [
    "two_objects",
    "three_objects"
], indirect=True)
def test_collision_gradient(test_scenes):

    x0 = test_scenes['x0']
    dx = test_scenes['dx']
    obj_ids = test_scenes['obj_ids']
    is_static = test_scenes['is_static']

    # Collision parameters
    dt = 0.01
    collision_radius = 0.05
    detection_ratio = 1.5
    impenetrable_barrier_ratio = 0.5
    friction = 0.2
    collision = collisions.Collision(dt=dt,
                                     collision_particle_radius=collision_radius,
                                     detection_ratio=detection_ratio,
                                     impenetrable_barrier_ratio=impenetrable_barrier_ratio,
                                     friction=friction)

    collision.detect_collisions(dx, x0, obj_ids, is_static)
    wp_gradient = collision.gradient(dx, x0, coeff=1.0)
    gradient = wp.to_torch(wp_gradient) if wp_gradient.shape[0] > 0 else torch.zeros(0, 3, device=wp.device_to_torch(wp_gradient.device))

    t_x0 = wp.to_torch(x0)
    t_dx = wp.to_torch(dx)
    t_indices_a = wp.to_torch(collision.collision_indices_a)
    t_indices_b = wp.to_torch(collision.collision_indices_b)

    dEdx_fd = torch.zeros(collision.num_contacts, 3, device=t_dx.device, dtype=t_dx.dtype)
    eps = 1e-5
    for i in range(dEdx_fd.shape[0]):
        for j in range(dEdx_fd.shape[1]):
            pair = (t_indices_a[i].item(), t_indices_b[i].item())
            t_dx[pair[0], j] += eps
            E1 = wp.to_torch(collision.energy(wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=1.0))
            t_dx[pair[0], j] -= 2.0 * eps
            E2 = wp.to_torch(collision.energy(wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=1.0))
            t_dx[pair[0], j] += eps
            dEdx_fd[i, j] = (E1 - E2) / (2.0 * eps)

    assert torch.allclose(gradient, dEdx_fd, rtol=1e-2, atol=1e-2), \
        "Barrier-only collision gradients don't match finite difference"


def test_collision_friction_gradient():
    """Friction gradient with tangential motion; includes d(mu_fn)/d(offset) term."""
    device = 'cuda'
    sep = 0.08
    collision_radius = 0.05
    dt = 0.01
    impenetrable_barrier_ratio = 0.5
    friction = 0.5
    num_points_per_object = 10

    # Two slabs of 10 points each; pairs spaced along y so only matched pairs collide.
    pair_spacing = 0.5
    y = torch.arange(num_points_per_object, device=device, dtype=torch.float32) * pair_spacing
    z = torch.zeros(num_points_per_object, device=device)
    x0_obj0 = torch.stack([torch.zeros_like(y), y, z], dim=1)
    x0_obj1 = torch.stack([torch.full_like(y, sep), y, z], dim=1)
    x0_t = torch.cat([x0_obj0, x0_obj1], dim=0)
    dx_t = torch.zeros(2 * num_points_per_object, 3, device=device)
    obj_ids_t = torch.cat([
        torch.zeros(num_points_per_object, device=device, dtype=torch.int32),
        torch.ones(num_points_per_object, device=device, dtype=torch.int32),
    ])
    is_static_t = torch.zeros(2 * num_points_per_object, device=device, dtype=torch.int32)

    x0 = wp.array(x0_t, dtype=wp.vec3)
    dx = wp.array(dx_t, dtype=wp.vec3)
    obj_ids = wp.array(obj_ids_t, dtype=wp.int32)
    is_static = wp.array(is_static_t, dtype=wp.int32)

    collision = collisions.Collision(
        dt=dt,
        collision_particle_radius=collision_radius,
        detection_ratio=1.5,
        impenetrable_barrier_ratio=impenetrable_barrier_ratio,
        friction=friction,
    )
    collision.detect_collisions(dx, x0, obj_ids, is_static)
    assert collision.num_contacts == num_points_per_object

    mu_dt = dt * collision.friction_reg
    nu = collision.friction_fluid * collision.friction_reg
    rc = 2.0 * collision_radius

    t_dx = wp.to_torch(dx).clone()
    t_indices_a = wp.to_torch(collision.collision_indices_a)
    t_indices_b = wp.to_torch(collision.collision_indices_b)

    # Tangential slip on each contact's particle a (normal is along x).
    for c in range(collision.num_contacts):
        idx_a = t_indices_a[c].item()
        t_dx[idx_a, 1] = 0.005 + 0.001 * c
    dx = wp.from_torch(t_dx, dtype=wp.vec3)

    wp_gradient = wp.to_torch(collision.gradient(dx, x0, coeff=1.0))

    t_dx_start = wp.to_torch(collision.cp_dx_at_nm_iteration_0)
    t_kinematic_gaps = wp.to_torch(collision.collision_kinematic_gaps)
    t_normals = wp.to_torch(collision.collision_normals)

    #Autodiff check
    for c in range(collision.num_contacts):
        idx_a = t_indices_a[c].item()
        idx_b = t_indices_b[c].item()
        expected_gradient = _collision_contact_gradient_from_energy(
            t_dx, t_dx_start, t_kinematic_gaps[c], t_normals[c],
            idx_a, idx_b, rc, impenetrable_barrier_ratio, friction, mu_dt, nu)
        assert torch.allclose(wp_gradient[c], expected_gradient, rtol=1e-4, atol=1e-5), \
            f"Friction collision gradient for contact {c} doesn't match autodiff of analytical energy"

    # FD check with tangential motion so mu_fn chain rule matters.
    dEdx_fd = torch.zeros(collision.num_contacts, 3, device=device, dtype=t_dx.dtype)
    eps = 1e-5
    for c in range(collision.num_contacts):
        idx_a = t_indices_a[c].item()
        for j in range(3):
            t_dx[idx_a, j] += eps
            E1 = wp.to_torch(collision.energy(wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=1.0))
            t_dx[idx_a, j] -= 2.0 * eps
            E2 = wp.to_torch(collision.energy(wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=1.0))
            t_dx[idx_a, j] += eps
            dEdx_fd[c, j] = (E1 - E2) / (2.0 * eps)

    assert torch.allclose(wp_gradient, dEdx_fd, rtol=1e-3, atol=1e-5), \
        "Friction collision gradient doesn't match finite difference"


def test_collision_bounds_indexing():
    """Regression test: Jacobian offsets must be indexed with 3*c (not c).

    Each contact occupies 3 rows in the BSR Jacobian (x, y, z components).
    The bug used offsets[c] instead of offsets[3*c], causing the wrong DOF
    columns to be bounded for any contact c >= 1.

    Strategy: construct a controlled scene with exactly 2 contacts between
    distinct particle pairs, then move ONLY the particle from the SECOND
    contact (c=1) toward its partner. Verify that the DOF columns linked to
    that particle (via the correct 3*1 row offset) are bounded (< 1.0),
    while the DOF columns only reachable via the buggy (c=1) indexing —
    which would point to row 1 (y-component of contact 0) — are NOT bounded.

    Particle layout (separation = 0.08, barrier rp = 0.05, detection = 0.15):
      p0 (obj 0) at (0.0, 0, 0) -- close to --> p2 (obj 1) at (0.08, 0, 0)
      p1 (obj 0) at (1.0, 0, 0) -- close to --> p3 (obj 1) at (1.08, 0, 0)
    Both pairs are inside detection radius and outside the barrier, so
    gap_cur = rp - 0.08 = -0.03 < 0, ensuring the bounds kernel proceeds.
    """
    device = 'cuda'
    # separation chosen to be in (rp=0.05, detection=0.15)
    SEP = 0.08

    x0_t = torch.tensor([
        [0.0, 0.0, 0.0],        # p0, object 0
        [1.0, 0.0, 0.0],        # p1, object 0
        [SEP, 0.0, 0.0],        # p2, object 1
        [1.0 + SEP, 0.0, 0.0],  # p3, object 1
    ], device=device)
    dx_t = torch.zeros(4, 3, device=device)
    obj_ids_t = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.int32)
    is_static_t = torch.zeros(4, device=device, dtype=torch.int32)
    # One handle per particle (identity weights) so DOF block col == particle index
    weights_t = torch.eye(4, device=device)

    x0 = wp.array(x0_t, dtype=wp.vec3)
    dx = wp.array(dx_t, dtype=wp.vec3)
    obj_ids = wp.array(obj_ids_t, dtype=wp.int32)
    is_static = wp.array(is_static_t, dtype=wp.int32)
    weights = wp.from_torch(weights_t.contiguous())

    collision_radius = 0.05
    collision = collisions.Collision(dt=0.01,
                                     collision_particle_radius=collision_radius,
                                     detection_ratio=1.5,
                                     impenetrable_barrier_ratio=0.5,
                                     friction=0.0)

    collision.detect_collisions(dx, x0, obj_ids, is_static)
    assert collision.num_contacts == 2, \
        f"Expected exactly 2 contacts, got {collision.num_contacts}"

    collision.calculate_jacobian(weights, x0, is_static)

    J_a = collision.collision_J_a
    assert J_a.offsets.shape[0] >= 3 * 2 + 1

    t_normals = wp.to_torch(collision.collision_normals)
    t_indices_a = wp.to_torch(collision.collision_indices_a)

    # Move ONLY the particle from contact c=1 toward its partner
    t_delta_dx = torch.zeros(4, 3, device=device)
    idx_a_1 = t_indices_a[1].item()
    if idx_a_1 >= 0:
        t_delta_dx[idx_a_1] = -t_normals[1] * 0.02  # toward partner → delta_d_a < 0

    cp_delta_dx = wp.from_torch(t_delta_dx, dtype=wp.vec3)
    cp_dx = wp.from_torch(torch.zeros(4, 3, device=device), dtype=wp.vec3)

    bounds = collision.get_bounds(cp_delta_dx, cp_dx, x0)
    t_bounds = wp.to_torch(bounds)

    J_a_offsets = wp.to_torch(J_a.offsets)
    J_a_columns = wp.to_torch(J_a.columns)

    # DOF block columns that the CORRECT (3*c) indexing uses for contact c=1
    correct_row = 3 * 1
    correct_cols = J_a_columns[J_a_offsets[correct_row]:J_a_offsets[correct_row + 1]]

    # DOF block columns the BUGGY (c) indexing would have used for c=1 (row 1 = y of contact 0)
    buggy_row = 1
    buggy_cols = J_a_columns[J_a_offsets[buggy_row]:J_a_offsets[buggy_row + 1]]

    assert len(correct_cols) > 0, "No DOF columns for contact c=1 particle a"
    # Only meaningful if the bug and fix target different DOF columns
    if not torch.equal(correct_cols, buggy_cols):
        for col in correct_cols.tolist():
            s = col * J_a.block_shape[1]
            e = s + J_a.block_shape[1]
            assert (t_bounds[s:e] < 1.0).any(), \
                f"DOF block col {col} should be bounded for contact c=1 (3*c indexing)"

        buggy_only = set(buggy_cols.tolist()) - set(correct_cols.tolist())
        for col in buggy_only:
            s = col * J_a.block_shape[1]
            e = s + J_a.block_shape[1]
            assert (t_bounds[s:e] == 1.0).all(), \
                f"DOF block col {col} should NOT be bounded (only reachable via buggy c indexing)"


@pytest.mark.parametrize("test_scenes", [
    "one_object",
    "two_objects",
    "three_objects"
], indirect=True)
def test_collision_hessian(test_scenes):

    x0 = test_scenes['x0']
    dx = test_scenes['dx']
    weights = test_scenes['weights']
    obj_ids = test_scenes['obj_ids']
    is_static = test_scenes['is_static']

    # Collision parameters
    dt = 0.01
    collision_radius = 0.05
    detection_ratio = 1.5
    impenetrable_barrier_ratio = 0.5
    friction = 0.0
    collision = collisions.Collision(dt=dt,
                                     collision_particle_radius=collision_radius,
                                     detection_ratio=detection_ratio,
                                     impenetrable_barrier_ratio=impenetrable_barrier_ratio,
                                     friction=friction)  # other parameters are default

    collision.detect_collisions(dx, x0, obj_ids, is_static)
    wp_hessian_blocks = collision.hessian(dx, x0, coeff=1.0)
    hessian_blocks = wp.to_torch(wp_hessian_blocks) if wp_hessian_blocks.shape[0] > 0 else torch.zeros(
        0, 3, 3, device=wp.device_to_torch(wp_hessian_blocks.device))
    
    # diagonal matrix from nx3x3 blocks
    hessian = torch.block_diag(*hessian_blocks).to(hessian_blocks.device)
    
    t_x0 = wp.to_torch(x0)
    t_dx = wp.to_torch(dx)
    t_x = t_x0 + t_dx
    t_indices_a = wp.to_torch(collision.collision_indices_a)
    t_indices_b = wp.to_torch(collision.collision_indices_b)

    
    if weights.shape[1] == 1:
        # one object, no gradients since we have no self-collisions
        G0 = collision.gradient(dx, x0, coeff=1.0)
        assert G0.numpy().shape[0] == 0, "Gradient should be empty for one object"
        return

    # Finite difference for collision gradients
    # loop through t_x pairs of points and calculate the distance between them
    G0 = wp.to_torch(collision.gradient(dx, x0, coeff=1.0)).flatten()

    hessian_fd = torch.zeros(G0.shape[0], G0.shape[0],
                          device=t_x.device, dtype=t_x.dtype)
    eps = 1e-4
    
    hessian_row = 0
    for i in range(collision.num_contacts):
        for j in range(3):
            pair = (t_indices_a[i].item(), t_indices_b[i].item())
            t_dx[pair[0], j] += eps
            G1 = wp.to_torch(collision.gradient(
                wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=1.0)).flatten()
            t_dx[pair[0], j] -= 2.0*eps
            G2 = wp.to_torch(collision.gradient(
                wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=1.0)).flatten()
            t_dx[pair[0], j] += eps
            hessian_fd[hessian_row] = (G1 - G2) / (2.0*eps)
            hessian_row += 1

    assert torch.allclose(hessian, hessian_fd, rtol=1e-1), \
        "Collision hessian doesn't match analytical calculation"


def test_collision_friction_hessian():
    """Friction Hessian with tangential motion; FD of gradient vs analytical blocks."""
    device = 'cuda'
    sep = 0.08
    collision_radius = 0.05
    dt = 0.01
    impenetrable_barrier_ratio = 0.5
    friction = 0.5

    x0_t = torch.tensor([
        [0.0, 0.0, 0.0],
        [sep, 0.0, 0.0],
    ], device=device)
    dx_t = torch.zeros(2, 3, device=device)
    obj_ids_t = torch.tensor([0, 1], device=device, dtype=torch.int32)
    is_static_t = torch.zeros(2, device=device, dtype=torch.int32)

    x0 = wp.array(x0_t, dtype=wp.vec3)
    dx = wp.array(dx_t, dtype=wp.vec3)
    obj_ids = wp.array(obj_ids_t, dtype=wp.int32)
    is_static = wp.array(is_static_t, dtype=wp.int32)

    collision = collisions.Collision(
        dt=dt,
        collision_particle_radius=collision_radius,
        detection_ratio=1.5,
        impenetrable_barrier_ratio=impenetrable_barrier_ratio,
        friction=friction,
    )
    collision.detect_collisions(dx, x0, obj_ids, is_static)
    assert collision.num_contacts == 1

    t_dx = wp.to_torch(dx).clone()
    t_dx[0, 1] = 0.005
    dx = wp.from_torch(t_dx, dtype=wp.vec3)

    hessian_blocks = wp.to_torch(collision.hessian(dx, x0, coeff=1.0))
    hessian = hessian_blocks[0]

    idx_a = wp.to_torch(collision.collision_indices_a)[0].item()
    G0 = wp.to_torch(collision.gradient(dx, x0, coeff=1.0))[0]

    hessian_fd = torch.zeros(3, 3, device=device, dtype=t_dx.dtype)
    eps = 1e-4
    for j in range(3):
        t_dx[idx_a, j] += eps
        G1 = wp.to_torch(collision.gradient(
            wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=1.0))[0]
        t_dx[idx_a, j] -= 2.0 * eps
        G2 = wp.to_torch(collision.gradient(
            wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=1.0))[0]
        t_dx[idx_a, j] += eps
        hessian_fd[:, j] = (G1 - G2) / (2.0 * eps)

    zero_threshold = 1e-2
    analytical_nonzeros = (hessian.abs() > zero_threshold)
    fd_nonzeros = (hessian_fd.abs() > zero_threshold)
    assert torch.all(analytical_nonzeros == fd_nonzeros), \
        "Friction hessian sparsity mismatch"
    assert torch.allclose(hessian[fd_nonzeros], hessian_fd[fd_nonzeros], rtol=1e-1, atol=1e-1), \
        "Friction collision hessian doesn't match finite difference"


@pytest.mark.parametrize("test_scenes", [
    "two_objects", "three_objects", "two_objects_one_static"], indirect=True)
def test_capturable_launch_matches_host(test_scenes):
    r"""Launching over the fixed capacity with the device-count guard must match the
    host-sized launch.

    Per-contact kernels used to launch at ``dim=num_contacts``, a host value. Under
    ``capturable`` they launch at ``dim=max_contacting_pairs`` and skip inactive slots
    via ``c >= num_contacts[0]``, so no launch dimension depends on the host.

    ``two_objects_one_static`` is included deliberately: static partners are marked with
    ``NULL_ELEMENT_INDEX`` (-1), a *different* concept from "unused slot". Guarding on
    the index sign instead of the count would silently drop every contact against static
    geometry, and this case is what catches that.
    """
    x0 = test_scenes['x0']
    dx = test_scenes['dx']
    obj_ids = test_scenes['obj_ids']
    is_static = test_scenes['is_static']

    max_pairs = 512
    collision = collisions.Collision(
        dt=0.01, collision_particle_radius=0.05, detection_ratio=1.5,
        impenetrable_barrier_ratio=0.5, friction=0.5,
        max_contacting_pairs=max_pairs)
    collision.detect_collisions(dx, x0, obj_ids, is_static)

    nc = collision.num_contacts
    # Asserted, not skipped: the fixture is seeded, so zero contacts here means detection
    # regressed, and a skip would report that as green.
    assert nc > 0, "scene produced no contacts; detection has regressed"

    # Host-sized launches, freshly allocated outputs (the long-standing behaviour).
    collision.capturable = False
    g_host = torch.as_tensor(collision.gradient(dx, x0, 1.0).numpy()).clone()
    h_host = torch.as_tensor(collision.hessian(dx, x0, 1.0).numpy()).clone()

    # Full-capacity launches into preallocated buffers.
    collision.capturable = True
    g_buf = wp.zeros(max_pairs, dtype=wp.vec3, device=dx.device)
    h_buf = wp.zeros(max_pairs, dtype=wp.mat33, device=dx.device)
    g_cap = torch.as_tensor(
        collision.gradient(dx, x0, 1.0, gradient=g_buf).numpy()).clone()
    h_cap = torch.as_tensor(
        collision.hessian(dx, x0, 1.0, hessian_blocks=h_buf).numpy()).clone()
    collision.capturable = False

    # Per-slot writes with no reduction, so these must be exact -- not merely close.
    assert torch.equal(g_host, g_cap[:nc]), "gradient differs under capacity launch"
    assert torch.equal(h_host, h_cap[:nc]), "hessian differs under capacity launch"

    # The padded tail must be exactly zero or it pollutes J^T H J downstream.
    assert torch.count_nonzero(g_cap[nc:]) == 0, "gradient tail not zeroed"
    assert torch.count_nonzero(h_cap[nc:]) == 0, "hessian tail not zeroed"


@pytest.mark.parametrize("test_scenes", ["three_objects"], indirect=True)
def test_count_guard_suppresses_stale_slots(test_scenes):
    r"""The count guard must suppress slots left populated by a denser previous frame.

    This is the test that can actually fail. `test_capturable_launch_matches_host` cannot:
    it detects once on a freshly constructed Collision, so every padded slot still holds
    the construction-time `wp.zeros` (indices 0, zero normal), which evaluates to an
    exactly-zero gradient anyway. Deleting the guard leaves that test green.

    Here detection runs twice -- first on a configuration that produces many contacts,
    then on one that produces fewer -- without reconstructing the Collision. The tail
    slots are therefore populated with *valid* indices and normals from the dense frame.
    If the guard is removed or inverted, those stale contacts contribute real forces and
    the assertions below fail.
    """
    x0 = test_scenes['x0']
    obj_ids = test_scenes['obj_ids']
    is_static = test_scenes['is_static']
    n_pts = x0.shape[0]

    max_pairs = 4096
    collision = collisions.Collision(
        dt=0.01, collision_particle_radius=0.05, detection_ratio=1.5,
        impenetrable_barrier_ratio=0.5, friction=0.5,
        max_contacting_pairs=max_pairs)

    # Dense frame: a large radius pulls many pairs into contact.
    collision.collision_radius = 0.5
    dense_dx = wp.array(torch.zeros(n_pts, 3, device='cuda'), dtype=wp.vec3)
    collision.detect_collisions(dense_dx, x0, obj_ids, is_static)
    n_dense = collision.num_contacts

    # Sparse frame: shrink the radius so far fewer pairs qualify. Slots
    # [n_sparse, n_dense) now hold live data from the dense frame.
    collision.collision_radius = 0.02
    collision.detect_collisions(dense_dx, x0, obj_ids, is_static)
    n_sparse = collision.num_contacts

    assert n_dense > n_sparse > 0, \
        f"need a strictly denser first frame to leave stale slots ({n_dense} vs {n_sparse})"

    g = torch.as_tensor(collision.gradient(dense_dx, x0, 1.0).numpy())
    h = torch.as_tensor(collision.hessian(dense_dx, x0, 1.0).numpy())

    # The host-sized launch only covers [0, n_sparse), so stale slots are out of range.
    # The capacity launch covers all of them and must rely on the guard.
    g_buf = wp.zeros(max_pairs, dtype=wp.vec3, device='cuda')
    h_buf = wp.zeros(max_pairs, dtype=wp.mat33, device='cuda')
    collision.capturable = True
    g_cap = torch.as_tensor(
        collision.gradient(dense_dx, x0, 1.0, gradient=g_buf).numpy()).clone()
    h_cap = torch.as_tensor(
        collision.hessian(dense_dx, x0, 1.0, hessian_blocks=h_buf).numpy()).clone()
    collision.capturable = False

    # Live slots agree exactly.
    assert torch.equal(g, g_cap[:n_sparse])
    assert torch.equal(h, h_cap[:n_sparse])

    # The stale range must contribute nothing. Without the guard these hold real
    # contact forces carried over from the dense frame.
    stale_g = g_cap[n_sparse:n_dense]
    stale_h = h_cap[n_sparse:n_dense]
    assert torch.count_nonzero(stale_g) == 0, (
        f"{int(torch.count_nonzero(stale_g))} stale gradient entries survived in slots "
        f"[{n_sparse}, {n_dense}) -- the count guard is not suppressing them")
    assert torch.count_nonzero(stale_h) == 0, (
        f"{int(torch.count_nonzero(stale_h))} stale hessian entries survived in slots "
        f"[{n_sparse}, {n_dense})")


def _make_collision(**kwargs):
    r"""Collision configured the way every test in this file configures it."""
    params = dict(dt=0.01, collision_particle_radius=0.05, detection_ratio=1.5,
                  impenetrable_barrier_ratio=0.5, friction=0.5)
    params.update(kwargs)
    return collisions.Collision(**params)


def _detect_and_gather(test_scenes, **kwargs):
    r"""Runs one detection and returns everything the chunk kernels need.

    Detection happens exactly once and the resulting contact arrays are then shared by
    every comparison in the caller. Re-detecting from the same state would measure
    ``wp.atomic_add`` slot-assignment scheduling, not implementation agreement: the
    contact *set* is stable but its storage *order* is not.
    """
    x0, dx = test_scenes['x0'], test_scenes['dx']
    weights, obj_ids = test_scenes['weights'], test_scenes['obj_ids']
    is_static = test_scenes['is_static']

    collision = _make_collision(**kwargs)
    collision.detect_collisions(dx, x0, obj_ids, is_static)

    t_B = lbs_matrix(wp.to_torch(x0), wp.to_torch(weights)).contiguous()
    return collision, wp.from_torch(t_B), t_B


def _chunk_reference(t_B, ind_a, ind_b, start, chunk, num_contacts):
    r"""Row-gather reference for one chunk, guarding NULL_ELEMENT_INDEX on both sides."""
    device = t_B.device
    out = torch.zeros(3 * chunk, t_B.shape[1], device=device)
    offs = torch.arange(3, device=device)
    for c in range(chunk):
        g = start + c
        if g >= num_contacts:
            continue  # padding: stays zero
        for idx, sign in ((int(ind_a[g]), 1.0), (int(ind_b[g]), -1.0)):
            if idx != collisions.NULL_ELEMENT_INDEX:
                out[3 * c + offs] += sign * t_B[3 * idx + offs]
    return out


@pytest.mark.parametrize("test_scenes", ["two_objects", "three_objects"], indirect=True)
def test_jacobian_chunk_matches_dense_jacobian(test_scenes):
    r"""Every chunk reproduces the corresponding slice of the sparse-assembled Jacobian.

    This is the identity the whole chunked design rests on: the collision Jacobian is a
    pure row gather from the dense subspace basis, so a chunk can be built at any offset
    without ever materializing the full (3*max_contacting_pairs, num_dofs) matrix.
    Equality is *exact*, not approximate -- both sides gather the same floats and the
    only arithmetic is one subtraction, so any tolerance here would be hiding something.
    """
    collision, b_dense, t_B = _detect_and_gather(test_scenes)
    collision.calculate_jacobian(test_scenes['weights'], test_scenes['x0'],
                                 test_scenes['is_static'])
    num_contacts = collision.num_contacts
    assert num_contacts > 1, "fixture produced too few contacts to chunk meaningfully"

    expected = collision.collision_J_dense
    chunk = max(1, num_contacts // 2)
    chunk_start = wp.zeros(1, dtype=int)
    j_chunk = wp.zeros((3 * chunk, t_B.shape[1]), dtype=wp.float32)

    for start in range(0, num_contacts, chunk):
        chunk_start.fill_(start)
        collision.build_jacobian_chunk(b_dense, chunk_start, j_chunk)

        rows = min(chunk, num_contacts - start)
        got = wp.to_torch(j_chunk)[:3 * rows]
        want = expected[3 * start:3 * (start + rows)]
        assert torch.equal(got, want), \
            f"chunk at contact {start} differs from the dense Jacobian slice"


@pytest.mark.parametrize("test_scenes", ["two_objects", "three_objects"], indirect=True)
def test_jacobian_chunk_zeroes_past_contact_count(test_scenes):
    r"""Rows past the live contact count are zero, not stale and not garbage.

    The reduction runs over the whole chunk, so a padded row that carried any value at
    all would add a spurious rank-1 term to :math:`J^T H J`. Two cases matter: a chunk
    that straddles the count, and one entirely beyond it.
    """
    collision, b_dense, t_B = _detect_and_gather(test_scenes)
    num_contacts = collision.num_contacts
    assert num_contacts > 1

    chunk_start = wp.zeros(1, dtype=int)
    # Sized so the chunk starting one contact before the count straddles it.
    chunk = max(2, num_contacts // 2)
    j_chunk = wp.zeros((3 * chunk, t_B.shape[1]), dtype=wp.float32)

    # Poison the buffer first: zeros are only meaningful if the kernel wrote them.
    wp.to_torch(j_chunk).fill_(7.0)

    straddle = num_contacts - 1
    chunk_start.fill_(straddle)
    collision.build_jacobian_chunk(b_dense, chunk_start, j_chunk)
    got = wp.to_torch(j_chunk)
    assert got[:3].abs().sum() > 0, "the one live contact in the straddling chunk is zero"
    assert torch.equal(got[3:], torch.zeros_like(got[3:])), \
        "rows past the contact count were not zeroed"

    wp.to_torch(j_chunk).fill_(7.0)
    chunk_start.fill_(num_contacts)
    collision.build_jacobian_chunk(b_dense, chunk_start, j_chunk)
    got = wp.to_torch(j_chunk)
    assert torch.equal(got, torch.zeros_like(got)), \
        "a chunk entirely past the contact count must be all zero"


@pytest.mark.parametrize("test_scenes", ["two_objects_one_static"], indirect=True)
def test_jacobian_chunk_guards_static_sentinel(test_scenes):
    r"""A static side contributes no rows -- it must not wrap to the last point.

    ``NULL_ELEMENT_INDEX`` is -1 and Warp's ``index()`` folds negatives from the end, so
    an unguarded gather at a static contact silently reads the *last* contact point's
    rows and attributes that motion to whichever object owns it.

    The host sparse path does exactly that today (``sparse_collision_jacobian_matrix``
    populates all 3 rows for a contact whose index is -1), so ``collision_J_dense`` is
    deliberately *not* the oracle here -- the guarded gather is. This is latent rather
    than live: ``simulation.py`` calls ``detect_collisions`` with ``cp_is_static=None``,
    so production never writes the sentinel.
    """
    collision, b_dense, t_B = _detect_and_gather(test_scenes)
    num_contacts = collision.num_contacts
    ind_a = wp.to_torch(collision.collision_indices_a[:num_contacts]).cpu()
    ind_b = wp.to_torch(collision.collision_indices_b[:num_contacts]).cpu()
    static_side = (ind_a == collisions.NULL_ELEMENT_INDEX) | \
                  (ind_b == collisions.NULL_ELEMENT_INDEX)
    assert num_contacts > 0 and bool(static_side.any()), \
        "fixture produced no static-partner contacts, so this guards nothing"

    chunk_start = wp.zeros(1, dtype=int)
    j_chunk = wp.zeros((3 * num_contacts, t_B.shape[1]), dtype=wp.float32)
    collision.build_jacobian_chunk(b_dense, chunk_start, j_chunk)
    got = wp.to_torch(j_chunk)

    want = _chunk_reference(t_B, ind_a, ind_b, 0, num_contacts, num_contacts)
    assert torch.equal(got, want), \
        "static-partner contacts were not gathered with the sentinel guarded"

    # The specific failure mode: rows equal to +/- the last point's basis rows.
    last = t_B[3 * (t_B.shape[0] // 3 - 1):]
    for c in torch.nonzero(static_side).squeeze(1).tolist():
        assert not torch.equal(got[3 * c:3 * c + 3].abs(), last.abs()), \
            f"contact {c} gathered the last point's rows -- the -1 index wrapped"


@pytest.mark.parametrize("test_scenes", ["two_objects", "three_objects"], indirect=True)
def test_hessian_chunk_gather(test_scenes):
    r"""The per-contact 3x3 blocks gather by chunk, with padding zeroed."""
    collision, b_dense, t_B = _detect_and_gather(test_scenes)
    num_contacts = collision.num_contacts
    assert num_contacts > 1

    capacity = collision.max_contacting_pairs
    h_full = wp.zeros(capacity, dtype=wp.mat33)
    collision.hessian(test_scenes['dx'], test_scenes['x0'], 1.0,
                      hessian_blocks=h_full)
    t_h_full = wp.to_torch(h_full)

    chunk = max(2, num_contacts // 2)
    chunk_start = wp.zeros(1, dtype=int)
    h_chunk = wp.zeros(chunk, dtype=wp.mat33)

    for start in range(0, num_contacts, chunk):
        wp.to_torch(h_chunk).fill_(7.0)
        chunk_start.fill_(start)
        collision.gather_hessian_chunk(h_full, chunk_start, h_chunk)
        got = wp.to_torch(h_chunk)

        rows = min(chunk, num_contacts - start)
        assert torch.equal(got[:rows], t_h_full[start:start + rows]), \
            f"hessian chunk at contact {start} does not match the full array"
        assert torch.equal(got[rows:], torch.zeros_like(got[rows:])), \
            "hessian blocks past the contact count were not zeroed"


def _dense_collision_hessian(collision, weights, x0, dx, num_dofs):
    r"""Reference :math:`J^T H J` built the way the host path builds it today."""
    collision.calculate_jacobian(weights, x0)
    num_contacts = collision.num_contacts
    h_full = wp.zeros(collision.max_contacting_pairs, dtype=wp.mat33)
    collision.hessian(dx, x0, 1.0, hessian_blocks=h_full)
    t_h = wp.to_torch(h_full)[:num_contacts]
    return hess_reduction(collision.collision_J_dense, t_h), h_full, num_contacts


@pytest.mark.parametrize("test_scenes", ["two_objects", "three_objects"], indirect=True)
def test_chunked_hessian_matches_dense_reduction(test_scenes):
    r"""The chunked reduction reproduces the monolithic :math:`J^T H J`.

    Not bit-identical, and it cannot be: one large GEMM and a sum of small ones
    accumulate the same products in different orders, and float32 addition is not
    associative. The tolerance below is derived from that mechanism -- float32 epsilon
    times the number of accumulated terms times the result magnitude -- rather than
    fitted to whatever the run happens to produce.
    """
    x0, dx = test_scenes['x0'], test_scenes['dx']
    weights, obj_ids = test_scenes['weights'], test_scenes['obj_ids']

    collision = _make_collision(max_contacting_pairs=64, capturable=True)
    collision.detect_collisions(dx, x0, obj_ids, test_scenes['is_static'])

    t_B = lbs_matrix(wp.to_torch(x0), wp.to_torch(weights)).contiguous()
    num_dofs = t_B.shape[1]
    expected, h_full, num_contacts = _dense_collision_hessian(
        collision, weights, x0, dx, num_dofs)
    assert num_contacts > 1

    reducer = collisions.ChunkedCollisionHessian(collision, num_dofs, chunk_size=8)
    out = torch.zeros(num_dofs, num_dofs, device=t_B.device)
    reducer.reduce(wp.from_torch(t_B), h_full, out)

    scale = max(float(expected.abs().max()), 1.0)
    tol = torch.finfo(torch.float32).eps * (3 * num_contacts) * scale
    err = (out - expected).abs().max().item()
    assert err <= tol, \
        f"chunked reduction differs from the dense one by {err:.3e} (bound {tol:.3e})"


@pytest.mark.parametrize("test_scenes", ["three_objects"], indirect=True)
def test_chunked_hessian_is_invariant_to_chunk_size(test_scenes):
    r"""Chunking is a partition of the sum, so the split point must not matter.

    A chunk size that divides the contact count differently exercises different padding
    boundaries; if the padding were not zeroed, the results would diverge as the number
    of empty chunks changed.
    """
    x0, dx = test_scenes['x0'], test_scenes['dx']
    weights, obj_ids = test_scenes['weights'], test_scenes['obj_ids']

    collision = _make_collision(max_contacting_pairs=64, capturable=True)
    collision.detect_collisions(dx, x0, obj_ids, test_scenes['is_static'])
    t_B = lbs_matrix(wp.to_torch(x0), wp.to_torch(weights)).contiguous()
    num_dofs = t_B.shape[1]
    b_dense = wp.from_torch(t_B)

    h_full = wp.zeros(64, dtype=wp.mat33)
    collision.hessian(dx, x0, 1.0, hessian_blocks=h_full)

    results = []
    for chunk_size in (4, 7, 8, 16, 64):
        reducer = collisions.ChunkedCollisionHessian(
            collision, num_dofs, chunk_size=chunk_size)
        out = torch.zeros(num_dofs, num_dofs, device=t_B.device)
        results.append(reducer.reduce(b_dense, h_full, out).clone())

    scale = max(float(results[0].abs().max()), 1.0)
    tol = torch.finfo(torch.float32).eps * 3 * collision.num_contacts * scale
    for chunk_size, got in zip((7, 8, 16, 64), results[1:]):
        err = (got - results[0]).abs().max().item()
        assert err <= tol, \
            f"chunk_size={chunk_size} differs from chunk_size=4 by {err:.3e}"


@pytest.mark.parametrize("test_scenes", ["two_objects"], indirect=True)
def test_chunked_hessian_accepts_indivisible_chunk_size(test_scenes):
    r"""A chunk size that does not divide the capacity is allowed, and agrees.

    This used to raise, on the theory that a partial final chunk would read past the
    contact arrays. It cannot: ``count <= capacity`` is enforced by detection and by the
    device-side clamp, and all three gather kernels return early on ``g >= count``, so an
    over-hanging lane zeroes itself before touching memory. The old restriction forced
    callers to search downward for a divisor, which collapsed to a chunk size of 1 -- one
    ``capture_while`` iteration per contact -- for a capacity like 10007.
    """
    x0, dx = test_scenes['x0'], test_scenes['dx']
    weights, obj_ids = test_scenes['weights'], test_scenes['obj_ids']

    collision = _make_collision(max_contacting_pairs=64, capturable=True)
    collision.detect_collisions(dx, x0, obj_ids, test_scenes['is_static'])
    t_B = lbs_matrix(wp.to_torch(x0), wp.to_torch(weights)).contiguous()
    num_dofs = t_B.shape[1]
    b_dense = wp.from_torch(t_B)
    h_full = wp.zeros(64, dtype=wp.mat33)
    collision.hessian(dx, x0, 1.0, hessian_blocks=h_full)
    assert collision.num_contacts > 1

    out = torch.zeros(num_dofs, num_dofs, device=t_B.device)
    reference = collisions.ChunkedCollisionHessian(
        collision, num_dofs, chunk_size=8).reduce(b_dense, h_full, out).clone()

    scale = max(float(reference.abs().max()), 1.0)
    tol = torch.finfo(torch.float32).eps * (3 * collision.num_contacts) * scale
    for chunk_size in (7, 13, 60, 100):   # 100 > capacity, so it clamps to 64
        reducer = collisions.ChunkedCollisionHessian(
            collision, num_dofs, chunk_size=chunk_size)
        got = reducer.reduce(b_dense, h_full, torch.zeros_like(out))
        err = (got - reference).abs().max().item()
        assert err <= tol, f"chunk_size={chunk_size} differs by {err:.3e}"


@pytest.mark.parametrize("test_scenes", ["two_objects"], indirect=True)
def test_chunked_hessian_rejects_nonpositive_chunk_size(test_scenes):
    collision = _make_collision(max_contacting_pairs=64, capturable=True)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        collisions.ChunkedCollisionHessian(collision, 24, chunk_size=0)


def test_collision_rejects_nonpositive_capacity():
    r"""A zero capacity used to surface as a bare ZeroDivisionError from the chunk sizing."""
    with pytest.raises(ValueError, match="max_contacting_pairs must be positive"):
        _make_collision(max_contacting_pairs=0)


@pytest.mark.parametrize("test_scenes", ["two_objects"], indirect=True)
def test_chunked_hessian_ignores_stale_contacts_past_the_count(test_scenes):
    r"""Contacts written by an earlier, busier step must not leak into the reduction.

    The buffers are fixed-capacity and are not cleared between detections, so slots past
    the current count still hold the previous step's indices. Only the device count
    distinguishes live from stale -- this asserts the reduction respects it.
    """
    x0, dx = test_scenes['x0'], test_scenes['dx']
    weights, obj_ids = test_scenes['weights'], test_scenes['obj_ids']

    collision = _make_collision(max_contacting_pairs=64, capturable=True)
    collision.detect_collisions(dx, x0, obj_ids, test_scenes['is_static'])
    t_B = lbs_matrix(wp.to_torch(x0), wp.to_torch(weights)).contiguous()
    num_dofs = t_B.shape[1]
    b_dense = wp.from_torch(t_B)
    num_contacts = collision.num_contacts
    assert 0 < num_contacts < 60

    h_full = wp.zeros(64, dtype=wp.mat33)
    collision.hessian(dx, x0, 1.0, hessian_blocks=h_full)

    reducer = collisions.ChunkedCollisionHessian(collision, num_dofs, chunk_size=8)
    out = torch.zeros(num_dofs, num_dofs, device=t_B.device)
    reducer.reduce(b_dense, h_full, out)
    baseline = out.clone()

    # Forge plausible-looking contacts in the stale tail: real indices, real Hessian
    # blocks. Nothing but the count marks them dead. The helper picks the largest live
    # block deliberately -- forging with block 0 would often copy an exactly-zero barrier
    # block, and the assertion below would hold no matter what the reduction did.
    _forge_extra_contacts(collision, h_full, num_contacts, 64)

    reducer.reduce(b_dense, h_full, out)
    assert torch.equal(out, baseline), \
        "stale contacts past the device count contributed to the reduction"


def _forge_extra_contacts(collision, h_full, num_contacts, upto):
    r"""Fills stale slots with copies of the most significant live contact.

    Copies the *largest* block rather than block 0: barrier Hessians are exactly zero for
    contacts outside the barrier distance, and block 0 frequently is. Forging with a zero
    block makes every "does the count matter?" assertion pass vacuously.
    """
    t_h = wp.to_torch(h_full)
    block_norms = t_h.abs().sum(dim=(1, 2))[:num_contacts]
    big = int(block_norms.argmax())
    assert float(block_norms[big]) > 0.0, "no live contact has a nonzero Hessian block"

    t_ia = wp.to_torch(collision.collision_indices_a)
    t_ib = wp.to_torch(collision.collision_indices_b)
    t_ia[num_contacts:upto] = t_ia[big]
    t_ib[num_contacts:upto] = t_ib[big]
    t_h[num_contacts:upto] = t_h[big]


@pytest.mark.parametrize("chunk_size", [8, 7],
                         ids=["divides-capacity", "partial-final-chunk"])
@pytest.mark.parametrize("test_scenes", ["three_objects"], indirect=True)
def test_chunked_hessian_captures_and_tracks_live_contact_count(test_scenes, chunk_size):
    r"""The captured reduction replays correctly at contact counts it never saw.

    This is the point of expressing the chunk loop as ``wp.capture_while`` rather than a
    Python loop. A Python loop bakes its trip count into the graph, so a step with more
    contacts than the capture-time step would silently drop the excess. The device
    predicate makes the trip count a property of replay: capture at 2 chunks' worth of
    contacts, replay at 5, and get the 5-chunk answer without re-capturing.

    Each replay is checked against the host-side full-capacity loop, which is bit-identical
    rather than merely close -- chunks past the count contribute exact zeros, and adding
    0.0 is exact.
    """
    x0, dx = test_scenes['x0'], test_scenes['dx']
    weights, obj_ids = test_scenes['weights'], test_scenes['obj_ids']

    collision = _make_collision(max_contacting_pairs=64, capturable=True)
    collision.detect_collisions(dx, x0, obj_ids, test_scenes['is_static'])
    num_contacts = collision.num_contacts
    assert 0 < num_contacts < 16

    t_B = lbs_matrix(wp.to_torch(x0), wp.to_torch(weights)).contiguous()
    num_dofs = t_B.shape[1]
    b_dense = wp.from_torch(t_B)
    h_full = wp.zeros(64, dtype=wp.mat33)
    collision.hessian(dx, x0, 1.0, hessian_blocks=h_full)

    reducer = collisions.ChunkedCollisionHessian(collision, num_dofs,
                                                 chunk_size=chunk_size)
    out = torch.zeros(num_dofs, num_dofs, device=t_B.device)

    # chunk_size=7 does not divide the 64-contact capacity, so the last chunk hangs off
    # the end. Captured, that is the case where an unguarded lane would read past the
    # arrays -- the count guard is what makes it read nothing instead.
    #
    # No eager call first: capture must succeed from cold, which it only does because
    # the reducer forces cuBLAS to create its handle in __init__.
    graph, _ = capture_function_torch(lambda: reducer.reduce_capturable(b_dense, h_full, out))

    out.zero_()
    wp.capture_launch(graph)
    torch.cuda.synchronize()
    at_capture = out.clone()

    _forge_extra_contacts(collision, h_full, num_contacts, 40)

    seen_sums = []
    for n_live in (40, 24, 8, 0, num_contacts):
        collision.count.fill_(n_live)

        out.zero_()
        wp.capture_launch(graph)
        torch.cuda.synchronize()
        captured = out.clone()

        out.zero_()
        expected = reducer.reduce(b_dense, h_full, out).clone()

        assert torch.equal(captured, expected), \
            f"captured replay at count={n_live} differs from the host loop"
        seen_sums.append(float(captured.abs().sum()))

    assert seen_sums[-1] == pytest.approx(float(at_capture.abs().sum())), \
        "replaying at the capture-time count should reproduce the capture-time result"
    assert seen_sums[3] == 0.0, "zero contacts must reduce to zero"
    # Strictly decreasing over 40 > 24 > 8 > 0: the trip count really is following the
    # count, not replaying a fixed number of chunks.
    assert seen_sums[0] > seen_sums[1] > seen_sums[2] > seen_sums[3], \
        f"reduction did not scale with the live contact count: {seen_sums}"


@pytest.mark.parametrize("test_scenes", ["three_objects"], indirect=True)
def test_chunked_hessian_capture_allocates_nothing_on_replay(test_scenes):
    r"""Replay must not allocate.

    ``wp.capture_while`` bodies become CUDA conditional graph nodes, which are stricter
    than a plain capture: they tolerate no allocation at all. Measured with
    ``memory_stats`` rather than ``TorchDispatchMode``, which is blind to allocations made
    below the dispatcher (cuBLAS/cuSOLVER workspaces).
    """
    x0, dx = test_scenes['x0'], test_scenes['dx']
    weights, obj_ids = test_scenes['weights'], test_scenes['obj_ids']

    collision = _make_collision(max_contacting_pairs=64, capturable=True)
    collision.detect_collisions(dx, x0, obj_ids, test_scenes['is_static'])
    t_B = lbs_matrix(wp.to_torch(x0), wp.to_torch(weights)).contiguous()
    num_dofs = t_B.shape[1]
    b_dense = wp.from_torch(t_B)
    h_full = wp.zeros(64, dtype=wp.mat33)
    collision.hessian(dx, x0, 1.0, hessian_blocks=h_full)

    reducer = collisions.ChunkedCollisionHessian(collision, num_dofs, chunk_size=8)
    out = torch.zeros(num_dofs, num_dofs, device=t_B.device)
    graph, _ = capture_function_torch(lambda: reducer.reduce_capturable(b_dense, h_full, out))

    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()['allocation.all.allocated']
    for _ in range(5):
        wp.capture_launch(graph)
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()['allocation.all.allocated']

    assert after == before, f"{after - before} allocations during graph replay"


@pytest.mark.parametrize("test_scenes", ["two_objects_one_static"], indirect=True)
def test_target_distance_guards_static_sentinel_on_both_sides(test_scenes):
    r"""A contact against static geometry gets a one-radius target, not two.

    Detection enforces ``idx_a < idx_b``, so a static object added early in the scene puts
    ``NULL_ELEMENT_INDEX`` in ``indices_a`` for *all* of its contacts -- which is exactly
    what this fixture produces. A helper that tested only ``indices_b`` would return
    ``2*radius`` for every one of them.

    Asserted through the energy, because ``rc`` is not observable on its own: the barrier
    is active only for ``rp_ratio < d/rc <= 1``, so doubling ``rc`` halves ``d_hat`` and
    changes the energy by a large factor rather than a roundoff. The analytical helper is
    evaluated at both candidate ``rc`` values and the kernel must match the one-radius one.
    """
    x0, dx = test_scenes['x0'], test_scenes['dx']
    obj_ids, is_static = test_scenes['obj_ids'], test_scenes['is_static']

    radius, barrier_ratio, friction = 0.05, 0.5, 0.5
    dt = 0.01
    collision = _make_collision(collision_particle_radius=radius,
                                impenetrable_barrier_ratio=barrier_ratio,
                                friction=friction, dt=dt)
    collision.detect_collisions(dx, x0, obj_ids, is_static)
    num_contacts = collision.num_contacts

    ind_a = wp.to_torch(collision.collision_indices_a[:num_contacts]).cpu()
    ind_b = wp.to_torch(collision.collision_indices_b[:num_contacts]).cpu()
    assert num_contacts > 0 and bool((ind_a == collisions.NULL_ELEMENT_INDEX).all()), (
        "fixture must put the sentinel in indices_a; it is the case the old code missed")
    assert bool((ind_b >= 0).all()), "indices_b should be live for this fixture"

    coeff = 1.0
    energy = wp.zeros(1, dtype=float)
    collision.energy(dx, x0, coeff, energy)
    got = float(wp.to_torch(energy)[0])

    t_dx = wp.to_torch(dx)
    t_gaps = wp.to_torch(collision.collision_kinematic_gaps)
    t_nor = wp.to_torch(collision.collision_normals)
    t_start = wp.to_torch(collision.cp_dx_at_nm_iteration_0)
    mu_dt = dt * collision.friction_reg

    def analytical(rc):
        total = 0.0
        for c in range(num_contacts):
            # idx_a is the sentinel here, so side a contributes no motion.
            b = int(ind_b[c])
            offset = t_gaps[c] - (t_dx[b] - t_start[b])
            total += float(_collision_contact_energy_analytical(
                offset, t_nor[c], rc, barrier_ratio, friction, mu_dt,
                collision.friction_fluid))
        return total

    want_1r = analytical(1.0 * radius)
    want_2r = analytical(2.0 * radius)
    assert abs(want_1r - want_2r) > 1e-6 * max(abs(want_1r), 1.0), (
        "the two target distances give the same energy here, so this test cannot "
        "distinguish them -- adjust the fixture geometry")

    tol = 1e-4 * max(abs(want_1r), 1.0)
    assert abs(got - want_1r) <= tol, (
        f"energy {got:.6e} does not match the one-radius target {want_1r:.6e}; "
        f"the two-radius value is {want_2r:.6e}, so the sentinel guard on indices_a "
        f"is not being applied")

    # want_1r is 0.0 for this fixture -- at one radius these pairs sit outside the
    # barrier band entirely, which is the point: the unguarded helper invented contact
    # forces for pairs that are not touching. But "expected 0" would also be satisfied by
    # an energy kernel that always returned 0, so cross-check against a scene where the
    # correct answer is non-zero. Doubling the radius makes the guarded one-radius target
    # equal the old unguarded two-radius one, so the kernel must now reproduce want_2r.
    doubled = _make_collision(collision_particle_radius=2.0 * radius,
                              impenetrable_barrier_ratio=barrier_ratio,
                              friction=friction, dt=dt)
    doubled.detect_collisions(dx, x0, obj_ids, is_static)
    energy_2r = wp.zeros(1, dtype=float)
    doubled.energy(dx, x0, coeff, energy_2r)
    got_2r = float(wp.to_torch(energy_2r)[0])
    assert abs(want_2r) > 1e-6, "cross-check is vacuous if the two-radius energy is zero"
    assert abs(got_2r - want_2r) <= 1e-4 * abs(want_2r), (
        f"at twice the radius the guarded helper should reproduce the two-radius energy "
        f"{want_2r:.6e}, got {got_2r:.6e}")


@pytest.mark.parametrize("test_scenes", ["three_objects"], indirect=True)
def test_capturable_energy_ignores_stale_contacts_past_the_count(test_scenes):
    r"""The energy kernel's stale-slot guard is load-bearing and until now untested.

    Every other ``.energy(`` call in this file builds a ``Collision`` with the default
    ``capturable=False``, where the launch dimension *is* the contact count and the guard
    is trivially true. So deleting ``if c >= num_contacts[0]: return`` from
    ``_collision_energy_wp_kernel`` left the whole suite green while feeding a captured
    line search barrier energy from the previous frame's contacts.

    Same shape as ``test_count_guard_suppresses_stale_slots``, which covers the gradient
    and hessian kernels. Tolerance rather than equality: energy is an ``atomic_add`` into
    a single scalar, so the accumulation order varies between launches even for identical
    inputs.
    """
    x0, dx = test_scenes['x0'], test_scenes['dx']
    obj_ids, is_static = test_scenes['obj_ids'], test_scenes['is_static']

    collision = _make_collision(max_contacting_pairs=64, capturable=True)
    collision.detect_collisions(dx, x0, obj_ids, is_static)
    num_contacts = collision.num_contacts
    assert 0 < num_contacts < 60, f"need headroom for stale slots, got {num_contacts}"

    coeff = 1.0
    baseline = wp.zeros(1, dtype=float)
    collision.energy(dx, x0, coeff, baseline)
    want = float(wp.to_torch(baseline)[0])
    assert abs(want) > 0.0, "energy is zero, so stale contributions would be invisible"

    # Fill the dead tail with copies of a real contact: valid indices, valid normals,
    # valid gaps. Only the device count marks them as no longer live.
    #
    # Copy the most *active* contact, not slot 0. The barrier is piecewise -- zero unless
    # rp_ratio < d/rc <= 1 -- so a typical contact contributes exactly nothing to the
    # energy, and forging from one makes this test pass against a kernel with no guard at
    # all. Per-contact gradient magnitude identifies an active one.
    dEdx = wp.zeros(collision.max_contacting_pairs, dtype=wp.vec3)
    collision.gradient(dx, x0, coeff, gradient=dEdx)
    per_contact = wp.to_torch(dEdx)[:num_contacts].abs().sum(dim=1)
    active = int(per_contact.argmax())
    assert float(per_contact[active]) > 0.0, \
        "no contact is inside the barrier, so a forged copy would contribute nothing"

    t_ia = wp.to_torch(collision.collision_indices_a)
    t_ib = wp.to_torch(collision.collision_indices_b)
    t_nor = wp.to_torch(collision.collision_normals)
    t_gap = wp.to_torch(collision.collision_kinematic_gaps)
    for arr in (t_ia, t_ib, t_nor, t_gap):
        arr[num_contacts:] = arr[active]

    after = wp.zeros(1, dtype=float)
    collision.energy(dx, x0, coeff, after)
    got = float(wp.to_torch(after)[0])

    # float32 atomic reduction over the same summands in a different order.
    tol = torch.finfo(torch.float32).eps * (num_contacts ** 0.5) * abs(want)
    assert abs(got - want) <= max(tol, 1e-12), (
        f"stale contacts past the device count contributed {got - want:.6e} to the "
        f"energy (baseline {want:.6e}); the count guard is not suppressing them")
