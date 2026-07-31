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
from kaolin.utils.testing import with_seed


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
    if nc == 0:
        pytest.skip("scene produced no contacts; comparison would be vacuous")

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


@pytest.mark.parametrize("test_scenes", ["two_objects", "three_objects"], indirect=True)
def test_pair_matrix_matches_host_object_pairs(test_scenes):
    r"""The device-side pair matrix must agree exactly with the host object_pairs list.

    `object_pairs` is built with torch.unique(...).cpu().numpy() -- three device syncs
    and a variable-length host array driving a Python loop in the Hessian assembly. The
    matrix is its capturable replacement: a single element of it is a legal
    `wp.capture_if` predicate, so per-pair blocks can be skipped on device.

    It is filled during detection rather than by a later kernel, because after detection
    a static point's index is NULL_ELEMENT_INDEX and its object identity is gone -- and
    a -1 subscript into the object map would wrap to the last point, attributing the
    contact to the wrong object. `two_objects_one_static` is what covers that.
    """
    x0 = test_scenes['x0']
    dx = test_scenes['dx']
    obj_ids = test_scenes['obj_ids']
    is_static = test_scenes['is_static']
    n_obj = int(wp.to_torch(obj_ids).max()) + 1

    collision = collisions.Collision(
        dt=0.01, collision_particle_radius=0.5, detection_ratio=1.5,
        impenetrable_barrier_ratio=0.5, friction=0.5,
        max_contacting_pairs=4096, num_objects=n_obj)
    collision.detect_collisions(dx, x0, obj_ids, is_static)

    if collision.num_contacts == 0:
        pytest.skip("scene produced no contacts")

    device_matrix = torch.as_tensor(collision.collision_pair_matrix.numpy())

    # Rebuild the same relation from the host-side list.
    host_matrix = torch.zeros(n_obj, n_obj, dtype=device_matrix.dtype)
    for i, j in collision.object_pairs:
        host_matrix[int(i), int(j)] = 1

    assert torch.equal(device_matrix, host_matrix), (
        f"device pair matrix disagrees with host object_pairs\n"
        f"device:\n{device_matrix}\nhost:\n{host_matrix}")

    # Symmetric with self-pairs set, matching what the Hessian assembly needs.
    assert torch.equal(device_matrix, device_matrix.T), "pair matrix must be symmetric"


@pytest.mark.parametrize("test_scenes", ["three_objects"], indirect=True)
def test_pair_matrix_clears_between_detections(test_scenes):
    r"""Stale pairs must not persist: detection only ever sets entries, never clears."""
    x0 = test_scenes['x0']
    dx = test_scenes['dx']
    obj_ids = test_scenes['obj_ids']
    is_static = test_scenes['is_static']
    n_obj = int(wp.to_torch(obj_ids).max()) + 1

    collision = collisions.Collision(
        dt=0.01, collision_particle_radius=0.5, detection_ratio=1.5,
        impenetrable_barrier_ratio=0.5, friction=0.5,
        max_contacting_pairs=4096, num_objects=n_obj)

    collision.detect_collisions(dx, x0, obj_ids, is_static)
    dense = torch.as_tensor(collision.collision_pair_matrix.numpy()).clone()

    # Shrink the radius so nothing is in contact; the matrix must go empty.
    collision.collision_radius = 1e-6
    collision.detect_collisions(dx, x0, obj_ids, is_static)
    sparse = torch.as_tensor(collision.collision_pair_matrix.numpy())

    assert dense.sum() > 0, "first detection should have found contacts"
    assert collision.num_contacts == 0, "second detection should have found none"
    assert sparse.sum() == 0, (
        f"pair matrix retained {int(sparse.sum())} stale entries from the previous frame")


@pytest.mark.parametrize("test_scenes", ["two_objects_one_static"], indirect=True)
def test_pair_matrix_records_static_contacts_that_host_list_loses(test_scenes):
    r"""With a static object the device matrix is correct and `object_pairs` is not.

    `object_pairs` is built host-side as `obj_ids[ind_a]`, but by then a static point's
    index is NULL_ELEMENT_INDEX (-1), and torch wraps a -1 subscript to the LAST entry.
    So the static object's contacts get attributed to whichever object owns the last
    contact point, and the genuine (static, dynamic) coupling disappears from the list.
    The Hessian assembly loops over that list, so the block coupling a kinematic collider
    to a dynamic body would never be assembled.

    It is not fixable host-side: once the sentinel is written the object identity is
    gone. That is precisely why the pair matrix is filled inside the detection kernel,
    where obj_a and obj_b are still in scope. This test pins the difference so the
    device matrix is not "corrected" to match the broken list.
    """
    x0 = test_scenes['x0']
    dx = test_scenes['dx']
    obj_ids = test_scenes['obj_ids']
    is_static = test_scenes['is_static']

    collision = collisions.Collision(
        dt=0.01, collision_particle_radius=0.5, detection_ratio=1.5,
        impenetrable_barrier_ratio=0.5, friction=0.5,
        max_contacting_pairs=4096, num_objects=2)
    collision.detect_collisions(dx, x0, obj_ids, is_static)
    assert collision.num_contacts > 0

    device_matrix = torch.as_tensor(collision.collision_pair_matrix.numpy())

    # Object 0 is static, object 1 dynamic, and they are in contact. The device matrix
    # must record the cross coupling in both directions.
    assert device_matrix[0, 1] == 1 and device_matrix[1, 0] == 1, (
        f"device matrix lost the static/dynamic coupling:\n{device_matrix}")

    # The host list, by contrast, never mentions object 0 -- documenting the bug rather
    # than asserting the matrix should reproduce it.
    host_objects = {int(i) for pair in collision.object_pairs for i in pair}
    assert 0 not in host_objects, (
        "object_pairs unexpectedly contains the static object -- if this now passes, "
        "the host-side -1 wraparound has been fixed and this test should become an "
        "equality check against the device matrix")
