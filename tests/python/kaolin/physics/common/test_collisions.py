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
        
        # make one object static
        _stacked_is_static[0] = torch.ones(num_points, device=device)

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
    friction = 0.0
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
    "one_object",
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
    collision.get_jacobian(weights, x0, is_static)
    collision_jacobian = collision.collision_J_dense

    if collision.num_contacts == 0:
        return
    
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
    t_is_static = wp.to_torch(is_static)

    # Analytically calculate collision energy
    # loop through all pairs of points and calculate the distance between them
    expected_energy = torch.tensor(0.0, device=t_x.device, dtype=t_x.dtype)
    for i in range(len(t_x0)):
        for j in range(i+1, len(t_x0)):
            if t_obj_ids[i] == t_obj_ids[j]:
                pass  # ignore self collisions
            else:
                pos_a = t_x[i]
                pos_b = t_x[j]
                dist = (pos_a - pos_b).norm()
                normal = (pos_a - pos_b)/dist
                
                kinematic_gap = torch.zeros(3, device=t_x.device, dtype=t_x.dtype)
                offset = pos_a - pos_b 
                rc = 2.0*collision_radius

                if impenetrable_barrier_ratio*rc < dist and dist<=rc:
                    d_hat = dist/rc 
                    d_min_l_squared = (d_hat - 1.0) * (d_hat - 1.0)  # quadratic term ensures energy is 0 when d = rc
                    expected_energy += -d_min_l_squared * torch.log(d_hat - impenetrable_barrier_ratio)  # log barrier term. inf when two rp's overlaps.


                    #### friction energy
                    dc = d_hat - 1.0
                    dp = d_hat - impenetrable_barrier_ratio
                    barrier = 2.0 * torch.log(dp)

                    dE_d_hat = -dc * (barrier + dc / dp)
                    vt = (offset - dist * normal) / mu_dt  # tangential velocity
                    vt_norm = vt.norm()

                    mu_fn = -mu * dE_d_hat / rc  # yield force
                    expected_energy += (
                        mu_fn * mu_dt
                        * (
                            0.5 * nu * vt_norm * vt_norm
                            + torch.where(
                                vt_norm < 1.0,
                                vt_norm * vt_norm * (1.0 - vt_norm / 3.0),
                                vt_norm - 1.0 / 3.0
                            )
                        )
                    )
                else:
                    expected_energy += 0.0


    assert torch.allclose(wp.to_torch(energy)[0], expected_energy, rtol=1e-5), \
        "Collision energy doesn't match analytical calculation"

        
@pytest.mark.parametrize("test_scenes", [
    "one_object",
    "two_objects",
    "three_objects"
], indirect=True)
def test_collision_gradient(test_scenes):

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
                                     friction=friction) # other parameters are default

    collision.detect_collisions(dx, x0, obj_ids, is_static)
    wp_gradient = collision.gradient(dx, x0, coeff=1.0)
    gradient = wp.to_torch(wp_gradient) if wp_gradient.shape[0] > 0 else torch.zeros(0, 3, device=wp.device_to_torch(wp_gradient.device))
    
    
    t_x0 = wp.to_torch(x0)
    t_dx = wp.to_torch(dx)
    t_x = t_x0 + t_dx
    t_indices_a = wp.to_torch(collision.collision_indices_a)
    t_indices_b = wp.to_torch(collision.collision_indices_b)

    # Finite difference for collision gradients
    # loop through t_x pairs of points and calculate the distance between them
    E0 = wp.to_torch(collision.energy(dx, x0, coeff=1.0))
    dEdx_fd = torch.zeros(collision.num_contacts, 3, device=t_x.device, dtype=t_x.dtype)
    eps = 1e-5
    for i in range(dEdx_fd.shape[0]):
        for j in range(dEdx_fd.shape[1]):
            pair = (t_indices_a[i].item(), t_indices_b[i].item())
            t_dx[pair[0],j] += eps
            E1 = wp.to_torch(collision.energy(wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=1.0))
            t_dx[pair[0], j] -= 2.0*eps
            E2 = wp.to_torch(collision.energy(wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=1.0))
            t_dx[pair[0], j] += eps
            dEdx_fd[i, j] = (E1 - E2) / (2.0*eps)


    assert torch.allclose(gradient, dEdx_fd, rtol=1e-1), \
        "Collision gradients doesn't match analytical calculation"
        

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

    collision.get_jacobian(weights, x0, is_static)

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
    eps = 1e-3
    
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
