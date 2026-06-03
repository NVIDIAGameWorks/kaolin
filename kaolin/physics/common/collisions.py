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

import logging
import torch
import warp as wp
import warp.sparse as wps

from kaolin.physics.simplicits.precomputed import sparse_collision_jacobian_matrix
from kaolin.physics.utils.warp_utilities import _bsr_to_torch, _warp_csr_from_torch_dense

__all__ = ['Collision']

# TODO: Separate the cps from qps. Currently we use qps for both.
# TODO: Currently self collisions are disabled via high immune radius.
# TODO: Consider floor friction, alternatively use a floor object and set friction for that
# TODO: Object-wise friction parameters
# TODO: Allow variable collision radii for different objects.


NULL_ELEMENT_INDEX = wp.constant(-1)


@wp.kernel
def _detect_particle_collisions_wp_kernel(
    max_contacts: int,                    # max number of contacts to detect
    grid: wp.uint64,                      # hashgrid for current points
    radius: float,                        # collision radius
    self_collision_immune_radius: float,  # ignore self collisions within radius
    pos_cur: wp.array(dtype=wp.vec3),     # current positions of points B*z + x0
    pos_rest: wp.array(dtype=wp.vec3),    # rest positions of points x0
    pos_delta: wp.array(dtype=wp.vec3),   # displacements of points ... velocity of points * dt (how much they moved in current timestep)B*z_k - B*z_0 where k is newton iteration
    qp_obj_ids: wp.array(dtype=int),      # point to object id mapping
    cp_is_static: wp.array(dtype=int),    # 1 for true, 0 for false
    count: wp.array(dtype=int),           # number of contacts detected
    normals: wp.array(dtype=wp.vec3),     # contact normals
    kinematic_gaps: wp.array(dtype=wp.vec3),  # kinematic gaps
    indices_a: wp.array(dtype=int),       # collision indices pairs a-b
    indices_b: wp.array(dtype=int),       # collision indices pairs a-b
):  # pragma: no cover
    tid = wp.tid()

    # Get current point's index, object and position
    idx_a = wp.hash_grid_point_id(grid, tid)
    obj_a = qp_obj_ids[idx_a]
    pos_a = pos_cur[idx_a]

    # Query grid for nearby points within radius
    for idx_b in wp.hash_grid_query(grid, pos_a, radius):
        if idx_a >= idx_b:
            continue  # symmetric, already checked, skip

        # If both points are in the same object,
        # and they're within the immune radius, skip
        obj_b = qp_obj_ids[idx_b]
        if (
            obj_a == obj_b
            and wp.length_sq(pos_rest[idx_a] - pos_rest[idx_b])
            < self_collision_immune_radius
        ):
            continue  # nearby points in the same object, skip

        # If the points are within the radius,
        # we have a collision
        pos_b = pos_cur[idx_b]
        d = wp.length(pos_a - pos_b)
        if d <= radius:
            # indx of the current collision pair
            idx = wp.atomic_add(count, 0, 1)
            if idx >= max_contacts:
                return

            n = wp.normalize(pos_a - pos_b)
            normals[idx] = n

            # The kinematic gap tracks how far apart the points would be without their current motion.
            kinematic_gaps[idx] = (
                wp.dot(pos_a - pos_b -
                       (pos_delta[idx_a] - pos_delta[idx_b]), n) * n
            )
            
            
            if cp_is_static[idx_a] == 1:
                indices_a[idx] = NULL_ELEMENT_INDEX
            else:
                indices_a[idx] = idx_a
                
            if cp_is_static[idx_b] == 1:
                indices_b[idx] = NULL_ELEMENT_INDEX
            else:
                indices_b[idx] = idx_b


@wp.func
def _collision_offset_wp_func(
    c: int,
    dx_cur: wp.array(dtype=wp.vec3),
    dx_start_of_timestep: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):  # pragma: no cover
    r"""
    Compute the relative offset between two colliding points.
    
    This function calculates the current relative position between two points involved
    in a collision, accounting for their motion since the start of the timestep and
    any kinematic gaps (initial separations).
    
    Args:
        c (int): Index of the collision pair
        dx_cur (wp.array(dtype=wp.vec3)): Current displacements of all points
        dx_start_of_timestep (wp.array(dtype=wp.vec3)): Displacements at the start of the timestep
        kinematic_gaps (wp.array(dtype=wp.vec3)): Initial separation vectors between colliding points
        indices_a (wp.array(dtype=int)): Indices of the first point in each collision pair
        indices_b (wp.array(dtype=int)): Indices of the second point in each collision pair
        
    Returns:
        wp.vec3: The relative offset vector from point B to point A, accounting for
                motion and kinematic gaps. If point B is static (NULL_ELEMENT_INDEX),
                only point A's motion is considered.
    """
    idx_a = indices_a[c]
    idx_b = indices_b[c]

    pos_delta_a = dx_cur[idx_a] - dx_start_of_timestep[idx_a]
    pos_delta_b = dx_cur[idx_b] - dx_start_of_timestep[idx_b]

    offset = pos_delta_a + kinematic_gaps[c]
    if idx_b != NULL_ELEMENT_INDEX:
        offset -= pos_delta_b
    return offset


@wp.func
def _collision_target_distance_wp_func(
    c: int,
    radius: float,
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):  # pragma: no cover
    return wp.where(indices_b[c] == NULL_ELEMENT_INDEX, 1.0, 2.0) * radius
    # return 2.0 * radius


@wp.kernel
def _collision_energy_wp_kernel(
    coeff: float,
    radius: float,
    barrier_distance_ratio: float,
    mu: float,
    dt: float,
    nu: float,
    dx_cur: wp.array(dtype=wp.vec3),
    dx_start_of_timestep: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
    energies: wp.array(dtype=float),
):  # pragma: no cover
    r"""
    Compute the collision energy for each collision pair.
    
    This function calculates the energy of a collision pair, accounting for the barrier distance ratio,
    the kinematic gaps, and the normal vector. 
    
    Args:
        dx_cur (wp.array(dtype=wp.vec3)): Current displacements of all points
        dx_start_of_timestep (wp.array(dtype=wp.vec3)): Displacements at the start of the timestep
        kinematic_gaps (wp.array(dtype=wp.vec3)): Initial separation vectors between colliding points
        normals (wp.array(dtype=wp.vec3)): Normal vectors of the collision pairs
        indices_a (wp.array(dtype=int)): Indices of the first point in each collision pair
        indices_b (wp.array(dtype=int)): Indices of the second point in each collision pair
        energies (wp.array(dtype=float)): Energy of each collision pair
    """
    c = wp.tid()

    offset = _collision_offset_wp_func(
        c, dx_cur, dx_start_of_timestep, kinematic_gaps, indices_a, indices_b)
    rc = _collision_target_distance_wp_func(c, radius, indices_a, indices_b)
    rp_ratio = barrier_distance_ratio

    nor = normals[c]
    d = wp.dot(offset, nor)
    d_hat = d / rc

    # Check if within the active collision range: d_hat in (rp_ratio, 1]
    if rp_ratio < d_hat and d_hat <= 1.0:
        # d_hat = d / rc, where d is the normal-projected gap and rc is the collision target distance
        # d_min_l_squared is a quadratic barrier, nonzero when d_hat < 1 (objects start to overlap)
        d_min_l_squared = (d_hat - 1.0) * (
            d_hat - 1.0
        )  # quadratic penalty, ensures E is 0 at d = rc (no contact), positive when overlapping

        # Log barrier energy: becomes infinite as the gap closes to rp_ratio (impenetrable barrier)
        E = -d_min_l_squared * wp.log(
            d_hat - rp_ratio
        )  # adds infinite energy as d_hat approaches rp_ratio

        # --- Friction (tangential) energy terms ---
        # dc is distance past rc threshold (negative inside collision)
        dc = d_hat - 1.0
        # dp is gap past the hard barrier
        dp = d_hat - rp_ratio
        # 'barrier' is twice the log barrier, appears in the friction yield force
        barrier = 2.0 * wp.log(dp)

        # dE_d_hat: derivative of the barrier energy w.r.t. d_hat, sets normal force scale
        dE_d_hat = -dc * (barrier + dc / dp)

        # Tangential velocity: subtract projection onto the normal, dividing by dt
        vt = (offset - d * nor) / dt  # tangential slip vector per unit time
        vt_norm = wp.length(vt)       # tangential slip speed

        # mu_fn: yield friction force magnitude, scales with normal force
        mu_fn = -mu * dE_d_hat / rc  # Coulomb friction coefficient (frictional yield magnitude)

        # Add frictional (dissipative and regularized yield) energy term:
        #   - 0.5 * nu * vt_norm^2: fluid-style (velocity-squared) regularizer
        #   - (case statement): regularized stick-slip below vt_norm=1, classic friction above
        E += (
            mu_fn
            * dt
            * (
                0.5 * nu * vt_norm * vt_norm
                + wp.where(
                    vt_norm < 1.0,
                    vt_norm * vt_norm * (1.0 - vt_norm / 3.0),
                    vt_norm - 1.0 / 3.0,
                )
            )
        )
        ##

    else:
        # Outside the collision range (no overlap) — energy is zero
        E = 0.0

    wp.atomic_add(energies, 0, coeff * E)


@wp.kernel
def _collision_gradient_wp_kernel(coeff: float,
                        radius: float,
                        barrier_distance_ratio: float,
                        mu: float,
                        dt: float,
                        nu: float,
                        dx_cur: wp.array(dtype=wp.vec3),
                        dx_start_of_timestep: wp.array(dtype=wp.vec3),
                        kinematic_gaps: wp.array(dtype=wp.vec3),
                        normals: wp.array(dtype=wp.vec3),
                        indices_a: wp.array(dtype=int),
                        indices_b: wp.array(dtype=int),
                                  gradient: wp.array(dtype=wp.vec3)):  # pragma: no cover
    r"""
    Calculates the collision gradient for each collision pair.
    
    This function calculates the gradient of the collision energy for each collision pair,
    accounting for the barrier distance ratio, the kinematic gaps, and the normal vector.
    
    Args:
        dx_cur (wp.array(dtype=wp.vec3)): Current displacements of all points
        dx_start_of_timestep (wp.array(dtype=wp.vec3)): Displacements at the start of the timestep
        kinematic_gaps (wp.array(dtype=wp.vec3)): Initial separation vectors between colliding points
        normals (wp.array(dtype=wp.vec3)): Normal vectors of the collision pairs
        indices_a (wp.array(dtype=int)): Indices of the first point in each collision pair
        indices_b (wp.array(dtype=int)): Indices of the second point in each collision pair
        gradient (wp.array(dtype=wp.vec3)): Gradient of the collision energy for each collision pair
    """
    c = wp.tid()

    offset = _collision_offset_wp_func(
        c, dx_cur, dx_start_of_timestep, kinematic_gaps, indices_a, indices_b)
    rc = _collision_target_distance_wp_func(c, radius, indices_a, indices_b)
    rp_ratio = barrier_distance_ratio

    nor = normals[c]
    d = wp.dot(offset, nor)
    d_hat = d / rc

    # The following block computes the gradient of the barrier collision energy and friction term.
    # 
    # Let:
    # - d_hat: normalized gap along the collision normal (distance between points divided by the "collision radius rc")
    # - rp_ratio: normalized ratio for beginning of the barrier region ("impenetrable barrier ratio")
    # - dc = d_hat - 1.0: signed penetration depth normalized by rc (negative if interpenetrating)
    # - dp = d_hat - rp_ratio: normalized distance into the barrier energy region
    # 
    # The collision (barrier) potential is nonzero only when rp_ratio < d_hat <= 1.0,
    # i.e., contacts are within the active barrier region but haven't separated.

    if rp_ratio < d_hat and d_hat <= 1.0:
        dc = d_hat - 1.0
        dp = d_hat - rp_ratio
        barrier = 2.0 * wp.log(dp)  # log-barrier term in the potential

        # dE/d(d_hat): 
        #   Derivative of the barrier energy with respect to normalized displacement.
        #   This combines the log-barrier slope plus a quadratic for distance inside the region.
        dE_d_hat = -dc * (barrier + dc / dp)

        # Chain rule: convert to Cartesian gradient in world space along the normal direction
        gradient[c] = dE_d_hat / rc * nor

        # ---- Friction Terms ----
        # Friction acts tangentially to the contact plane.
        # vt: tangential slip/velocity between contacting points
        vt = (offset - d * nor) / dt  # tangential velocity
        vt_norm = wp.length(vt)

        # Effective friction force magnitude ("yield force", a la Coulomb friction).
        # Proportional to normal barrier force (mu_fn ~ mu * |normal force|).
        mu_fn = -mu * dE_d_hat / rc  # yield force

        # Nonlinear regularization for vt_norm -> 0 (improves smoothness)
        # - For vt_norm < 1, f1_over_vt_norm = 2-vt_norm (quadratic regularization)
        # - For vt_norm >= 1, f1_over_vt_norm = 1/vt_norm (Coulomb friction regime)
        f1_over_vt_norm = wp.where(
            vt_norm < 1.0,  2.0 - vt_norm, 1.0 / vt_norm)

        # Add friction term to the gradient in tangential direction (proportional to mu, vt, and fluid regularization nu)
        gradient[c] += mu_fn * (f1_over_vt_norm + nu) * vt

        # H_vt: Dissipation term for differentiable stick/slip friction (see DFG/PolyFric)
        h_vt = (
            0.5 * nu * vt_norm * vt_norm
            + wp.where(
                vt_norm < 1.0,
                vt_norm * vt_norm * (1.0 - vt_norm / 3.0),
                vt_norm - 1.0 / 3.0,
            )
        )

        # Second derivatives of the barrier energy with respect to d_hat, needed for friction gradient's nontrivial Jacobian
        dbarrier_d_hat = 2.0 / dp
        ddcdp_d_hat = (dp - dc) / (dp * dp)
        d2E_d_hat2 = -(barrier + dc / dp) - dc * (dbarrier_d_hat + ddcdp_d_hat)

        # Add the last frictional correction term, projected along the normal (stick/slip dissipation effect)
        gradient[c] += -mu * dt * h_vt * d2E_d_hat2 / (rc * rc) * nor
        ###

    else:
        # Outside the barrier/penetration region: energy and its gradient are zero
        gradient[c] = wp.vec3(0.0)

    # Scale by the collision coefficient (typically 1.0 for main collision energy, 0.0 for friction)
    gradient[c] = coeff * gradient[c]


@wp.kernel
def _collision_hessian_diag_blocks_wp_kernel(coeff: float,
                                   radius: float,
                                   barrier_distance_ratio: float,
                                   mu: float,
                                   dt: float,
                                   nu: float,
                                   dx_cur: wp.array(dtype=wp.vec3),
                                   dx_start_of_timestep: wp.array(dtype=wp.vec3),
                                   kinematic_gaps: wp.array(dtype=wp.vec3),
                                   normals: wp.array(dtype=wp.vec3),
                                   indices_a: wp.array(dtype=int),
                                   indices_b: wp.array(dtype=int),
                                             hessian: wp.array(dtype=wp.mat33)):  # pragma: no cover
    r"""
    Compute the Hessian of the collision energy for each collision pair.

    This function calculates the Hessian of the collision energy for each collision pair,
    accounting for the barrier distance ratio, the kinematic gaps, and the normal vector.

    Args:
        dx_cur (wp.array(dtype=wp.vec3)): Current displacements of all points
        dx_start_of_timestep (wp.array(dtype=wp.vec3)): Displacements at the start of the timestep
        kinematic_gaps (wp.array(dtype=wp.vec3)): Initial separation vectors between colliding points
        normals (wp.array(dtype=wp.vec3)): Normal vectors of the collision pairs
        indices_a (wp.array(dtype=int)): Indices of the first point in each collision pair
        indices_b (wp.array(dtype=int)): Indices of the second point in each collision pair
        hessian (wp.array(dtype=wp.mat33)): Hessian of the collision energy for each collision pair
    """
    c = wp.tid()

    offset = _collision_offset_wp_func(
        c, dx_cur, dx_start_of_timestep, kinematic_gaps, indices_a, indices_b)
    rc = _collision_target_distance_wp_func(c, radius, indices_a, indices_b)
    rp_ratio = barrier_distance_ratio

    nor = normals[c]
    d = wp.dot(offset, nor)
    d_hat = d / rc

    if rp_ratio < d_hat and d_hat <= 1.0:
        # dc = normal projected gap past contact threshold (dc < 0 when in contact)
        dc = d_hat - 1.0
        # dp = gap past the barrier; dp ~ 0 at hard collision, dp > 0 for separation
        dp = d_hat - rp_ratio
        # barrier is 2 * log(dp); diverges as dp -> 0, the log barrier for impenetrability
        barrier = 2.0 * wp.log(dp)

        # dE/d_dhat: normal force scale, from differentiating the log barrier energy
        dE_d_hat = -dc * (barrier + dc / dp)

        # First and second derivatives of barrier term wrt d_hat:
        dbarrier_d_hat = 2.0 / dp  # d(barrier)/d(d_hat)
        ddcdp_d_hat = (dp - dc) / (dp * dp)  # d(dc/dp)/d(d_hat)

        # d2E/d_dhat2: second derivative, normal force stiffness (curvature of energy wrt normal gap)
        d2E_d_hat2 = -(barrier + dc / dp) - dc * (dbarrier_d_hat + ddcdp_d_hat)
        # Outer product with nor gives the 3x3 Hessian in the normal direction
        hessian[c] = d2E_d_hat2 / (rc * rc) * wp.outer(nor, nor)

        # friction hessian: slip term (mu_fn * f1_nu * vt) + chain rule term
        # through -mu * dt * h_vt * d2E_d_hat2 / rc^2 * nor

        # vt: tangential slip vector (velocity tangent to contact)
        vt = (offset - d * nor) / dt  # tangential velocity
        vt_norm = wp.length(vt)       # tangential slip speed

        # mu_fn: frictional force yield magnitude
        mu_fn = -mu * dE_d_hat / rc  # yield force (Coulomb friction)
        mu_fn_p = -mu * d2E_d_hat2 / rc  # d(mu_fn) / d(d_hat)

        # f1_over_vt_norm: regularized interpolation between sticking (quadratic penalty) and kinetic friction (linear), per D.E. Terzopoulos's classic formulation
        f1_over_vt_norm = wp.where(
            vt_norm < 1.0,  2.0 - vt_norm, 1.0 / vt_norm)
        f1_nu = f1_over_vt_norm + nu
        tangent_proj = wp.identity(3, dtype=wp.float32) - wp.outer(nor, nor)  # projection operator onto tangential plane

        slip_vtn_eps = 1.0e-4  # epsilon to handle tangential slip~0

        if vt_norm < slip_vtn_eps:
            # For near-zero tangential speed, limit of f1_nu * vt Hessian is f1_nu / dt * tangent_proj
            hessian[c] += mu_fn / dt * f1_nu * tangent_proj
        elif vt_norm < 1.0:
            # Stick regime (quadratic friction); extra out-of-plane curvature from vt
            hessian[c] += mu_fn / dt * (
                f1_nu * tangent_proj - wp.outer(vt, vt) / (vt_norm * dt)
            )
        else:
            # Slip regime (linear kinetic friction); out-of-plane curvature term
            f1_p = -1.0 / (vt_norm * vt_norm)
            hessian[c] += mu_fn * (
                f1_p / (vt_norm * dt) * wp.outer(vt, vt)
                + f1_nu / dt * tangent_proj
            )
        # mu_fn_p * f1_nu / rc * vt ⊗ nor: cross term in Hessian for friction-yield gradient vs normal direction
        hessian[c] += mu_fn_p * f1_nu / rc * wp.outer(vt, nor)

        # h_vt: tangential friction energy (quadratic for stick, linear past slip threshold)
        h_vt = (
            0.5 * nu * vt_norm * vt_norm
            + wp.where(
                vt_norm < 1.0,
                vt_norm * vt_norm * (1.0 - vt_norm / 3.0),
                vt_norm - 1.0 / 3.0,
            )
        )
        # h_vt_p: derivative of frictional energy wrt tangential slip speed
        h_vt_p = wp.where(
            vt_norm < 1.0,
            nu * vt_norm + 2.0 * vt_norm - vt_norm * vt_norm,
            nu * vt_norm + 1.0,
        )

        # Higher order derivatives of barrier/log-barrier and quadratic slip term
        d2barrier_d_hat = -2.0 / (dp * dp)
        dddcdp_d_hat = -2.0 * ddcdp_d_hat / dp
        df_d_hat = dbarrier_d_hat - dc / (dp * dp)
        dg_d_hat = d2barrier_d_hat + dddcdp_d_hat
        # d3E/d_dhat3: third derivative, for chain rule in frictional-tangential coupling
        d3E_d_hat3 = -df_d_hat - dg_d_hat * dc - (dbarrier_d_hat + ddcdp_d_hat)

        # dvtn_doffset: directional derivative of v_t norm w.r.t. offset (tangential motion)
        dvtn_doffset = wp.where(
            vt_norm > slip_vtn_eps,
            vt / (vt_norm * dt),
            wp.vec3(0.0),
        )

        # Cross-term coupling: normal and tangential directions via friction energy's chain rule
        chain_coeff = -mu * dt / (rc * rc)
        hessian[c] += chain_coeff * (
            d2E_d_hat2 * h_vt_p * wp.outer(nor, dvtn_doffset)
            + h_vt * d3E_d_hat3 / rc * wp.outer(nor, nor)
        )
        ###

    else:
        # Outside active barrier region: Hessian is zero
        hessian[c] = wp.mat33(0.0)

    hessian[c] = coeff * hessian[c]


@wp.kernel
def _get_collision_bounds_wp_kernel(
    radius: float,
    barrier_distance_ratio: float,
    dx_cur: wp.array(dtype=wp.vec3),
    dx_start_of_timestep: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
    delta_dx: wp.array(dtype=wp.vec3),
    jacobian_a_offsets: wp.array(dtype=int),
    jacobian_a_columns: wp.array(dtype=int),
    jacobian_b_offsets: wp.array(dtype=int),
    jacobian_b_columns: wp.array(dtype=int),
    dof_t_max: wp.array(dtype=float),
):  # pragma: no cover
    c = wp.tid()

    # Distance delta
    nor = normals[c]

    idx_a = indices_a[c]
    idx_b = indices_b[c]

    delta_d_a = wp.dot(nor, delta_dx[idx_a])

    # If idx_b is -1, then there is no second colliding particle
    if idx_b == NULL_ELEMENT_INDEX:
        delta_d_b = 0.0
    else:
        delta_d_b = -wp.dot(nor, delta_dx[idx_b])

    # Current distance
    offset = _collision_offset_wp_func(
        c, dx_cur, dx_start_of_timestep, kinematic_gaps, indices_a, indices_b)
    rc = _collision_target_distance_wp_func(c, radius, indices_a, indices_b)
    rp = barrier_distance_ratio * rc
    gap_cur = rp - wp.dot(offset, nor)

    if gap_cur >= 0.0:
        # Missed due to too large timestep. Can't do anything now
        return

    MAX_PROGRESS = 0.75
    max_delta_d = 0.5 * MAX_PROGRESS * gap_cur

    # TODO: Change this to use the cp_to_dof mapping in the future. In case I don't have these J_a, J_b matrices
    #
    # Jacobian tells me which dofs affect which colliding particles
    # Using two jacobians Ja, Jb you can tell which DOFs affect the first colliding particle
    # and the second colliding particle
    # Using warp sparse matrices I can use the same kernel to compute the bounds
    if delta_d_a < 0.0:  # getting closer
        t_max = wp.clamp(max_delta_d / delta_d_a, 0.0, 1.0)
        if t_max < 1.0:
            dof_beg = jacobian_a_offsets[3*c]
            dof_end = jacobian_a_offsets[3*c + 1]
            for dof in range(dof_beg, dof_end):
                wp.atomic_min(dof_t_max, jacobian_a_columns[dof], t_max)

    if delta_d_b < 0.0:  # getting closer
        t_max = wp.clamp(max_delta_d / delta_d_b, 0.0, 1.0)
        if t_max < 1.0:
            dof_beg = jacobian_b_offsets[3*c]
            dof_end = jacobian_b_offsets[3*c + 1]
            for dof in range(dof_beg, dof_end):
                wp.atomic_min(dof_t_max, jacobian_b_columns[dof], t_max)


class Collision:
    def __init__(self,
                 dt,
                 collision_particle_radius=0.1,
                 detection_ratio=1.5,
                 impenetrable_barrier_ratio=0.5,
                 ignore_self_collision_ratio=100000.0,
                 collision_penalty_stiffness=100.0,
                 friction_regularization=0.1,
                 friction_fluid=0.1,
                 friction=0.5,
                 max_contacting_pairs=10000,
                 bounds=True):
        r"""
        Initialize the collision class. This class operates on the whole scene

        Args:
            dt (float): Time step.
            collision_particle_radius (float): Radius of the collision particle at which penalty begins to apply. Defaults to 0.1.
            detection_ratio (float): Collision detection radius described as a ratio relative to the collision_particle_radius. Should be larger than collision_particle_radius. Defaults to 1.5.
            impenetrable_barrier_ratio (float): Collision barrier radius described as a ratio relative to the collision_particle_radius. Should be smaller than collision_particle_radius. Defaults to 0.25.
            ignore_self_collision_ratio (float): Collision immune radius described as a ratio relative to the collision_particle_radius. Defaults to 100000.0.
            collision_penalty_stiffness (float): Penalty stiffness of the collision interactions. Defaults to 100.0.
            friction_regularization (float): Friction regularization. Keeps friction forces proportional to timestep. Defaults to 0.1.
            friction_fluid (float): Dampens and smoothens the friction forces. Defaults to 0.1.
            friction (float): Friction coefficient. Defaults to 0.5.
            max_contacting_pairs (int): Number of contact points. Defaults to 10000.
            bounds (bool): Bounds the dofs in the line search to prevent any interpenetration. Defaults to True.
        """

        # Collision constants
        self.num_contacts = 0
        self.bounds = bounds
        self.collision_radius = collision_particle_radius

        self.collision_detection_ratio = detection_ratio
        self.collision_barrier_ratio = impenetrable_barrier_ratio
        self.ignore_self_collision_ratio = ignore_self_collision_ratio
        self.collision_penalty_stiffness = collision_penalty_stiffness

        # Friction constants
        self.friction_reg = friction_regularization
        self.friction_fluid = friction_fluid
        self.friction = friction
        self.dt = dt

        # Buffers for collisions get updated per timestep
        self.collision_indices_a = wp.empty(max_contacting_pairs, dtype=int)
        self.collision_indices_b = wp.empty(max_contacting_pairs, dtype=int)
        self.collision_normals = wp.empty(max_contacting_pairs, dtype=wp.vec3)
        self.collision_kinematic_gaps = wp.empty(
            max_contacting_pairs, dtype=wp.vec3)

        # Jacobians used to map from cps of contact pairs back to dofs
        self.collision_J_a = None  # Size 3*num_cps x num_dofs
        self.collision_J_b = None  # Size 3*num_cps x num_dofs
        self.collision_J = None  # Size 3*num_cps x num_dofs

        # stores the pos at start of timestep

        self.cp_dx_at_nm_iteration_0 = None

        # Hashgrid for broadphase collision detection
        self.hashgrid = wp.HashGrid(128, 128, 128)

    def detect_collisions(self, cp_dx, cp_x0, cp_obj_ids, cp_is_static=None):
        r""" Detects collisions between contact points and stores the results in the collision buffers.

        Args:
            cp_dx (wp.array(dtype=wp.vec3)): Current contact point displacements.
            cp_x0 (wp.array(dtype=wp.vec3)): Rest contact point positions.
            cp_obj_ids (wp.array(dtype=int)): Map from contact point to object id.
            cp_is_static (wp.array(dtype=int), optional): Array indicating which contact points are static (1 for static, 0 for dynamic). Defaults to None.

        Note:
            The function sets the collision indices in the collision_indices_a and collision_indices_b buffers, 
            collision normals in the collision_normals buffer, 
            and kinematic gaps between the contact points in the collision_kinematic_gaps buffer.
            
            The number of contacts is stored in the num_contacts attribute.
            
        """
        # TODO: If we call this function multiple times per timestep, we need to store the
        # cp_dx_at_nm_iteration_0_torch at the start of each timestep, not here.
        self.cp_dx_at_nm_iteration_0 = wp.clone(cp_dx)

        # current position of contact points
        current_cp = wp.from_torch(wp.to_torch(
            cp_dx) + wp.to_torch(cp_x0), dtype=wp.vec3)

        # Get change in cp position since start of timestep = cp_dx - cp_dx_at_nm_iteration_0
        # pos_delta = wp.from_torch(wp.to_torch(
        #     cp_dx) - wp.to_torch(self.cp_dx_at_nm_iteration_0), dtype=wp.vec3)
        pos_delta = wp.zeros_like(current_cp)

        # Build hashgrid from current contact points
        self.hashgrid.build(current_cp, radius=2.0*self.collision_radius)

        # Kernel inputs
        max_contacts = self.collision_indices_a.shape[0]
        detection_radius = self.collision_radius * self.collision_detection_ratio
        collision_immune_radius = self.collision_radius * self.ignore_self_collision_ratio
        
        if cp_is_static is None:
            cp_is_static = wp.zeros_like(cp_obj_ids) # none are static

        # Kernel outputs
        count = wp.zeros(1, dtype=int)
        
        # Find collisions
        wp.launch(
            kernel=_detect_particle_collisions_wp_kernel,
            dim=current_cp.shape[0],
            inputs=[max_contacts,
                    self.hashgrid.id,
                    2.0*detection_radius,  # 2x (for both particles)
                    collision_immune_radius,
                    current_cp,
                    cp_x0,
                    pos_delta,
                    cp_obj_ids,
                    cp_is_static, # indices of static objects
                    count,
                    self.collision_normals,
                    self.collision_kinematic_gaps,
                    self.collision_indices_a,
                    self.collision_indices_b],
        )
        

        self.num_contacts = int(count.numpy()[0])

        if self.num_contacts > max_contacts:
            logging.warning('contact buffer size exceed, some have been ignored')
            self.num_contacts = max_contacts

        # self.build_jacobian(cp_w, cp_x0, cp_obj_ids)

        # If there are any collision contacts detected
        if self.num_contacts > 0:
            # Get the indices of colliding points, truncated to actual number of contacts
            ind_a = wp.to_torch(self.collision_indices_a)[
                :self.num_contacts]  # size (num_contacts,)
            ind_b = wp.to_torch(self.collision_indices_b)[
                :self.num_contacts]  # size (num_contacts,)
            
            # Map collision point indices to their object IDs
            obj_ids = wp.to_torch(cp_obj_ids)  # size (num_cps,)
            obj_ids_a = obj_ids[ind_a]  # size (num_contacts,)
            obj_ids_b = obj_ids[ind_b]  # size (num_contacts,)

            # Create pairs of colliding object IDs
            # size (num_contacts, 2)
            object_pairs = torch.stack((obj_ids_a, obj_ids_b), dim=1)
            # size (num_unique_contacts, 2)
            unique_pairs = torch.unique(object_pairs, dim=0).cpu()

            # Flip-flop, reverse and self interaction pairs for the hessian matrix
            # (A,B), (B,A), (A,A) and (B,B)
            object_pairs = torch.vstack(
                (
                    unique_pairs[:, [0, 1]],  # Original pairs
                    unique_pairs[:, [1, 0]],  # Reversed pairs
                    unique_pairs[:, [0, 0]],  # Self pairs for first objects
                    unique_pairs[:, [1, 1]],  # Self pairs for second objects
                )
            )
            # Get unique interaction pairs
            self.object_pairs = torch.unique(object_pairs, dim=0).numpy() # needed for indexing in the hessian matrix
        else:
            # If no collisions, empty list
            self.object_pairs = []

        return

    def calculate_jacobian(self, cp_w, cp_x0, cp_is_static=None, qr_tfm=None):
        r""" Builds the jacobians of the collision points w.r.t the dofs. For contact pairs :math:`x_a \in \mathbb{R}^3, x_b \in \mathbb{R}^3`, the jacobians are:

        .. math::
            J_a = \frac{\partial x_a}{\partial z} \in \mathbb{R}^{3 \times n}
            J_b = \frac{\partial x_b}{\partial z} \in \mathbb{R}^{3 \times n}
            J = J_a - J_b \in \mathbb{R}^{3 \times n}

        The difference, :math:`J = J_a - J_b`, is the jacobian of the collision gaps.

        Args:
            cp_w (wp.array2d(dtype=wp.float32)): Contact point skinning weights of size :math:`(\text{num_pts}, \text{num_handles})`
            cp_x0 (wp.array(dtype=wp.vec3)): Rest contact point positions of size :math:`(\text{num_pts}, 3)`
            cp_is_static (wp.array(dtype=int), optional): Array indicating which contact points are static (1 for static, 0 for dynamic). Defaults to None.
            qr_tfm (torch.Tensor, optional): Block-diagonal handle-DOF rotation that maps the raw (pre-QR) basis to the post-QR basis used for elastic/inertia terms. When provided, ``collision_J`` and ``collision_J_dense`` are rotated into the post-QR basis for gradient/Hessian assembly, while ``collision_J_a``/``collision_J_b`` are kept in the raw basis so the bounds kernel can still read meaningful per-DOF sparsity. Defaults to None.

        Note:
            The jacobian set by this function is a sparse matrix of size :math:`(3 \times \text{num_contacts}, 12 \times \text{num_handles})`.
        """

        # indices of the colliding point pairs
        num_handles = cp_w.shape[1]
        if self.num_contacts == 0:
            num_rows = 0
            num_cols = 12*num_handles

            self.collision_J_a = wps.bsr_zeros(
                num_rows, num_cols, wp.float32)
            self.collision_J_b = wps.bsr_zeros(
                num_rows, num_cols, wp.float32)
        else:
            if cp_is_static is None:
                cp_is_static = wp.zeros(cp_x0.shape[0], dtype=wp.int32, device=cp_x0.device)

            ind_a = wp.clone(self.collision_indices_a[:self.num_contacts])
            ind_b = wp.clone(self.collision_indices_b[:self.num_contacts])

            J_a = sparse_collision_jacobian_matrix(cp_w, cp_x0, indices=ind_a, cp_is_static=cp_is_static)
            J_b = sparse_collision_jacobian_matrix(cp_w, cp_x0, indices=ind_b, cp_is_static=cp_is_static)

            self.collision_J_a = J_a
            self.collision_J_b = J_b
            self.collision_J_a.nnz_sync()
            self.collision_J_b.nnz_sync()

        self.collision_J = self.collision_J_a - self.collision_J_b #wps.bsr_copy(J, block_shape=(3, 12))
        self.collision_J.nnz_sync()

        if self.num_contacts > 0:
            self.collision_J_dense = _bsr_to_torch(self.collision_J).to_dense()
        else:
            self.collision_J_dense = torch.zeros(self.collision_J.shape, device=wp.device_to_torch(self.collision_J.device), dtype=wp.dtype_to_torch(self.collision_J.dtype))

        # QR mode: rotate the consumer-facing collision_J / collision_J_dense into the
        # post-QR basis. collision_J_a / collision_J_b stay raw so get_bounds can read
        # the original LBS sparsity (a row-subset of pre-QR B with per-handle column
        # blocks). The line search wraps _apply_bounds with the inverse rotation so the
        # clamp still happens in the basis where the bounds were computed.
        if qr_tfm is not None and self.num_contacts > 0:
            self.collision_J_dense = self.collision_J_dense @ qr_tfm
            # Match the (1, 4) block shape used elsewhere (e.g. simulation.py:140
            # for sim_B), so bsr_mv downstream sees the same layout it did pre-QR.
            self.collision_J = wps.bsr_copy(
                _warp_csr_from_torch_dense(self.collision_J_dense), block_shape=(1, 4))
            self.collision_J.nnz_sync()

        return

    def get_bounds(self, cp_delta_dx, cp_dx, cp_x0):
        r""" Compute the bounds of the update for each dof. This is used to guarantee intersection-free contact. See :func:`kaolin.physics.optimization.apply_bounds` for more details.

        Args:
            cp_delta_dx (wp.array(dtype=wp.vec3)): :math:`(B*dz).reshape(-1, 3)` where :math:`dz` is the newton update of size :math:`(\text{num_pts}, 3)`
            cp_dx (wp.array(dtype=wp.vec3)): :math:`(B*z).reshape(-1, 3)` where :math:`z` is the current dofs of size :math:`(\text{num_pts}, 3)`
            cp_x0 (wp.array(dtype=wp.vec3)): Rest contact point positions of size :math:`(\text{num_pts}, 3)`
            
        Returns:
            wp.array(dtype=float): Bounds of the update for each dof of size :math:`(\text{num_dofs},)`
        """
        if self.num_contacts == 0 and not self.bounds:
            return None

        # Inputs: Position increments of the contact points

        # Output: vector of size num_column_blocks in J_a. If J_a is csr, then num_blocks=J.shape[1]
        blockwise_bounds = wp.ones(
            (self.collision_J_a.ncol), dtype=float, device=self.collision_J_a.device)

        wp.launch(
            _get_collision_bounds_wp_kernel,
            dim=self.num_contacts,
            inputs=[
                self.collision_radius,
                self.collision_barrier_ratio,
                cp_dx,
                self.cp_dx_at_nm_iteration_0,
                self.collision_kinematic_gaps,  # kinematic gaps
                self.collision_normals,      # contact normals
                self.collision_indices_a,    # indices of colliding point pairs
                self.collision_indices_b,    # indices of colliding point pairs
                # step delta (dz) applied to cps: B*dz
                cp_delta_dx,
                self.collision_J_a.offsets,  # offsets of the jacobian blocks
                self.collision_J_a.columns,  # columns of the jacobian blocks
                self.collision_J_b.offsets,  # offsets of the jacobian blocks
                self.collision_J_b.columns,  # columns of the jacobian blocks
                blockwise_bounds,               # Output: bounds for each handle
            ],
        )

        # print(
        #     "blockwise_bounds: left is rbf, right is mlp. If rbf > mlp is true, thats good.")
        # left = wp.to_torch(blockwise_bounds[0:blockwise_bounds.shape[0]//2])
        # right = wp.to_torch(blockwise_bounds[blockwise_bounds.shape[0]//2:])
        # print(left)
        # print(right)

        # we have one bound per block column of J.
        # expand to one bound per scalar column, as that is what apply_bounds expect
        dof_bounds = wp.from_torch(
            wp.to_torch(blockwise_bounds).unsqueeze(1).repeat(1, self.collision_J_a.block_shape[1]).flatten())

        return dof_bounds

    def energy(self, dx, x0, coeff, energy=None):
        r"""
        Compute the collision energy.

        Args:
            dx (wp.array(dtype=wp.vec3)): Current CP displacements with the current dofs of size :math:`(\text{num_pts}, 3)`
            x0 (wp.array(dtype=wp.vec3)): Rest contact point positions of size :math:`(\text{num_pts}, 3)`
            coeff (float): Coefficient for the collision energy.
            energy (wp.array(dtype=float)): Output energy. Used for cuda-graph capture of size :math:`1`

        Returns:
            wp.array(dtype=float): Optional output energy of size :math:`1`
        """
        
        if energy is None:
            energy = wp.zeros(1, dtype=float)

        wp.launch(
            kernel=_collision_energy_wp_kernel,
            dim=self.num_contacts,
            inputs=[coeff,
                    self.collision_radius,
                    self.collision_barrier_ratio,
                    self.friction,
                    self.dt*self.friction_reg,
                    self.friction_fluid*self.friction_reg,
                    dx,
                    self.cp_dx_at_nm_iteration_0,
                    self.collision_kinematic_gaps,
                    self.collision_normals,
                    self.collision_indices_a,
                    self.collision_indices_b],
            outputs=[energy],
            adjoint=False
        )
        return energy
        # print("collision energy: ", self.num_contacts, energy.numpy())

    def gradient(self, dx, x0, coeff):
        r"""
        Compute the gradient of the collision energy.

        Args:
            dx (wp.array(dtype=wp.vec3)): Current CP displacements with the current dofs of size :math:`(\text{num_pts}, 3)`
            x0 (wp.array(dtype=wp.vec3)): Rest contact point positions of size :math:`(\text{num_pts}, 3)`
            coeff (float): Coefficient for the collision energy.

        Returns:
            wp.array(dtype=wp.vec3): Gradient of the collision energy of size :math:`(\text{num_contacts}, 3)`
        """
        gradient = wp.zeros(
            self.num_contacts, dtype=wp.vec3, device=dx.device)

        wp.launch(
            kernel=_collision_gradient_wp_kernel,
            dim=self.num_contacts,
            inputs=[coeff,
                    self.collision_radius,
                    self.collision_barrier_ratio,
                    self.friction,
                    self.dt*self.friction_reg,
                    self.friction_fluid*self.friction_reg,
                    dx,
                    self.cp_dx_at_nm_iteration_0,
                    self.collision_kinematic_gaps,
                    self.collision_normals,
                    self.collision_indices_a,
                    self.collision_indices_b],
            outputs=[gradient],
            adjoint=False
        )

        return gradient

    def hessian(self, dx, x0, coeff):
        r"""
        Compute the hessian of the collision energy.

        Args:
            dx (wp.array(dtype=wp.vec3)): Current CP displacements with the current dofs of size :math:`(\text{num_pts}, 3)`
            x0 (wp.array(dtype=wp.vec3)): Rest contact point positions of size :math:`(\text{num_pts}, 3)`
            coeff (float): Coefficient for the collision energy.

        Returns:
            wp.array(dtype=wp.mat33): Hessian of the collision energy of size :math:`(\text{num_contacts}, 3, 3)`
        """

        hessian_blocks = wp.zeros(
            self.num_contacts, dtype=wp.mat33, device=dx.device)

        wp.launch(
            kernel=_collision_hessian_diag_blocks_wp_kernel,
            dim=self.num_contacts,
            inputs=[coeff,
                    self.collision_radius,
                    self.collision_barrier_ratio,
                    self.friction,
                    self.dt*self.friction_reg,
                    self.friction_fluid*self.friction_reg,
                    dx,
                    self.cp_dx_at_nm_iteration_0,
                    self.collision_kinematic_gaps,
                    self.collision_normals,
                    self.collision_indices_a,
                    self.collision_indices_b],
            outputs=[hessian_blocks],
            adjoint=False
        )
        return hessian_blocks
