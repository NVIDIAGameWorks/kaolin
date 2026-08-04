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
from kaolin.physics.utils.torch_utilities import hess_reduction

__all__ = ['Collision', 'ChunkedCollisionHessian']

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


@wp.kernel
def _collision_jacobian_chunk_wp_kernel(
    b_dense: wp.array2d(dtype=wp.float32),    # (3*num_cps, num_dofs) subspace basis
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
    num_contacts: wp.array(dtype=int),
    chunk_start: wp.array(dtype=int),         # device-resident, so the loop never syncs
    j_chunk: wp.array2d(dtype=wp.float32),    # (3*chunk, num_dofs) output
):  # pragma: no cover
    r"""Materializes one chunk of the collision Jacobian as a row gather from ``B``.

    ``J[3c+k, :] = B[3*idx_a[c]+k, :] - B[3*idx_b[c]+k, :]`` -- verified exactly equal to
    the sparse-assembled ``collision_J_dense``. Because it is only a gather, a chunk can
    be built directly at any offset, so the full ``(3*max_contacting_pairs, num_dofs)``
    Jacobian never has to exist: memory is ``chunk * 3 * num_dofs`` regardless of
    capacity. At the capacities kaolin's examples use that is 14 MiB rather than 343 MiB.

    ``chunk_start`` is an array, not an int, because the enclosing ``wp.capture_while``
    advances it on device; a Python int would bake the offset into the graph.
    """
    c, j = wp.tid()
    g = chunk_start[0] + c

    # Past the live contact count: zero the rows. The reduction runs over the whole
    # chunk, so padding must contribute exactly nothing to J^T H J.
    if g >= num_contacts[0]:
        for k in range(3):
            j_chunk[3 * c + k, j] = 0.0
        return

    idx_a = indices_a[g]
    idx_b = indices_b[g]
    for k in range(3):
        v = float(0.0)
        # A static side has no DOFs to differentiate against, so it contributes no rows.
        # Guarded for the same reason as _collision_offset_wp_func: NULL_ELEMENT_INDEX is
        # a marker, and gathering at -1 would wrap to the last row of B.
        if idx_a != NULL_ELEMENT_INDEX:
            v += b_dense[3 * idx_a + k, j]
        if idx_b != NULL_ELEMENT_INDEX:
            v -= b_dense[3 * idx_b + k, j]
        j_chunk[3 * c + k, j] = v


@wp.kernel
def _collision_hessian_chunk_wp_kernel(
    h_full: wp.array(dtype=wp.mat33),
    num_contacts: wp.array(dtype=int),
    chunk_start: wp.array(dtype=int),
    h_chunk: wp.array(dtype=wp.mat33),
):  # pragma: no cover
    r"""Gathers the per-contact 3x3 Hessian blocks for one chunk, zeroing the padding."""
    c = wp.tid()
    g = chunk_start[0] + c
    if g >= num_contacts[0]:
        h_chunk[c] = wp.mat33(0.0)
    else:
        h_chunk[c] = h_full[g]


@wp.kernel
def _collision_gradient_chunk_wp_kernel(
    g_full: wp.array(dtype=wp.vec3),
    num_contacts: wp.array(dtype=int),
    chunk_start: wp.array(dtype=int),
    g_chunk: wp.array(dtype=wp.vec3),
):  # pragma: no cover
    r"""Gathers the per-contact :math:`dE/dx` for one chunk, zeroing the padding."""
    c = wp.tid()
    g = chunk_start[0] + c
    if g >= num_contacts[0]:
        g_chunk[c] = wp.vec3(0.0)
    else:
        g_chunk[c] = g_full[g]


@wp.kernel
def _clamp_contact_count_wp_kernel(
    count: wp.array(dtype=int),
    capacity: int,
):  # pragma: no cover
    r"""Clamps the detected contact count to the buffer capacity, on device.

    The detection kernel increments ``count`` *before* testing the capacity, so on
    overflow it holds the raw detected total while only ``capacity`` slots were written.
    Every per-contact kernel guards on this array, so an unclamped count would let threads
    read slots that were never filled.

    Done here rather than from the host because ``count <= capacity`` has to hold without
    anyone reading it back: it is what makes a partial final chunk safe in
    :class:`ChunkedCollisionHessian` (a thread passing ``g >= count`` therefore also
    satisfies ``g < capacity``), and it is the last thing that forced a D2H per step.
    """
    count[0] = wp.min(count[0], capacity)


@wp.kernel
def _advance_chunk_start_wp_kernel(
    chunk_start: wp.array(dtype=int),
    chunk_size: int,
):  # pragma: no cover
    r"""Advances the chunk cursor on device, so the loop never syncs to advance it."""
    chunk_start[0] = chunk_start[0] + chunk_size


@wp.kernel
def _chunk_loop_cond_wp_kernel(
    chunk_start: wp.array(dtype=int),
    num_contacts: wp.array(dtype=int),
    cond: wp.array(dtype=int),
):  # pragma: no cover
    r"""``wp.capture_while`` predicate: are there still live contacts left to reduce?

    Written as a device array so the loop trip count follows the *actual* contact count
    at replay rather than the capacity baked in at capture time.
    """
    cond[0] = wp.where(chunk_start[0] < num_contacts[0], 1, 0)


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
                motion and kinematic gaps. Either side may be static
                (NULL_ELEMENT_INDEX), in which case that side contributes no motion;
                if both are, the offset is just the kinematic gap.
    """
    idx_a = indices_a[c]
    idx_b = indices_b[c]

    # A static point contributes no motion, so its term is simply omitted -- the same
    # semantics the idx_b branch below has always had. NULL_ELEMENT_INDEX is a marker,
    # not an index: Warp's index() does `if (i < 0) i += shape[0]`, so gathering at -1
    # silently returns the LAST contact point in the scene and folds an unrelated
    # particle's displacement into the offset.
    #
    # Unreachable today (simulation.py calls detect_collisions with cp_is_static=None,
    # so the sentinel is never written), but detect_collisions is public and takes
    # cp_is_static; the moment a caller supplies it, detection's `idx_a < idx_b`
    # ordering puts the sentinel in indices_a for *every* contact against a
    # low-indexed static object, not occasionally.
    offset = kinematic_gaps[c]
    if idx_a != NULL_ELEMENT_INDEX:
        offset += dx_cur[idx_a] - dx_start_of_timestep[idx_a]
    if idx_b != NULL_ELEMENT_INDEX:
        offset -= dx_cur[idx_b] - dx_start_of_timestep[idx_b]
    return offset


@wp.func
def _collision_target_distance_wp_func(
    c: int,
    radius: float,
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):  # pragma: no cover
    r"""Target separation for contact ``c``: one radius against static geometry, two
    between two dynamic particles.

    Both indices are tested, not just ``indices_b``. Detection enforces ``idx_a < idx_b``,
    so a kinematic object added early in the scene -- as ``simplicits_friction_slab`` adds
    its slab -- puts the sentinel in ``indices_a`` for *every* one of its contacts, not
    occasionally. Testing only ``indices_b`` would then return ``2*radius`` for all of
    them, and since ``rc`` divides through everything downstream (``d_hat = d/rc``,
    ``rp = barrier_ratio*rc``, gradient ``~ dE/rc``, Hessian ``~ d2E/rc^2``) the barrier
    would engage at twice the intended gap with the force halved and the Hessian quartered.
    Warp folds a ``-1`` subscript to the last element rather than raising, so it would fail
    silently.
    """
    # No short-circuit `or` inside wp.where, so combine the two tests arithmetically.
    static_a = wp.where(indices_a[c] == NULL_ELEMENT_INDEX, 1, 0)
    static_b = wp.where(indices_b[c] == NULL_ELEMENT_INDEX, 1, 0)
    return wp.where(static_a + static_b > 0, 1.0, 2.0) * radius


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
    num_contacts: wp.array(dtype=int),
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

    # Slot validity, launched over the fixed max_contacting_pairs capacity. This is a
    # no-op when the launch dim is exactly num_contacts (every tid passes), so the
    # non-capturable path is unaffected. Deliberately NOT `indices_b[c] < 0`: that
    # sentinel means "partner is static geometry", a distinct concept that
    # _collision_offset_wp_func and _collision_target_distance_wp_func each handle by
    # testing both indices, and conflating the two silently drops every contact against a
    # kinematic collider.
    if c >= num_contacts[0]:
        return

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
                        num_contacts: wp.array(dtype=int),
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

    # Slot validity, launched over the fixed max_contacting_pairs capacity. This is a
    # no-op when the launch dim is exactly num_contacts (every tid passes), so the
    # non-capturable path is unaffected. Deliberately NOT `indices_b[c] < 0`: that
    # sentinel means "partner is static geometry", a distinct concept that
    # _collision_offset_wp_func and _collision_target_distance_wp_func each handle by
    # testing both indices, and conflating the two silently drops every contact against a
    # kinematic collider.
    if c >= num_contacts[0]:
        return

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
                                   num_contacts: wp.array(dtype=int),
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

    # Slot validity, launched over the fixed max_contacting_pairs capacity. This is a
    # no-op when the launch dim is exactly num_contacts (every tid passes), so the
    # non-capturable path is unaffected. Deliberately NOT `indices_b[c] < 0`: that
    # sentinel means "partner is static geometry", a distinct concept that
    # _collision_offset_wp_func and _collision_target_distance_wp_func each handle by
    # testing both indices, and conflating the two silently drops every contact against a
    # kinematic collider.
    if c >= num_contacts[0]:
        return

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
    num_contacts: wp.array(dtype=int),
    dof_t_max: wp.array(dtype=float),
):  # pragma: no cover
    c = wp.tid()

    # Slot validity, launched over the fixed max_contacting_pairs capacity. This is a
    # no-op when the launch dim is exactly num_contacts (every tid passes), so the
    # non-capturable path is unaffected. Deliberately NOT `indices_b[c] < 0`: that
    # sentinel means "partner is static geometry", a distinct concept that
    # _collision_offset_wp_func and _collision_target_distance_wp_func each handle by
    # testing both indices, and conflating the two silently drops every contact against a
    # kinematic collider.
    if c >= num_contacts[0]:
        return

    # Distance delta
    nor = normals[c]

    idx_a = indices_a[c]
    idx_b = indices_b[c]

    # If idx_a is -1 the first point is static and does not move. Mirrors the idx_b
    # branch below; without it, delta_dx[-1] wraps to the last contact point.
    if idx_a == NULL_ELEMENT_INDEX:
        delta_d_a = 0.0
    else:
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


@wp.kernel
def _get_collision_bounds_dense_wp_kernel(
    radius: float,
    barrier_distance_ratio: float,
    dx_cur: wp.array(dtype=wp.vec3),
    dx_start_of_timestep: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
    delta_dx: wp.array(dtype=wp.vec3),
    b_dense: wp.array2d(dtype=wp.float32),
    block_width: int,
    num_contacts: wp.array(dtype=int),
    dof_t_max: wp.array(dtype=float),
):  # pragma: no cover
    r"""Capturable form of :func:`_get_collision_bounds_wp_kernel`.

    Identical math; the only difference is where the "which DOFs does this contact point
    move?" question is answered. The original walks the BSR structure of ``collision_J_a``
    / ``collision_J_b``, which the capturable path no longer builds. Since those Jacobians
    are row gathers of the dense basis, row ``3c`` of :math:`J_a` *is* row ``3*idx_a`` of
    ``B``, so the same block sparsity is read straight from ``b_dense``.

    Launched 2D over (contact, DOF block). Each thread redoes the (cheap) gap arithmetic
    so that the sparsity test parallelizes over DOFs instead of looping inside one thread.
    """
    c, blk = wp.tid()

    if c >= num_contacts[0]:
        return

    nor = normals[c]
    idx_a = indices_a[c]
    idx_b = indices_b[c]

    if idx_a == NULL_ELEMENT_INDEX:
        delta_d_a = 0.0
    else:
        delta_d_a = wp.dot(nor, delta_dx[idx_a])

    if idx_b == NULL_ELEMENT_INDEX:
        delta_d_b = 0.0
    else:
        delta_d_b = -wp.dot(nor, delta_dx[idx_b])

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

    col0 = block_width * blk

    if delta_d_a < 0.0 and idx_a != NULL_ELEMENT_INDEX:  # getting closer
        t_max = wp.clamp(max_delta_d / delta_d_a, 0.0, 1.0)
        if t_max < 1.0:
            # int(0) rather than False: Warp requires an explicit dynamic-variable
            # declaration for anything mutated inside a dynamic loop.
            touched = int(0)
            for m in range(block_width):
                if b_dense[3 * idx_a, col0 + m] != 0.0:
                    touched = 1
            if touched == 1:
                for m in range(block_width):
                    wp.atomic_min(dof_t_max, col0 + m, t_max)

    if delta_d_b < 0.0 and idx_b != NULL_ELEMENT_INDEX:  # getting closer
        t_max = wp.clamp(max_delta_d / delta_d_b, 0.0, 1.0)
        if t_max < 1.0:
            touched = int(0)
            for m in range(block_width):
                if b_dense[3 * idx_b, col0 + m] != 0.0:
                    touched = 1
            if touched == 1:
                for m in range(block_width):
                    wp.atomic_min(dof_t_max, col0 + m, t_max)


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
                 bounds=True,
                 capturable=False):
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
            capturable (bool): Launch every per-contact kernel over the fixed
                ``max_contacting_pairs`` capacity, relying on the in-kernel device-count
                guard, so no launch dimension depends on a host value. Required for cuda
                graph capture. Note this changes the length of the arrays ``gradient`` and
                ``hessian`` return when given preallocated outputs. Defaults to False.
        """

        # Collision constants
        # num_contacts is a memoized property backed by self.count -- see below.
        self._num_contacts_cache = None
        # Initialized here because get_bounds and _assemble_hessians read it before the
        # first detect_collisions call on a freshly built scene.
        self.object_pairs = []
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

        if max_contacting_pairs <= 0:
            raise ValueError(
                f"max_contacting_pairs must be positive, got {max_contacting_pairs}.")
        self.max_contacting_pairs = max_contacting_pairs

        # Buffers for collisions get updated per timestep.
        # wp.zeros rather than wp.empty: kernels launched over the full capacity read
        # the tail as well, and uninitialized indices would be out-of-bounds gathers in
        # _collision_offset_wp_func. The device-side count guard is what makes the tail
        # inert, but the buffers must still be deterministic.
        self.collision_indices_a = wp.zeros(max_contacting_pairs, dtype=int)
        self.collision_indices_b = wp.zeros(max_contacting_pairs, dtype=int)
        self.collision_normals = wp.zeros(max_contacting_pairs, dtype=wp.vec3)
        self.collision_kinematic_gaps = wp.zeros(
            max_contacting_pairs, dtype=wp.vec3)

        # Contact count, kept on device. This is the single source of truth: the
        # num_contacts property below reads it on demand rather than at detection time,
        # so a captured step never pays a D2H for a number it does not use.
        self.count = wp.zeros(1, dtype=int)

        # When True, every per-contact kernel launches over the fixed
        # max_contacting_pairs capacity and relies on the in-kernel device-count guard,
        # so no launch dimension depends on a host-side value. Required for graph
        # capture; off by default so the existing path is unchanged.
        self.capturable = capturable

        # Jacobians used to map from cps of contact pairs back to dofs
        self.collision_J_a = None  # Size 3*num_cps x num_dofs
        self.collision_J_b = None  # Size 3*num_cps x num_dofs
        self.collision_J = None  # Size 3*num_cps x num_dofs

        # stores the pos at start of timestep

        self.cp_dx_at_nm_iteration_0 = None

        # Hashgrid for broadphase collision detection
        self.hashgrid = wp.HashGrid(128, 128, 128)

    def set_start_of_timestep_dx(self, cp_dx):
        r"""Records the contact-point displacements at the start of the timestep.

        Writes **in place** into a persistent buffer rather than rebinding the attribute.
        All four per-contact kernels read this array, so a captured graph records its
        pointer; rebinding it (``wp.clone``, or an assignment from outside) would leave
        the graph replaying against memory the scene no longer owns. Same reasoning as
        the persistent ``self.count``.

        Args:
            cp_dx (wp.array(dtype=wp.vec3)): Displacements to record.
        """
        if (self.cp_dx_at_nm_iteration_0 is None
                or self.cp_dx_at_nm_iteration_0.shape[0] != cp_dx.shape[0]):
            # First call, or the contact-point count changed (scene rebuilt). Allocating
            # here is fine: it happens outside any capture, and the pointer is then
            # stable for every subsequent timestep.
            self.cp_dx_at_nm_iteration_0 = wp.zeros_like(cp_dx)
        wp.copy(dest=self.cp_dx_at_nm_iteration_0, src=cp_dx)

    @property
    def num_contacts(self):
        r"""Number of live contacts. Read back from the device once per detection.

        Lazy *and* memoized, and it has to be both.

        Lazy, so the capturable path never pays for it: the count is already on device
        and every per-contact kernel guards against it there, so a captured step needs no
        host-side copy at all. Reading it eagerly in ``detect_collisions`` cost one
        blocking D2H per step for a value that path discards.

        Memoized, so the host path does not pay for it repeatedly. It reads this
        constantly -- ``_assemble_energies`` alone touches it once per energy evaluation,
        which with the default ``max_newton_steps=5`` / ``max_ls_steps=10`` is up to ~55
        times a step, and ``_contact_launch_dim`` reads it again inside each of those.
        Measured at 111 reads per step. Uncached, every one of those is its own device
        sync, which trades one stall per step for a hundred.

        The cache is dropped at the start of each detection and nowhere else, so it is
        valid for exactly as long as the contact set is. Code that writes ``self.count``
        directly (tests do) must call :func:`invalidate_contact_count` afterwards.

        The clamp mirrors :func:`_clamp_contact_count_wp_kernel`, which has already
        applied it on device; it is repeated here so the overflow warning has somewhere
        to live now that detection does not look at the count.

        Returns:
            int: Contact count, capped at ``max_contacting_pairs``.
        """
        if self._num_contacts_cache is None:
            n = int(self.count.numpy()[0])
            if n > self.max_contacting_pairs:
                logging.warning('contact buffer size exceed, some have been ignored')
                n = self.max_contacting_pairs
            self._num_contacts_cache = n
        return self._num_contacts_cache

    def invalidate_contact_count(self):
        r"""Drops the memoized :attr:`num_contacts`, forcing a re-read on next access.

        Called automatically by :func:`detect_collisions`. Only needed externally by code
        that writes ``self.count`` behind the scene's back.
        """
        self._num_contacts_cache = None

    def _contact_launch_dim(self):
        r"""Launch dimension for per-contact kernels.

        ``max_contacting_pairs`` under ``capturable`` (fixed at capture time, with the
        in-kernel ``c >= num_contacts[0]`` guard making the unused tail inert), otherwise
        the live host-side count, which is what the non-capturable path has always used.
        """
        return self.max_contacting_pairs if self.capturable else self.num_contacts

    def build_jacobian_chunk(self, b_dense, chunk_start, j_chunk):
        r"""Materializes ``chunk`` consecutive contacts of the collision Jacobian.

        Equivalent to ``self.collision_J_dense[3*s : 3*(s+chunk), :]`` for
        ``s = chunk_start[0]``, but built as a direct row gather from the dense subspace
        basis rather than assembled sparsely, so no ``bsr_from_triplets``, no
        ``nnz_sync()``, and no host readback. Rows past the live contact count are zeroed.

        This is what lets the full ``(3*max_contacting_pairs, num_dofs)`` Jacobian stay
        unallocated: peak memory is set by ``chunk``, not by capacity.

        Args:
            b_dense (wp.array2d(dtype=wp.float32)): Dense subspace basis of size
                :math:`(3 \times \text{num_pts}, \text{num_dofs})`.
            chunk_start (wp.array(dtype=int)): Single-element device array holding the
                index of the first contact in this chunk. Device-resident so the
                enclosing ``wp.capture_while`` can advance it without a host sync.
            j_chunk (wp.array2d(dtype=wp.float32)): Output of size
                :math:`(3 \times \text{chunk}, \text{num_dofs})`. Fully overwritten.
        """
        if j_chunk.shape[0] % 3 != 0:
            raise ValueError(
                f"j_chunk must have a multiple of 3 rows, got {j_chunk.shape[0]}.")
        if j_chunk.shape[1] != b_dense.shape[1]:
            raise ValueError(
                f"j_chunk has {j_chunk.shape[1]} columns but b_dense has "
                f"{b_dense.shape[1]}; both must be num_dofs.")
        wp.launch(
            kernel=_collision_jacobian_chunk_wp_kernel,
            dim=(j_chunk.shape[0] // 3, j_chunk.shape[1]),
            inputs=[b_dense, self.collision_indices_a, self.collision_indices_b,
                    self.count, chunk_start],
            outputs=[j_chunk],
            device=b_dense.device)

    def gather_gradient_chunk(self, g_full, chunk_start, g_chunk):
        r"""Gathers the per-contact :math:`dE/dx` vectors for one chunk.

        Args:
            g_full (wp.array(dtype=wp.vec3)): Per-contact gradients over the full
                capacity, as written by :func:`gradient`.
            chunk_start (wp.array(dtype=int)): Single-element device array holding the
                index of the first contact in this chunk.
            g_chunk (wp.array(dtype=wp.vec3)): Output of size ``chunk``.
        """
        wp.launch(
            kernel=_collision_gradient_chunk_wp_kernel,
            dim=g_chunk.shape[0],
            inputs=[g_full, self.count, chunk_start],
            outputs=[g_chunk],
            device=g_full.device)

    def gather_hessian_chunk(self, h_full, chunk_start, h_chunk):
        r"""Gathers the per-contact :math:`3 \times 3` Hessian blocks for one chunk.

        The companion to :func:`build_jacobian_chunk`: together they give the
        :math:`J_c^T H_c J_c` operands for one chunk. Blocks past the live contact count
        are zeroed, so padded slots contribute nothing to the reduction.

        Args:
            h_full (wp.array(dtype=wp.mat33)): Per-contact blocks over the full capacity,
                as written by :func:`hessian`.
            chunk_start (wp.array(dtype=int)): Single-element device array holding the
                index of the first contact in this chunk.
            h_chunk (wp.array(dtype=wp.mat33)): Output of size ``chunk``.
        """
        wp.launch(
            kernel=_collision_hessian_chunk_wp_kernel,
            dim=h_chunk.shape[0],
            inputs=[h_full, self.count, chunk_start],
            outputs=[h_chunk],
            device=h_full.device)

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
        self.set_start_of_timestep_dx(cp_dx)

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

        # Kernel outputs. Reuse the persistent device buffer rather than allocating a
        # fresh one per detection: a captured graph records the pointer, so the count
        # must live at a stable address.
        self.count.zero_()
        # The memoized host mirror is stale from here on. Dropped before the launch, so
        # any read after this point re-syncs and sees the new contact set.
        self._num_contacts_cache = None
        count = self.count

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
        

        # Enforce count <= capacity on device. Nothing reads the count back here; see
        # _clamp_contact_count_wp_kernel for why that invariant has to hold anyway.
        wp.launch(_clamp_contact_count_wp_kernel, dim=1,
                  inputs=[self.count, max_contacts], device=self.count.device)

        if self.capturable:
            # object_pairs below costs two blocking D2H (torch.unique with dim= has a
            # data-dependent output shape, so it sizes its output from the host, then
            # .cpu()/.numpy() again). Its only consumer is the host Newton path's
            # _assemble_hessians, which builds a sparse block matrix from the pair list.
            # The capturable assembly reduces J^T H J full-width and never reads it, so
            # for a captured scene this is pure dead work plus the last host syncs in the
            # step. Returning here is what makes the step actually sync-free.
            return

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
            dim=self._contact_launch_dim(),
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
                self.count,                  # device-side contact count (slot guard)
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

    def get_bounds_capturable(self, cp_delta_dx, cp_dx, b_dense, dof_bounds,
                              block_width=4):
        r"""Per-DOF step bounds, without touching the sparse Jacobian.

        Capturable counterpart of :func:`get_bounds`: fixed launch dimension, preallocated
        output, no allocation and no host read. See
        :func:`_get_collision_bounds_dense_wp_kernel` for why ``b_dense`` can stand in for
        the BSR structure.

        Args:
            cp_delta_dx (wp.array(dtype=wp.vec3)): :math:`B \, dz`, the proposed step at
                the contact points.
            cp_dx (wp.array(dtype=wp.vec3)): :math:`B \, z`, current displacements.
            b_dense (wp.array2d(dtype=wp.float32)): Dense subspace basis.
            dof_bounds (wp.array(dtype=float)): Preallocated ``(num_dofs,)`` output. Reset
                to 1.0 here, so callers need not.
            block_width (int, optional): DOF block granularity at which bounds are
                applied, matching ``collision_J_a``'s block shape. Defaults to 4.

        Returns:
            wp.array(dtype=float): ``dof_bounds``.
        """
        num_dofs = b_dense.shape[1]
        if dof_bounds.shape[0] != num_dofs:
            raise ValueError(
                f"dof_bounds has {dof_bounds.shape[0]} entries but b_dense has "
                f"{num_dofs} columns.")
        if num_dofs % block_width != 0:
            raise ValueError(
                f"num_dofs ({num_dofs}) must be a multiple of block_width "
                f"({block_width}).")

        dof_bounds.fill_(1.0)
        wp.launch(
            _get_collision_bounds_dense_wp_kernel,
            dim=(self._contact_launch_dim(), num_dofs // block_width),
            inputs=[
                self.collision_radius,
                self.collision_barrier_ratio,
                cp_dx,
                self.cp_dx_at_nm_iteration_0,
                self.collision_kinematic_gaps,
                self.collision_normals,
                self.collision_indices_a,
                self.collision_indices_b,
                cp_delta_dx,
                b_dense,
                block_width,
                self.count,
            ],
            outputs=[dof_bounds],
            device=b_dense.device)
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
            dim=self._contact_launch_dim(),
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
                    self.collision_indices_b,
                    self.count],
            outputs=[energy],
            adjoint=False
        )
        return energy
        # print("collision energy: ", self.num_contacts, energy.numpy())

    def gradient(self, dx, x0, coeff, gradient=None):
        r"""
        Compute the gradient of the collision energy.

        Args:
            dx (wp.array(dtype=wp.vec3)): Current CP displacements with the current dofs of size :math:`(\text{num_pts}, 3)`
            x0 (wp.array(dtype=wp.vec3)): Rest contact point positions of size :math:`(\text{num_pts}, 3)`
            coeff (float): Coefficient for the collision energy.
            gradient (wp.array(dtype=wp.vec3), optional): Preallocated output, normally of
                size ``max_contacting_pairs``. Required for cuda-graph capture, which
                forbids allocation. Zeroed on entry, since the count guard leaves the
                unused tail unwritten. Defaults to allocating a ``num_contacts``-sized
                array, which is what the non-capturable path expects.

        Returns:
            wp.array(dtype=wp.vec3): Gradient of the collision energy of size :math:`(\text{num_contacts}, 3)`
        """
        if gradient is None:
            gradient = wp.zeros(
                self.num_contacts, dtype=wp.vec3, device=dx.device)
        else:
            # The kernel writes slot c for every c below the device count, which is
            # independent of this buffer's length -- so an undersized buffer is an
            # out-of-bounds write, and Warp release builds strip the bounds assert.
            if gradient.shape[0] < self._contact_launch_dim():
                raise ValueError(
                    f"gradient buffer holds {gradient.shape[0]} entries but the launch "
                    f"covers {self._contact_launch_dim()}; size it to "
                    f"max_contacting_pairs ({self.max_contacting_pairs}).")
            gradient.zero_()

        wp.launch(
            kernel=_collision_gradient_wp_kernel,
            dim=self._contact_launch_dim(),
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
                    self.collision_indices_b,
                    self.count],
            outputs=[gradient],
            adjoint=False
        )

        return gradient

    def hessian(self, dx, x0, coeff, hessian_blocks=None):
        r"""
        Compute the hessian of the collision energy.

        Args:
            dx (wp.array(dtype=wp.vec3)): Current CP displacements with the current dofs of size :math:`(\text{num_pts}, 3)`
            x0 (wp.array(dtype=wp.vec3)): Rest contact point positions of size :math:`(\text{num_pts}, 3)`
            coeff (float): Coefficient for the collision energy.
            hessian_blocks (wp.array(dtype=wp.mat33), optional): Preallocated output,
                normally of size ``max_contacting_pairs``. Required for cuda-graph
                capture, which forbids allocation. Zeroed on entry, since the count guard
                leaves the unused tail unwritten -- and the padded blocks must be exactly
                zero so they contribute nothing to :math:`J^T H J`. Defaults to
                allocating a ``num_contacts``-sized array.

        Returns:
            wp.array(dtype=wp.mat33): Hessian of the collision energy of size :math:`(\text{num_contacts}, 3, 3)`
        """
        if hessian_blocks is None:
            hessian_blocks = wp.zeros(
                self.num_contacts, dtype=wp.mat33, device=dx.device)
        else:
            # See gradient(): an undersized buffer is an out-of-bounds write that Warp
            # release builds will not catch.
            if hessian_blocks.shape[0] < self._contact_launch_dim():
                raise ValueError(
                    f"hessian buffer holds {hessian_blocks.shape[0]} entries but the "
                    f"launch covers {self._contact_launch_dim()}; size it to "
                    f"max_contacting_pairs ({self.max_contacting_pairs}).")
            hessian_blocks.zero_()

        wp.launch(
            kernel=_collision_hessian_diag_blocks_wp_kernel,
            dim=self._contact_launch_dim(),
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
                    self.collision_indices_b,
                    self.count],
            outputs=[hessian_blocks],
            adjoint=False
        )
        return hessian_blocks


class ChunkedCollisionHessian:
    r"""Reduces the collision Hessian :math:`J^T H J` a chunk of contacts at a time.

    :math:`H` is block diagonal -- one :math:`3 \times 3` block per contact -- so the
    reduction is separable over contacts:

    .. math::
        J^T H J = \sum_c J_c^T H_c J_c

    where :math:`J_c` are the rows of :math:`J` belonging to chunk :math:`c`. Each term
    is a full :math:`(\text{num_dofs}, \text{num_dofs})` matrix that simply accumulates,
    so the chunks can be visited one at a time and the full
    :math:`(3 \times \text{max_contacting_pairs}, \text{num_dofs})` Jacobian never has to
    exist. That is the whole point: peak memory is set by ``chunk_size`` rather than by
    contact capacity. For ``simplicits_stacking_cubes`` (600 DOFs, 50000 contact capacity)
    a full dense Jacobian is 343 MiB against 14 MiB for a 2048-contact chunk, and the
    ``H @ J`` scratch is the same size again in both cases, so the ratio holds overall.

    Every buffer is allocated once, in ``__init__``, and every launch dimension is fixed,
    so ``reduce`` performs no allocation and no host sync. The trip count is the one
    remaining host-visible quantity, and :func:`reduce` deliberately walks the full
    capacity rather than reading the live count; :func:`reduce_capturable` replaces the
    Python loop with a ``wp.capture_while`` over a device predicate, which both makes it
    graph-capturable and skips the empty tail.

    Args:
        collision (Collision): Source of the contact arrays and the device contact count.
        num_dofs (int): Number of simulation DOFs, i.e. columns of the subspace basis.
        chunk_size (int, optional): Contacts per chunk. Need not divide
            ``collision.max_contacting_pairs``; a partial final chunk is safe because the
            gather kernels zero any lane past the live contact count. Clamped down to the
            capacity. Defaults to 2048.
        device (optional): Warp device for the buffers. Defaults to ``collision.count``'s.
    """

    def __init__(self, collision, num_dofs, chunk_size=2048, device=None):
        capacity = collision.max_contacting_pairs
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

        # chunk_size deliberately need NOT divide the capacity. A final chunk that hangs
        # off the end is safe because `count <= capacity` is an enforced invariant --
        # detection drops slots past the capacity (_detect_particle_collisions_wp_kernel)
        # and _clamp_contact_count_wp_kernel pins the count on device afterwards. So a
        # lane with `g >= count` also has `g >= capacity`, and all three gather kernels
        # return early on exactly that test, zeroing their output. Requiring exact
        # divisibility bought nothing and forced callers into a divisor search that
        # collapsed to chunk_size=1 for capacities with no convenient factor.
        chunk_size = min(chunk_size, capacity)

        if device is None:
            device = collision.count.device

        self.collision = collision
        self.chunk_size = chunk_size
        self.num_chunks = -(-capacity // chunk_size)  # ceil
        self.num_dofs = num_dofs
        self.device = device

        # Device-resident cursor and loop predicate. Both must be arrays rather than
        # Python ints: a captured graph bakes in host values, so an int cursor would
        # replay every step at whatever offset it happened to hold at capture time.
        self.chunk_start = wp.zeros(1, dtype=int, device=device)
        self.loop_cond = wp.zeros(1, dtype=int, device=device)

        self.j_chunk = wp.zeros((3 * chunk_size, num_dofs),
                                dtype=wp.float32, device=device)
        self.h_chunk = wp.zeros(chunk_size, dtype=wp.mat33, device=device)
        self.g_chunk = wp.zeros(chunk_size, dtype=wp.vec3, device=device)

        # Torch views onto the same memory -- the reduction is a pair of GEMMs, which
        # torch does far better than a hand-written kernel. wp.to_torch aliases rather
        # than copies, so these stay valid for the lifetime of the buffers above.
        self._t_j_chunk = wp.to_torch(self.j_chunk)
        self._t_h_chunk = wp.to_torch(self.h_chunk)
        # (chunk, 3) -> (3*chunk,), a view: this is the vector J_c^T multiplies.
        self._t_g_chunk = wp.to_torch(self.g_chunk).reshape(3 * chunk_size)
        # Scratch for the intermediate H @ J. Preallocated because torch.bmm would
        # otherwise allocate inside the capture region.
        self._hj = torch.zeros(chunk_size, 3, num_dofs,
                               dtype=self._t_j_chunk.dtype,
                               device=self._t_j_chunk.device)

        # cuBLAS creates its handle lazily, on the first GEMM of a given device/dtype, and
        # cublasCreate is illegal inside a capture region -- it fails with
        # CUBLAS_STATUS_NOT_INITIALIZED and poisons the CUDA context. So force the handle
        # into existence here, on the buffers that reduce_capturable will use, while we
        # are still guaranteed to be outside any capture. The buffers are zeroed, so this
        # computes nothing; only the handle matters. The rest of the capturable path gets
        # away without this only because scene setup incidentally runs matmuls first,
        # which is not something a standalone reducer should have to rely on.
        _warm = torch.zeros(num_dofs, num_dofs, dtype=self._t_j_chunk.dtype,
                            device=self._t_j_chunk.device)
        hess_reduction(self._t_j_chunk, self._t_h_chunk, out=_warm, HJ=self._hj)
        _warm[0].addmv_(self._t_j_chunk.transpose(0, 1), self._t_g_chunk)

    def _advance(self):
        wp.launch(kernel=_advance_chunk_start_wp_kernel, dim=1,
                  inputs=[self.chunk_start, self.chunk_size], device=self.device)

    def _reduce_one_chunk(self, b_dense, h_full, out):
        r"""Builds the chunk at the current cursor and accumulates its contribution."""
        self.collision.build_jacobian_chunk(b_dense, self.chunk_start, self.j_chunk)
        self.collision.gather_hessian_chunk(h_full, self.chunk_start, self.h_chunk)
        hess_reduction(self._t_j_chunk, self._t_h_chunk, out=out, HJ=self._hj,
                       accumulate=True)
        self._advance()

    def _reduce_one_gradient_chunk(self, b_dense, g_full, out):
        r"""Accumulates :math:`J_c^T \, (dE/dx)_c` for the chunk at the current cursor."""
        self.collision.build_jacobian_chunk(b_dense, self.chunk_start, self.j_chunk)
        self.collision.gather_gradient_chunk(g_full, self.chunk_start, self.g_chunk)
        # addmv_ is the in-place out += A @ v GEMV; no allocation, no extra buffer.
        out.addmv_(self._t_j_chunk.transpose(0, 1), self._t_g_chunk)
        self._advance()

    def accumulate_gradient_capturable(self, b_dense, g_full, out):
        r"""Adds :math:`J^T \, dE/dx` into ``out``, chunked, under ``wp.capture_while``.

        The host path does this with one ``bsr_mv`` against the sparse collision
        Jacobian, whose topology changes with the contact set and so cannot be captured.
        This is the same product, accumulated chunk by chunk from gathered rows.

        Unlike the Hessian reduction this deliberately does **not** zero ``out``: the
        scene gradient already holds the elastic and point-wise terms by the time
        collisions are added.

        Args:
            b_dense (wp.array2d(dtype=wp.float32)): Dense subspace basis.
            g_full (wp.array(dtype=wp.vec3)): Per-contact :math:`dE/dx` over capacity.
            out (torch.Tensor): ``(num_dofs,)`` accumulator, added into.

        Returns:
            torch.Tensor: ``out``.
        """
        if out.numel() != self.num_dofs:
            raise ValueError(
                f"out has {out.numel()} entries, expected {self.num_dofs}.")
        self.chunk_start.zero_()
        self._update_cond()

        def while_body():
            self._reduce_one_gradient_chunk(b_dense, g_full, out)
            self._update_cond()

        wp.capture_while(self.loop_cond, while_body=while_body)
        return out

    def reduce(self, b_dense, h_full, out):
        r"""Accumulates :math:`J^T H J` into ``out`` with a host-side chunk loop.

        Walks the full contact *capacity* rather than the live count, so the trip count
        is independent of the contact state and nothing is read back to the host. Chunks
        beyond the count contribute exactly zero -- both gather kernels zero their
        padding -- so the result is identical to reducing only the live contacts, just at
        the cost of some empty GEMMs.

        Args:
            b_dense (wp.array2d(dtype=wp.float32)): Dense subspace basis.
            h_full (wp.array(dtype=wp.mat33)): Per-contact Hessian blocks over capacity.
            out (torch.Tensor): Preallocated ``(num_dofs, num_dofs)`` output, overwritten.

        Returns:
            torch.Tensor: ``out``.
        """
        self._validate(b_dense, h_full, out)
        out.zero_()
        self.chunk_start.zero_()
        for _ in range(self.num_chunks):
            self._reduce_one_chunk(b_dense, h_full, out)
        return out

    def reduce_capturable(self, b_dense, h_full, out, accumulate=False):
        r"""Same reduction, but with the chunk loop as a ``wp.capture_while``.

        Two things change versus :func:`reduce`. The loop becomes a device-side
        conditional graph node, so the whole reduction can live inside a captured graph;
        and the predicate tests the live contact count, so replay stops after
        ``ceil(count / chunk_size)`` chunks instead of always walking the capacity.

        Args:
            b_dense (wp.array2d(dtype=wp.float32)): Dense subspace basis.
            h_full (wp.array(dtype=wp.mat33)): Per-contact Hessian blocks over capacity.
            out (torch.Tensor): Preallocated ``(num_dofs, num_dofs)`` output.
            accumulate (bool, optional): Add into ``out`` rather than overwriting it.
                Lets the scene Hessian be accumulated in place, avoiding a second
                ``(num_dofs, num_dofs)`` scratch -- 56 MiB at 3840 DOF. Defaults to False.

        Returns:
            torch.Tensor: ``out``.
        """
        self._validate(b_dense, h_full, out)
        if not accumulate:
            out.zero_()
        self.chunk_start.zero_()
        self._update_cond()

        def while_body():
            self._reduce_one_chunk(b_dense, h_full, out)
            self._update_cond()

        wp.capture_while(self.loop_cond, while_body=while_body)
        return out

    def _update_cond(self):
        wp.launch(kernel=_chunk_loop_cond_wp_kernel, dim=1,
                  inputs=[self.chunk_start, self.collision.count, self.loop_cond],
                  device=self.device)

    def _validate(self, b_dense, h_full, out):
        if b_dense.shape[1] != self.num_dofs:
            raise ValueError(
                f"b_dense has {b_dense.shape[1]} columns but this reducer was built for "
                f"{self.num_dofs} DOFs.")
        if h_full.shape[0] < self.collision.max_contacting_pairs:
            raise ValueError(
                f"h_full holds {h_full.shape[0]} blocks but capacity is "
                f"{self.collision.max_contacting_pairs}.")
        if tuple(out.shape) != (self.num_dofs, self.num_dofs):
            raise ValueError(
                f"out has shape {tuple(out.shape)}, expected "
                f"({self.num_dofs}, {self.num_dofs}).")
