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
from kaolin.physics.utils.warp_utilities import bsr_to_torch

__all__ = ['Collision']

# TODO:Consider calling detect_particle_collisions each iteration of newton step
# - Pros:
#   - Allows for smaller collision radius
# - Cons:
#   - Slower than once per timestep
# TODO: Separate the cps from qps. Currently we use qps for both.
# TODO: Currently self collisions are disabled via high immune radius.
# TODO: Consider floor friction.
# TODO: In collision bounds kernel, swap out J_a, J_b indices with cp_to_obj_id, and obj_id_to_dof maps.
# TODO: Allow variable collision radii for different objects.
# TODO: Expose friction parameters to users

NULL_ELEMENT_INDEX = wp.constant(-1)


@wp.kernel
def _detect_particle_collisions(
    max_contacts: int,                    # max number of contacts to detect
    grid: wp.uint64,                      # hashgrid for current points
    radius: float,                        # collision radius
    self_collision_immune_radius: float,  # ignore self collisions within radius
    # current positions of points B*z + x0
    pos_cur: wp.array(dtype=wp.vec3),
    pos_rest: wp.array(dtype=wp.vec3),    # rest positions of points x0
    # displacements of points ... velocity of points * dt (how much they moved in current timestep)B*z_k - B*z_0 where k is newton iteration
    pos_delta: wp.array(dtype=wp.vec3),
    qp_obj_ids: wp.array(dtype=int),      # point to object id mapping
    cp_is_static: wp.array(dtype=int),    # 1 for true, 0 for false
    count: wp.array(dtype=int),           # number of contacts detected
    normals: wp.array(dtype=wp.vec3),     # contact normals
    kinematic_gaps: wp.array(dtype=wp.vec3),  # kinematic gaps
    indices_a: wp.array(dtype=int),       # collision indices pairs a-b
    indices_b: wp.array(dtype=int),       # collision indices pairs a-b
):
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
def _collision_offset(
    c: int,
    dx_cur: wp.array(dtype=wp.vec3),
    dx_start_of_timestep: wp.array(dtype=wp.vec3),
    kinematic_gaps: wp.array(dtype=wp.vec3),
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):
    idx_a = indices_a[c]
    idx_b = indices_b[c]

    pos_delta_a = dx_cur[idx_a] - dx_start_of_timestep[idx_a]
    pos_delta_b = dx_cur[idx_b] - dx_start_of_timestep[idx_b]

    offset = pos_delta_a + kinematic_gaps[c]
    if idx_b != NULL_ELEMENT_INDEX:
        offset -= pos_delta_b
    return offset


@wp.func
def _collision_target_distance(
    c: int,
    radius: float,
    indices_a: wp.array(dtype=int),
    indices_b: wp.array(dtype=int),
):
    return wp.select(indices_b[c] == NULL_ELEMENT_INDEX, 2.0, 1.0) * radius
    # return 2.0 * radius


@wp.kernel
def _collision_energy(
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
):
    c = wp.tid()

    offset = _collision_offset(
        c, dx_cur, dx_start_of_timestep, kinematic_gaps, indices_a, indices_b)
    rc = _collision_target_distance(c, radius, indices_a, indices_b)
    rp_ratio = barrier_distance_ratio

    nor = normals[c]
    d = wp.dot(offset, nor)
    d_hat = d / rc

    # Check its within radiuses.
    if rp_ratio < d_hat and d_hat <= 1.0:
        d_min_l_squared = (d_hat - 1.0) * (
            d_hat - 1.0
        )  # quadratic term ensures energy is 0 when d = rc
        E = -d_min_l_squared * wp.log(
            d_hat - rp_ratio
        )  # log barrier term. inf when two rp's overlaps.

        # friction energy
        dc = d_hat - 1.0
        dp = d_hat - rp_ratio
        barrier = 2.0 * wp.log(dp)

        dE_d_hat = -dc * (barrier + dc / dp)
        vt = (offset - d * nor) / dt  # tangential velocity
        vt_norm = wp.length(vt)

        mu_fn = -mu * dE_d_hat / rc  # yield force

        E += (
            mu_fn
            * dt
            * (
                0.5 * nu * vt_norm * vt_norm
                + wp.select(
                    vt_norm < 1.0,
                    vt_norm - 1.0 / 3.0,
                    vt_norm * vt_norm * (1.0 - vt_norm / 3.0),
                )
            )
        )
        ##

    else:
        E = 0.0

    wp.atomic_add(energies, 0, coeff * E)


@wp.kernel
def _collision_gradient(coeff: float,
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
                        gradient: wp.array(dtype=wp.vec3)):
    """Calculates the collision energy for each object point
    """
    c = wp.tid()

    offset = _collision_offset(
        c, dx_cur, dx_start_of_timestep, kinematic_gaps, indices_a, indices_b)
    rc = _collision_target_distance(c, radius, indices_a, indices_b)
    rp_ratio = barrier_distance_ratio

    nor = normals[c]
    d = wp.dot(offset, nor)
    d_hat = d / rc

    if rp_ratio < d_hat and d_hat <= 1.0:
        dc = d_hat - 1.0
        dp = d_hat - rp_ratio
        barrier = 2.0 * wp.log(dp)

        dE_d_hat = -dc * (barrier + dc / dp)
        gradient[c] = dE_d_hat / rc * nor

        # friction
        vt = (offset - d * nor) / dt  # tangential velocity
        vt_norm = wp.length(vt)
        vt_dir = wp.normalize(vt)  # avoids dealing with 0

        mu_fn = -mu * dE_d_hat / rc  # yield force

        f1_over_vt_norm = wp.select(
            vt_norm < 1.0, 1.0 / vt_norm, 2.0 - vt_norm)
        gradient[c] += mu_fn * (f1_over_vt_norm + nu) * vt
        ###

    else:
        gradient[c] = wp.vec3(0.0)

    gradient[c] = coeff * gradient[c]


@wp.kernel
def _collision_hessian_diag_blocks(coeff: float,
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
                                   hessian: wp.array(dtype=wp.mat33)):
    c = wp.tid()

    offset = _collision_offset(
        c, dx_cur, dx_start_of_timestep, kinematic_gaps, indices_a, indices_b)
    rc = _collision_target_distance(c, radius, indices_a, indices_b)
    rp_ratio = barrier_distance_ratio

    nor = normals[c]
    d = wp.dot(offset, nor)
    d_hat = d / rc

    if rp_ratio < d_hat and d_hat <= 1.0:
        dc = d_hat - 1.0
        dp = d_hat - rp_ratio
        barrier = 2.0 * wp.log(dp)

        dE_d_hat = -dc * (barrier + dc / dp)

        dbarrier_d_hat = 2.0 / dp
        ddcdp_d_hat = (dp - dc) / (dp * dp)

        d2E_d_hat2 = -(barrier + dc / dp) - dc * (dbarrier_d_hat + ddcdp_d_hat)
        hessian[c] = d2E_d_hat2 / (rc * rc) * wp.outer(nor, nor)

        # friction

        vt = (offset - d * nor) / dt  # tangential velocity
        vt_norm = wp.length(vt)
        vt_dir = wp.normalize(vt)  # avoids dealing with 0

        mu_fn = -mu * dE_d_hat / rc  # yield force

        f1_over_vt_norm = wp.select(
            vt_norm < 1.0, 1.0 / vt_norm, 2.0 - vt_norm)

        # regularization such that f / H dt <= k v (penalizes friction switching dir)
        friction_slip_reg = 0.1
        df1_d_vtn = wp.max(
            2.0 * (1.0 - vt_norm),
            friction_slip_reg / (0.5 * friction_slip_reg + vt_norm),
        )

        vt_perp = wp.cross(vt_dir, nor)
        hessian[c] += (
            mu_fn
            / dt
            * (
                (df1_d_vtn + nu) * wp.outer(vt_dir, vt_dir)
                + (f1_over_vt_norm + nu) * wp.outer(vt_perp, vt_perp)
            )
        )
        ###

    else:
        hessian[c] = wp.mat33(0.0)

    hessian[c] = coeff * hessian[c]


@wp.kernel
def _compute_collision_bounds(
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
):
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
    offset = _collision_offset(
        c, dx_cur, dx_start_of_timestep, kinematic_gaps, indices_a, indices_b)
    rc = _collision_target_distance(c, radius, indices_a, indices_b)
    rp = barrier_distance_ratio * rc
    gap_cur = rp - wp.dot(offset, nor)

    if gap_cur >= 0.0:
        # Missed due to too large timestep. Can't do anything now
        return

    MAX_PROGRESS = 0.75
    max_delta_d = 0.5 * MAX_PROGRESS * gap_cur

    # TODO: Change this to use the cp_to_dof mapping? In case I don't have these J_a, J_b matrices
    #
    # Jacobian tells me which dofs affect which colliding particles
    # Using two jacobians Ja, Jb you can tell which DOFs affect the first colliding particle
    # and the second colliding particle
    # Using warp sparse matrices I can use the same kernel to compute the bounds
    if delta_d_a < 0.0:  # getting closer
        t_max = wp.clamp(max_delta_d / delta_d_a, 0.0, 1.0)
        if t_max < 1.0:
            dof_beg = jacobian_a_offsets[c]
            dof_end = jacobian_a_offsets[c + 1]
            for dof in range(dof_beg, dof_end):
                wp.atomic_min(dof_t_max, jacobian_a_columns[dof], t_max)

    if delta_d_b < 0.0:  # getting closer
        t_max = wp.clamp(max_delta_d / delta_d_b, 0.0, 1.0)
        if t_max < 1.0:
            dof_beg = jacobian_b_offsets[c]
            dof_end = jacobian_b_offsets[c + 1]
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

        Returns:
            None
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
            kernel=_detect_particle_collisions,
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
            # TODO: Ask if .cpu() is necessary?
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
            unique_pairs = torch.unique(object_pairs, dim=0)
            self.object_pairs = unique_pairs.numpy()  # TODO: Ask if .numpy() is necessary?
        else:
            # If no collisions, empty list
            self.object_pairs = []

        return

    def build_jacobian(self, cp_w, cp_x0, cp_is_static=None):
        r""" Builds the jacobians of the collision points w.r.t the dofs. For contact pairs :math:`x_a \in \mathbb{R}^3, x_b \in \mathbb{R}^3`, the jacobians are:

        .. math::
            J_a = \frac{\partial x_a}{\partial z} \in \mathbb{R}^{3 \times n}
            J_b = \frac{\partial x_b}{\partial z} \in \mathbb{R}^{3 \times n}
            J = J_a - J_b \in \mathbb{R}^{3 \times n}

        The difference, :math:`J = J_a - J_b`, is the jacobian of the collision gaps.

        Args:
            cp_w (wp.array2d(dtype=wp.float32)): Contact point skinning weights.
            cp_x0 (wp.array(dtype=wp.vec3)): Rest contact point positions.
        """

        # indices of the colliding point pairs
        num_pts = self.num_contacts
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
            
            t_ind_a = wp.to_torch(ind_a)
            t_ind_b = wp.to_torch(ind_b)

            J_a = sparse_collision_jacobian_matrix(cp_w, cp_x0, indices=ind_a, cp_is_static=cp_is_static)
            J_b = sparse_collision_jacobian_matrix(cp_w, cp_x0, indices=ind_b, cp_is_static=cp_is_static)

            self.collision_J_a = J_a
            self.collision_J_b = J_b
            self.collision_J_a.nnz_sync()
            self.collision_J_b.nnz_sync()

        J = self.collision_J_a - self.collision_J_b
        self.collision_J = J #wps.bsr_copy(J, block_shape=(3, 12))
        self.collision_J.nnz_sync()

        self.collision_J_dense = bsr_to_torch(self.collision_J).to_dense()

        return

    def compute_bounds(self, cp_delta_dx, cp_dx, cp_x0):
        r""" Compute the bounds of the update for each dof. This is used to guarantee intersection-free contact. See :func:`kaolin.physics.optimization.apply_bounds` for more details.

        Args:
            cp_delta_dx (wp.array(dtype=wp.vec3)): B*dz where dz is the newton update
            cp_dx (wp.array(dtype=wp.vec3)): B*z where z is the current dofs
            cp_x0 (wp.array(dtype=wp.vec3)): Rest contact point positions.
        """
        if self.num_contacts == 0 and not self.bounds:
            return None

        # Notes:
        # TODO: Remove later
        # u_0 is dofs at start of newton step
        # delta_du : current newton step
        # du is accumulated update over the newton iterations
        # du + delta_du : the updated dofs relative to start dofs
        # u <- du + delta_du + u_0
        # delta_du = -inv(H) *rhs..... dz = z - z_0  ..,  current dofs - start of dofs at newton step

        # Inputs: Position increments of the contact points

        # Output: vector of size num_column_blocks in J_a. If J_a is csr, then num_blocks=J.shape[1]
        blockwise_bounds = wp.ones(
            (self.collision_J_a.ncol), dtype=float, device=self.collision_J_a.device)

        wp.launch(
            _compute_collision_bounds,
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
            dx (wp.array(dtype=wp.vec3)): Current CP displacements with the current dofs.
            x0 (wp.array(dtype=wp.vec3)): Rest contact point positions.
            coeff (float): Coefficient for the collision energy.
            energy (wp.array(dtype=float)): Output energy. Used for cuda-graph capture
        """

        if energy is None:
            energy = wp.zeros(1, dtype=float)

        wp.launch(
            kernel=_collision_energy,
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
            dx (wp.array(dtype=wp.vec3)): Current CP displacements with the current dofs.
            x0 (wp.array(dtype=wp.vec3)): Rest contact point positions.
            coeff (float): Coefficient for the collision energy.
        """
        gradient = wp.zeros(
            self.num_contacts, dtype=wp.vec3, device=dx.device)

        wp.launch(
            kernel=_collision_gradient,
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
            dx (wp.array(dtype=wp.vec3)): Current CP displacements with the current dofs.
            x0 (wp.array(dtype=wp.vec3)): Rest contact point positions.
            coeff (float): Coefficient for the collision energy.
        """

        hessian_blocks = wp.zeros(
            self.num_contacts, dtype=wp.mat33, device=dx.device)

        wp.launch(
            kernel=_collision_hessian_diag_blocks,
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
