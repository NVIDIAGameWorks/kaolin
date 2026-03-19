# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

import warp as wp
import torch
from newton import Contacts
from newton._src.geometry import ParticleFlags
from newton._src.sim.model import Model
from newton._src.sim.state import State

from newton._src.math import orthonormal_basis as build_orthonormal_basis


class mat32(wp.types.matrix(shape=(3, 2), dtype=wp.types.float32)):
    r"""Warp 3x2 float32 matrix type; used for tangent-plane basis (e.g. friction)."""
    pass

# From Ty Trusty's soft_robots code
# change to sum over particles

@wp.kernel
def _contact_subspace_energy(
    particle_pos: wp.array(dtype=wp.vec3),
    particle_prev_pos: wp.array(dtype=wp.vec3),
    particle_vol: wp.array(dtype=float),
    contact_samples: wp.array(dtype=int),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    shape_ke: wp.array(dtype=float),
    shape_kf: wp.array(dtype=float),
    shape_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    particle_ke: float,
    particle_kf: float,
    particle_mu: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    dt: float,
    friction_epsilon: float,
    lagged_body_contact_force_norm: wp.array(dtype=float),
    friction_use_lagged_body_contact_force_norm: bool,
    coeff_of_restitution: float,
    velocity_penalty_kv: float,
    # output
    contact_energy: wp.array(dtype=float),
):
    r"""
    Compute contact energy including collision, friction, and velocity-based restitution terms.

    This kernel computes the contact energy for particle-shape collisions using IPC-style friction
    and optional velocity-level restitution penalty. The total energy consists of collision penalty,
    friction dissipation, and optional velocity penalty for bounce effects. Launched with
    dim=soft_contact_max; each thread processes one contact and atomically adds to contact_energy[0].

    Args:
        particle_pos (wp.array(dtype=wp.vec3)): Current particle positions (world), length num_particles.
        particle_prev_pos (wp.array(dtype=wp.vec3)): Previous-step particle positions (world), length num_particles.
        particle_vol (wp.array(dtype=float)): Particle volumes, length num_particles.
        contact_samples (wp.array(dtype=int)): Maps contact sample index -> particle index; length num_samples.
        particle_radius (wp.array(dtype=float)): Per-particle radius, length num_particles.
        particle_flags (wp.array(dtype=wp.int32)): Per-particle flags (e.g. ACTIVE), length num_particles.
        contact_count (wp.array(dtype=int)): Single-element array holding the number of active contacts.
        contact_particle (wp.array(dtype=int)): For each contact, the sample index; length soft_contact_max.
        contact_shape (wp.array(dtype=int)): For each contact, the shape index; length soft_contact_max.
        contact_body_pos (wp.array(dtype=wp.vec3)): For each contact, body-space contact point; length soft_contact_max.
        contact_body_vel (wp.array(dtype=wp.vec3)): For each contact, body-space contact velocity; length soft_contact_max.
        contact_normal (wp.array(dtype=wp.vec3)): For each contact, world-space contact normal; length soft_contact_max.
        shape_ke (wp.array(dtype=float)): Per-shape stiffness; length num_shapes.
        shape_kf (wp.array(dtype=float)): Per-shape friction stiffness; length num_shapes.
        shape_mu (wp.array(dtype=float)): Per-shape friction coefficient; length num_shapes.
        shape_body (wp.array(dtype=int)): Per-shape rigid body index; length num_shapes.
        particle_ke (float): Global particle contact stiffness.
        particle_kf (float): Global particle contact friction stiffness.
        particle_mu (float): Global particle contact friction coefficient.
        body_q (wp.array(dtype=wp.transform)): Body transforms (world); length num_bodies.
        body_qd (wp.array(dtype=wp.spatial_vector)): Body spatial velocities; length num_bodies.
        body_com (wp.array(dtype=wp.vec3)): Body center-of-mass in body frame; length num_bodies.
        dt (float): Time step.
        friction_epsilon (float): IPC friction smoothing parameter.
        lagged_body_contact_force_norm (wp.array(dtype=float)): Lagged normal force magnitude per sample; length num_samples.
        friction_use_lagged_body_contact_force_norm (bool): If True, use lagged force norm for friction.
        coeff_of_restitution (float): Coefficient of restitution for velocity penalty.
        velocity_penalty_kv (float): Stiffness for velocity-level restitution penalty.
        contact_energy (wp.array(dtype=float)): Single-element array; kernel atomically adds the total contact energy here.
    """
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return

    sample_index = contact_particle[tid]
    particle_index = contact_samples[sample_index]
    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    shape_index = contact_shape[tid]
    if shape_index < 0:
        return
    body_index = shape_body[shape_index]

    # Get material properties
    ke = 0.5 * (particle_ke + shape_ke[shape_index])

    # Get particle position and radius
    pos = particle_pos[sample_index]
    radius = particle_radius[particle_index]
    vol = particle_vol[particle_index]
    prev_pos = particle_prev_pos[sample_index]

    # Get body transform
    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    # Body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])

    # Contact normal
    n = contact_normal[tid]

    # Penetration depth
    penetration_depth = wp.dot(n, pos - bx) - radius

    if penetration_depth > 0:
        return

    dx = pos - prev_pos

    # use the instantaneous velocity
    r = bx - wp.transform_point(X_wb, X_com)
    body_v_s = wp.spatial_vector()
    if body_index >= 0:
        body_v_s = body_qd[body_index]

    body_w = wp.spatial_top(body_v_s)
    body_v = wp.spatial_bottom(body_v_s)

    # compute the body velocity at the particle position
    bv = body_v + wp.cross(body_w, r) + \
        wp.transform_vector(X_wb, contact_body_vel[tid])

    relative_translation = dx - bv * dt

    # friction
    e0, e1 = build_orthonormal_basis(n)

    T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

    u = wp.transpose(T) * relative_translation
    eps_u = friction_epsilon * dt

    mu = 0.5 * (particle_mu + shape_mu[shape_index])

    u_norm = wp.length(u)
    if u_norm > eps_u:
        f0 = u_norm - eps_u / 3.0
    else:
        u_norm_div_eps_u = u_norm / eps_u
        f0 = u_norm * u_norm_div_eps_u * (1.0 - u_norm_div_eps_u / 3.0)

    body_contact_force_norm = ke * wp.abs(penetration_depth)
    if friction_use_lagged_body_contact_force_norm:
        friction_energy = mu * \
            lagged_body_contact_force_norm[sample_index] * f0
    else:
        friction_energy = mu * body_contact_force_norm * f0

    collision_energy = 0.5 * ke * penetration_depth * penetration_depth
   

    # Velocity-level restitution penalty (bounce):
    # vn = n · relative_velocity, where relative_velocity ≈ relative_translation / dt
    vn = wp.dot(n, relative_translation) / dt
    neg_vn = -vn
    vel_energy = 0.0
    if neg_vn > 0.0:
        vterm = (1.0 + coeff_of_restitution) * neg_vn
        vel_energy = 0.5 * velocity_penalty_kv * vterm * vterm

    # e = vol * (friction_energy)
    e = vol * (collision_energy + friction_energy + vel_energy)
    if shape_index >= 0:
        wp.atomic_add(contact_energy, 0, e)


@wp.kernel
def _contact_subspace_gradient(
    particle_pos: wp.array(dtype=wp.vec3),
    particle_prev_pos: wp.array(dtype=wp.vec3),
    particle_vol: wp.array(dtype=float),
    contact_samples: wp.array(dtype=int),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    shape_ke: wp.array(dtype=float),
    shape_kf: wp.array(dtype=float),
    shape_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    particle_ke: float,
    particle_kf: float,
    particle_mu: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    dt: float,
    friction_epsilon: float,
    lagged_body_contact_force_norm: wp.array(dtype=float),
    friction_use_lagged_body_contact_force_norm: bool,
    coeff_of_restitution: float,
    velocity_penalty_kv: float,
    # output
    particle_gradients: wp.array(dtype=wp.vec3),
):
    r"""
    Compute contact gradient (force) including collision, friction, and velocity-based restitution.

    This kernel computes the gradient of the contact energy with respect to particle positions,
    which represents the contact forces. The gradient includes contributions from collision response,
    IPC-style friction, and optional velocity-level restitution penalty. Launched with
    dim=soft_contact_max; each thread adds (atomically) to particle_gradients using sample indices.

    Args:
        particle_pos (wp.array(dtype=wp.vec3)): Current particle positions (world), length num_particles.
        particle_prev_pos (wp.array(dtype=wp.vec3)): Previous-step particle positions (world), length num_particles.
        particle_vol (wp.array(dtype=float)): Particle volumes, length num_particles.
        contact_samples (wp.array(dtype=int)): Maps contact sample index -> particle index; length num_samples.
        particle_radius (wp.array(dtype=float)): Per-particle radius, length num_particles.
        particle_flags (wp.array(dtype=wp.int32)): Per-particle flags (e.g. ACTIVE), length num_particles.
        contact_count (wp.array(dtype=int)): Single-element array holding the number of active contacts.
        contact_particle (wp.array(dtype=int)): For each contact, the sample index; length soft_contact_max.
        contact_shape (wp.array(dtype=int)): For each contact, the shape index; length soft_contact_max.
        contact_body_pos (wp.array(dtype=wp.vec3)): For each contact, body-space contact point; length soft_contact_max.
        contact_body_vel (wp.array(dtype=wp.vec3)): For each contact, body-space contact velocity; length soft_contact_max.
        contact_normal (wp.array(dtype=wp.vec3)): For each contact, world-space contact normal; length soft_contact_max.
        shape_ke (wp.array(dtype=float)): Per-shape stiffness; length num_shapes.
        shape_kf (wp.array(dtype=float)): Per-shape friction stiffness; length num_shapes.
        shape_mu (wp.array(dtype=float)): Per-shape friction coefficient; length num_shapes.
        shape_body (wp.array(dtype=int)): Per-shape rigid body index; length num_shapes.
        particle_ke (float): Global particle contact stiffness.
        particle_kf (float): Global particle contact friction stiffness.
        particle_mu (float): Global particle contact friction coefficient.
        body_q (wp.array(dtype=wp.transform)): Body transforms (world); length num_bodies.
        body_qd (wp.array(dtype=wp.spatial_vector)): Body spatial velocities; length num_bodies.
        body_com (wp.array(dtype=wp.vec3)): Body center-of-mass in body frame; length num_bodies.
        dt (float): Time step.
        friction_epsilon (float): IPC friction smoothing parameter.
        lagged_body_contact_force_norm (wp.array(dtype=float)): Lagged normal force magnitude per sample; length num_samples.
        friction_use_lagged_body_contact_force_norm (bool): If True, use lagged force norm for friction.
        coeff_of_restitution (float): Coefficient of restitution for velocity penalty.
        velocity_penalty_kv (float): Stiffness for velocity-level restitution penalty.
        particle_gradients (wp.array(dtype=wp.vec3)): Gradient w.r.t. particle positions (sample layout); length num_samples.
            Kernel atomically adds the contact force (negative gradient) per contact to the
            corresponding sample index.
    """
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return

    sample_index = contact_particle[tid]
    particle_index = contact_samples[sample_index]
    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    shape_index = contact_shape[tid]
    if shape_index < 0:
        return
    body_index = shape_body[shape_index]

    # Get material properties
    ke = 0.5 * (particle_ke + shape_ke[shape_index])

    # Get particle position and radius
    pos = particle_pos[sample_index]
    radius = particle_radius[particle_index]
    vol = particle_vol[particle_index]

    # Get body transform
    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    # Body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])

    # Contact normal
    n = contact_normal[tid]

    # Penetration depth
    penetration_depth = wp.dot(n, pos - bx) - radius
    # penetration_depth = wp.dot(n, pos - bx)

    if penetration_depth > 0:
        return

    collision_gradient = n * ke * penetration_depth

    dx = particle_pos[sample_index] - particle_prev_pos[sample_index]

    # use the instantaneous velocity
    r = bx - wp.transform_point(X_wb, X_com)
    body_v_s = wp.spatial_vector()
    if body_index >= 0:
        body_v_s = body_qd[body_index]

    body_w = wp.spatial_top(body_v_s)
    body_v = wp.spatial_bottom(body_v_s)

    # compute the body velocity at the particle position
    bv = body_v + wp.cross(body_w, r) + \
        wp.transform_vector(X_wb, contact_body_vel[tid])

    relative_translation = dx - bv * dt

    # friction
    e0, e1 = build_orthonormal_basis(n)

    T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

    u = wp.transpose(T) * relative_translation
    eps_u = friction_epsilon * dt

    mu = 0.5 * (particle_mu + shape_mu[shape_index])

    u_norm = wp.length(u)
    # IPC friction
    if u_norm > eps_u:
        # constant stage
        f1_SF_over_x = 1.0 / u_norm
    else:
        # smooth transition
        f1_SF_over_x = (-u_norm / eps_u + 2.0) / eps_u

    body_contact_force_norm = ke * wp.abs(penetration_depth)
    if friction_use_lagged_body_contact_force_norm:
        friction_gradient = mu * \
            lagged_body_contact_force_norm[sample_index] * \
            T * (f1_SF_over_x * u)

    else:
        friction_gradient = mu * \
            body_contact_force_norm * T * (f1_SF_over_x * u)

    # Velocity-level restitution penalty gradient
    vn = wp.dot(n, relative_translation) / dt
    neg_vn = -vn
    vel_grad = wp.vec3(0.0, 0.0, 0.0)
    if neg_vn > 0.0:
        coeff = velocity_penalty_kv * (1.0 + coeff_of_restitution) * (1.0 + coeff_of_restitution) * neg_vn
        # d(neg_vn)/d(particle_pos) = -n / dt (since rel_trans depends linearly on pos)
        vel_grad = - (coeff / dt) * n

    # Contact force
    # contact_gradient = vol * (friction_gradient)
    contact_gradient = vol * (collision_gradient + friction_gradient + vel_grad)

    # Write contact force directly to particle_f using atomic_sub
    # Note: particle_f is already sized for samples, so use sample_index
    if shape_index >= 0:
        wp.atomic_add(particle_gradients, sample_index, contact_gradient)
    


@wp.func
def outer_over_norm(u: wp.vec2) -> wp.mat22:
    r"""
    Compute the outer product of a 2D vector divided by its norm.

    Returns the matrix (u ⊗ u) / ||u||, or zero matrix if u has zero length.
    Used in friction Hessian computation.

    Args:
        u (wp.vec2): 2D vector.

    Returns:
        wp.mat22: 2x2 matrix representing the normalized outer product.
    """
    u_norm = wp.length(u)
    if u_norm == 0.0:
        return wp.mat22()
    else:
        return wp.outer(u, u) / wp.length(u)


@wp.kernel
def _contact_subspace_hessian(
    particle_pos: wp.array(dtype=wp.vec3),
    particle_prev_pos: wp.array(dtype=wp.vec3),
    particle_vol: wp.array(dtype=float),
    contact_samples: wp.array(dtype=int),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    shape_ke: wp.array(dtype=float),
    shape_kf: wp.array(dtype=float),
    shape_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    particle_ke: float,
    particle_kf: float,
    particle_mu: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    dt: float,
    friction_epsilon: float,
    lagged_body_contact_force_norm: wp.array(dtype=float),
    friction_use_lagged_body_contact_force_norm: bool,
    coeff_of_restitution: float,
    velocity_penalty_kv: float,
    # output
    particle_hessians: wp.array(dtype=wp.mat33),
):
    r"""
    Compute contact Hessian (stiffness) including collision, friction, and velocity restitution.

    This kernel computes the second derivative (Hessian) of the contact energy with respect to
    particle positions. The Hessian includes contributions from collision stiffness, IPC-style
    friction Hessian, and optional velocity-level restitution penalty Hessian. Used for implicit
    time integration. Launched with dim=soft_contact_max; each thread writes to
    particle_hessians[sample_index] (overwrites, no atomic; one sample may have multiple contacts).

    Args:
        particle_pos (wp.array(dtype=wp.vec3)): Current particle positions (world), length num_particles.
        particle_prev_pos (wp.array(dtype=wp.vec3)): Previous-step particle positions (world), length num_particles.
        particle_vol (wp.array(dtype=float)): Particle volumes, length num_particles.
        contact_samples (wp.array(dtype=int)): Maps contact sample index -> particle index; length num_samples.
        particle_radius (wp.array(dtype=float)): Per-particle radius, length num_particles.
        particle_flags (wp.array(dtype=wp.int32)): Per-particle flags (e.g. ACTIVE), length num_particles.
        contact_count (wp.array(dtype=int)): Single-element array holding the number of active contacts.
        contact_particle (wp.array(dtype=int)): For each contact, the sample index; length soft_contact_max.
        contact_shape (wp.array(dtype=int)): For each contact, the shape index; length soft_contact_max.
        contact_body_pos (wp.array(dtype=wp.vec3)): For each contact, body-space contact point; length soft_contact_max.
        contact_body_vel (wp.array(dtype=wp.vec3)): For each contact, body-space contact velocity; length soft_contact_max.
        contact_normal (wp.array(dtype=wp.vec3)): For each contact, world-space contact normal; length soft_contact_max.
        shape_ke (wp.array(dtype=float)): Per-shape stiffness; length num_shapes.
        shape_kf (wp.array(dtype=float)): Per-shape friction stiffness; length num_shapes.
        shape_mu (wp.array(dtype=float)): Per-shape friction coefficient; length num_shapes.
        shape_body (wp.array(dtype=int)): Per-shape rigid body index; length num_shapes.
        particle_ke (float): Global particle contact stiffness.
        particle_kf (float): Global particle contact friction stiffness.
        particle_mu (float): Global particle contact friction coefficient.
        body_q (wp.array(dtype=wp.transform)): Body transforms (world); length num_bodies.
        body_qd (wp.array(dtype=wp.spatial_vector)): Body spatial velocities; length num_bodies.
        body_com (wp.array(dtype=wp.vec3)): Body center-of-mass in body frame; length num_bodies.
        dt (float): Time step.
        friction_epsilon (float): IPC friction smoothing parameter.
        lagged_body_contact_force_norm (wp.array(dtype=float)): Lagged normal force magnitude per sample; length num_samples.
        friction_use_lagged_body_contact_force_norm (bool): If True, use lagged force norm for friction.
        coeff_of_restitution (float): Coefficient of restitution for velocity penalty.
        velocity_penalty_kv (float): Stiffness for velocity-level restitution penalty.
        particle_hessians (wp.array(dtype=wp.mat33)): 3x3 Hessian blocks per sample (sample layout); length num_samples.
            Each thread writes vol * (collision_hessian + friction_hessian + vel_hessian) at
            sample_index; caller should zero this array before launch if needed.
    """
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return

    sample_index = contact_particle[tid]
    particle_index = contact_samples[sample_index]
    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    shape_index = contact_shape[tid]
    if shape_index < 0:
        return
    body_index = shape_body[shape_index]

    # Get material properties
    ke = 0.5 * (particle_ke + shape_ke[shape_index])

    # Get particle position and radius
    pos = particle_pos[sample_index]
    radius = particle_radius[particle_index]
    vol = particle_vol[particle_index]

    # Get body transform
    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    # Body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])

    # Contact normal
    n = contact_normal[tid]

    # Penetration depth
    penetration_depth = wp.dot(n, pos - bx) - radius

    if penetration_depth > 0:
        return

    dx = particle_pos[sample_index] - particle_prev_pos[sample_index]

    # use the instantaneous velocity
    r = bx - wp.transform_point(X_wb, X_com)
    body_v_s = wp.spatial_vector()
    if body_index >= 0:
        body_v_s = body_qd[body_index]

    body_w = wp.spatial_top(body_v_s)
    body_v = wp.spatial_bottom(body_v_s)

    # compute the body velocity at the particle position
    bv = body_v + wp.cross(body_w, r) + \
        wp.transform_vector(X_wb, contact_body_vel[tid])

    relative_translation = dx - bv * dt

    # friction
    e0, e1 = build_orthonormal_basis(n)

    T = mat32(e0[0], e1[0], e0[1], e1[1], e0[2], e1[2])

    u = wp.transpose(T) * relative_translation
    eps_u = friction_epsilon * dt

    mu = 0.5 * (particle_mu + shape_mu[shape_index])

    u_norm = wp.length(u)
    # IPC friction
    if u_norm > eps_u:
        # constant stage
        f1_SF_over_x = 1.0 / u_norm
    else:
        # smooth transition
        f1_SF_over_x = (-u_norm / eps_u + 2.0) / eps_u

    collision_hessian = wp.outer(n, n) * ke

    if friction_use_lagged_body_contact_force_norm:
        body_contact_force_norm = lagged_body_contact_force_norm[sample_index]
    else:
        body_contact_force_norm = ke * wp.abs(penetration_depth)

    if u_norm > eps_u:
        outer_term = -f1_SF_over_x * wp.outer(u, u) / wp.dot(u, u)
    else:
        # uu^T / ||u||
        outer_term = -outer_over_norm(u) / eps_u / eps_u

    friction_hessian = mu * body_contact_force_norm * (T *
                                                       (f1_SF_over_x * wp.identity(2,
                                                        float) + outer_term)
                                                       # (f1_SF_over_x * wp.identity(2, float))
                                                       * wp.transpose(T))

    # Velocity-level restitution penalty Hessian (active only when approaching)
    vn = wp.dot(n, relative_translation) / dt
    neg_vn = -vn
    vel_hessian = wp.mat33()
    if neg_vn > 0.0:
        kfac = velocity_penalty_kv * (1.0 + coeff_of_restitution) * (1.0 + coeff_of_restitution) / (dt * dt)
        vel_hessian = kfac * wp.outer(n, n)

    if shape_index >= 0:
        wp.atomic_add(particle_hessians, sample_index,
                      vol * (collision_hessian + friction_hessian + vel_hessian))


@wp.kernel
def _update_lagged_body_contact_force_norm_kernel(
    particle_pos: wp.array(dtype=wp.vec3),
    particle_prev_pos: wp.array(dtype=wp.vec3),
    particle_vol: wp.array(dtype=float),
    contact_samples: wp.array(dtype=int),
    particle_radius: wp.array(dtype=float),
    particle_flags: wp.array(dtype=wp.int32),
    contact_count: wp.array(dtype=int),
    contact_particle: wp.array(dtype=int),
    contact_shape: wp.array(dtype=int),
    contact_body_pos: wp.array(dtype=wp.vec3),
    contact_body_vel: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    shape_ke: wp.array(dtype=float),
    shape_kf: wp.array(dtype=float),
    shape_mu: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    particle_ke: float,
    particle_kf: float,
    particle_mu: float,
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    body_com: wp.array(dtype=wp.vec3),
    dt: float,
    friction_epsilon: float,
    friction_use_lagged_body_contact_force_norm: bool,
    # output
    lagged_body_contact_force_norm: wp.array(dtype=float),
):
    r"""
    Update the lagged contact force norm used for friction computation.

    This kernel computes the contact force magnitude (`ke * |penetration_depth|`) for each
    particle-shape contact. These values are stored and used in subsequent iterations
    when friction_use_lagged_body_contact_force_norm is enabled, allowing for a lagged
    friction formulation that improves stability. Launched with dim=`num_samples` (e.g.
    `integration_pt_volume.shape[0]`); each thread may write to lagged_body_contact_force_norm
    at the contact's sample index (non-penetrating contacts set 0).

    Args:
        particle_pos (wp.array(dtype=wp.vec3)): Current particle positions (world), length num_particles.
        particle_prev_pos (wp.array(dtype=wp.vec3)): Previous-step particle positions (unused in this kernel), length num_particles.
        particle_vol (wp.array(dtype=float)): Particle volumes, length num_particles.
        contact_samples (wp.array(dtype=int)): Maps contact sample index -> particle index; length num_samples.
        particle_radius (wp.array(dtype=float)): Per-particle radius, length num_particles.
        particle_flags (wp.array(dtype=wp.int32)): Per-particle flags (e.g. ACTIVE), length num_particles.
        contact_count (wp.array(dtype=int)): Single-element array holding the number of active contacts.
        contact_particle (wp.array(dtype=int)): For each contact, the sample index; length soft_contact_max.
        contact_shape (wp.array(dtype=int)): For each contact, the shape index; length soft_contact_max.
        contact_body_pos (wp.array(dtype=wp.vec3)): For each contact, body-space contact point; length soft_contact_max.
        contact_body_vel (wp.array(dtype=wp.vec3)): For each contact, body-space contact velocity; length soft_contact_max.
        contact_normal (wp.array(dtype=wp.vec3)): For each contact, world-space contact normal; length soft_contact_max.
        shape_ke (wp.array(dtype=float)): Per-shape stiffness; length num_shapes.
        shape_kf (wp.array(dtype=float)): Per-shape friction stiffness; length num_shapes.
        shape_mu (wp.array(dtype=float)): Per-shape friction coefficient; length num_shapes.
        shape_body (wp.array(dtype=int)): Per-shape rigid body index; length num_shapes.
        particle_ke (float): Global particle contact stiffness.
        particle_kf (float): Global particle contact friction stiffness.
        particle_mu (float): Global particle contact friction coefficient.
        body_q (wp.array(dtype=wp.transform)): Body transforms (world); length num_bodies.
        body_qd (wp.array(dtype=wp.spatial_vector)): Body spatial velocities; length num_bodies.
        body_com (wp.array(dtype=wp.vec3)): Body center-of-mass in body frame; length num_bodies.
        dt (float): Time step.
        friction_epsilon (float): IPC friction smoothing parameter (unused in this kernel).
        friction_use_lagged_body_contact_force_norm (bool): Unused in this kernel; for API consistency.
        lagged_body_contact_force_norm (wp.array(dtype=float)): Per-sample normal force magnitude; length num_samples.
            For each contact, sample_index gets ke * |penetration_depth| if penetrating, else 0.
    """
    tid = wp.tid()

    count = contact_count[0]
    if tid >= count:
        return

    sample_index = contact_particle[tid]
    particle_index = contact_samples[sample_index]
    if (particle_flags[particle_index] & ParticleFlags.ACTIVE) == 0:
        return

    shape_index = contact_shape[tid]
    if shape_index < 0:
        return
    body_index = shape_body[shape_index]

    # Get material properties
    ke = 0.5 * (particle_ke + shape_ke[shape_index])

    # Get particle position and radius
    pos = particle_pos[sample_index]
    radius = particle_radius[particle_index]
    vol = particle_vol[particle_index]

    # Get body transform
    X_wb = wp.transform_identity()
    X_com = wp.vec3()
    if body_index >= 0:
        X_wb = body_q[body_index]
        X_com = body_com[body_index]

    # Body position in world space
    bx = wp.transform_point(X_wb, contact_body_pos[tid])

    # Contact normal
    n = contact_normal[tid]

    # Penetration depth
    penetration_depth = wp.dot(n, pos - bx) - radius

    if penetration_depth > 0:
        lagged_body_contact_force_norm[sample_index] = 0.0
    else:
        lagged_body_contact_force_norm[sample_index] = ke * \
            wp.abs(penetration_depth)


class SimplicitsParticleNewtonShapeSoftContact:
    r"""
    Handler for soft contact between Simplicits particles and Newton rigid body shapes,
    which are both in the same model.
    
    This class manages contact energy, gradient, and Hessian computations for particle-shape
    collisions in a coupled Newton-Simplicits simulation. It supports IPC-style friction,
    lagged friction formulation, and optional velocity-based restitution penalties.
    """
    def __init__(
        self,
        model: Model,
        integration_pt_volume: wp.array,
        dt: float,
        friction_use_lagged_body_contact_force_norm: bool = True,
        velocity_penalty_kv_scale: float = 0.0,
        friction_epsilon: float = 1e-2,
        coeff_of_restitution: float = 0.0,
    ):
        r"""
        Initialize the soft contact handler.

        Args:
            model (Model): Newton Model containing particles, shapes, bodies, and contact material params.
            integration_pt_volume (wp.array): Volume per integration (sample) point; dtype=float, length num_samples.
                Used to scale contact energy/gradient/Hessian.
            dt (float): Simulation time step (used in contact energy/gradient/Hessian kernels).
            friction_use_lagged_body_contact_force_norm (bool): If True, use lagged normal force magnitude
                for friction (recommended for stability); if False, use current penetration-based norm.
            velocity_penalty_kv_scale (float): Dimensionless scale factor for the velocity-level restitution
                penalty stiffness, expressed as a multiple of soft_contact_ke. The actual stiffness
                is ``velocity_penalty_kv_scale * soft_contact_ke``. Default 0.0 (disabled).
            friction_epsilon (float): IPC friction smoothing parameter.
            coeff_of_restitution (float): Coefficient of restitution for velocity penalty.

        Returns:
            None
        """
        self.integration_pt_volume = integration_pt_volume
        self.hessians_blocks = wp.zeros(
            integration_pt_volume.shape[0], dtype=wp.mat33, device=integration_pt_volume.device)

        self.contact_samples = wp.array(torch.arange(
            self.integration_pt_volume.shape[0]), dtype=int, device=self.integration_pt_volume.device)

        self.model: Model = model
        self.contacts: Contacts = None
        self.state: State = None

        self.dt = dt
        self.friction_epsilon = friction_epsilon
        self.coeff_of_restitution = coeff_of_restitution
        self.velocity_penalty_kv = velocity_penalty_kv_scale * float(model.soft_contact_ke)

        self.lagged_body_contact_force_norm = wp.zeros(
            self.integration_pt_volume.shape[0], dtype=float, device=self.integration_pt_volume.device)
        self.friction_use_lagged_body_contact_force_norm = friction_use_lagged_body_contact_force_norm

        self.buffer = wp.zeros(2)
        self.particle_pos = wp.zeros(
            self.model.simplicits_scene.sim_pts.shape[0], dtype=wp.vec3, device=self.model.simplicits_scene.sim_pts.device)

    def _set_state(self, state: State):
        r"""
        Set the simulation state for contact computation.

        Args:
            state (State): Newton state containing particle and body positions/velocities.
        """
        self.state = state

    def _set_contacts(self, contacts: Contacts):
        r"""
        Set the contacts object for this collision handler.

        Args:
            contacts (Contacts): Newton Contacts object containing contact pair information.
        """
        self.contacts = contacts
        # soft_contact_max is default to num of particles x num of bodies
        assert contacts.soft_contact_max % self.integration_pt_volume.shape[0] == 0

    def _copy_contacts(self, contacts: Contacts):
        r"""
        Copy contact data from another Contacts object to this handler's internal storage.

        Args:
            contacts (Contacts): Newton Contacts object to copy from.
        """
        for attr in contacts.__dict__:
            contacts_attr = getattr(contacts, attr)
            self_attr = getattr(self.contacts, attr)
            if isinstance(contacts_attr, wp.array):
                wp.copy(dest=self_attr, src=contacts_attr)
            else:
                setattr(self.contacts, attr, contacts_attr)
        # soft_contact_count = contacts.soft_contact_count.numpy()

    def update_lagged_body_contact_force_norm(self, dx, x0):
        r"""
        Update the lagged contact force norms for all particles in contact.

        This method computes the current contact force magnitudes and stores them
        for use in the lagged friction formulation. Should be called before solving
        when using lagged friction.

        Args:
            dx (wp.array): Particle displacement from rest positions; dtype=wp.vec3.
            x0 (wp.array): Rest particle positions; dtype=wp.vec3.
        """
        particle_pos = dx + x0
        wp.launch(
            kernel=_update_lagged_body_contact_force_norm_kernel,
            dim=self.integration_pt_volume.shape[0],
            inputs=[
                particle_pos,  # Use contact positions
                self.state.particle_q,
                self.integration_pt_volume,
                self.contact_samples,  # Pass contact samples array
                self.model.particle_radius,
                self.model.particle_flags,
                self.contacts.soft_contact_count,
                self.contacts.soft_contact_particle,
                self.contacts.soft_contact_shape,
                self.contacts.soft_contact_body_pos,
                self.contacts.soft_contact_body_vel,
                self.contacts.soft_contact_normal,
                self.model.shape_material_ke,
                self.model.shape_material_kf,
                self.model.shape_material_mu,
                self.model.shape_body,
                self.model.soft_contact_ke,
                self.model.soft_contact_kf,
                self.model.soft_contact_mu,
                self.state.body_q,
                self.state.body_qd,
                self.model.body_com,
                self.dt,
                self.friction_epsilon,
                self.friction_use_lagged_body_contact_force_norm,
            ],
            outputs=[self.lagged_body_contact_force_norm],
            device=x0.device,
        )

    def energy(self, dx, x0, coeff, energy=None):
        r"""
        Compute the newton-simplicits particle softcontact energy.

        Args:
            dx (wp.array): Current CP displacements with the current dofs; dtype=wp.vec3, size :math:`(\text{num_pts}, 3)`.
            x0 (wp.array): Rest contact point positions; dtype=wp.vec3, size :math:`(\text{num_pts}, 3)`.
            coeff (float): Coefficient for the contact energy.
            energy (wp.array): Output energy, used for cuda-graph capture; dtype=float, size :math:`1`.

        Returns:
            wp.array: Optional output energy; dtype=float, size :math:`1`.
        """
        if energy is None:
            energy = wp.zeros(1, dtype=float)

        if self.contacts is None or self.contacts.soft_contact_max == 0:
            return energy

        wp.copy(dest=self.buffer, src=energy, dest_offset=0, count=1)

        # particle_pos = dx + x0
        wp.copy(dest=self.particle_pos, src=dx)
        self.particle_pos += x0
        wp.launch(
            kernel=_contact_subspace_energy,
            dim=self.contacts.soft_contact_max,
            inputs=[
                self.particle_pos,  # Use contact positions
                self.state.particle_q,
                self.integration_pt_volume,
                self.contact_samples,  # Pass contact samples array
                self.model.particle_radius,
                self.model.particle_flags,
                self.contacts.soft_contact_count,
                self.contacts.soft_contact_particle,
                self.contacts.soft_contact_shape,
                self.contacts.soft_contact_body_pos,
                self.contacts.soft_contact_body_vel,
                self.contacts.soft_contact_normal,
                self.model.shape_material_ke,
                self.model.shape_material_kf,
                self.model.shape_material_mu,
                self.model.shape_body,
                self.model.soft_contact_ke,
                self.model.soft_contact_kf,
                self.model.soft_contact_mu,
                self.state.body_q,
                self.state.body_qd,
                self.model.body_com,
                self.dt,
                self.friction_epsilon,
                self.lagged_body_contact_force_norm,
                self.friction_use_lagged_body_contact_force_norm,
                self.coeff_of_restitution,
                self.velocity_penalty_kv,
            ],
            outputs=[energy],
            device=x0.device,
        )

        return energy*coeff

    def gradient(self, dx, x0, coeff, gradients):
        r"""
        Compute the newton-simplicits particle softcontact gradient.

        Args:
            dx (wp.array): Current CP displacements with the current dofs; dtype=wp.vec3, size :math:`(\text{num_pts}, 3)`.
            x0 (wp.array): Rest contact point positions; dtype=wp.vec3, size :math:`(\text{num_pts}, 3)`.
            coeff (float): Coefficient for the contact gradient.
            gradients (wp.array): Output gradient; dtype=wp.vec3, size :math:`(\text{num_pts}, 3)`.

        Returns:
            wp.array: Optional output gradient; dtype=wp.vec3, size :math:`(\text{num_pts}, 3)`.
        """
        if gradients is None:
            gradients = wp.zeros_like(dx)

        if self.contacts is None or self.contacts.soft_contact_max == 0:
            return gradients

        # particle_pos = dx + x0
        wp.copy(dest=self.particle_pos, src=dx)
        self.particle_pos += x0

        # Launch kernel to compute contact forces
        wp.launch(
            kernel=_contact_subspace_gradient,
            dim=self.contacts.soft_contact_max,
            inputs=[
                self.particle_pos,  # Use contact positions
                self.state.particle_q,
                self.integration_pt_volume,
                self.contact_samples,  # Pass contact samples array
                self.model.particle_radius,
                self.model.particle_flags,
                self.contacts.soft_contact_count,
                self.contacts.soft_contact_particle,
                self.contacts.soft_contact_shape,
                self.contacts.soft_contact_body_pos,
                self.contacts.soft_contact_body_vel,
                self.contacts.soft_contact_normal,
                self.model.shape_material_ke,
                self.model.shape_material_kf,
                self.model.shape_material_mu,
                self.model.shape_body,
                self.model.soft_contact_ke,
                self.model.soft_contact_kf,
                self.model.soft_contact_mu,
                self.state.body_q,
                self.state.body_qd,
                self.model.body_com,
                self.dt,
                self.friction_epsilon,
                self.lagged_body_contact_force_norm,
                self.friction_use_lagged_body_contact_force_norm,
                self.coeff_of_restitution,
                self.velocity_penalty_kv,
            ],
            outputs=[gradients],
            device=x0.device,
        )


        return gradients*coeff

    def hessian(self, dx, x0, coeff):
        r"""
        Compute the newton-simplicits particle softcontact Hessian.

        Evaluates the second derivative of contact energy w.r.t. particle positions and stores
        the result in self.hessians_blocks (3x3 blocks per sample). Used for implicit solvers.

        Args:
            dx (wp.array): Current integration-point displacements; dtype=wp.vec3, size :math:`(\text{num_pts}, 3)`.
            x0 (wp.array): Rest integration-point positions; dtype=wp.vec3, size :math:`(\text{num_pts}, 3)`.
            coeff (float): Scaling coefficient applied to the Hessian (e.g. time-step factor).

        Returns:
            wp.array: Hessian blocks; dtype=wp.mat33, size :math:`(\text{num_pts}, 3, 3)` (same
                as self.hessians_blocks), scaled by coeff.
        """

        self.hessians_blocks.zero_()
        if self.contacts is None or self.contacts.soft_contact_max == 0:
            return self.hessians_blocks


        # particle_pos = dx + x0
        wp.copy(dest=self.particle_pos, src=dx)
        self.particle_pos += x0

        wp.launch(
            kernel=_contact_subspace_hessian,
            dim=self.contacts.soft_contact_max,
            inputs=[
                self.particle_pos,  # Use contact positions
                self.state.particle_q,
                self.integration_pt_volume,
                self.contact_samples,  # Pass contact samples array
                self.model.particle_radius,
                self.model.particle_flags,
                self.contacts.soft_contact_count,
                self.contacts.soft_contact_particle,
                self.contacts.soft_contact_shape,
                self.contacts.soft_contact_body_pos,
                self.contacts.soft_contact_body_vel,
                self.contacts.soft_contact_normal,
                self.model.shape_material_ke,
                self.model.shape_material_kf,
                self.model.shape_material_mu,
                self.model.shape_body,
                self.model.soft_contact_ke,
                self.model.soft_contact_kf,
                self.model.soft_contact_mu,
                self.state.body_q,
                self.state.body_qd,
                self.model.body_com,
                self.dt,
                self.friction_epsilon,
                self.lagged_body_contact_force_norm,
                self.friction_use_lagged_body_contact_force_norm,
                self.coeff_of_restitution,
                self.velocity_penalty_kv,
            ],
            outputs=[self.hessians_blocks],
            device=x0.device,
        )


        return self.hessians_blocks*coeff

