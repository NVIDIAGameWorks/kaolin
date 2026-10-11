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

"""Tests for Newton collisions.

Note: Fixtures from conftest.py are automatically available!
"""

import torch
import warp as wp
from newton._src.sim.collide import CollisionPipeline
from kaolin.experimental.newton.builder import SimplicitsModelBuilder
from kaolin.experimental.newton.solver import SimplicitsSolver
import pytest


def test_object_contacting_floor(simplicits_object):
    r"""Test that object does not fall below the floor using the custom collision code.
    The purpose of this test is to ensure that the custom collision code is working correctly, not to test any Newton API code.
    Args:
        simplicits_object: The Simplicits object to test.
    """

    AXIS = 1
    FLOOR_HEIGHT = -1.0
    OBJECT_HEIGHT = 2.0
    builder = SimplicitsModelBuilder(up_axis="y")

    builder.add_simplicits_object(simplicits_object, num_qp=1024,
                    init_transform=torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0, OBJECT_HEIGHT],
                                                [0.0, 0.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda'))
    # Add ground plane
    builder.add_shape_plane(
        plane=(*builder.up_vector, FLOOR_HEIGHT),
        width=0.0,
        length=0.0,
        cfg=SimplicitsModelBuilder.ShapeConfig(
            ke=1e4, mu=0.5, kd=100.0, kf=1.0
        ),
        label="ground_plane",
    )

    model = builder.finalize()
    assert model is not None
    assert model.simplicits_scene.force_dict["pt_wise"]["newton_soft_collisions"] is not None
    state0 = model.state()
    state1 = model.state()
    assert state0 is not None
    assert state0.particle_q is not None
    assert state0.particle_q.shape[0] == 1024

    solver = SimplicitsSolver(model)

    dt = 0.05

    collision_pipeline = CollisionPipeline(model, soft_contact_margin=0.05)
    contacts = model.collide(state0, collision_pipeline=collision_pipeline)
    assert contacts.soft_contact_count.numpy()[0] == 0

    for i in range(20):
        contacts = model.collide(state0)
        solver.step(state0, state1, None, contacts, dt)
        state0, state1 = state1, state0


    assert state0.particle_q.numpy()[0, AXIS] > FLOOR_HEIGHT
    assert contacts.soft_contact_count.numpy()[0] > 0


def test_object_contacting_rigid_shape(simplicits_object):
    """Test that soft object contacts a rigid box shape and eventually the ground plane."""
    FLOOR_HEIGHT = 0
    OBJECT_HEIGHT = 2.0
    dt = 0.05
    builder = SimplicitsModelBuilder(up_axis="y")
    builder.add_simplicits_object(simplicits_object, num_qp=1024,
                    init_transform=torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0, OBJECT_HEIGHT],
                                                [0.0, 0.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda'))
    # Add ground plane
    builder.add_shape_plane(
        plane=(*builder.up_vector, FLOOR_HEIGHT),
        width=0.0,
        length=0.0,
        cfg=SimplicitsModelBuilder.ShapeConfig(
            ke=1e4, mu=0.5, kd=100.0, kf=1.0
        ),
        label="ground_plane",
    )

    # Add a rigid cube
    xform = wp.transform(wp.vec3(0.4, 0.5, 0.0), wp.quat_identity())
    body = builder.add_body(xform=xform)
    builder.add_shape_box(body=body, hx=0.5, hy=0.5, hz=0.5)

    model = builder.finalize()
    assert model is not None
    assert model.simplicits_scene.force_dict["pt_wise"]["newton_soft_collisions"] is not None
    assert model.state() is not None
    state0 = model.state()
    state1 = model.state()
    assert state0.particle_q is not None
    assert state0.particle_q.shape[0] == 1024
    assert len(state0.body_q.numpy().flatten()) == 7


    solver = SimplicitsSolver(model)
    solver.step(state0, state1, None, None, dt)
    state0, state1 = state1, state0

    sim_particle_radius = model.particle_radius.numpy()[0]
    contacts = model.collide(state0)
    assert contacts is not None

    assert contacts.soft_contact_count.numpy()[0] == 0

    for i in range(10):
        contacts = model.collide(state0)
        solver.step(state0, state1, None, contacts, dt)
        state0, state1 = state1, state0

    assert contacts.soft_contact_count.numpy()[0] > 0
    # Assert that no contact involves the ground plane (shape index 0)
    assert not (contacts.soft_contact_shape.numpy()[:contacts.soft_contact_count.numpy()[0]] == 0).any(), \
        "Found contact(s) with ground plane (shape index 0) in contacts."

    for i in range(20):
        contacts = model.collide(state0)
        solver.step(state0, state1, None, contacts, dt)
        state0, state1 = state1, state0

    assert contacts.soft_contact_count.numpy()[0] > 0
    # Assert that no contact involves the ground plane (shape index 0)
    assert  (contacts.soft_contact_shape.numpy()[:contacts.soft_contact_count.numpy()[0]] == 0).any(), \
        "Found NO contact(s) with ground plane in the last 10 steps."


def test_contact_energy(simplicits_object):
    """Test that contact energy is positive when contacts exist."""
    FLOOR_HEIGHT = 0
    OBJECT_HEIGHT = 1.5
    builder = SimplicitsModelBuilder(up_axis="y")
    builder.add_simplicits_object(simplicits_object, num_qp=100,
                    init_transform=torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0, OBJECT_HEIGHT],
                                                [0.0, 0.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda'))
    # Add ground plane
    builder.add_shape_plane(
        plane=(*builder.up_vector, FLOOR_HEIGHT),
        width=0.0,
        length=0.0,
        cfg=SimplicitsModelBuilder.ShapeConfig(
            ke=1e4, mu=0.5, kd=100.0, kf=1.0
        ),
        label="ground_plane",
    )

    # Add a rigid cube of size 0.5x0.5x0.5
    xform = wp.transform(wp.vec3(0.4, 0.5, 0.0), wp.quat_identity())
    body = builder.add_body(xform=xform)
    builder.add_shape_box(body=body, hx=0.5, hy=0.5, hz=0.5)

    model = builder.finalize()
    state0 = model.state()

    contacts = model.collide(state0)
    assert contacts is not None
    assert contacts.soft_contact_count.numpy()[0] > 0

    # Get the collision handler
    collision_handler = model.simplicits_scene.force_dict["pt_wise"]["newton_soft_collisions"]["object"]
    assert collision_handler is not None

    # Set state and contacts on the collision handler
    collision_handler._set_state(state0)
    collision_handler._set_contacts(contacts)

    # Get particle positions from the state
    # dx is the displacement from rest, x0 is the rest position
    # For this test, we'll use current positions as x0 and zero displacement
    x0 = state0.particle_q
    dx = wp.zeros_like(x0)


    # Test energy computation with zero displacement
    energy = collision_handler.energy(dx=dx, x0=x0, coeff=1.0)
    assert energy is not None
    energy_value = energy.numpy()[0]
    assert energy_value > 0.0, "Contact energy should be positive when contacts exist"

def test_contact_gradient(simplicits_object):
    r"""Test the contact gradient against finite differences of the contact energy.

    Four quadrature points are placed well inside a static plane, making the reference
    energy independent of contact activation. Evaluating at rest leaves velocity friction
    inactive, and mu=0 disables it for the finite-difference perturbations. The velocity
    penalty is covered by test_contact_vel_gradient, which evaluates it away from the
    kink at neg_vn = 0.
    """
    FLOOR_HEIGHT = 0
    OBJECT_HEIGHT = -2.0
    builder = SimplicitsModelBuilder(up_axis="y")
    builder.add_simplicits_object(
        simplicits_object,
        num_qp=4,
        init_transform=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, OBJECT_HEIGHT],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda'))
    builder.add_shape_plane(
        plane=(*builder.up_vector, FLOOR_HEIGHT),
        width=0.0,
        length=0.0,
        cfg=SimplicitsModelBuilder.ShapeConfig(
            ke=1e4, mu=0.0, kd=100.0, kf=1.0
        ),
        label="ground_plane",
    )

    model = builder.finalize()
    state0 = model.state()

    contacts = model.collide(state0)
    assert contacts is not None
    assert contacts.soft_contact_count.numpy()[0] > 0

    # Get the collision handler
    collision_handler = model.simplicits_scene.force_dict["pt_wise"]["newton_soft_collisions"]["object"]
    collision_handler.velocity_penalty_kv = 0.0
    assert collision_handler is not None

    # Use state0 as both prev and current: relative_translation=0 → velocity term inactive
    collision_handler._set_state(state0)
    collision_handler._set_contacts(contacts)

    # Get particle positions from the state
    # dx is the displacement from rest, x0 is the rest position
    # For this test, we'll use current positions as x0 and zero displacement
    x0 = state0.particle_q
    dx = wp.zeros_like(x0)
    coeff = model.simplicits_scene.force_dict["pt_wise"]["newton_soft_collisions"]["coeff"]
    wp_gradient = collision_handler.gradient(dx, x0, coeff=coeff, gradients=None)
    gradient = wp.to_torch(wp_gradient) if wp_gradient.shape[0] > 0 else torch.zeros(0, 3, device=wp.device_to_torch(wp_gradient.device))


    t_x0 = wp.to_torch(x0)
    t_dx = wp.to_torch(dx)

    # Finite difference for collision gradients
    dEdx_fd = torch.zeros_like(t_x0)
    eps = 1e-4
    for i in range(dEdx_fd.shape[0]):
        for j in range(dEdx_fd.shape[1]):
            t_dx[i,j] += eps
            E1 = wp.to_torch(collision_handler.energy(wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=coeff))
            t_dx[i, j] -= 2.0*eps
            E2 = wp.to_torch(collision_handler.energy(wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=coeff))
            t_dx[i, j] += eps
            dEdx_fd[i, j] = (E1 - E2) / (2.0*eps)

    assert torch.allclose(gradient, dEdx_fd, atol=1e-2,rtol=1e-1), \
        "Collision energy doesn't match analytical calculation"


@pytest.mark.parametrize("velocity_penalty_scale", [0.1, 0.2])
def test_contact_vel_gradient(simplicits_object, velocity_penalty_scale):
    r"""Test the velocity-penalty term of the contact gradient in isolation.

    Isolates the vel term by zeroing ke/mu: no collision energy, and no friction, whose
    non-lagged normal-force magnitude ke*|penetration| is differentiated by energy() but
    held fixed by gradient(). Advances one step so neg_vn > 0, putting the penalty in the
    smooth region away from the kink at neg_vn = 0, where central differences of the
    one-sided quadratic return 0.25 * kv * (1 + e)^2 * eps / dt^2 instead of 0.
    """
    FLOOR_HEIGHT = 0
    OBJECT_HEIGHT = 0.0
    dt = 0.05
    builder = SimplicitsModelBuilder(up_axis="y")
    builder.add_simplicits_object(simplicits_object, num_qp=50,
                                  init_transform=torch.tensor(
                                      [[1.0, 0.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0, OBJECT_HEIGHT],
                                       [0.0, 0.0, 1.0, 0.0],
                                       [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda'))
    builder.add_shape_plane(
        plane=(*builder.up_vector, FLOOR_HEIGHT),
        width=0.0,
        length=0.0,
        cfg=SimplicitsModelBuilder.ShapeConfig(ke=0.0, mu=0.0, kd=100.0, kf=1.0),
        label="ground_plane",
    )

    model = builder.finalize()
    # Zero out model-level ke/mu so collision+friction energy is exactly 0
    model.soft_contact_ke = 0.0
    model.soft_contact_mu = 0.0

    state0 = model.state()
    state1 = model.state()

    contacts_init = model.collide(state0)
    assert contacts_init is not None
    assert contacts_init.soft_contact_count.numpy()[0] > 0, \
        "Expected initial contacts; OBJECT_HEIGHT=0.0 should place QPs below floor"

    # Advance one step so particles have downward velocity (neg_vn > 0),
    # moving the vel penalty away from the kink at neg_vn = 0.
    solver = SimplicitsSolver(model)
    solver.step(state0, state1, None, contacts_init, dt)

    contacts = model.collide(state1)
    assert contacts is not None
    assert contacts.soft_contact_count.numpy()[0] > 0

    collision_handler = model.simplicits_scene.force_dict["pt_wise"]["newton_soft_collisions"]["object"]
    # Use a fixed reference scale (1e3) since model.soft_contact_ke is 0
    collision_handler.velocity_penalty_kv = velocity_penalty_scale * 1e3
    assert collision_handler is not None

    # state0 = prev_pos so relative_translation != 0 → neg_vn > 0
    collision_handler._set_state(state0)
    collision_handler._set_contacts(contacts)

    x0 = state1.particle_q
    dx = wp.zeros_like(x0)
    coeff = model.simplicits_scene.force_dict["pt_wise"]["newton_soft_collisions"]["coeff"]

    gradient = wp.to_torch(collision_handler.gradient(dx, x0, coeff=coeff, gradients=None)).clone()
    assert gradient.abs().max() > 0.0, "Expected a non-zero velocity-penalty gradient"

    t_dx = wp.to_torch(dx)

    dEdx_fd = torch.zeros_like(gradient)
    eps = 1e-4
    for i in range(dEdx_fd.shape[0]):
        for j in range(dEdx_fd.shape[1]):
            t_dx[i, j] += eps
            E1 = wp.to_torch(collision_handler.energy(wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=coeff))
            t_dx[i, j] -= 2.0 * eps
            E2 = wp.to_torch(collision_handler.energy(wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=coeff))
            t_dx[i, j] += eps
            dEdx_fd[i, j] = (E1 - E2) / (2.0 * eps)

    assert torch.allclose(gradient, dEdx_fd, atol=1e-2, rtol=1e-1), \
        "Velocity-penalty gradient doesn't match finite difference of energy"


@pytest.mark.parametrize("velocity_penalty_scale", [0.1, 0.2])
def test_contact_vel_hessian(simplicits_object, velocity_penalty_scale):
    r"""Test the velocity-penalty term of the contact hessian in isolation.

    Isolates the vel term by zeroing ke/mu (no collision/friction energy).
    Advances one step so neg_vn > 0, putting the penalty in the smooth region
    away from the kink at neg_vn = 0.
    """
    FLOOR_HEIGHT = 0
    OBJECT_HEIGHT = 0.0
    dt = 0.05
    builder = SimplicitsModelBuilder(up_axis="y")
    builder.add_simplicits_object(simplicits_object, num_qp=50,
                    init_transform=torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0, OBJECT_HEIGHT],
                                                [0.0, 0.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda'))
    builder.add_shape_plane(
        plane=(*builder.up_vector, FLOOR_HEIGHT),
        width=0.0,
        length=0.0,
        cfg=SimplicitsModelBuilder.ShapeConfig(ke=0.0, mu=0.0, kd=100.0, kf=1.0),
        label="ground_plane",
    )

    model = builder.finalize()
    # Zero out model-level ke/mu so collision+friction energy is exactly 0
    model.soft_contact_ke = 0.0
    model.soft_contact_mu = 0.0

    state0 = model.state()
    state1 = model.state()

    contacts_init = model.collide(state0)
    assert contacts_init is not None
    assert contacts_init.soft_contact_count.numpy()[0] > 0, \
        "Expected initial contacts; OBJECT_HEIGHT=0.0 should place QPs below floor"

    # Advance one step so particles have downward velocity (neg_vn > 0),
    # moving the vel penalty away from the kink at neg_vn = 0.
    solver = SimplicitsSolver(model)
    solver.step(state0, state1, None, contacts_init, dt)

    contacts = model.collide(state1)
    assert contacts is not None
    assert contacts.soft_contact_count.numpy()[0] > 0

    collision_handler = model.simplicits_scene.force_dict["pt_wise"]["newton_soft_collisions"]["object"]
    # Use a fixed reference scale (1e3) since model.soft_contact_ke is 0
    collision_handler.velocity_penalty_kv = velocity_penalty_scale * 1e3
    assert collision_handler is not None

    # state0 = prev_pos so relative_translation != 0 → neg_vn > 0
    collision_handler._set_state(state0)
    collision_handler._set_contacts(contacts)

    x0 = state1.particle_q
    dx = wp.zeros_like(x0)
    coeff = model.simplicits_scene.force_dict["pt_wise"]["newton_soft_collisions"]["coeff"]

    wp_hessian_blocks = collision_handler.hessian(dx, x0, coeff=coeff)
    hessian_blocks = wp.to_torch(wp_hessian_blocks) if wp_hessian_blocks.shape[0] > 0 else torch.zeros(
        0, 3, 3, device=wp.device_to_torch(wp_hessian_blocks.device))
    hessian = torch.block_diag(*hessian_blocks).to(hessian_blocks.device)

    t_x0 = wp.to_torch(x0)
    t_dx = wp.to_torch(dx)

    hessian_fd = torch.zeros((t_x0.shape[0] * t_x0.shape[1], t_x0.shape[0] * t_x0.shape[1]),
                             device=t_x0.device, dtype=t_x0.dtype)
    eps = 1e-5
    for i in range(hessian_blocks.shape[0]):
        for k in range(3):
            t_dx[i, k] += eps
            G1 = wp.to_torch(collision_handler.gradient(
                wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=coeff, gradients=None)).flatten()
            t_dx[i, k] -= 2.0 * eps
            G2 = wp.to_torch(collision_handler.gradient(
                wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=coeff, gradients=None)).flatten()
            t_dx[i, k] += eps
            hessian_fd[3 * i + k] = (G1 - G2) / (2.0 * eps)

    zero_threshold = 1e-2
    analytical_nonzeros = (hessian.abs() > zero_threshold)
    fd_nonzeros = (hessian_fd.abs() > zero_threshold)

    sparsity_match = torch.all(analytical_nonzeros == fd_nonzeros)
    assert sparsity_match, \
        f"Vel hessian sparsity mismatch: analytical {analytical_nonzeros.sum().item()} non-zeros, FD {fd_nonzeros.sum().item()}"
    assert torch.allclose(hessian[fd_nonzeros], hessian_fd[fd_nonzeros], atol=1e-1, rtol=1e-1), \
        "Vel hessian values don't match finite difference approximation"


def test_contact_collision_hessian(simplicits_object):
    r"""Test the collision (penetration) term of the contact hessian in isolation.

    Isolates the collision term by setting mu=0 (no friction) and vel_kv=0 (no vel penalty).
    Uses state0 for both prev and current so relative_translation=0 → neg_vn=0 → vel term off.
    OBJECT_HEIGHT=0.0 places QPs below the floor so penetration_depth > 0 at t=0 (away from kink).
    """
    FLOOR_HEIGHT = 0
    OBJECT_HEIGHT = 0.0
    builder = SimplicitsModelBuilder(up_axis="y")
    builder.add_simplicits_object(simplicits_object, num_qp=50,
                    init_transform=torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0, OBJECT_HEIGHT],
                                                [0.0, 0.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda'))
    builder.add_shape_plane(
        plane=(*builder.up_vector, FLOOR_HEIGHT),
        width=0.0,
        length=0.0,
        cfg=SimplicitsModelBuilder.ShapeConfig(ke=1e4, mu=0.0, kd=100.0, kf=1.0),
        label="ground_plane",
    )

    model = builder.finalize()
    # Zero out friction at model level
    model.soft_contact_mu = 0.0

    state0 = model.state()

    contacts = model.collide(state0)
    assert contacts is not None
    assert contacts.soft_contact_count.numpy()[0] > 0, \
        "Expected initial contacts; OBJECT_HEIGHT=0.0 should place QPs below floor"

    collision_handler = model.simplicits_scene.force_dict["pt_wise"]["newton_soft_collisions"]["object"]
    collision_handler.velocity_penalty_kv = 0.0
    assert collision_handler is not None

    # Use state0 as both prev and current: relative_translation=0 → vel term inactive
    collision_handler._set_state(state0)
    collision_handler._set_contacts(contacts)

    x0 = state0.particle_q
    dx = wp.zeros_like(x0)
    coeff = model.simplicits_scene.force_dict["pt_wise"]["newton_soft_collisions"]["coeff"]

    wp_hessian_blocks = collision_handler.hessian(dx, x0, coeff=coeff)
    hessian_blocks = wp.to_torch(wp_hessian_blocks) if wp_hessian_blocks.shape[0] > 0 else torch.zeros(
        0, 3, 3, device=wp.device_to_torch(wp_hessian_blocks.device))
    hessian = torch.block_diag(*hessian_blocks).to(hessian_blocks.device)

    t_x0 = wp.to_torch(x0)
    t_dx = wp.to_torch(dx)

    hessian_fd = torch.zeros((t_x0.shape[0] * t_x0.shape[1], t_x0.shape[0] * t_x0.shape[1]),
                             device=t_x0.device, dtype=t_x0.dtype)
    eps = 1e-5
    for i in range(hessian_blocks.shape[0]):
        for k in range(3):
            t_dx[i, k] += eps
            G1 = wp.to_torch(collision_handler.gradient(
                wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=coeff, gradients=None)).flatten()
            t_dx[i, k] -= 2.0 * eps
            G2 = wp.to_torch(collision_handler.gradient(
                wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=coeff, gradients=None)).flatten()
            t_dx[i, k] += eps
            hessian_fd[3 * i + k] = (G1 - G2) / (2.0 * eps)

    zero_threshold = 1e-2
    analytical_nonzeros = (hessian.abs() > zero_threshold)
    fd_nonzeros = (hessian_fd.abs() > zero_threshold)


    sparsity_match = torch.all(analytical_nonzeros == fd_nonzeros)
    assert sparsity_match, \
        f"Collision hessian sparsity mismatch: analytical {analytical_nonzeros.sum().item()} non-zeros, FD {fd_nonzeros.sum().item()}"
    assert torch.allclose(hessian[fd_nonzeros], hessian_fd[fd_nonzeros], atol=1e-1, rtol=1e-1), \
        "Collision hessian values don't match finite difference approximation"


def test_contact_friction_hessian(simplicits_object):
    r"""Test the friction term of the contact hessian in isolation.

    Isolates friction by enabling ke>0 and mu>0 while setting vel_kv=0.
    Uses state0 for both prev and current (rel_translation=0 → vel term off).
    Calls update_lagged_body_contact_force_norm before FD so the lagged normal-force
    norm is populated and constant during FD sweeps — matching the analytical hessian
    assumption. With eps=1e-5 and eps_u=5e-4 the IPC smoothing yields ~1% friction
    hessian error, well within rtol=10%.

    The comparison is resolution-aware: differencing the gradient in float32 cannot
    resolve curvature below ULP(G_c) / (2 * eps) for the column it differences, and the
    tangent/normal cross terms sit under that floor because their column carries the
    large normal force. Those entries come back as exactly zero, so the floor is folded
    into the per-entry tolerance instead of asserting identical sparsity.
    """
    FLOOR_HEIGHT = 0
    OBJECT_HEIGHT = 0.0
    builder = SimplicitsModelBuilder(up_axis="y")
    builder.add_simplicits_object(simplicits_object, num_qp=50,
                    init_transform=torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0, OBJECT_HEIGHT],
                                                [0.0, 0.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda'))
    builder.add_shape_plane(
        plane=(*builder.up_vector, FLOOR_HEIGHT),
        width=0.0,
        length=0.0,
        cfg=SimplicitsModelBuilder.ShapeConfig(ke=1e4, mu=0.5, kd=100.0, kf=1.0),
        label="ground_plane",
    )

    model = builder.finalize()
    state0 = model.state()

    contacts = model.collide(state0)
    assert contacts is not None
    assert contacts.soft_contact_count.numpy()[0] > 0, \
        "Expected initial contacts; OBJECT_HEIGHT=0.0 should place QPs below floor"

    collision_handler = model.simplicits_scene.force_dict["pt_wise"]["newton_soft_collisions"]["object"]
    collision_handler.velocity_penalty_kv = 0.0
    assert collision_handler is not None

    # Use state0 as both prev and current: relative_translation=0 → vel term inactive
    collision_handler._set_state(state0)
    collision_handler._set_contacts(contacts)

    x0 = state0.particle_q
    dx = wp.zeros_like(x0)
    coeff = model.simplicits_scene.force_dict["pt_wise"]["newton_soft_collisions"]["coeff"]

    # Populate lagged body contact force norm before FD so it stays constant during
    # the FD sweep (matching the analytical hessian's fixed-norm assumption).
    collision_handler.update_lagged_body_contact_force_norm(dx, x0)

    wp_hessian_blocks = collision_handler.hessian(dx, x0, coeff=coeff)
    hessian_blocks = wp.to_torch(wp_hessian_blocks) if wp_hessian_blocks.shape[0] > 0 else torch.zeros(
        0, 3, 3, device=wp.device_to_torch(wp_hessian_blocks.device))
    hessian = torch.block_diag(*hessian_blocks).to(hessian_blocks.device)

    # Unperturbed gradient, used below to size the FD resolution floor per column.
    gradient = wp.to_torch(collision_handler.gradient(dx, x0, coeff=coeff, gradients=None)).flatten().clone()

    t_x0 = wp.to_torch(x0)
    t_dx = wp.to_torch(dx)

    hessian_fd = torch.zeros((t_x0.shape[0] * t_x0.shape[1], t_x0.shape[0] * t_x0.shape[1]),
                             device=t_x0.device, dtype=t_x0.dtype)
    eps = 1e-5
    for i in range(hessian_blocks.shape[0]):
        for k in range(3):
            t_dx[i, k] += eps
            G1 = wp.to_torch(collision_handler.gradient(
                wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=coeff, gradients=None)).flatten()
            t_dx[i, k] -= 2.0 * eps
            G2 = wp.to_torch(collision_handler.gradient(
                wp.from_torch(t_dx, dtype=wp.vec3), x0, coeff=coeff, gradients=None)).flatten()
            t_dx[i, k] += eps
            hessian_fd[3 * i + k] = (G1 - G2) / (2.0 * eps)

    zero_threshold = 1e-2

    # The tangential diagonal entries come solely from the friction term; requiring them
    # on both sides keeps the comparison below from passing vacuously. Up axis is y, so
    # the tangential components are x and z.
    tangential = torch.zeros(hessian.shape[0], dtype=torch.bool, device=hessian.device)
    tangential[0::3] = True
    tangential[2::3] = True
    assert (torch.diagonal(hessian)[tangential].abs() > zero_threshold).any(), \
        "No friction contribution in the analytical hessian"
    assert (torch.diagonal(hessian_fd)[tangential].abs() > zero_threshold).any(), \
        "Finite differences resolved no friction contribution"

    # Entry (r, c) differences gradient component c, so float32 cancellation limits it to
    # curvature above ULP(G_c) / (2 * eps); below that the quotient rounds to zero.
    fd_floor = (torch.finfo(torch.float32).eps * gradient.abs() / (2.0 * eps)).unsqueeze(0).expand_as(hessian_fd)
    tol = 1e-1 + 1e-1 * hessian.abs() + 4.0 * fd_floor
    excess = ((hessian - hessian_fd).abs() - tol).max().item()
    assert excess <= 0.0, \
        f"Friction hessian differs from finite differences beyond FD resolution (worst excess {excess:.4g})"


def test_contact_energy_with_velocity_penalty(simplicits_object):
    r"""Test that velocity penalty increases contact energy for approaching particles.

    Sets up a falling object above a floor+box, advances one time step so that particles
    have non-zero downward velocity (approaching the contact surface), then verifies that
    the contact energy with velocity_penalty_kv > 0 is strictly greater than with kv = 0.
    """
    FLOOR_HEIGHT = 0
    OBJECT_HEIGHT = 1.5
    dt = 0.05
    builder = SimplicitsModelBuilder(up_axis="y")
    builder.add_simplicits_object(simplicits_object, num_qp=100,
                    init_transform=torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                                [0.0, 1.0, 0.0, OBJECT_HEIGHT],
                                                [0.0, 0.0, 1.0, 0.0],
                                                [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda'))
    builder.add_shape_plane(
        plane=(*builder.up_vector, FLOOR_HEIGHT),
        width=0.0,
        length=0.0,
        cfg=SimplicitsModelBuilder.ShapeConfig(ke=1e4, mu=0.5, kd=100.0, kf=1.0),
        label="ground_plane",
    )
    xform = wp.transform(wp.vec3(0.4, 0.5, 0.0), wp.quat_identity())
    body = builder.add_body(xform=xform)
    builder.add_shape_box(body=body, hx=0.5, hy=0.5, hz=0.5)

    model = builder.finalize()
    state0 = model.state()
    state1 = model.state()

    solver = SimplicitsSolver(model)
    # Run one step: gravity pulls particles downward (toward the box/floor).
    contacts0 = model.collide(state0)
    solver.step(state0, state1, None, contacts0, dt)
    # state0 = positions at t=0 (previous), state1 = positions at t=dt (current)

    # Verify contacts still exist after the step
    contacts1 = model.collide(state1)
    assert contacts1.soft_contact_count.numpy()[0] > 0, \
        "Expected contacts after one step; object should still be near the box"

    collision_handler = model.simplicits_scene.force_dict["pt_wise"]["newton_soft_collisions"]["object"]

    # Use state0 as prev_pos so that relative_translation = (state1.particle_q - state0.particle_q),
    # capturing the downward motion of falling particles.
    collision_handler._set_state(state0)
    collision_handler._set_contacts(contacts1)

    x0 = state1.particle_q  # current positions (after one step)
    dx = wp.zeros_like(x0)

    # Energy without velocity penalty
    collision_handler.velocity_penalty_kv = 0.0
    energy_no_kv = collision_handler.energy(dx=dx, x0=x0, coeff=1.0).numpy()[0]

    # Energy with velocity penalty scaled to 10% of soft_contact_ke
    collision_handler.velocity_penalty_kv = 0.1 * float(model.soft_contact_ke)
    energy_with_kv = collision_handler.energy(dx=dx, x0=x0, coeff=1.0).numpy()[0]

    assert energy_with_kv > energy_no_kv, (
        f"Energy with velocity penalty ({energy_with_kv:.6f}) should exceed "
        f"energy without ({energy_no_kv:.6f}); "
        "check that particles are approaching the contact surface"
    )


def test_coeff_reaches_the_output_buffers(simplicits_object):
    """coeff must scale what lands in the caller's buffer, not just the return value.

    This is the one thing no other test here can see. Every other coeff in this file is
    literally 1.0, and all four assembler call sites in SimplicitsScene discard the
    return and read the buffer they passed in. So the previous shape -- kernels with no
    coeff parameter and `return buffer * coeff` at the end -- looked correct from the
    return value while contributing an unscaled term to the scene, and reverting to it
    would pass the rest of this file unchanged.

    Scaling is checked between two coeffs rather than against an absolute value, so the
    test says nothing about what the contact model should compute.
    """
    FLOOR_HEIGHT, OBJECT_HEIGHT = 0, 1.5
    builder = SimplicitsModelBuilder(up_axis="y")
    builder.add_simplicits_object(
        simplicits_object, num_qp=100,
        init_transform=torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                     [0.0, 1.0, 0.0, OBJECT_HEIGHT],
                                     [0.0, 0.0, 1.0, 0.0],
                                     [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda'))
    builder.add_shape_plane(
        plane=(*builder.up_vector, FLOOR_HEIGHT), width=0.0, length=0.0,
        cfg=SimplicitsModelBuilder.ShapeConfig(ke=1e4, mu=0.5, kd=100.0, kf=1.0),
        label="ground_plane")
    xform = wp.transform(wp.vec3(0.4, 0.5, 0.0), wp.quat_identity())
    builder.add_shape_box(body=builder.add_body(xform=xform), hx=0.5, hy=0.5, hz=0.5)

    model = builder.finalize()
    state0 = model.state()
    contacts = model.collide(state0)
    assert contacts.soft_contact_count.numpy()[0] > 0, "test is vacuous without contacts"

    handler = model.simplicits_scene.force_dict["pt_wise"]["newton_soft_collisions"]["object"]
    handler._set_state(state0)
    handler._set_contacts(contacts)

    x0 = state0.particle_q
    dx = wp.zeros_like(x0)
    n = x0.shape[0]
    small, large = 0.25, 1.0
    ratio = large / small

    def energy_at(coeff):
        out = wp.zeros(1, dtype=wp.float32, device=x0.device)
        handler.energy(dx=dx, x0=x0, coeff=coeff, energy=out)
        return float(out.numpy()[0])

    def gradient_at(coeff):
        out = wp.zeros(n, dtype=wp.vec3, device=x0.device)
        handler.gradient(dx, x0, coeff, out)
        return wp.to_torch(out).clone()

    def hessian_at(coeff):
        return wp.to_torch(handler.hessian(dx, x0, coeff)).clone()

    e_small, e_large = energy_at(small), energy_at(large)
    assert e_small != 0.0, "test is vacuous if the contact energy is zero"
    assert e_large == pytest.approx(ratio * e_small, rel=1e-4), (
        "coeff did not reach the energy buffer the assembler reads")

    g_small, g_large = gradient_at(small), gradient_at(large)
    assert g_small.abs().max() > 0.0, "test is vacuous if the contact gradient is zero"
    assert torch.allclose(g_large, ratio * g_small, rtol=1e-4), (
        "coeff did not reach the gradient buffer the assembler reads")

    h_small, h_large = hessian_at(small), hessian_at(large)
    assert h_small.abs().max() > 0.0, "test is vacuous if the contact hessian is zero"
    assert torch.allclose(h_large, ratio * h_small, rtol=1e-4), (
        "coeff did not scale the hessian")
