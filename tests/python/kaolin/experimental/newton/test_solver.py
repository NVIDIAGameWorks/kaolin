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

"""Tests for SimplicitsSolver.

Note: Fixtures from conftest.py are automatically available!
"""

import torch
import warp as wp
import kaolin
from kaolin.experimental.newton.solver import SimplicitsSolver
from kaolin.experimental.newton.builder import SimplicitsModelBuilder
from pathlib import Path

UP_AXIS = "y"
AXIS = 1 if UP_AXIS == "y" else 2

def run_regression_test(solver, model, tol=1e-2, test_name="fem_test"):
    p = Path(__file__).resolve().parent.parent.parent / "physics" / "simplicits" / "regression_test_data" / "wpfem_vertex_deformations_beam.pth"
    fem_data = torch.load(p, weights_only=False)

    faces = fem_data["mesh_faces"]  # beam faces
    start_verts = fem_data["v0"]  # beam start verts
    frame_1_verts = fem_data["v1"]  # beam at frame 1
    frame_100_verts = fem_data["v_end"]  # beam verts at frame 100
    dt = 0.05

    # Checking deformation at start
    our_start_verts = model.simplicits_scene.get_object_deformed_pts(0, start_verts)  # find OUR starting deformation on the fem beam's verts
    cd = kaolin.metrics.pointcloud.chamfer_distance(start_verts.unsqueeze(0),our_start_verts.unsqueeze(0), w1=1.0, w2=1.0, squared=True)
    assert cd[0].item() < tol*tol, f"Chamfer distance at start is {cd[0].item()}. This is too high. This is a very basic test, something is terribly wrong in {test_name}."

    # Checking deformation at frame 1
    state_0 = model.state()
    state_1 = model.state()
    solver.step(state_0, state_1, None, None, dt)
    state_1, state_0 = state_0, state_1

    our_frame_1_verts = model.simplicits_scene.get_object_deformed_pts(0, start_verts)

    cd = kaolin.metrics.pointcloud.chamfer_distance(frame_1_verts.unsqueeze(0), our_frame_1_verts.unsqueeze(0), w1=1.0, w2=1.0, squared=True)
    assert cd[0].item() < tol*tol, f"Chamfer distance at frame 1 is {cd[0].item()}. This is too high. This is a basic test, something is terribly wrong in {test_name}."

    # Checking deformation at frame 100
    for i in range(99):
        state_0.clear_forces()
        state_1.clear_forces()
        solver.step(state_0, state_1, None, None, dt)
        state_1, state_0 = state_0, state_1

    our_frame_100_verts = model.simplicits_scene.get_object_deformed_pts(0, start_verts)


    cd = kaolin.metrics.pointcloud.chamfer_distance(frame_100_verts.unsqueeze(0),
                                                    our_frame_100_verts.unsqueeze(0),
                                                    w1=1.0, w2=1.0, squared=True)

    assert cd[0].item() < tol, f"Chamfer distance at frame 100 is {cd[0].item()}. This is too high. Something is likely wrong in {test_name}."

def test_solver_initialization(simplicits_object):
    """Test that solver can be created."""
    builder = SimplicitsModelBuilder(up_axis=UP_AXIS)
    builder.add_simplicits_object(simplicits_object, num_qp=1024)

    model = builder.finalize()
    assert model is not None
    assert model.state() is not None
    state = model.state()
    assert state.sim_z is not None and state.sim_z.shape[0] == simplicits_object.num_handles*12
    assert state.sim_z_prev is not None and state.sim_z_prev.shape[0] == simplicits_object.num_handles*12
    assert state.sim_z_dot is not None and state.sim_z_dot.shape[0] == simplicits_object.num_handles*12
    solver = SimplicitsSolver(model)
    assert solver is not None

def test_solver_regression_cantilever_beam(cantilever_beam_object):
    """Test that solver can be created."""
    simplicits_object = cantilever_beam_object

    builder = SimplicitsModelBuilder(up_axis=UP_AXIS)
    builder.add_simplicits_object(simplicits_object, num_qp=1024)
    builder.add_simplicits_object_boundary_condition(0, "right", lambda x: x[:, 0] >= 0.98, bdry_penalty=10000.0)
    model = builder.finalize()
    model.simplicits_scene.max_newton_steps = 10
    model.simplicits_scene.max_ls_steps = 20
    model.simplicits_scene.newton_hessian_regularizer = 1e-3
    model.simplicits_scene.direct_solve = True
    assert model is not None
    assert model.state() is not None
    state = model.state()
    assert state.sim_z is not None and state.sim_z.shape[0] == simplicits_object.num_handles*12
    assert state.sim_z_prev is not None and state.sim_z_prev.shape[0] == simplicits_object.num_handles*12
    assert state.sim_z_dot is not None and state.sim_z_dot.shape[0] == simplicits_object.num_handles*12

    solver = SimplicitsSolver(model)
    assert solver is not None
    assert state.sim_z is not None and state.sim_z.shape[0] == simplicits_object.num_handles*12
    assert state.sim_z_prev is not None and state.sim_z_prev.shape[0] == simplicits_object.num_handles*12
    assert state.sim_z_dot is not None and state.sim_z_dot.shape[0] == simplicits_object.num_handles*12

    run_regression_test(solver, model, tol=0.02, test_name="solver_step_test")

def test_solver_step(simplicits_object):
    """Test that solver step works."""
    builder = SimplicitsModelBuilder(up_axis=UP_AXIS)
    builder.add_simplicits_object(simplicits_object, num_qp=1024)

    model = builder.finalize()
    assert model is not None
    assert model.state() is not None
    state = model.state()
    assert state.sim_z is not None and state.sim_z.shape[0] == simplicits_object.num_handles*12
    assert state.sim_z_prev is not None and state.sim_z_prev.shape[0] == simplicits_object.num_handles*12
    assert state.sim_z_dot is not None and state.sim_z_dot.shape[0] == simplicits_object.num_handles*12
    solver = SimplicitsSolver(model)
    assert solver is not None
    state_in = model.state()
    state_out = model.state()
    dt = 0.05
    solver.step(state_in, state_out, None, None, dt)
    # Get the points of the object at state_in
    rest_pts = wp.to_torch(model.sim_z_to_full(state_in.sim_z))
    deformed_pts = wp.to_torch(model.sim_z_to_full(state_out.sim_z))
    assert deformed_pts.shape == rest_pts.shape

    #Ensure that the object has slightly dropped due in y direction to gravity
    assert deformed_pts[:, 1].mean() < rest_pts[:, 1].mean()

    # Drop the object for 20 steps, no floor
    for _ in range(20):
        state_in.clear_forces()
        state_out.clear_forces()
        solver.step(state_in, state_out, None, None, dt)
        state_in, state_out = state_out, state_in

    deformed_pts = wp.to_torch(model.sim_z_to_full(state_in.sim_z))
    assert deformed_pts[:, AXIS].mean() < -1.0 # below floor in y direction
    assert deformed_pts[:, AXIS].mean() < rest_pts[:, AXIS].mean()

def test_solver_with_newton_floor(simplicits_object):
    """Test that solver step works with a newton floor."""
    builder = SimplicitsModelBuilder(up_axis=UP_AXIS)
    builder.add_simplicits_object(simplicits_object,
    init_transform=torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 2.0],
                                [0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda'),
     num_qp=512)

    # Add ground plane
    builder.add_shape_plane(
        plane=(*builder.up_vector, -1.0),
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
    assert model.state() is not None
    state = model.state()
    assert state.sim_z is not None and state.sim_z.shape[0] == simplicits_object.num_handles*12
    assert state.sim_z_prev is not None and state.sim_z_prev.shape[0] == simplicits_object.num_handles*12
    assert state.sim_z_dot is not None and state.sim_z_dot.shape[0] == simplicits_object.num_handles*12
    solver = SimplicitsSolver(model)
    assert solver is not None
    state_in = model.state()
    state_out = model.state()
    dt = 0.05
    contacts = model.collide(state_in)
    solver.step(state_in, state_out, None, contacts, dt)
    # Get the points of the object at state_in
    rest_pts = wp.to_torch(model.sim_z_to_full(state_in.sim_z))
    deformed_pts = wp.to_torch(model.sim_z_to_full(state_out.sim_z))
    assert deformed_pts.shape == rest_pts.shape

    #Ensure that the object has slightly dropped due in y direction to gravity
    assert deformed_pts[:, AXIS].mean() < rest_pts[:, AXIS].mean()

    # Drop the object for 20 steps with floor collision
    for _ in range(20):
        state_in.clear_forces()
        state_out.clear_forces()
        contacts = model.collide(state_in)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in

    deformed_pts = wp.to_torch(model.sim_z_to_full(state_in.sim_z))
    assert deformed_pts[:, AXIS].mean() > -1.0  # stayed above floor due to collision


def test_solver_kinematic_object_does_not_move(simplicits_object):
    """Kinematic object stays fixed even when penetrating the floor;
    dynamic object is correctly stopped by the floor."""
    num_qp = 200
    builder = SimplicitsModelBuilder(up_axis=UP_AXIS)

    # Obj 0: kinematic, positioned below the floor (penetrating)
    builder.add_simplicits_object(
        simplicits_object,
        num_qp=num_qp,
        is_kinematic=True,
        init_transform=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, -0.5],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda'),
    )

    # Obj 1: dynamic, positioned above the floor
    builder.add_simplicits_object(
        simplicits_object,
        num_qp=num_qp,
        is_kinematic=False,
        init_transform=torch.tensor(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 2.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device='cuda'),
    )

    # Newton floor at y = -1.0
    builder.add_shape_plane(
        plane=(*builder.up_vector, -1.0),
        width=0.0,
        length=0.0,
        cfg=SimplicitsModelBuilder.ShapeConfig(ke=1e4, mu=0.5, kd=100.0, kf=1.0),
        label="ground_plane",
    )

    model = builder.finalize()
    solver = SimplicitsSolver(model)
    state_in = model.state()
    state_out = model.state()

    # Snapshot kinematic DOFs before any stepping
    num_kin_dofs = simplicits_object.num_handles * 12
    z_initial = wp.to_torch(state_in.sim_z).clone()

    dt = 0.02
    for _ in range(20):
        state_in.clear_forces()
        state_out.clear_forces()
        contacts = model.collide(state_in)
        solver.step(state_in, state_out, None, contacts, dt)
        state_in, state_out = state_out, state_in

    z_final = wp.to_torch(state_in.sim_z)

    # Kinematic DOFs must be exactly unchanged (floor did not push it out)
    assert torch.allclose(z_final[:num_kin_dofs], z_initial[:num_kin_dofs]), \
        "Kinematic object DOFs changed during simulation"

    # Dynamic object must have been stopped above the floor
    dyn_pts = wp.to_torch(model.sim_z_to_full(state_in.sim_z))[num_qp:]
    assert dyn_pts[:, AXIS].mean() > -1.0, \
        "Dynamic object fell through the floor"
