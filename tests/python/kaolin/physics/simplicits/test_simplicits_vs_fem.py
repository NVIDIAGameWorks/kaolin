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

import torch
import os
import warp as wp
import kaolin
import numpy as np
from typing import Any
from kaolin.physics.simplicits import SimplicitsScene, SimplicitsObject, PhysicsPoints
from kaolin.physics.simplicits.network import SkinningModule

import pytest
from kaolin.utils.testing import with_seed

def run_regression_test(simplicits_scene, fem_data, tol=1e-2, test_name="fem_test"):
    start_verts = fem_data["v0"]  # beam start verts
    frame_1_verts = fem_data["v1"]  # beam at frame 1
    frame_100_verts = fem_data["v_end"]  # beam verts at frame 100

    # Checking deformation at start
    our_start_verts = simplicits_scene.get_object_deformed_pts(0, points='rendered')

    cd = kaolin.metrics.pointcloud.chamfer_distance(start_verts.unsqueeze(0),
                                                    our_start_verts.unsqueeze(0),
                                                    w1=1.0, w2=1.0, squared=True)
    assert cd[0].item() < tol*tol, f"Chamfer distance at start is {cd[0].item()}. This is too high. This is a very basic test, something is terribly wrong in {test_name}."

    # Checking deformation at frame 1
    simplicits_scene.run_sim_step()

    our_frame_1_verts = simplicits_scene.get_object_deformed_pts(0, points='rendered')

    cd = kaolin.metrics.pointcloud.chamfer_distance(frame_1_verts.unsqueeze(0),
                                                    our_frame_1_verts.unsqueeze(0),
                                                    w1=1.0, w2=1.0, squared=True)
    assert cd[0].item(
    ) < tol*tol + 1e-5, f"Chamfer distance at frame 1 is {cd[0].item()}. This is too high. This is a basic test, something is terribly wrong in {test_name}."

    # Checking deformation at frame 100
    for i in range(99):
        simplicits_scene.run_sim_step()

    our_frame_100_verts = simplicits_scene.get_object_deformed_pts(0, points='rendered')

    cd = kaolin.metrics.pointcloud.chamfer_distance(frame_100_verts.unsqueeze(0),
                                                    our_frame_100_verts.unsqueeze(0),
                                                    w1=1.0, w2=1.0, squared=True)

    assert cd[0].item() < tol, f"Chamfer distance at frame 100 is {cd[0].item()}. This is too high. Something is likely wrong in {test_name}."


@pytest.fixture
@with_seed(0, 0, 0)
def cantilever_beam_data(device, dtype):
    """Load cantilever beam mesh and sample interior points with material properties."""
    mesh_file = os.path.dirname(os.path.realpath(
        __file__)) + "/regression_test_data/beam_surf.obj"
    mesh = kaolin.io.import_mesh(mesh_file, triangulate=True).cuda()

    num_samples = 100000
    uniform_pts = torch.rand(num_samples, 3, device=device) * (mesh.vertices.max(
        dim=0).values - mesh.vertices.min(dim=0).values) + mesh.vertices.min(dim=0).values
    boolean_signs = kaolin.ops.mesh.check_sign(mesh.vertices.unsqueeze(
        0), mesh.faces, uniform_pts.unsqueeze(0), hash_resolution=512)

    pts = uniform_pts[boolean_signs.squeeze()]  # m
    yms = torch.full((pts.shape[0],) , 1e5, dtype=torch.float32, device=device)  # kg/m/s^2
    prs = torch.full((pts.shape[0],) , 0.45, dtype=torch.float32, device=device)  # unitless
    rhos = torch.full((pts.shape[0],) , 500.0, dtype=torch.float32, device=device)  # kg/m^3
    object_vol = (mesh.vertices.max(dim=0)[0] - mesh.vertices.min(dim=0)[0]).prod()  # m^3

    fem_data = torch.load(os.path.dirname(os.path.realpath(
        __file__)) + "/regression_test_data/wpfem_vertex_deformations_beam.pth", weights_only=False)

    return mesh, pts, yms, prs, rhos, object_vol, fem_data


@pytest.fixture(params=["rkpm", "trained"])
@with_seed(0,0,0)
def cantilever_beam_setup(request, device, dtype, cantilever_beam_data):
    """Fixture to set up cantilever beam scene for testing."""
    mesh, pts, yms, prs, rhos, object_vol, fem_data = cantilever_beam_data

    phys = PhysicsPoints(pts=pts, yms=yms, prs=prs, rhos=rhos, appx_vol=object_vol)
    if request.param == "rkpm":
        simplicits_object = SimplicitsObject.create_with_rkpm(
            physics_points=phys,
            num_handles=32, num_points=8192, num_nodes=1024, dtype=torch.float64)
    elif request.param == "trained":
        weights_file = os.path.dirname(os.path.realpath(
            __file__)) + "/regression_test_data/beam_weights_fcn_32_handles.pth"
        data = torch.load(weights_file, weights_only=False)
        fcn = SkinningModule.from_function(data['model'], data['bb_min'], data['bb_max'])
        simplicits_object = SimplicitsObject.create_from_function(
            physics_points=phys, fcn=fcn)

    rendered_pts = fem_data["v0"]

    scene = SimplicitsScene(
        device=device, timestep=0.05,
        max_newton_steps=10, max_ls_steps=20)
    scene.newton_hessian_regularizer = 0
    scene.direct_solve = True

    scene.add_object(simplicits_object, num_qp=1024,
                     renderable_pts=rendered_pts)

    scene.set_scene_gravity(torch.tensor([0, 9.8, 0]))
    scene.set_scene_floor(floor_height=-1.0, floor_axis=1,
                          floor_penalty=10000.0, flip_floor=False)
    scene.set_object_boundary_condition(
        0, "right", lambda x: x[:, 0] >= 0.98, bdry_penalty=10000.0)

    if request.param == "rkpm":
        tol = 0.005
    elif request.param == "trained":
        tol = 0.02

    return mesh, scene, tol, fem_data


@pytest.fixture(params=["rkpm", "trained"])
@with_seed(0, 0, 0)
def cube_drop_setup(request, device, dtype):
    """Fixture to set up cube drop scene for testing."""
    mesh_file = os.path.dirname(os.path.realpath(
        __file__)) + "/regression_test_data/cube_surf.obj"
    mesh = kaolin.io.import_mesh(mesh_file, triangulate=True).cuda()

    num_samples = 100000

    uniform_pts = torch.rand(num_samples, 3, device=device) * (mesh.vertices.max(
        dim=0).values - mesh.vertices.min(dim=0).values) + mesh.vertices.min(dim=0).values

    boolean_signs = kaolin.ops.mesh.check_sign(mesh.vertices.unsqueeze(
        0), mesh.faces, uniform_pts.unsqueeze(0), hash_resolution=512)

    pts = uniform_pts[boolean_signs.squeeze()]  # m
    yms = torch.full((pts.shape[0],) , 1e4, dtype=torch.float32, device=device)  # kg/m/s^2
    prs = torch.full((pts.shape[0],) , 0.45, dtype=torch.float32, device=device)  # unitless
    rhos = torch.full((pts.shape[0],) , 500.0, dtype=torch.float32, device=device)  # kg/m^3
    object_vol = (mesh.vertices.max(dim=0)[
                  0] - mesh.vertices.min(dim=0)[0]).prod()  # m^3 #bbx volume
    dt = 0.05  # s

    fem_data = torch.load(os.path.dirname(os.path.realpath(
        __file__)) + "/regression_test_data/wpfem_vertex_deformations_cube.pth", weights_only=False)

    phys = PhysicsPoints(pts=pts, yms=yms, prs=prs, rhos=rhos, appx_vol=object_vol)
    if request.param == "rkpm":
        simplicits_object = SimplicitsObject.create_with_rkpm(
            physics_points=phys,
            num_handles=32, num_points=8192, num_nodes=1024, dtype=torch.float64)
    elif request.param == "trained":
        weights_file = os.path.dirname(os.path.realpath(
            __file__)) + "/regression_test_data/cube_weights_fcn_32_handles.pth"
        data = torch.load(weights_file, weights_only=False)
        fcn = SkinningModule.from_function(data['model'], data['bb_min'], data['bb_max'])
        simplicits_object = SimplicitsObject.create_from_function(
            physics_points=phys, fcn=fcn)

    rendered_pts = fem_data["v0"]

    scene = SimplicitsScene(
        device=device, timestep=dt,
        max_newton_steps=10, max_ls_steps=20)
    scene.newton_hessian_regularizer = 0
    scene.direct_solve = True

    scene.add_object(simplicits_object, num_qp=1000,
                     renderable_pts=rendered_pts)

    scene.set_scene_gravity(torch.tensor([0, 9.8, 0]))
    scene.set_scene_floor(floor_height=-1.0, floor_axis=1,
                          floor_penalty=10000.0, flip_floor=False)

    if request.param == "rkpm":
        tol = 0.0015
    elif request.param == "trained":
        tol = 0.0015

    return mesh, scene, tol, fem_data


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cantilever_beam(device, dtype, cantilever_beam_setup):
    r"""Using the Siggraph Asia 2025 results to compare a hanging cantilever beam with the right edge fixed in warp.fem vs simplicits.
    Tests that all chamfer distances are close to 0 (or below 1e-2). Since the simplicits beam is reduced there will be numerical stiffness even at 32 handles.
    The final rest position will be different from the fem beam, but the chamfer distance tolerance is set to account for this.
    
    Common parameters: 
        Ym: 1e5
        Pr: 0.45
        Rho: 500
        dt: 5e-2
        Steps: 100
        Vol: 0.0625
    
    Simplicits params:
        Handles: 32
        Training steps: 20k
        Lr: 1e-3
        Le_coeff: 1e-1
        Lo_coeff: 1e6
    """
    
    _, simplicits_scene, tol, data = cantilever_beam_setup

    run_regression_test(simplicits_scene, data, tol=tol, test_name="cantilever_beam")
    

@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cube_drop(device, dtype, cube_drop_setup):
    """Using the Siggraph Asia 2025 results to compare a unit cube drop in warp.fem vs simplicits.
    Tests that all chamfer distances are close to 0 (or below 1e-3).

    Common parameters:
        Ym: 1e4
        Pr: 0.45
        Rho: 500
        dt: 5e-2
        Steps: 100
        Vol: 1

    Simplicits params:
        Handles: 32
        Training steps: 20k
        Lr: 1e-3
        Le_coeff: 1e-1
        Lo_coeff: 1e6
    """

    _, simplicits_scene, tol, data = cube_drop_setup

    run_regression_test(simplicits_scene, data, tol=tol, test_name="cube_drop")


# ---------------------------------------------------------------------------
# Fixtures parametrized over individual flags (rkpm object, cantilever beam)
# ---------------------------------------------------------------------------

@pytest.fixture
@with_seed(0, 0, 0)
def cantilever_beam_dFdz_setup(request, device, dtype, cantilever_beam_data):
    """Cantilever beam with rkpm object, analytical dFdz."""
    mesh, pts, yms, prs, rhos, object_vol, fem_data = cantilever_beam_data
    phys = PhysicsPoints(pts=pts, yms=yms, prs=prs, rhos=rhos, appx_vol=object_vol)
    simplicits_object = SimplicitsObject.create_with_rkpm(
        physics_points=phys,
        num_handles=32, num_points=8192, num_nodes=1024, dtype=torch.float64)
    rendered_pts = fem_data["v0"]

    scene = SimplicitsScene(device=device, timestep=0.05,
                            max_newton_steps=10, max_ls_steps=20)
    scene.newton_hessian_regularizer = 0
    scene.direct_solve = True
    scene.add_object(simplicits_object, num_qp=1024,
                     renderable_pts=rendered_pts)
    scene.set_scene_gravity(torch.tensor([0, 9.8, 0]))
    scene.set_scene_floor(floor_height=-1.0, floor_axis=1,
                          floor_penalty=10000.0, flip_floor=False)
    scene.set_object_boundary_condition(
        0, "right", lambda x: x[:, 0] >= 0.98, bdry_penalty=10000.0)
    return mesh, scene, fem_data


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cantilever_beam_dFdz_consistency(device, dtype, cantilever_beam_dFdz_setup):
    """Analytical dFdz should pass the cantilever beam FEM regression test."""
    _, simplicits_scene, data = cantilever_beam_dFdz_setup
    run_regression_test(simplicits_scene, data, tol=0.005, test_name="cantilever_beam_dFdz")

