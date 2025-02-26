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
from kaolin.physics.simplicits.easy_api import SimplicitsScene, SimplicitsObject
import pytest
from kaolin.utils.testing import with_seed

def run_regression_test(simplicits_scene, fem_data, tol=1e-2, test_name="fem_test"):
    faces = fem_data["mesh_faces"]  # beam faces
    start_verts = fem_data["v0"]  # beam start verts
    frame_1_verts = fem_data["v1"]  # beam at frame 1
    frame_100_verts = fem_data["v_end"]  # beam verts at frame 100

    # Checking deformation at start
    our_start_verts = simplicits_scene.get_object_deformed_pts(
        0, start_verts)  # find OUR starting deformation on the fem beam's verts

    cd = kaolin.metrics.pointcloud.chamfer_distance(start_verts.unsqueeze(0),
                                                    our_start_verts.unsqueeze(
                                                        0),
                                                    w1=1.0, w2=1.0, squared=True)
    assert cd[0].item() < tol*tol, f"Chamfer distance at start is {cd[0].item()}. This is too high. This is a very basic test, something is terribly wrong in {test_name}."

    # Checking deformation at frame 1
    simplicits_scene.run_sim_step()

    our_frame_1_verts = simplicits_scene.get_object_deformed_pts(
        0, start_verts)

    cd = kaolin.metrics.pointcloud.chamfer_distance(frame_1_verts.unsqueeze(0),
                                                    our_frame_1_verts.unsqueeze(
                                                        0),
                                                    w1=1.0, w2=1.0, squared=True)
    assert cd[0].item(
    ) < tol*tol, f"Chamfer distance at frame 1 is {cd[0].item()}. This is too high. This is a basic test, something is terribly wrong in {test_name}."

    # Checking deformation at frame 100
    for i in range(99):
        simplicits_scene.run_sim_step()

    our_frame_100_verts = simplicits_scene.get_object_deformed_pts(
        0, start_verts)

    cd = kaolin.metrics.pointcloud.chamfer_distance(frame_100_verts.unsqueeze(0),
                                                    our_frame_100_verts.unsqueeze(
                                                        0),
                                                    w1=1.0, w2=1.0, squared=True)

    assert cd[0].item() < tol, f"Chamfer distance at frame 100 is {cd[0].item()}. This is too high. Something is likely wrong in {test_name}."

@with_seed(0,0,0)
def cantilever_beam_scene_setup(device, dtype):
    """Set up cantilever beam scene for testing."""
    mesh_file = os.path.dirname(os.path.realpath(
        __file__)) + "/regression_test_data/beam_surf.obj"
    mesh = kaolin.io.import_mesh(mesh_file, triangulate=True).cuda()

    num_samples = 100000

    uniform_pts = torch.rand(num_samples, 3, device=device) * (mesh.vertices.max(
        dim=0).values - mesh.vertices.min(dim=0).values) + mesh.vertices.min(dim=0).values

    boolean_signs = kaolin.ops.mesh.check_sign(mesh.vertices.unsqueeze(
        0), mesh.faces, uniform_pts.unsqueeze(0), hash_resolution=512)

    pts = uniform_pts[boolean_signs.squeeze()]  # m
    yms = 1e5*torch.ones(pts.shape[0], device=device, dtype=dtype)  # kg/m/s^2
    prs = 0.45*torch.ones(pts.shape[0], device=device, dtype=dtype)  # unitless
    rhos = 500*torch.ones(pts.shape[0], device=device, dtype=dtype)  # kg/m^3
    object_vol = (mesh.vertices.max(dim=0)[
                  0] - mesh.vertices.min(dim=0)[0]).prod()  # m^3 #bbx volume
    dt = 0.05  # s

    # train simplicits object
    simplicits_object = SimplicitsObject.create_trained(
                                        pts, 
                                        yms, 
                                        prs, 
                                        rhos, 
                                        object_vol, 
                                        num_handles=32, 
                                        num_samples=1000, 
                                        model_layers=10, 
                                        training_batch_size=10, 
                                        training_num_steps=25000, 
                                        training_lr_start=1e-3, 
                                        training_lr_end=1e-3, 
                                        training_le_coeff=1e-1, 
                                        training_lo_coeff=1e6, 
                                        training_log_every=1000, 
                                        normalize_for_training=True)


    scene = SimplicitsScene(
        device=device,
        dtype=dtype,
        timestep=dt,
        max_newton_steps=10,  # run to near convergence
        max_ls_steps=20,
    )
    scene.newton_hessian_regularizer = 0
    scene.direct_solve = True
    
    scene.add_object(simplicits_object, num_qp=1024)

    scene.set_scene_gravity(torch.tensor([0, 9.8, 0]))
    scene.set_scene_floor(floor_height=-1.0, floor_axis=1,
                          floor_penalty=10000.0, flip_floor=False)

    scene.set_object_boundary_condition(
        0, "right", lambda x: x[:, 0] >= 0.98, bdry_penalty=10000.0)

    return mesh, scene


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_cantilever_beam_simulation(device, dtype):
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
    
    # Load simplicits scene 
    _, simplicits_scene = cantilever_beam_scene_setup(device, dtype)
    
    # Load FEM beam results
    data = torch.load(os.path.dirname(os.path.realpath(
        __file__)) + "/regression_test_data/wpfem_vertex_deformations_beam.pth", weights_only=False)
    
    run_regression_test(simplicits_scene, data, tol=0.015, test_name="cantilever_beam")