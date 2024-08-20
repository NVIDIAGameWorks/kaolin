# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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


import pytest
import torch
import shutil
import sys
import os

import kaolin as kal
from kaolin.utils.testing import FLOAT_TYPES, with_seed

########### Setup Functions ##############


#######################################
@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_regression_fox_sim(device, dtype):
    # Import and triangulate to enable rasterization; move to GPU
    mesh = kal.io.import_mesh(kal.__path__[0] + "/../sample_data/meshes/fox.obj", triangulate=True).cuda()
    mesh.vertices = kal.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)
    orig_vertices = mesh.vertices.clone()  # Also save original undeformed vertices

    # Physics material parameters
    soft_youngs_modulus = 1e5
    poisson_ratio = 0.45
    rho = 500  # kg/m^3
    approx_volume = 0.5  # m^3

    # Points sampled over the object
    num_samples = 1000000
    uniform_pts = torch.rand(num_samples, 3, device='cuda') * (orig_vertices.max(dim=0).values -
                                                               orig_vertices.min(dim=0).values) + orig_vertices.min(dim=0).values
    boolean_signs = kal.ops.mesh.check_sign(mesh.vertices.unsqueeze(
        0), mesh.faces, uniform_pts.unsqueeze(0), hash_resolution=512)
    pts = uniform_pts[boolean_signs.squeeze()]
    yms = torch.full((pts.shape[0],), soft_youngs_modulus, device="cuda")
    prs = torch.full((pts.shape[0],), poisson_ratio, device="cuda")
    rhos = torch.full((pts.shape[0],), rho, device="cuda")

    sim_obj = kal.physics.simplicits.SimplicitsObject(pts, yms, prs, rhos, torch.tensor(
        [approx_volume], dtype=torch.float32, device="cuda"), num_handles=5)
    sim_obj.load_model(os.path.dirname(os.path.realpath(__file__)) +
                       "/regression_test_data/fox_mesh_model_10k_steps.pt")

    assert (sim_obj.num_handles == 5)
    assert (sim_obj.model is not None)
    assert (sim_obj.model_plus_rigid(pts).shape[1] == 6)

    # Scene setup. should be identical to the reference sim scene setup
    scene = kal.physics.simplicits.SimplicitsScene()  # default empty scene
    scene.max_newton_steps = 3
    obj_idx = scene.add_object(sim_obj)
    scene.set_scene_gravity(acc_gravity=torch.tensor([0, 9.8, 0]))
    scene.set_scene_floor(floor_height=-0.8, floor_axis=1, floor_penalty=1000)
    scene.set_object_materials(obj_idx, yms=torch.tensor(1e4, device='cuda', dtype=torch.float))

    reference_sim = torch.load(os.path.dirname(os.path.realpath(__file__)) +
                               "/regression_test_data/fox_sim_reference_every_10th_step.pt")
    for s in range(101):
        scene.run_sim_step()
        if (s % 10 == 0):
            updated_verts = scene.get_object_deformed_pts(obj_idx, orig_vertices).squeeze()
            print(updated_verts)
            print(reference_sim[s // 10])
            print("------")
            assert (torch.allclose(updated_verts, reference_sim[s // 10], atol=1))
