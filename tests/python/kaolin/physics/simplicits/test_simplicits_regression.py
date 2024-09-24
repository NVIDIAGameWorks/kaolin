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
import logging
import os

import kaolin as kal
from kaolin.utils.testing import FLOAT_TYPES, with_seed

########### Setup Functions ##############

# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = False

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False


def load_and_set_fox(device, dtype):
    # Import and triangulate to enable rasterization; move to GPU
    mesh = kal.io.import_mesh(kal.__path__[0] + "/../sample_data/meshes/fox.obj", triangulate=True).to(device)
    mesh.vertices = kal.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)
    orig_vertices = mesh.vertices.clone()  # Also save original undeformed vertices

    # Physics material parameters
    soft_youngs_modulus = 1e4
    poisson_ratio = 0.45
    rho = 500  # kg/m^3
    approx_volume = 0.5  # m^3

    # Points sampled over the object
    num_samples = 1000000
    torch.manual_seed(0)
    uniform_pts = torch.rand(num_samples, 3, device=device) * (orig_vertices.max(dim=0).values -
                                                               orig_vertices.min(dim=0).values) + orig_vertices.min(dim=0).values
    boolean_signs = kal.ops.mesh.check_sign(mesh.vertices.unsqueeze(
        0), mesh.faces, uniform_pts.unsqueeze(0), hash_resolution=512)
    pts = uniform_pts[boolean_signs.squeeze()]
    yms = torch.full((pts.shape[0],), soft_youngs_modulus, device=device)
    prs = torch.full((pts.shape[0],), poisson_ratio, device=device)
    rhos = torch.full((pts.shape[0],), rho, device=device)
    return pts, yms, prs, rhos, approx_volume, orig_vertices


def load_and_set_box(device, dtype):
    # Import and triangulate to enable rasterization; move to GPU
    mesh = kal.io.import_mesh(kal.__path__[0] + "/../sample_data/meshes/fox.obj", triangulate=True).to(device)
    mesh.vertices = kal.ops.pointcloud.center_points(mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)
    orig_vertices = mesh.vertices.clone()  # Also save original undeformed vertices

    # Physics material parameters
    soft_youngs_modulus = 1e4
    poisson_ratio = 0.45
    rho = 500  # kg/m^3
    approx_volume = 0.5  # m^3

    # Points sampled over the object
    num_samples = 10000
    torch.manual_seed(0)
    uniform_pts = torch.rand(num_samples, 3, device=device) * (orig_vertices.max(dim=0).values -
                                                               orig_vertices.min(dim=0).values) + orig_vertices.min(dim=0).values
    pts = uniform_pts
    yms = torch.full((pts.shape[0],), soft_youngs_modulus, device=device)
    prs = torch.full((pts.shape[0],), poisson_ratio, device=device)
    rhos = torch.full((pts.shape[0],), rho, device=device)
    return pts, yms, prs, rhos, approx_volume, orig_vertices

#######################################


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_regression_fox_train(device, dtype):
    logging.getLogger('kaolin.physics').setLevel(logging.DEBUG)

    r"Step 1: Load and Setup Object"
    torch.manual_seed(0)
    pts, yms, prs, rhos, approx_volume, orig_vertices = load_and_set_box(device, dtype)
    print(pts)

    r"Step 2: Create Simplicits Object"
    torch.manual_seed(0)
    sim_obj = kal.physics.simplicits.SimplicitsObject(pts, yms, prs, rhos, torch.tensor(
        [approx_volume], dtype=dtype, device=device))
    torch.manual_seed(0)
    training_vals = sim_obj.train(num_steps=4000)
    print(training_vals)
    r"Step 3: Read Reference Train Vals"
    filename = os.path.dirname(os.path.realpath(__file__)) + \
        "/regression_test_data/box_training_reference_log_4k_steps.txt"
    reference_training_val = torch.load(filename)

    i = 0
    r"Step 4: Asserts Training Match"
    for tvals in training_vals:
        le = reference_training_val[i][0]
        lo = reference_training_val[i][1]
        assert (abs(le - tvals[0]) < 0.005 * le)
        assert (abs(lo - tvals[1]) < 0.005 * lo)
        i += 1


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float32])
def test_regression_fox_train_warp(device, dtype):
    logging.getLogger('kaolin.physics').setLevel(logging.DEBUG)

    r"Step 1: Load and Setup Object"
    torch.manual_seed(0)
    pts, yms, prs, rhos, approx_volume, orig_vertices = load_and_set_box(device, dtype)

    r"Step 2: Create Simplicits Object"
    torch.manual_seed(0)
    sim_obj = kal.physics.simplicits.SimplicitsObject(pts, yms, prs, rhos, torch.tensor(
        [approx_volume], dtype=dtype, device=device), warp_training=True)
    torch.manual_seed(0)
    training_vals = sim_obj.train(num_steps=4000)
    print(training_vals)
    r"Step 3: Read Reference Train Vals"
    filename = os.path.dirname(os.path.realpath(__file__)) + \
        "/regression_test_data/box_training_reference_log_4k_steps.txt"
    reference_training_val = torch.load(filename)

    i = 0
    r"Step 4: Asserts Training Match"
    for tvals in training_vals:
        le = reference_training_val[i][0]
        lo = reference_training_val[i][1]
        assert (abs(le - tvals[0]) < 0.005 * le)
        assert (abs(lo - tvals[1]) < 0.005 * lo)
        i += 1


# @pytest.mark.parametrize('device', ['cuda'])
# @pytest.mark.parametrize('dtype', [torch.float])
# def test_regression_fox_sim(device, dtype):
#     logging.getLogger('kaolin.physics').setLevel(logging.CRITICAL)

#     r"Step 1: Load and Setup Object"
#     torch.manual_seed(0)
#     pts, yms, prs, rhos, approx_volume, orig_vertices = load_and_set_fox(device, dtype)

#     r"Step 2: Create Simplicits Object"
#     torch.manual_seed(0)
#     sim_obj = kal.physics.simplicits.SimplicitsObject(pts, yms, prs, rhos, torch.tensor(
#         [approx_volume], dtype=dtype, device=device))

#     r"Step 3: Load Pre-Saved Network"
#     sim_obj.load_model(os.path.dirname(os.path.realpath(__file__)) +
#                        "/regression_test_data/fox_mesh_model_10k_steps.pt")

#     r"Step 4: Assert Parameter Match"
#     assert (sim_obj.num_handles == 10)
#     assert (sim_obj.model is not None)
#     assert (sim_obj.model_plus_rigid(pts).shape[1] == 11)
#     assert (sim_obj.num_samples == 1000)
#     assert (torch.allclose(sim_obj.yms, 1e4 * torch.ones_like(sim_obj.yms)))
#     assert (torch.allclose(sim_obj.prs, 0.45 * torch.ones_like(sim_obj.prs)))
#     assert (torch.allclose(sim_obj.rhos, 500 * torch.ones_like(sim_obj.rhos)))

#     r"Step 5: Setup Scene"
#     # Scene setup. should be identical to the reference sim scene setup
#     scene = kal.physics.simplicits.SimplicitsScene()  # default empty scene
#     scene.max_newton_steps = 3
#     obj_idx = scene.add_object(sim_obj)
#     scene.set_scene_gravity(acc_gravity=torch.tensor([0, 9.8, 0]))
#     scene.set_scene_floor(floor_height=-0.8, floor_axis=1, floor_penalty=1000)
#     scene.set_object_materials(obj_idx, yms=torch.tensor(1e4, device=device, dtype=dtype))

#     reference_sim = torch.load(os.path.dirname(os.path.realpath(__file__)) +
#                                "/regression_test_data/fox_sim_reference_every_10th_step.pt")
#     print(len(reference_sim))
#     r"Step 6: Simulate and Assert Match at Every 10 Steps"
#     torch.manual_seed(0)
#     for s in range(101):
#         scene.run_sim_step()
#         if (s % 10 == 0):
#             updated_verts = scene.get_object_deformed_pts(obj_idx, orig_vertices).squeeze()
#             print(updated_verts)
#             print(reference_sim[s // 10])
#             print("------")
#             assert (torch.allclose(updated_verts, reference_sim[s // 10], atol=1))
