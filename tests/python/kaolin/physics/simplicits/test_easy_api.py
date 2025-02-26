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

import os
import shutil
import torch
import numpy as np
import pytest
from kaolin.utils.testing import FLOAT_TYPES, with_seed

from kaolin.physics.simplicits import SimplicitsObject, SimplicitsScene


import logging
logging.disable(logging.INFO)

########### Setup Functions ##############


def sdBox(p):
    SDBOXSIZE = [1, 1, 1]
    b = np.array(SDBOXSIZE)
    q = np.absolute(p) - b
    return np.linalg.norm(np.array([max(q[0], 0.0), max(q[1], 0.0), max(q[2], 0.0)])) + min(max(q[0], max(q[1], q[2])), 0.0)


def example_unit_cube_object(num_points=100000, yms=1e5, prs=0.45, rhos=100, device='cuda', dtype=torch.float):
    uniform_points = np.random.uniform([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5], size=(num_points, 3))
    sdf_vals = np.apply_along_axis(sdBox, 1, uniform_points)
    keep_points = np.nonzero(sdf_vals <= 0)[0]  # keep points where sd is not positive
    X0 = uniform_points[keep_points, :]
    X0_sdfval = sdf_vals[keep_points]

    YMs = yms * np.ones(X0.shape[0])
    PRs = prs * np.ones_like(YMs)
    Rhos = rhos * np.ones_like(YMs)

    bb_vol = (np.max(uniform_points[:, 0]) - np.min(uniform_points[:, 0])) * (np.max(uniform_points[:, 1]) -
                                                                              np.min(uniform_points[:, 1])) * (np.max(uniform_points[:, 2]) - np.min(uniform_points[:, 2]))
    vol_per_sample = bb_vol / uniform_points.shape[0]
    appx_vol = vol_per_sample * X0.shape[0]
    return torch.tensor(X0, device=device, dtype=dtype), torch.tensor(X0_sdfval, device=device, dtype=dtype), torch.tensor(YMs, device=device, dtype=dtype), torch.tensor(PRs, device=device, dtype=dtype), torch.tensor(Rhos, device=device, dtype=dtype), torch.tensor(appx_vol, device=device, dtype=dtype)


def example_rigid_cube(offset=torch.tensor([0,0,0]), device='cuda', dtype=torch.float):
    offset = offset.to(device).to(dtype)
    x0, x0_sdfval, yms, prs, rhos, appx_vol = example_unit_cube_object(
        device=device, dtype=dtype)
    x0 += offset.unsqueeze(0)
    rigid_obj = SimplicitsObject.create_rigid(
        pts=x0, yms=yms, prs=prs, rhos=rhos, appx_vol=appx_vol)
    return rigid_obj

@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_simplicits_object_create_rigid(device, dtype):
    rigid_obj = example_rigid_cube(device=device, dtype=dtype)
    assert rigid_obj.num_handles == 1


# @pytest.mark.parametrize('device', ['cuda'])
# @pytest.mark.parametrize('dtype', [torch.float])
# def test_easy_api_scene_setup(device, dtype):
#     rigid_obj_one = example_rigid_cube(device=device, dtype=dtype)
#     rigid_obj_two = example_rigid_cube(device=device, dtype=dtype)
#     scene = SimplicitsScene(device=device, dtype=dtype)
#     scene.add_object(rigid_obj_one)
#     scene.add_object(rigid_obj_two)
#     scene.get_scene_ready_for_sim()
#     assert scene.sim_z.shape[0] == 24
#     assert scene.sim_B.shape[0] == 3*scene.sim_pts.shape[0] and scene.sim_B.shape[1] == scene.sim_z.shape[0]
#     assert scene.sim_BMB.shape[0] == scene.sim_z.shape[0] and scene.sim_BMB.shape[1] == scene.sim_z.shape[0]
#     assert scene.sim_dFdz.shape[0] == 9*scene.sim_pts.shape[0] and scene.sim_dFdz.shape[1] == scene.sim_z.shape[0]


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_easy_api_scene_setup_kinematic(device, dtype):
    # 1. set kinematic object to be at 0, 10, 0 
    # 2. assert after setup that mean y coordinate is 10, store difference in y coordinate from dynamic object
    # 3. run simulation
    # 4. assert after a few steps that mean y coordinate is 10, assert difference in y coordinate from dynamic object is larger
    # 5. set kinematic object to be at 0, 5, 0
    # 6. assert after setup that mean y coordinate is 5
    # 7. run simulation
    # 8. assert after a few steps that mean y coordinate is 5, assert difference in y coordinate from dynamic object is smaller
    
    
    rigid_obj_one = example_rigid_cube(device=device, dtype=dtype)
    rigid_obj_two = example_rigid_cube(device=device, dtype=dtype)
    
    scene = SimplicitsScene(device=device, dtype=dtype)
    scene.timestep = 0.1
    scene.add_object(rigid_obj_one, 
                    init_transform=torch.tensor([[1, 0, 0, 0], 
                                       [0, 1, 0, 10], 
                                       [0, 0, 1, 0], 
                                       [0, 0, 0, 1]], device=device, dtype=dtype),
                         is_kinematic=True)
    scene.add_object(rigid_obj_two,
                    init_transform=torch.tensor([[1, 0, 0, 0],
                                   [0, 1, 0, 5],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]], device=device, dtype=dtype))
    
    scene.set_scene_gravity()
    scene.set_scene_floor()
    scene.set_object_materials(0)
    
    # 2. assert after setup that mean y coordinate is 10
    kin_pts = scene.get_object_deformed_pts(0)
    expected_kin_pts = scene.get_object(
        0).sample_pts + torch.tensor([0, 10, 0], device=device, dtype=dtype).unsqueeze(0)
    assert torch.allclose(
        kin_pts, expected_kin_pts), f"Kinematic object is not at 0, 10, 0"
    
    dyn_pts=scene.get_object_deformed_pts(1)
    mean_dyn_y=dyn_pts[:, 1].mean()
    mean_kin_y = kin_pts[:, 1].mean()
    assert abs(mean_dyn_y - 5.0) < 1.0, f"Mean y coordinate of dynamic object {mean_dyn_y} is not within 1 unit of 5" 
    
    diff_in_y = abs(mean_kin_y - mean_dyn_y) # should be 5, but store this to measure later
    
    # 3. run simulation
    for i in range(10):
        scene.run_sim_step()
        
    # 4. assert that mean y coordinate of kinematic object is 10
    kin_pts = scene.get_object_deformed_pts(0)
    expected_kin_pts = scene.get_object(
        0).sample_pts + torch.tensor([0, 10, 0], device=device, dtype=dtype).unsqueeze(0)
    assert torch.allclose(
        kin_pts, expected_kin_pts), f"Kinematic object is not at 0, 10, 0"
    
    dyn_pts=scene.get_object_deformed_pts(1)
    mean_dyn_y=dyn_pts[:, 1].mean()
    mean_kin_y = kin_pts[:, 1].mean()
    assert abs(mean_kin_y - mean_dyn_y) > diff_in_y, f"Kinematic object is at y = {mean_kin_y}, dynamic object is at y = {mean_dyn_y}. The difference in y should be larger than {diff_in_y}. Either both objects are moving, or the dynamic object is not moving."
    
        
    # 5. Move kinematic object to 0, 1, 0 
    scene.set_kinematic_object_transform(0, 
                                         torch.tensor([[1,0,0,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], device=device, dtype=dtype))
    
    # 6. assert after that mean y coordinate of kinematic object is 1
    kin_pts = scene.get_object_deformed_pts(0)
    expected_kin_pts = scene.get_object(
        0).sample_pts + torch.tensor([0, 1, 0], device=device, dtype=dtype).unsqueeze(0)
    assert torch.allclose(
        kin_pts, expected_kin_pts), f"Kinematic object is not at 0, 1, 0"
    
    dyn_pts = scene.get_object_deformed_pts(1)
    mean_dyn_y = dyn_pts[:, 1].mean()
    mean_kin_y = kin_pts[:, 1].mean()
    assert abs(mean_kin_y - mean_dyn_y) < diff_in_y, f"Difference in y coordinate of dynamic object {mean_dyn_y} is not smaller than {diff_in_y} after moving kinematic object. Kinematic object was not moved."
    

@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_easy_api_scene_reset_kinematic(device, dtype):
    # 1. set dyn and kinematic objects to be at 0, 10, 0 and 0, 5, 0
    # 2. assert placement is correct
    # 3. run simulation
    # 4. assert locations 
    # 5. set kinematic object to be at 0, 1, 0
    # 5. reset scene
    # 6. assert dynamic object is at 0, 5, 0, kinematic object is at 0, 1, 0
    
    rigid_obj_one = example_rigid_cube(device=device, dtype=dtype)
    rigid_obj_two = example_rigid_cube(device=device, dtype=dtype)

    scene = SimplicitsScene(device=device, dtype=dtype)
    scene.timestep = 0.1
    scene.add_object(rigid_obj_one,
                     init_transform=torch.tensor([[1, 0, 0, 0],
                                                 [0, 1, 0, 10],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]], device=device, dtype=dtype),
                     is_kinematic=True)
    scene.add_object(rigid_obj_two,
                     init_transform=torch.tensor([[1, 0, 0, 0],
                                                 [0, 1, 0, 5],
                                                  [0, 0, 1, 0],
                                                  [0, 0, 0, 1]], device=device, dtype=dtype))

    scene.set_scene_gravity()
    scene.set_scene_floor()
    scene.set_object_materials(0)

    # 2. assert after setup that mean y coordinate is 10
    kin_pts = scene.get_object_deformed_pts(0)
    mean_kin_y = kin_pts[:, 1].mean()
    expected_kin_pts = scene.get_object(0).sample_pts + torch.tensor([0, 10, 0], device=device, dtype=dtype).unsqueeze(0)
    assert torch.allclose(kin_pts, expected_kin_pts)
    
    dyn_pts = scene.get_object_deformed_pts(1)
    mean_dyn_y = dyn_pts[:, 1].mean()
    assert abs(
        mean_dyn_y - 5.0) < 1.0, f"Mean y coordinate of dynamic object {mean_dyn_y} is not within 1 unit of 5"

    # should be ~5, but store this to measure later
    diff_in_y = abs(mean_kin_y - mean_dyn_y)

    # 3. run simulation
    for i in range(10):
        scene.run_sim_step()

    # 4. assert that mean y coordinate of kinematic object is 10
    kin_pts = scene.get_object_deformed_pts(0)
    expected_kin_pts = scene.get_object(
        0).sample_pts + torch.tensor([0, 10, 0], device=device, dtype=dtype).unsqueeze(0)
    assert torch.allclose(
        kin_pts, expected_kin_pts), f"Kinematic object is not at 0, 10, 0"
    dyn_pts = scene.get_object_deformed_pts(1)
    mean_dyn_y = dyn_pts[:, 1].mean()
    mean_kin_y = kin_pts[:, 1].mean()
    assert abs(
        mean_kin_y - mean_dyn_y) > diff_in_y, f"Kinematic object is at y = {mean_kin_y}, dynamic object is at y = {mean_dyn_y}. The difference in y should be larger than {diff_in_y}. "

    # 5. Move kinematic object to 0, 1, 0
    scene.set_kinematic_object_transform(0,torch.tensor([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], device=device, dtype=dtype))

    # 6. Reset scene
    scene.reset_scene()
    
    # 7. assert dynamic object is at 0, 5, 0, kinematic object is at 0, 1, 0
    kin_pts = scene.get_object_deformed_pts(0)
    expected_kin_pts = scene.get_object(
        0).sample_pts + torch.tensor([0, 1, 0], device=device, dtype=dtype).unsqueeze(0)
    assert torch.allclose(kin_pts, expected_kin_pts), f"Kinematic object is not at 0, 1, 0"
    dyn_pts = scene.get_object_deformed_pts(1)
    mean_dyn_y = dyn_pts[:, 1].mean()
    assert abs(
        mean_dyn_y - 5.0) < 1.0, f"Mean y coordinate of dynamic object {mean_dyn_y} is not within 1 unit of 5"


  

