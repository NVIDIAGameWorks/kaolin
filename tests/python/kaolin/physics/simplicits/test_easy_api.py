#1. Test simplicits object with 0 handles
#2. Test simplicits object with 10 handles

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

import os, sys 
import torch 
import numpy as np
import pytest
from kaolin.utils.testing import FLOAT_TYPES, with_seed

import kaolin as kal
import kaolin.physics.simplicits as simplicits

import logging 
logging.disable(logging.INFO)

########### Setup Functions ##############
def sdBox(p):
    SDBOXSIZE  = [1,1,1]
    b = np.array(SDBOXSIZE)
    q = np.absolute(p) - b
    return  np.linalg.norm(np.array([max(q[0], 0.0), max(q[1], 0.0), max(q[2], 0.0)])) + min(max(q[0],max(q[1],q[2])),0.0)

def example_unit_cube_object(num_points=100000, yms=1e5, prs=0.45, rhos=100, device='cuda' ,dtype=torch.float):
    uniform_points = np.random.uniform([-0.5,-0.5,-0.5], [0.5,0.5,0.5], size=(num_points, 3))
    sdf_vals = np.apply_along_axis(sdBox, 1, uniform_points)
    keep_points = np.nonzero(sdf_vals <= 0)[0] # keep points where sd is not positive
    X0 = uniform_points[keep_points, :]
    X0_sdfval = sdf_vals[keep_points]

    YMs = yms*np.ones(X0.shape[0])
    PRs = prs*np.ones_like(YMs)
    Rhos = rhos*np.ones_like(YMs)

    bb_vol = (np.max(uniform_points[:,0]) - np.min(uniform_points[:,0])) * (np.max(uniform_points[:,1]) - np.min(uniform_points[:,1])) * (np.max(uniform_points[:,2]) - np.min(uniform_points[:,2]))
    vol_per_sample = bb_vol / uniform_points.shape[0]
    appx_vol = vol_per_sample*X0.shape[0]
    return torch.tensor(X0, device=device, dtype=dtype), torch.tensor(X0_sdfval, device=device, dtype=dtype), torch.tensor(YMs, device=device, dtype=dtype), torch.tensor(PRs, device=device, dtype=dtype), torch.tensor(Rhos, device=device, dtype=dtype), torch.tensor(appx_vol, device=device, dtype=dtype)


def get_toy_object(device, dtype):
	so_pts, _, so_yms, so_prs, so_rhos, so_appx_vol = example_unit_cube_object(device=device, dtype=dtype)
	# move it up in y-coordinates by 1
	so_pts[:,1] += 1

	sim_obj = simplicits.SimplicitsObject(so_pts, so_yms, so_prs, so_rhos, so_appx_vol, num_handles=0)

	# Useful for extensively testing the training process
	# Slows down unit tests a lot, so use it in the future as a notebook test/integration test
	# FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
	# if os.path.exists(FILE_DIR + "/data/example_unit_cube_model.pth"):
	# 	sim_obj.load_model(FILE_DIR + "/data/example_unit_cube_model.pth")
	# else:
	# 	sim_obj.train(num_steps=100) # do nothing if num_handles = 0
	# 	sim_obj.save_model(FILE_DIR + "/data/example_unit_cube_model.pth")
	
	return sim_obj
#######################################

@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float])
def test_easy_api_scene_fcns(device, dtype):
	sim_obj = get_toy_object(device, dtype)
	scene = simplicits.SimplicitsScene() # default empty scene
	scene.max_newton_steps = 3 
	
	# Add our object to the scene. The scene copies it into an internal SimulatableObject utility class
	init_transform = torch.tensor([[0,0,0,1],[0,0,0,0],[0,0,0,0]], device=device, dtype=dtype)
	obj_idx = scene.add_object(sim_obj, init_tfm=init_transform)
	assert(len(scene.sim_obj_dict) == 1)
	assert(scene.sim_obj_dict[obj_idx])
	assert(scene.current_id == 1)
	assert(torch.allclose(scene.sim_obj_dict[obj_idx].z.reshape(-1,3,4)[-1],  init_transform))

	# Add our object to the scene. The scene copies it into an internal SimulatableObject utility class
	obj_idx2 = scene.add_object(sim_obj)
	assert(len(scene.sim_obj_dict) == 2)
	assert(scene.sim_obj_dict[obj_idx2])
	assert(scene.current_id == 2)
 
	# Remove object 2 from scene
	scene.remove_object(obj_idx2)
	assert(len(scene.sim_obj_dict) == 1)
	assert(obj_idx2 not in scene.sim_obj_dict)
	assert(scene.current_id == 2) # never decrement id, just keeps going up

	# Add gravity to the scene
	# TODO (Vismay): Add a way to access the sim_obj's force dict through the scene
	# Should not be accessing the utility class at all here (and other tests below)
	scene.set_scene_gravity(acc_gravity=torch.tensor([0, 9.8, 0]))
	assert("pt_wise" in scene.sim_obj_dict[obj_idx].force_dict)
	assert("gravity" in scene.sim_obj_dict[obj_idx].force_dict["pt_wise"])

	# Add floor to the scene
	scene.set_scene_floor(floor_height=-1, floor_axis=1, floor_penalty=1000)
	assert("pt_wise" in scene.sim_obj_dict[obj_idx].force_dict)
	assert("floor" in scene.sim_obj_dict[obj_idx].force_dict["pt_wise"])
	assert("floor" in scene.sim_obj_dict[obj_idx].force_dict["pt_wise"])

	# Add boundary to the scene
	bdry_fcn = lambda pts: pts[:,1]>1.45
	scene.set_object_boundary_condition(obj_idx, "top_boundary", bdry_fcn, bdry_penalty=10000)
	assert("pt_wise" in scene.sim_obj_dict[obj_idx].force_dict)
	assert("top_boundary" in scene.sim_obj_dict[obj_idx].force_dict["pt_wise"])
	assert(torch.all(scene.sim_obj_dict[obj_idx].force_dict["pt_wise"]["top_boundary"]["force_object"].pinned_vertices[:,1]>1.45))

	# Add neohookean to the scene
	scene.set_object_materials(obj_idx, yms=torch.tensor(1e5, device=device, dtype=dtype))
	assert("defo_grad_wise" in scene.sim_obj_dict[obj_idx].force_dict)
	assert("material" in scene.sim_obj_dict[obj_idx].force_dict["defo_grad_wise"])


	# Remove some forces from the scene
	scene.remove_scene_force("gravity")
	assert("gravity" not in scene.sim_obj_dict[obj_idx].force_dict["pt_wise"])
	scene.remove_scene_force("floor")
	assert("gravity" not in scene.sim_obj_dict[obj_idx].force_dict["pt_wise"])

	# Set gravity again and drop
	scene.set_scene_gravity(acc_gravity=torch.tensor([0, 9.8, 0]))

	for s in range(20):
		scene.run_sim_step()

	# Check that obj not on floor
	assert(torch.all(scene.get_object_deformed_pts(obj_idx, scene.get_object(obj_idx).sim_pts).squeeze()[:,1]> (-1+0.05)))


	# Now reset scene.
	scene.reset_object(obj_idx)

	#remove boundary
	scene.sim_obj_dict[obj_idx].remove_force("top_boundary")
	assert("top_boundary" not in scene.sim_obj_dict[obj_idx].force_dict["pt_wise"])

	# Drop again
	for s in range(20):
		scene.run_sim_step()

	# Check that obj not on floor
	assert(torch.all(scene.get_object_deformed_pts(obj_idx, scene.get_object(obj_idx).sim_pts).squeeze()[:,1] < (1.45)))

