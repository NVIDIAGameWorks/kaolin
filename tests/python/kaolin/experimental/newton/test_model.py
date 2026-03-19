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


"""Tests for SimplicitsModel.

Note: Fixtures like 'cube_mesh' and 'simplicits_object' are automatically
available from conftest.py - no import needed!
"""

import torch
import warp as wp
from kaolin.experimental.newton.model import SimplicitsModel


def test_sim_z_to_full_uninitialized():
    model = SimplicitsModel()
    result = model.sim_z_to_full(wp.zeros(0, dtype=wp.float32))
    assert result.shape == (0,)
    assert result.dtype == wp.vec3


def test_sim_z_dot_to_full_uninitialized():
    model = SimplicitsModel()
    result = model.sim_z_dot_to_full(wp.zeros(0, dtype=wp.float32))
    assert result.shape == (0,)
    assert result.dtype == wp.vec3


def test_sim_z_to_full_shape(simplicits_object):
    model = SimplicitsModel()
    model.simplicits_scene.add_object(simplicits_object)
    model.simplicits_scene.set_scene_gravity()
    scene = model.simplicits_scene
    num_pts = scene.sim_pts.shape[0]
    result = model.sim_z_to_full(scene.sim_z)
    assert result.shape == (num_pts,)
    assert result.dtype == wp.vec3


def test_sim_z_dot_to_full_shape(simplicits_object):
    model = SimplicitsModel()
    model.simplicits_scene.add_object(simplicits_object)
    model.simplicits_scene.set_scene_gravity()
    scene = model.simplicits_scene
    num_pts = scene.sim_pts.shape[0]
    result = model.sim_z_dot_to_full(wp.zeros_like(scene.sim_z))
    assert result.shape == (num_pts,)
    assert result.dtype == wp.vec3


def test_sim_z_dot_to_full_zero_velocity(simplicits_object):
    model = SimplicitsModel()
    model.simplicits_scene.add_object(simplicits_object)
    model.simplicits_scene.set_scene_gravity()
    scene = model.simplicits_scene
    result = model.sim_z_dot_to_full(wp.zeros_like(scene.sim_z))
    result_torch = wp.to_torch(result)
    assert torch.allclose(result_torch, torch.zeros_like(result_torch))


def test_sim_z_to_full_matches_get_object_deformed_pts(simplicits_object):
    model = SimplicitsModel()
    obj_idx = model.simplicits_scene.add_object(simplicits_object)
    model.simplicits_scene.set_scene_gravity()
    scene = model.simplicits_scene
    result_torch = wp.to_torch(model.sim_z_to_full(scene.sim_z))
    result_lbs = scene.get_object_deformed_pts(obj_idx)
    assert torch.allclose(result_torch, result_lbs, atol=1e-5)

    num_pts = scene.sim_pts.shape[0]
    assert result_torch.shape[0] == num_pts 
    assert result_torch.shape[1] == 3


def test_sim_z_dot_to_full_matches_get_object_deformed_pts(simplicits_object):
    model = SimplicitsModel()
    model.simplicits_scene.add_object(simplicits_object)
    model.simplicits_scene.set_scene_gravity()
    scene = model.simplicits_scene
    result_torch = wp.to_torch(model.sim_z_dot_to_full(scene.sim_z_dot))
    assert torch.allclose(result_torch, torch.zeros_like(result_torch))

    
