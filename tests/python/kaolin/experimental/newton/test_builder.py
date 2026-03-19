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

"""Tests for SimplicitsModelBuilder.

Note: Fixtures from conftest.py are automatically available!
"""

import torch
import pytest
from kaolin.experimental.newton.builder import SimplicitsModelBuilder


def test_builder_initialization():
    """Test that builder can be created."""
    builder = SimplicitsModelBuilder()
    model = builder.finalize()
    assert builder is not None
    assert model is not None
    assert model.simplicits_scene is not None
    assert model.simplicits_scene.sim_z is None
    assert model.simplicits_scene.sim_pts is None


def test_builder_add_object(simplicits_object):
    """Test adding a simplicits object to builder.

    Uses the 'simplicits_object' fixture from conftest.py automatically.
    """
    num_qp = 100
    builder = SimplicitsModelBuilder()
    builder.add_simplicits_object(simplicits_object, num_qp=num_qp)

    model = builder.finalize()
    assert model is not None
    assert model.simplicits_scene is not None
    assert model.simplicits_scene.sim_z is not None
    assert model.simplicits_scene.sim_z.shape[0] == simplicits_object.num_handles*12
    assert model.simplicits_scene.sim_pts.shape[0] == num_qp
    assert model.state() is not None
    state = model.state()
    assert state.sim_z is not None
    assert state.sim_z.shape[0] == simplicits_object.num_handles*12
    assert state.particle_q.shape[0] == num_qp


def test_builder_with_gravity(simplicits_object):
    """Test builder with gravity force."""
    builder = SimplicitsModelBuilder()
    builder.add_simplicits_object(simplicits_object, num_qp=100)
    model = builder.finalize()

    assert model.simplicits_scene.force_dict["pt_wise"]["gravity"] is not None

def test_builder_with_boundary_conditions(simplicits_object):
    """Test builder with boundary conditions."""
    builder = SimplicitsModelBuilder()
    builder.add_simplicits_object(simplicits_object, num_qp=100)
    builder.add_simplicits_object_boundary_condition(obj_idx=0, name="left", fcn=lambda x: x[:, 0] < -0.4, bdry_penalty=10000.0)
    model = builder.finalize()

    assert model.simplicits_scene.force_dict["pt_wise"]["left"] is not None

def test_builder_add_multiple_objects(simplicits_object):
    """Test that adding two objects produces correctly sized combined buffers."""
    num_qp = 100
    builder = SimplicitsModelBuilder()
    builder.add_simplicits_object(simplicits_object, num_qp=num_qp)
    builder.add_simplicits_object(simplicits_object, num_qp=num_qp)

    model = builder.finalize()

    # Two objects registered in the scene
    assert len(model.simplicits_scene.sim_obj_dict) == 2

    # sim_z spans all handles of both objects
    assert model.simplicits_scene.sim_z.shape[0] == simplicits_object.num_handles * 12 * 2

    # sim_pts spans all quadrature points of both objects
    assert model.simplicits_scene.sim_pts.shape[0] == num_qp * 2

    # State reflects the same totals
    state = model.state()
    assert state.sim_z.shape[0] == simplicits_object.num_handles * 12 * 2
    assert state.particle_q.shape[0] == num_qp * 2


def test_builder_with_collisions(simplicits_object):
    """Test builder with collisions."""
    builder = SimplicitsModelBuilder()
    builder.add_simplicits_object(simplicits_object, num_qp=100)
    builder.add_simplicits_object(simplicits_object, num_qp=100)
    builder.add_simplicits_collisions(collision_particle_radius=0.1, detection_ratio=1.5, impenetrable_barrier_ratio=0.25, collision_penalty=1000.0, max_contact_pairs=10000, friction=0.5)
    model = builder.finalize()

    assert model.simplicits_scene.force_dict["collision"]["object"] is not None


def test_builder_add_object_num_qp_respected(simplicits_object):
    """Test that num_qp is correctly forwarded to the scene."""
    num_qp = 100
    builder = SimplicitsModelBuilder()
    builder.add_simplicits_object(simplicits_object, num_qp=num_qp)
    model = builder.finalize()

    assert model.simplicits_scene.sim_pts.shape[0] == num_qp
    assert model.simplicits_scene.sim_obj_dict[0].num_qp == num_qp


def test_builder_add_object_init_transform_3x4(simplicits_object):
    """Test that a non-identity 3x4 init_transform is forwarded and stored as a delta."""
    # Translation of +1 in x: row 0 col 3 = 1
    init_transform = torch.eye(4, device='cuda')[:3, :]  # (3,4) identity
    init_transform[0, 3] = 1.0  # translate x by 1

    builder = SimplicitsModelBuilder()
    builder.add_simplicits_object(simplicits_object, num_qp=100, init_transform=init_transform)
    model = builder.finalize()

    stored = model.simplicits_scene.sim_obj_dict[0].init_transform
    # delta = init_transform - identity, so it should be non-zero
    assert not torch.all(stored == 0), "init_transform delta should be non-zero for a non-identity transform"
    assert stored.shape == (3, 4)
    assert torch.allclose(stored, torch.tensor([[0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]], device='cuda', dtype=torch.float32))


def test_builder_add_object_init_transform_4x4(simplicits_object):
    """Test that a 4x4 init_transform is accepted and stored as a non-zero delta."""
    init_transform = torch.eye(4, device='cuda')
    init_transform[1, 3] = 0.5  # translate y by 0.5

    builder = SimplicitsModelBuilder()
    builder.add_simplicits_object(simplicits_object, num_qp=100, init_transform=init_transform)
    model = builder.finalize()

    stored = model.simplicits_scene.sim_obj_dict[0].init_transform
    assert not torch.all(stored == 0), "4x4 init_transform delta should be non-zero"
    assert stored.shape == (3, 4)
    assert torch.allclose(stored, torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0.5], [0, 0, 0, 0]], device='cuda', dtype=torch.float32))


def test_builder_add_object_bad_init_transform(simplicits_object):
    """Test that an invalid init_transform shape raises ValueError during finalize."""
    bad_transform = torch.ones(2, 4, device='cuda')

    builder = SimplicitsModelBuilder()
    builder.add_simplicits_object(simplicits_object, num_qp=100, init_transform=bad_transform)

    with pytest.raises(ValueError):
        builder.finalize()


def test_builder_add_object_is_kinematic(simplicits_object):
    """Test that is_kinematic is correctly forwarded for each object."""
    builder = SimplicitsModelBuilder()
    builder.add_simplicits_object(simplicits_object, num_qp=100, is_kinematic=True)
    builder.add_simplicits_object(simplicits_object, num_qp=100, is_kinematic=False)
    model = builder.finalize()

    assert model.simplicits_scene.sim_obj_dict[0].is_kinematic is True
    assert model.simplicits_scene.sim_obj_dict[1].is_kinematic is False


def test_builder_collisions_default_params(simplicits_object):
    """Test that add_simplicits_collisions() defaults are forwarded correctly."""
    builder = SimplicitsModelBuilder()
    builder.add_simplicits_object(simplicits_object, num_qp=100)
    builder.add_simplicits_collisions()
    model = builder.finalize()

    col = model.simplicits_scene.force_dict["collision"]["object"]
    assert col.collision_radius == 0.1
    assert col.collision_detection_ratio == 1.5
    assert col.collision_barrier_ratio == 0.25
    assert col.collision_penalty_stiffness == 1000.0
    assert col.friction == 0.5
    assert model.simplicits_scene.force_dict["collision"]["coeff"] == 1000.0


def test_builder_collisions_custom_params(simplicits_object):
    """Test that custom collision parameters are forwarded correctly."""
    builder = SimplicitsModelBuilder()
    builder.add_simplicits_object(simplicits_object, num_qp=100)
    builder.add_simplicits_collisions(
        collision_particle_radius=0.2,
        detection_ratio=2.0,
        impenetrable_barrier_ratio=0.3,
        collision_penalty=500.0,
        max_contact_pairs=5000,
        friction=0.3,
    )
    model = builder.finalize()

    col = model.simplicits_scene.force_dict["collision"]["object"]
    assert col.collision_radius == 0.2
    assert col.collision_detection_ratio == 2.0
    assert col.collision_barrier_ratio == 0.3
    assert col.collision_penalty_stiffness == 500.0
    assert col.friction == 0.3
    assert model.simplicits_scene.force_dict["collision"]["coeff"] == 500.0


def test_builder_finalize_twice_returns_distinct_models(simplicits_object):
    """Test that two successive finalize() calls return independent model objects."""
    num_qp = 100
    builder = SimplicitsModelBuilder()
    builder.add_simplicits_object(simplicits_object, num_qp=num_qp)

    model1 = builder.finalize()
    model2 = builder.finalize()

    # Identity: distinct objects
    assert model1 is not model2
    assert model1.simplicits_scene is not model2.simplicits_scene

    # Both have valid Simplicits state
    assert model1.simplicits_scene.sim_z is not None
    assert model2.simplicits_scene.sim_z is not None
    assert model1.simplicits_scene.sim_pts is not None
    assert model2.simplicits_scene.sim_pts is not None

    # Same shapes — same objects were added to each scene
    assert model1.simplicits_scene.sim_z.shape == model2.simplicits_scene.sim_z.shape
    assert model1.simplicits_scene.sim_pts.shape == model2.simplicits_scene.sim_pts.shape

    # Independent memory — buffers are distinct allocations
    assert model1.simplicits_scene.sim_z.ptr != model2.simplicits_scene.sim_z.ptr
    assert model1.simplicits_scene.sim_pts.ptr != model2.simplicits_scene.sim_pts.ptr

    # Both models have the correct particle count (not double due to accumulation bug)
    state1 = model1.state()
    state2 = model2.state()
    assert state1.particle_q.shape[0] == num_qp
    assert state2.particle_q.shape[0] == num_qp
