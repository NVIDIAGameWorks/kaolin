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

"""
Shared pytest fixtures for Newton tests.

Fixtures defined here are automatically available to all test files
in this directory and subdirectories without needing to import them.
"""

import pytest

pytest.importorskip("newton")

import os
import torch
import kaolin
from examples.tutorial.tutorial_common import COMMON_DATA_DIR
from kaolin.physics.simplicits import SimplicitsObject

# Constants for test objects
NUM_SAMPLES = 10000
SOFT_YOUNGS_MODULUS = 1e5
POISSON_RATIO = 0.45
DENSITY = 500
APPROX_VOLUME = 0.5


@pytest.fixture(scope="session")
def cube_mesh():
    """Load and prepare the cube mesh for simulation.

    Fixture scope is 'session' to load mesh once for all tests.
    """
    mesh_path = os.path.join(COMMON_DATA_DIR, "meshes")
    mesh = kaolin.io.import_mesh(
        mesh_path + "/cube.obj", triangulate=True).to('cuda')
    mesh.vertices = kaolin.ops.pointcloud.center_points(
        mesh.vertices.unsqueeze(0), normalize=True).squeeze(0)
    return mesh


@pytest.fixture
def simplicits_object(cube_mesh):
    """Create a Simplicits object from cube mesh.

    This fixture depends on the 'cube_mesh' fixture above.
    Creates a new object for each test to ensure test isolation.
    """
    orig_vertices = cube_mesh.vertices.clone()

    # Sample points uniformly over the bounding box
    uniform_pts = torch.rand(NUM_SAMPLES, 3, device='cuda') * (
        orig_vertices.max(dim=0).values - orig_vertices.min(dim=0).values
    ) + orig_vertices.min(dim=0).values

    # Create material property tensors
    yms = torch.full((NUM_SAMPLES,), SOFT_YOUNGS_MODULUS, device='cuda')
    prs = torch.full((NUM_SAMPLES,), POISSON_RATIO, device='cuda')
    rhos = torch.full((NUM_SAMPLES,), DENSITY, device='cuda')

    # Create rigid Simplicits object
    sim_obj = SimplicitsObject.create_rigid(
        uniform_pts, yms, prs, rhos, APPROX_VOLUME
    )
    return sim_obj


@pytest.fixture
def simplicits_model_with_object(simplicits_object):
    """Create a SimplicitsModel with an object already added.

    Useful for tests that need a pre-configured model.
    """
    from kaolin.experimental.newton.model import SimplicitsModel

    model = SimplicitsModel()
    obj_idx = model.simplicits_scene.add_object(simplicits_object)
    return model, obj_idx
