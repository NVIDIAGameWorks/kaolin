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

import logging

import torch
import kaolin.physics.simplicits

logger = logging.getLogger(__name__)


class PhysicsRunner:
    """Trains Simplicits objects and drives the simulation loop.

    Designed to be constructed on the main thread and then have `setup()` +
    repeated `step()` calls issued from a background thread.
    """

    def __init__(self, objects, timestep, newton_steps, enable_collisions, device):
        """
        Args:
            objects: list of ObjectState (rest-pose gaussians + init transforms).
            timestep: simulation timestep in seconds.
            newton_steps: max Newton iterations per step.
            enable_collisions: whether to enable inter-object collision handling.
            device: torch device string ('cuda' or 'cpu').
        """
        self._objects = objects
        self._timestep = float(timestep)
        self._newton_steps = int(newton_steps)
        self._enable_collisions = enable_collisions
        self._device = device
        self._scene = None
        self._current_gaussians = [None] * len(objects)
        self._obj_indices = []

    def setup(self):
        """Train one SimplicitsObject per loaded object and build the scene.

        Blocks until training is complete (may take several minutes per object).
        """
        scene = kaolin.physics.simplicits.SimplicitsScene(
            device=self._device,
            timestep=self._timestep,
            max_newton_steps=self._newton_steps,
        )

        for i, obj_state in enumerate(self._objects):
            gs = obj_state.gaussians.to(self._device)

            if obj_state.baked_physics is None:
                raise ValueError(
                    f"Object '{obj_state.name}' has no pre-baked physics. "
                    "Load it from a USD file processed by VoMP."
                )
            baked = obj_state.baked_physics.to(self._device)
            logger.info(f'[{obj_state.name}] Using pre-baked SkinnedPhysicsPoints.')

            init_transform = obj_state.transform_matrix().to(self._device)
            obj_idx = scene.add_object(baked, init_transform=init_transform)
            self._obj_indices.append(obj_idx)

            # Initialise to the transformed rest pose so the first frame looks right.
            per_pt = init_transform.unsqueeze(0).expand(gs.positions.shape[0], -1, -1)
            self._current_gaussians[i] = gs.as_transformed(per_pt)
            logger.info(f'[{obj_state.name}] Added as scene object {obj_idx}.')

        scene.set_scene_gravity(acc_gravity=torch.tensor([0.0, 0.0, 9.8], device=self._device))
        scene.set_scene_floor(floor_height=-2.0, floor_axis=2, floor_penalty=1000, flip_floor=False)

        if self._enable_collisions and len(self._objects) > 1:
            scene.enable_collisions(
                collision_particle_radius=0.05,
                detection_ratio=1.5,
                impenetrable_barrier_ratio=0.25,
                collision_penalty=1000.0,
                max_contact_pairs=10000,
                friction=0.5,
            )

        self._scene = scene
        logger.info('Physics scene ready — simulation can begin.')

    def step(self):
        """Run one backward-Euler Newton step and update deformed Gaussian models."""
        if self._scene is None:
            return
        self._scene.run_sim_step()
        for i, (obj_state, obj_idx) in enumerate(zip(self._objects, self._obj_indices)):
            with torch.no_grad():
                per_pt_transforms = self._scene.get_object_point_transforms(obj_idx, 'rendered')
                gs = obj_state.gaussians.to(self._device)
                self._current_gaussians[i] = gs.as_transformed(per_pt_transforms)

    def get_current_gaussians(self) -> list:
        """Return the list of currently-deformed GaussianSplatModels (one per object)."""
        return [g for g in self._current_gaussians if g is not None]
