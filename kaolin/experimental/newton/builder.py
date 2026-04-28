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

import warnings

import torch
import warp as wp
import newton
from newton._src.core.types import AxisType, Axis
from kaolin.experimental.newton.model import SimplicitsModel
from kaolin.experimental.newton.collisions import SimplicitsParticleNewtonShapeSoftContact
from kaolin.physics.simplicits import SimplicitsObject
import logging

__all__ = [
    'SimplicitsModelBuilder',
]

class SimplicitsModelBuilder(newton.ModelBuilder):
    r"""Extends Newton's ModelBuilder to handle SimplicitsModel construction.

    Attributes:
        model (SimplicitsModel): The SimplicitsModel being constructed by this builder.
    """
    def __init__(self, up_axis: AxisType = Axis.Z, gravity: float = -9.81):
        r"""Initialize the builder with an empty SimplicitsModel.

        Args:
            up_axis (AxisType): The axis to use as the "up" direction in the simulation. Defaults to Axis.Z.
            gravity (float): The magnitude of gravity to apply along the up axis. Defaults to -9.81.
        """
        super().__init__(up_axis, gravity)
        self._pending_objects = []
        self._pending_boundary_conditions = []   # list of (obj_idx, name, fcn, bdry_penalty, pinned_x)
        self._pending_collisions = None           # tuple of collision kwargs, or None


    def add_simplicits_object(self, sim_object: SimplicitsObject, num_qp=1000, init_transform=None,
                              is_kinematic=False, renderable_pts=None):
        r"""Add a Simplicits soft-body object to the model.

        Wraps SimplicitsScene.add_object() to add deformable objects to the simulation.

        Args:
            sim_object (SimplicitsObject): Simplicits object wrapped into a SimulatedObject for this scene.
            num_qp (int): Number of quadrature points (sample points to integrate over).
            init_transform (torch.Tensor or None): 3x4 or 4x4 tensor for the object's initial skinning transform.
                Takes a standard transformation, not a delta; the identity matrix is subtracted and the delta is saved.
            is_kinematic (bool): If True, object is kinematic and not solved during dynamics.
            renderable_pts (torch.Tensor or None): Optional rest positions for a separate rendered point set
                (see :meth:`kaolin.physics.simplicits.simulation.SimplicitsScene.add_object`).

        """
        # obj_id = self.model.simplicits_scene.add_object(sim_object, num_qp, init_transform, is_kinematic)
        self._pending_objects.append((sim_object, num_qp, init_transform, is_kinematic, renderable_pts))

    def add_simplicits_collisions(self, collision_particle_radius=0.1,
                                        detection_ratio=1.5,
                                        impenetrable_barrier_ratio=0.25,
                                        collision_penalty=1000.0,
                                        max_contact_pairs=10000,
                                        friction=0.5):
        r"""Enable soft-body to soft-body collisions between Simplicits objects.

        Wraps SimplicitsScene.enable_collisions() for self-collisions and inter-object collisions.
        The call is deferred until finalize().

        Args:
            collision_particle_radius (float): Scene-wide collision particle radius; penalty begins here. Defaults to 0.1.
            detection_ratio (float): Collision detection radius as ratio of collision_particle_radius. Defaults to 1.5.
            impenetrable_barrier_ratio (float): Collision barrier radius as ratio of collision_particle_radius. Defaults to 0.25.
            collision_penalty (float): Stiffness of the collision interaction. Defaults to 1000.0.
            max_contact_pairs (int): Maximum number of contact pairs to detect. Defaults to 10000.
            friction (float): Friction coefficient. Defaults to 0.5.
        """
        self._pending_collisions = (collision_particle_radius, detection_ratio, impenetrable_barrier_ratio, collision_penalty, max_contact_pairs, friction)

    def add_simplicits_object_boundary_condition(self, obj_idx, name, fcn, bdry_penalty=10000.0, pinned_x=None):
        r"""Add boundary conditions to the Simplicits scene.

        Wraps SimplicitsScene.set_object_boundary_condition() for a specific object.
        The call is deferred until finalize().

        Args:
            obj_idx (int): Id of the object.
            name (str): Boundary condition name.
            fcn (Callable): Function defining which indices the boundary condition applies to; returns a boolean
                :math:`(n,)` vector where boundary indices are True.
            bdry_penalty (float): Boundary condition penalty coefficient.
            pinned_x (torch.Tensor or None): Pinned positions for the boundary. If None, positions are taken
                from the current object positions.
        """
        self._pending_boundary_conditions.append((obj_idx, name, fcn, bdry_penalty, pinned_x))

    def finalize(self, device='cuda', requires_grad=False, **kwargs) -> SimplicitsModel:
        r"""Finalize and build the SimplicitsModel instance.

        Registers Simplicits particles with Newton, finalizes the base model, and automatically
        enables soft-rigid body collisions if objects exist.

        Args:
            device (str or torch.device): Target device for the model.
            requires_grad (bool): Whether gradients are required. Defaults to False. If True, a warning is
                issued; Simplicits is not differentiable yet and finalize still runs with ``requires_grad=False``.
            **kwargs: Forwarded to :meth:`newton.ModelBuilder.finalize` (e.g. validation skips).

        Returns:
            (SimplicitsModel): Fully constructed model ready for simulation.
        """
        if requires_grad:
            warnings.warn(
                "Simplicits is not differentiable yet; SimplicitsModelBuilder.finalize() proceeds with "
                "requires_grad=False.",
                UserWarning,
                stacklevel=2,
            )

        # Truncate any Simplicits particles appended by a previous finalize() call,
        # so each call starts from the same Newton-builder base state.
        if hasattr(self, '_simplicits_base_particle_count'):
            base = self._simplicits_base_particle_count
            self.particle_q      = self.particle_q[:base]
            self.particle_qd     = self.particle_qd[:base]
            self.particle_mass   = self.particle_mass[:base]
            self.particle_radius = self.particle_radius[:base]
            self.particle_flags  = self.particle_flags[:base]
            self.particle_world  = self.particle_world[:base]

        model = SimplicitsModel(device)

        for sim_object, num_qp, init_transform, is_kinematic, renderable_pts in self._pending_objects:
            obj_id = model.simplicits_scene.add_object(
                sim_object, num_qp, init_transform, is_kinematic, renderable_pts)
            logging.info(f"Added Simplicits object with ID {obj_id}")

        has_simplicits_objects = len(self._pending_objects) > 0

        if has_simplicits_objects:
            # Set gravity
            acc_gravity = torch.zeros(3)
            acc_gravity[self.up_axis.value] = -self.gravity
            model.simplicits_scene.set_scene_gravity(acc_gravity)

            # Apply deferred boundary conditions
            for (obj_idx, name, fcn, bdry_penalty, pinned_x) in self._pending_boundary_conditions:
                model.simplicits_scene.set_object_boundary_condition(obj_idx, name, fcn, bdry_penalty, pinned_x)

            # Apply deferred collisions
            if self._pending_collisions is not None:
                model.simplicits_scene.enable_collisions(*self._pending_collisions)

            # Get particle data from SimplicitsScene
            sim_pts = model.simplicits_scene.sim_pts.numpy()
            sim_masses = model.simplicits_scene.sim_M.values.numpy()[::3]
            assert sim_masses.shape[0] == sim_pts.shape[0]

            # Capture base count once (before any Simplicits particles are added)
            if not hasattr(self, '_simplicits_base_particle_count'):
                self._simplicits_base_particle_count = len(self.particle_q)

            # Store the starting index of Simplicits particles
            self.simplicits_particle_start = len(self.particle_q)

            # Add Simplicits particles to Newton's global particle arrays.
            # SimplicitsSolver updates these particles each step via sim_z;
            # Newton's own cloth/particle solvers should not be applied to them.
            self.add_particles(
                pos=[p for p in sim_pts],
                vel=[(0.0, 0.0, 0.0)] * sim_pts.shape[0],
                mass=sim_masses,
                radius=[0.05] * sim_pts.shape[0],
            )

            self.simplicits_particle_end = len(self.particle_q)

        # Forward all ModelBuilder.finalize keyword-only options (skip_* validations, etc.).
        base_m = super().finalize(device, requires_grad=False, **kwargs)

        # copy all attributes from base model
        model.__dict__.update(base_m.__dict__)

        if has_simplicits_objects:
            # Store Simplicits particle indices in the model for easy access
            model.simplicits_particle_start = self.simplicits_particle_start
            model.simplicits_particle_end = self.simplicits_particle_end

            # Auto-add collisions with Newton rigid objects if not already registered
            if "newton_soft_collisions" not in model.simplicits_scene.force_dict["pt_wise"]:
                model.simplicits_scene.force_dict["pt_wise"]["newton_soft_collisions"] = {
                    "object": SimplicitsParticleNewtonShapeSoftContact(model,
                                                wp.ones_like(
                                                    model.simplicits_scene.sim_vols),
                                                dt=model.simplicits_scene.timestep,
                                                friction_use_lagged_body_contact_force_norm=False),
                    "coeff": 0.001,
                }

        return model
