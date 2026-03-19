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

import warp as wp
import kaolin

import newton

from kaolin.experimental.newton.state import SimplicitsState


__all__ = [
    'SimplicitsModel',
]

class SimplicitsModel(newton.Model):
    r"""Extends Newton's Model for Simplicits physics simulations.

    Attributes:
        simplicits_scene (kaolin.physics.simplicits.SimplicitsScene): Simplicits scene instance.
    """
    def __init__(self, device = None):
        r"""Initialize the model with an empty Simplicits scene.

        Args:
            device (wp.device or str or None): Target device for simulation (GPU).
        """
        super().__init__(device)

        # Empty simplicits scene ( currently with default values )
        self.simplicits_scene = kaolin.physics.simplicits.SimplicitsScene()
        # Index range [simplicits_particle_start, simplicits_particle_end) into
        # Newton's global particle arrays (particle_q, particle_qd).
        # SimplicitsSolver owns their dynamics: each step it converts the
        # reduced DOFs (sim_z) back to world-space positions and writes them
        # into this slice of the state's particle arrays, so that Newton
        # collision/contact queries see up-to-date Simplicits particle positions.
        # Set by SimplicitsModelBuilder.finalize() after add_particles().
        self.simplicits_particle_start = None
        self.simplicits_particle_end = None

    def state(self, requires_grad: bool | None = None) -> SimplicitsState:
        r"""Create a new simulation state initialized from the Simplicits scene.

        Args:
            requires_grad (bool or None): Whether to enable gradient tracking for state variables.

        Returns:
            (SimplicitsState): State with initialized position, velocity, and DOF arrays.
        """
        s = super().state(requires_grad=requires_grad)

        # Scene dofs should not be None.
        # Object added and dofs should be set before calling state.
        if self.simplicits_scene.sim_z is not None:
            sim_dof = self.simplicits_scene.sim_z.shape[0]

            s.sim_z = wp.clone(self.simplicits_scene.sim_z)
            s.sim_z_dot = wp.zeros(sim_dof, dtype=wp.float32, device=self.device)
            s.sim_z_prev = wp.zeros(sim_dof, dtype=wp.float32, device=self.device)

            # Copy Simplicits particles to the correct range in state arrays
            # These are set in the builder.finalize() method.
            if self.simplicits_particle_start is not None and self.simplicits_particle_end is not None:
                sim_particle_q = self.sim_z_to_full(s.sim_z)
                sim_particle_qd = wp.array(self.simplicits_scene.sim_B @ s.sim_z_dot, dtype=wp.vec3)
                start = self.simplicits_particle_start
                end = self.simplicits_particle_end
                wp.copy(dest=s.particle_q, src=sim_particle_q,
                        dest_offset=start, src_offset=0, count=end-start)
                wp.copy(dest=s.particle_qd, src=sim_particle_qd,
                        dest_offset=start, src_offset=0, count=end-start)

        else:
            s.sim_z = None
            s.sim_z_dot = None
            s.sim_z_prev = None
        return s

    def sim_z_to_full(self, sim_z: wp.array):
        r"""Convert reduced coordinates to full particle positions.

        Args:
            sim_z (wp.array): Reduced coordinate state vector.

        Returns:
            (wp.array(dtype=wp.vec3)): Full particle positions, or empty array if scene uninitialized.
        """
        if self.simplicits_scene.sim_z and self.simplicits_scene.sim_z_dot is not None:
            return wp.array(self.simplicits_scene.sim_B @ sim_z, dtype=wp.vec3) + self.simplicits_scene.sim_pts
        else:
            return wp.zeros((0,), dtype=wp.vec3)

    def sim_z_dot_to_full(self, sim_z_dot: wp.array):
        r"""Convert reduced coordinate velocities to full particle velocities.

        Args:
            sim_z_dot (wp.array): Reduced coordinate velocity vector.

        Returns:
            (wp.array(dtype=wp.vec3)): Full particle velocities, or empty array if scene uninitialized.
        """
        if self.simplicits_scene.sim_z_dot is not None:
            return wp.array(self.simplicits_scene.sim_B @ sim_z_dot, dtype=wp.vec3)
        else:
            return wp.zeros((0,), dtype=wp.vec3)
