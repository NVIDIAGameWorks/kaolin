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
from newton.solvers import SolverBase
from newton import Contacts, Control

from kaolin.experimental.newton.model import SimplicitsModel
from kaolin.experimental.newton.state import SimplicitsState



__all__ = [
    'SimplicitsSolver',
]

class SimplicitsSolver(SolverBase):
    r"""Integrates Simplicits's solver with other newton solvers.

    The :attr:`model` attribute (inherited from the base solver) holds the
    SimplicitsModel instance containing scene and simulation data.
    """
    model: SimplicitsModel

    def __init__(self, model: SimplicitsModel):
        r"""Initialize the solver with a SimplicitsModel.

        Args:
            model (SimplicitsModel): The SimplicitsModel to simulate.
        """
        super().__init__(model)

    def step(self, state_in: SimplicitsState, state_out: SimplicitsState, control: Control, contacts: Contacts, dt: float) -> SimplicitsState:
        r"""Advance simulation by one timestep.

        If state includes simplicits DOFs, copies the state to Simplicits scene and runs
        the simulation in the Simplicits scene with collision handling. Only updates
        the Simplicits particle subset of the state's particle arrays.

        Args:
            state_in (SimplicitsState): Current simulation state.
            state_out (SimplicitsState): Output state to populate.
            control (Control): Control inputs (unused by Simplicits).
            contacts (Contacts): Contact information for soft-rigid collisions.
            dt (float): Timestep size in seconds.

        Returns:
            (SimplicitsState): Updated state_out with new simulation state.
        """
        if state_in.sim_z is not None:
            wp.copy(dest=self.model.simplicits_scene.sim_z,
                    src=state_in.sim_z)
            wp.copy(dest=self.model.simplicits_scene.sim_z_dot,
                    src=state_in.sim_z_dot)
            self.model.simplicits_scene.timestep = dt

            # Collisions with newton objects and floors are handled by this force.
            if "newton_soft_collisions" in self.model.simplicits_scene.force_dict["pt_wise"] and contacts is not None:
                self.model.simplicits_scene.force_dict["pt_wise"]["newton_soft_collisions"]["object"]._set_state(
                    state_in)
                self.model.simplicits_scene.force_dict["pt_wise"]["newton_soft_collisions"]["object"]._set_contacts(
                    contacts)

            self.model.simplicits_scene.run_sim_step()

            # Update Simplicits DOFs
            wp.copy(dest=state_out.sim_z, src=self.model.simplicits_scene.sim_z)
            wp.copy(dest=state_out.sim_z_dot, src=self.model.simplicits_scene.sim_z_dot)

            # Update only the Simplicits particle subset in the state's particle arrays
            sim_particle_q = self.model.sim_z_to_full(state_out.sim_z)
            sim_particle_qd = self.model.sim_z_dot_to_full(state_out.sim_z_dot)

            # Copy Simplicits particles to the correct range in state arrays
            start = self.model.simplicits_particle_start
            end = self.model.simplicits_particle_end
            wp.copy(dest=state_out.particle_q, src=sim_particle_q,
                    dest_offset=start, src_offset=0, count=end-start)
            wp.copy(dest=state_out.particle_qd, src=sim_particle_qd,
                    dest_offset=start, src_offset=0, count=end-start)

        return state_out
