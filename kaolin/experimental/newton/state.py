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

import newton

__all__ = [
    'SimplicitsState',
]

class SimplicitsState(newton.State):
    r"""Extended Newton state for flattened Simplicits DOFs.

    Attributes:
        sim_z (wp.array or None): Flattened dofs with shape :math:`(3*4*\text{num_handles}, )`.
        sim_z_prev (wp.array or None): Flattened dofs with shape :math:`(3*4*\text{num_handles},)`.
        sim_z_dot (wp.array or None): Flattened dofs with shape :math:`(3*4*\text{num_handles},)`.
    """
    def __init__(self):
        r"""Initialize a SimplicitsState with None values for simulation variables (no arguments)."""
        super().__init__()

        self.sim_z = None
        self.sim_z_prev = None
        self.sim_z_dot = None
