
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

"""Tests for SimplicitsState.

Note: Fixtures from conftest.py are automatically available!
"""

from kaolin.experimental.newton.state import SimplicitsState

# Test empty state
# @pytest.mark.parametrize('device', ['cuda', 'cpu'])
# @pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_state():
    state = SimplicitsState()
    assert state.sim_z is None
    assert state.sim_z_prev is None
    assert state.sim_z_dot is None
