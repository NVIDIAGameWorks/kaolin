# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

import os
import torch
import pytest

import kaolin

if os.getenv('KAOLIN_TEST_NVDIFFRAST', '0') == '0':
    pytest.skip('test is ignored as KAOLIN_TEST_NVDIFFRAST is not set', allow_module_level=True)

class TestNvidiaContext:

    @pytest.mark.parametrize('device', ['cuda:0', 'cuda'])
    @pytest.mark.parametrize('raise_error', [True, False])
    def test_default_nvdiffrast_context(self, device, raise_error):
        import nvdiffrast.torch
        ctx = kaolin.render.mesh.nvdiffrast_context.default_nvdiffrast_context(device, raise_error)
        assert isinstance(ctx, nvdiffrast.torch.RasterizeCudaContext)
