# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
# Kornia components Copyright (c) 2019 Kornia project authors
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

import torch
import pytest


# From kornia
# https://github.com/arraiyopensource/kornia/
def get_test_devices():
    """Creates a list of strings indicating available devices to test on.
    Checks for CUDA devices, primarily. Assumes CPU is always available.

    Return:
        list (str): list of device names

    """

    # Assumption: CPU is always available
    devices = ['cpu']

    if torch.cuda.is_available():
        devices.append('cuda')

    return devices


# Setup devices to run unit tests
TEST_DEVICES = get_test_devices()


@pytest.fixture()
def device_type(request):
    typ = request.config.getoption('--typetest')
    return typ
