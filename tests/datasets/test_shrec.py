# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import pytest

import kaolin as kal

SHREC16_ROOT = "/data/SHREC16/"
CACHE_DIR = "tests/datasets/cache"

# Tests below can only be run is a ShapeNet dataset is available
REASON = "SHREC16 not found at default location: {}".format(SHREC16_ROOT)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.skipif(not os.path.exists(SHREC16_ROOT), reason=REASON)
def test_SHREC16(device):
    models = kal.datasets.SHREC16(
        basedir=SHREC16_ROOT, categories=["ants"], split="test"
    )
    assert len(models) == 4
