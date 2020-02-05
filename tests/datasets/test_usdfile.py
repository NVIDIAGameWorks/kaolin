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

import pytest
import shutil
from pathlib import Path

from kaolin.datasets.usdfile import USDMeshes


def test_usd_meshes():
    fpath = './tests/model.usd'
    cache_dir = './tests/datasets_eval/USDMeshes/'
    usd_dataset = USDMeshes(usd_filepath=fpath, cache_dir=cache_dir)
    assert len(usd_dataset) == 1

    # test caching
    assert len(list(Path(cache_dir).glob('**/*.p'))) == 1
    shutil.rmtree('tests/datasets_eval/USDMeshes')

# Tests below must be run with KitchenSet dataset

# def test_usd_meshes():
#     fpath = 'data/Kitchen_set/Kitchen_set.usd'
#     cache_dir = './tests/datasets_eval/USDMeshes/'
#     usd_dataset = USDMeshes(usd_filepath=fpath, cache_dir=cache_dir)
#     assert len(usd_dataset) == 740

#     # test caching
#     assert len(list(Path(cache_dir).glob('**/*.npz'))) == 740
