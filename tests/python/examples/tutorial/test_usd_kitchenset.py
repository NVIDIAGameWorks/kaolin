# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
import shutil
import zipfile
import urllib.request

import pytest


@pytest.fixture(scope='module')
def kitchen_set_dir():
    # Create temporary output directory
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_data')
    os.makedirs(data_dir, exist_ok=True)
    kitchen_set_path = os.path.join(data_dir, 'kitchenset.zip')

    kitchen_set_url = 'http://graphics.pixar.com/usd/files/Kitchen_set.zip'
    urllib.request.urlretrieve(kitchen_set_url, kitchen_set_path)
    with zipfile.ZipFile(kitchen_set_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    yield os.path.join(data_dir, 'Kitchen_set')
    shutil.rmtree(data_dir)


@pytest.fixture(scope='module')
def out_dir():
    # Create temporary output directory
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_out')
    os.makedirs(out_dir, exist_ok=True)
    yield out_dir
    shutil.rmtree(out_dir)


class TestUsdKitchenSet:
    def test_runs(self, kitchen_set_dir, out_dir):
        args = f'--kitchen_set_dir={kitchen_set_dir} --output_dir={out_dir}'
        os.system(f'python examples/tutorial/usd_kitchenset.py {args}')

        # Confirm that there are 426 meshes exported
        assert len(os.listdir(out_dir)) == 426
