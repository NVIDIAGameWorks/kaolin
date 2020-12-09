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
import pytest
import shutil
import torch

import kaolin

from kaolin.utils.testing import tensor_info


@pytest.fixture(scope='module')
def out_dir():
    # Create temporary output directory
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_viz_out')
    os.makedirs(out_dir, exist_ok=True)
    yield out_dir
    shutil.rmtree(out_dir)  # Note: comment to keep output directory


@pytest.fixture(scope='module')
def obj_paths():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    samples_dir = os.path.join(cur_dir, os.pardir, os.pardir, os.pardir, 'samples')
    return [os.path.join(samples_dir, 'rocket.obj'),
            os.path.join(samples_dir, 'model.obj')]


class TestVisualizeMain:
    def test_runs(self, obj_paths, out_dir):
        objs = ','.join(obj_paths)

        # Check that the main function runs
        # Note: to run and capture output do:
        # pytest --capture=tee-sys tests/python/examples/
        args = '--skip_normalization --test_objs={} --output_dir={}'.format(objs, out_dir)
        os.system('python examples/tutorial/visualize_main.py {}'.format(args))

        # Spot check one of the outputs
        for i in range(len(obj_paths)):
            expected = kaolin.io.obj.import_mesh(obj_paths[i])
            expected_usd = os.path.join(out_dir, 'output', 'mesh_%d.usd' % i)
            assert os.path.exists(expected_usd)
            actual_start = kaolin.io.usd.import_mesh(expected_usd, time=0)
            actual_end = kaolin.io.usd.import_mesh(expected_usd, time=1000)

            assert torch.allclose(expected.vertices, actual_end.vertices, rtol=1e-03)
            assert not torch.allclose(expected.vertices, actual_start.vertices)
