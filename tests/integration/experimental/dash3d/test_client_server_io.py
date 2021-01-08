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

import numpy as np
import os
import pytest
import shutil
import torch

import kaolin

from kaolin.utils.testing import tensor_info

from kaolin.experimental.dash3d.util import meshes_to_binary
from kaolin.experimental.dash3d.util import point_clouds_to_binary

@pytest.fixture(scope='module')
def out_dir():
    # Create temporary output directory
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_out')
    os.makedirs(out_dir, exist_ok=True)
    yield out_dir
    shutil.rmtree(out_dir)  # Note: comment to keep output directory


@pytest.fixture(scope='module')
def meshes():
    vertices0 = np.array([[1.0, 2.0, 3.0],
                          [10.0, 20.0, 30.0],
                          [2.0, 4.0, 6.0],
                          [15.0, 25.0, 35.0]], dtype=np.float32)
    faces0 = np.array([[0, 1, 2],
                       [2, 1, 3]], dtype=np.int32)
    vertices1 = np.arange(0, 300).reshape((-1, 3))
    faces1 = np.stack([np.arange(0, 100),
                       np.mod(np.arange(0, 100) + 1, 100),
                       np.mod(np.arange(0, 100) + 2, 100)]).astype(np.int32).reshape((-1, 3))
    vertices2 = np.random.random((1, 9000)).reshape((-1, 3))
    faces2 = np.stack([np.mod(np.arange(0, 6000), 1000),
                       np.ones((6000,)),
                       np.random.randint(0, 2999 + 1, (6000,))]).astype(np.int32).reshape((-1, 3))
    return {"faces": [faces0, faces1, faces2],
            "vertices": [vertices0, vertices1, vertices2]}


@pytest.fixture(scope='module')
def pointclouds():
    pts0 = np.array([[1.0, 2.0, 3.0],
                     [10.0, 20.0, 30.0],
                     [2.0, 4.0, 6.0],
                     [15.0, 25.0, 35.0]], dtype=np.float32)
    pts1 = np.arange(0, 300).astype(np.float32).reshape((-1, 3))
    pts2 = np.random.random((1, 9000)).astype(np.float32).reshape((-1, 3))
    return {"positions": [pts0, pts1, pts2]}


class TestBinaryEncoding:
    def test_server_client_binary_compatibility(self, meshes, pointclouds, out_dir):
        # Encode and write mesh0+mesh1 and mesh2 to binary files
        binstr = meshes_to_binary(meshes['vertices'][0:2], meshes['faces'][0:2])
        with open(os.path.join(out_dir, 'meshes0_1.bin'), 'wb') as f:
            f.write(binstr)
        binstr = meshes_to_binary([meshes['vertices'][2]], [meshes['faces'][2]])
        with open(os.path.join(out_dir, 'meshes2.bin'), 'wb') as f:
            f.write(binstr)

        # Encode and write ptcloud0+ptcloud1 and pointcloud2 to binary files
        binstr = point_clouds_to_binary(pointclouds['positions'][0:2])
        with open(os.path.join(out_dir, 'clouds0_1.bin'), 'wb') as f:
            f.write(binstr)
        binstr = point_clouds_to_binary([pointclouds['positions'][2]])
        with open(os.path.join(out_dir, 'clouds2.bin'), 'wb') as f:
            f.write(binstr)

        # Execute javascript test that checks that these are parsed correctly
        js_test = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test_binary_parse.js')
        os.system('npx mocha {}'.format(js_test))  # TODO: will npx work for everyone?
