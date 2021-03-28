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

import torch

from kaolin.io import obj

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../samples/')

# TODO(cfujitsang): Add sanity test over a dataset like ModelNet

class TestLoadOff:
    @pytest.fixture(autouse=True)
    def expected_vertices(self):
        return torch.FloatTensor([
            [-0.1, -0.1, -0.1],
            [0.1, -0.1, -0.1],
            [-0.1, 0.1, -0.1],
            [0.1, 0.1, -0.1],
            [-0.1, -0.1, 0.1],
            [0.1, -0.1, 0.1]
        ])

    @pytest.fixture(autouse=True)
    def expected_faces(self):
        return torch.LongTensor([
            [0, 1, 3, 2],
            [1, 0, 4, 5]
        ])

    @pytest.fixture(autouse=True)
    def expected_face_colors(self):
        return torch.LongTensor([
            [128, 128, 128],
            [0, 0, 255]
        ])

    @pytest.mark.parametrize('with_face_colors', [False, True])
    def test_import_mesh(self, with_face_colors, expected_vertices, expected_faces,
                         expected_face_colors):
        outputs = obj.import_mesh(os.path.join(SIMPLE_DIR, 'model.off'),
                                  with_face_colors=with_face_colors)
        assert torch.equal(outputs.vertices, expected_vertices)
        assert torch.equal(outputs.faces, expected_faces)
        if with_face_colors:
            assert torch.equal(outputs.face_colors, expected_face_colors)
        else:
            assert outputs.face_colors is None
