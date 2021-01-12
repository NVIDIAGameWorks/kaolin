# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
from pathlib import Path

import pytest
import torch

from kaolin.datasets.shapenet import ShapeNet
from kaolin.io.obj import ObjMesh

SHAPENET_PATH = '/data/ShapeNet'
SHAPENET_TEST_CATEGORY_SYNSETS = ['03001627']
SHAPENET_TEST_CATEGORY_LABELS = ['chair']


class TestShapeNet(object):

    @pytest.fixture(autouse=True)
    def shapenet_dataset(self):
        return ShapeNet(root=SHAPENET_PATH,
                        categories=SHAPENET_TEST_CATEGORY_SYNSETS,
                        train=True, split=1.0)

    def test_basic_getitem(self, shapenet_dataset):
        assert len(shapenet_dataset) > 0
        item = shapenet_dataset[0]
        data = item.data
        attributes = item.attributes
        assert isinstance(data, ObjMesh)
        assert isinstance(attributes, dict)

        assert isinstance(data.vertices, torch.Tensor)
        assert len(data.vertices.shape) == 2
        assert data.vertices.shape[1] == 3
        assert isinstance(data.faces, torch.Tensor)
        assert len(data.faces.shape) == 2

        assert isinstance(attributes['name'], str)
        assert isinstance(attributes['path'], Path)
        assert isinstance(attributes['synset'], str)
        assert isinstance(attributes['label'], str)

    def test_init_by_labels(self):
        dataset = ShapeNet(root=SHAPENET_PATH,
                           categories=SHAPENET_TEST_CATEGORY_LABELS,
                           train=True, split=1.0)

        assert isinstance(dataset[0].data, ObjMesh)
        assert isinstance(dataset[0].attributes, dict)
