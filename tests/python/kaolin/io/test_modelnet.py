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

import os

import pytest
import torch

from kaolin.io.off import return_type
from kaolin.io.modelnet import ModelNet

MODELNET_PATH = '/home/jiehanw/Downloads/ModelNet10'
MODELNET_TEST_CATEGORY_LABELS = ['bathtub']
MODELNET_TEST_CATEGORY_LABELS_2 = ['desk']
MODELNET_TEST_CATEGORY_LABELS_MULTI = ['bathtub', 'desk']

ALL_CATEGORIES = [
    MODELNET_TEST_CATEGORY_LABELS,
    MODELNET_TEST_CATEGORY_LABELS_2,
    MODELNET_TEST_CATEGORY_LABELS_MULTI,
]

# Skip test in a CI environment
@pytest.mark.skipif(os.getenv('CI') == 'true', reason="CI does not have dataset")
@pytest.mark.parametrize('categories', ALL_CATEGORIES)
@pytest.mark.parametrize('split', ['train', 'test'])
@pytest.mark.parametrize('index', [0, -1])
class TestModelNet(object):

    @pytest.fixture(autouse=True)
    def modelnet_dataset(self, categories, split):
        return ModelNet(root=MODELNET_PATH,
                        categories=categories,
                        split=split)

    def test_basic_getitem(self, modelnet_dataset, index):
        assert len(modelnet_dataset) > 0

        if index == -1:
            index = len(modelnet_dataset) - 1

        item = modelnet_dataset[index]
        data = item.data
        attributes = item.attributes
        assert isinstance(data, return_type)
        assert isinstance(attributes, dict)

        assert isinstance(data.vertices, torch.Tensor)
        assert len(data.vertices.shape) == 2
        assert data.vertices.shape[1] == 3
        assert isinstance(data.faces, torch.Tensor)
        assert len(data.faces.shape) == 2

        assert isinstance(attributes['name'], str)
        assert isinstance(attributes['path'], Path)
        assert isinstance(attributes['label'], str)

        assert isinstance(data.face_colors, torch.Tensor)
    
