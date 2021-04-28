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

from kaolin.io.obj import return_type
from kaolin.io.shrec import SHREC16

SHREC16_PATH = '/data/shrec16'
SHREC16_TEST_CATEGORY_SYNSETS = ['02691156']
SHREC16_TEST_CATEGORY_LABELS = ['airplane']
SHREC16_TEST_CATEGORY_SYNSETS_2 = ['02958343']
SHREC16_TEST_CATEGORY_LABELS_2 = ['car']
SHREC16_TEST_CATEGORY_SYNSETS_MULTI = ['02691156', '02958343']
SHREC16_TEST_CATEGORY_LABELS_MULTI = ['airplane', 'car']

ALL_CATEGORIES = [
    SHREC16_TEST_CATEGORY_SYNSETS,
    SHREC16_TEST_CATEGORY_LABELS,
    SHREC16_TEST_CATEGORY_SYNSETS_2,
    SHREC16_TEST_CATEGORY_LABELS_2,
    SHREC16_TEST_CATEGORY_SYNSETS_MULTI,
    SHREC16_TEST_CATEGORY_LABELS_MULTI,
]


# Skip test in a CI environment
@pytest.mark.skipif(os.getenv('CI') == 'true', reason="CI does not have dataset")
@pytest.mark.parametrize('categories', ALL_CATEGORIES)
@pytest.mark.parametrize('split', ['val'])
class TestSHREC16(object):

    @pytest.fixture(autouse=True)
    def shrec16_dataset(self, categories, split):
        return SHREC16(root=SHREC16_PATH,
                       categories=categories,
                       split=split)

    @pytest.mark.parametrize('index', [0, -1])
    def test_basic_getitem(self, shrec16_dataset, index):
        assert len(shrec16_dataset) > 0

        if index == -1:
            index = len(shrec16_dataset) - 1

        item = shrec16_dataset[index]
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
        assert isinstance(attributes['synset'], str)
        assert isinstance(attributes['label'], str)
