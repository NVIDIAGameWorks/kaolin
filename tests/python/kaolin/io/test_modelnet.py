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
import copy

import pytest
import torch

from kaolin.io.dataset import KaolinDatasetItem
from kaolin.io.off import return_type
from kaolin.io.modelnet import ModelNet

MODELNET_PATH = os.getenv('KAOLIN_TEST_MODELNET_PATH')
MODELNET_TEST_CATEGORY_LABELS = ['bathtub']
MODELNET_TEST_CATEGORY_LABELS_2 = ['desk']
MODELNET_TEST_CATEGORY_LABELS_MULTI = ['bathtub', 'desk']

ALL_CATEGORIES = [
    MODELNET_TEST_CATEGORY_LABELS,
    MODELNET_TEST_CATEGORY_LABELS_2,
    MODELNET_TEST_CATEGORY_LABELS_MULTI,
]

# Skip test in a CI environment
@pytest.mark.skipif(MODELNET_PATH is None,
                    reason="'KAOLIN_TEST_MODELNET_PATH' environment variable is not set.")
@pytest.mark.parametrize('categories', ALL_CATEGORIES)
@pytest.mark.parametrize('split', ['train', 'test'])
@pytest.mark.parametrize('index', [0, -1])
@pytest.mark.parametrize('use_transform', [True, False])
@pytest.mark.parametrize('output_dict', [True, False])
class TestModelNet(object):

    @pytest.fixture(autouse=True)
    def transform(self, output_dict, use_transform):
        if use_transform:
            if output_dict:
                def transform(inputs):
                    outputs = copy.copy(inputs)
                    outputs['mesh'] = return_type(
                        vertices=outputs['mesh'].vertices + 1.,
                        faces=outputs['mesh'].faces,
                        face_colors=outputs['mesh'].face_colors
                    )
                    return outputs
                return transform
            else:
                def transform(inputs):
                    outputs = KaolinDatasetItem(
                        data=return_type(
                            vertices=inputs.data.vertices + 1.,
                            faces=inputs.data.faces,
                            face_colors=inputs.data.face_colors
                        ),
                        attributes=inputs.attributes)
                    return outputs
                return transform
        else:
            return None

    @pytest.fixture(autouse=True)
    def modelnet_dataset(self, categories, split, transform, output_dict):
        return ModelNet(root=MODELNET_PATH,
                        categories=categories,
                        split=split,
                        transform=transform,
                        output_dict=output_dict)

    def test_basic_getitem(self, modelnet_dataset, index, output_dict):
        assert len(modelnet_dataset) > 0

        if index == -1:
            index = len(modelnet_dataset) - 1

        item = modelnet_dataset[index]
        if output_dict:
            data = item['mesh']
            attributes = item
        else:
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
    
