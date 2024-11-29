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

from kaolin.rep import SurfaceMesh
from kaolin.io.dataset import KaolinDatasetItem
from kaolin.io.shrec import SHREC16

SHREC16_PATH = os.getenv('KAOLIN_TEST_SHREC16_PATH')
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
@pytest.mark.skipif(SHREC16_PATH is None,
                    reason="'KAOLIN_TEST_SHREC16_PATH' environment variable is not set.")
@pytest.mark.parametrize('categories', ALL_CATEGORIES)
@pytest.mark.parametrize('split', ['train', 'val', 'test'])
@pytest.mark.parametrize('use_transform', [True, False])
@pytest.mark.parametrize('output_dict', [True, False])
class TestSHREC16(object):

    @pytest.fixture(autouse=True)
    def transform(self, output_dict, use_transform):
        if use_transform:
            if output_dict:
                def transform(inputs):
                    outputs = copy.copy(inputs)
                    outputs['mesh'] = SurfaceMesh(
                        vertices=outputs['mesh'].vertices + 1.,
                        faces=outputs['mesh'].faces,
                        uvs=outputs['mesh'].uvs,
                        face_uvs_idx=outputs['mesh'].face_uvs_idx,
                        materials=outputs['mesh'].materials,
                        material_assignments=outputs['mesh'].material_assignments,
                        normals=outputs['mesh'].normals,
                        face_normals_idx=outputs['mesh'].face_normals_idx
                    )
                    return outputs
                return transform
            else:
                def transform(inputs):
                    outputs = KaolinDatasetItem(
                        data=SurfaceMesh(
                            vertices=inputs.data.vertices + 1.,
                            faces=inputs.data.faces,
                            uvs=inputs.data.uvs,
                            face_uvs_idx=inputs.data.face_uvs_idx,
                            materials=inputs.data.materials,
                            material_assignments=inputs.data.material_assignments,
                            normals=inputs.data.normals,
                            face_normals_idx=inputs.data.face_normals_idx
                        ),
                        attributes=inputs.attributes)
                    return outputs
                return transform
        else:
            return None

    @pytest.fixture(autouse=True)
    def shrec16_dataset(self, categories, split, transform, output_dict):
        return SHREC16(root=SHREC16_PATH,
                       categories=categories,
                       split=split,
                       transform=transform,
                       output_dict=output_dict)

    @pytest.mark.parametrize('index', [0, -1])
    def test_basic_getitem(self, shrec16_dataset, index, split, output_dict):
        assert len(shrec16_dataset) > 0

        if index == -1:
            index = len(shrec16_dataset) - 1

        item = shrec16_dataset[index]
        if output_dict:
            data = item['mesh']
            attributes = item
        else:
            data = item.data
            attributes = item.attributes
        assert isinstance(data, SurfaceMesh)
        assert isinstance(attributes, dict)

        assert isinstance(data.vertices, torch.Tensor)
        assert len(data.vertices.shape) == 2
        assert data.vertices.shape[1] == 3
        assert isinstance(data.faces, torch.Tensor)
        assert len(data.faces.shape) == 2

        assert isinstance(attributes['name'], str)
        assert isinstance(attributes['path'], Path)
        
        if split == "test":
            assert attributes['synset'] is None
            assert attributes['labels'] is None
        else:
            assert isinstance(attributes['synset'], str)
            assert isinstance(attributes['labels'], list)
    
    @pytest.mark.parametrize('index', [-1, -2])
    def test_neg_index(self, shrec16_dataset, index, output_dict):

        assert len(shrec16_dataset) > 0

        gt_item = shrec16_dataset[len(shrec16_dataset) + index]
        item = shrec16_dataset[index]
        if output_dict:
            data = item['mesh']
            attributes = item
            gt_data = gt_item['mesh']
            gt_attributes = gt_item
        else:
            data = item.data
            attributes = item.attributes
            gt_data = gt_item.data
            gt_attributes = gt_item.attributes

        assert torch.equal(data.vertices, gt_data.vertices)
        assert torch.equal(data.faces, gt_data.faces)

        assert attributes['name'] == gt_attributes['name']
        assert attributes['path'] == gt_attributes['path']
        assert attributes['synset'] == gt_attributes['synset']
