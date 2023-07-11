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
import random

from kaolin.rep import SurfaceMesh
from kaolin.io.dataset import KaolinDatasetItem
from kaolin.io import shapenet
from kaolin.utils.testing import contained_torch_equal

SHAPENETV1_PATH = os.getenv('KAOLIN_TEST_SHAPENETV1_PATH')
SHAPENETV2_PATH = os.getenv('KAOLIN_TEST_SHAPENETV2_PATH')
SHAPENET_TEST_CATEGORY_SYNSETS = ['02933112']
SHAPENET_TEST_CATEGORY_LABELS = ['dishwasher']
SHAPENET_TEST_CATEGORY_MULTI = ['mailbox', '04379243']

ALL_CATEGORIES = [
    SHAPENET_TEST_CATEGORY_SYNSETS,
    SHAPENET_TEST_CATEGORY_LABELS,
    SHAPENET_TEST_CATEGORY_MULTI
]

@pytest.mark.parametrize('version', [
    pytest.param('v1', marks=pytest.mark.skipif(
        SHAPENETV1_PATH is None,
        reason="'KAOLIN_TEST_SHAPENETV1_PATH' environment variable is not set."
    )),
    pytest.param('v2', marks=pytest.mark.skipif(
        SHAPENETV2_PATH is None,
        reason="'KAOLIN_TEST_SHAPENETV2_PATH' environment variable is not set."
    ))
])
@pytest.mark.parametrize('categories', ALL_CATEGORIES)
@pytest.mark.parametrize('with_materials', [True, False])
@pytest.mark.parametrize('output_dict', [True, False])
@pytest.mark.parametrize('use_transform', [True, False])
class TestShapeNet(object):

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
    def shapenet_dataset(self, version, categories, with_materials, transform, output_dict):
        if version == 'v1':
            ds = shapenet.ShapeNetV1(root=SHAPENETV1_PATH,
                                     categories=categories,
                                     train=True,
                                     split=0.7,
                                     with_materials=with_materials,
                                     transform=transform,
                                     output_dict=output_dict)
        elif version == 'v2':
            ds = shapenet.ShapeNetV2(root=SHAPENETV2_PATH,
                                     categories=categories,
                                     train=True,
                                     split=0.7,
                                     with_materials=with_materials,
                                     transform=transform,
                                     output_dict=output_dict)
        else:
            raise ValueError(f"version {version} not recognized")
        return ds

    @pytest.mark.parametrize('index', [0, -1, None, None])
    def test_basic_getitem(self, shapenet_dataset, index, with_materials, output_dict):
        if index is None:
            index = random.randint(0, len(shapenet_dataset) - 1)
        assert len(shapenet_dataset) > 0

        item = shapenet_dataset[index]
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
        assert data.vertices.shape[0] > 0
        assert data.vertices.shape[1] == 3

        assert isinstance(data.faces, torch.LongTensor)
        assert len(data.faces.shape) == 2
        assert data.faces.shape[0] > 0
        assert data.faces.shape[1] == 3

        if with_materials:
            assert isinstance(data.uvs, torch.Tensor)
            assert len(data.uvs.shape) == 2
            assert data.uvs.shape[1] == 2

            assert isinstance(data.face_uvs_idx, torch.LongTensor)
            assert data.face_uvs_idx.shape == data.faces.shape
            assert isinstance(data.materials, list)
            assert len(data.materials) > 0
            assert isinstance(data.material_assignments, torch.ShortTensor)
            assert list(data.material_assignments.shape) == [data.faces.shape[0]]
        else:
            assert data.uvs is None
            assert data.face_uvs_idx is None
            assert data.materials is None
            assert data.material_assignments is None

        assert isinstance(attributes['name'], str)
        assert isinstance(attributes['path'], Path)
        assert isinstance(attributes['synset'], str)
        assert isinstance(attributes['labels'], list)

    @pytest.mark.parametrize('index', [-1, -2])
    def test_neg_index(self, shapenet_dataset, index, output_dict):

        assert len(shapenet_dataset) > 0

        gt_item = shapenet_dataset[len(shapenet_dataset) + index]
        if output_dict:
            gt_data = gt_item['mesh']
        else:
            gt_data = gt_item.data

        item = shapenet_dataset[index]
        if output_dict:
            data = item['mesh']
        else:
            data = item.data

        contained_torch_equal(item, gt_item)

    def test_test_split(self, shapenet_dataset, with_materials, output_dict,
                        version, categories):
        if version == 'v1':
            test_dataset = shapenet.ShapeNetV1(root=SHAPENETV1_PATH,
                                               categories=categories,
                                               train=False,
                                               split=0.7,
                                               with_materials=with_materials,
                                               output_dict=output_dict)
        else:
            test_dataset = shapenet.ShapeNetV2(root=SHAPENETV2_PATH,
                                               categories=categories,
                                               train=False,
                                               split=0.7,
                                               with_materials=with_materials,
                                               output_dict=output_dict)
        train_item = shapenet_dataset[0]
        test_item = test_dataset[0]
        if output_dict:
            train_attributes = train_item
            test_attributes = test_item
        else:
            train_attributes = train_item.attributes
            test_attributes = test_item.attributes
        assert train_attributes['name'] != test_attributes['name']
        assert test_attributes['name'] not in shapenet_dataset.names
