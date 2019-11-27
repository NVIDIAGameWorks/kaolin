# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pytest
import shutil
import torch

import kaolin as kal
import kaolin.transforms.transforms as tfs


MODELNET_ROOT = 'data/ModelNet10/'
CACHE_DIR = 'tests/datasets/cache'


# Tests below can only be run is a ShapeNet dataset is available
REASON = 'ShapeNet not found at default location: {}'.format(MODELNET_ROOT)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.skipif(not os.path.exists(MODELNET_ROOT), reason=REASON)
def test_ModelNet(device): 
    models = kal.datasets.ModelNet(basedir=MODELNET_ROOT, categories=['bathtub'], split='test')

    assert len(models) == 50
    for obj, category in models: 
        assert category.item() == 0
        assert isinstance(obj, kal.rep.Mesh)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.skipif(not os.path.exists(MODELNET_ROOT), reason=REASON)
def test_ModelNetPointCloud(device): 
    transform = tfs.Compose([
        tfs.TriangleMeshToPointCloud(num_samples=32),
        tfs.NormalizePointCloud()
    ])
    models = kal.datasets.ModelNet(basedir=MODELNET_ROOT, categories=['bathtub'], split='test', transform=transform)

    assert len(models) == 50
    for obj, category in models: 
        assert category.item() == 0
        assert isinstance(obj, torch.Tensor)
        assert obj.size(0) == 32


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.skipif(not os.path.exists(MODELNET_ROOT), reason=REASON)
def test_ModelNetVoxels(device): 
    models = kal.datasets.ModelNetVoxels(basedir=MODELNET_ROOT, cache_dir=CACHE_DIR, categories=['bathtub'], split='test', resolutions=[30])

    assert len(models) == 50
    for obj in models: 
        assert obj['attributes']['category'] == 'bathtub'
        assert set(obj['data']['30'].shape) == set([30, 30, 30]) 
    shutil.rmtree(CACHE_DIR)
