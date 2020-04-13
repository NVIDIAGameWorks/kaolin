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
from nuscenes.nuscenes import NuScenes

import kaolin as kal
import kaolin.transforms.transforms as tfs
from kaolin.rep import PointCloud


NUSCENES_ROOT = '/data/nuscenes_mini/'


# Tests below can only be run if a ModelNet dataset is available
REASON = 'Nuscenes mini not found at default location: {}'.format(NUSCENES_ROOT)


@pytest.mark.skipif(not os.path.exists(NUSCENES_ROOT), reason=REASON)
def test_nusc():
    nusc = NuScenes(version='v1.0-mini',
                    dataroot=NUSCENES_ROOT,
                    verbose=False)

    traindata = kal.datasets.NuscDetection(nusc, train=True, nsweeps=5)
    traindata_large = kal.datasets.NuscDetection(nusc, train=True, nsweeps=10)
    valdata = kal.datasets.NuscDetection(nusc, train=False, nsweeps=5)

    assert len(traindata) == 323
    assert len(valdata) == 81

    inst = traindata[10]
    assert isinstance(inst['data']['pc'], PointCloud)

    # check dimension of point cloud for 5 lidar sweeps
    N, D = inst['data']['pc'].points.shape
    assert N == 129427
    assert D == 5

    # check dimension of point cloud for 10 lidar sweeps
    N, D = traindata_large[10]['data']['pc'].points.shape
    assert N == 258188
    assert D == 5
