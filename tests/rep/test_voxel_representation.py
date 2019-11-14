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

import pytest
import torch

import kaolin as kal


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_voxelgrid(device):
    voxels = torch.ones([32, 32, 32]).to(device)
    kal.rep.VoxelGrid(voxels=voxels, copy=False)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_scale_down(device):
    voxel = torch.ones([32, 32, 32]).to(device)
    down = kal.conversions.voxelgridconversions.downsample(voxel, [2, 2, 2])
    assert (set(down.shape) == set([16, 16, 16]))
    down = kal.conversions.voxelgridconversions.downsample(voxel, [3, 3, 3])
    assert (set(down.shape) == set([10, 10, 10]))
    down = kal.conversions.voxelgridconversions.downsample(voxel, [3, 2, 1])
    assert (set(down.shape) == set([10, 16, 32]))


    voxel = torch.zeros([128, 128, 128]).to(device)
    down = kal.conversions.voxelgridconversions.downsample(voxel, [2, 2, 2])
    assert (set(down.shape) == set([64, 64, 64]))
    down = kal.conversions.voxelgridconversions.downsample(voxel, [3, 3, 3])
    assert (set(down.shape) == set([42, 42, 42]))
    down = kal.conversions.voxelgridconversions.downsample(voxel, [3, 2, 1])
    assert (set(down.shape) == set([42, 64, 128]))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_scale_up(device):
    voxel = torch.ones([32, 32, 32]).to(device)
    down = kal.conversions.voxelgridconversions.upsample(voxel, 64)
    assert (set(down.shape) == set([64, 64, 64]))
    down = kal.conversions.voxelgridconversions.upsample(voxel, 33)
    assert (set(down.shape) == set([33, 33, 33]))
    down = kal.conversions.voxelgridconversions.upsample(voxel, 256)
    assert (set(down.shape) == set([256, 256, 256]))


    voxel = torch.zeros([128, 128, 128]).to(device)
    down = kal.conversions.voxelgridconversions.upsample(voxel, 128)
    assert (set(down.shape) == set([128, 128, 128]))

    down = kal.conversions.voxelgridconversions.upsample(voxel, 150)
    assert (set(down.shape) == set([150, 150, 150]))

    down = kal.conversions.voxelgridconversions.upsample(voxel, 256 )
    assert (set(down.shape) == set([256, 256, 256]))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_threshold(device):
    voxel = torch.ones([32, 32, 32]).to(device)
    binary = kal.conversions.voxelgridconversions.threshold(voxel, .5)
    assert binary.sum() == 32*32*32
    assert (set(binary.shape) == set([32, 32, 32]))

    voxel = torch.ones([32, 32, 32]).to(device) * .3
    binary = kal.conversions.voxelgridconversions.threshold(voxel, .5)
    assert binary.sum() == 0
    assert (set(binary.shape) == set([32, 32, 32]))

    voxel = torch.ones([64, 64, 64]).to(device) * .7
    binary = kal.conversions.voxelgridconversions.threshold(voxel, .5)
    assert binary.sum() == 64*64*64
    assert (set(binary.shape) == set([64, 64, 64]))


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_fill(device):
    voxel = torch.ones([32, 32, 32]).to(device)
    voxel[16, 16, 16] = 0 
    filled_voxel = kal.conversions.voxelgridconversions.fill(voxel)
    assert voxel.sum() < filled_voxel.sum()
    assert filled_voxel.sum() == 32*32*32

    voxel = torch.ones([32, 32, 32]).to(device)
    filled_voxel = kal.conversions.voxelgridconversions.fill(voxel)
    assert voxel.sum()  == filled_voxel.sum()
    assert filled_voxel.sum() == 32*32*32

    voxel = torch.zeros([64, 64, 64]).to(device)
    filled_voxel = kal.conversions.voxelgridconversions.fill(voxel)
    assert voxel.sum()  == filled_voxel.sum()
    assert filled_voxel.sum() == 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_extract_surface(device):
    voxel = torch.ones([32, 32, 32]).to(device)
    surface_voxel = kal.conversions.voxelgridconversions.extract_surface(voxel)
    assert voxel.sum() > surface_voxel.sum()
    assert surface_voxel.sum() == 2*(32*32 + 32*30 + 30*30)
    assert kal.conversions.voxelgridconversions.extract_surface(surface_voxel).sum() == surface_voxel.sum()

    voxel = torch.zeros([32, 32, 32]).to(device)
    surface_voxel = kal.conversions.voxelgridconversions.extract_surface(voxel)
    assert voxel.sum() == surface_voxel.sum()
    assert surface_voxel.sum() == 0
    assert kal.conversions.voxelgridconversions.extract_surface(surface_voxel).sum() == surface_voxel.sum()	

    voxel = torch.ones([64, 64, 64]).to(device)
    surface_voxel = kal.conversions.voxelgridconversions.extract_surface(voxel)
    assert voxel.sum() > surface_voxel.sum()
    assert surface_voxel.sum() == 2*(64*64 + 64*62 + 62*62)
    assert kal.conversions.voxelgridconversions.extract_surface(surface_voxel).sum() == surface_voxel.sum()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_extract_odms(device):
    voxel = torch.rand([32, 32, 32]).to(device)
    voxel = kal.conversions.voxelgridconversions.threshold(voxel, .9)
    odms = kal.conversions.voxelgridconversions.extract_odms(voxel)
    assert (set(odms.shape) == set([6, 32, 32]))
    assert (odms.max() == 32)
    assert (odms.min() == 0)

    voxel = torch.ones([32, 32, 32]).to(device)
    voxel = kal.conversions.voxelgridconversions.threshold(voxel, .5)
    odms = kal.conversions.voxelgridconversions.extract_odms(voxel)
    assert (set(odms.shape) == set([6, 32, 32]))
    assert (odms.max() == 0)

    voxel = torch.zeros([128, 128, 128]).to(device)
    voxel = kal.conversions.voxelgridconversions.threshold(voxel, .5)
    odms = kal.conversions.voxelgridconversions.extract_odms(voxel)
    assert (set(odms.shape) == set([6, 128, 128]))
    assert (odms.max() == 128)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_project_odms(device): 

    odms = torch.rand([32, 32, 32]).to(device) *32
    odms = odms.int()
    voxel = kal.conversions.voxelgridconversions.project_odms(odms)
    assert (set(voxel.shape) == set([32, 32, 32]))

    odms = torch.rand([128, 128, 128]).to(device) *128
    odms = odms.int()
    voxel = kal.conversions.voxelgridconversions.project_odms(odms)
    assert (set(voxel.shape) == set([128, 128, 128]))

    voxel = torch.ones([32, 32, 32]).to(device)
    voxel = kal.conversions.voxelgridconversions.threshold(voxel, .9)
    odms = kal.conversions.voxelgridconversions.extract_odms(voxel)
    new_voxel = kal.conversions.voxelgridconversions.project_odms(odms)
    assert (set(new_voxel.shape) == set([32, 32, 32]))
    assert (torch.abs(voxel - new_voxel).sum() == 0)
