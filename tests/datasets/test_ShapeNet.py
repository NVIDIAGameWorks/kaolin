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
import os
import shutil
from pathlib import Path

import kaolin as kal
from torch.utils.data import DataLoader
from kaolin.datasets import shapenet


SHAPENET_ROOT = 'data/ShapeNet/'
CACHE_DIR = 'tests/datasets/cache'


# Tests below can only be run is a ShapeNet dataset is available
REASON = 'ShapeNet not found at {}'.format(SHAPENET_ROOT)


@pytest.mark.skipif(not Path(SHAPENET_ROOT).exists(), reason=REASON)
def test_Meshes():
    meshes1 = shapenet.ShapeNet_Meshes(root=SHAPENET_ROOT,
                                       categories=['can'], train=True, split=.7)
    assert len(meshes1) > 0
    for mesh in meshes1:
        assert Path(mesh['attributes']['path']).is_file()
        assert mesh['data']['vertices'].shape[0] > 0

    meshes2 = shapenet.ShapeNet_Meshes(root=SHAPENET_ROOT,
                                       categories=['can', 'bowl'], train=True, split=.7)
    assert len(meshes2) > len(meshes1)


@pytest.mark.skipif(not Path(SHAPENET_ROOT).exists(), reason=REASON)
def test_Voxels():
    voxels = shapenet.ShapeNet_Voxels(root=SHAPENET_ROOT, cache_dir=CACHE_DIR,
                                      categories=['can'], train=True, split=.7, resolutions=[32])
    assert len(voxels) == 75
    assert voxels.cache_dir.exists()
    assert len(list(voxels.cache_dir.rglob('*.npz'))) == 75
    for obj in voxels:
        # assert os.path.isfile(obj['32_name'])
        assert (set(obj['data']['32'].shape) == set([32, 32, 32]))

    voxels = shapenet.ShapeNet_Voxels(root=SHAPENET_ROOT, cache_dir=CACHE_DIR,
                                      categories=['can'], train=False, split=.7, resolutions=[32])
    assert len(voxels) == 33

    shutil.rmtree('tests/datasets/cache/voxels')


# def test_Images():
#     images = shapenet.ShapeNet_Images(root=SHAPENET_ROOT, cache_dir=CACHE_DIR,
#                                       categories=['phone'], views=1, train=True, split=.7)
#     assert len(images) == 736
#     for obj in images:
#         assert set(obj['data']['images'].shape) == set([137, 137, 4])
#         assert os.path.isfile(obj['attributes']['name'] + '/rendering/00.png')
#         assert set(obj['data']['params']['cam_mat'].shape) == set([3, 3])
#         assert set(obj['data']['params']['cam_pos'].shape) == set([3])


@pytest.mark.skipif(not Path(SHAPENET_ROOT).exists(), reason=REASON)
def test_Surface_Meshes():
    surface_meshes = shapenet.ShapeNet_Surface_Meshes(root=SHAPENET_ROOT, cache_dir=CACHE_DIR,
                                                      categories=['can'], train=True, split=.1,
                                                      resolution=100, smoothing_iterations=3,
                                                      mode='Tri')
    assert len(surface_meshes) == 10
    assert surface_meshes.cache_dir.exists()
    assert len(list(surface_meshes.cache_dir.rglob('*.npz'))) == 10
    for smesh in surface_meshes:
        assert smesh['data']['vertices'].shape[0] > 0
        assert smesh['data']['faces'].shape[1] == 3

    shutil.rmtree('tests/datasets/cache/surface_meshes')

    surface_meshes = shapenet.ShapeNet_Surface_Meshes(root=SHAPENET_ROOT, cache_dir=CACHE_DIR,
                                                      categories=['can'], train=True, split=.1,
                                                      resolution=100, smoothing_iterations=3,
                                                      mode='Quad')
    assert len(surface_meshes) == 10
    assert surface_meshes.cache_dir.exists()
    assert len(list(surface_meshes.cache_dir.rglob('*.npz'))) == 10
    for smesh in surface_meshes:
        assert smesh['data']['vertices'].shape[0] > 0
        assert smesh['data']['faces'].shape[1] == 4
    shutil.rmtree('tests/datasets/cache/voxels')
    shutil.rmtree('tests/datasets/cache/surface_meshes')


@pytest.mark.skipif(not Path(SHAPENET_ROOT).exists(), reason=REASON)
def test_Points():
    points = shapenet.ShapeNet_Points(root=SHAPENET_ROOT, cache_dir=CACHE_DIR,
                                      categories=['can'], train=True, split=.1,
                                      resolution=100, smoothing_iterations=3, num_points=5000,
                                      surface=False, normals=False)

    assert len(points) == 10
    assert points.cache_dir.exists()
    assert len(list(points.cache_dir.rglob('*.npz'))) == 10
    for obj in points:
        assert set(obj['data']['points'].shape) == set([5000, 3])
        assert set(obj['data']['normals'].shape) == set([5000, 3])

    shutil.rmtree('tests/datasets/cache/points')

    points = shapenet.ShapeNet_Points(root=SHAPENET_ROOT, cache_dir=CACHE_DIR,
                                      categories=['can'], train=True, split=.1,
                                      resolution=100, smoothing_iterations=3, num_points=5000,
                                      surface=True, normals=True)

    assert len(points) == 10
    assert points.cache_dir.exists()
    assert len(list(points.cache_dir.rglob('*.npz'))) == 10
    for obj in points:
        assert set(obj['data']['points'].shape) == set([5000, 3])
        assert set(obj['data']['normals'].shape) == set([5000, 3])

    shutil.rmtree('tests/datasets/cache/points')
    shutil.rmtree('tests/datasets/cache/voxels')
    shutil.rmtree('tests/datasets/cache/surface_meshes')


@pytest.mark.skipif(not Path(SHAPENET_ROOT).exists(), reason=REASON)
def test_SDF_Points():
    sdf_points = shapenet.ShapeNet_SDF_Points(root=SHAPENET_ROOT, cache_dir=CACHE_DIR,
                                              categories=['can'], train=True, split=.1,
                                              resolution=100, smoothing_iterations=3,
                                              num_points=5000, occ=False, sample_box=True)

    assert len(sdf_points) == 10
    assert sdf_points.cache_dir.exists()
    assert len(list(sdf_points.cache_dir.rglob('*.npz'))) == 10
    for obj in sdf_points:
        assert set(obj['data']['sdf_points'].shape) == set([5000, 3])
        assert set(obj['data']['sdf_distances'].shape) == set([5000])

    shutil.rmtree('tests/datasets/cache/sdf_points')

    sdf_points = shapenet.ShapeNet_SDF_Points(root=SHAPENET_ROOT, cache_dir=CACHE_DIR,
                                              categories=['can'], train=True, split=.1,
                                              resolution=100, smoothing_iterations=3,
                                              num_points=5000, occ=True, sample_box=True)

    assert len(sdf_points) == 10
    assert sdf_points.cache_dir.exists()
    assert len(list(sdf_points.cache_dir.rglob('*.npz'))) == 10
    for obj in sdf_points:
        assert set(obj['data']['occ_points'].shape) == set([5000, 3])
        assert set(obj['data']['occ_values'].shape) == set([5000])

    shutil.rmtree('tests/datasets/cache/sdf_points')
    shutil.rmtree('tests/datasets/cache/voxels')
    shutil.rmtree('tests/datasets/cache/surface_meshes')


@pytest.mark.skipif(not Path(SHAPENET_ROOT).exists(), reason=REASON)
def test_Combination():
    dataset_params = {
        'root': SHAPENET_ROOT,
        'categories': ['can'],
        'train': True,
        'split': .8,
    }
    # images = shapenet.ShapeNet_Images(root=SHAPENET_ROOT, cache_dir=CACHE_DIR,
    #                                   categories=['bowl'], views=1, train=True, split=.8)
    meshes = shapenet.ShapeNet_Meshes(**dataset_params)
    voxels = shapenet.ShapeNet_Voxels(**dataset_params, cache_dir=CACHE_DIR, resolutions=[32])
    sdf_points = shapenet.ShapeNet_SDF_Points(**dataset_params, cache_dir=CACHE_DIR,
                                              smoothing_iterations=3, num_points=500, occ=False,
                                              sample_box=True)

    points = shapenet.ShapeNet_Points(**dataset_params, cache_dir=CACHE_DIR, resolution=100,
                                      smoothing_iterations=3, num_points=500,
                                      surface=False, normals=True)

    dataset = shapenet.ShapeNet_Combination([voxels, sdf_points, points])

    for obj in dataset:
        obj_data = obj['data']
        assert set(obj['data']['sdf_points'].shape) == set([500, 3])
        assert set(obj['data']['sdf_distances'].shape) == set([500])
        assert set(obj['data']['32'].shape) == set([32, 32, 32])
        assert set(obj['data']['points'].shape) == set([500, 3])
        assert set(obj['data']['normals'].shape) == set([500, 3])

    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=8)
    for batch in train_loader:
        assert set(batch['data']['sdf_points'].shape) == set([2, 500, 3])
        assert set(batch['data']['sdf_distances'].shape) == set([2, 500])
        assert set(batch['data']['32'].shape) == set([2, 32, 32, 32])
        assert set(batch['data']['points'].shape) == set([2, 500, 3])
        assert set(batch['data']['normals'].shape) == set([2, 500, 3])

    shutil.rmtree('tests/datasets/cache/sdf_points')
    shutil.rmtree('tests/datasets/cache/points')
    shutil.rmtree('tests/datasets/cache/voxels')
    shutil.rmtree('tests/datasets/cache/surface_meshes')
