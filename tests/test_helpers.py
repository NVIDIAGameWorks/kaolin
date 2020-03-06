import os
import shutil

import pytest
import torch
import numpy as np

import kaolin as kal
from kaolin import helpers


CACHE_DIR = 'tests/cache'


@pytest.fixture(autouse=True)
def cleanup():
    """Cleanup after each test. """
    yield
    shutil.rmtree(CACHE_DIR)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_cache_tensor(device):
    tensor = torch.ones(5, device=device)

    cache = helpers.Cache(func=lambda x: x, cache_dir=CACHE_DIR, cache_key='test')
    cache('tensor', x=tensor)

    # Make sure cache is created
    assert os.path.exists(os.path.join(CACHE_DIR, 'test', 'tensor.p'))

    # Confirm loaded tensor is correct and on CPU device
    loaded = cache('tensor')
    assert torch.all(loaded.eq(tensor.cpu()))

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_cache_dict(device):
    dictionary = {
        'a': torch.ones(5, device=device),
        'b': np.zeros(5),
    }

    cache = helpers.Cache(func=lambda x: x, cache_dir=CACHE_DIR, cache_key='test')
    cache('dictionary', x=dictionary)

    # Make sure cache is created
    assert os.path.exists(os.path.join(CACHE_DIR, 'test', 'dictionary.p'))

    # Confirm loaded dict is correct and on CPU device
    loaded = cache('dictionary')
    assert torch.all(loaded['a'].eq(dictionary['a'].cpu()))
    assert np.all(np.isclose(loaded['b'], dictionary['b']))

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_cache_mesh(device):
    vertices = torch.ones(10, 3, device=device)
    faces = torch.ones(20, 3, device=device, dtype=torch.long)
    mesh = kal.rep.TriangleMesh.from_tensors(vertices, faces)

    cache = helpers.Cache(func=lambda x: x, cache_dir=CACHE_DIR, cache_key='test')
    cache('mesh', x=mesh)

    # Make sure cache is created
    assert os.path.exists(os.path.join(CACHE_DIR, 'test', 'mesh.p'))

    # Confirm loaded mesh is correct and on CPU device
    loaded = cache('mesh')
    assert torch.all(loaded.vertices.eq(vertices.cpu()))
    assert torch.all(loaded.faces.eq(faces.cpu()))

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_cache_voxelgrid(device):
    voxels = torch.ones(3, 3, 3, device=device)
    voxelgrid = kal.rep.VoxelGrid(voxels)

    cache = helpers.Cache(func=lambda x: x, cache_dir=CACHE_DIR, cache_key='test')
    cache('voxelgrid', x=voxelgrid)

    # Make sure cache is created
    assert os.path.exists(os.path.join(CACHE_DIR, 'test', 'voxelgrid.p'))

    # Confirm loaded voxelgrid is correct and on CPU device
    loaded = cache('voxelgrid')
    assert torch.all(loaded.voxels.eq(voxels.cpu()))

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_cache_pointcloud(device):
    points = torch.ones(10, 3, device=device)
    pointcloud = kal.rep.PointCloud(points)

    cache = helpers.Cache(func=lambda x: x, cache_dir=CACHE_DIR, cache_key='test')
    cache('pointcloud', x=pointcloud)

    # Make sure cache is created
    assert os.path.exists(os.path.join(CACHE_DIR, 'test', 'pointcloud.p'))

    # Confirm loaded pointcloud is correct and on CPU device
    loaded = cache('pointcloud')
    assert torch.all(loaded.points.eq(points.cpu()))
