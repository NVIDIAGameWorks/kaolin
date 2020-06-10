import os
import sys
import pytest
import torch
from pathlib import Path

from kaolin.rep import TriangleMesh, VoxelGrid, PointCloud
from kaolin.conversions.meshconversions import trianglemesh_to_pointcloud, trianglemesh_to_voxelgrid

# Skip test if import fails unless on CI and not on Windows
if os.environ.get('CI') and not sys.platform == 'win32':
    from kaolin.visualize.vis_usd import VisUsd
else:
    VisUsd = pytest.importorskip('kaolin.visualize.vis_usd.VisUsd', reason='The pxr library could not be imported')

root = Path('tests/visualize/results')
root.mkdir(exist_ok=True)
mesh = TriangleMesh.from_obj('tests/model.obj')
voxels = VoxelGrid(trianglemesh_to_voxelgrid(mesh, 32))
pc = PointCloud(trianglemesh_to_pointcloud(mesh, 500)[0])

vis = VisUsd()

@pytest.mark.parametrize('object_3d', [mesh, voxels, pc])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('meet_ground', [True, False])
@pytest.mark.parametrize('center_on_stage', [True, False])
@pytest.mark.parametrize('fit_to_stage', [True, False])
def test_vis(object_3d, device, meet_ground, center_on_stage, fit_to_stage):
    if device == 'cuda':
        if isinstance(object_3d, TriangleMesh):
            object_3d.cuda()
        elif isinstance(object_3d, PointCloud):
            object_3d.points = object_3d.points.to(torch.device(device))
        elif isinstance(object_3d, VoxelGrid):
            object_3d.voxels = object_3d.voxels.to(torch.device(device))

    vis.set_stage(filepath=str(root / f'{type(object_3d).__name__}_{device}.usda'))
    vis.visualize(object_3d, meet_ground=meet_ground, center_on_stage=center_on_stage,
                  fit_to_stage=fit_to_stage)
