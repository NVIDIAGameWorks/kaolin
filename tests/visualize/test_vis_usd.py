import pytest
import torch

from kaolin.rep import TriangleMesh, VoxelGrid, PointCloud
from kaolin.conversions.meshconversions import trianglemesh_to_pointcloud, trianglemesh_to_voxelgrid
from kaolin.visualize.vis_usd import VisUsd

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

    vis.set_stage(filepath=f'tests/{type(object_3d).__name__}_{device}.usda')
    vis.visualize(object_3d, meet_ground=meet_ground, center_on_stage=center_on_stage,
                  fit_to_stage=fit_to_stage)
