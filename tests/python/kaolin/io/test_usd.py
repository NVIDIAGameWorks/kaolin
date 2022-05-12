# Copyright (c) 2019, 20-21 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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


import os
import shutil

import torch
import pytest
from pxr import Usd, UsdGeom

from kaolin.io import usd, obj
from kaolin.io import utils
from kaolin.ops.conversions import trianglemeshes_to_voxelgrids


@pytest.fixture(scope='class')
def out_dir():
    # Create temporary output directory
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_out')
    os.makedirs(out_dir, exist_ok=True)
    yield out_dir
    shutil.rmtree(out_dir)


@pytest.fixture(scope='module')
def mesh():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    obj_mesh = obj.import_mesh(os.path.join(cur_dir, os.pardir, os.pardir,
                               os.pardir, 'samples/rocket.obj'), with_normals=True,
                               with_materials=True, error_handler=obj.skip_error_handler)
    return obj_mesh

@pytest.fixture(scope='module')
def mesh_alt():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    obj_mesh = obj.import_mesh(os.path.join(cur_dir, os.pardir, os.pardir,
                               os.pardir, 'samples/model.obj'), with_normals=True,
                               with_materials=True, error_handler=obj.skip_error_handler)
    return obj_mesh

@pytest.fixture(scope='module')
def hetero_mesh_path():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cur_dir, os.pardir, os.pardir,
                        os.pardir, 'samples/rocket_hetero.usd')

@pytest.fixture(scope='module')
def hetero_subsets_materials_mesh_path():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(cur_dir, os.pardir, os.pardir,
                        os.pardir, 'samples/rocket_hetero_subsets_materials.usd')

@pytest.fixture(scope='module')
def pointcloud():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    pointcloud, color, normals = usd.import_pointcloud(
        os.path.join(cur_dir, os.pardir, os.pardir,
                     os.pardir, 'samples/rocket_pointcloud_GeomPoints.usda'),
        '/World/pointcloud')
    return pointcloud

@pytest.fixture(scope='module')
def pointcloud_instancer():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    pointcloud, color, normals = usd.import_pointcloud(
        os.path.join(cur_dir, os.pardir, os.pardir,
                     os.pardir, 'samples/rocket_pointcloud.v0.9.0.usda'),
        '/World/pointcloud')
    return pointcloud

@pytest.fixture(scope='module')
def pointcloud_with_color():
    cur_dir = os.path.dirname(os.path.realpath(__file__))
    pointcloud, color, normals = usd.import_pointcloud(
        os.path.join(cur_dir, os.pardir, os.pardir,
                     os.pardir, 'samples/golden/pointcloud_GeomPoints_colors.usda'),
        '/World/pointcloud')
    return (pointcloud, color)

class TestMeshes:
    def setup_method(self):
        self.file_name = 'meshes.usda'

    @pytest.fixture(scope='class')
    def scene_paths(self):
        num_meshes = 2
        return [f'/World/mesh_{i}' for i in range(num_meshes)]

    def test_export_single(self, scene_paths, out_dir, mesh):
        out_path = os.path.join(out_dir, f'single_{self.file_name}')

        # Export a mesh
        stage = usd.export_mesh(out_path, scene_paths[0], mesh.vertices, mesh.faces)

        # Confirm exported USD matches golden file
        golden = os.path.join(out_dir, os.pardir, os.pardir, os.pardir,
                              os.pardir, 'samples/golden/mesh.usda')
        assert open(golden).read() == open(out_path).read()

    def test_export_multiple(self, scene_paths, out_dir, mesh, mesh_alt):
        out_path = os.path.join(out_dir, self.file_name)

        # Export some meshes
        vertices_list = [mesh.vertices, mesh_alt.vertices]
        faces_list = [mesh.faces, mesh_alt.faces]
        stage = usd.export_meshes(out_path, scene_paths, vertices_list, faces_list)

    def test_import_bad_prim(self, scene_paths, out_dir, mesh):
        """Test that import fails when reaching invalid prims"""
        out_path = os.path.join(out_dir, self.file_name)
        with pytest.raises(ValueError):
            usd.import_meshes(out_path, ['/foo'] + scene_paths)

    def test_import_hetero_fail_import_meshes(self, scene_paths, out_dir, hetero_mesh_path):
        """Test that import fails when importing heterogeneous mesh without handler"""
        with pytest.raises(utils.NonHomogeneousMeshError):
            usd.import_meshes(hetero_mesh_path, ['/Root'])

    def test_import_hetero_fail_import_mesh(self, scene_paths, out_dir, hetero_mesh_path):
        """Test that import fails when importing heterogeneous mesh without handler"""
        with pytest.raises(utils.NonHomogeneousMeshError):
            usd.import_mesh(file_path=hetero_mesh_path, scene_path='/Root')

    def test_import_hetero_skip(self, scene_paths, out_dir, hetero_mesh_path, mesh):
        """Test that import skips mesh when importing heterogeneous mesh with skip handler"""
        meshes = usd.import_meshes(hetero_mesh_path, ['/Root'],
                                   heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_skip)
        assert len(meshes) == 0

        # Test skip on a batch of mixed homogeneous and heterogeneous meshes
        homogeneous_meshes = usd.import_meshes(os.path.join(out_dir, self.file_name))
        mixed = meshes + homogeneous_meshes
        out_path = os.path.join(out_dir, 'mixed.usda')
        usd.export_meshes(out_path, vertices=[m.vertices for m in mixed], faces=[m.faces for m in mixed])

        mixed_in = usd.import_meshes(out_path, heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_skip)
        assert len(mixed_in) == 2

    def test_import_hetero_empty_import_meshes(self, scene_paths, out_dir, hetero_mesh_path):
        """Test that imports empty mesh when importing heterogeneous mesh with empty handler"""
        mesh = usd.import_meshes(hetero_mesh_path, ['/Root'],
                                 heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_empty)
        for attr in mesh:
            assert len(attr[0]) == 0

    def test_import_hetero_empty_import_mesh(self, scene_paths, out_dir, hetero_mesh_path):
        """Test that imports empty mesh when importing heterogeneous mesh with empty handler"""
        mesh = usd.import_mesh(hetero_mesh_path, scene_path='/Root', heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_empty)
        assert len(mesh[0]) == 0

    def test_import_hetero_homogenize_import_meshes(self, scene_paths, out_dir, hetero_mesh_path):
        """Test that imports homogeneous mesh when importing heterogeneous mesh with naive homogenize handler"""
        # TODO(jlafleche) Render meshes before/after homogenize operation
        out_path = os.path.join(out_dir, 'homogenized.usda')
        mesh = usd.import_meshes(hetero_mesh_path, ['/Root'],
                                 heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_naive_homogenize)
        usd.export_mesh(out_path, '/World/Rocket', vertices=mesh[0].vertices, faces=mesh[0].faces)

        # Confirm we now have a triangle mesh
        assert mesh[0].faces.size(1) == 3

        # Confirm exported USD matches golden file
        golden = os.path.join(out_dir, '../../../../samples/golden/rocket_homogenized.usda')
        assert open(golden).read() == open(out_path).read()

    def test_import_hetero_homogenize_import_mesh(self, scene_paths, out_dir, hetero_mesh_path):
        """Test that imports homogeneous mesh when importing heterogeneous mesh with naive homogenize handler"""
        # TODO(jlafleche) Render meshes before/after homogenize operation
        out_path = os.path.join(out_dir, 'homogenized.usda')
        mesh = usd.import_mesh(hetero_mesh_path, scene_path='/Root',
                               heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_naive_homogenize)
        usd.export_mesh(out_path, '/World/Rocket', vertices=mesh.vertices, faces=mesh.faces)

        # Confirm we now have a triangle mesh
        assert mesh.faces.size(1) == 3

        # Confirm exported USD matches golden file
        golden = os.path.join(out_dir, '../../../../samples/golden/rocket_homogenized.usda')
        assert open(golden).read() == open(out_path).read()

    def test_import_material_subsets(self, scene_paths, out_dir, hetero_subsets_materials_mesh_path):
        """Test that imports materials from mesh with subsets"""
        out_path = os.path.join(out_dir, 'homogenized_materials.usda')
        mesh = usd.import_mesh(hetero_subsets_materials_mesh_path, scene_path='/Root',
                               heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_naive_homogenize,
                               with_materials=True)
        usd.export_mesh(out_path, '/World/Rocket', vertices=mesh.vertices, faces=mesh.faces,
                        materials_order=mesh.materials_order, materials=mesh.materials, uvs=mesh.uvs)

        # Confirm we now have a triangle mesh
        assert mesh.faces.size(1) == 3

        # Confirm exported USD matches golden file
        golden = os.path.join(out_dir, '../../../../samples/golden/rocket_homogenized_materials.usda')
        assert open(golden).read() == open(out_path).read()

    def test_import_with_material(self, scene_paths, out_dir, hetero_subsets_materials_mesh_path):
        """Test that imports materials from mesh with subsets"""
        out_path = os.path.join(out_dir, 'homogenized_materials.usda')
        mesh = usd.import_mesh(hetero_subsets_materials_mesh_path, scene_path='/Root',
                               heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_naive_homogenize,
                               with_materials=False)
        assert mesh.materials is None
        assert mesh.materials_order is None

        mesh = usd.import_mesh(hetero_subsets_materials_mesh_path, scene_path='/Root',
                               heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_naive_homogenize,
                               with_materials=True)
        assert mesh.materials is not None
        assert mesh.materials_order is not None


    def test_import_multiple(self, scene_paths, out_dir, mesh, mesh_alt):
        out_path = os.path.join(out_dir, self.file_name)
        meshes_in = usd.import_meshes(out_path, scene_paths)

        # Confirm imported vertices and faces match original input
        assert torch.allclose(mesh.vertices, meshes_in[0].vertices)
        assert torch.all(mesh.faces.eq(meshes_in[0].faces))
        assert torch.allclose(mesh_alt.vertices, meshes_in[1].vertices)
        assert torch.all(mesh_alt.faces.eq(meshes_in[1].faces))

    def test_import_multiple_no_paths(self, out_dir, mesh, mesh_alt):
        out_path = os.path.join(out_dir, self.file_name)
        meshes_in = usd.import_meshes(out_path)

        # Confirm imported vertices and faces match original input
        assert torch.allclose(mesh.vertices, meshes_in[0].vertices)
        assert torch.all(mesh.faces.eq(meshes_in[0].faces))
        assert torch.allclose(mesh_alt.vertices, meshes_in[1].vertices)
        assert torch.all(mesh_alt.faces.eq(meshes_in[1].faces))

    def test_import_single_flattened(self, scene_paths, out_dir, mesh, mesh_alt):
        """Will flatten all meshes in USD into a single mesh."""
        out_path = os.path.join(out_dir, self.file_name)
        mesh_in = usd.import_mesh(out_path)
        assert len(mesh_in.vertices) == (len(mesh.vertices) + len(mesh_alt.vertices))
        assert len(mesh_in.faces) == (len(mesh.faces) + len(mesh_alt.faces))

    def test_export_only_vertices(self, out_dir, mesh):
        out_path = os.path.join(out_dir, 'only_vert.usda')
        usd.export_mesh(out_path, vertices=mesh.vertices)
        mesh_in = usd.import_mesh(out_path)
        assert torch.allclose(mesh_in.vertices, mesh.vertices)

    def test_export_only_faces(self, out_dir, mesh):
        out_path = os.path.join(out_dir, 'only_faces.usda')
        usd.export_mesh(out_path, faces=mesh.faces)
        mesh_in = usd.import_mesh(out_path)
        assert torch.allclose(mesh_in.faces, mesh.faces)

    def test_export_only_face_uvs(self, out_dir, mesh):
        out_path = os.path.join(out_dir, 'only_uvs.usda')
        usd.export_mesh(out_path, vertices=mesh.vertices, faces=mesh.faces, uvs=mesh.uvs)
        mesh_in = usd.import_mesh(out_path)
        assert torch.allclose(mesh_in.uvs.view(-1, 2), mesh.uvs.view(-1, 2))

    def test_import_st_indices_facevarying(self, out_dir, mesh):
        out_path = os.path.join(out_dir, 'st_indices.usda')
        uvs = torch.rand((mesh.faces.view(-1).size(0), 2))
        scene_path = '/World/mesh_0'
        face_uvs_idx = (torch.rand(mesh.faces.shape[:2]) * 99).long()
        usd.export_mesh(out_path, scene_path=scene_path, vertices=mesh.vertices,
                        faces=mesh.faces, uvs=uvs, face_uvs_idx=face_uvs_idx)

        # check that interpolation was set correctly to 'faceVarying'
        stage = Usd.Stage.Open(out_path)
        pv = UsdGeom.Mesh(stage.GetPrimAtPath(scene_path)).GetPrimvar('st')
        assert pv.GetInterpolation() == 'faceVarying'

        mesh_in = usd.import_mesh(out_path)
        assert torch.allclose(mesh_in.uvs, uvs)

    def test_import_st_no_indices_vertex(self, out_dir, mesh):
        out_path = os.path.join(out_dir, 'st_no_indices_vertex.usda')
        uvs = torch.rand((mesh.vertices.size(0), 2))
        scene_path = '/World/mesh_0'
        usd.export_mesh(out_path, scene_path=scene_path, vertices=mesh.vertices,
                        faces=mesh.faces, uvs=uvs)

        # check that interpolation was set correctly to 'vertex'
        stage = Usd.Stage.Open(out_path)
        pv = UsdGeom.Mesh(stage.GetPrimAtPath(scene_path)).GetPrimvar('st')
        assert pv.GetInterpolation() == 'vertex'

        mesh_in = usd.import_mesh(out_path)
        assert torch.allclose(mesh_in.uvs, uvs)

    def test_import_st_no_indices_facevarying(self, out_dir, mesh):
        out_path = os.path.join(out_dir, 'st_no_indices_face_varying.usda')
        uvs = torch.rand((mesh.faces.size(0) * mesh.faces.size(1), 2))
        scene_path = '/World/mesh_0'
        usd.export_mesh(out_path, scene_path=scene_path, vertices=mesh.vertices,
                        faces=mesh.faces, uvs=uvs)

        # check that interpolation was set correctly to 'faceVarying'
        stage = Usd.Stage.Open(out_path)
        pv = UsdGeom.Mesh(stage.GetPrimAtPath(scene_path)).GetPrimvar('st')
        assert pv.GetInterpolation() == 'faceVarying'

        mesh_in = usd.import_mesh(out_path)
        assert torch.allclose(mesh_in.uvs, uvs)

    def test_import_st_no_indices_uniform(self, out_dir, mesh):
        out_path = os.path.join(out_dir, 'st_no_indices_face_uniform.usda')
        uvs = torch.rand((mesh.faces.size(0), 2))
        scene_path = '/World/mesh_0'
        usd.export_mesh(out_path, scene_path=scene_path, vertices=mesh.vertices,
                        faces=mesh.faces, uvs=uvs)

        # check that interpolation was set correctly to 'uniform'
        stage = Usd.Stage.Open(out_path)
        pv = UsdGeom.Mesh(stage.GetPrimAtPath(scene_path)).GetPrimvar('st')
        assert pv.GetInterpolation() == 'uniform'

        # TODO(jlafleche) add support for `uniform` interpolation
        # mesh_in = usd.import_mesh(out_path)
        # assert torch.allclose(mesh_in.uvs, uvs)

    def test_export_only_face_normals(self, out_dir, mesh):
        out_path = os.path.join(out_dir, 'only_normals.usda')
        usd.export_mesh(out_path, face_normals=mesh.vertex_normals[mesh.face_normals])
        mesh_in = usd.import_mesh(out_path, with_normals=True)
        assert torch.allclose(mesh_in.face_normals.view(-1, 3), mesh.vertex_normals[mesh.face_normals].view(-1, 3))


class TestPointCloud:
    def setup_method(self):
        self.scene_path = '/World/pointcloud'
        self.num_multiple = 3

    def test_export_single(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'pointcloud.usda')
        usd.export_pointcloud(pointcloud=pointcloud, file_path=out_path, scene_path=self.scene_path, points_type='usd_geom_points')

        # Confirm exported USD matches golden file
        golden = os.path.join(out_dir, '../../../../samples/golden/pointcloud_GeomPoints.usda')
        assert open(golden).read() == open(out_path).read()

    def test_export_single_instancer(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'pointcloud_instancer.usda')
        usd.export_pointcloud(pointcloud=pointcloud, file_path=out_path, scene_path=self.scene_path)

        # Confirm exported USD matches golden file
        golden = os.path.join(out_dir, '../../../../samples/golden/pointcloud_PointInstancer.usda')
        assert open(golden).read() == open(out_path).read()

    def test_export_multiple(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'pointclouds.usda')

        # Export some meshes using default scene paths
        usd.export_pointclouds(pointclouds=[pointcloud for _ in range(self.num_multiple)],
                               file_path=out_path, points_type='usd_geom_points')

        # Test that can get their scene paths later
        scene_paths = usd.get_pointcloud_scene_paths(out_path)
        assert len(scene_paths) == self.num_multiple

    def test_export_multiple_instancer(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'pointclouds_instancer.usda')

        usd.export_pointclouds(pointclouds=[pointcloud for _ in range(self.num_multiple)],
                               file_path=out_path)

        # Test that can get their scene paths later
        scene_paths = usd.get_pointcloud_scene_paths(out_path)
        assert len(scene_paths) == self.num_multiple

    def test_import_single(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'pointcloud.usda')
        pointcloud_in = usd.import_pointcloud(file_path=out_path, scene_path=self.scene_path).points

        # Confirm imported pointcloud matches original input
        assert torch.allclose(pointcloud, pointcloud_in)

    def test_import_multiple(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'pointclouds.usda')
        pointcloud_in_list = usd.import_pointclouds(file_path=out_path)

        # Confirm imported pointcloud matches original input
        assert len(pointcloud_in_list) == self.num_multiple
        for pointcloud_in, colors_in, normals_in in pointcloud_in_list:
            assert torch.allclose(pointcloud, pointcloud_in)

    def test_import_single_instancer(self, out_dir, pointcloud_instancer):
        # Test that the read from UsdPointInstancer is the same as the read from UsdGeomPoints
        out_path = os.path.join(out_dir, 'pointcloud.usda')
        pointcloud_in, colors_in, normals_in = usd.import_pointcloud(file_path=out_path, scene_path=self.scene_path)

        # Confirm imported pointcloud matches original input
        assert torch.allclose(pointcloud_instancer, pointcloud_in)

    def test_import_multiple_instancer(self, out_dir, pointcloud_instancer):
        # Test that the read from UsdPointInstancer is the same as the read from UsdGeomPoints
        out_path = os.path.join(out_dir, 'pointclouds.usda')
        pointcloud_in_list = usd.import_pointclouds(file_path=out_path)

        # Confirm imported pointcloud matches original input
        assert len(pointcloud_in_list) == self.num_multiple
        for pointcloud_in, colors_in, normals_in in pointcloud_in_list:
            assert torch.allclose(pointcloud_instancer, pointcloud_in)

    def test_export_single_colors(self, out_dir, pointcloud_with_color):
        # Export a single pointcloud with colors
        pointcloud, color = pointcloud_with_color

        out_path = os.path.join(out_dir, 'pointcloud_colors.usda')
        usd.export_pointcloud(pointcloud=pointcloud, file_path=out_path, color=color, scene_path=self.scene_path, points_type='usd_geom_points')

        # Confirm exported USD matches golden file
        golden = os.path.join(out_dir, '../../../../samples/golden/pointcloud_GeomPoints_colors.usda')
        assert open(golden).read() == open(out_path).read()

    def test_import_single_color(self, out_dir, pointcloud):
        out_path = os.path.join(out_dir, 'pointcloud_colors.usda')
        pointcloud_in, color, _ = usd.import_pointcloud(file_path=out_path, scene_path=self.scene_path)

        # Confirm imported pointcloud matches original input
        assert torch.allclose(pointcloud, pointcloud_in)

        # Confirm that points have the same shape as color
        assert pointcloud_in.shape == color.shape

class TestVoxelGrid:
    def setup_method(self):
        self.scene_path = '/World/voxelgrid'
        self.num_multiple = 3

    @staticmethod
    def make_voxelgrid(mesh):
        resolution = 64
        voxelgrid = trianglemeshes_to_voxelgrids(mesh.vertices.unsqueeze(0), mesh.faces,
                                                 resolution)
        return voxelgrid[0].bool()

    @pytest.fixture(scope='class')
    def voxelgrid(self, mesh):
        return TestVoxelGrid.make_voxelgrid(mesh)

    def test_export_single(self, out_dir, voxelgrid):
        out_path = os.path.join(out_dir, 'voxelgrid.usda')
        usd.export_voxelgrid(file_path=out_path, voxelgrid=voxelgrid, scene_path=self.scene_path)

        # Confirm exported USD matches golden file
        golden = os.path.join(out_dir, '../../../../samples/golden/voxelgrid.usda')
        assert open(golden).read() == open(out_path).read()

    def test_export_multiple(self, out_dir, voxelgrid):
        out_path = os.path.join(out_dir, 'voxelgrids.usda')

        # Export multiple voxelgrids using default paths
        usd.export_voxelgrids(file_path=out_path, voxelgrids=[voxelgrid for _ in range(self.num_multiple)])

    def test_import_single(self, out_dir, voxelgrid):
        out_path = os.path.join(out_dir, 'voxelgrid.usda')
        voxelgrid_in = usd.import_voxelgrid(file_path=out_path, scene_path=self.scene_path)
        assert torch.equal(voxelgrid, voxelgrid_in)

    def test_import_multiple(self, out_dir, voxelgrid):
        out_path = os.path.join(out_dir, 'voxelgrids.usda')
        voxelgrid_in_list = usd.import_voxelgrids(file_path=out_path)

        # Confirm imported voxelgrid matches original input
        assert len(voxelgrid_in_list) == self.num_multiple
        for voxelgrid_in in voxelgrid_in_list:
            assert torch.equal(voxelgrid, voxelgrid_in)


class TestMisc:
    @pytest.fixture(scope='class')
    def voxelgrid(self, mesh):
        return TestVoxelGrid.make_voxelgrid(mesh)

    def test_get_authored_time_samples_untimed(self, out_dir, mesh, voxelgrid):
        out_path = os.path.join(out_dir, 'untimed.usda')
        usd.export_voxelgrid(file_path=out_path, voxelgrid=voxelgrid, scene_path='/World/voxelgrid')
        usd.export_mesh(out_path, scene_path='/World/meshes', vertices=mesh.vertices, faces=mesh.faces)

        times = usd.get_authored_time_samples(out_path)
        assert times == []

    def test_get_authored_time_samples_timed(self, out_dir, mesh, voxelgrid, pointcloud):
        out_path = os.path.join(out_dir, 'timed.usda')
        usd.export_voxelgrid(file_path=out_path, voxelgrid=voxelgrid, scene_path='/World/voxelgrid')
        times = usd.get_authored_time_samples(out_path)
        assert times == []

        usd.export_voxelgrid(file_path=out_path, voxelgrid=voxelgrid, scene_path='/World/voxelgrid', time=1)
        times = usd.get_authored_time_samples(out_path)
        assert times == [1]

        usd.export_mesh(out_path, scene_path='/World/meshes', vertices=mesh.vertices, faces=mesh.faces, time=20)
        usd.export_mesh(out_path, scene_path='/World/meshes', vertices=mesh.vertices, faces=None, time=250)
        times = usd.get_authored_time_samples(out_path)
        assert times == [1.0, 20.0, 250.0]

        usd.export_pointcloud(out_path, pointcloud)
