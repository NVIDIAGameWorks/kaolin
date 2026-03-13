# Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES.
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
import glob
import filecmp

import torch
import pytest
from pxr import Usd, UsdGeom

from kaolin.io import usd, obj, gltf
from kaolin.io import utils
from kaolin.rep import SurfaceMesh
from kaolin.utils.testing import print_namedtuple_attributes, print_dict_attributes, \
    check_tensor_attribute_shapes, contained_torch_equal, check_allclose, file_contents_equal

__test_dir = os.path.dirname(os.path.realpath(__file__))
__samples_path = os.path.join(__test_dir, os.pardir, os.pardir, os.pardir, os.pardir, 'samples')


def io_data_path(fname):
    """ Return path relative to tests/samples/io"""
    return os.path.join(__samples_path, 'io', fname)


def samples_data_path(*args):
    return os.path.join(__samples_path, *args)


def read_raw_usd_attributes(fname_or_stage):
    if type(fname_or_stage) == str:
        stage = Usd.Stage.Open(fname_or_stage)
    else:
        stage = fname_or_stage

    paths = usd.utils.get_scene_paths(stage, prim_types=["Mesh"])
    return [usd.get_raw_mesh_prim_geometry(stage.GetPrimAtPath(p), with_normals=True, with_uvs=True, time=0)
            for p in paths]


@pytest.fixture(scope='class')
def out_dir():
    # Create temporary output directory
    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_out')
    os.makedirs(out_dir, exist_ok=True)
    yield out_dir
    shutil.rmtree(out_dir)

@pytest.fixture(scope='module')
def mesh():
    obj_mesh = obj.import_mesh(os.path.join(__samples_path, 'rocket.obj'), with_normals=True,
                               with_materials=True, error_handler=obj.skip_error_handler)
    return obj_mesh

@pytest.fixture(scope='class')
def golden_overwrite_path(out_dir, mesh):
    path = os.path.join(out_dir, 'golden_overwrite.usda')
    usd.export_mesh(path, vertices=mesh.vertices, faces=mesh.faces)
    return path


@pytest.fixture(scope='module')
def mesh_alt():
    obj_mesh = obj.import_mesh(os.path.join(__samples_path, 'model.obj'), with_normals=True,
                               with_materials=True, error_handler=obj.skip_error_handler)
    return obj_mesh

@pytest.fixture(scope='module')
def mesh_path():
    return os.path.join(__samples_path, 'golden', 'mesh.usda')   # rocket  # TODO: rename

@pytest.fixture(scope='module')
def homogenized_golden_path():
    return os.path.join(__samples_path, 'golden', 'rocket_homogenized.usda')

@pytest.fixture(scope='module')
def homo_mesh_path():
    return os.path.join(__samples_path, 'rocket_model_hom.usda')

@pytest.fixture(scope='module')
def mixed_mesh_path():
    return os.path.join(__samples_path, 'mixed.usdc')

@pytest.fixture(scope='module')
def hetero_mesh_path():
    return os.path.join(__samples_path, 'rocket_hetero.usd')

@pytest.fixture(scope='module')
def hetero_subsets_materials_mesh_path():
    return os.path.join(__samples_path, 'rocket_hetero_subsets_materials.usd')

def _unset_material_names(materials):
    if materials is None:
        return
    for m in materials:
        if type(m) == list:
            for m2 in m:
                m2.material_name = ''
        else:
            m.material_name = ''

class TestMeshes:
    @pytest.fixture(scope='class')
    def scene_paths(self):
        num_meshes = 2
        return [f'/World/mesh_{i}' for i in range(num_meshes)]

    def test_input_as_expected(self, mesh, mesh_alt):
        # DEBUG INFORMATION (uncomment when debugging)
        # print_namedtuple_attributes(mesh, 'test input read from rocket.obj')
        # print_namedtuple_attributes(mesh_alt, 'test input read from model.obj')

        # Note normals are not vertex normals here, they are in fact normals not associated to vertices
        # in the order that is in vertices.
        assert check_tensor_attribute_shapes(
            mesh, throw=True,
            vertices=[426, 3], faces=[832, 3], uvs=[493, 2],
            face_uvs_idx=[832, 3],
            normals=[430, 3],
            face_normals_idx=[832, 3])

        assert check_tensor_attribute_shapes(
            mesh_alt, throw=True,
            vertices=[482, 3], faces=[960, 3], uvs=[610, 2],
            face_uvs_idx=[960, 3],
            normals=[584, 3],
            face_normals_idx=[960, 3])

    def test_export_single(self, scene_paths, out_dir, mesh, mesh_path):
        out_path = os.path.join(out_dir, f'single_mesh.usda')

        # Export a mesh
        stage = usd.export_mesh(out_path, scene_paths[0], mesh.vertices, mesh.faces)

        # Check against golden usd file path
        assert filecmp.cmp(mesh_path, out_path)
        #assert open(mesh_path).read() == open(out_path).read()

    @pytest.mark.parametrize('input_stage', [False, True])
    @pytest.mark.parametrize('with_paths', [False, True])
    def test_export_import_multiple(self, scene_paths, out_dir, mesh, mesh_alt, input_stage, with_paths):
        out_path = os.path.join(out_dir, 'partial_meshes.usda')

        # Export some meshes
        meshes = [mesh, mesh_alt]
        vertices_list = [mesh.vertices, mesh_alt.vertices]
        faces_list = [mesh.faces, mesh_alt.faces]
        actual_face_normals_list = [mesh.normals[mesh.face_normals_idx],
                                    mesh_alt.normals[mesh_alt.face_normals_idx]]
        # Try exporting just vertices and faces
        usd.export_meshes(out_path, scene_paths, vertices_list, faces_list, overwrite=True)

        # Now export all the attributes
        out_path = os.path.join(out_dir, 'meshes.usda')
        # TODO: properly export with materials once can convert OBJ materials to USD
        usd.export_meshes(out_path, scene_paths, vertices_list, faces_list,
                          uvs=[mesh.uvs, mesh_alt.uvs],
                          face_uvs_idx=[mesh.face_uvs_idx, mesh_alt.face_uvs_idx],
                          face_normals=actual_face_normals_list,
                          overwrite=True)

        path_or_stage = Usd.Stage.Open(out_path) if input_stage else out_path

        # TODO: once above is fixed use with_materials=True here
        meshes_in = usd.import_meshes(path_or_stage, scene_paths if with_paths else None, with_normals=True, return_list=False)
        assert len(meshes_in) == len(meshes)
        for i, orig_mesh in enumerate(meshes):
            in_mesh = meshes_in[scene_paths[i]]
            # DEBUG INFORMATION (uncomment when debugging)
            # print_namedtuple_attributes(orig_mesh, f'Orig mesh [{i}]')
            # print_namedtuple_attributes(in_mesh, f'Imported mesh [{i}]')

            # We check key attributes
            assert contained_torch_equal({
                'vertices': orig_mesh.vertices,
                'faces': orig_mesh.faces,
                'uvs': orig_mesh.uvs,
                'face_uvs_idx': orig_mesh.face_uvs_idx,
                'face_normals': actual_face_normals_list[i]
            },
            {
                'vertices': in_mesh.vertices,
                'faces': in_mesh.faces,
                'uvs': in_mesh.uvs,
                'face_uvs_idx': in_mesh.face_uvs_idx,
                'face_normals': in_mesh.face_normals
            })

        # Test that import_mesh returns a NONE-batched (flattened) mesh with all meshes concatenated
        mesh_in = usd.import_mesh(path_or_stage, with_normals=True)
        assert mesh_in.batching == SurfaceMesh.Batching.NONE
        assert len(mesh_in.vertices) == (len(mesh.vertices) + len(mesh_alt.vertices))
        assert len(mesh_in.faces) == (len(mesh.faces) + len(mesh_alt.faces))
        assert contained_torch_equal({
            'vertices': torch.cat([mesh.vertices, mesh_alt.vertices], dim=0),
            'faces': torch.cat([mesh.faces, mesh_alt.faces + mesh.vertices.shape[0]], dim=0),
            'uvs': torch.cat([mesh.uvs, mesh_alt.uvs], dim=0),
            'face_uvs_idx': torch.cat([mesh.face_uvs_idx, mesh_alt.face_uvs_idx + mesh.uvs.shape[0]], dim=0),
            'face_normals': torch.cat(actual_face_normals_list, dim=0)
        },
        {
            'vertices': mesh_in.vertices,
            'faces': mesh_in.faces,
            'uvs': mesh_in.uvs,
            'face_uvs_idx': mesh_in.face_uvs_idx,
            'face_normals': mesh_in.face_normals
         })

    def test_import_bad_prim(self, scene_paths, mesh_path):
        """Test that import fails when reaching invalid prims"""
        with pytest.raises(ValueError):
            usd.import_meshes(mesh_path, ['/foo'] + scene_paths, return_list=False)

    def test_get_mesh_scene_paths(self, scene_paths, out_dir, mesh, mesh_alt):
        """get_mesh_scene_paths returns expected paths from exported stage"""
        out_path = os.path.join(out_dir, 'meshes_scene_paths.usda')
        usd.export_meshes(out_path, scene_paths,
                          vertices=[mesh.vertices, mesh_alt.vertices],
                          faces=[mesh.faces, mesh_alt.faces], overwrite=True)
        actual = usd.get_mesh_scene_paths(out_path)
        assert set(str(p) for p in actual) == set(scene_paths)

    def test_get_mesh_scene_paths_scene_path(self, out_dir, mesh, mesh_alt):
        """get_mesh_scene_paths with scene_path returns only paths under that prefix"""
        all_scene_paths = ['/World/Foo/mesh_0', '/World/Foo/mesh_1', '/World/Bar/mesh_0']
        out_path = os.path.join(out_dir, 'meshes_scene_path.usda')
        usd.export_meshes(out_path, all_scene_paths,
                          vertices=[mesh.vertices, mesh_alt.vertices, mesh.vertices],
                          faces=[mesh.faces, mesh_alt.faces, mesh.faces], overwrite=True)
        actual = usd.get_mesh_scene_paths(out_path, scene_path='/World/Foo')
        assert set(str(p) for p in actual) == {'/World/Foo/mesh_0', '/World/Foo/mesh_1'}

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_meshes_auto_discover(self, homo_mesh_path, input_stage):
        """import_meshes with scene_paths=None auto-discovers all mesh prims and returns a dict"""
        all_scene_paths = usd.get_mesh_scene_paths(homo_mesh_path)
        assert len(all_scene_paths) == 2

        path_or_stage = Usd.Stage.Open(homo_mesh_path) if input_stage else homo_mesh_path
        meshes_explicit = usd.import_meshes(homo_mesh_path, all_scene_paths, with_normals=True, return_list=False)
        meshes_auto = usd.import_meshes(path_or_stage, with_normals=True, return_list=False)

        assert set(meshes_auto.keys()) == set(str(p) for p in all_scene_paths)
        for sp in all_scene_paths:
            ref_mesh = meshes_explicit[str(sp)]
            in_mesh = meshes_auto[str(sp)]
            assert contained_torch_equal(
                {'vertices': ref_mesh.vertices, 'faces': ref_mesh.faces},
                {'vertices': in_mesh.vertices, 'faces': in_mesh.faces}
            )

    @pytest.mark.parametrize('input_stage', [False, True])
    @pytest.mark.parametrize('triangulate', [False, True])
    def test_import_hetero_fail(self, hetero_mesh_path, input_stage, triangulate):
        """Test that import fails when importing heterogeneous mesh without handler"""
        path_or_stage = Usd.Stage.Open(hetero_mesh_path) if input_stage else hetero_mesh_path

        if not triangulate:
            with pytest.raises(utils.NonHomogeneousMeshError):
                usd.import_meshes(file_path_or_stage=path_or_stage, scene_paths=['/Root/Rocket'], triangulate=triangulate, return_list=False)

            with pytest.raises(utils.NonHomogeneousMeshError):
                usd.import_mesh(file_path_or_stage=path_or_stage, scene_path='/Root', triangulate=triangulate)

        else:
            usd.import_meshes(file_path_or_stage=path_or_stage, scene_paths=['/Root/Rocket'], triangulate=triangulate, return_list=False)
            usd.import_mesh(file_path_or_stage=path_or_stage, scene_path='/Root', triangulate=triangulate)

    @pytest.mark.parametrize('input_stage', [False, True])
    @pytest.mark.parametrize('triangulate', [False, True])
    def test_import_hetero_skip(self, scene_paths, hetero_mesh_path, homo_mesh_path,
                                mixed_mesh_path, input_stage, triangulate):
        """Test that import skips mesh when importing heterogeneous mesh with skip handler"""
        path_or_stage = Usd.Stage.Open(hetero_mesh_path) if input_stage else hetero_mesh_path
        hetero_scene_paths = usd.get_mesh_scene_paths(hetero_mesh_path)
        meshes = usd.import_meshes(path_or_stage, hetero_scene_paths,
                                   heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_skip, return_list=False)
        assert len(meshes) == 0
        meshes2 = usd.import_meshes(path_or_stage,
                                    heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_skip, return_list=False)
        assert meshes.keys() == meshes2.keys()
        for k in meshes.keys():
            assert contained_torch_equal(meshes[k].as_dict(), meshes2[k].as_dict())
                                   
        # All sub-meshes are skipped, so import_mesh has nothing to return.
        mesh = usd.import_mesh(path_or_stage,
                               heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_skip)
        assert mesh is None

        # Test skip on a batch of homogeneous meshes
        path_or_stage = Usd.Stage.Open(homo_mesh_path) if input_stage else homo_mesh_path
        homo_scene_paths = usd.get_mesh_scene_paths(homo_mesh_path)
        meshes = usd.import_meshes(path_or_stage, homo_scene_paths,
                                   heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_skip, return_list=False)
        assert len(meshes) == 2
        meshes2 = usd.import_meshes(path_or_stage,
                                    heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_skip, return_list=False)
        assert meshes.keys() == meshes2.keys()
        for k in meshes.keys():
            assert contained_torch_equal(meshes[k].as_dict(), meshes2[k].as_dict())
        # import_mesh must produce the same flattened result built from the per-prim meshes above.
        mesh = usd.import_mesh(path_or_stage,
                               heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_skip)
        expected = SurfaceMesh.flatten(list(meshes.values()), group_materials_by_name=True)
        assert mesh.batching == SurfaceMesh.Batching.NONE
        assert contained_torch_equal(mesh.as_dict(), expected.as_dict())

        # Test skip on a batch of mixed homogeneous and heterogeneous meshes
        # Note we can't export heterogeneous meshes, so previous test did not work
        path_or_stage = Usd.Stage.Open(mixed_mesh_path) if input_stage else mixed_mesh_path
        mixed_scene_paths = usd.get_mesh_scene_paths(mixed_mesh_path)
        meshes = usd.import_meshes(path_or_stage, mixed_scene_paths,
                                   heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_skip, return_list=False)
        assert len(meshes) == 1
        meshes2 = usd.import_meshes(path_or_stage,
                                    heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_skip, return_list=False)
        assert meshes.keys() == meshes2.keys()
        for k in meshes.keys():
            assert contained_torch_equal(meshes[k].as_dict(), meshes2[k].as_dict())
        # Only the homogeneous sub-mesh survives; import_mesh returns it with its transform applied.
        mesh = usd.import_mesh(path_or_stage,
                               heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_skip)
        expected = next(iter(meshes.values())).as_transformed()
        assert mesh.batching == SurfaceMesh.Batching.NONE
        assert contained_torch_equal(mesh.as_dict(), expected.as_dict())


    @pytest.mark.parametrize('use_triangulate_shortcut', [True, False])
    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_hetero_homogenize_import_meshes(self, out_dir, hetero_mesh_path, homogenized_golden_path,
                                                    input_stage, use_triangulate_shortcut):
        """Test that imports homogeneous mesh when importing heterogeneous mesh with naive homogenize handler"""
        # TODO(jlafleche) Render meshes before/after homogenize operation
        path_or_stage = Usd.Stage.Open(hetero_mesh_path) if input_stage else hetero_mesh_path
        if use_triangulate_shortcut:
            kwargs = {'triangulate': True}
        else:
            kwargs = {'heterogeneous_mesh_handler': utils.mesh_handler_naive_triangulate}
        mesh = usd.import_meshes(path_or_stage, ['/Root/Rocket'], **kwargs, return_list=False)
        # Confirm we now have a triangle mesh
        assert mesh['/Root/Rocket'].faces.size(1) == 3

        out_path = os.path.join(out_dir, 'homogenized.usda')
        usd.export_mesh(out_path, '/World/Rocket', vertices=mesh['/Root/Rocket'].vertices, faces=mesh['/Root/Rocket'].faces, overwrite=True)

        # Confirm exported USD matches golden file
        assert filecmp.cmp(homogenized_golden_path, out_path)
        #assert open(homogenized_golden_path).read() == open(out_path).read()

    @pytest.mark.parametrize('use_triangulate_shortcut', [True, False])
    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_hetero_homogenize_import_mesh(self, out_dir,  hetero_mesh_path, homogenized_golden_path,
                                                  input_stage, use_triangulate_shortcut):
        """Test that imports homogeneous mesh when importing heterogeneous mesh with naive homogenize handler"""
        # TODO(jlafleche) Render meshes before/after homogenize operation
        path_or_stage = Usd.Stage.Open(hetero_mesh_path) if input_stage else hetero_mesh_path
        if use_triangulate_shortcut:
            kwargs = {'triangulate': True}
        else:
            kwargs = {'heterogeneous_mesh_handler': utils.mesh_handler_naive_triangulate}
        mesh = usd.import_mesh(path_or_stage, scene_path='/Root', **kwargs)

        # Confirm we now have a triangle mesh
        assert mesh.faces.size(1) == 3

        out_path = os.path.join(out_dir, 'homogenized.usda')
        usd.export_mesh(out_path, '/World/Rocket', vertices=mesh.vertices, faces=mesh.faces, overwrite=True)

        # Confirm exported USD matches golden file
        assert filecmp.cmp(homogenized_golden_path, out_path)
        #assert open(homogenized_golden_path).read() == open(out_path).read()

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_with_transform(self, scene_paths, out_dir, hetero_mesh_path, input_stage):
        """Test that import_mesh applies the local-to-world transform to vertices and import_meshes returns it separately"""
        path_or_stage = Usd.Stage.Open(hetero_mesh_path) if input_stage else hetero_mesh_path
        mesh = usd.import_mesh(path_or_stage, scene_path='/Root',
                               heterogeneous_mesh_handler=utils.mesh_handler_naive_triangulate)
        out_path = os.path.join(out_dir, 'transformed.usda')
        stage = usd.create_stage(out_path)
        prim = usd.add_mesh(stage, '/World/Rocket', vertices=mesh.vertices, faces=mesh.faces)
        UsdGeom.Xformable(prim).AddTranslateOp().Set((10, 10, 10))
        stage.Save()

        # import_mesh applies the transform: vertices are in world space
        mesh_import = usd.import_mesh(out_path)
        assert torch.allclose(mesh_import.vertices, mesh.vertices + torch.tensor([10., 10., 10.]))
        assert mesh_import.transform is None

        # import_meshes returns local-space vertices with the transform set
        meshes_dict = usd.import_meshes(out_path, return_list=False)
        mesh_local = meshes_dict['/World/Rocket']
        assert torch.allclose(mesh_local.vertices, mesh.vertices)
        assert mesh_local.transform is not None
        assert mesh_local.transform.shape == (4, 4)
        assert torch.allclose(mesh_local.transform[:3, 3], torch.tensor([10., 10., 10.]))

    def test_export_import_roundtrip_with_transform(self, out_dir):
        """Test that the local-to-world transform is preserved via import_meshes, and import_mesh applies it to vertices"""
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int64)
        transform = torch.tensor([[2., 0., 0., 0.5],
                                  [0., 0., -2., 0.],
                                  [0., 2., 0., 0.],
                                  [0., 0., 0., 1.]], dtype=torch.float32)
        out_path = os.path.join(out_dir, 'mesh_transform_roundtrip.usda')

        usd.export_mesh(out_path, scene_path='/World/Meshes/mesh_0',
                        vertices=vertices, faces=faces, local_to_world=transform, overwrite=True)

        # import_meshes preserves the transform without applying it
        meshes_dict = usd.import_meshes(out_path, ['/World/Meshes/mesh_0'], return_list=False)
        mesh_local = meshes_dict['/World/Meshes/mesh_0']
        assert torch.allclose(mesh_local.vertices, vertices)
        assert mesh_local.transform is not None
        assert torch.allclose(mesh_local.transform, transform, atol=1e-5)

        # import_mesh applies the transform: vertices are in world space
        mesh_import = usd.import_mesh(out_path, scene_path='/World/Meshes/mesh_0')
        assert mesh_import.transform is None
        # Verify world-space vertices: transform @ [x,y,z,1]^T
        verts_h = torch.cat([vertices, torch.ones(vertices.shape[0], 1)], dim=1)
        expected_verts = (verts_h @ transform.T)[:, :3]
        assert torch.allclose(mesh_import.vertices, expected_verts, atol=1e-5)

    def test_export_meshes_batched_local_to_world(self, out_dir):
        """export_meshes accepts local_to_world as a (N, 4, 4) batched tensor (per-mesh)
        and as a single (4, 4) tensor (broadcast to every mesh), aligned with SurfaceMesh.transform."""
        vertices = [torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], dtype=torch.float32),
                    torch.tensor([[0., 0., 0.], [0., 1., 0.], [0., 0., 1.]], dtype=torch.float32)]
        faces = [torch.tensor([[0, 1, 2]], dtype=torch.int64),
                 torch.tensor([[0, 1, 2]], dtype=torch.int64)]
        transform_0 = torch.tensor([[2., 0., 0., 0.5],
                                    [0., 0., -2., 0.],
                                    [0., 2., 0., 0.],
                                    [0., 0., 0., 1.]], dtype=torch.float32)
        transform_1 = torch.tensor([[1., 0., 0., 3.],
                                    [0., 1., 0., -1.],
                                    [0., 0., 1., 2.],
                                    [0., 0., 0., 1.]], dtype=torch.float32)
        scene_paths = ['/World/Meshes/mesh_0', '/World/Meshes/mesh_1']

        # (N, 4, 4) per-mesh
        per_mesh = torch.stack([transform_0, transform_1], dim=0)
        out_path = os.path.join(out_dir, 'export_meshes_batched_per_mesh.usda')
        usd.export_meshes(out_path, scene_paths, vertices=vertices, faces=faces,
                          local_to_world=per_mesh, overwrite=True)
        meshes_dict = usd.import_meshes(out_path, scene_paths, return_list=False)
        assert torch.allclose(meshes_dict[scene_paths[0]].transform, transform_0, atol=1e-5)
        assert torch.allclose(meshes_dict[scene_paths[1]].transform, transform_1, atol=1e-5)

        # (4, 4) broadcast
        out_path = os.path.join(out_dir, 'export_meshes_batched_broadcast.usda')
        usd.export_meshes(out_path, scene_paths, vertices=vertices, faces=faces,
                          local_to_world=transform_0, overwrite=True)
        meshes_dict = usd.import_meshes(out_path, scene_paths, return_list=False)
        assert torch.allclose(meshes_dict[scene_paths[0]].transform, transform_0, atol=1e-5)
        assert torch.allclose(meshes_dict[scene_paths[1]].transform, transform_0, atol=1e-5)

    def test_import_vertex_interpolation_normals(self, out_dir):
        """Normals authored with interpolation='vertex' must round-trip into SurfaceMesh.vertex_normals.

        Kaolin's own export only writes faceVarying normals, so this test authors a USD
        with vertex-interpolated normals via the raw USD API to exercise the
        vertex-interpolation branch of `_get_mesh_prim_attributes` / `set_normals`.
        """
        vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                                dtype=torch.float32)
        faces = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.int64)
        vertex_normals = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.], [-1., 0., 0.]],
                                      dtype=torch.float32)

        out_path = os.path.join(out_dir, 'mesh_vertex_normals.usda')
        stage = usd.create_stage(out_path)
        prim = usd.add_mesh(stage, '/World/Mesh', vertices=vertices, faces=faces)
        usd_mesh = UsdGeom.Mesh(prim)
        usd_mesh.GetNormalsAttr().Set(vertex_normals.numpy())
        UsdGeom.PointBased(usd_mesh).SetNormalsInterpolation('vertex')
        stage.Save()

        meshes_dict = usd.import_meshes(out_path, ['/World/Mesh'], with_normals=True, return_list=False)
        mesh_local = meshes_dict['/World/Mesh']
        assert mesh_local.vertex_normals is not None
        assert torch.allclose(mesh_local.vertex_normals, vertex_normals)

    @pytest.mark.parametrize('function_variant', ['export_mesh', 'export_meshes'])
    @pytest.mark.parametrize('input_stage', [False, True])
    @pytest.mark.parametrize('overwrite_textures', [False, True])
    def test_import_material_subsets(self, scene_paths, out_dir, hetero_subsets_materials_mesh_path,
                                     input_stage, function_variant, overwrite_textures):
        """Test that imports materials from mesh with subsets"""
        if input_stage:
            path_or_stage = Usd.Stage.Open(hetero_subsets_materials_mesh_path)
        else:
            path_or_stage = hetero_subsets_materials_mesh_path

        # Read and homogenize the mesh
        mesh = usd.import_mesh(path_or_stage, scene_path='/Root',
                               heterogeneous_mesh_handler=utils.mesh_handler_naive_triangulate,
                               with_normals=True, with_materials=True)
        # avoid any automatic computation of normals
        mesh.unset_attributes_return_none = True

        # Confirm we now have a triangulated mesh
        assert mesh.faces.size(1) == 3

        # Check material assignments
        expected_material_assignments = torch.zeros((mesh.faces.size(0),), dtype=torch.short)
        expected_material_assignments[:18 * 2] = 1  # first 18 quads use 2nd material
        expected_material_assignments[788 + 18:] = 2  # last faces (offset by extra triangles created by quads) use 3rd material
        assert torch.equal(mesh.material_assignments, expected_material_assignments)

        # Also read in the golden mesh
        golden_path = samples_data_path('golden', 'rocket_homogenized_materials.usda')
        golden_mesh = usd.import_mesh(golden_path,
                                      heterogeneous_mesh_handler=utils.mesh_handler_naive_triangulate,
                                      with_normals=True, with_materials=True)
        golden_mesh.unset_attributes_return_none = True

        # Spot check against raw USD attributes
        raw_attributes = read_raw_usd_attributes(path_or_stage)[0]
        assert torch.sum(raw_attributes['face_sizes'] - 2) == 832, "Bug in read_raw_usd_attributes"
        assert mesh.faces.size(0) == torch.sum(raw_attributes['face_sizes'] - 2)   # Expected for fan triangulation

        # Spot check against raw mesh attributes for a few initial quads
        assert torch.equal(raw_attributes['face_sizes'][:5], torch.tensor([4, 4, 4, 4, 4]))  # make sure first few face sizes are as expected
        for quad_idx in range(5):
            tri_idx = quad_idx * 2  # Only true for the first few quads

            # Check the processed mesh
            expected_vertices = raw_attributes['vertices'][raw_attributes['faces'][quad_idx * 4: quad_idx * 4 + 3], :]
            expected_normals = raw_attributes['normals'][quad_idx * 4: quad_idx * 4 + 3, :]
            expected_uvs = raw_attributes['uvs'][quad_idx * 4: quad_idx * 4 + 3, :]
            expected_uvs[:, 1] = 1 - expected_uvs[:, 1]
            assert torch.allclose(mesh.vertices[mesh.faces[tri_idx, :], :], expected_vertices)
            assert torch.allclose(mesh.face_normals[tri_idx, ...], expected_normals)
            assert torch.allclose(mesh.uvs[mesh.face_uvs_idx[tri_idx, :], :], expected_uvs)

            # Also sanity check the golden mesh (to catch human error)
            assert torch.allclose(golden_mesh.vertices[mesh.faces[tri_idx, :], :], expected_vertices)
            assert torch.allclose(golden_mesh.face_normals[tri_idx, ...], expected_normals)
            assert torch.allclose(golden_mesh.uvs[mesh.face_uvs_idx[tri_idx, :], :], expected_uvs)
        # Write the homogenized mesh to file
        out_path = os.path.join(out_dir, 'rocket_homogenized_materials.usda')
        num_writes = 1 if overwrite_textures else 2  # write twice to ensure texture files are different
        for i in range(num_writes):
            if function_variant == 'export_mesh':
                usd.export_mesh(out_path, '/World/Rocket', vertices=mesh.vertices, faces=mesh.faces,
                                face_uvs_idx=mesh.face_uvs_idx, face_normals=mesh.face_normals, uvs=mesh.uvs,
                                material_assignments=mesh.material_assignments, materials=mesh.materials,
                                overwrite=True, overwrite_textures=overwrite_textures)
            else:
                usd.export_meshes(out_path, ['/World/Rocket'], vertices=[mesh.vertices], faces=[mesh.faces],
                                  face_uvs_idx=[mesh.face_uvs_idx], face_normals=[mesh.face_normals], uvs=[mesh.uvs],
                                  material_assignments=[mesh.material_assignments],
                                  materials=[mesh.materials],
                                  overwrite=True, overwrite_textures=overwrite_textures)

        # Confirm exported USD matches golden file
        # Note: we can't match UVs due to small arithmetic changes introduced due to UV convention fixing
        files_equal = file_contents_equal(golden_path, out_path, exclude_pattern='primvars:st =')
        if overwrite_textures:
            assert files_equal
        else:
            assert not files_equal  # texture references should be different

        # Confirm we read identical mesh after writing
        reimported_mesh = usd.import_mesh(out_path, scene_path='/World/Rocket', with_materials=True, with_normals=True)
        reimported_mesh.unset_attributes_return_none = True

        # Since comparison of materials is not implemented, we override materials with diffuse colors first
        assert len(mesh.materials) == len(reimported_mesh.materials)
        # Unset material names, which will not match after write
        _unset_material_names(mesh.materials)
        _unset_material_names(reimported_mesh.materials)
        assert contained_torch_equal(mesh, reimported_mesh, print_error_context='')

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_with_material(self, scene_paths, out_dir, hetero_subsets_materials_mesh_path, input_stage):
        """Test that imports materials from mesh with subsets"""
        if input_stage:
            path_or_stage = Usd.Stage.Open(hetero_subsets_materials_mesh_path)
        else:
            path_or_stage = hetero_subsets_materials_mesh_path

        mesh = usd.import_mesh(path_or_stage, scene_path='/Root',
                               heterogeneous_mesh_handler=utils.mesh_handler_naive_triangulate,
                               with_materials=False)
        assert mesh.materials is None
        assert mesh.material_assignments is None

        mesh = usd.import_mesh(path_or_stage, scene_path='/Root',
                               heterogeneous_mesh_handler=utils.mesh_handler_naive_triangulate,
                               with_materials=True)
        assert mesh.materials is not None
        assert mesh.material_assignments is not None

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


    def test_fail_export_twice(self, out_dir, mesh):
        out_path = os.path.join(out_dir, 'fail_exported_twice.usda')
        usd.export_mesh(out_path, vertices=mesh.vertices, faces=mesh.faces)
        with pytest.raises(FileExistsError):
            usd.export_mesh(out_path, vertices=mesh.vertices, faces=mesh.faces)


    def test_export_twice(self, out_dir, mesh, golden_overwrite_path):
        out_path = os.path.join(out_dir, 'exported_twice.usda')
        usd.export_mesh(out_path, vertices=mesh.vertices + 10., faces=mesh.faces)
        usd.export_mesh(out_path, vertices=mesh.vertices, faces=mesh.faces,
                        overwrite=True)
        assert filecmp.cmp(out_path, golden_overwrite_path)

    def test_export_overwrite_full_file(self, out_dir, mesh, golden_overwrite_path):
        out_path = os.path.join(out_dir, 'exported_overwrite_full_file.usda')
        usd.export_meshes(out_path,
                          vertices=[mesh.vertices + 10, mesh.vertices + 20],
                          faces=[mesh.faces, mesh.faces])
        usd.export_mesh(out_path, vertices=mesh.vertices, faces=mesh.faces,
                        overwrite=True)
        assert filecmp.cmp(out_path, golden_overwrite_path)

    def test_export_overwrite_pointcloud(self, out_dir, mesh, golden_overwrite_path):
        out_path = os.path.join(out_dir, 'exported_overwrite_pointcloud.usda')
        usd.export_pointcloud(out_path, pointcloud=torch.rand((100, 3)))
        usd.export_mesh(out_path, vertices=mesh.vertices, faces=mesh.faces,
                        overwrite=True)
        assert filecmp.cmp(out_path, golden_overwrite_path)

    def test_export_overwrite_voxelgrid(self, out_dir, mesh, golden_overwrite_path):
        out_path = os.path.join(out_dir, 'exported_overwrite_voxelgrid.usda')
        usd.export_voxelgrid(out_path, voxelgrid=(torch.rand(32, 32, 32) > 0.7))
        usd.export_mesh(out_path, vertices=mesh.vertices, faces=mesh.faces,
                        overwrite=True)
        assert filecmp.cmp(out_path, golden_overwrite_path)

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_st_indices_facevarying(self, out_dir, mesh, input_stage):
        out_path = os.path.join(out_dir, 'st_indices.usda')
        uvs = torch.rand((mesh.faces.view(-1).size(0), 2))
        scene_path = '/World/mesh_0'
        face_uvs_idx = (torch.rand(mesh.faces.shape[:2]) * 99).long()
        usd.export_mesh(out_path, scene_path=scene_path, vertices=mesh.vertices,
                        faces=mesh.faces, uvs=uvs, face_uvs_idx=face_uvs_idx,
                        overwrite=True)

        # check that interpolation was set correctly to 'faceVarying'
        stage = Usd.Stage.Open(out_path)
        pv = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(scene_path)).GetPrimvar('st')
        assert pv.GetInterpolation() == 'faceVarying'

        if input_stage:
            path_or_stage = Usd.Stage.Open(out_path)
        else:
            path_or_stage = out_path
        mesh_in = usd.import_mesh(path_or_stage)
        assert torch.allclose(mesh_in.uvs, uvs)

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_st_no_indices_vertex(self, out_dir, mesh, input_stage):
        out_path = os.path.join(out_dir, 'st_no_indices_vertex.usda')
        uvs = torch.rand((mesh.vertices.size(0), 2))
        scene_path = '/World/mesh_0'
        usd.export_mesh(out_path, scene_path=scene_path, vertices=mesh.vertices,
                        faces=mesh.faces, uvs=uvs, overwrite=True)

        # check that interpolation was set correctly to 'vertex'
        stage = Usd.Stage.Open(out_path)
        pv = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(scene_path)).GetPrimvar('st')
        assert pv.GetInterpolation() == 'vertex'

        if input_stage:
            path_or_stage = Usd.Stage.Open(out_path)
        else:
            path_or_stage = out_path
        mesh_in = usd.import_mesh(path_or_stage)
        assert torch.allclose(mesh_in.uvs, uvs)

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_st_no_indices_facevarying(self, out_dir, mesh, input_stage):
        out_path = os.path.join(out_dir, 'st_no_indices_face_varying.usda')
        uvs = torch.rand((mesh.faces.size(0) * mesh.faces.size(1), 2))
        scene_path = '/World/mesh_0'
        usd.export_mesh(out_path, scene_path=scene_path, vertices=mesh.vertices,
                        faces=mesh.faces, uvs=uvs, overwrite=True)

        # check that interpolation was set correctly to 'faceVarying'
        stage = Usd.Stage.Open(out_path)
        pv = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(scene_path)).GetPrimvar('st')
        assert pv.GetInterpolation() == 'faceVarying'

        if input_stage:
            path_or_stage = Usd.Stage.Open(out_path)
        else:
            path_or_stage = out_path
        mesh_in = usd.import_mesh(path_or_stage)
        assert torch.allclose(mesh_in.uvs, uvs)

    def test_import_st_no_indices_uniform(self, out_dir, mesh):
        out_path = os.path.join(out_dir, 'st_no_indices_face_uniform.usda')
        uvs = torch.rand((mesh.faces.size(0), 2))
        scene_path = '/World/mesh_0'
        usd.export_mesh(out_path, scene_path=scene_path, vertices=mesh.vertices,
                        faces=mesh.faces, uvs=uvs)

        # check that interpolation was set correctly to 'uniform'
        stage = Usd.Stage.Open(out_path)
        pv = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(scene_path)).GetPrimvar('st')

        assert pv.GetInterpolation() == 'uniform'

        # TODO(jlafleche) add support for `uniform` interpolation
        # mesh_in = usd.import_mesh(out_path)
        # assert torch.allclose(mesh_in.uvs, uvs)

    def test_export_only_face_normals(self, out_dir, mesh):
        out_path = os.path.join(out_dir, 'only_normals.usda')
        usd.export_mesh(out_path, face_normals=mesh.normals[mesh.face_normals_idx])
        mesh_in = usd.import_mesh(out_path, with_normals=True)
        assert torch.allclose(mesh_in.face_normals.view(-1, 3), mesh.normals[mesh.face_normals_idx].view(-1, 3))
        # TODO: support and test normals for various interpolations

    @pytest.mark.parametrize('with_normals', [False, True])
    @pytest.mark.parametrize('with_materials', [False, True])
    @pytest.mark.parametrize('flatten', [True, False])
    def test_import_triangulate(self, with_normals, with_materials, flatten):
        input_path = io_data_path(f'amsterdam.usda')  # Multiple quad meshes
        per_mesh_vertices = [4, 98, 98, 98, 386, 386, 98, 8, 98, 98, 98, 4, 4, 4, 386, 98, 4, 4]
        per_mesh_quads = [1, 96, 96, 96, 384, 384, 96, 6, 96, 96, 96, 1, 1, 1, 384, 96, 1, 1]
        if flatten:
            # import_mesh auto-discovers all prims and flattens them into one NONE-batched mesh
            orig = [usd.import_mesh(input_path, with_materials=with_materials, with_normals=with_normals)]
            triangulated = [usd.import_mesh(input_path, with_materials=with_materials,
                                            with_normals=with_normals, triangulate=True)]
            assert orig[0].batching == SurfaceMesh.Batching.NONE
            assert triangulated[0].batching == SurfaceMesh.Batching.NONE
            expected_num_vertices = [sum(per_mesh_vertices)]
            expected_num_quads = [sum(per_mesh_quads)]
        else:
            # Import as multiple meshes individually; iterate in scene_paths order rather than dict order.
            scene_paths = [str(p) for p in usd.get_mesh_scene_paths(input_path)]
            orig_dict = usd.import_meshes(
                input_path, scene_paths, with_materials=with_materials, with_normals=with_normals, return_list=False)
            triangulated_dict = usd.import_meshes(
                input_path, scene_paths, with_materials=with_materials, with_normals=with_normals,
                triangulate=True, return_list=False)
            orig = [orig_dict[p] for p in scene_paths]
            triangulated = [triangulated_dict[p] for p in scene_paths]
            assert len(orig) == 18
            assert len(triangulated) == 18
            expected_num_vertices = per_mesh_vertices
            expected_num_quads = per_mesh_quads

        for i in range(len(orig)):
            qmesh = orig[i]           # quad mesh
            tmesh = triangulated[i]   # triangle mesh

            # disallow automatic computation of properties (specifically face_normals can be auto-computed)
            qmesh.allow_auto_compute = False
            tmesh.allow_auto_compute = False

            check_tensor_attribute_shapes(
                qmesh, vertices=[expected_num_vertices[i], 3], faces=[expected_num_quads[i], 4])
            check_tensor_attribute_shapes(
                tmesh, vertices=[expected_num_vertices[i], 3], faces=[expected_num_quads[i] * 2, 3])
            assert torch.allclose(qmesh.vertices, tmesh.vertices)
            if with_materials:
                assert tmesh.materials is not None
                assert len(tmesh.materials) > 0
                assert contained_torch_equal([mat.diffuse_color for mat in qmesh.materials],
                                             [mat.diffuse_color for mat in tmesh.materials], approximate=True)
            else:
                assert tmesh.materials is None
                assert tmesh.material_assignments is None

            # Spot check all values for a given quad
            qidx = expected_num_quads[i] // 2  # quad index
            tidx = qidx * 2                    # triangle index
            assert torch.allclose(qmesh.vertices[qmesh.faces[qidx, :3], :], tmesh.vertices[tmesh.faces[tidx, :]])
            assert torch.allclose(qmesh.uvs[qmesh.face_uvs_idx[qidx, :3]], tmesh.uvs[tmesh.face_uvs_idx[tidx, :]])

            if with_normals:
                assert torch.allclose(qmesh.face_normals[qidx, :3, :], tmesh.face_normals[tidx, ...])
            else:
                assert tmesh.face_normals is None

            if with_materials:
                assert torch.equal(qmesh.material_assignments[qidx], tmesh.material_assignments[tidx])


class TestDiverseInputs:
    @pytest.fixture(scope='class')
    def expected_mesh_counts(self):
        return {
            'ico_smooth': 1,
            'ico_flat': 1,
            'fox': 1,
            'pizza': 1,
            'armchair': 3,
            'amsterdam': 18,
            'avocado': 1,
        }

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    @pytest.mark.parametrize('fname', ['ico_flat.usda', 'ico_flat.usdz', 'ico_smooth.usda', 'ico_smooth.usdz', 'fox.usdc', 'fox.usdz', 'pizza.usda', 'pizza.usdz', 'amsterdam.usda', 'amsterdam.usdz', 'armchair.usdc', 'armchair.usdz'])
    def test_read_write_read_consistency(self, device, fname, out_dir, expected_mesh_counts):
        model_name, input_ext = os.path.splitext(fname)
        output_ext = '.usda' if input_ext == '.usdz' else input_ext

        fpath = io_data_path(fname)
        # import as multiple meshes
        in_scene_paths = [str(p) for p in usd.get_mesh_scene_paths(fpath)]
        in_dict = usd.import_meshes(fpath, in_scene_paths, with_normals=True, with_materials=True, return_list=False)
        read_usd_mesh = SurfaceMesh.cat(
            [in_dict[p] for p in in_scene_paths], fixed_topology=False).to(device)
        assert len(read_usd_mesh) == expected_mesh_counts[model_name]
        assert len(read_usd_mesh.materials[0]) != 0

        # Now write the USD to file, read it back and make sure attributes match the original mesh
        out_path = os.path.join(out_dir, f'reexport_multi_{model_name}{output_ext}')
        usd.export_meshes(out_path, vertices=read_usd_mesh.vertices,
                          faces=read_usd_mesh.faces,
                          uvs=read_usd_mesh.uvs,
                          face_uvs_idx=read_usd_mesh.face_uvs_idx,
                          face_normals=read_usd_mesh.face_normals,
                          material_assignments=read_usd_mesh.material_assignments,
                          materials=read_usd_mesh.materials,
                          overwrite=True)

        out_scene_paths = [str(p) for p in usd.get_mesh_scene_paths(out_path)]
        out_dict = usd.import_meshes(out_path, out_scene_paths, with_normals=True, with_materials=True, return_list=False)
        exported_usd_mesh = SurfaceMesh.cat(
            [out_dict[p] for p in out_scene_paths], fixed_topology=False).to(device)

        # transform is not preserved through export/reimport (transforms are not exported)
        ignore_attrs = {'transform'}
        assert set(read_usd_mesh.get_attributes()) - ignore_attrs == set(exported_usd_mesh.get_attributes()) - ignore_attrs
        for att in read_usd_mesh.get_attributes(only_tensors=True):
            if att in ignore_attrs:
                continue
            assert contained_torch_equal(read_usd_mesh.get_attribute(att),
                                         exported_usd_mesh.get_attribute(att),
                                         print_error_context=f'Failed for attribute {att}',
                                         approximate=True, rtol=1e-5, atol=1e-8)

        for mesh_idx in range(len(read_usd_mesh)):
            materials_orig = read_usd_mesh.materials[mesh_idx]
            materials_exported = exported_usd_mesh.materials[mesh_idx]
            assert len(materials_orig) == len(materials_exported), f'Material number mismatch for mesh {mesh_idx}'
            _unset_material_names(materials_orig)
            _unset_material_names(materials_exported)
            assert contained_torch_equal(
                materials_orig, materials_exported,
                approximate=True, rtol=1e-2, atol=1e-3, print_error_context=f'Material mismatch for mesh {mesh_idx}')

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_read_write_read_consistency_multi_write(self, device, out_dir, expected_mesh_counts):
        # Same as above, but tests that multiple exports don't overwrite each other's textures (which would be the
        # case, unless setting texture paths carefully)
        all_bnames = ['ico_flat', 'ico_smooth', 'fox', 'pizza', 'amsterdam', 'armchair']

        # First we export all at once
        for bname in all_bnames:
            fname = glob.glob(io_data_path(f'{bname}.usd') + '*')[0]
            out_path = os.path.join(out_dir, f'reexport_multi_{bname}.usda')
            # import as multiple meshes
            in_scene_paths = [str(p) for p in usd.get_mesh_scene_paths(fname)]
            in_dict = usd.import_meshes(fname, in_scene_paths, with_normals=True, with_materials=True, return_list=False)
            read_usd_mesh = SurfaceMesh.cat(
                [in_dict[p] for p in in_scene_paths], fixed_topology=False).to(device)

            # Now write the USD to file, read it back and make sure attributes match the original mesh
            usd.export_meshes(out_path,
                              vertices=read_usd_mesh.vertices,
                              faces=read_usd_mesh.faces,
                              uvs=read_usd_mesh.uvs,
                              face_uvs_idx=read_usd_mesh.face_uvs_idx,
                              face_normals=read_usd_mesh.face_normals,
                              material_assignments=read_usd_mesh.material_assignments,
                              materials=read_usd_mesh.materials,
                              overwrite=True)

        # Next, we import original and exported and check consistency
        for bname in all_bnames:
            fname = glob.glob(io_data_path(f'{bname}.usd') + '*')[0]
            out_path = os.path.join(out_dir, f'reexport_multi_{bname}.usda')

            # import as multiple meshes
            in_scene_paths = [str(p) for p in usd.get_mesh_scene_paths(fname)]
            in_dict = usd.import_meshes(fname, in_scene_paths, with_normals=True, with_materials=True, return_list=False)
            read_usd_mesh = SurfaceMesh.cat(
                [in_dict[p] for p in in_scene_paths], fixed_topology=False)

            out_scene_paths = [str(p) for p in usd.get_mesh_scene_paths(out_path)]
            out_dict = usd.import_meshes(out_path, out_scene_paths, with_normals=True, with_materials=True, return_list=False)
            exported_usd_mesh = SurfaceMesh.cat(
                [out_dict[p] for p in out_scene_paths], fixed_topology=False)

            # transform is not preserved through export/reimport (transforms are not exported)
            ignore_attrs = {'transform'}
            assert set(read_usd_mesh.get_attributes()) - ignore_attrs == set(exported_usd_mesh.get_attributes()) - ignore_attrs
            for att in read_usd_mesh.get_attributes(only_tensors=True):
                if att in ignore_attrs:
                    continue
                assert contained_torch_equal(read_usd_mesh.get_attribute(att),
                                             exported_usd_mesh.get_attribute(att),
                                             approximate=True, rtol=1e-5, atol=1e-8), f'Failed for attribute {att}'

            for mesh_idx in range(len(read_usd_mesh)):
                materials_orig = read_usd_mesh.materials[mesh_idx]
                materials_exported = exported_usd_mesh.materials[mesh_idx]
                assert len(materials_orig) == len(materials_exported), f'Material number mismatch {bname} {mesh_idx}'
                _unset_material_names(materials_orig)
                _unset_material_names(materials_exported)
                assert contained_torch_equal(
                    materials_orig, materials_exported,
                    approximate=True, rtol=1e-2, atol=1e-3, print_error_context=f'Material mismatch {bname} {mesh_idx}')


class TestGoldenFiles:
    def test_rocket_homogenized_materials_golden_equivalence(self):
        """Old and new golden files for rocket_homogenized_materials differ only in signed-zero
        text representation (-0 vs 0); verify they import identically."""
        old_path = samples_data_path('golden', 'rocket_homogenized_materials_old.usda')
        new_path = samples_data_path('golden', 'rocket_homogenized_materials.usda')

        def _load(path):
            m = usd.import_mesh(path, with_normals=True, with_materials=True)
            m.unset_attributes_return_none = True
            return m

        old = _load(old_path)
        new = _load(new_path)

        assert contained_torch_equal(old.as_dict(), new.as_dict())


class TestV018Deprecation:
    """v0.18 call forms for mesh I/O keep working."""

    def test_import_meshes_default_returns_list(self, out_dir, mesh, mesh_alt):
        # v0.18 default: returns a list of SurfaceMesh.
        out_path = os.path.join(out_dir, 'legacy_default_list.usda')
        scene_paths = ['/World/mesh_0', '/World/mesh_1']
        usd.export_meshes(out_path,
                          scene_paths=scene_paths,
                          vertices=[mesh.vertices, mesh_alt.vertices],
                          faces=[mesh.faces, mesh_alt.faces],
                          overwrite=True)
        result = usd.import_meshes(out_path)
        assert isinstance(result, list)
        assert len(result) == 2
        for got, src in zip(result, [mesh, mesh_alt]):
            assert contained_torch_equal(
                {'vertices': got.vertices, 'faces': got.faces},
                {'vertices': src.vertices, 'faces': src.faces},
                approximate=True, atol=1e-5)
