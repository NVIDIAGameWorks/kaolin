# Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES.
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
from kaolin.utils.testing import print_namedtuple_attributes, print_dict_attributes, \
    check_tensor_attribute_shapes, contained_torch_equal, check_allclose

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
        assert open(mesh_path).read() == open(out_path).read()

    @pytest.mark.parametrize('input_stage', [False, True, 'generated'])
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
        stage = usd.export_meshes(out_path, scene_paths, vertices_list, faces_list)

        # Now export all the attributes
        out_path = os.path.join(out_dir, 'meshes.usda')
        # TODO: properly export with materials once can convert OBJ materials to USD
        stage = usd.export_meshes(out_path, scene_paths, vertices_list, faces_list,
                                  uvs=[mesh.uvs, mesh_alt.uvs],
                                  face_uvs_idx=[mesh.face_uvs_idx, mesh_alt.face_uvs_idx],
                                  face_normals=actual_face_normals_list)

        # Test that we can read both meshes correctly with/without paths
        args = {}
        if with_paths:
            args = {'scene_paths': scene_paths}

        if input_stage == 'generated':
            path_or_stage = stage  # Also test stage output by export_meshes
        else:
            path_or_stage = Usd.Stage.Open(out_path) if input_stage else out_path

        # TODO: once above is fixed use with_materials=True here
        meshes_in = usd.import_meshes(path_or_stage, with_normals=True, **args)
        assert len(meshes_in) == len(meshes)
        for i, orig_mesh in enumerate(meshes):
            in_mesh = meshes_in[i]
            # DEBUG INFORMATION (uncomment when debugging)
            # print_namedtuple_attributes(orig_mesh, f'Orig mesh [{i}]')
            # print_namedtuple_attributes(in_mesh, f'Imported mesh [{i}]')

            # We check key attributes
            assert contained_torch_equal(
                {'vertices': orig_mesh.vertices, 'faces': orig_mesh.faces, 'uvs': orig_mesh.uvs,
                 'face_uvs_idx': orig_mesh.face_uvs_idx, 'face_normals': actual_face_normals_list[i]},
                {'vertices': in_mesh.vertices, 'faces': in_mesh.faces, 'uvs': in_mesh.uvs,
                 'face_uvs_idx': in_mesh.face_uvs_idx, 'face_normals': in_mesh.face_normals},
                approximate=True, rtol=1e-5, atol=1e-8)

        # Test that can also read the flattened mesh and check that attributes are correctly shifted
        mesh_in = usd.import_mesh(path_or_stage, with_normals=True)
        assert len(mesh_in.vertices) == (len(mesh.vertices) + len(mesh_alt.vertices))
        assert len(mesh_in.faces) == (len(mesh.faces) + len(mesh_alt.faces))
        assert contained_torch_equal(
            {'vertices': torch.cat([mesh.vertices, mesh_alt.vertices], dim=0),
             'faces': torch.cat([mesh.faces, mesh_alt.faces + mesh.vertices.shape[0]], dim=0),
             'uvs': torch.cat([mesh.uvs, mesh_alt.uvs], dim=0),
             'face_uvs_idx': torch.cat([mesh.face_uvs_idx, mesh_alt.face_uvs_idx + mesh.uvs.shape[0]], dim=0),
             'face_normals': torch.cat(actual_face_normals_list, dim=0)
             },
            {'vertices': mesh_in.vertices,
             'faces': mesh_in.faces,
             'uvs': mesh_in.uvs,
             'face_uvs_idx': mesh_in.face_uvs_idx,
             'face_normals': mesh_in.face_normals
             },
            approximate=True, rtol=1e-5, atol=1e-8)

    def test_import_bad_prim(self, scene_paths, mesh_path):
        """Test that import fails when reaching invalid prims"""
        with pytest.raises(ValueError):
            usd.import_meshes(mesh_path, ['/foo'] + scene_paths)

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_hetero_fail(self, hetero_mesh_path, input_stage):
        """Test that import fails when importing heterogeneous mesh without handler"""
        path_or_stage = Usd.Stage.Open(hetero_mesh_path) if input_stage else hetero_mesh_path

        with pytest.raises(utils.NonHomogeneousMeshError):
            usd.import_meshes(file_path_or_stage=path_or_stage, scene_paths=['/Root'])

        with pytest.raises(utils.NonHomogeneousMeshError):
            usd.import_mesh(file_path_or_stage=path_or_stage, scene_path='/Root')

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_hetero_skip(self, scene_paths, hetero_mesh_path, homo_mesh_path, mixed_mesh_path, input_stage):
        """Test that import skips mesh when importing heterogeneous mesh with skip handler"""
        path_or_stage = Usd.Stage.Open(hetero_mesh_path) if input_stage else hetero_mesh_path
        meshes = usd.import_meshes(path_or_stage, ['/Root'],
                                   heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_skip)
        assert len(meshes) == 0

        path_or_stage = Usd.Stage.Open(homo_mesh_path) if input_stage else homo_mesh_path
        meshes = usd.import_meshes(path_or_stage,
                                   heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_skip)
        assert len(meshes) == 2

        # Test skip on a batch of mixed homogeneous and heterogeneous meshes
        # Note we can't export heterogeneous meshes, so previous test did not work
        path_or_stage = Usd.Stage.Open(mixed_mesh_path) if input_stage else mixed_mesh_path
        meshes = usd.import_meshes(path_or_stage,
                                   heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_skip)
        assert len(meshes) == 1

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_hetero_homogenize_import_meshes(self, out_dir, hetero_mesh_path, homogenized_golden_path,
                                                    input_stage):
        """Test that imports homogeneous mesh when importing heterogeneous mesh with naive homogenize handler"""
        # TODO(jlafleche) Render meshes before/after homogenize operation
        path_or_stage = Usd.Stage.Open(hetero_mesh_path) if input_stage else hetero_mesh_path
        mesh = usd.import_meshes(path_or_stage, ['/Root'],
                                 heterogeneous_mesh_handler=utils.mesh_handler_naive_triangulate)
        # Confirm we now have a triangle mesh
        assert mesh[0].faces.size(1) == 3

        out_path = os.path.join(out_dir, 'homogenized.usda')
        usd.export_mesh(out_path, '/World/Rocket', vertices=mesh[0].vertices, faces=mesh[0].faces)

        # Confirm exported USD matches golden file
        assert open(homogenized_golden_path).read() == open(out_path).read()

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_hetero_homogenize_import_mesh(self, out_dir,  hetero_mesh_path, homogenized_golden_path, input_stage):
        """Test that imports homogeneous mesh when importing heterogeneous mesh with naive homogenize handler"""
        # TODO(jlafleche) Render meshes before/after homogenize operation
        path_or_stage = Usd.Stage.Open(hetero_mesh_path) if input_stage else hetero_mesh_path
        mesh = usd.import_mesh(path_or_stage, scene_path='/Root',
                               heterogeneous_mesh_handler=utils.mesh_handler_naive_triangulate)
        # Confirm we now have a triangle mesh
        assert mesh.faces.size(1) == 3

        out_path = os.path.join(out_dir, 'homogenized.usda')
        usd.export_mesh(out_path, '/World/Rocket', vertices=mesh.vertices, faces=mesh.faces)

        # Confirm exported USD matches golden file
        assert open(homogenized_golden_path).read() == open(out_path).read()

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_with_transform(self, scene_paths, out_dir, hetero_mesh_path, input_stage):
        """Test that mesh transforms are correctly applied during import"""
        path_or_stage = Usd.Stage.Open(hetero_mesh_path) if input_stage else hetero_mesh_path
        mesh = usd.import_mesh(path_or_stage, '/Root',
                               heterogeneous_mesh_handler=utils.mesh_handler_naive_triangulate)
        out_path = os.path.join(out_dir, 'transformed.usda')
        stage = usd.create_stage(out_path)
        prim = usd.add_mesh(stage, '/World/Rocket', vertices=mesh.vertices, faces=mesh.faces)
        UsdGeom.Xformable(prim).AddTranslateOp().Set((10, 10, 10))
        stage.Save()

        mesh_import = usd.import_mesh(out_path)
        assert torch.allclose(mesh_import.vertices, mesh.vertices + 10.)

    @pytest.mark.parametrize('function_variant', ['export_mesh', 'export_meshes'])
    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_material_subsets(self, scene_paths, out_dir, hetero_subsets_materials_mesh_path,
                                     input_stage, function_variant):
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
            assert torch.allclose(mesh.vertices[mesh.faces[tri_idx, :], :], expected_vertices)
            assert torch.allclose(mesh.face_normals[tri_idx, ...], expected_normals)
            assert torch.allclose(mesh.uvs[mesh.face_uvs_idx[tri_idx, :], :], expected_uvs)

            # Also sanity check the golden mesh (to catch human error)
            assert torch.allclose(golden_mesh.vertices[mesh.faces[tri_idx, :], :], expected_vertices)
            assert torch.allclose(golden_mesh.face_normals[tri_idx, ...], expected_normals)
            assert torch.allclose(golden_mesh.uvs[mesh.face_uvs_idx[tri_idx, :], :], expected_uvs)
        # Write the homogenized mesh to file
        out_path = os.path.join(out_dir, 'rocket_homogenized_materials.usda')
        if function_variant == 'export_mesh':
            usd.export_mesh(out_path, '/World/Rocket', vertices=mesh.vertices, faces=mesh.faces,
                            face_uvs_idx=mesh.face_uvs_idx, face_normals=mesh.face_normals, uvs=mesh.uvs,
                            material_assignments=mesh.material_assignments, materials=mesh.materials)
        else:
            usd.export_meshes(out_path, ['/World/Rocket'], vertices=[mesh.vertices], faces=[mesh.faces],
                              face_uvs_idx=[mesh.face_uvs_idx], face_normals=[mesh.face_normals], uvs=[mesh.uvs],
                              material_assignments=[mesh.material_assignments],
                              materials=[mesh.materials])

        # Confirm exported USD matches golden file
        assert open(golden_path).read() == open(out_path).read()

        # Confirm we read identical mesh after writing
        reimported_mesh = usd.import_mesh(out_path, scene_path='/World/Rocket', with_materials=True, with_normals=True)
        reimported_mesh.unset_attributes_return_none = True

        # Since comparison of materials is not implemented, we override materials with diffuse colors first
        assert len(mesh.materials) == len(reimported_mesh.materials)
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

    @pytest.mark.parametrize('input_stage', [False, True])
    def test_import_st_indices_facevarying(self, out_dir, mesh, input_stage):
        out_path = os.path.join(out_dir, 'st_indices.usda')
        uvs = torch.rand((mesh.faces.view(-1).size(0), 2))
        scene_path = '/World/mesh_0'
        face_uvs_idx = (torch.rand(mesh.faces.shape[:2]) * 99).long()
        usd.export_mesh(out_path, scene_path=scene_path, vertices=mesh.vertices,
                        faces=mesh.faces, uvs=uvs, face_uvs_idx=face_uvs_idx)

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
                        faces=mesh.faces, uvs=uvs)

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
                        faces=mesh.faces, uvs=uvs)

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
        input_path = io_data_path(f'amsterdam.usd')  # Multiple quad meshes
        if flatten:
            # Import as one mesh
            orig = [usd.import_mesh(input_path, with_materials=with_materials, with_normals=with_normals)]
            triangulated = [usd.import_mesh(input_path, with_materials=with_materials, with_normals=with_normals,
                                            triangulate=True)]
            assert len(orig) == 1
            assert len(triangulated) == 1
            expected_num_vertices = [1974]
            expected_num_quads = [1932]
        else:
            # Import as multiple meshes
            orig = usd.import_meshes(input_path, with_materials=with_materials, with_normals=with_normals)
            triangulated = usd.import_meshes(input_path, with_materials=with_materials, with_normals=with_normals,
                                             triangulate=True)
            assert len(orig) == 18
            assert len(triangulated) == 18
            expected_num_vertices = [4, 98, 98, 98, 386, 386, 98, 8, 98, 98, 98, 4, 4, 4, 386, 98, 4, 4]
            expected_num_quads = [1, 96, 96, 96, 384, 384, 96, 6, 96, 96, 96, 1, 1, 1, 384, 96, 1, 1]

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
    def expected_sizes(self):
        return {'ico_smooth': {'vertices': [42, 3], 'faces': [80, 3]},
                'ico_flat': {'vertices': [42, 3], 'faces': [80, 3]},
                'fox': {'vertices': [5002, 3], 'faces':  [10000, 3]},
                'pizza': {'vertices': [482, 3], 'faces': [960, 3]},
                'armchair': {'vertices': [9204, 3], 'faces': [9200, 4]},
                'amsterdam': {'vertices': [1974, 3], 'faces': [1932, 4]}}

    @pytest.fixture(scope='class')
    def expected_material_counts(self):
        return {'ico_smooth': 1,
                'ico_flat': 1,
                'fox': 1,
                'pizza': 2,
                'armchair': 2,
                'amsterdam': 14}

    # TODO: add armchair
    @pytest.mark.parametrize('bname', ['ico_flat', 'ico_smooth', 'fox', 'pizza', 'amsterdam'])
    def test_read_write_read_consistency(self, bname, out_dir, expected_sizes, expected_material_counts):
        # Read USD version, flattening all meshes into one
        fname = io_data_path(f'{bname}.usd')
        read_usd_mesh = usd.import_mesh(fname, with_normals=True, with_materials=True)
        assert check_tensor_attribute_shapes(read_usd_mesh, **expected_sizes[bname])

        # Read OBJ version
        fname = io_data_path(f'{bname}.obj')
        read_obj_mesh = obj.import_mesh(fname, with_normals=True, with_materials=True)

        # DEBUG INFORMATION (uncomment to help diagnose failures)
        # stage = Usd.Stage.Open(io_data_path(f'{bname}.usd'))
        # paths = usd.utils.get_scene_paths(stage, prim_types=["Mesh"])
        # #assert len(paths) == 1
        # prim = stage.GetPrimAtPath(paths[0])
        # raw_usd = usd.get_raw_mesh_prim_geometry(prim, with_normals=True, with_uvs=True, time=0)
        # print_namedtuple_attributes(read_usd_mesh, f'Read USD mesh {bname}')
        # print_dict_attributes(raw_usd, name=f'RAW USD {bname}')
        # print_namedtuple_attributes(read_obj_mesh, f'Read OBJ mesh {bname}')

        # Ensure vertex order is consistent before performing any further checks
        check_allclose(read_obj_mesh.vertices, read_usd_mesh.vertices, atol=1e-04)

        # Check that final face values between the two meshes agree (note the OBJ and USD may store
        # and index uvs and faces differently, but final per-face per-vertex values must agree
        assert torch.allclose(read_usd_mesh.face_uvs, read_obj_mesh.face_uvs, atol=1e-04)
        assert torch.allclose(read_usd_mesh.face_normals, read_obj_mesh.face_normals, atol=1e-04)

        # Check material consistency
        assert len(read_usd_mesh.materials) == expected_material_counts[bname]
        assert len(read_usd_mesh.materials) == len(read_obj_mesh.materials)
        assert len(read_usd_mesh.material_assignments) > 0
        assert torch.equal(read_usd_mesh.material_assignments, read_obj_mesh.material_assignments)

        # Now write the USD to file, read it back and make sure attributes are as expected
        out_path = os.path.join(out_dir, f'reexport_{bname}.usda')
        # TODO: the export fails with materials; add a test and fix this in test_materials.py and here
        # Note: specular value is expected to be  a tuple, not single value as in this case
        usd.export_mesh(out_path, vertices=read_usd_mesh.vertices, faces=read_usd_mesh.faces,
                        uvs=read_usd_mesh.uvs, face_uvs_idx=read_usd_mesh.face_uvs_idx,
                        face_normals=read_usd_mesh.face_normals)

        # Because we don't want to compare materials, read original mesh and exported mesh
        fname = io_data_path(f'{bname}.usd')
        read_usd_mesh = usd.import_mesh(fname, with_normals=True)
        # Read exported mesh
        exported_usd_mesh = usd.import_mesh(out_path, with_normals=True)
        assert contained_torch_equal(read_usd_mesh, exported_usd_mesh, approximate=True, rtol=1e-5, atol=1e-8)
