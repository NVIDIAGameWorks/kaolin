# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import glob
import torch
import pytest

from kaolin.io import usd, obj, gltf, import_mesh
from kaolin.utils.testing import check_tensor_attribute_shapes, contained_torch_equal, check_allclose

__test_dir = os.path.dirname(os.path.realpath(__file__))
__samples_path = os.path.join(__test_dir, os.pardir, os.pardir, os.pardir, 'samples')


def io_data_path(fname):
    """ Return path relative to tests/samples/io"""
    return os.path.join(__samples_path, 'io', fname)


class TestDiverseInputs:
    @pytest.fixture(scope='class')
    def expected_sizes(self):
        return {'ico_smooth': {'vertices': [42, 3], 'faces': [80, 3]},
                'ico_flat': {'vertices': [42, 3], 'faces': [80, 3]},
                'fox': {'vertices': [5002, 3], 'faces':  [10000, 3]},
                'pizza': {'vertices': [482, 3], 'faces': [960, 3]},
                'armchair': {'vertices': [9204, 3], 'faces': [9200, 4]},
                'amsterdam': {'vertices': [1974, 3], 'faces': [1932, 4]},
                'avocado': {'vertices': [406, 3], 'faces': [682, 3]}}

    @pytest.fixture(scope='class')
    def expected_material_counts(self):
        return {'ico_smooth': 1,
                'ico_flat': 1,
                'fox': 1,
                'pizza': 2,
                'armchair': 2,
                'amsterdam': 14,
                'avocado': 1}

    @pytest.fixture(scope='class')
    def expected_mesh_counts(self):
        return {'ico_smooth': 1,
                'ico_flat': 1,
                'fox': 1,
                'pizza': 1,
                'armchair': 3,
                'amsterdam': 18,
                'avocado': 1}

    @pytest.mark.parametrize("triangulate", [True, False])
    @pytest.mark.parametrize("method_to_test", ["generic", "named"])
    @pytest.mark.parametrize('bname', ['ico_flat', 'ico_smooth', 'fox', 'pizza', 'amsterdam', 'armchair', 'avocado'])
    def test_read_usd_obj_consistency(self, bname, expected_sizes, expected_material_counts, method_to_test, triangulate):
        # Read USD version, flattening all meshes into one
        fname = glob.glob(io_data_path(f'{bname}.usd') + '*')[0]
        if method_to_test == 'generic':
            read_usd_mesh = import_mesh(fname, triangulate=triangulate)
        else:
            read_usd_mesh = usd.import_mesh(fname, with_normals=True, with_materials=True, triangulate=triangulate)
        if not triangulate:
            assert check_tensor_attribute_shapes(read_usd_mesh, **expected_sizes[bname])

        # Read OBJ version
        fname = io_data_path(f'{bname}.obj')
        if method_to_test == 'generic':
            read_obj_mesh = import_mesh(fname, triangulate=triangulate)
        else:
            read_obj_mesh = obj.import_mesh(fname, with_normals=True, with_materials=True, triangulate=triangulate)

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
        assert torch.equal(read_usd_mesh.faces, read_obj_mesh.faces)

        # Check that final face values between the two meshes agree (note the OBJ and USD may store
        # and index uvs and faces differently, but final per-face per-vertex values must agree
        assert torch.allclose(read_usd_mesh.face_uvs, read_obj_mesh.face_uvs, atol=1e-04)
        assert torch.allclose(read_usd_mesh.face_normals, read_obj_mesh.face_normals, atol=1e-04, rtol=1e-03)

        # Check material consistency
        assert len(read_usd_mesh.materials) == expected_material_counts[bname]
        assert len(read_usd_mesh.materials) == len(read_obj_mesh.materials)
        assert len(read_usd_mesh.material_assignments) > 0
        assert torch.equal(read_usd_mesh.material_assignments, read_obj_mesh.material_assignments)

    @pytest.mark.parametrize("method_to_test", ["generic", "named"])
    def test_read_usd_obj_gltf_consistency(self, expected_sizes, expected_material_counts, method_to_test):
        bname = 'avocado'
        # Read USD version
        fname = glob.glob(io_data_path(f'{bname}.usd') + '*')[0]
        if method_to_test == 'generic':
            read_usd_mesh = import_mesh(fname, triangulate=True)
        else:
            print("fname")
            read_usd_mesh = usd.import_mesh(fname, with_normals=True, with_materials=True, triangulate=True)

        # Read gltf version
        fname = io_data_path(f'{bname}.gltf')
        if method_to_test == 'generic':
            read_gltf_mesh = import_mesh(fname)
        else:
            read_gltf_mesh = gltf.import_mesh(fname)

        # Read obj version
        fname = io_data_path(f'{bname}.obj')
        if method_to_test == 'generic':
            read_obj_mesh = import_mesh(fname)
        else:
            read_obj_mesh = obj.import_mesh(fname, with_normals=True, with_materials=True, raw_materials=False)

        # Ensure vertex order is consistent before performing any further checks
        check_allclose(read_gltf_mesh.vertices, read_usd_mesh.vertices, atol=1e-04)
        assert torch.equal(read_usd_mesh.faces, read_gltf_mesh.faces)

        # Check that final face values between the two meshes agree (note the OBJ and USD may store
        # and index uvs and faces differently, but final per-face per-vertex values must agree
        assert torch.allclose(read_usd_mesh.face_uvs, read_gltf_mesh.face_uvs, atol=1e-04)
        assert torch.allclose(read_usd_mesh.face_normals, read_gltf_mesh.face_normals, atol=1e-03, rtol=1e-03)

        # Check material consistency
        assert len(read_usd_mesh.materials) == expected_material_counts[bname]
        assert len(read_usd_mesh.materials) == len(read_gltf_mesh.materials)
        assert len(read_usd_mesh.material_assignments) > 0
        assert torch.equal(read_usd_mesh.material_assignments, read_gltf_mesh.material_assignments)

        for usd_mat, gltf_mat, obj_mat in zip(read_usd_mesh.materials, read_gltf_mesh.materials, read_obj_mesh.materials):
            usd_mat.material_name = ''
            gltf_mat.material_name = ''
            obj_mat.material_name = ''
            assert contained_torch_equal(usd_mat, gltf_mat, approximate=True, print_error_context='', rtol=1e-2, atol=1e-2)
            assert contained_torch_equal(usd_mat, obj_mat, approximate=True, print_error_context='', rtol=1e-2, atol=1e-2)
