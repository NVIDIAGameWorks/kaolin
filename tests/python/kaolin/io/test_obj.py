# Copyright (c) 2019,20-22, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pytest

import torch

from kaolin.io import utils
from kaolin.io import obj
from kaolin.utils.testing import print_namedtuple_attributes, print_dict_attributes, \
    check_tensor_attribute_shapes, contained_torch_equal, check_allclose

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, 'samples')
SIMPLE_DIR = os.path.join(ROOT_DIR, 'simple_obj/')


def io_data_path(fname):
    """ Return path relative to tests/samples/io"""
    return os.path.join(ROOT_DIR, 'io', fname)


# TODO(cfujitsang): Add sanity test over a dataset like ShapeNet

def _get_mtl_names(mtl_path):
    names = []
    with open(mtl_path, 'r') as f:
        for line in f.readlines():
            data = line.split()
            if len(data) == 0:
                continue
            if data[0] == 'newmtl':
                names.append(data[1])
    names.sort()
    return names


class TestLoadObj:
    @pytest.fixture(autouse=True)
    def expected_vertices(self):
        return torch.FloatTensor([
            [-0.1, -0.1, -0.1],
            [0.1, -0.1, -0.1],
            [-0.1, 0.1, -0.1],
            [0.1, 0.1, -0.1],
            [-0.1, -0.1, 0.1],
            [0.1, -0.1, 0.1]
        ])

    @pytest.fixture(autouse=True)
    def expected_faces(self):
        return torch.LongTensor([
            [0, 1, 3, 2],
            [1, 0, 4, 5]
        ])

    @pytest.fixture(autouse=True)
    def expected_faces_triangulated(self):
        return torch.LongTensor([
            [0, 1, 3], [0, 3, 2],
            [1, 0, 4], [1, 4, 5]
        ])

    @pytest.fixture(autouse=True)
    def expected_faces_heterogeneous(self):
        return torch.LongTensor([
            [0, 1, 3],
            [0, 3, 2],
            [1, 0, 4]
        ])

    @pytest.fixture(autouse=True)
    def expected_uvs(self):
        return torch.Tensor([
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0]
        ])

    @pytest.fixture(autouse=True)
    def expected_face_uvs_idx(self):
        return torch.LongTensor([
            [0, 1, 3, 2],
            [3, 1, 0, 2]
        ])

    @pytest.fixture(autouse=True)
    def expected_face_uvs_idx_triangulated(self):
        return torch.LongTensor([
            [0, 1, 3], [0, 3, 2],
            [3, 1, 0], [3, 0, 2]
        ])

    @pytest.fixture(autouse=True)
    def expected_face_uvs_idx_heterogeneous(self):
        return torch.LongTensor([
            [0, 1, 3],
            [0, 3, 2],
            [3, 1, 0]
        ])

    @pytest.fixture(autouse=True)
    def expected_normals(self):
        return torch.FloatTensor([
            [0., 0., -1.],
            [0., -1., 0.],
            [-0.333334, -0.333333, -0.333333],
            [0.333334, -0.333333, -0.333333]
        ])

    @pytest.fixture(autouse=True)
    def expected_face_normals_idx(self):
        return torch.LongTensor([
            [2, 3, 0, 0],
            [3, 2, 1, 1]
        ])

    @pytest.fixture(autouse=True)
    def expected_face_normals_idx_triangulated(self):
        return torch.LongTensor([
            [2, 3, 0], [2, 0, 0],
            [3, 2, 1], [3, 1, 1]
        ])

    @pytest.fixture(autouse=True)
    def expected_face_normals_idx_heterogeneous(self):
        return torch.LongTensor([
            [2, 3, 0],
            [2, 0, 0],
            [3, 2, 1]
        ])

    @pytest.fixture(autouse=True)
    def expected_materials(self):
        return [
            {'material_name': 'Material.001',
             'Ka': torch.tensor([0.5, 0.6, 0.7]),
             'Kd': torch.tensor([0.4, 0.3, 0.2]),
             'Ks': torch.tensor([0.1, 0.3, 0.5]),
             'map_Ka': torch.ByteTensor([[[102, 127, 178], [153, 178, 178]],
                                         [[127, 127, 153], [127, 178, 204]]]),
             'map_Kd': torch.ByteTensor([[[102, 102, 76],  [102, 51,  25]],
                                         [[76,  76,  51],  [127, 76,  51]]])
            },
            {'material_name': 'Material.002',
             'Ka': torch.tensor([0.7, 0.7, 0.7]),
             'Kd': torch.tensor([0.2, 0.2, 0.2]),
             'Ks': torch.tensor([0.1, 0.1, 0.1]),
             'map_Ka': torch.ByteTensor([[[178, 178, 153], [178, 178, 102]],
                                         [[178, 178, 204], [178, 178, 229]]]),
             'map_Kd': torch.ByteTensor([[[51,  51,  51],  [51,  51,  51]],
                                         [[51,  51,  51],  [51,  51,  51]]]),
             'map_Ks': torch.ByteTensor([[[0,   0,   25],  [0,   0,   25]],
                                         [[51,  51,  25],  [51,  51,  25]]])
            },
            {'material_name': 'Material.003',
             'Ka': torch.tensor([0., 0., 0.]),
             'Kd': torch.tensor([0., 0., 0.]),
             'Ks': torch.tensor([0., 0., 0.])
            }
        ]

    @pytest.fixture(autouse=True)
    def expected_material_assignments_heterogeneous(self):
        return torch.ShortTensor([0, 0, 1])

    @pytest.fixture(autouse=True)
    def expected_material_assignments(self):
        return torch.ShortTensor([0, 1])

    @pytest.fixture(autouse=True)
    def expected_material_assignments_triangulated(self):
        return torch.ShortTensor([0, 0, 1, 1])

    @pytest.mark.parametrize('with_normals', [False, True])
    @pytest.mark.parametrize('with_materials', [False, True])
    def test_import_mesh(self, with_normals, with_materials,
                         expected_vertices, expected_faces, expected_uvs, expected_face_uvs_idx, expected_normals,
                         expected_face_normals_idx, expected_materials, expected_material_assignments):
        outputs = obj.import_mesh(os.path.join(SIMPLE_DIR, 'model.obj'),
                                  with_materials=with_materials, with_normals=with_normals,
                                  error_handler=obj.skip_error_handler)
        assert torch.equal(outputs.vertices, expected_vertices)
        assert torch.equal(outputs.faces, expected_faces)
        if with_materials:
            assert torch.allclose(outputs.uvs, expected_uvs)
            assert torch.equal(outputs.face_uvs_idx, expected_face_uvs_idx)
            assert contained_torch_equal(outputs.materials, expected_materials, approximate=True)
            assert contained_torch_equal(outputs.material_assignments, expected_material_assignments)
        else:
            assert outputs.materials is None
            assert outputs.material_assignments is None
        if with_normals:
            assert torch.equal(outputs.normals, expected_normals)
            assert torch.equal(outputs.face_normals_idx, expected_face_normals_idx)
        else:
            assert outputs.normals is None
            assert outputs.face_normals_idx is None

    @pytest.mark.parametrize('with_normals', [False, True])
    @pytest.mark.parametrize('with_materials', [False, True])
    def test_import_mesh_triangulate(self, with_normals, with_materials,
                                     expected_vertices, expected_faces_triangulated,
                                     expected_uvs, expected_face_uvs_idx_triangulated,
                                     expected_normals, expected_face_normals_idx_triangulated, expected_materials,
                                     expected_material_assignments_triangulated):
        outputs = obj.import_mesh(os.path.join(SIMPLE_DIR, 'model.obj'),
                                  with_materials=with_materials, with_normals=with_normals,
                                  error_handler=obj.skip_error_handler,
                                  triangulate=True)
        # TODO: might want to write a function for this if/else testing block; it's repeated everywhere
        assert torch.equal(outputs.vertices, expected_vertices)
        assert torch.equal(outputs.faces, expected_faces_triangulated)
        if with_materials:
            assert torch.allclose(outputs.uvs, expected_uvs)
            assert torch.equal(outputs.face_uvs_idx, expected_face_uvs_idx_triangulated)
            assert contained_torch_equal(outputs.materials, expected_materials, approximate=True)
            assert contained_torch_equal(outputs.material_assignments, expected_material_assignments_triangulated)
        else:
            assert outputs.materials is None
            assert outputs.material_assignments is None

        if with_normals:
            assert torch.equal(outputs.normals, expected_normals)
            assert torch.equal(outputs.face_normals_idx, expected_face_normals_idx_triangulated)
        else:
            assert outputs.normals is None
            assert outputs.face_normals_idx is None

    @pytest.mark.parametrize('with_normals', [False, True])
    def test_error_import_mesh(self, with_normals):
        with pytest.raises(obj.MaterialLoadError):
            outputs = obj.import_mesh(os.path.join(SIMPLE_DIR, 'model.obj'),
                                      with_materials=True, with_normals=with_normals,
                                      error_handler=obj.default_error_handler)

    @pytest.mark.parametrize('with_normals', [False, True])
    def test_warn_import_mesh(self, with_normals):
        with pytest.warns(UserWarning):
            outputs = obj.import_mesh(os.path.join(SIMPLE_DIR, "model.obj"),
                                      with_materials=True, with_normals=with_normals,
                                      error_handler=obj.skip_error_handler)

    @pytest.mark.parametrize('with_normals', [False, True])
    @pytest.mark.parametrize('with_materials', [False, True])
    def test_import_mesh_heterogeneous(self, with_normals, with_materials, expected_vertices,
                                       expected_faces_heterogeneous, expected_face_uvs_idx_heterogeneous,
                                       expected_uvs, expected_materials, expected_material_assignments_heterogeneous,
                                       expected_normals, expected_face_normals_idx_heterogeneous):
        outputs = obj.import_mesh(os.path.join(SIMPLE_DIR, 'model_heterogeneous.obj'),
                                  with_materials=with_materials, with_normals=with_normals,
                                  error_handler=obj.skip_error_handler,
                                  heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_naive_homogenize)
        assert torch.equal(outputs.vertices, expected_vertices)
        assert torch.equal(outputs.faces, expected_faces_heterogeneous)

        if with_materials:
            assert torch.allclose(outputs.uvs, expected_uvs)
            assert torch.equal(outputs.face_uvs_idx, expected_face_uvs_idx_heterogeneous)
            assert contained_torch_equal(outputs.materials, expected_materials, approximate=True)
            assert contained_torch_equal(outputs.material_assignments, expected_material_assignments_heterogeneous)
        else:
            assert outputs.materials is None
            assert outputs.material_assignments is None

        if with_normals:
            assert torch.equal(outputs.normals, expected_normals)
            assert torch.equal(outputs.face_normals_idx, expected_face_normals_idx_heterogeneous)
        else:
            assert outputs.normals is None
            assert outputs.face_normals_idx is None

    def test_import_mesh_heterogeneous_skip(self):
        outputs = obj.import_mesh(os.path.join(SIMPLE_DIR, 'model_heterogeneous.obj'),
                                  with_materials=True, with_normals=True,
                                  error_handler=obj.skip_error_handler,
                                  heterogeneous_mesh_handler=utils.heterogeneous_mesh_handler_skip)
        assert outputs is None

    @pytest.fixture(autouse=True)
    def expected_large_values(self):
        num_vertices = 0
        num_faces = 0
        num_uvs = 0
        num_normals = 0
        num_face_groups = 0
        material_names = []

        # Process core attributes and materials
        with open(os.path.join(ROOT_DIR, "model.obj")) as f:
            for line in f.readlines():
                data = line.split()
                if len(data) == 0:
                    continue
                if data[0] == 'f':
                    num_faces += 1
                elif data[0] == 'v':
                    num_vertices += 1
                elif data[0] == 'vt':
                    num_uvs += 1
                elif data[0] == 'vn':
                    num_normals += 1
                elif data[0] == 'usemtl':
                    num_face_groups += 1
                elif data[0] == 'mtllib':
                    material_names.extend(_get_mtl_names(os.path.join(ROOT_DIR, data[1])))

        material_names.sort()
        num_materials = len(material_names)

        # Process material assignments in alphabetical order
        active_mtl = None
        face_idx = 0
        material_assignments = torch.zeros((num_faces,)).short() - 1
        with open(os.path.join(ROOT_DIR, "model.obj")) as f:
            for line in f.readlines():
                data = line.split()
                if len(data) == 0:
                    continue
                if data[0] == 'f':
                    if active_mtl is not None:
                        material_assignments[face_idx] = active_mtl
                    face_idx += 1
                elif data[0] == 'usemtl':
                    active_mtl = material_names.index(data[1])

        return {'num_vertices': num_vertices,
                'num_faces': num_faces,
                'num_uvs': num_uvs,
                'num_normals': num_normals,
                'num_materials': num_materials,
                'num_face_groups': num_face_groups,
                'material_assignments': material_assignments}

    @pytest.mark.parametrize('with_normals', [False, True])
    @pytest.mark.parametrize('with_materials', [False, True])
    def test_large_obj(self, with_materials, with_normals, expected_large_values):
        outputs = obj.import_mesh(os.path.join(ROOT_DIR, "model.obj"),
                               with_materials=with_materials, with_normals=with_normals)
        assert outputs.vertices.shape == (expected_large_values['num_vertices'], 3)
        assert outputs.faces.shape == (expected_large_values['num_faces'], 3)
        if with_materials:
            assert outputs.uvs.shape == (expected_large_values['num_uvs'], 2)
            assert outputs.face_uvs_idx.shape == (expected_large_values['num_faces'], 3)
            assert len(outputs.materials) == expected_large_values['num_materials']
            assert contained_torch_equal(outputs.material_assignments, expected_large_values['material_assignments'])
        else:
            assert outputs.uvs is None
            assert outputs.face_uvs_idx is None
            assert outputs.materials is None
            assert outputs.material_assignments is None
        if with_normals:
            assert outputs.normals.shape == (expected_large_values['num_normals'], 3)
            assert outputs.face_normals_idx.shape == (expected_large_values['num_faces'], 3)
        else:
            assert outputs.normals is None
            assert outputs.face_normals_idx is None




class TestDiverseInputs:
    @pytest.fixture(scope='class')
    def expected_sizes(self):
        # TODO: compare actual face UVs and normals once consistent between OBJ and USD
        return {'ico_smooth': {'vertices': [42, 3], 'faces': [80, 3], 'normals': [42, 3], 'uvs': [63, 2]},
                'ico_flat': {'vertices': [42, 3], 'faces': [80, 3], 'normals': [80, 3], 'uvs': [63, 2]},
                'fox': {'vertices': [5002, 3], 'faces':  [10000, 3], 'normals': [5002, 3], 'uvs': [5505, 2]}}

    @pytest.mark.parametrize('bname', ['ico_smooth', 'ico_flat', 'fox'])
    def test_read_write_read(self, bname, expected_sizes):
        # TODO: also test materials
        fname = io_data_path(f'{bname}.obj')
        read_mesh = obj.import_mesh(fname, with_normals=True, with_materials=True)

        # DEBUG INFORMATION (uncomment when debugging)
        # print_namedtuple_attributes(read_mesh, f'Read OBJ mesh {bname}')
        assert check_tensor_attribute_shapes(read_mesh, **expected_sizes[bname])
