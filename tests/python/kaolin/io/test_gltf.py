# Copyright (c) 2023,26 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import base64
import json
import pytest
from PIL import Image
import numpy as np

import torch

import kaolin
from kaolin.io import utils, gltf
from kaolin.utils.testing import print_namedtuple_attributes, print_dict_attributes, \
    check_tensor_attribute_shapes, contained_torch_equal, check_allclose

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, 'samples')
SAMPLE_DIR = os.path.join(ROOT_DIR, 'io')
GT_DIR = os.path.join(SAMPLE_DIR, 'gltf')

# TODO(cfujitsang): Add sanity test over a dataset like ShapeNet
class TestAvocadoGLTF:
    @pytest.fixture(autouse=True)
    def expected_vertices(self):
        return torch.load(os.path.join(GT_DIR, 'Avocado_gt_vertices.pt'))

    @pytest.fixture(autouse=True)
    def expected_faces(self):
        return torch.load(os.path.join(GT_DIR, 'Avocado_gt_faces.pt'))

    @pytest.fixture(autouse=True)
    def expected_uvs(self):
        return torch.load(os.path.join(GT_DIR, 'Avocado_gt_uvs.pt'))

    @pytest.fixture(autouse=True)
    def expected_face_uvs_idx(self, expected_faces):
        return expected_faces

    @pytest.fixture(autouse=True)
    def expected_diffuse_texture(self):
        img = Image.open(os.path.join(SAMPLE_DIR, 'textures', 'Avocado_baseColor.png'))
        return torch.as_tensor(np.array(img)).float() * (1. / 255.)

    @pytest.fixture(autouse=True)
    def expected_metallic_roughness_texture(self):
        img = Image.open(os.path.join(SAMPLE_DIR, 'textures', 'Avocado_roughnessMetallic.png'))
        return torch.as_tensor(np.array(img)).float() * (1. / 255.)

    @pytest.fixture(autouse=True)
    def expected_roughness_texture(self, expected_metallic_roughness_texture):
        return expected_metallic_roughness_texture[..., 1:2]

    @pytest.fixture(autouse=True)
    def expected_metallic_texture(self, expected_metallic_roughness_texture):
        return expected_metallic_roughness_texture[..., 2:3]

    @pytest.fixture(autouse=True)
    def expected_normals_texture(self):
        img = Image.open(os.path.join(SAMPLE_DIR, 'textures', 'Avocado_normal.png'))
        return (torch.as_tensor(np.array(img)).float() * (2. / 255.) - 1.)

    @pytest.fixture(autouse=True)
    def expected_material_assignments(self, expected_faces):
        return torch.zeros((expected_faces.shape[0],), dtype=torch.short)

    def test_import_mesh(
        self, expected_vertices, expected_faces,
        expected_uvs, expected_face_uvs_idx,
        expected_diffuse_texture, expected_roughness_texture,
        expected_metallic_texture, expected_normals_texture,
        expected_material_assignments
    ):
        mesh = gltf.import_mesh(os.path.join(
            SAMPLE_DIR,
            'avocado.gltf'
        ))
        assert mesh.has_attribute('vertex_normals')
        assert torch.equal(mesh.vertices, expected_vertices)
        assert torch.equal(mesh.faces, expected_faces)
        assert torch.equal(mesh.uvs, expected_uvs)
        assert torch.equal(mesh.face_uvs_idx, expected_face_uvs_idx)
        assert torch.equal(mesh.materials[0].diffuse_texture,
                           expected_diffuse_texture)
        assert torch.equal(mesh.materials[0].roughness_texture,
                           expected_roughness_texture)
        assert torch.equal(mesh.materials[0].metallic_texture,
                           expected_metallic_texture)
        assert torch.equal(mesh.materials[0].normals_texture,
                           expected_normals_texture)
        assert torch.equal(mesh.material_assignments,
                           expected_material_assignments)

        # Make sure transmittance extension isn't loaded
        assert mesh.materials[0].transmittance_value is None
        assert mesh.materials[0].transmittance_texture is None


class TestDiverseInputs:
    @pytest.fixture(scope='class')
    def expected_sizes_flat(self):
        return {'armchair': {'vertices': [10800, 3], 'faces': [18400, 3]},
                'avocado': {'vertices': [406, 3], 'faces': [682, 3]}}

    @pytest.fixture(scope='class')
    def expected_sizes(self):
        # Obtained from reading the gltf text file
        return {'armchair': [{'vertices': [4454, 3], 'faces': [8288, 3]},
                             {'vertices': [1896, 3], 'faces': [1824, 3]},
                             {'vertices': [4450, 3], 'faces': [8288, 3]}],
                'avocado': [{'vertices': [406, 3], 'faces': [682, 3]}]}

    @pytest.mark.parametrize("bname", ["avocado", "armchair"])
    def test_import_basics(self, bname, expected_sizes_flat):
        mesh = gltf.import_mesh(os.path.join(SAMPLE_DIR, bname + '.gltf'))

        expected_attributes = {'faces', 'vertices', 'vertex_normals', 'vertex_tangents',
                               'uvs', 'face_uvs_idx', 'material_assignments'}
        assert set(mesh.get_attributes(True)) == expected_attributes
        assert check_tensor_attribute_shapes(mesh, **expected_sizes_flat[bname])

    @pytest.mark.parametrize("bname", ["avocado", "armchair"])
    def test_import_meshes(self, bname, expected_sizes):
        meshes = gltf.import_meshes(os.path.join(SAMPLE_DIR, bname + '.gltf'))
        assert len(meshes) == len(expected_sizes[bname])

        expected_attributes = {'faces', 'vertices', 'vertex_normals', 'vertex_tangents',
                               'uvs', 'face_uvs_idx', 'material_assignments'}
        for idx, m in enumerate(meshes):
            assert set(m.get_attributes(True)) == expected_attributes
            assert check_tensor_attribute_shapes(m, **expected_sizes[bname][idx])


class TestTransmittanceExtensionGLTF:
    """ Note: The asset used in this test has been modified from its original transmittance values.
    Do not use original rendered values for comparison.
    """

    @pytest.fixture(autouse=True)
    def expected_transmittance_texture(self):
        img = Image.open(os.path.join(SAMPLE_DIR, 'textures', 'transmit_test_texture14184.jpg'))
        img = torch.as_tensor(np.array(img)).float() * (1. / 255.)
        return img[:,:,0:1]

    def test_import_mesh(
        self, expected_transmittance_texture
    ):
        mesh = gltf.import_mesh(os.path.join(
            SAMPLE_DIR,
            'transmission_test.gltf'
        ))
        assert mesh.materials[0].transmittance_value is None
        assert mesh.materials[0].transmittance_texture is None

        assert mesh.materials[1].transmittance_value is None
        assert torch.equal(mesh.materials[1].transmittance_texture, expected_transmittance_texture)

        assert torch.equal(mesh.materials[2].transmittance_value, torch.tensor([1.0]))
        assert mesh.materials[2].transmittance_texture is None

        assert mesh.materials[3].transmittance_value is None
        assert torch.equal(mesh.materials[3].transmittance_texture, expected_transmittance_texture)

        assert torch.equal(mesh.materials[4].transmittance_value, torch.tensor([0.83]))
        assert mesh.materials[4].transmittance_texture is None

        assert mesh.materials[5].transmittance_value is None
        assert torch.equal(mesh.materials[5].transmittance_texture, 0.5 * expected_transmittance_texture)

        assert torch.equal(mesh.materials[6].transmittance_value, torch.tensor([0.0]))
        assert mesh.materials[6].transmittance_texture is None


class TestUnassignedMaterials:
    """Primitives without a ``material`` are valid glTF and must stay unassigned (-1)."""

    @staticmethod
    def _write_gltf(path, primitives, materials=None):
        # Four vertices and one uint16 index accessor per primitive, stored in a data URI buffer.
        positions = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [1., 1., 0.]], dtype=np.float32)
        blob = positions.tobytes()
        buffer_views = [{'buffer': 0, 'byteOffset': 0, 'byteLength': len(blob)}]
        accessors = [{'bufferView': 0, 'componentType': 5126, 'count': 4, 'type': 'VEC3',
                      'min': [0., 0., 0.], 'max': [1., 1., 0.]}]
        gltf_primitives = []
        for indices, material in primitives:
            indices = np.array(indices, dtype=np.uint16).tobytes()
            buffer_views.append({'buffer': 0, 'byteOffset': len(blob), 'byteLength': len(indices)})
            accessors.append({'bufferView': len(buffer_views) - 1, 'componentType': 5123,
                              'count': len(indices) // 2, 'type': 'SCALAR'})
            blob += indices + b'\0' * (-len(indices) % 4)
            primitive = {'attributes': {'POSITION': 0}, 'indices': len(accessors) - 1}
            if material is not None:
                primitive['material'] = material
            gltf_primitives.append(primitive)
        content = {
            'asset': {'version': '2.0'}, 'scene': 0, 'scenes': [{'nodes': [0]}],
            'nodes': [{'mesh': 0}], 'meshes': [{'primitives': gltf_primitives}],
            'buffers': [{'byteLength': len(blob), 'uri': 'data:application/octet-stream;base64,' +
                         base64.b64encode(blob).decode()}],
            'bufferViews': buffer_views, 'accessors': accessors
        }
        if materials is not None:
            content['materials'] = [{'name': name, 'pbrMetallicRoughness': {}} for name in materials]
        with open(path, 'w') as f:
            json.dump(content, f)

    @pytest.mark.parametrize('materials,primitives,expected_names,expected_assignments', [
        (None, [([0, 1, 2], None), ([1, 3, 2], None)], [], [-1, -1]),
        (['red'], [([0, 1, 2], 0), ([1, 3, 2], None)], ['red'], [0, -1]),
        (['red', 'blue'], [([0, 1, 2], 0), ([1, 3, 2], None)], ['red'], [0, -1]),
    ])
    def test_import_meshes(self, tmp_path, materials, primitives, expected_names, expected_assignments):
        path = str(tmp_path / 'unassigned.gltf')
        self._write_gltf(path, primitives, materials)
        meshes = gltf.import_meshes(path)
        assert len(meshes) == 1
        assert [m.material_name for m in meshes[0].materials] == expected_names
        assert meshes[0].material_assignments.tolist() == expected_assignments

        # Must agree with import_mesh, which keeps the global material list
        mesh = gltf.import_mesh(path)
        assert mesh.material_assignments.tolist() == [
            -1 if idx == -1 else materials.index(expected_names[idx]) for idx in expected_assignments]
