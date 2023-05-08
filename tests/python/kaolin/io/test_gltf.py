# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from PIL import Image
import numpy as np

import torch

import kaolin
from kaolin.io import utils, gltf
from kaolin.utils.testing import print_namedtuple_attributes, print_dict_attributes, \
    check_tensor_attribute_shapes, contained_torch_equal, check_allclose

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, 'samples')
SAMPLE_DIR = os.path.join(ROOT_DIR, 'gltf/')

# TODO(cfujitsang): Add sanity test over a dataset like ShapeNet
class TestGLTF:
    @pytest.fixture(autouse=True)
    def expected_vertices(self):
        return torch.load(os.path.join(SAMPLE_DIR, 'gt_vertices.pt'))

    @pytest.fixture(autouse=True)
    def expected_faces(self):
        return torch.load(os.path.join(SAMPLE_DIR, 'gt_faces.pt'))

    @pytest.fixture(autouse=True)
    def expected_uvs(self):
        return torch.load(os.path.join(SAMPLE_DIR, 'gt_uvs.pt'))

    @pytest.fixture(autouse=True)
    def expected_face_uvs_idx(self, expected_faces):
        return expected_faces

    @pytest.fixture(autouse=True)
    def expected_diffuse_texture(self):
        img = Image.open(os.path.join(SAMPLE_DIR, 'Avocado_baseColor.png'))
        return torch.as_tensor(np.array(img)).float() * (1. / 255.)

    @pytest.fixture(autouse=True)
    def expected_metallic_roughness_texture(self):
        img = Image.open(os.path.join(SAMPLE_DIR, 'Avocado_roughnessMetallic.png'))
        return torch.as_tensor(np.array(img)).float() * (1. / 255.)

    @pytest.fixture(autouse=True)
    def expected_roughness_texture(self, expected_metallic_roughness_texture):
        return expected_metallic_roughness_texture[..., 1:2]

    @pytest.fixture(autouse=True)
    def expected_metallic_texture(self, expected_metallic_roughness_texture):
        return expected_metallic_roughness_texture[..., 2:3]

    @pytest.fixture(autouse=True)
    def expected_normals_texture(self):
        img = Image.open(os.path.join(SAMPLE_DIR, 'Avocado_normal.png'))
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
            'Avocado.gltf'
        ))
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
