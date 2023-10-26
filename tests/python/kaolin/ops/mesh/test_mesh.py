# Copyright (c) 2019,20-22-23 NVIDIA CORPORATION & AFFILIATES.
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

import math
import pytest
import os

import torch

from kaolin.utils.testing import FLOAT_TYPES
from kaolin.ops import mesh
from kaolin.io import obj

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, os.pardir, os.pardir, os.pardir, 'samples/')


@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
def test_adjacency_matrix_sparse(device, dtype):
    num_vertices = 5
    faces = torch.tensor([[1, 3, 2],
                          [1, 4, 0]], dtype=torch.long, device=device)

    output = mesh.adjacency_matrix(num_vertices, faces).to_dense()
    expected = torch.tensor([[0, 1, 0, 0, 1],
                             [1, 0, 1, 1, 1],
                             [0, 1, 0, 1, 0],
                             [0, 1, 1, 0, 0],
                             [1, 1, 0, 0, 0]], dtype=torch.float, device=device)

    assert torch.equal(output, expected)


@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
def test_adjacency_matrix_dense(device, dtype):
    num_vertices = 5
    faces = torch.tensor([[1, 3, 2],
                          [1, 4, 0]], dtype=torch.long, device=device)

    output = mesh.adjacency_matrix(num_vertices, faces, sparse=False)
    expected = torch.tensor([[0, 1, 0, 0, 1],
                             [1, 0, 1, 1, 1],
                             [0, 1, 0, 1, 0],
                             [0, 1, 1, 0, 0],
                             [1, 1, 0, 0, 0]], dtype=torch.float, device=device)
    assert torch.equal(output, expected)


@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
def test_adjacency_consistent(device, dtype):
    test_mesh = obj.import_mesh(os.path.join(ROOT_DIR, 'model.obj'))
    vertices = test_mesh.vertices
    faces = test_mesh.faces

    num_vertices = vertices.shape[0]

    sparse = mesh.adjacency_matrix(num_vertices, faces)
    sparse_to_dense = sparse.to_dense()
    dense = mesh.adjacency_matrix(num_vertices, faces, sparse=False)

    assert torch.equal(sparse_to_dense, dense)


@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
class TestUniformLaplacian:

    def test_uniform_laplacian(self, device, dtype):

        num_vertices = 5
        faces = torch.tensor([[1, 3, 2],
                              [1, 4, 0]], dtype=torch.long, device=device)

        output = mesh.uniform_laplacian(num_vertices, faces)
        expected = torch.tensor([[-1, 0.5, 0, 0, 0.5],
                                 [0.25, -1, 0.25, 0.25, 0.25],
                                 [0, 0.5, -1, 0.5, 0],
                                 [0, 0.5, 0.5, -1, 0],
                                 [0.5, 0.5, 0, 0, -1]], dtype=torch.float, device=device)

        assert torch.equal(output, expected)

    def test_not_connected_mesh(self, device, dtype):
        num_vertices = 4
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        result = mesh.uniform_laplacian(num_vertices, faces)

        # Any row and column related to V3 is zeros.
        assert torch.equal(result[3, :3], torch.zeros((3), device=device, dtype=torch.float))
        assert torch.equal(result[:3, 3], torch.zeros((3), device=device, dtype=torch.float))

@pytest.mark.parametrize('device,dtype', FLOAT_TYPES)
class TestComputeVertexNormals:
    def test_compute_vertex_normals(self, device, dtype):
        # Faces are a fan around the 0th vertex
        faces = torch.tensor([[0, 2, 1],
                              [0, 3, 2],
                              [0, 4, 3]],
                             device=device, dtype=torch.long)
        B = 3
        F = faces.shape[0]
        FSize = faces.shape[1]
        V = 6  # one vertex not in faces
        face_normals = torch.rand((B, F, FSize, 3), device=device, dtype=dtype)

        expected = torch.zeros((B, V, 3), device=device, dtype=dtype)
        for b in range(B):
            expected[b, 0, :] = (face_normals[b, 0, 0, :] + face_normals[b, 1, 0, :] + face_normals[b, 2, 0, :]) / 3
            expected[b, 1, :] = face_normals[b, 0, 2, :]
            expected[b, 2, :] = (face_normals[b, 0, 1, :] + face_normals[b, 1, 2, :]) / 2
            expected[b, 3, :] = (face_normals[b, 1, 1, :] + face_normals[b, 2, 2, :]) / 2
            expected[b, 4, :] = face_normals[b, 2, 1, :]
            expected[b, 5, :] = 0  # DNE in faces

        vertex_normals = mesh.compute_vertex_normals(faces, face_normals, num_vertices=V)
        assert torch.allclose(expected, vertex_normals)

        # Now let's not pass in num_vertices; we will not get normals for the last vertex which is not in faces
        vertex_normals = mesh.compute_vertex_normals(faces, face_normals)
        assert torch.allclose(expected[:, :5, :], vertex_normals)
