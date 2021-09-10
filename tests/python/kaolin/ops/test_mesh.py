# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

from kaolin.ops.batch import get_first_idx, tile_to_packed, list_to_packed

from kaolin.utils.testing import FLOAT_TYPES, with_seed, check_tensor
from kaolin.ops import mesh
from kaolin.ops.mesh.trianglemesh import _unbatched_subdivide_vertices
from kaolin.io import obj

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../samples/')

@pytest.mark.parametrize("device,dtype", FLOAT_TYPES)
class TestFaceAreas:
    def test_face_areas(self, device, dtype):
        vertices = torch.tensor([[[0., 0., 0.],
                                  [0., 0., 1.],
                                  [0., 1., 0.],
                                  [2., 0., 0.2]],
                                 [[-1., -1., -1.],
                                  [-1., -1., 1.],
                                  [-1, 1., -1.],
                                  [3, -1., -0.6]]],
                                device=device, dtype=dtype)
        faces = torch.tensor([[0, 1, 2],
                              [1, 0, 3]],
                             device=device, dtype=torch.long)
        output = mesh.face_areas(vertices, faces)
        expected_output = torch.tensor([[0.5, 1.], [2., 4.]], device=device, dtype=dtype)
        assert torch.equal(output, expected_output)

    def test_packed_face_areas(self, device, dtype):
        vertices = torch.tensor([[0., 0., 0.],
                                 [0., 0., 1.],
                                 [0., 1., 0.],
                                 [2., 0., 0.2],
                                 [0., 0., 0.],
                                 [0., 1., 1.],
                                 [2., 0., 0.]],
                                device=device, dtype=dtype)
        faces = torch.tensor([[0, 1, 2],
                              [1, 0, 3],
                              [0, 1, 2]], device=device, dtype=torch.long)
        first_idx_vertices = torch.LongTensor([0, 4, 7], device='cpu')
        num_faces_per_mesh = torch.LongTensor([2, 1], device='cpu')
        output = mesh.packed_face_areas(vertices, first_idx_vertices,
                                                faces, num_faces_per_mesh)
        expected_output = torch.tensor([0.5, 1., math.sqrt(2.)], device=device, dtype=dtype)
        assert torch.allclose(output, expected_output)


@pytest.mark.parametrize("device,dtype", FLOAT_TYPES)
class TestSamplePoints:

    @pytest.fixture(autouse=True)
    def vertices(self, device, dtype):
         vertices = torch.tensor([[[0., 0., 0.],
                                  [0., 0., 1.],
                                  [0., 1., 0.],
                                  [2., 0., 0.2]],
                                 [[-1., -1., -1.],
                                  [-1., -1., 1.],
                                  [-1, 1., -1.],
                                  [3, -1., -0.6]]],
                                device=device, dtype=dtype)
         return vertices

    @pytest.fixture(autouse=True)
    def faces(self, device, dtype):
        faces = torch.tensor([[0, 1, 2],
                              [1, 0, 3]],
                             device=device, dtype=torch.long)
        return faces

    ######## FIXED ########
    def test_sample_points(self, vertices, faces, device, dtype):
        batch_size, num_vertices = vertices.shape[:2]
        num_faces = faces.shape[0]
        num_samples = 1000

        points, face_choices = mesh.sample_points(vertices, faces, num_samples)

        check_tensor(points, shape=(batch_size, num_samples, 3),
                     dtype=dtype, device=device)
        check_tensor(face_choices, shape=(batch_size, num_samples),
                     dtype=torch.long, device=device)

        # check that all faces are sampled
        num_0 = torch.sum(face_choices == 0, dim=1)
        assert torch.all(num_0 + torch.sum(face_choices == 1, dim=1) == num_samples)
        sampling_prob = num_samples / 3.
        tolerance = sampling_prob * 0.2
        assert torch.all(num_0 < sampling_prob + tolerance) and \
               torch.all(num_0 > sampling_prob - tolerance)


        face_vertices = mesh.index_vertices_by_faces(vertices, faces)

        face_vertices_choices = torch.gather(
            face_vertices, 1, face_choices[:, :, None, None].repeat(1, 1, 3, 3))

        # compute distance from the point to the plan of the face picked
        face_normals = mesh.face_normals(face_vertices_choices, unit=True)

        v0_p = points - face_vertices_choices[:, :, 0]  # batch_size x num_points x 3
        len_v0_p = torch.sqrt(torch.sum(v0_p ** 2, dim=-1))
        cos_a = torch.matmul(v0_p.reshape(-1, 1, 3),
                             face_normals.reshape(-1, 3, 1)).reshape(
            batch_size, num_samples) / len_v0_p
        point_to_face_dist = len_v0_p * cos_a

        if dtype == torch.half:
            atol = 1e-2
            rtol = 1e-3
        else:
            atol = 1e-4
            rtol = 1e-5

        # check that the point is close to the plan
        assert torch.allclose(point_to_face_dist,
                              torch.zeros((batch_size, num_samples), device=device, dtype=dtype),
                              atol=atol, rtol=rtol)

        # check that the point lie in the triangle
        edges0 = face_vertices_choices[:, :, 1] - face_vertices_choices[:, :, 0]
        edges1 = face_vertices_choices[:, :, 2] - face_vertices_choices[:, :, 1]
        edges2 = face_vertices_choices[:, :, 0] - face_vertices_choices[:, :, 2]

        v0_p = points - face_vertices_choices[:, :, 0]
        v1_p = points - face_vertices_choices[:, :, 1]
        v2_p = points - face_vertices_choices[:, :, 2]

        # Normals of the triangle formed by an edge and the point
        normals1 = torch.cross(edges0, v0_p)
        normals2 = torch.cross(edges1, v1_p)
        normals3 = torch.cross(edges2, v2_p)
        # cross-product of those normals with the face normals must be positive
        margin = -5e-3 if dtype == torch.half else 0.
        assert torch.all(torch.matmul(normals1.reshape(-1, 1, 3),
                                      face_normals.reshape(-1, 3, 1)) >= margin)
        assert torch.all(torch.matmul(normals2.reshape(-1, 1, 3),
                                      face_normals.reshape(-1, 3, 1)) >= margin)
        assert torch.all(torch.matmul(normals3.reshape(-1, 1, 3),
                                      face_normals.reshape(-1, 3, 1)) >= margin)

    def test_sample_points_with_areas(self, vertices, faces, dtype, device):
        num_samples = 1000
        face_areas = mesh.face_areas(vertices, faces)
        points1, face_choices1 = with_seed(1234)(
            mesh.sample_points)(vertices, faces, num_samples, face_areas)
        points2, face_choices2 = with_seed(1234)(
            mesh.sample_points)(vertices, faces, num_samples)
        assert torch.allclose(points1, points2)
        assert torch.equal(face_choices1, face_choices2)

    def test_diff_sample_points(self, vertices, faces, device, dtype):
        num_samples = 1000
        points1, face_choices1 = with_seed(1234)(
            mesh.sample_points)(vertices, faces, num_samples)
        points2, face_choices2 = with_seed(1235)(
            mesh.sample_points)(vertices, faces, num_samples)
        assert not torch.equal(points1, points2)
        assert not torch.equal(face_choices1, face_choices2)

    ######## PACKED ########
    @pytest.fixture(autouse=True)
    def packed_vertices_info(self, device, dtype):
        vertices = torch.tensor([[0., 0., 0.],
                                 [0., 0., 1.],
                                 [0., 1., 0.],
                                 [2., 0., 0.2],
                                 [0., 0., 0.],
                                 [0., 1., 1.],
                                 [2., 0., 0.]],
                                device=device, dtype=dtype)
        first_idx_vertices = torch.LongTensor([0, 4, 7], device='cpu')
        return vertices, first_idx_vertices

    @pytest.fixture(autouse=True)
    def packed_faces_info(self, device, dtype):
        faces = torch.tensor([[0, 1, 2],
                              [1, 0, 3],
                              [0, 1, 2]], device=device, dtype=torch.long)
        num_faces_per_mesh = torch.LongTensor([2, 1], device='cpu')
        return faces, num_faces_per_mesh

    def test_packed_sample_points(self, packed_vertices_info, packed_faces_info,
                                  device, dtype):
        vertices, first_idx_vertices = packed_vertices_info
        faces, num_faces_per_mesh = packed_faces_info

        total_num_vertices = vertices.shape[0]
        total_num_faces = faces.shape[0]
        batch_size = num_faces_per_mesh.shape[0]
        num_samples = 1000

        points, face_choices = mesh.packed_sample_points(
            vertices, first_idx_vertices, faces, num_faces_per_mesh, num_samples)

        check_tensor(points, shape=(batch_size, num_samples, 3),
                     dtype=dtype, device=device)
        check_tensor(face_choices, shape=(batch_size, num_samples),
                     dtype=torch.long, device=device)

        # check that all faces are sampled
        assert torch.all(face_choices[1] == 2)
        num_0 = torch.sum(face_choices[0] == 0)
        assert num_0 + torch.sum(face_choices[0] == 1) == num_samples
        sampling_prob = num_samples / 3.
        tolerance = sampling_prob * 0.2
        assert (num_0 < sampling_prob + tolerance) and \
               (num_0 > sampling_prob - tolerance)

        merged_faces = faces + tile_to_packed(first_idx_vertices[:-1].to(vertices.device),
                                              num_faces_per_mesh)

        face_vertices = torch.index_select(
            vertices, 0, merged_faces.reshape(-1)).reshape(total_num_faces, 3, 3)

        face_vertices_choices = torch.gather(
            face_vertices, 0, face_choices.reshape(-1, 1, 1).repeat(1, 3, 3)
        ).reshape(batch_size, num_samples, 3, 3)

        # compute distance from the point to the plan of the face picked
        face_normals = mesh.face_normals(face_vertices_choices, unit=True)
        v0_p = points - face_vertices_choices[:, :, 0]  # batch_size x num_points x 3
        len_v0_p = torch.sqrt(torch.sum(v0_p ** 2, dim=-1))
        cos_a = torch.matmul(v0_p.reshape(-1, 1, 3),
                             face_normals.reshape(-1, 3, 1)).reshape(
            batch_size, num_samples) / len_v0_p
        point_to_face_dist = len_v0_p * cos_a

        if dtype == torch.half:
            atol = 1e-2
            rtol = 1e-3
        else:
            atol = 1e-4
            rtol = 1e-5

        # check that the point is close to the plan
        assert torch.allclose(point_to_face_dist,
                              torch.zeros((batch_size, num_samples),
                                          device=device, dtype=dtype),
                              atol=atol, rtol=rtol)

        # check that the point lie in the triangle
        edges0 = face_vertices_choices[:, :, 1] - face_vertices_choices[:, :, 0]
        edges1 = face_vertices_choices[:, :, 2] - face_vertices_choices[:, :, 1]
        edges2 = face_vertices_choices[:, :, 0] - face_vertices_choices[:, :, 2]

        v0_p = points - face_vertices_choices[:, :, 0]
        v1_p = points - face_vertices_choices[:, :, 1]
        v2_p = points - face_vertices_choices[:, :, 2]

        # Normals of the triangle formed by an edge and the point
        normals1 = torch.cross(edges0, v0_p)
        normals2 = torch.cross(edges1, v1_p)
        normals3 = torch.cross(edges2, v2_p)
        # cross-product of those normals with the face normals must be positive
        margin = -2e-3 if dtype == torch.half else 0.
        assert torch.all(torch.matmul(normals1.reshape(-1, 1, 3),
                                      face_normals.reshape(-1, 3, 1)) >= margin)
        assert torch.all(torch.matmul(normals2.reshape(-1, 1, 3),
                                      face_normals.reshape(-1, 3, 1)) >= margin)
        assert torch.all(torch.matmul(normals3.reshape(-1, 1, 3),
                                      face_normals.reshape(-1, 3, 1)) >= margin)

    def test_packed_sample_points_with_areas(self, packed_vertices_info, packed_faces_info,
                                             dtype, device):
        num_samples = 1000
        vertices, first_idx_vertices = packed_vertices_info
        faces, num_faces_per_mesh = packed_faces_info

        face_areas = mesh.packed_face_areas(vertices, first_idx_vertices,
                                                    faces, num_faces_per_mesh)

        points1, face_choices1 = with_seed(1234)(mesh.packed_sample_points)(
            vertices, first_idx_vertices, faces, num_faces_per_mesh, num_samples, face_areas)

        points2, face_choices2 = with_seed(1234)(mesh.packed_sample_points)(
            vertices, first_idx_vertices, faces, num_faces_per_mesh, num_samples)

        assert torch.allclose(points1, points2)
        assert torch.equal(face_choices1, face_choices2)

    def test_diff_packed_sample_points(self, packed_vertices_info, packed_faces_info,
                                       dtype, device):
        num_samples = 1000
        vertices, first_idx_vertices = packed_vertices_info
        faces, num_faces_per_mesh = packed_faces_info

        points1, face_choices1 = with_seed(1234)(mesh.packed_sample_points)(
            vertices, first_idx_vertices, faces, num_faces_per_mesh, num_samples)
        points2, face_choices2 = with_seed(1235)(mesh.packed_sample_points)(
            vertices, first_idx_vertices, faces, num_faces_per_mesh, num_samples)

        assert not torch.equal(points1, points2)
        assert not torch.equal(face_choices1, face_choices2)

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

@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
class TestSubdivide:

    def test_subdivide(self, device, dtype):
        vertices = torch.tensor([[0, 0, 0],
                                 [1, 0, 0],
                                 [0, 0, 1]], dtype=dtype, device=device)

        faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        new_vertices = _unbatched_subdivide_vertices(vertices, faces, 3)
        expected_vertices = torch.tensor([[0.0000, 0.0000, 0.0000],
                                          [0.0000, 0.0000, 0.1250],
                                          [0.0000, 0.0000, 0.2500],
                                          [0.0000, 0.0000, 0.3750],
                                          [0.0000, 0.0000, 0.5000],
                                          [0.0000, 0.0000, 0.6250],
                                          [0.0000, 0.0000, 0.7500],
                                          [0.0000, 0.0000, 0.8750],
                                          [0.0000, 0.0000, 1.0000],
                                          [0.1250, 0.0000, 0.0000],
                                          [0.1250, 0.0000, 0.1250],
                                          [0.1250, 0.0000, 0.2500],
                                          [0.1250, 0.0000, 0.3750],
                                          [0.1250, 0.0000, 0.5000],
                                          [0.1250, 0.0000, 0.6250],
                                          [0.1250, 0.0000, 0.7500],
                                          [0.1250, 0.0000, 0.8750],
                                          [0.2500, 0.0000, 0.0000],
                                          [0.2500, 0.0000, 0.1250],
                                          [0.2500, 0.0000, 0.2500],
                                          [0.2500, 0.0000, 0.3750],
                                          [0.2500, 0.0000, 0.5000],
                                          [0.2500, 0.0000, 0.6250],
                                          [0.2500, 0.0000, 0.7500],
                                          [0.3750, 0.0000, 0.0000],
                                          [0.3750, 0.0000, 0.1250],
                                          [0.3750, 0.0000, 0.2500],
                                          [0.3750, 0.0000, 0.3750],
                                          [0.3750, 0.0000, 0.5000],
                                          [0.3750, 0.0000, 0.6250],
                                          [0.5000, 0.0000, 0.0000],
                                          [0.5000, 0.0000, 0.1250],
                                          [0.5000, 0.0000, 0.2500],
                                          [0.5000, 0.0000, 0.3750],
                                          [0.5000, 0.0000, 0.5000],
                                          [0.6250, 0.0000, 0.0000],
                                          [0.6250, 0.0000, 0.1250],
                                          [0.6250, 0.0000, 0.2500],
                                          [0.6250, 0.0000, 0.3750],
                                          [0.7500, 0.0000, 0.0000],
                                          [0.7500, 0.0000, 0.1250],
                                          [0.7500, 0.0000, 0.2500],
                                          [0.8750, 0.0000, 0.0000],
                                          [0.8750, 0.0000, 0.1250],
                                          [1.0000, 0.0000, 0.0000]], dtype=dtype, device=device)

        assert torch.equal(new_vertices, expected_vertices)

    def test_subdivide_2(self, device, dtype):
        vertices = torch.tensor([[0, 0, 0],
                                 [1, 0, 0],
                                 [0, 0, 1]], dtype=dtype, device=device)

        faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        new_vertices = _unbatched_subdivide_vertices(vertices, faces, 2)
        expected_vertices = torch.tensor([[0.0000, 0.0000, 0.0000],
                                          [0.0000, 0.0000, 0.1250],
                                          [0.0000, 0.0000, 0.2500],
                                          [0.0000, 0.0000, 0.3750],
                                          [0.0000, 0.0000, 0.5000],
                                          [0.0000, 0.0000, 0.6250],
                                          [0.0000, 0.0000, 0.7500],
                                          [0.0000, 0.0000, 0.8750],
                                          [0.0000, 0.0000, 1.0000],
                                          [0.1250, 0.0000, 0.0000],
                                          [0.1250, 0.0000, 0.1250],
                                          [0.1250, 0.0000, 0.2500],
                                          [0.1250, 0.0000, 0.3750],
                                          [0.1250, 0.0000, 0.5000],
                                          [0.1250, 0.0000, 0.6250],
                                          [0.1250, 0.0000, 0.7500],
                                          [0.1250, 0.0000, 0.8750],
                                          [0.2500, 0.0000, 0.0000],
                                          [0.2500, 0.0000, 0.1250],
                                          [0.2500, 0.0000, 0.2500],
                                          [0.2500, 0.0000, 0.3750],
                                          [0.2500, 0.0000, 0.5000],
                                          [0.2500, 0.0000, 0.6250],
                                          [0.2500, 0.0000, 0.7500],
                                          [0.3750, 0.0000, 0.0000],
                                          [0.3750, 0.0000, 0.1250],
                                          [0.3750, 0.0000, 0.2500],
                                          [0.3750, 0.0000, 0.3750],
                                          [0.3750, 0.0000, 0.5000],
                                          [0.3750, 0.0000, 0.6250],
                                          [0.5000, 0.0000, 0.0000],
                                          [0.5000, 0.0000, 0.1250],
                                          [0.5000, 0.0000, 0.2500],
                                          [0.5000, 0.0000, 0.3750],
                                          [0.5000, 0.0000, 0.5000],
                                          [0.6250, 0.0000, 0.0000],
                                          [0.6250, 0.0000, 0.1250],
                                          [0.6250, 0.0000, 0.2500],
                                          [0.6250, 0.0000, 0.3750],
                                          [0.7500, 0.0000, 0.0000],
                                          [0.7500, 0.0000, 0.1250],
                                          [0.7500, 0.0000, 0.2500],
                                          [0.8750, 0.0000, 0.0000],
                                          [0.8750, 0.0000, 0.1250],
                                          [1.0000, 0.0000, 0.0000]], device=device, dtype=dtype)

        assert torch.equal(new_vertices, expected_vertices)

    def test_subdivide_3(self, device, dtype):
        vertices = torch.tensor([[0, 0, 0],
                                 [0, 0.5, 0],
                                 [0, 0, 1]], dtype=dtype, device=device)

        faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        new_vertices = _unbatched_subdivide_vertices(vertices, faces, 2)
        expected_vertices = torch.tensor([[0.0000, 0.0000, 0.0000],
                                          [0.0000, 0.0000, 0.1250],
                                          [0.0000, 0.0000, 0.2500],
                                          [0.0000, 0.0000, 0.3750],
                                          [0.0000, 0.0000, 0.5000],
                                          [0.0000, 0.0000, 0.6250],
                                          [0.0000, 0.0000, 0.7500],
                                          [0.0000, 0.0000, 0.8750],
                                          [0.0000, 0.0000, 1.0000],
                                          [0.0000, 0.0625, 0.0000],
                                          [0.0000, 0.0625, 0.1250],
                                          [0.0000, 0.0625, 0.2500],
                                          [0.0000, 0.0625, 0.3750],
                                          [0.0000, 0.0625, 0.5000],
                                          [0.0000, 0.0625, 0.6250],
                                          [0.0000, 0.0625, 0.7500],
                                          [0.0000, 0.0625, 0.8750],
                                          [0.0000, 0.1250, 0.0000],
                                          [0.0000, 0.1250, 0.1250],
                                          [0.0000, 0.1250, 0.2500],
                                          [0.0000, 0.1250, 0.3750],
                                          [0.0000, 0.1250, 0.5000],
                                          [0.0000, 0.1250, 0.6250],
                                          [0.0000, 0.1250, 0.7500],
                                          [0.0000, 0.1875, 0.0000],
                                          [0.0000, 0.1875, 0.1250],
                                          [0.0000, 0.1875, 0.2500],
                                          [0.0000, 0.1875, 0.3750],
                                          [0.0000, 0.1875, 0.5000],
                                          [0.0000, 0.1875, 0.6250],
                                          [0.0000, 0.2500, 0.0000],
                                          [0.0000, 0.2500, 0.1250],
                                          [0.0000, 0.2500, 0.2500],
                                          [0.0000, 0.2500, 0.3750],
                                          [0.0000, 0.2500, 0.5000],
                                          [0.0000, 0.3125, 0.0000],
                                          [0.0000, 0.3125, 0.1250],
                                          [0.0000, 0.3125, 0.2500],
                                          [0.0000, 0.3125, 0.3750],
                                          [0.0000, 0.3750, 0.0000],
                                          [0.0000, 0.3750, 0.1250],
                                          [0.0000, 0.3750, 0.2500],
                                          [0.0000, 0.4375, 0.0000],
                                          [0.0000, 0.4375, 0.1250],
                                          [0.0000, 0.5000, 0.0000]], dtype=dtype, device=device)

        assert torch.equal(new_vertices, expected_vertices)

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
class TestCheckSign:

    @pytest.fixture(autouse=True)
    def verts(self, device):
        verts = []
        verts.append(torch.tensor([[0., 0., 0.],
                              [1., 0.5, 1.],
                              [0.5, 1., 1.],
                              [1., 1., 0.5]], device = device))
        verts.append(torch.tensor([[0., 0., 0.],
                              [1., 0, 0],
                              [0, 0, 1.],
                              [0, 1., 0]], device = device))
        return torch.stack(verts)

    @pytest.fixture(autouse=True)
    def faces(self, device):
        faces = torch.tensor([[0, 3, 1],
                              [0, 1, 2],
                              [0, 2, 3],
                              [3, 2, 1]], device = device)
        return faces

    @pytest.fixture(autouse=True)
    def points(self, device):
        axis = torch.linspace(0.1, 0.9, 3, device = device)
        p_x, p_y, p_z = torch.meshgrid(axis + 0.01, axis + 0.02, axis + 0.03)
        points = torch.cat((p_x.unsqueeze(-1), p_y.unsqueeze(-1), p_z.unsqueeze(-1)), dim=3)
        points = points.view(1, -1, 3).expand(2, -1, -1)
        return points

    @pytest.fixture(autouse=True)
    def expected(self, device):
        expected = []
        expected.append(torch.tensor([ True, False, False, False, False, False, False, False, 
                                    False, False, False, False, False,  True, False, False, 
                                    False,  True, False, False, False, False, False,  True, 
                                    False,  True, False], device=device))
        expected.append(torch.tensor([ True,  True, False,  True, False, False, False, False, False,  True,
         False, False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False], device=device))
        return torch.stack(expected)

    def test_verts_type(self, verts, faces, points):
        with pytest.raises(TypeError,
                           match=r"Expected verts entries to be torch.float32 "
                                 r"but got torch.float64."):
            verts = verts.double()
            mesh.check_sign(verts, faces, points)

    def test_faces_type(self, verts, faces, points):
        with pytest.raises(TypeError,
                           match=r"Expected faces entries to be torch.int64 "
                                 r"but got torch.int32."):
            faces = faces.int()
            mesh.check_sign(verts, faces, points)

    def test_points_type(self, verts, faces, points):
        with pytest.raises(TypeError,
                           match=r"Expected points entries to be torch.float32 "
                                 r"but got torch.float16."):
            points = points.half()
            mesh.check_sign(verts, faces, points)

    def test_hash_resolution_type(self, verts, faces, points):
        with pytest.raises(TypeError,
                           match=r"Expected hash_resolution to be int "
                                 r"but got <class 'float'>."):
            mesh.check_sign(verts, faces, points, 512.0)

    def test_verts_ndim(self, verts, faces, points):
        with pytest.raises(ValueError,
                           match=r"Expected verts to have 3 dimensions "
                                 r"but got 4 dimensions."):
            verts = verts.unsqueeze(-1)
            mesh.check_sign(verts, faces, points)

    def test_faces_ndim(self, verts, faces, points):
        with pytest.raises(ValueError,
                           match=r"Expected faces to have 2 dimensions "
                                 r"but got 3 dimensions."):
            faces = faces.unsqueeze(-1)
            mesh.check_sign(verts, faces, points)

    def test_points_ndim(self, verts, faces, points):
        with pytest.raises(ValueError,
                           match=r"Expected points to have 3 dimensions "
                                 r"but got 4 dimensions."):
            points = points.unsqueeze(-1)
            mesh.check_sign(verts, faces, points)

    def test_verts_shape(self, verts, faces, points):
        with pytest.raises(ValueError,
                           match=r"Expected verts to have 3 coordinates "
                                 r"but got 2 coordinates."):
            verts = verts[...,:2]
            mesh.check_sign(verts, faces, points)

    def test_faces_shape(self, verts, faces, points):
        with pytest.raises(ValueError,
                           match=r"Expected faces to have 3 vertices "
                                 r"but got 2 vertices."):
            faces = faces[:,:2]
            mesh.check_sign(verts, faces, points)

    def test_points_shape(self, verts, faces, points):
        with pytest.raises(ValueError,
                           match=r"Expected points to have 3 coordinates "
                                 r"but got 2 coordinates."):
            points = points[...,:2]
            mesh.check_sign(verts, faces, points)

    def test_single_batch(self, verts, faces, points, expected):
        output = mesh.check_sign(verts[0:1], faces, points[0:1])
        assert(torch.equal(output, expected[0:1]))

    def test_meshes(self, verts, faces, points, expected):
        output = mesh.check_sign(verts, faces, points)
        assert(torch.equal(output, expected))

    def test_faces_with_zero_area(self, verts, faces, points, expected):
        faces = torch.cat([faces, torch.tensor([[1, 1, 1],
                              [0, 0, 0],
                              [2, 2, 2],
                              [3, 3, 3]]).to(faces.device)])
        output = mesh.check_sign(verts, faces, points)
        assert(torch.equal(output, expected))

