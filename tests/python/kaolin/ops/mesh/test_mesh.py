# Copyright (c) 2019,20-22 NVIDIA CORPORATION & AFFILIATES.
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

from kaolin.ops.batch import get_first_idx, tile_to_packed, list_to_packed

from kaolin.utils.testing import FLOAT_TYPES, with_seed, check_tensor
from kaolin.ops import mesh
from kaolin.ops.mesh.trianglemesh import _unbatched_subdivide_vertices, subdivide_trianglemesh
from kaolin.io import obj

ROOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, os.pardir, os.pardir, os.pardir, 'samples/')


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
        # TODO(cfujitsang): extend the test with Z variation
        return torch.tensor([[[0., 0., 0.],
                              [0., 1., 0.],
                              [1., 0., 0.],
                              [-1, 0., 0.]],
                             [[1., 1., 3.],
                              [1., 1.5, 3.],
                              [1.5, 1., 3.],
                              [0.5, 1., 3.]]],
                            device=device, dtype=dtype)
        return vertices

    @pytest.fixture(autouse=True)
    def faces(self, device, dtype):
        return torch.tensor([[0, 1, 2],
                             [1, 0, 3]],
                            device=device, dtype=torch.long)

    @pytest.fixture(autouse=True)
    def face_features(self, device, dtype):
        return torch.tensor(
            [[[[0., 0.], [0., 1.], [0., 2.]],
              [[1., 3.], [1., 4.], [1., 5.]]],
             [[[2., 6.], [2., 7.], [2., 8.]],
              [[3., 9.], [3., 10.], [3., 11.]]]],
            device=device, dtype=dtype)

    ######## FIXED ########
    @pytest.mark.parametrize('use_features', [False, True])
    def test_sample_points(self, vertices, faces, face_features,
                           use_features, device, dtype):
        batch_size, num_vertices = vertices.shape[:2]
        num_faces = faces.shape[0]
        num_samples = 1000

        if use_features:
            points, face_choices, interpolated_features = mesh.sample_points(
                vertices, faces, num_samples, face_features=face_features)
        else:
            points, face_choices = mesh.sample_points(
                vertices, faces, num_samples)

        check_tensor(points, shape=(batch_size, num_samples, 3),
                     dtype=dtype, device=device)
        check_tensor(face_choices, shape=(batch_size, num_samples),
                     dtype=torch.long, device=device)

        # check that all faces are sampled
        num_0 = torch.sum(face_choices == 0, dim=1)
        assert torch.all(num_0 + torch.sum(face_choices == 1, dim=1) == num_samples)
        sampling_prob = num_samples / 2
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
        margin = -5e-3 if dtype == torch.half else 0.
        assert torch.all(torch.matmul(normals1.reshape(-1, 1, 3),
                                      face_normals.reshape(-1, 3, 1)) >= margin)
        assert torch.all(torch.matmul(normals2.reshape(-1, 1, 3),
                                      face_normals.reshape(-1, 3, 1)) >= margin)
        assert torch.all(torch.matmul(normals3.reshape(-1, 1, 3),
                                      face_normals.reshape(-1, 3, 1)) >= margin)
        if use_features:
            feat_dim = face_features.shape[-1]
            check_tensor(interpolated_features, shape=(batch_size, num_samples, feat_dim),
                         dtype=dtype, device=device)
            # face_vertices_choices (batch_size, num_samples, 3, 3)
            # points (batch_size, num_samples, 3)
            ax = face_vertices_choices[:, :, 0, 0]
            ay = face_vertices_choices[:, :, 0, 1]
            bx = face_vertices_choices[:, :, 1, 0]
            by = face_vertices_choices[:, :, 1, 1]
            cx = face_vertices_choices[:, :, 2, 0]
            cy = face_vertices_choices[:, :, 2, 1]
            m = bx - ax
            p = by - ay
            n = cx - ax
            q = cy - ay
            s = points[:, :, 0] - ax
            t = points[:, :, 1] - ay

            # sum_weights = torch.sum(weights, dim=-1)
            # zeros_idxs = torch.where(sum_weights == 0)
            #weights = weights / torch.sum(weights, keepdims=True, dim=-1)
            k1 = s * q - n * t
            k2 = m * t - s * p
            k3 = m * q - n * p
            w1 = k1 / (k3 + 1e-7)
            w2 = k2 / (k3 + 1e-7)
            w0 = (1. - w1) - w2
            weights = torch.stack([w0, w1, w2], dim=-1)

            gt_points = torch.sum(
                face_vertices_choices * weights.unsqueeze(-1), dim=-2)
            assert torch.allclose(points, gt_points, atol=atol, rtol=rtol)

            _face_choices = face_choices[..., None, None].repeat(1, 1, 3, feat_dim)
            face_features_choices = torch.gather(face_features, 1, _face_choices)

            gt_interpolated_features = torch.sum(
                face_features_choices * weights.unsqueeze(-1), dim=-2)
            assert torch.allclose(interpolated_features, gt_interpolated_features,
                                  atol=atol, rtol=rtol)

    def test_sample_points_with_areas(self, vertices, faces, dtype, device):
        num_samples = 1000
        face_areas = mesh.face_areas(vertices, faces)
        points1, face_choices1 = with_seed(1234)(
            mesh.sample_points)(vertices, faces, num_samples, face_areas)
        points2, face_choices2 = with_seed(1234)(
            mesh.sample_points)(vertices, faces, num_samples)
        assert torch.allclose(points1, points2)
        assert torch.equal(face_choices1, face_choices2)

    def test_sample_points_with_areas_with_features(self, vertices, faces,
                                                    face_features, dtype, device):
        num_samples = 1000
        face_areas = mesh.face_areas(vertices, faces)
        points1, face_choices1, interpolated_features1 = with_seed(1234)(
            mesh.sample_points)(vertices, faces, num_samples, face_areas,
                                face_features=face_features)
        points2, face_choices2, interpolated_features2 = with_seed(1234)(
            mesh.sample_points)(vertices, faces, num_samples,
                                face_features=face_features)
        assert torch.allclose(points1, points2)
        assert torch.equal(face_choices1, face_choices2)
        assert torch.allclose(interpolated_features1, interpolated_features2)

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
class TestSubdivideTrianglemesh:

    @pytest.fixture(autouse=True)
    def vertices_icosahedron(self, device):
        return torch.tensor([[[-0.5257, 0.8507, 0.0000],
                              [0.5257, 0.8507, 0.0000],
                              [-0.5257, -0.8507, 0.0000],
                              [0.5257, -0.8507, 0.0000],
                              [0.0000, -0.5257, 0.8507],
                              [0.0000, 0.5257, 0.8507],
                              [0.0000, -0.5257, -0.8507],
                              [0.0000, 0.5257, -0.8507],
                              [0.8507, 0.0000, -0.5257],
                              [0.8507, 0.0000, 0.5257],
                              [-0.8507, 0.0000, -0.5257],
                              [-0.8507, 0.0000, 0.5257]]], dtype=torch.float, device=device)

    @pytest.fixture(autouse=True)
    def faces_icosahedron(self, device):
        return torch.tensor([[0, 11, 5],
                             [0, 5, 1],
                             [0, 1, 7],
                             [0, 7, 10],
                             [0, 10, 11],
                             [1, 5, 9],
                             [5, 11, 4],
                             [11, 10, 2],
                             [10, 7, 6],
                             [7, 1, 8],
                             [3, 9, 4],
                             [3, 4, 2],
                             [3, 2, 6],
                             [3, 6, 8],
                             [3, 8, 9],
                             [4, 9, 5],
                             [2, 4, 11],
                             [6, 2, 10],
                             [8, 6, 7],
                             [9, 8, 1]], dtype=torch.long, device=device)

    @pytest.fixture(autouse=True)
    def expected_vertices_default_alpha(self, device):
        return torch.tensor([[[-0.4035, 0.6529, 0.0000],
                              [0.4035, 0.6529, 0.0000],
                              [-0.4035, -0.6529, 0.0000],
                              [0.4035, -0.6529, 0.0000],
                              [0.0000, -0.4035, 0.6529],
                              [0.0000, 0.4035, 0.6529],
                              [0.0000, -0.4035, -0.6529],
                              [0.0000, 0.4035, -0.6529],
                              [0.6529, 0.0000, -0.4035],
                              [0.6529, 0.0000, 0.4035],
                              [-0.6529, 0.0000, -0.4035],
                              [-0.6529, 0.0000, 0.4035],
                              [0.0000, 0.7694, 0.0000],
                              [-0.2378, 0.6225, 0.3847],
                              [-0.2378, 0.6225, -0.3847],
                              [-0.6225, 0.3847, -0.2378],
                              [-0.6225, 0.3847, 0.2378],
                              [0.2378, 0.6225, 0.3847],
                              [0.2378, 0.6225, -0.3847],
                              [0.6225, 0.3847, -0.2378],
                              [0.6225, 0.3847, 0.2378],
                              [0.0000, -0.7694, 0.0000],
                              [-0.2378, -0.6225, 0.3847],
                              [-0.2378, -0.6225, -0.3847],
                              [-0.6225, -0.3847, -0.2378],
                              [-0.6225, -0.3847, 0.2378],
                              [0.2378, -0.6225, 0.3847],
                              [0.2378, -0.6225, -0.3847],
                              [0.6225, -0.3847, -0.2378],
                              [0.6225, -0.3847, 0.2378],
                              [0.0000, 0.0000, 0.7694],
                              [0.3847, -0.2378, 0.6225],
                              [-0.3847, -0.2378, 0.6225],
                              [0.3847, 0.2378, 0.6225],
                              [-0.3847, 0.2378, 0.6225],
                              [0.0000, 0.0000, -0.7694],
                              [0.3847, -0.2378, -0.6225],
                              [-0.3847, -0.2378, -0.6225],
                              [0.3847, 0.2378, -0.6225],
                              [-0.3847, 0.2378, -0.6225],
                              [0.7694, 0.0000, 0.0000],
                              [-0.7694, 0.0000, 0.0000]]], dtype=torch.float, device=device)

    @pytest.fixture(autouse=True)
    def expected_vertices_zero_alpha(self, device):
        return torch.tensor([[[-0.5257, 0.8507, 0.0000],
                            [0.5257, 0.8507, 0.0000],
                            [-0.5257, -0.8507, 0.0000],
                            [0.5257, -0.8507, 0.0000],
                            [0.0000, -0.5257, 0.8507],
                            [0.0000, 0.5257, 0.8507],
                            [0.0000, -0.5257, -0.8507],
                            [0.0000, 0.5257, -0.8507],
                            [0.8507, 0.0000, -0.5257],
                            [0.8507, 0.0000, 0.5257],
                            [-0.8507, 0.0000, -0.5257],
                            [-0.8507, 0.0000, 0.5257],
                            [0.0000, 0.7694, 0.0000],
                            [-0.2378, 0.6225, 0.3847],
                            [-0.2378, 0.6225, -0.3847],
                            [-0.6225, 0.3847, -0.2378],
                            [-0.6225, 0.3847, 0.2378],
                            [0.2378, 0.6225, 0.3847],
                            [0.2378, 0.6225, -0.3847],
                            [0.6225, 0.3847, -0.2378],
                            [0.6225, 0.3847, 0.2378],
                            [0.0000, -0.7694, 0.0000],
                            [-0.2378, -0.6225, 0.3847],
                            [-0.2378, -0.6225, -0.3847],
                            [-0.6225, -0.3847, -0.2378],
                            [-0.6225, -0.3847, 0.2378],
                            [0.2378, -0.6225, 0.3847],
                            [0.2378, -0.6225, -0.3847],
                            [0.6225, -0.3847, -0.2378],
                            [0.6225, -0.3847, 0.2378],
                            [0.0000, 0.0000, 0.7694],
                            [0.3847, -0.2378, 0.6225],
                            [-0.3847, -0.2378, 0.6225],
                            [0.3847, 0.2378, 0.6225],
                            [-0.3847, 0.2378, 0.6225],
                            [0.0000, 0.0000, -0.7694],
                            [0.3847, -0.2378, -0.6225],
                            [-0.3847, -0.2378, -0.6225],
                            [0.3847, 0.2378, -0.6225],
                            [-0.3847, 0.2378, -0.6225],
                            [0.7694, 0.0000, 0.0000],
                            [-0.7694, 0.0000, 0.0000]]], dtype=torch.float, device=device)

    @pytest.fixture(autouse=True)
    def expected_faces_icosahedron_1_iter(self, device):
        return torch.tensor([[11, 34, 16],
                            [0, 16, 13],
                            [5, 13, 34],
                            [13, 16, 34],
                            [5, 17, 13],
                            [0, 13, 12],
                            [1, 12, 17],
                            [12, 13, 17],
                            [1, 18, 12],
                            [0, 12, 14],
                            [7, 14, 18],
                            [14, 12, 18],
                            [7, 39, 14],
                            [0, 14, 15],
                            [10, 15, 39],
                            [15, 14, 39],
                            [10, 41, 15],
                            [0, 15, 16],
                            [11, 16, 41],
                            [16, 15, 41],
                            [5, 33, 17],
                            [1, 17, 20],
                            [9, 20, 33],
                            [20, 17, 33],
                            [11, 32, 34],
                            [5, 34, 30],
                            [4, 30, 32],
                            [30, 34, 32],
                            [10, 24, 41],
                            [11, 41, 25],
                            [2, 25, 24],
                            [25, 41, 24],
                            [7, 35, 39],
                            [10, 39, 37],
                            [6, 37, 35],
                            [37, 39, 35],
                            [1, 19, 18],
                            [7, 18, 38],
                            [8, 38, 19],
                            [38, 18, 19],
                            [9, 31, 29],
                            [3, 29, 26],
                            [4, 26, 31],
                            [26, 29, 31],
                            [4, 22, 26],
                            [3, 26, 21],
                            [2, 21, 22],
                            [21, 26, 22],
                            [2, 23, 21],
                            [3, 21, 27],
                            [6, 27, 23],
                            [27, 21, 23],
                            [6, 36, 27],
                            [3, 27, 28],
                            [8, 28, 36],
                            [28, 27, 36],
                            [8, 40, 28],
                            [3, 28, 29],
                            [9, 29, 40],
                            [29, 28, 40],
                            [9, 33, 31],
                            [4, 31, 30],
                            [5, 30, 33],
                            [30, 31, 33],
                            [4, 32, 22],
                            [2, 22, 25],
                            [11, 25, 32],
                            [25, 22, 32],
                            [2, 24, 23],
                            [6, 23, 37],
                            [10, 37, 24],
                            [37, 23, 24],
                            [6, 35, 36],
                            [8, 36, 38],
                            [7, 38, 35],
                            [38, 36, 35],
                            [8, 19, 40],
                            [9, 40, 20],
                            [1, 20, 19],
                            [20, 40, 19]], dtype=torch.long, device=device)

    def test_subdivide_trianglemesh_1_iter_default_alpha(self, vertices_icosahedron, faces_icosahedron, expected_vertices_default_alpha, expected_faces_icosahedron_1_iter):
        new_vertices, new_faces = subdivide_trianglemesh(vertices_icosahedron, faces_icosahedron, 1)
        assert torch.allclose(new_vertices, expected_vertices_default_alpha, atol=1e-04)
        assert torch.equal(new_faces, expected_faces_icosahedron_1_iter)

    def test_subdivide_trianglemesh_1_iter_zero_alpha(self, vertices_icosahedron, faces_icosahedron, expected_vertices_zero_alpha, expected_faces_icosahedron_1_iter):
        alpha = torch.zeros_like(vertices_icosahedron[..., 0])
        new_vertices, new_faces = subdivide_trianglemesh(vertices_icosahedron, faces_icosahedron, 1, alpha)
        assert torch.allclose(new_vertices, expected_vertices_zero_alpha, atol=1e-04)
        assert torch.equal(new_faces, expected_faces_icosahedron_1_iter)

    def test_subdivide_trianglemesh_5_iter(self, vertices_icosahedron, faces_icosahedron):
        new_vertices, new_faces = subdivide_trianglemesh(vertices_icosahedron, faces_icosahedron, 5)
        # check total area of all faces
        assert torch.allclose(mesh.face_areas(new_vertices, new_faces).sum(),
                              torch.tensor([6.2005], dtype=new_vertices.dtype, device=new_faces.device), atol=1e-04)
        assert new_faces.shape[0] == faces_icosahedron.shape[0] * 4 ** 5


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
