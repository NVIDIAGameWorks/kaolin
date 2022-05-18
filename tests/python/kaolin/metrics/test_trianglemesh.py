# Copyright (c) 2019,20-21 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import random

from kaolin.metrics import trianglemesh
from kaolin.ops.mesh import index_vertices_by_faces
from kaolin.utils.testing import FLOAT_TYPES, CUDA_FLOAT_TYPES

@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
def test_unbatched_naive_triangle_distance(device, dtype):
    pointcloud = torch.tensor([[0., -1., -1.],
                               [1., -1., -1.],
                               [-1., -1., -1.],
                               [0., -1., 2.],
                               [1., -1., 2.],
                               [-1, -1., 2.],
                               [0., 2., 0.5],
                               [1., 2., 0.5],
                               [-1., 2., 0.5],
                               [0., -1., 0.5],
                               [1., -1., 0.5],
                               [-1., -1., 0.5],
                               [0., 1., 1.],
                               [1., 1., 1.],
                               [-1., 1., 1.],
                               [0., 1., 0.],
                               [1., 1., 0.],
                               [-1., 1., 0.],
                               [1., 0.5, 0.5],
                               [-1., 0.5, 0.5]],
                              device=device, dtype=dtype)

    vertices = torch.tensor([[0., 0., 0.],
                             [0., 0., 1.],
                             [0., 1., 0.5],
                             [0.5, 0., 0.],
                             [0.5, 0., 1.],
                             [0.5, 1., 0.5]],
                            device=device, dtype=dtype)

    faces = torch.tensor([[0, 1, 2], [3, 4, 5]], device=device, dtype=torch.long)

    face_vertices = index_vertices_by_faces(vertices.unsqueeze(0), faces)[0]

    expected_dist = torch.tensor(
        [2.0000, 2.2500, 3.0000, 2.0000, 2.2500, 3.0000, 1.0000, 1.2500, 2.0000,
         1.0000, 1.2500, 2.0000, 0.2000, 0.4500, 1.2000, 0.2000, 0.4500, 1.2000,
         0.2500, 1.0000], device=device, dtype=dtype)

    expected_face_idx = torch.tensor(
        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
        device=device, dtype=torch.long)

    expected_dist_type = torch.tensor(
        [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 0, 0],
        device=device, dtype=torch.int)

    dist, face_idx, dist_type = trianglemesh._unbatched_naive_point_to_mesh_distance(
        pointcloud, face_vertices)

    assert torch.allclose(dist, expected_dist)
    assert torch.equal(face_idx, expected_face_idx)
    assert torch.equal(dist_type, expected_dist_type)

@pytest.mark.parametrize('num_points', [1025])
@pytest.mark.parametrize('num_faces', [1025])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
class TestUnbatchedTriangleDistanceCuda:
    @pytest.fixture(autouse=True)
    def pointcloud(self, num_points, dtype):
        return torch.randn((num_points, 3), device='cuda', dtype=dtype)

    @pytest.fixture(autouse=True)
    def face_vertices(self, num_faces, dtype):
        return torch.randn((num_faces, 3, 3), device='cuda', dtype=dtype)

    def test_face_vertices(self, pointcloud, face_vertices):
        dist, face_idx, dist_type = trianglemesh._UnbatchedTriangleDistanceCuda.apply(
            pointcloud, face_vertices)
        dist2, face_idx2, dist_type2 = trianglemesh._unbatched_naive_point_to_mesh_distance(
            pointcloud, face_vertices)
        assert torch.allclose(dist, dist2)
        assert torch.equal(face_idx, face_idx2)
        assert torch.equal(dist_type, dist_type2)

    def test_face_vertices_grad(self, pointcloud, face_vertices):
        pointcloud = pointcloud.detach()
        pointcloud.requires_grad = True
        face_vertices = face_vertices.detach()
        face_vertices.requires_grad = True
        pointcloud2 = pointcloud.detach()
        pointcloud2.requires_grad = True
        face_vertices2 = face_vertices.detach()
        face_vertices2.requires_grad = True
        dist, face_idx, dist_type = trianglemesh._UnbatchedTriangleDistanceCuda.apply(
            pointcloud, face_vertices)
        dist2, face_idx2, dist_type2 = trianglemesh._unbatched_naive_point_to_mesh_distance(
            pointcloud2, face_vertices2)
        grad_out = torch.rand_like(dist)
        dist.backward(grad_out)
        dist2.backward(grad_out)
        diff_idxs = torch.where(~torch.isclose(pointcloud.grad, pointcloud2.grad))
        assert torch.allclose(pointcloud.grad, pointcloud2.grad,
                              rtol=1e-5, atol=1e-5)
        assert torch.allclose(face_vertices.grad, face_vertices2.grad,
                              rtol=1e-5, atol=1e-5)

@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('num_points', [11, 1025])
@pytest.mark.parametrize('num_faces', [11, 1025])
def test_triangle_distance(batch_size, num_points, num_faces, device, dtype):
    pointclouds = torch.randn((batch_size, num_points, 3), device=device,
                              dtype=dtype)
    face_vertices = torch.randn((batch_size, num_faces, 3, 3), device=device,
                                dtype=dtype)
    expected_dist = []
    expected_face_idx = []
    expected_dist_type = []

    for i in range(batch_size):
        _expected_dist, _expected_face_idx, _expected_dist_type = \
            trianglemesh._unbatched_naive_point_to_mesh_distance(
                pointclouds[i], face_vertices[i])
        expected_dist.append(_expected_dist)
        expected_face_idx.append(_expected_face_idx)
        expected_dist_type.append(_expected_dist_type)
    expected_dist = torch.stack(expected_dist, dim=0)
    expected_face_idx = torch.stack(expected_face_idx, dim=0)
    expected_dist_type = torch.stack(expected_dist_type, dim=0)
    dist, face_idx, dist_type = trianglemesh.point_to_mesh_distance(
        pointclouds, face_vertices)
    assert torch.allclose(dist, expected_dist)
    assert torch.equal(face_idx, expected_face_idx)
    assert torch.equal(dist_type, expected_dist_type)

@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
class TestEdgeLength:

    @pytest.fixture(autouse=True)
    def get_tol(self, device, dtype):
        if dtype == torch.half:
            return 1e-2, 1e-2
        elif dtype == torch.float:
            return 1e-5, 1e-4
        elif dtype == torch.double:
            return 1e-6, 1e-5

    def test_edge_length(self, device, dtype, get_tol):

        atol, rtol = get_tol
        vertices = torch.tensor([[[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]],

                                 [[3, 0, 0],
                                  [0, 4, 0],
                                  [0, 0, 5]]], dtype=dtype, device=device)

        faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        output = trianglemesh.average_edge_length(vertices, faces)
        expected = torch.tensor([[1.4142], [5.7447]], device=device, dtype=dtype)

        assert torch.allclose(output, expected, atol=atol, rtol=rtol)

@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
def test_laplacian_smooth(device, dtype):
    vertices = torch.tensor([[[0, 0, 1],
                              [2, 1, 2],
                              [3, 1, 2]],
                             [[3, 1, 2],
                              [0, 0, 3],
                              [0, 3, 3]]], dtype=dtype, device=device)
    faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)
    output = trianglemesh.uniform_laplacian_smoothing(vertices, faces)
    expected = torch.tensor([[[2.5000, 1.0000, 2.0000],
                              [1.5000, 0.5000, 1.5000],
                              [1.0000, 0.5000, 1.5000]],
                             [[0.0000, 1.5000, 3.0000],
                              [1.5000, 2.0000, 2.5000],
                              [1.5000, 0.5000, 2.5000]]], dtype=dtype, device=device)

    assert torch.equal(output, expected)
