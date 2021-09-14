# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

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
from kaolin.utils.testing import FLOAT_TYPES, CUDA_FLOAT_TYPES, with_seed

torch_seed = 456
random_seed = 439

@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
class TestTriangleDistance:
    @pytest.fixture(autouse=True)
    def base_pointcloud(self, device, dtype):
        return torch.tensor([[0., -1., -1.],
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

    @pytest.fixture(autouse=True)
    def pointclouds(self, base_pointcloud):
        return torch.stack([base_pointcloud,
                            base_pointcloud * 0.2,
                            torch.flip(base_pointcloud, dims=(-1,))],
                           dim=0)

    @pytest.fixture(autouse=True)
    def base_vertices(self, device, dtype):
        return torch.tensor([[0., 0., 0.],
                             [0., 0., 1.],
                             [0., 1., 0.5],
                             [0.5, 0., 0.],
                             [0.5, 0., 1.],
                             [0.5, 1., 0.5]],
                            device=device, dtype=dtype,
                            requires_grad=True)

    @pytest.fixture(autouse=True)
    def vertices(self, base_vertices):
        return torch.stack([base_vertices,
                            base_vertices * 0.2,
                            torch.flip(base_vertices, dims=(-1,))],
                           dim=0)

    @pytest.fixture(autouse=True)
    def faces(self, device):
        return torch.tensor([[0, 1, 2], [3, 4, 5]], device=device, dtype=torch.long)

    @pytest.fixture(autouse=True)
    def expected_base_dist(self, device, dtype):
        return torch.tensor([2.0000, 2.2500, 3.0000, 2.0000, 2.2500, 3.0000, 1.0000, 1.2500, 2.0000,
            1.0000, 1.2500, 2.0000, 0.2000, 0.4500, 1.2000, 0.2000, 0.4500, 1.2000,
            0.2500, 1.0000], device=device, dtype=dtype)

    @pytest.fixture(autouse=True)
    def expected_dist(self, expected_base_dist):
        return torch.stack([expected_base_dist,
                            expected_base_dist * (0.2 ** 2),
                            expected_base_dist], dim=0)

    @pytest.fixture(autouse=True)
    def expected_face_idx(self, device, dtype):
        face_idx = torch.tensor([[0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0]],
                                device=device, dtype=torch.long)
        return face_idx.repeat(3, 1)

    @pytest.fixture(autouse=True)
    def expected_dist_type(self, device, dtype):
        dist_type = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 0, 0]],
                                 device=device, dtype=torch.int)
        return dist_type.repeat(3, 1)

    @with_seed(torch_seed=torch_seed, random_seed=random_seed)
    def test_triangle_distance_batch(self, pointclouds, vertices, faces,
                                     expected_dist, expected_face_idx, expected_dist_type):
        dist, face_idx, dist_type = trianglemesh.point_to_mesh_distance(
            pointclouds, vertices, faces)

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
