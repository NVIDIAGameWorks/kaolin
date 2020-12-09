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
class Test_UnbatchedTriangleDistance:

    @pytest.fixture(autouse=True)
    def get_tol(self, device, dtype):
        if dtype == torch.float:
            return 1e-5, 1e-4
        elif dtype == torch.double:
            return 1e-6, 1e-5

    @pytest.fixture(autouse=True)
    @with_seed(torch_seed=torch_seed, random_seed=random_seed)
    def input_random_data(self, dtype, device):
        V = 100
        F = 50
        P = 80
        vertices = torch.randint(0, 10, (V, 3), device=device, dtype=dtype)
        vertices = torch.unique(vertices, dim=0)
        vertices.requires_grad = True

        V = vertices.shape[0]
        faces = []
        for i in range(F):
            faces.append(random.sample(range(V), 3))

        faces = torch.tensor(faces, device='cuda', dtype=torch.long)

        points =  torch.randint(0, 10, (P, 3), device=device, dtype=dtype)

        return points, vertices, faces

    @pytest.fixture(autouse=True)
    @with_seed(torch_seed=torch_seed, random_seed=random_seed)
    def input_double_data(self, device):
        """
        We choose the input data on purpose, so that when grad check offset the input points a little bit,
        it won't change the vertices it corresponds to. If the gradcheck offset changes vertices that this point
        corresponds to, the gradcheck will fail.
        """
        V = 200
        F = 200
        vertices = torch.randint(0, 20, (V, 3), device='cuda', dtype=torch.double)
        vertices = torch.unique(vertices, dim=0)
        vertices.requires_grad = True

        V = vertices.shape[0]
        faces = []
        for i in range(F):
            faces.append(random.sample(range(V), 3))

        faces = torch.tensor(faces, device='cuda', dtype=torch.long)

        v1 = torch.index_select(vertices, 0, faces[:, 0])
        v2 = torch.index_select(vertices, 0, faces[:, 1])
        v3 = torch.index_select(vertices, 0, faces[:, 2])

        v21 = v2 - v1  # (F, 3)
        v32 = v3 - v2
        v13 = v1 - v3

        all_edges = torch.stack((v21, v32, v13))

        all_vertices_by_faces = torch.stack((v1, v2, v3))
        # Select point that is closer to the edge
        random_split = torch.rand((F, 3), device='cuda')
        random_split = torch.clamp(random_split, 0.3, 0.7)

        cases = torch.randint(0, 3, size=(F, 1), dtype=torch.long, device='cuda')
        order = torch.arange(0, F, dtype=torch.long, device='cuda').view(F, 1)

        cases_idx = torch.cat((cases, order), dim=-1)

        noise = (-1 - 1) * torch.rand((F, 3), device='cuda') + 1

        selected_points = all_edges[cases[:, 0], cases_idx[:, 1]] * random_split + all_vertices_by_faces[cases[:, 0], cases_idx[:, 1]] + noise
        selected_points = selected_points.detach()
        selected_points.requires_grad=True

        return selected_points, vertices, faces

    @pytest.fixture(autouse=True)# Use for choose seed, will remove once ready to merge
    @with_seed(torch_seed=torch_seed, random_seed=random_seed)
    def target_grad_double(self, input_double_data):
        # if test_gradcheck passed the gradient using torch.double inputs is trustable
        points, vertices, faces = input_double_data
    
        points = points.detach()
        points.requires_grad = True

        v1 = torch.index_select(vertices, 0, faces[:, 0])
        v2 = torch.index_select(vertices, 0, faces[:, 1])
        v3 = torch.index_select(vertices, 0, faces[:, 2])

        v1 = v1.detach()
        v2 = v2.detach()
        v3 = v3.detach()

        v1.requires_grad = True
        v2.requires_grad = True
        v3.requires_grad = True

        outputs = torch.sum(trianglemesh._UnbatchedTriangleDistance.apply(points, v1, v2, v3)[0])
        outputs.backward()
        return points.grad.clone(), v1.grad.clone(), v2.grad.clone(), v3.grad.clone()

    @with_seed(torch_seed=torch_seed, random_seed=random_seed)
    def test_random_data(self, device, dtype, input_random_data, get_tol):
        points, vertices, faces = input_random_data
        atol, rtol = get_tol

        v1 = torch.index_select(vertices, 0, faces[:, 0])
        v2 = torch.index_select(vertices, 0, faces[:, 1])
        v3 = torch.index_select(vertices, 0, faces[:, 2])

        output_dist, output_idx, output_dist_type = \
            trianglemesh._UnbatchedTriangleDistance.apply(points, v1, v2, v3)
        expected_dist, expected_idx, expected_dist_type = \
            trianglemesh._point_to_mesh_distance_cpu(points, vertices, faces)

        assert torch.allclose(output_dist, expected_dist.unsqueeze(0), atol=atol, rtol=rtol)

        # Make sure every distance type is tested.
        for i in range(4):
            assert i in output_dist_type

    @with_seed(torch_seed=torch_seed, random_seed=random_seed)
    def test_grad_check(self, device, dtype, input_double_data):
        if dtype != torch.double:
            pytest.skip("Gradient check only works in double.")

        points, vertices, faces = input_double_data

        v1 = torch.index_select(vertices, 0, faces[:, 0])
        v2 = torch.index_select(vertices, 0, faces[:, 1])
        v3 = torch.index_select(vertices, 0, faces[:, 2])

        v1 = v1.detach()
        v2 = v2.detach()
        v3 = v3.detach()

        v1.requires_grad = True
        v2.requires_grad = True
        v3.requires_grad = True

        grad_result = torch.autograd.gradcheck(trianglemesh._UnbatchedTriangleDistance.apply,
                                               (points, v1, v2, v3), eps=1e-8, atol=1e-6)

        assert grad_result

    @with_seed(torch_seed=torch_seed, random_seed=random_seed)
    def test_grad_check_other_type(self, device, dtype, input_double_data, target_grad_double):
        if dtype == torch.double:
            pytest.skip("Gradient check for double already tested.")
        
        points, vertices, faces = input_double_data
        points = points.to(dtype).detach()
        vertices = vertices.to(dtype).detach()
        faces = faces.detach()

        points.requires_grad = True
        
        v1 = torch.index_select(vertices, 0, faces[:, 0])
        v2 = torch.index_select(vertices, 0, faces[:, 1])
        v3 = torch.index_select(vertices, 0, faces[:, 2])

        v1 = v1.detach()
        v2 = v2.detach()
        v3 = v3.detach()

        v1.requires_grad = True
        v2.requires_grad = True
        v3.requires_grad = True

        output = trianglemesh._UnbatchedTriangleDistance.apply(points, v1, v2, v3)[0]
        torch.sum(output).backward()
        target_grad_points, target_grad_v1, target_grad_v2, target_grad_v3 = target_grad_double

        target_grad_points = target_grad_points.to(dtype)
        target_grad_v1 = target_grad_v1.to(dtype)
        target_grad_v2 = target_grad_v2.to(dtype)
        target_grad_v3 = target_grad_v3.to(dtype)

        assert torch.allclose(target_grad_points, points.grad, rtol=1e-2, atol=5e-2)

@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
class TestTriangleDistance:

    @pytest.fixture(autouse=True)
    def get_tol(self, device, dtype):
        if dtype == torch.float:
            return 1e-5, 1e-4
        elif dtype == torch.double:
            return 1e-6, 1e-5

    @pytest.fixture(autouse=True)
    @with_seed(torch_seed=torch_seed, random_seed=random_seed)
    def input_random_data(self, dtype, device):
        V = 100
        F = 50
        P = 80
        vertices = torch.randint(0, 10, (V, 3), device=device, dtype=dtype)
        vertices = torch.unique(vertices, dim=0)
        vertices.requires_grad = True

        V = vertices.shape[0]
        faces = []
        for i in range(F):
            faces.append(random.sample(range(V), 3))

        faces = torch.tensor(faces, device='cuda', dtype=torch.long)

        points =  torch.randint(0, 10, (P, 3), device=device, dtype=dtype)

        return points, vertices, faces

    @with_seed(torch_seed=torch_seed, random_seed=random_seed)
    def test_triangle_distance_batch(self, device, dtype):
        vertices = torch.tensor([[[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]],
                                 
                                 [[0, 0, 0],
                                  [0, 1, 1],
                                  [1, 0, 1]]], device=device, dtype=dtype)
            
        faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        point1 = torch.tensor([[[2, 0.5, 0]],
                               [[0.5, 0.5, 0.5]]], device=device, dtype=dtype)
        
        dist, idx, dist_type = trianglemesh.point_to_mesh_distance(point1, vertices, faces)

        # Distance is not square rooted.
        expected_dist = torch.tensor([[4], [0.0833]], dtype=dtype, device=device)
        expected_idx = torch.tensor([[0], [0]], dtype=torch.long, device=device)
        # closest to the edge 0-1
        expected_dist_type = torch.tensor([[0], [3]], dtype=torch.int, device=device)

        assert torch.allclose(dist, expected_dist, atol=1e-4, rtol=1e-4)
        assert torch.equal(idx, expected_idx)
        assert torch.equal(dist_type, expected_dist_type)

    @with_seed(torch_seed=torch_seed, random_seed=random_seed)
    def test_random_data(self, device, dtype, input_random_data, get_tol):
        points, vertices, faces = input_random_data
        atol, rtol = get_tol

        output_dist, output_idx, output_dist_type = \
            trianglemesh.point_to_mesh_distance(points.unsqueeze(0), vertices.unsqueeze(0), faces)
        expected_dist, expected_idx, expected_dist_type = \
            trianglemesh._point_to_mesh_distance_cpu(points, vertices, faces)

        assert torch.allclose(output_dist, expected_dist.unsqueeze(0), atol=atol, rtol=rtol)

        # Make sure every distance type is tested.
        for i in range(4):
            assert i in output_dist_type

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
