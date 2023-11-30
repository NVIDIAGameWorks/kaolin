# Copyright (c) 2023 YOUR_ORGANIZATION_NAME.
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

import pytest

import torch
from kaolin.non_commercial.flexicubes.flexicubes import FlexiCubes  # Please adjust this according to your project structure.


def cube_sdf(x_nx3):
    sdf_values = 0.5 - torch.abs(x_nx3)
    sdf_values = torch.clamp(sdf_values, min=0.0)
    sdf_values = sdf_values[:, 0] * sdf_values[:, 1] * sdf_values[:, 2]
    sdf_values = -1.0 * sdf_values

    return sdf_values.view(-1, 1)


def cube_sdf_gradient(x_nx3):
    gradients = []
    for i in range(x_nx3.shape[0]):
        x, y, z = x_nx3[i]
        grad_x, grad_y, grad_z = 0, 0, 0

        max_val = max(abs(x) - 0.5, abs(y) - 0.5, abs(z) - 0.5)

        if max_val == abs(x) - 0.5:
            grad_x = 1.0 if x > 0 else -1.0
        if max_val == abs(y) - 0.5:
            grad_y = 1.0 if y > 0 else -1.0
        if max_val == abs(z) - 0.5:
            grad_z = 1.0 if z > 0 else -1.0

        gradients.append(torch.tensor([grad_x, grad_y, grad_z]))

    return torch.stack(gradients).to(x_nx3.device)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
class TestFlexiCubes:

    @pytest.fixture(autouse=True)
    def input_data(self, device):
        sdf_n = torch.tensor([0.6660, 0.5500, 0.5071, 0.5500, 0.6660, 0.5500, 0.4124, 0.3590,
                              0.4124, 0.5500, 0.5071, 0.3590, 0.3000, 0.3590, 0.5071, 0.5500,
                              0.4124, 0.3590, 0.4124, 0.5500, 0.6660, 0.5500, 0.5071, 0.5500,
                              0.6660, 0.5500, 0.4124, 0.3590, 0.4124, 0.5500, 0.4124, 0.2330,
                              0.1536, 0.2330, 0.4124, 0.3590, 0.1536, 0.0500, 0.1536, 0.3590,
                              0.4124, 0.2330, 0.1536, 0.2330, 0.4124, 0.5500, 0.4124, 0.3590,
                              0.4124, 0.5500, 0.5071, 0.3590, 0.3000, 0.3590, 0.5071, 0.3590,
                              0.1536, 0.0500, 0.1536, 0.3590, 0.3000, 0.0500, -0.2000, 0.0500,
                              0.3000, 0.3590, 0.1536, 0.0500, 0.1536, 0.3590, 0.5071, 0.3590,
                              0.3000, 0.3590, 0.5071, 0.5500, 0.4124, 0.3590, 0.4124, 0.5500,
                              0.4124, 0.2330, 0.1536, 0.2330, 0.4124, 0.3590, 0.1536, 0.0500,
                              0.1536, 0.3590, 0.4124, 0.2330, 0.1536, 0.2330, 0.4124, 0.5500,
                              0.4124, 0.3590, 0.4124, 0.5500, 0.6660, 0.5500, 0.5071, 0.5500,
                              0.6660, 0.5500, 0.4124, 0.3590, 0.4124, 0.5500, 0.5071, 0.3590,
                              0.3000, 0.3590, 0.5071, 0.5500, 0.4124, 0.3590, 0.4124, 0.5500,
                              0.6660, 0.5500, 0.5071, 0.5500, 0.6660],
                             dtype=torch.float,
                             device=device)
        return sdf_n

    @pytest.fixture(autouse=True)
    def expected_trimesh_output(self, device):
        expected_vertices = torch.tensor([[-0.0667, -0.0667, -0.0667],
                                          [-0.0667, -0.0667, 0.0667],
                                          [-0.0667, 0.0667, -0.0667],
                                          [-0.0667, 0.0667, 0.0667],
                                          [0.0667, -0.0667, -0.0667],
                                          [0.0667, -0.0667, 0.0667],
                                          [0.0667, 0.0667, -0.0667],
                                          [0.0667, 0.0667, 0.0667]],
                                         dtype=torch.float,
                                         device=device)

        expected_faces = torch.tensor([[0, 1, 2],
                                       [2, 1, 3],
                                       [0, 2, 4],
                                       [4, 2, 6],
                                       [2, 3, 6],
                                       [6, 3, 7],
                                       [4, 5, 0],
                                       [0, 5, 1],
                                       [5, 7, 1],
                                       [1, 7, 3],
                                       [6, 7, 4],
                                       [4, 7, 5]],
                                      dtype=torch.long,
                                      device=device)
        return expected_vertices, expected_faces

    @pytest.fixture(autouse=True)
    def expected_tetmesh_output(self, device):
        expected_vertices = torch.tensor([[-0.0667, -0.0667, -0.0667],
                                          [-0.0667, -0.0667, 0.0667],
                                          [-0.0667, 0.0667, -0.0667],
                                          [-0.0667, 0.0667, 0.0667],
                                          [0.0667, -0.0667, -0.0667],
                                          [0.0667, -0.0667, 0.0667],
                                          [0.0667, 0.0667, -0.0667],
                                          [0.0667, 0.0667, 0.0667],
                                          [0.0000, 0.0000, 0.0000]],
                                         dtype=torch.float,
                                         device=device)

        expected_tets = torch.tensor([[0, 1, 2, 8],
                                      [2, 1, 3, 8],
                                      [0, 2, 4, 8],
                                      [4, 2, 6, 8],
                                      [2, 3, 6, 8],
                                      [6, 3, 7, 8],
                                      [4, 5, 0, 8],
                                      [0, 5, 1, 8],
                                      [5, 7, 1, 8],
                                      [1, 7, 3, 8],
                                      [6, 7, 4, 8],
                                      [4, 7, 5, 8]],
                                     dtype=torch.long,
                                     device=device)
        return expected_vertices, expected_tets

    @pytest.fixture(autouse=True)
    def expected_grid(self, device):
        expected_x_nx3 = torch.tensor([[-0.5000, -0.5000, -0.5000],
                                       [-0.5000, -0.5000, 0.0000],
                                       [-0.5000, -0.5000, 0.5000],
                                       [-0.5000, 0.0000, -0.5000],
                                       [-0.5000, 0.0000, 0.0000],
                                       [-0.5000, 0.0000, 0.5000],
                                       [-0.5000, 0.5000, -0.5000],
                                       [-0.5000, 0.5000, 0.0000],
                                       [-0.5000, 0.5000, 0.5000],
                                       [0.0000, -0.5000, -0.5000],
                                       [0.0000, -0.5000, 0.0000],
                                       [0.0000, -0.5000, 0.5000],
                                       [0.0000, 0.0000, -0.5000],
                                       [0.0000, 0.0000, 0.0000],
                                       [0.0000, 0.0000, 0.5000],
                                       [0.0000, 0.5000, -0.5000],
                                       [0.0000, 0.5000, 0.0000],
                                       [0.0000, 0.5000, 0.5000],
                                       [0.5000, -0.5000, -0.5000],
                                       [0.5000, -0.5000, 0.0000],
                                       [0.5000, -0.5000, 0.5000],
                                       [0.5000, 0.0000, -0.5000],
                                       [0.5000, 0.0000, 0.0000],
                                       [0.5000, 0.0000, 0.5000],
                                       [0.5000, 0.5000, -0.5000],
                                       [0.5000, 0.5000, 0.0000],
                                       [0.5000, 0.5000, 0.5000]],
                                      dtype=torch.float,
                                      device=device)

        expected_cube_fx8 = torch.tensor([[0, 9, 3, 12, 1, 10, 4, 13],
                                          [1, 10, 4, 13, 2, 11, 5, 14],
                                          [3, 12, 6, 15, 4, 13, 7, 16],
                                          [4, 13, 7, 16, 5, 14, 8, 17],
                                          [9, 18, 12, 21, 10, 19, 13, 22],
                                          [10, 19, 13, 22, 11, 20, 14, 23],
                                          [12, 21, 15, 24, 13, 22, 16, 25],
                                          [13, 22, 16, 25, 14, 23, 17, 26]],
                                         dtype=torch.long,
                                         device=device)
        return expected_x_nx3, expected_cube_fx8

    @pytest.fixture(autouse=True)
    def expected_qef_output(self, device):
        expected_vertices = torch.tensor([[-0.5, -0.5, -0.5],
                                          [-0.5, -0.5, 0.0],
                                          [-0.5, -0.5, 0.5],
                                          [-0.5, 0.0, -0.5],
                                          [-0.5, 0.0, 0.0],
                                          [-0.5, 0.0, 0.5],
                                          [-0.5, 0.5, -0.5],
                                          [-0.5, 0.5, 0.0],
                                          [-0.5, 0.5, 0.5],
                                          [0.0, -0.5, -0.5],
                                          [0.0, -0.5, 0.0],
                                          [0.0, -0.5, 0.5],
                                          [0.0, 0.0, -0.5],
                                          [0.0, 0.0, 0.5],
                                          [0.0, 0.5, -0.5],
                                          [0.0, 0.5, 0.0],
                                          [0.0, 0.5, 0.5],
                                          [0.5, -0.5, -0.5],
                                          [0.5, -0.5, 0.0],
                                          [0.5, -0.5, 0.5],
                                          [0.5, 0.0, -0.5],
                                          [0.5, 0.0, 0.0],
                                          [0.5, 0.0, 0.5],
                                          [0.5, 0.5, -0.5],
                                          [0.5, 0.5, 0.0],
                                          [0.5, 0.5, 0.5]],
                                         dtype=torch.float,
                                         device=device)

        expected_faces_1 = torch.tensor([[3, 4, 0],
                                         [0, 4, 1],
                                         [4, 5, 1],
                                         [1, 5, 2],
                                         [6, 7, 3],
                                         [3, 7, 4],
                                         [7, 8, 5],
                                         [7, 5, 4],
                                         [9, 12, 0],
                                         [0, 12, 3],
                                         [9, 10, 0],
                                         [0, 10, 1],
                                         [10, 11, 1],
                                         [1, 11, 2],
                                         [11, 13, 2],
                                         [2, 13, 5],
                                         [12, 14, 3],
                                         [3, 14, 6],
                                         [13, 16, 8],
                                         [13, 8, 5],
                                         [14, 15, 6],
                                         [6, 15, 7],
                                         [15, 16, 7],
                                         [7, 16, 8],
                                         [17, 20, 12],
                                         [17, 12, 9],
                                         [17, 18, 10],
                                         [17, 10, 9],
                                         [20, 21, 17],
                                         [17, 21, 18],
                                         [18, 19, 10],
                                         [10, 19, 11],
                                         [19, 22, 13],
                                         [19, 13, 11],
                                         [21, 22, 18],
                                         [18, 22, 19],
                                         [20, 23, 12],
                                         [12, 23, 14],
                                         [23, 24, 20],
                                         [20, 24, 21],
                                         [22, 25, 13],
                                         [13, 25, 16],
                                         [24, 25, 22],
                                         [24, 22, 21],
                                         [23, 24, 15],
                                         [23, 15, 14],
                                         [24, 25, 16],
                                         [24, 16, 15]],
                                        dtype=torch.long,
                                        device=device)
        expected_faces_2 = torch.tensor([[3, 4, 0],
                                         [0, 4, 1],
                                         [4, 5, 1],
                                         [1, 5, 2],
                                         [6, 7, 3],
                                         [3, 7, 4],
                                         [7, 8, 5],
                                         [7, 5, 4],
                                         [9, 12, 0],
                                         [0, 12, 3],
                                         [9, 10, 0],
                                         [0, 10, 1],
                                         [10, 11, 1],
                                         [1, 11, 2],
                                         [11, 13, 2],
                                         [2, 13, 5],
                                         [12, 14, 3],
                                         [3, 14, 6],
                                         [13, 16, 5],
                                         [5, 16, 8],
                                         [14, 15, 6],
                                         [6, 15, 7],
                                         [15, 16, 8],
                                         [15, 8, 7],
                                         [17, 20, 9],
                                         [9, 20, 12],
                                         [17, 18, 9],
                                         [9, 18, 10],
                                         [20, 21, 17],
                                         [17, 21, 18],
                                         [18, 19, 10],
                                         [10, 19, 11],
                                         [19, 22, 13],
                                         [19, 13, 11],
                                         [21, 22, 18],
                                         [18, 22, 19],
                                         [20, 23, 12],
                                         [12, 23, 14],
                                         [23, 24, 20],
                                         [20, 24, 21],
                                         [22, 25, 13],
                                         [13, 25, 16],
                                         [24, 25, 22],
                                         [24, 22, 21],
                                         [23, 24, 15],
                                         [23, 15, 14],
                                         [24, 25, 15],
                                         [15, 25, 16]],
                                        dtype=torch.long,
                                        device=device)
        return expected_vertices, expected_faces_1, expected_faces_2

    def test_grid_construction(self, expected_grid, device):
        fc = FlexiCubes(device)
        x_nx3, cube_fx8 = fc.construct_voxel_grid(2)
        assert torch.allclose(x_nx3, expected_grid[0], atol=1e-4)
        assert torch.equal(cube_fx8, expected_grid[1])

    def test_trimesh_extraction(self, input_data, expected_trimesh_output, device):
        fc = FlexiCubes(device)
        x_nx3, cube_fx8 = fc.construct_voxel_grid(4)
        output = fc(x_nx3, input_data, cube_fx8, 4)

        assert torch.allclose(output[0], expected_trimesh_output[0], atol=1e-4)
        assert torch.equal(output[1], expected_trimesh_output[1])

    def test_tetmesh_extraction(self, input_data, expected_tetmesh_output, device):
        fc = FlexiCubes(device)
        x_nx3, cube_fx8 = fc.construct_voxel_grid(4)
        output = fc(x_nx3, input_data, cube_fx8, 4, output_tetmesh=True)

        assert torch.allclose(output[0], expected_tetmesh_output[0], atol=1e-4)
        assert torch.equal(output[1], expected_tetmesh_output[1])

    def test_qef_extraction_grad_func(self, expected_qef_output, device):
        fc = FlexiCubes(device)
        x_nx3, cube_fx8 = fc.construct_voxel_grid(3)
        sdf_n = cube_sdf(x_nx3)
        output = fc(x_nx3, sdf_n, cube_fx8, 3, grad_func=cube_sdf_gradient)

        assert torch.allclose(output[0], expected_qef_output[0], atol=1e-4)
        # in this example, both triangulations are correct
        assert torch.equal(output[1], expected_qef_output[1]) or torch.equal(output[1], expected_qef_output[2])
