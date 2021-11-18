# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
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
from kaolin.ops.conversions import tetmesh as tm


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
class TestMarchingTetrahedra:

    @pytest.fixture(autouse=True)
    def vertices(self, device):
        vertices = torch.tensor([[-1., -1., -1.],
                                 [1., -1., -1.],
                                 [-1., 1., -1.],
                                 [1., 1., -1.],
                                 [-1., -1., 1.],
                                 [1., -1., 1.],
                                 [-1., 1., 1.],
                                 [1., 1., 1.]],
                                dtype=torch.float,
                                device=device).unsqueeze(0).expand(4, -1, -1)
        return vertices

    @pytest.fixture(autouse=True)
    def tets(self, device):
        tets = torch.tensor([[0, 1, 3, 5],
                             [4, 5, 0, 6],
                             [0, 3, 2, 6],
                             [5, 3, 6, 7],
                             [0, 5, 3, 6]],
                            dtype=torch.long,
                            device=device)
        return tets

    @pytest.fixture(autouse=True)
    def sdf(self, device):
        sdf = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1],  # 1st case: empty
                            [1, 1, 1, 1, -1, 1, 1, 1],  # 2nd case: one triangle
                            [1, 1, 1, 1, -1, -1, 1, 1],  # 3rd case: multiple triangles
                            [1, 1, 1, 1, -0.5, -0.7, 1, 1]],  # 4th case: same topology as 3rd case but different zero-crossings

                           dtype=torch.float,
                           device=device)
        return sdf

    @pytest.fixture(autouse=True)
    def expected_verts(self, device):
        expected_verts = []
        expected_verts.append(torch.zeros((0, 3), device=device))
        expected_verts.append(torch.tensor([[-1., -1., 0.],
                                            [0., -1., 1.],
                                            [-1., 0., 1.]],
                                           dtype=torch.float,
                                           device=device))

        expected_verts.append(torch.tensor([[-1., -1., 0.],
                                            [0., -1., 0.],
                                            [1., -1., 0.],
                                            [1., 0., 0.],
                                            [-1., 0., 1.],
                                            [0., 0., 1.],
                                            [1., 0., 1.]],
                                           dtype=torch.float,
                                           device=device))

        expected_verts.append(torch.tensor([[-1.0000, -1.0000, 0.3333],
                                            [0.1765, -1.0000, 0.1765],
                                            [1.0000, -1.0000, 0.1765],
                                            [1.0000, -0.1765, 0.1765],
                                            [-1.0000, -0.3333, 1.0000],
                                            [0.1765, -0.1765, 1.0000],
                                            [1.0000, -0.1765, 1.0000]],
                                           dtype=torch.float,
                                           device=device))

        return expected_verts

    @pytest.fixture(autouse=True)
    def expected_faces(self, device):

        expected_faces = []

        expected_faces.append(torch.zeros(
            (0, 3), dtype=torch.long, device=device))

        expected_faces.append(torch.tensor([[2, 1, 0]],
                                           dtype=torch.long,
                                           device=device))

        expected_faces.append(torch.tensor([[2, 1, 3],
                                            [6, 3, 5],
                                            [3, 1, 5],
                                            [5, 0, 4],
                                            [5, 1, 0]],
                                           dtype=torch.long,
                                           device=device))

        expected_faces.append(torch.tensor([[2, 1, 3],
                                            [6, 3, 5],
                                            [3, 1, 5],
                                            [5, 0, 4],
                                            [5, 1, 0]],
                                           dtype=torch.long,
                                           device=device))

        return expected_faces

    @pytest.fixture(autouse=True)
    def expected_tet_idx(self, device):

        expected_tet_idx = []
        expected_tet_idx.append(torch.zeros(
            (0), dtype=torch.long, device=device))

        expected_tet_idx.append(torch.tensor([1],
                                             dtype=torch.long,
                                             device=device))

        expected_tet_idx.append(torch.tensor([0, 3, 4, 1, 1],
                                             dtype=torch.long,
                                             device=device))

        expected_tet_idx.append(torch.tensor([0, 3, 4, 1, 1],
                                             dtype=torch.long,
                                             device=device))

        return expected_tet_idx

    def test_output_value(self, vertices, tets, sdf, expected_verts, expected_faces, expected_tet_idx):

        verts_list, faces_list, tet_idx_list = tm.marching_tetrahedra(vertices, tets, sdf, True)
        for i in range(0, 4):
            assert torch.allclose(
                verts_list[i], expected_verts[i], atol=1e-4)
            assert torch.equal(
                faces_list[i], expected_faces[i])
            assert torch.equal(
                tet_idx_list[i], expected_tet_idx[i])
