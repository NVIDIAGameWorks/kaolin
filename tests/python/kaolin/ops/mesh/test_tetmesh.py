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

from kaolin.ops.mesh import tetmesh


class TestTetMeshOps:

    def test_validate_tetrahedrons_wrong_ndim(self):
        wrong_ndim_tet = torch.randn(size=(2, 2))
        with pytest.raises(Exception):
            tetmesh._validate_tetrahedrons(wrong_ndim_tet)

    def test_validate_tetrahedrons_wrong_third_dimension(self):
        wrong_third_dim_tet = torch.randn(size=(2, 2, 3))
        with pytest.raises(Exception):
            tetmesh._validate_tetrahedrons(wrong_third_dim_tet)

    def test_validate_tetrahedrons_wrong_fourth_dimension(self):
        wrong_fourth_dim_tet = torch.randn(size=(2, 2, 4, 2))
        with pytest.raises(Exception):
            tetmesh._validate_tetrahedrons(wrong_fourth_dim_tet)

    def test_inverse_vertices_offset(self):
        tetrahedrons = torch.tensor([[[[-0.0500, 0.0000, 0.0500],
                                       [-0.0250, -0.0500, 0.0000],
                                       [0.0000, 0.0000, 0.0500],
                                       [0.5000, 0.5000, 0.4500]]]])
        oracle = torch.tensor([[[[0.0000, 20.0000, 0.0000],
                                 [79.9999, -149.9999, 10.0000],
                                 [-99.9999, 159.9998, -10.0000]]]])
        torch.allclose(tetmesh.inverse_vertices_offset(tetrahedrons), oracle)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
class TestSubdivideTetmesh:

    @pytest.fixture(autouse=True)
    def vertices_single_tet(self, device):
        return torch.tensor([[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=torch.float, device=device)

    @pytest.fixture(autouse=True)
    def faces_single_tet(self, device):
        return torch.tensor([[0, 1, 2, 3]], dtype=torch.long, device=device)

    @pytest.fixture(autouse=True)
    def expected_vertices_single_tet(self, device):
        return torch.tensor([[[0.0000, 0.0000, 0.0000],
                              [1.0000, 0.0000, 0.0000],
                              [0.0000, 1.0000, 0.0000],
                              [0.0000, 0.0000, 1.0000],
                              [0.5000, 0.0000, 0.0000],
                              [0.0000, 0.5000, 0.0000],
                              [0.0000, 0.0000, 0.5000],
                              [0.5000, 0.5000, 0.0000],
                              [0.5000, 0.0000, 0.5000],
                              [0.0000, 0.5000, 0.5000]]], dtype=torch.float, device=device)

    @pytest.fixture(autouse=True)
    def expected_faces_single_tet(self, device):
        return torch.tensor([[0, 4, 5, 6],
                             [1, 7, 4, 8],
                             [2, 5, 7, 9],
                             [3, 6, 9, 8],
                             [4, 5, 6, 8],
                             [4, 5, 8, 7],
                             [9, 5, 8, 6],
                             [9, 5, 7, 8]], dtype=torch.long, device=device)

    @pytest.fixture(autouse=True)
    def faces_two_tets(self, device):
        return torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long, device=device)

    @pytest.fixture(autouse=True)
    def expected_faces_two_tets(self, device):
        return torch.tensor([[0, 4, 5, 6],
                            [0, 4, 5, 6],
                            [1, 7, 4, 8],
                            [1, 7, 4, 8],
                            [2, 5, 7, 9],
                            [2, 5, 7, 9],
                            [3, 6, 9, 8],
                            [3, 6, 9, 8],
                            [4, 5, 6, 8],
                            [4, 5, 6, 8],
                            [4, 5, 8, 7],
                            [4, 5, 8, 7],
                            [9, 5, 8, 6],
                            [9, 5, 8, 6],
                            [9, 5, 7, 8],
                            [9, 5, 7, 8]], dtype=torch.long, device=device)

    @pytest.fixture(autouse=True)
    def features_single_tet(self, device):
        return torch.tensor([[[-1, 2], [-1, 4], [0.5, -2], [0.5, -3]]], dtype=torch.float, device=device)

    @pytest.fixture(autouse=True)
    def expected_features_single_tet(self, device):
        return torch.tensor([[[-1.0000, 2.0000],
                              [-1.0000, 4.0000],
                              [0.5000, -2.0000],
                              [0.5000, -3.0000],
                              [-1.0000, 3.0000],
                              [-0.2500, 0.0000],
                              [-0.2500, -0.5000],
                              [-0.2500, 1.0000],
                              [-0.2500, 0.5000],
                              [0.5000, -2.5000]]], dtype=torch.float, device=device)

    def test_subdivide_tetmesh_no_features(self, vertices_single_tet, faces_single_tet, expected_vertices_single_tet, expected_faces_single_tet):
        new_vertices, new_faces = tetmesh.subdivide_tetmesh(vertices_single_tet, faces_single_tet)
        assert torch.equal(new_vertices, expected_vertices_single_tet)
        assert torch.equal(new_faces, expected_faces_single_tet)

    def test_subdivide_tetmesh_no_features(self, vertices_single_tet, faces_single_tet, expected_vertices_single_tet, expected_faces_single_tet, features_single_tet, expected_features_single_tet):
        new_vertices, new_faces, new_features = tetmesh.subdivide_tetmesh(
            vertices_single_tet, faces_single_tet, features_single_tet)
        assert torch.equal(new_vertices, expected_vertices_single_tet)
        assert torch.equal(new_faces, expected_faces_single_tet)
        assert torch.equal(new_features, expected_features_single_tet)

    def test_subdivide_tetmesh_shared_verts(self, vertices_single_tet, faces_two_tets, expected_vertices_single_tet, expected_faces_two_tets, features_single_tet, expected_features_single_tet):
        # check if redundant vertices are generated
        new_vertices, new_faces, new_features = tetmesh.subdivide_tetmesh(
            vertices_single_tet, faces_two_tets, features_single_tet)
        assert torch.equal(new_vertices, expected_vertices_single_tet)
        assert torch.equal(new_faces, expected_faces_two_tets)
        assert torch.equal(new_features, expected_features_single_tet)
