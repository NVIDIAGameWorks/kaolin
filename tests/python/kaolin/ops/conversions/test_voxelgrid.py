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
import itertools

import torch
import random
from kaolin.ops.conversions import voxelgrid as vg

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
class TestVoxelgridsToCubicMeshes:

    @pytest.fixture(autouse=True)
    def voxelgrids(self, device):
        voxelgrids = torch.ones([4, 2, 2, 2], dtype=torch.bool, device=device)

        voxelgrids[0, :] = 0  # 1st case: all empty
        voxelgrids[2, :, 0, 0] = 0
        voxelgrids[3, 0, 0, :] = 0
        voxelgrids[3, 0, 1, 0] = 0
        voxelgrids[3, 1, 1, 1] = 0

        return voxelgrids

    @pytest.fixture(autouse=True)
    def expected_verts(self, device):
        expected_verts = []
        expected_verts.append(torch.zeros((0, 3), device=device))
        expected_verts.append(torch.tensor([[0, 0, 0],
                                            [0, 0, 1],
                                            [0, 0, 2],
                                            [0, 1, 0],
                                            [0, 1, 1],
                                            [0, 1, 2],
                                            [0, 2, 0],
                                            [0, 2, 1],
                                            [0, 2, 2],
                                            [1, 0, 0],
                                            [1, 0, 1],
                                            [1, 0, 2],
                                            [1, 1, 0],
                                            [1, 1, 2],
                                            [1, 2, 0],
                                            [1, 2, 1],
                                            [1, 2, 2],
                                            [2, 0, 0],
                                            [2, 0, 1],
                                            [2, 0, 2],
                                            [2, 1, 0],
                                            [2, 1, 1],
                                            [2, 1, 2],
                                            [2, 2, 0],
                                            [2, 2, 1],
                                            [2, 2, 2]],
                                           dtype=torch.float,
                                           device=device))

        expected_verts.append(torch.tensor([[0, 0, 1],
                                            [0, 0, 2],
                                            [0, 1, 0],
                                            [0, 1, 1],
                                            [0, 1, 2],
                                            [0, 2, 0],
                                            [0, 2, 1],
                                            [0, 2, 2],
                                            [1, 0, 1],
                                            [1, 0, 2],
                                            [1, 1, 0],
                                            [1, 1, 1],
                                            [1, 1, 2],
                                            [1, 2, 0],
                                            [1, 2, 1],
                                            [1, 2, 2],
                                            [2, 0, 1],
                                            [2, 0, 2],
                                            [2, 1, 0],
                                            [2, 1, 1],
                                            [2, 1, 2],
                                            [2, 2, 0],
                                            [2, 2, 1],
                                            [2, 2, 2]],
                                           dtype=torch.float,
                                           device=device))

        expected_verts.append(torch.tensor([[0, 1, 1],
                                            [0, 1, 2],
                                            [0, 2, 1],
                                            [0, 2, 2],
                                            [1, 0, 0],
                                            [1, 0, 1],
                                            [1, 0, 2],
                                            [1, 1, 0],
                                            [1, 1, 1],
                                            [1, 1, 2],
                                            [1, 2, 0],
                                            [1, 2, 1],
                                            [1, 2, 2],
                                            [2, 0, 0],
                                            [2, 0, 1],
                                            [2, 0, 2],
                                            [2, 1, 0],
                                            [2, 1, 1],
                                            [2, 1, 2],
                                            [2, 2, 0],
                                            [2, 2, 1]],
                                           dtype=torch.float,
                                           device=device))

        return expected_verts

    @pytest.fixture(autouse=True)
    def expected_faces(self, device):

        expected_faces = []
        expected_faces.append(torch.zeros(
            (0, 4), dtype=torch.long, device=device))

        # 2nd case: all ones

        expected_faces.append(torch.tensor([[0,  3,  4,  1],
                                            [1,  4,  5,  2],
                                            [3,  6,  7,  4],
                                            [4,  7,  8,  5],
                                            [18, 21, 20, 17],
                                            [19, 22, 21, 18],
                                            [21, 24, 23, 20],
                                            [22, 25, 24, 21],
                                            [0,  1, 10,  9],
                                            [1,  2, 11, 10],
                                            [14, 15,  7,  6],
                                            [15, 16,  8,  7],
                                            [9, 10, 18, 17],
                                            [10, 11, 19, 18],
                                            [23, 24, 15, 14],
                                            [24, 25, 16, 15],
                                            [0,  9, 12,  3],
                                            [5, 13, 11,  2],
                                            [3, 12, 14,  6],
                                            [8, 16, 13,  5],
                                            [9, 17, 20, 12],
                                            [13, 22, 19, 11],
                                            [12, 20, 23, 14],
                                            [16, 25, 22, 13]],
                                           dtype=torch.long,
                                           device=device))
        # 3rd case

        expected_faces.append(torch.tensor([[0,  3,  4,  1],
                                            [2,  5,  6,  3],
                                            [3,  6,  7,  4],
                                            [17, 20, 19, 16],
                                            [19, 22, 21, 18],
                                            [20, 23, 22, 19],
                                            [0,  1,  9,  8],
                                            [2,  3, 11, 10],
                                            [13, 14,  6,  5],
                                            [14, 15,  7,  6],
                                            [8,  9, 17, 16],
                                            [10, 11, 19, 18],
                                            [21, 22, 14, 13],
                                            [22, 23, 15, 14],
                                            [0,  8, 11,  3],
                                            [4, 12,  9,  1],
                                            [2, 10, 13,  5],
                                            [7, 15, 12,  4],
                                            [8, 16, 19, 11],
                                            [12, 20, 17,  9],
                                            [10, 18, 21, 13],
                                            [15, 23, 20, 12]],
                                           dtype=torch.long,
                                           device=device))
        # 4th case
        expected_faces.append(torch.tensor([[0,  2,  3,  1],
                                            [4,  7,  8,  5],
                                            [5,  8,  9,  6],
                                            [7, 10, 11,  8],
                                            [9, 12, 11,  8],
                                            [14, 17, 16, 13],
                                            [15, 18, 17, 14],
                                            [17, 20, 19, 16],
                                            [0,  1,  9,  8],
                                            [11, 12,  3,  2],
                                            [4,  5, 14, 13],
                                            [5,  6, 15, 14],
                                            [17, 18,  9,  8],
                                            [19, 20, 11, 10],
                                            [0,  8, 11,  2],
                                            [3, 12,  9,  1],
                                            [4, 13, 16,  7],
                                            [9, 18, 15,  6],
                                            [7, 16, 19, 10],
                                            [11, 20, 17,  8]],
                                           dtype=torch.long,
                                           device=device))

        return expected_faces

    def test_output_value_trimesh(self, device, voxelgrids, expected_verts, expected_faces):

        tri_verts, tri_faces = vg.voxelgrids_to_cubic_meshes(
            voxelgrids, is_trimesh=True)

        for i in range(0, 4):
            assert torch.equal(
                tri_verts[i], expected_verts[i])

            tri_faces_nx3 = tri_faces[i]
            n = tri_faces_nx3.shape[0]
            tri_faces_nx3 = torch.cat(
                [tri_faces_nx3[:n // 2, [0, 2]], tri_faces_nx3[n // 2:, [0, 2]]], 1)
            assert torch.equal(
                tri_faces_nx3, expected_faces[i])

    def test_output_value_quadmesh(self, device, voxelgrids, expected_verts, expected_faces):

        verts, faces = vg.voxelgrids_to_cubic_meshes(
            voxelgrids, is_trimesh=False)

        for i in range(0, 4):
            assert torch.equal(
                verts[i], expected_verts[i])
            assert torch.equal(
                faces[i], expected_faces[i])

# TODO: Need comprehensive tests for every variation of each unique cases
class TestMarchingCube:
    def test_voxelgrids_to_trianglemeshes_empty(self):
        voxelgrid = torch.tensor([[[0, 0], 
                                   [0, 0]], 
                                  [[0, 0], 
                                   [0, 0]]], device='cuda', dtype=torch.uint8)
        
        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.zeros((0, 3), dtype=torch.float, device='cuda:0')

        expected_faces = torch.zeros((0, 3), device='cuda', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

    def test_voxelgrids_to_trianglemeshes_0(self):
        voxelgrid = torch.tensor([[[1, 0], 
                                   [0, 0]], 
                                  [[0, 0], 
                                   [0, 0]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[0.5000, 1.0000, 1.0000],
                                          [1.0000, 0.5000, 1.0000],
                                          [1.0000, 1.0000, 0.5000],
                                          [1.5000, 1.0000, 1.0000],
                                          [1.0000, 1.5000, 1.0000],
                                          [1.0000, 1.0000, 1.5000]], device='cuda:0')
        
        expected_faces = torch.tensor([[2, 1, 0],
                                       [1, 2, 3],
                                       [2, 0, 4],
                                       [3, 2, 4],
                                       [1, 5, 0],
                                       [1, 3, 5],
                                       [0, 5, 4],
                                       [3, 4, 5]], device='cuda:0', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_1(self):
        voxelgrid = torch.tensor([[[1, 1], 
                                   [0, 0]], 
                                  [[0, 0], 
                                   [0, 0]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[0.5000, 1.0000, 1.0000],
                                          [1.0000, 1.0000, 0.5000],
                                          [1.0000, 0.5000, 1.0000],
                                          [2.0000, 0.5000, 1.0000],
                                          [2.0000, 1.0000, 0.5000],
                                          [2.5000, 1.0000, 1.0000],
                                          [1.0000, 1.5000, 1.0000],
                                          [2.0000, 1.5000, 1.0000],
                                          [1.0000, 1.0000, 1.5000],
                                          [2.0000, 1.0000, 1.5000]], device='cuda')
        
        expected_faces = torch.tensor([[1, 2, 0],
                                       [1, 4, 3],
                                       [2, 1, 3],
                                       [3, 4, 5],
                                       [1, 0, 6],
                                       [4, 1, 6],
                                       [7, 4, 6],
                                       [5, 4, 7],
                                       [2, 8, 0],
                                       [2, 3, 9],
                                       [8, 2, 9],
                                       [3, 5, 9],
                                       [0, 8, 6],
                                       [7, 6, 8],
                                       [9, 7, 8],
                                       [5, 7, 9]], device='cuda', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)
            
    def test_voxelgrids_to_trianglemeshes_2(self):
        voxelgrid = torch.tensor([[[1, 0], 
                                   [0, 0]], 
                                  [[0, 1], 
                                   [0, 0]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[0.5000, 1.0000, 1.0000],
                                          [1.0000, 0.5000, 1.0000],
                                          [1.0000, 1.0000, 0.5000],
                                          [1.5000, 1.0000, 1.0000],
                                          [1.0000, 1.5000, 1.0000],
                                          [1.5000, 1.0000, 2.0000],
                                          [1.0000, 1.0000, 1.5000],
                                          [2.0000, 0.5000, 2.0000],
                                          [2.0000, 1.0000, 1.5000],
                                          [2.5000, 1.0000, 2.0000],
                                          [2.0000, 1.5000, 2.0000],
                                          [2.0000, 1.0000, 2.5000]], device='cuda')
        
        expected_faces = torch.tensor([[ 2,  1,  0],
                                       [ 1,  2,  3],
                                       [ 2,  0,  4],
                                       [ 3,  2,  4],
                                       [ 1,  6,  0],
                                       [ 5,  6,  1],
                                       [ 5,  1,  7],
                                       [ 3,  8,  1],
                                       [ 8,  7,  1],
                                       [ 7,  8,  9],
                                       [ 0,  6,  4],
                                       [ 6,  5, 10],
                                       [ 6, 10,  4],
                                       [ 8,  3, 10],
                                       [ 3,  4, 10],
                                       [ 9,  8, 10],
                                       [ 7, 11,  5],
                                       [ 7,  9, 11],
                                       [ 5, 11, 10],
                                       [ 9, 10, 11]], device='cuda', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_3(self):
        voxelgrid = torch.tensor([[[0, 1], 
                                   [1, 1]], 
                                  [[0, 0], 
                                   [0, 0]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[1.5000, 1.0000, 1.0000],
                                          [2.0000, 0.5000, 1.0000],
                                          [2.0000, 1.0000, 0.5000],
                                          [2.5000, 1.0000, 1.0000],
                                          [0.5000, 2.0000, 1.0000],
                                          [1.0000, 1.5000, 1.0000],
                                          [1.0000, 2.0000, 0.5000],
                                          [2.5000, 2.0000, 1.0000],
                                          [2.0000, 2.0000, 0.5000],
                                          [1.0000, 2.5000, 1.0000],
                                          [2.0000, 2.5000, 1.0000],
                                          [2.0000, 1.0000, 1.5000],
                                          [1.0000, 2.0000, 1.5000],
                                          [2.0000, 2.0000, 1.5000]], device='cuda')
        
        expected_faces = torch.tensor([[ 2,  1,  0],
                                       [ 1,  2,  3],
                                       [ 6,  5,  4],
                                       [ 0,  5,  6],
                                       [ 0,  6,  2],
                                       [ 2,  6,  8],
                                       [ 7,  3,  2],
                                       [ 8,  7,  2],
                                       [ 6,  4,  9],
                                       [ 8,  6,  9],
                                       [10,  8,  9],
                                       [ 7,  8, 10],
                                       [ 1, 11,  0],
                                       [ 1,  3, 11],
                                       [ 5, 12,  4],
                                       [ 5,  0, 11],
                                       [ 5, 11, 12],
                                       [12, 11, 13],
                                       [ 3,  7, 13],
                                       [11,  3, 13],
                                       [ 4, 12,  9],
                                       [10,  9, 12],
                                       [13, 10, 12],
                                       [ 7, 10, 13]], device='cuda', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_4(self):
        voxelgrid = torch.tensor([[[1, 1], 
                                   [1, 1]], 
                                  [[0, 0], 
                                   [0, 0]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[0.5000, 1.0000, 1.0000],
                                          [1.0000, 1.0000, 0.5000],
                                          [1.0000, 0.5000, 1.0000],
                                          [2.0000, 0.5000, 1.0000],
                                          [2.0000, 1.0000, 0.5000],
                                          [2.5000, 1.0000, 1.0000],
                                          [0.5000, 2.0000, 1.0000],
                                          [1.0000, 2.0000, 0.5000],
                                          [2.5000, 2.0000, 1.0000],
                                          [2.0000, 2.0000, 0.5000],
                                          [1.0000, 2.5000, 1.0000],
                                          [2.0000, 2.5000, 1.0000],
                                          [1.0000, 1.0000, 1.5000],
                                          [2.0000, 1.0000, 1.5000],
                                          [1.0000, 2.0000, 1.5000],
                                          [2.0000, 2.0000, 1.5000]], device='cuda')
        
        expected_faces = torch.tensor([[ 1,  2,  0],
                                       [ 1,  4,  3],
                                       [ 2,  1,  3],
                                       [ 3,  4,  5],
                                       [ 7,  1,  0],
                                       [ 6,  7,  0],
                                       [ 4,  1,  9],
                                       [ 9,  1,  7],
                                       [ 8,  5,  4],
                                       [ 9,  8,  4],
                                       [ 7,  6, 10],
                                       [ 9,  7, 10],
                                       [11,  9, 10],
                                       [ 8,  9, 11],
                                       [ 2, 12,  0],
                                       [ 2,  3, 13],
                                       [12,  2, 13],
                                       [ 3,  5, 13],
                                       [12, 14,  6],
                                       [ 0, 12,  6],
                                       [13, 15, 12],
                                       [15, 14, 12],
                                       [ 5,  8, 15],
                                       [13,  5, 15],
                                       [ 6, 14, 10],
                                       [11, 10, 14],
                                       [15, 11, 14],
                                       [ 8, 11, 15]], device='cuda', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_5(self):
        voxelgrid = torch.tensor([[[0, 1], 
                                   [1, 1]], 
                                  [[1, 0], 
                                   [0, 0]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[1.5000, 1.0000, 1.0000],
                                          [2.0000, 0.5000, 1.0000],
                                          [2.0000, 1.0000, 0.5000],
                                          [2.5000, 1.0000, 1.0000],
                                          [0.5000, 2.0000, 1.0000],
                                          [1.0000, 1.5000, 1.0000],
                                          [1.0000, 2.0000, 0.5000],
                                          [2.5000, 2.0000, 1.0000],
                                          [2.0000, 2.0000, 0.5000],
                                          [1.0000, 2.5000, 1.0000],
                                          [2.0000, 2.5000, 1.0000],
                                          [0.5000, 1.0000, 2.0000],
                                          [1.0000, 1.0000, 1.5000],
                                          [1.0000, 0.5000, 2.0000],
                                          [1.5000, 1.0000, 2.0000],
                                          [2.0000, 1.0000, 1.5000],
                                          [1.0000, 1.5000, 2.0000],
                                          [1.0000, 2.0000, 1.5000],
                                          [2.0000, 2.0000, 1.5000],
                                          [1.0000, 1.0000, 2.5000]], device='cuda')
        
        expected_faces = torch.tensor([[ 2,  1,  0],
                                       [ 1,  2,  3],
                                       [ 6,  5,  4],
                                       [ 0,  5,  6],
                                       [ 0,  6,  2],
                                       [ 2,  6,  8],
                                       [ 7,  3,  2],
                                       [ 8,  7,  2],
                                       [ 6,  4,  9],
                                       [ 8,  6,  9],
                                       [10,  8,  9],
                                       [ 7,  8, 10],
                                       [12, 13, 11],
                                       [12,  0,  1],
                                       [12,  1, 13],
                                       [15, 14,  1],
                                       [14, 13,  1],
                                       [ 1,  3, 15],
                                       [16, 17,  4],
                                       [16,  4, 11],
                                       [ 5, 12,  4],
                                       [12, 11,  4],
                                       [ 0, 12,  5],
                                       [14, 15, 16],
                                       [15, 17, 16],
                                       [15, 18, 17],
                                       [ 3,  7, 18],
                                       [15,  3, 18],
                                       [ 4, 17,  9],
                                       [10,  9, 17],
                                       [18, 10, 17],
                                       [ 7, 10, 18],
                                       [13, 19, 11],
                                       [13, 14, 19],
                                       [11, 19, 16],
                                       [14, 16, 19]], device='cuda', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)
    
    def test_voxelgrids_to_trianglemeshes_6(self):
        voxelgrid = torch.tensor([[[1, 0], 
                                   [0, 1]], 
                                  [[0, 1], 
                                   [1, 0]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[0.5000, 1.0000, 1.0000],
                                          [1.0000, 0.5000, 1.0000],
                                          [1.0000, 1.0000, 0.5000],
                                          [1.5000, 1.0000, 1.0000],
                                          [1.0000, 1.5000, 1.0000],
                                          [1.5000, 2.0000, 1.0000],
                                          [2.0000, 1.5000, 1.0000],
                                          [2.0000, 2.0000, 0.5000],
                                          [2.5000, 2.0000, 1.0000],
                                          [2.0000, 2.5000, 1.0000],
                                          [1.5000, 1.0000, 2.0000],
                                          [1.0000, 1.0000, 1.5000],
                                          [2.0000, 0.5000, 2.0000],
                                          [2.0000, 1.0000, 1.5000],
                                          [2.5000, 1.0000, 2.0000],
                                          [0.5000, 2.0000, 2.0000],
                                          [1.0000, 1.5000, 2.0000],
                                          [1.0000, 2.0000, 1.5000],
                                          [1.5000, 2.0000, 2.0000],
                                          [2.0000, 2.0000, 1.5000],
                                          [2.0000, 1.5000, 2.0000],
                                          [1.0000, 2.5000, 2.0000],
                                          [2.0000, 1.0000, 2.5000],
                                          [1.0000, 2.0000, 2.5000]], device='cuda')
        
        expected_faces = torch.tensor([[ 2,  1,  0],
                                       [ 1,  2,  3],
                                       [ 2,  0,  4],
                                       [ 4,  5,  7],
                                       [ 4,  7,  2],
                                       [ 6,  3,  7],
                                       [ 3,  2,  7],
                                       [ 6,  7,  8],
                                       [ 7,  5,  9],
                                       [ 8,  7,  9],
                                       [ 1, 11,  0],
                                       [10, 11,  1],
                                       [10,  1, 12],
                                       [ 3, 13,  1],
                                       [13, 12,  1],
                                       [12, 13, 14],
                                       [17,  4,  0],
                                       [17,  0, 15],
                                       [11, 16,  0],
                                       [16, 15,  0],
                                       [ 3,  6, 13],
                                       [10, 16, 11],
                                       [ 5,  4, 17],
                                       [20, 19, 18],
                                       [19, 20, 14],
                                       [19, 14,  8],
                                       [13,  6, 14],
                                       [ 6,  8, 14],
                                       [17, 15, 21],
                                       [18, 19,  9],
                                       [18,  9, 21],
                                       [ 5, 17,  9],
                                       [17, 21,  9],
                                       [ 8,  9, 19],
                                       [12, 22, 10],
                                       [12, 14, 22],
                                       [16, 23, 15],
                                       [16, 10, 22],
                                       [16, 22, 23],
                                       [20, 18, 22],
                                       [18, 23, 22],
                                       [14, 20, 22],
                                       [15, 23, 21],
                                       [18, 21, 23]], device='cuda', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_7(self):
        voxelgrid = torch.tensor([[[1, 0], 
                                   [1, 1]], 
                                  [[0, 0], 
                                   [1, 0]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[0.5000, 1.0000, 1.0000],
                                          [1.0000, 0.5000, 1.0000],
                                          [1.0000, 1.0000, 0.5000],
                                          [1.5000, 1.0000, 1.0000],
                                          [0.5000, 2.0000, 1.0000],
                                          [1.0000, 2.0000, 0.5000],
                                          [2.0000, 1.5000, 1.0000],
                                          [2.0000, 2.0000, 0.5000],
                                          [2.5000, 2.0000, 1.0000],
                                          [1.0000, 2.5000, 1.0000],
                                          [2.0000, 2.5000, 1.0000],
                                          [1.0000, 1.0000, 1.5000],
                                          [0.5000, 2.0000, 2.0000],
                                          [1.5000, 2.0000, 2.0000],
                                          [1.0000, 1.5000, 2.0000],
                                          [2.0000, 2.0000, 1.5000],
                                          [1.0000, 2.5000, 2.0000],
                                          [1.0000, 2.0000, 2.5000]], device='cuda')
        
        expected_faces = torch.tensor([[ 2,  1,  0],
                                       [ 1,  2,  3],
                                       [ 5,  2,  0],
                                       [ 4,  5,  0],
                                       [ 6,  3,  2],
                                       [ 6,  2,  7],
                                       [ 7,  2,  5],
                                       [ 6,  7,  8],
                                       [ 5,  4,  9],
                                       [ 7,  5,  9],
                                       [10,  7,  9],
                                       [ 8,  7, 10],
                                       [ 1, 11,  0],
                                       [ 1,  3, 11],
                                       [11, 14, 12],
                                       [11, 12,  0],
                                       [ 0, 12,  4],
                                       [15, 13, 14],
                                       [ 6, 15, 14],
                                       [ 6, 14, 11],
                                       [ 6, 11,  3],
                                       [ 6,  8, 15],
                                       [ 4, 12, 16],
                                       [ 9,  4, 16],
                                       [13, 15, 10],
                                       [13, 10, 16],
                                       [16, 10,  9],
                                       [ 8, 10, 15],
                                       [14, 17, 12],
                                       [14, 13, 17],
                                       [12, 17, 16],
                                       [13, 16, 17]], device='cuda', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_8(self):
        voxelgrid = torch.tensor([[[0, 1], 
                                   [1, 1]], 
                                  [[0, 0], 
                                   [1, 0]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[1.5000, 1.0000, 1.0000],
                                          [2.0000, 0.5000, 1.0000],
                                          [2.0000, 1.0000, 0.5000],
                                          [2.5000, 1.0000, 1.0000],
                                          [0.5000, 2.0000, 1.0000],
                                          [1.0000, 1.5000, 1.0000],
                                          [1.0000, 2.0000, 0.5000],
                                          [2.5000, 2.0000, 1.0000],
                                          [2.0000, 2.0000, 0.5000],
                                          [1.0000, 2.5000, 1.0000],
                                          [2.0000, 2.5000, 1.0000],
                                          [2.0000, 1.0000, 1.5000],
                                          [0.5000, 2.0000, 2.0000],
                                          [1.0000, 1.5000, 2.0000],
                                          [1.5000, 2.0000, 2.0000],
                                          [2.0000, 2.0000, 1.5000],
                                          [1.0000, 2.5000, 2.0000],
                                          [1.0000, 2.0000, 2.5000]], device='cuda')
        
        expected_faces = torch.tensor([[ 2,  1,  0],
                                       [ 1,  2,  3],
                                       [ 6,  5,  4],
                                       [ 0,  5,  6],
                                       [ 0,  6,  2],
                                       [ 2,  6,  8],
                                       [ 7,  3,  2],
                                       [ 8,  7,  2],
                                       [ 6,  4,  9],
                                       [ 8,  6,  9],
                                       [10,  8,  9],
                                       [ 7,  8, 10],
                                       [ 1, 11,  0],
                                       [ 1,  3, 11],
                                       [ 5, 13, 12],
                                       [ 4,  5, 12],
                                       [ 0, 13,  5],
                                       [ 0, 15, 13],
                                       [ 0, 11, 15],
                                       [14, 13, 15],
                                       [ 3,  7, 15],
                                       [11,  3, 15],
                                       [ 4, 12, 16],
                                       [ 9,  4, 16],
                                       [14, 15, 10],
                                       [14, 10, 16],
                                       [16, 10,  9],
                                       [ 7, 10, 15],
                                       [13, 17, 12],
                                       [13, 14, 17],
                                       [12, 17, 16],
                                       [14, 16, 17]], device='cuda', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_9(self):
        voxelgrid = torch.tensor([[[1, 0], 
                                   [0, 0]], 
                                  [[0, 0], 
                                   [0, 1]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[0.5000, 1.0000, 1.0000],
                                          [1.0000, 0.5000, 1.0000],
                                          [1.0000, 1.0000, 0.5000],
                                          [1.5000, 1.0000, 1.0000],
                                          [1.0000, 1.5000, 1.0000],
                                          [1.0000, 1.0000, 1.5000],
                                          [1.5000, 2.0000, 2.0000],
                                          [2.0000, 1.5000, 2.0000],
                                          [2.0000, 2.0000, 1.5000],
                                          [2.5000, 2.0000, 2.0000],
                                          [2.0000, 2.5000, 2.0000],
                                          [2.0000, 2.0000, 2.5000]], device='cuda')
        
        expected_faces = torch.tensor([[ 2,  1,  0],
                                       [ 1,  2,  3],
                                       [ 2,  0,  4],
                                       [ 3,  2,  4],
                                       [ 1,  5,  0],
                                       [ 1,  3,  5],
                                       [ 0,  5,  4],
                                       [ 3,  4,  5],
                                       [ 7,  6,  8],
                                       [ 7,  8,  9],
                                       [ 8,  6, 10],
                                       [ 9,  8, 10],
                                       [ 7, 11,  6],
                                       [ 7,  9, 11],
                                       [ 6, 11, 10],
                                       [ 9, 10, 11]], device='cuda', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_10(self):
        voxelgrid = torch.tensor([[[1, 1], 
                                   [0, 0]], 
                                  [[0, 0], 
                                   [0, 1]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[0.5000, 1.0000, 1.0000],
                                          [1.0000, 1.0000, 0.5000],
                                          [1.0000, 0.5000, 1.0000],
                                          [2.0000, 0.5000, 1.0000],
                                          [2.0000, 1.0000, 0.5000],
                                          [2.5000, 1.0000, 1.0000],
                                          [1.0000, 1.5000, 1.0000],
                                          [2.0000, 1.5000, 1.0000],
                                          [1.0000, 1.0000, 1.5000],
                                          [2.0000, 1.0000, 1.5000],
                                          [1.5000, 2.0000, 2.0000],
                                          [2.0000, 1.5000, 2.0000],
                                          [2.5000, 2.0000, 2.0000],
                                          [2.0000, 2.0000, 1.5000],
                                          [2.0000, 2.5000, 2.0000],
                                          [2.0000, 2.0000, 2.5000]], device='cuda')
        
        expected_faces = torch.tensor([[ 1,  2,  0],
                                       [ 1,  4,  3],
                                       [ 2,  1,  3],
                                       [ 3,  4,  5],
                                       [ 1,  0,  6],
                                       [ 4,  1,  6],
                                       [ 7,  4,  6],
                                       [ 5,  4,  7],
                                       [ 2,  8,  0],
                                       [ 2,  3,  9],
                                       [ 8,  2,  9],
                                       [ 3,  5,  9],
                                       [ 0,  8,  6],
                                       [ 7,  6, 10],
                                       [ 7, 10, 13],
                                       [ 6,  8, 10],
                                       [11, 10,  9],
                                       [ 8,  9, 10],
                                       [11,  9,  5],
                                       [11,  5, 12],
                                       [ 7, 13,  5],
                                       [13, 12,  5],
                                       [13, 10, 14],
                                       [12, 13, 14],
                                       [11, 15, 10],
                                       [11, 12, 15],
                                       [10, 15, 14],
                                       [12, 14, 15]], device='cuda', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_11(self):
        voxelgrid = torch.tensor([[[0, 1], 
                                   [0, 0]], 
                                  [[1, 0], 
                                   [0, 1]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[1.5000, 1.0000, 1.0000],
                                          [2.0000, 0.5000, 1.0000],
                                          [2.0000, 1.0000, 0.5000],
                                          [2.5000, 1.0000, 1.0000],
                                          [2.0000, 1.5000, 1.0000],
                                          [0.5000, 1.0000, 2.0000],
                                          [1.0000, 1.0000, 1.5000],
                                          [1.0000, 0.5000, 2.0000],
                                          [1.5000, 1.0000, 2.0000],
                                          [2.0000, 1.0000, 1.5000],
                                          [1.5000, 2.0000, 2.0000],
                                          [1.0000, 1.5000, 2.0000],
                                          [2.0000, 1.5000, 2.0000],
                                          [2.5000, 2.0000, 2.0000],
                                          [2.0000, 2.0000, 1.5000],
                                          [2.0000, 2.5000, 2.0000],
                                          [1.0000, 1.0000, 2.5000],
                                          [2.0000, 2.0000, 2.5000]], device='cuda')
        
        expected_faces = torch.tensor([[ 2,  1,  0],
                                       [ 1,  2,  3],
                                       [ 2,  0,  4],
                                       [ 3,  2,  4],
                                       [ 6,  7,  5],
                                       [ 6,  0,  1],
                                       [ 6,  1,  7],
                                       [ 9,  8,  1],
                                       [ 8,  7,  1],
                                       [ 1,  3,  9],
                                       [ 6,  5, 11],
                                       [ 4, 10, 14],
                                       [ 4, 11, 10],
                                       [ 4,  0, 11],
                                       [ 6, 11,  0],
                                       [ 9, 12,  8],
                                       [12,  9,  3],
                                       [12,  3, 13],
                                       [ 4, 14,  3],
                                       [14, 13,  3],
                                       [14, 10, 15],
                                       [13, 14, 15],
                                       [ 7, 16,  5],
                                       [ 7,  8, 16],
                                       [ 5, 16, 11],
                                       [10, 11, 16],
                                       [10, 16, 17],
                                       [ 8, 12, 16],
                                       [12, 17, 16],
                                       [12, 13, 17],
                                       [10, 17, 15],
                                       [13, 15, 17]], device='cuda', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_12(self):
        voxelgrid = torch.tensor([[[1, 0], 
                                   [0, 1]], 
                                  [[1, 0], 
                                   [0, 1]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[0.5000, 1.0000, 1.0000],
                                          [1.0000, 0.5000, 1.0000],
                                          [1.0000, 1.0000, 0.5000],
                                          [1.5000, 1.0000, 1.0000],
                                          [1.0000, 1.5000, 1.0000],
                                          [1.5000, 2.0000, 1.0000],
                                          [2.0000, 1.5000, 1.0000],
                                          [2.0000, 2.0000, 0.5000],
                                          [2.5000, 2.0000, 1.0000],
                                          [2.0000, 2.5000, 1.0000],
                                          [0.5000, 1.0000, 2.0000],
                                          [1.0000, 0.5000, 2.0000],
                                          [1.5000, 1.0000, 2.0000],
                                          [1.5000, 2.0000, 2.0000],
                                          [1.0000, 1.5000, 2.0000],
                                          [2.0000, 1.5000, 2.0000],
                                          [2.5000, 2.0000, 2.0000],
                                          [2.0000, 2.5000, 2.0000],
                                          [1.0000, 1.0000, 2.5000],
                                          [2.0000, 2.0000, 2.5000]], device='cuda')
        
        expected_faces = torch.tensor([[ 2,  1,  0],
                                       [ 1,  2,  3],
                                       [ 2,  0,  4],
                                       [ 4,  5,  7],
                                       [ 4,  7,  2],
                                       [ 6,  3,  7],
                                       [ 3,  2,  7],
                                       [ 6,  7,  8],
                                       [ 7,  5,  9],
                                       [ 8,  7,  9],
                                       [ 1, 11, 10],
                                       [ 0,  1, 10],
                                       [11,  1,  3],
                                       [12, 11,  3],
                                       [ 0, 10, 14],
                                       [ 4,  0, 14],
                                       [ 4, 13,  5],
                                       [ 4, 14, 13],
                                       [ 6, 15,  3],
                                       [15, 12,  3],
                                       [15,  6,  8],
                                       [16, 15,  8],
                                       [ 5, 13, 17],
                                       [ 9,  5, 17],
                                       [16,  8,  9],
                                       [17, 16,  9],
                                       [11, 18, 10],
                                       [11, 12, 18],
                                       [10, 18, 14],
                                       [13, 14, 18],
                                       [13, 18, 19],
                                       [12, 15, 18],
                                       [15, 19, 18],
                                       [15, 16, 19],
                                       [13, 19, 17],
                                       [16, 17, 19]], device='cuda', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_13(self):
        voxelgrid = torch.tensor([[[1, 0], 
                                   [1, 1]], 
                                  [[0, 0], 
                                   [0, 1]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[0.5000, 1.0000, 1.0000],
                                          [1.0000, 0.5000, 1.0000],
                                          [1.0000, 1.0000, 0.5000],
                                          [1.5000, 1.0000, 1.0000],
                                          [0.5000, 2.0000, 1.0000],
                                          [1.0000, 2.0000, 0.5000],
                                          [2.0000, 1.5000, 1.0000],
                                          [2.0000, 2.0000, 0.5000],
                                          [2.5000, 2.0000, 1.0000],
                                          [1.0000, 2.5000, 1.0000],
                                          [2.0000, 2.5000, 1.0000],
                                          [1.0000, 1.0000, 1.5000],
                                          [1.0000, 2.0000, 1.5000],
                                          [1.5000, 2.0000, 2.0000],
                                          [2.0000, 1.5000, 2.0000],
                                          [2.5000, 2.0000, 2.0000],
                                          [2.0000, 2.5000, 2.0000],
                                          [2.0000, 2.0000, 2.5000]], device='cuda')
        
        expected_faces = torch.tensor([[ 2,  1,  0],
                                       [ 1,  2,  3],
                                       [ 5,  2,  0],
                                       [ 4,  5,  0],
                                       [ 6,  3,  2],
                                       [ 6,  2,  7],
                                       [ 7,  2,  5],
                                       [ 6,  7,  8],
                                       [ 5,  4,  9],
                                       [ 7,  5,  9],
                                       [10,  7,  9],
                                       [ 8,  7, 10],
                                       [ 1, 11,  0],
                                       [ 1,  3, 11],
                                       [11, 12,  4],
                                       [ 0, 11,  4],
                                       [ 3, 12, 11],
                                       [ 3, 14, 12],
                                       [ 3,  6, 14],
                                       [14, 13, 12],
                                       [14,  6,  8],
                                       [15, 14,  8],
                                       [ 4, 12,  9],
                                       [12, 13, 16],
                                       [12, 16,  9],
                                       [ 9, 16, 10],
                                       [15,  8, 10],
                                       [16, 15, 10],
                                       [14, 17, 13],
                                       [14, 15, 17],
                                       [13, 17, 16],
                                       [15, 16, 17]], device='cuda', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def _flip(self, voxelgrid, expected_vertices, dims):
        voxelgrid_length = voxelgrid.shape[0] + 1

        new_expected_vertices = expected_vertices.detach().clone()

        for d in dims:
            if d == 0:
                new_expected_vertices[:, 2] = voxelgrid_length - expected_vertices[:, 2]
            elif d == 1:
                new_expected_vertices[:, 1] = voxelgrid_length - expected_vertices[:, 1]
            elif d == 2:
                new_expected_vertices[:, 0] = voxelgrid_length - expected_vertices[:, 0]

        if len(dims) == 0:
            new_voxelgrid = voxelgrid.detach().clone()
        else:
            new_voxelgrid = voxelgrid.flip(dims)

        new_vertices, _ = vg.voxelgrids_to_trianglemeshes(new_voxelgrid.unsqueeze(0))

        return new_vertices, new_expected_vertices, new_voxelgrid
    
    def _rotate(self, voxelgrid, expected_vertices, dims):

        new_voxelgrid = voxelgrid.permute(dims)
        new_vertices, _ = vg.voxelgrids_to_trianglemeshes(new_voxelgrid.unsqueeze(0))

        new_expected_vertices = expected_vertices.detach().clone()

        if dims == (0, 2, 1) or dims == [0, 2, 1]:
            new_expected_vertices[:, 0], new_expected_vertices[:, 1] = expected_vertices[:, 1], expected_vertices[:, 0]
        elif dims == (1, 0, 2) or dims == [1, 0, 2]:
            new_expected_vertices[:, 1], new_expected_vertices[:, 2] = expected_vertices[:, 2], expected_vertices[:, 1]
        elif dims == (2, 1, 0) or dims == [2, 1, 0]:
            new_expected_vertices[:, 0], new_expected_vertices[:, 2] = expected_vertices[:, 2], expected_vertices[:, 0]
        
        return new_vertices, new_expected_vertices, new_voxelgrid

    def _all_variations_test(self, voxelgrid, expected_vertices):

        all_rotations = [[0, 1, 2], [1, 0, 2], [2, 1, 0]]
        all_flips = [[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]

        for rotate in all_rotations:
            vertices, expected_vertices, voxelgrid = self._rotate(voxelgrid, expected_vertices, rotate)
            for flip in all_flips:
                new_vertices, new_expected_vertices, _ = self._flip(voxelgrid, expected_vertices, flip)
                new_vertices = torch.sort(new_vertices[0], dim=0)[0]
                new_expected_vertices = torch.sort(new_expected_vertices, dim=0)[0]

                assert torch.equal(new_vertices, new_expected_vertices)
    
    def test_print_timeout(self):
        voxelgrid = torch.tensor([[[1, 0], 
                                   [0, 0]], 
                                  [[0, 0], 
                                   [0, 0]]], device='cuda', dtype=torch.uint8)
        

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        import time
        start_time = time.time()
        print(vertices)

        assert time.time() - start_time <= 10