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

        expected_vertices = torch.tensor([[1.0000, 1.0000, 0.5000],
                                          [1.0000, 0.5000, 1.0000],
                                          [0.5000, 1.0000, 1.0000],
                                          [1.0000, 1.0000, 1.5000],
                                          [1.0000, 1.5000, 1.0000],
                                          [1.5000, 1.0000, 1.0000]], device='cuda:0')
    
        expected_faces = torch.tensor([[0, 1, 2],
                                       [3, 2, 1],
                                       [4, 0, 2],
                                       [4, 2, 3],
                                       [0, 5, 1],
                                       [5, 3, 1],
                                       [4, 5, 0],
                                       [5, 4, 3]], device='cuda:0', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_1(self):
        voxelgrid = torch.tensor([[[1, 1], 
                                   [0, 0]], 
                                  [[0, 0], 
                                   [0, 0]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[1.0000, 1.0000, 0.5000],
                                          [0.5000, 1.0000, 1.0000],
                                          [1.0000, 0.5000, 1.0000],
                                          [1.0000, 0.5000, 2.0000],
                                          [0.5000, 1.0000, 2.0000],
                                          [1.0000, 1.0000, 2.5000],
                                          [1.0000, 1.5000, 1.0000],
                                          [1.0000, 1.5000, 2.0000],
                                          [1.5000, 1.0000, 1.0000],
                                          [1.5000, 1.0000, 2.0000]], device='cuda:0')
        
        expected_faces = torch.tensor([[0, 2, 1],
                                       [3, 4, 1],
                                       [3, 1, 2],
                                       [5, 4, 3],
                                       [6, 0, 1],
                                       [6, 1, 4],
                                       [6, 4, 7],
                                       [7, 4, 5],
                                       [0, 8, 2],
                                       [9, 3, 2],
                                       [9, 2, 8],
                                       [9, 5, 3],
                                       [6, 8, 0],
                                       [8, 6, 7],
                                       [8, 7, 9],
                                       [9, 7, 5]], device='cuda:0', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)
            
    def test_voxelgrids_to_trianglemeshes_2(self):
        voxelgrid = torch.tensor([[[1, 0], 
                                   [0, 0]], 
                                  [[0, 1], 
                                   [0, 0]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[1.0000, 1.0000, 0.5000],
                                          [1.0000, 0.5000, 1.0000],
                                          [0.5000, 1.0000, 1.0000],
                                          [1.0000, 1.0000, 1.5000],
                                          [1.0000, 1.5000, 1.0000],
                                          [2.0000, 1.0000, 1.5000],
                                          [1.5000, 1.0000, 1.0000],
                                          [2.0000, 0.5000, 2.0000],
                                          [1.5000, 1.0000, 2.0000],
                                          [2.0000, 1.0000, 2.5000],
                                          [2.0000, 1.5000, 2.0000],
                                          [2.5000, 1.0000, 2.0000]], device='cuda:0')
        
        expected_faces = torch.tensor([[ 0,  1,  2],
                                       [ 3,  2,  1],
                                       [ 4,  0,  2],
                                       [ 4,  2,  3],
                                       [ 0,  6,  1],
                                       [ 1,  6,  5],
                                       [ 7,  1,  5],
                                       [ 1,  8,  3],
                                       [ 1,  7,  8],
                                       [ 9,  8,  7],
                                       [ 4,  6,  0],
                                       [10,  5,  6],
                                       [ 4, 10,  6],
                                       [10,  3,  8],
                                       [10,  4,  3],
                                       [10,  8,  9],
                                       [ 5, 11,  7],
                                       [11,  9,  7],
                                       [10, 11,  5],
                                       [11, 10,  9]], device='cuda:0', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_3(self):
        voxelgrid = torch.tensor([[[0, 1], 
                                   [1, 1]], 
                                  [[0, 0], 
                                   [0, 0]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[1.0000, 1.0000, 1.5000],
                                          [1.0000, 0.5000, 2.0000],
                                          [0.5000, 1.0000, 2.0000],
                                          [1.0000, 1.0000, 2.5000],
                                          [1.0000, 2.0000, 0.5000],
                                          [1.0000, 1.5000, 1.0000],
                                          [0.5000, 2.0000, 1.0000],
                                          [1.0000, 2.0000, 2.5000],
                                          [0.5000, 2.0000, 2.0000],
                                          [1.0000, 2.5000, 1.0000],
                                          [1.0000, 2.5000, 2.0000],
                                          [1.5000, 1.0000, 2.0000],
                                          [1.5000, 2.0000, 1.0000],
                                          [1.5000, 2.0000, 2.0000]], device='cuda:0')
        
        expected_faces = torch.tensor([[ 0,  1,  2],
                                       [ 3,  2,  1],
                                       [ 4,  5,  6],
                                       [ 6,  5,  0],
                                       [ 2,  6,  0],
                                       [ 8,  6,  2],
                                       [ 2,  3,  7],
                                       [ 2,  7,  8],
                                       [ 9,  4,  6],
                                       [ 9,  6,  8],
                                       [ 9,  8, 10],
                                       [10,  8,  7],
                                       [ 0, 11,  1],
                                       [11,  3,  1],
                                       [ 4, 12,  5],
                                       [11,  0,  5],
                                       [12, 11,  5],
                                       [13, 11, 12],
                                       [13,  7,  3],
                                       [13,  3, 11],
                                       [ 9, 12,  4],
                                       [12,  9, 10],
                                       [12, 10, 13],
                                       [13, 10,  7]], device='cuda:0', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_4(self):
        voxelgrid = torch.tensor([[[1, 1], 
                                   [1, 1]], 
                                  [[0, 0], 
                                   [0, 0]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[1.0000, 1.0000, 0.5000],
                                          [0.5000, 1.0000, 1.0000],
                                          [1.0000, 0.5000, 1.0000],
                                          [1.0000, 0.5000, 2.0000],
                                          [0.5000, 1.0000, 2.0000],
                                          [1.0000, 1.0000, 2.5000],
                                          [1.0000, 2.0000, 0.5000],
                                          [0.5000, 2.0000, 1.0000],
                                          [1.0000, 2.0000, 2.5000],
                                          [0.5000, 2.0000, 2.0000],
                                          [1.0000, 2.5000, 1.0000],
                                          [1.0000, 2.5000, 2.0000],
                                          [1.5000, 1.0000, 1.0000],
                                          [1.5000, 1.0000, 2.0000],
                                          [1.5000, 2.0000, 1.0000],
                                          [1.5000, 2.0000, 2.0000]], device='cuda:0')

        expected_faces = torch.tensor([[ 0,  2,  1],
                                       [ 3,  4,  1],
                                       [ 3,  1,  2],
                                       [ 5,  4,  3],
                                       [ 0,  1,  7],
                                       [ 0,  7,  6],
                                       [ 9,  1,  4],
                                       [ 7,  1,  9],
                                       [ 4,  5,  8],
                                       [ 4,  8,  9],
                                       [10,  6,  7],
                                       [10,  7,  9],
                                       [10,  9, 11],
                                       [11,  9,  8],
                                       [ 0, 12,  2],
                                       [13,  3,  2],
                                       [13,  2, 12],
                                       [13,  5,  3],
                                       [ 6, 14, 12],
                                       [ 6, 12,  0],
                                       [12, 15, 13],
                                       [12, 14, 15],
                                       [15,  8,  5],
                                       [15,  5, 13],
                                       [10, 14,  6],
                                       [14, 10, 11],
                                       [14, 11, 15],
                                       [15, 11,  8]], device='cuda:0', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_5(self):
        voxelgrid = torch.tensor([[[0, 1], 
                                   [1, 1]], 
                                  [[1, 0], 
                                   [0, 0]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[1.0000, 1.0000, 1.5000],
                                          [1.0000, 0.5000, 2.0000],
                                          [0.5000, 1.0000, 2.0000],
                                          [1.0000, 1.0000, 2.5000],
                                          [1.0000, 2.0000, 0.5000],
                                          [1.0000, 1.5000, 1.0000],
                                          [0.5000, 2.0000, 1.0000],
                                          [1.0000, 2.0000, 2.5000],
                                          [0.5000, 2.0000, 2.0000],
                                          [1.0000, 2.5000, 1.0000],
                                          [1.0000, 2.5000, 2.0000],
                                          [2.0000, 1.0000, 0.5000],
                                          [1.5000, 1.0000, 1.0000],
                                          [2.0000, 0.5000, 1.0000],
                                          [2.0000, 1.0000, 1.5000],
                                          [1.5000, 1.0000, 2.0000],
                                          [2.0000, 1.5000, 1.0000],
                                          [1.5000, 2.0000, 1.0000],
                                          [1.5000, 2.0000, 2.0000],
                                          [2.5000, 1.0000, 1.0000]], device='cuda:0')
        
        expected_faces = torch.tensor([[ 0,  1,  2],
                                       [ 3,  2,  1],
                                       [ 4,  5,  6],
                                       [ 6,  5,  0],
                                       [ 2,  6,  0],
                                       [ 8,  6,  2],
                                       [ 2,  3,  7],
                                       [ 2,  7,  8],
                                       [ 9,  4,  6],
                                       [ 9,  6,  8],
                                       [ 9,  8, 10],
                                       [10,  8,  7],
                                       [11, 13, 12],
                                       [ 1,  0, 12],
                                       [13,  1, 12],
                                       [ 1, 14, 15],
                                       [ 1, 13, 14],
                                       [15,  3,  1],
                                       [ 4, 17, 16],
                                       [11,  4, 16],
                                       [ 4, 12,  5],
                                       [ 4, 11, 12],
                                       [ 5, 12,  0],
                                       [16, 15, 14],
                                       [16, 17, 15],
                                       [17, 18, 15],
                                       [18,  7,  3],
                                       [18,  3, 15],
                                       [ 9, 17,  4],
                                       [17,  9, 10],
                                       [17, 10, 18],
                                       [18, 10,  7],
                                       [11, 19, 13],
                                       [19, 14, 13],
                                       [16, 19, 11],
                                       [19, 16, 14]], device='cuda:0', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)
    
    def test_voxelgrids_to_trianglemeshes_6(self):
        voxelgrid = torch.tensor([[[1, 0], 
                                   [0, 1]], 
                                  [[0, 1], 
                                   [1, 0]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[1.0000, 1.0000, 0.5000],
                                          [1.0000, 0.5000, 1.0000],
                                          [0.5000, 1.0000, 1.0000],
                                          [1.0000, 1.0000, 1.5000],
                                          [1.0000, 1.5000, 1.0000],
                                          [1.0000, 2.0000, 1.5000],
                                          [1.0000, 1.5000, 2.0000],
                                          [0.5000, 2.0000, 2.0000],
                                          [1.0000, 2.0000, 2.5000],
                                          [1.0000, 2.5000, 2.0000],
                                          [2.0000, 1.0000, 1.5000],
                                          [1.5000, 1.0000, 1.0000],
                                          [2.0000, 0.5000, 2.0000],
                                          [1.5000, 1.0000, 2.0000],
                                          [2.0000, 1.0000, 2.5000],
                                          [2.0000, 2.0000, 0.5000],
                                          [2.0000, 1.5000, 1.0000],
                                          [1.5000, 2.0000, 1.0000],
                                          [2.0000, 2.0000, 1.5000],
                                          [1.5000, 2.0000, 2.0000],
                                          [2.0000, 1.5000, 2.0000],
                                          [2.0000, 2.5000, 1.0000],
                                          [2.5000, 1.0000, 2.0000],
                                          [2.5000, 2.0000, 1.0000]], device='cuda:0')
        
        expected_faces = torch.tensor([[ 0,  1,  2],
                                       [ 3,  2,  1],
                                       [ 4,  0,  2],
                                       [ 7,  5,  4],
                                       [ 2,  7,  4],
                                       [ 7,  3,  6],
                                       [ 7,  2,  3],
                                       [ 8,  7,  6],
                                       [ 9,  5,  7],
                                       [ 9,  7,  8],
                                       [ 0, 11,  1],
                                       [ 1, 11, 10],
                                       [12,  1, 10],
                                       [ 1, 13,  3],
                                       [ 1, 12, 13],
                                       [14, 13, 12],
                                       [ 0,  4, 17],
                                       [15,  0, 17],
                                       [ 0, 16, 11],
                                       [ 0, 15, 16],
                                       [13,  6,  3],
                                       [11, 16, 10],
                                       [17,  4,  5],
                                       [18, 19, 20],
                                       [14, 20, 19],
                                       [ 8, 14, 19],
                                       [14,  6, 13],
                                       [14,  8,  6],
                                       [21, 15, 17],
                                       [ 9, 19, 18],
                                       [21,  9, 18],
                                       [ 9, 17,  5],
                                       [ 9, 21, 17],
                                       [19,  9,  8],
                                       [10, 22, 12],
                                       [22, 14, 12],
                                       [15, 23, 16],
                                       [22, 10, 16],
                                       [23, 22, 16],
                                       [22, 18, 20],
                                       [22, 23, 18],
                                       [22, 20, 14],
                                       [21, 23, 15],
                                       [23, 21, 18]], device='cuda:0', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_7(self):
        voxelgrid = torch.tensor([[[1, 0], 
                                   [1, 1]], 
                                  [[0, 0], 
                                   [1, 0]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[1.0000, 1.0000, 0.5000],
                                          [1.0000, 0.5000, 1.0000],
                                          [0.5000, 1.0000, 1.0000],
                                          [1.0000, 1.0000, 1.5000],
                                          [1.0000, 2.0000, 0.5000],
                                          [0.5000, 2.0000, 1.0000],
                                          [1.0000, 1.5000, 2.0000],
                                          [0.5000, 2.0000, 2.0000],
                                          [1.0000, 2.0000, 2.5000],
                                          [1.0000, 2.5000, 1.0000],
                                          [1.0000, 2.5000, 2.0000],
                                          [1.5000, 1.0000, 1.0000],
                                          [2.0000, 2.0000, 0.5000],
                                          [2.0000, 2.0000, 1.5000],
                                          [2.0000, 1.5000, 1.0000],
                                          [1.5000, 2.0000, 2.0000],
                                          [2.0000, 2.5000, 1.0000],
                                          [2.5000, 2.0000, 1.0000]], device='cuda:0')
        
        expected_faces = torch.tensor([[ 0,  1,  2],
                                       [ 3,  2,  1],
                                       [ 0,  2,  5],
                                       [ 0,  5,  4],
                                       [ 2,  3,  6],
                                       [ 7,  2,  6],
                                       [ 5,  2,  7],
                                       [ 8,  7,  6],
                                       [ 9,  4,  5],
                                       [ 9,  5,  7],
                                       [ 9,  7, 10],
                                       [10,  7,  8],
                                       [ 0, 11,  1],
                                       [11,  3,  1],
                                       [12, 14, 11],
                                       [ 0, 12, 11],
                                       [ 4, 12,  0],
                                       [14, 13, 15],
                                       [14, 15,  6],
                                       [11, 14,  6],
                                       [ 3, 11,  6],
                                       [15,  8,  6],
                                       [16, 12,  4],
                                       [16,  4,  9],
                                       [10, 15, 13],
                                       [16, 10, 13],
                                       [ 9, 10, 16],
                                       [15, 10,  8],
                                       [12, 17, 14],
                                       [17, 13, 14],
                                       [16, 17, 12],
                                       [17, 16, 13]], device='cuda:0', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_8(self):
        voxelgrid = torch.tensor([[[0, 1], 
                                   [1, 1]], 
                                  [[0, 0], 
                                   [1, 0]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[1.0000, 1.0000, 1.5000],
                                          [1.0000, 0.5000, 2.0000],
                                          [0.5000, 1.0000, 2.0000],
                                          [1.0000, 1.0000, 2.5000],
                                          [1.0000, 2.0000, 0.5000],
                                          [1.0000, 1.5000, 1.0000],
                                          [0.5000, 2.0000, 1.0000],
                                          [1.0000, 2.0000, 2.5000],
                                          [0.5000, 2.0000, 2.0000],
                                          [1.0000, 2.5000, 1.0000],
                                          [1.0000, 2.5000, 2.0000],
                                          [1.5000, 1.0000, 2.0000],
                                          [2.0000, 2.0000, 0.5000],
                                          [2.0000, 1.5000, 1.0000],
                                          [2.0000, 2.0000, 1.5000],
                                          [1.5000, 2.0000, 2.0000],
                                          [2.0000, 2.5000, 1.0000],
                                          [2.5000, 2.0000, 1.0000]], device='cuda:0')
        
        expected_faces = torch.tensor([[ 0,  1,  2],
                                       [ 3,  2,  1],
                                       [ 4,  5,  6],
                                       [ 6,  5,  0],
                                       [ 2,  6,  0],
                                       [ 8,  6,  2],
                                       [ 2,  3,  7],
                                       [ 2,  7,  8],
                                       [ 9,  4,  6],
                                       [ 9,  6,  8],
                                       [ 9,  8, 10],
                                       [10,  8,  7],
                                       [ 0, 11,  1],
                                       [11,  3,  1],
                                       [12, 13,  5],
                                       [12,  5,  4],
                                       [ 5, 13,  0],
                                       [13, 15,  0],
                                       [15, 11,  0],
                                       [15, 13, 14],
                                       [15,  7,  3],
                                       [15,  3, 11],
                                       [16, 12,  4],
                                       [16,  4,  9],
                                       [10, 15, 14],
                                       [16, 10, 14],
                                       [ 9, 10, 16],
                                       [15, 10,  7],
                                       [12, 17, 13],
                                       [17, 14, 13],
                                       [16, 17, 12],
                                       [17, 16, 14]], device='cuda:0', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_9(self):
        voxelgrid = torch.tensor([[[1, 0], 
                                   [0, 0]], 
                                  [[0, 0], 
                                   [0, 1]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[1.0000, 1.0000, 0.5000],
                                          [1.0000, 0.5000, 1.0000],
                                          [0.5000, 1.0000, 1.0000],
                                          [1.0000, 1.0000, 1.5000],
                                          [1.0000, 1.5000, 1.0000],
                                          [1.5000, 1.0000, 1.0000],
                                          [2.0000, 2.0000, 1.5000],
                                          [2.0000, 1.5000, 2.0000],
                                          [1.5000, 2.0000, 2.0000],
                                          [2.0000, 2.0000, 2.5000],
                                          [2.0000, 2.5000, 2.0000],
                                          [2.5000, 2.0000, 2.0000]], device='cuda:0')
        
        expected_faces = torch.tensor([[ 0,  1,  2],
                                       [ 3,  2,  1],
                                       [ 4,  0,  2],
                                       [ 4,  2,  3],
                                       [ 0,  5,  1],
                                       [ 5,  3,  1],
                                       [ 4,  5,  0],
                                       [ 5,  4,  3],
                                       [ 8,  6,  7],
                                       [ 9,  8,  7],
                                       [10,  6,  8],
                                       [10,  8,  9],
                                       [ 6, 11,  7],
                                       [11,  9,  7],
                                       [10, 11,  6],
                                       [11, 10,  9]], device='cuda:0', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_10(self):
        voxelgrid = torch.tensor([[[1, 1], 
                                   [0, 0]], 
                                  [[0, 0], 
                                   [0, 1]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[1.0000, 1.0000, 0.5000],
                                          [0.5000, 1.0000, 1.0000],
                                          [1.0000, 0.5000, 1.0000],
                                          [1.0000, 0.5000, 2.0000],
                                          [0.5000, 1.0000, 2.0000],
                                          [1.0000, 1.0000, 2.5000],
                                          [1.0000, 1.5000, 1.0000],
                                          [1.0000, 1.5000, 2.0000],
                                          [1.5000, 1.0000, 1.0000],
                                          [1.5000, 1.0000, 2.0000],
                                          [2.0000, 2.0000, 1.5000],
                                          [2.0000, 1.5000, 2.0000],
                                          [2.0000, 2.0000, 2.5000],
                                          [1.5000, 2.0000, 2.0000],
                                          [2.0000, 2.5000, 2.0000],
                                          [2.5000, 2.0000, 2.0000]], device='cuda:0')
                                          
        expected_faces = torch.tensor([[ 0,  2,  1],
                                       [ 3,  4,  1],
                                       [ 3,  1,  2],
                                       [ 5,  4,  3],
                                       [ 6,  0,  1],
                                       [ 6,  1,  4],
                                       [ 6,  4,  7],
                                       [ 7,  4,  5],
                                       [ 0,  8,  2],
                                       [ 9,  3,  2],
                                       [ 9,  2,  8],
                                       [ 9,  5,  3],
                                       [ 6,  8,  0],
                                       [10,  6,  7],
                                       [13, 10,  7],
                                       [10,  8,  6],
                                       [ 9, 10, 11],
                                       [10,  9,  8],
                                       [ 5,  9, 11],
                                       [12,  5, 11],
                                       [ 5, 13,  7],
                                       [ 5, 12, 13],
                                       [14, 10, 13],
                                       [14, 13, 12],
                                       [10, 15, 11],
                                       [15, 12, 11],
                                       [14, 15, 10],
                                       [15, 14, 12]], device='cuda:0', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_11(self):
        voxelgrid = torch.tensor([[[0, 1], 
                                   [0, 0]], 
                                  [[1, 0], 
                                   [0, 1]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[1.0000, 1.0000, 1.5000],
                                          [1.0000, 0.5000, 2.0000],
                                          [0.5000, 1.0000, 2.0000],
                                          [1.0000, 1.0000, 2.5000],
                                          [1.0000, 1.5000, 2.0000],
                                          [2.0000, 1.0000, 0.5000],
                                          [1.5000, 1.0000, 1.0000],
                                          [2.0000, 0.5000, 1.0000],
                                          [2.0000, 1.0000, 1.5000],
                                          [1.5000, 1.0000, 2.0000],
                                          [2.0000, 2.0000, 1.5000],
                                          [2.0000, 1.5000, 1.0000],
                                          [2.0000, 1.5000, 2.0000],
                                          [2.0000, 2.0000, 2.5000],
                                          [1.5000, 2.0000, 2.0000],
                                          [2.0000, 2.5000, 2.0000],
                                          [2.5000, 1.0000, 1.0000],
                                          [2.5000, 2.0000, 2.0000]], device='cuda:0')
        
        expected_faces = torch.tensor([[ 0,  1,  2],
                                       [ 3,  2,  1],
                                       [ 4,  0,  2],
                                       [ 4,  2,  3],
                                       [ 5,  7,  6],
                                       [ 1,  0,  6],
                                       [ 7,  1,  6],
                                       [ 1,  8,  9],
                                       [ 1,  7,  8],
                                       [ 9,  3,  1],
                                       [11,  5,  6],
                                       [14, 10,  4],
                                       [10, 11,  4],
                                       [11,  0,  4],
                                       [ 0, 11,  6],
                                       [ 8, 12,  9],
                                       [ 3,  9, 12],
                                       [13,  3, 12],
                                       [ 3, 14,  4],
                                       [ 3, 13, 14],
                                       [15, 10, 14],
                                       [15, 14, 13],
                                       [ 5, 16,  7],
                                       [16,  8,  7],
                                       [11, 16,  5],
                                       [16, 11, 10],
                                       [17, 16, 10],
                                       [16, 12,  8],
                                       [16, 17, 12],
                                       [17, 13, 12],
                                       [15, 17, 10],
                                       [17, 15, 13]], device='cuda:0', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_12(self):
        voxelgrid = torch.tensor([[[1, 0], 
                                   [0, 1]], 
                                  [[1, 0], 
                                   [0, 1]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[1.0000, 1.0000, 0.5000],
                                          [1.0000, 0.5000, 1.0000],
                                          [0.5000, 1.0000, 1.0000],
                                          [1.0000, 1.0000, 1.5000],
                                          [1.0000, 1.5000, 1.0000],
                                          [1.0000, 2.0000, 1.5000],
                                          [1.0000, 1.5000, 2.0000],
                                          [0.5000, 2.0000, 2.0000],
                                          [1.0000, 2.0000, 2.5000],
                                          [1.0000, 2.5000, 2.0000],
                                          [2.0000, 1.0000, 0.5000],
                                          [2.0000, 0.5000, 1.0000],
                                          [2.0000, 1.0000, 1.5000],
                                          [2.0000, 2.0000, 1.5000],
                                          [2.0000, 1.5000, 1.0000],
                                          [2.0000, 1.5000, 2.0000],
                                          [2.0000, 2.0000, 2.5000],
                                          [2.0000, 2.5000, 2.0000],
                                          [2.5000, 1.0000, 1.0000],
                                          [2.5000, 2.0000, 2.0000]], device='cuda:0')
        
        expected_faces = torch.tensor([[ 0,  1,  2],
                                       [ 3,  2,  1],
                                       [ 4,  0,  2],
                                       [ 7,  5,  4],
                                       [ 2,  7,  4],
                                       [ 7,  3,  6],
                                       [ 7,  2,  3],
                                       [ 8,  7,  6],
                                       [ 9,  5,  7],
                                       [ 9,  7,  8],
                                       [10, 11,  1],
                                       [10,  1,  0],
                                       [ 3,  1, 11],
                                       [ 3, 11, 12],
                                       [14, 10,  0],
                                       [14,  0,  4],
                                       [ 5, 13,  4],
                                       [13, 14,  4],
                                       [ 3, 15,  6],
                                       [ 3, 12, 15],
                                       [ 8,  6, 15],
                                       [ 8, 15, 16],
                                       [17, 13,  5],
                                       [17,  5,  9],
                                       [ 9,  8, 16],
                                       [ 9, 16, 17],
                                       [10, 18, 11],
                                       [18, 12, 11],
                                       [14, 18, 10],
                                       [18, 14, 13],
                                       [19, 18, 13],
                                       [18, 15, 12],
                                       [18, 19, 15],
                                       [19, 16, 15],
                                       [17, 19, 13],
                                       [19, 17, 16]], device='cuda:0', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def test_voxelgrids_to_trianglemeshes_13(self):
        voxelgrid = torch.tensor([[[1, 0], 
                                   [1, 1]], 
                                  [[0, 0], 
                                   [0, 1]]], device='cuda', dtype=torch.uint8)

        vertices, faces = vg.voxelgrids_to_trianglemeshes(voxelgrid.unsqueeze(0))

        expected_vertices = torch.tensor([[1.0000, 1.0000, 0.5000],
                                          [1.0000, 0.5000, 1.0000],
                                          [0.5000, 1.0000, 1.0000],
                                          [1.0000, 1.0000, 1.5000],
                                          [1.0000, 2.0000, 0.5000],
                                          [0.5000, 2.0000, 1.0000],
                                          [1.0000, 1.5000, 2.0000],
                                          [0.5000, 2.0000, 2.0000],
                                          [1.0000, 2.0000, 2.5000],
                                          [1.0000, 2.5000, 1.0000],
                                          [1.0000, 2.5000, 2.0000],
                                          [1.5000, 1.0000, 1.0000],
                                          [1.5000, 2.0000, 1.0000],
                                          [2.0000, 2.0000, 1.5000],
                                          [2.0000, 1.5000, 2.0000],
                                          [2.0000, 2.0000, 2.5000],
                                          [2.0000, 2.5000, 2.0000],
                                          [2.5000, 2.0000, 2.0000]], device='cuda:0')
        
        expected_faces = torch.tensor([[ 0,  1,  2],
                                       [ 3,  2,  1],
                                       [ 0,  2,  5],
                                       [ 0,  5,  4],
                                       [ 2,  3,  6],
                                       [ 7,  2,  6],
                                       [ 5,  2,  7],
                                       [ 8,  7,  6],
                                       [ 9,  4,  5],
                                       [ 9,  5,  7],
                                       [ 9,  7, 10],
                                       [10,  7,  8],
                                       [ 0, 11,  1],
                                       [11,  3,  1],
                                       [ 4, 12, 11],
                                       [ 4, 11,  0],
                                       [11, 12,  3],
                                       [12, 14,  3],
                                       [14,  6,  3],
                                       [12, 13, 14],
                                       [ 8,  6, 14],
                                       [ 8, 14, 15],
                                       [ 9, 12,  4],
                                       [16, 13, 12],
                                       [ 9, 16, 12],
                                       [10, 16,  9],
                                       [10,  8, 15],
                                       [10, 15, 16],
                                       [13, 17, 14],
                                       [17, 15, 14],
                                       [16, 17, 13],
                                       [17, 16, 15]], device='cuda:0', dtype=torch.long)

        assert torch.equal(expected_vertices, vertices[0])
        assert torch.equal(expected_faces, faces[0])

        self._all_variations_test(voxelgrid, expected_vertices)

    def _flip(self, voxelgrid, expected_vertices, dims):
        voxelgrid_length = voxelgrid.shape[0] + 1

        new_expected_vertices = expected_vertices.detach().clone()

        for d in dims:
            if d == 0:
                new_expected_vertices[:, 0] = voxelgrid_length - expected_vertices[:, 0]
            elif d == 1:
                new_expected_vertices[:, 1] = voxelgrid_length - expected_vertices[:, 1]
            elif d == 2:
                new_expected_vertices[:, 2] = voxelgrid_length - expected_vertices[:, 2]

        if len(dims) == 0:
            new_voxelgrid = voxelgrid.detach().clone()
        else:
            new_voxelgrid = voxelgrid.flip(dims)

        return new_expected_vertices, new_voxelgrid
    
    def _rotate(self, voxelgrid, expected_vertices, dims):

        new_voxelgrid = voxelgrid.permute(dims)

        new_expected_vertices = expected_vertices.detach().clone()

        if dims == (0, 2, 1) or dims == [0, 2, 1]:
            new_expected_vertices[:, 1], new_expected_vertices[:, 2] = expected_vertices[:, 2], expected_vertices[:, 1]
        elif dims == (1, 0, 2) or dims == [1, 0, 2]:
            new_expected_vertices[:, 0], new_expected_vertices[:, 1] = expected_vertices[:, 1], expected_vertices[:, 0]
        elif dims == (1, 2, 0) or dims == [1, 2, 0]:
            new_expected_vertices[:, 0] = expected_vertices[:, 1]
            new_expected_vertices[:, 1] = expected_vertices[:, 2]
            new_expected_vertices[:, 2] = expected_vertices[:, 0]
            # pass
        elif dims == (2, 1, 0) or dims == [2, 1, 0]:
            new_expected_vertices[:, 0], new_expected_vertices[:, 2] = expected_vertices[:, 2], expected_vertices[:, 0]
        elif dims == (2, 0, 1) or dims == [2, 0, 1]:
            new_expected_vertices[:, 0] = expected_vertices[:, 2]
            new_expected_vertices[:, 1] = expected_vertices[:, 0]
            new_expected_vertices[:, 2] = expected_vertices[:, 1]

        return new_expected_vertices, new_voxelgrid

    def _all_variations_test(self, voxelgrid, expected_vertices):

        all_rotations = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 1, 0], [2, 0, 1]]
        all_flips = [[], [0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]

        for rotate in all_rotations:
            new_expected_vertices, new_voxelgrid = self._rotate(voxelgrid, expected_vertices, rotate)
            for flip in all_flips:
                curr_expected_vertices, curr_voxelgrid = self._flip(new_voxelgrid, new_expected_vertices, flip)
                curr_vertices, _ = vg.voxelgrids_to_trianglemeshes(curr_voxelgrid.unsqueeze(0))

                curr_vertices = torch.sort(curr_vertices[0], dim=0)[0]
                curr_expected_vertices = torch.sort(curr_expected_vertices, dim=0)[0]

                assert torch.equal(curr_vertices, curr_expected_vertices)
    
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