# Copyright (c) 2019,20-22, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from kaolin.io import utils
from kaolin.utils.testing import contained_torch_equal


class TestUtils:
    @pytest.mark.parametrize(
        'handler', [utils.heterogeneous_mesh_handler_naive_homogenize, utils.mesh_handler_naive_triangulate])
    @pytest.mark.parametrize(
        'face_assignment_mode', [0, 1, 2])
    def test_mesh_handler_naive_triangulate(self, handler, face_assignment_mode):
        N = 15
        vertices = torch.rand((N, 3), dtype=torch.float32)
        face_vertex_counts = torch.LongTensor([3, 4, 5, 3, 6])
        faces = torch.LongTensor(
            [0, 1, 2,                  # Face 0 -> 1 face idx [0]
             2, 1, 3, 4,               # Face 1 -> 2 faces idx [1, 2]
             4, 5, 6, 7, 8,            # Face 2 -> 3 faces idx [3, 4, 5]
             3, 4, 6,                  # Face 3 -> 1 face idx [6]
             8, 9, 10, 11, 12, 13])    # Face 4 -> 4 faces idx [7, 8, 9, 10]
        expected_faces = torch.LongTensor(
            [[0, 1, 2],
             [2, 1, 3],   [2, 3, 4],
             [4, 5, 6],   [4, 6, 7],   [4, 7, 8],
             [3, 4, 6],
             [8, 9, 10],  [8, 10, 11], [8, 11, 12],   [8, 12, 13]])
        expected_num_faces = 11
        expected_face_vertex_counts = torch.LongTensor([3 for _ in range(expected_num_faces)])
        face_uvs_idx = torch.LongTensor(
            [0, 1, 2,                  # UVs for face 0
             10, 11, 12, 13,           # UVs for face 1
             20, 21, 22, 23, 24,       # UVs for face 2
             30, 31, 32,               # UVs for face 3
             40, 41, 42, 43, 44, 45])  # UVs for face 4
        expected_face_uvs_idx = torch.LongTensor(
            [[0, 1, 2],
             [10, 11, 12],   [10, 12, 13],
             [20, 21, 22],   [20, 22, 23],    [20, 23, 24],
             [30, 31, 32],
             [40, 41, 42],   [40, 42, 43],   [40, 43, 44],   [40, 44, 45]])

        # assignments to faces
        face_assignments = None
        expected_face_assignments = None
        with_assignments = face_assignment_mode > 0
        if with_assignments:
            if face_assignment_mode == 1:   # 1D tensors for face assignemtns replaced with new face indices
                face_assignments = {
                    '1': torch.LongTensor([0, 2]),
                    '2': torch.LongTensor([1, 3, 4])}
                expected_face_assignments = {
                    '1': torch.LongTensor([0, 3, 4, 5]),
                    '2': torch.LongTensor([1, 2, 6, 7, 8, 9, 10])}
            else:  # 2D tensors of start and end face_idx, replaced with new start and end face_idx
                face_assignments = {
                    'cat': torch.LongTensor([[0, 2], [3, 4], [2, 5]]),
                    'dog': torch.LongTensor([[1, 3]])}
                expected_face_assignments = {
                    'cat': torch.LongTensor([[0, 3], [6, 7], [3, 11]]),
                    'dog': torch.LongTensor([[1, 6]])}

        res = handler(
            vertices, face_vertex_counts, faces, face_uvs_idx, face_assignments=face_assignments)
        assert len(res) == (5 if with_assignments else 4)
        new_vertices = res[0]
        new_face_vertex_counts = res[1]
        new_faces = res[2]
        new_face_uvs_idx = res[3]

        assert torch.allclose(new_vertices, vertices)
        assert torch.equal(new_face_vertex_counts, expected_face_vertex_counts)
        assert torch.equal(new_faces, expected_faces)
        assert torch.equal(new_face_uvs_idx, expected_face_uvs_idx)

        if with_assignments:
            new_face_assignments = res[4]
            assert contained_torch_equal(new_face_assignments, expected_face_assignments)


