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

import kaolin as kal
from kaolin.ops.conversions import trianglemeshes_to_voxelgrids, unbatched_mesh_to_spc
from kaolin.utils.testing import FLOAT_TYPES


@pytest.mark.parametrize('device, dtype', FLOAT_TYPES)
@pytest.mark.parametrize('return_sparse', [True, False])
class TestTriangleMeshToVoxelgrid:

    def test_resolution_type(self, device, dtype, return_sparse):
        vertices = torch.tensor([[[0, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]]], dtype=dtype, device=device)

        faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        origins = torch.zeros((1, 3), dtype=dtype, device=device)

        scale = torch.ones((1), dtype=dtype, device=device)

        with pytest.raises(TypeError, match=r"Expected resolution to be int "
                                            r"but got .*"):
            trianglemeshes_to_voxelgrids(
                vertices, faces, 2.3, origins, scale, return_sparse
            )

    def test_mesh_to_voxel_batched(self, device, dtype, return_sparse):
        vertices = torch.tensor([[[0, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]],

                                 [[0, 0, 0],
                                  [0, 1, 0],
                                  [1, 0, 1]]], dtype=dtype, device=device)

        faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        origins = torch.zeros((2, 3), dtype=dtype, device=device)

        scale = torch.ones((2), dtype=dtype, device=device)

        output = trianglemeshes_to_voxelgrids(
            vertices, faces, 3, origins, scale, return_sparse
        )

        # output voxelgrid should have value around the corner
        expected = torch.tensor([[[[1., 1., 1.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]],

                                  [[1., 1., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]],

                                  [[1., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]]],

                                 [[[1., 0., 0.],
                                   [1., 0., 0.],
                                   [1., 0., 0.]],

                                  [[0., 1., 0.],
                                   [0., 1., 0.],
                                   [0., 0., 0.]],

                                  [[0., 0., 1.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]]]],device=device, dtype=dtype)

        if return_sparse:
            output = output.to_dense()
        assert torch.equal(output, expected)

    def test_mesh_to_voxel_origins(self, device, dtype, return_sparse):
        vertices = torch.tensor([[[0, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]]], dtype=dtype, device=device)

        faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        origins = torch.zeros((1, 3), dtype=dtype, device=device)
        origins[0][2] = 0.6

        scale = torch.ones((1), dtype=dtype, device=device)

        output = trianglemeshes_to_voxelgrids(
            vertices, faces, 3, origins, scale, return_sparse
        )

        expected = torch.tensor([[[[1., 1., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]],

                                  [[1., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]],

                                  [[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]]]], device=device, dtype=dtype)

        if return_sparse:
            output = output.to_dense()
        assert torch.equal(output, expected)

    def test_mesh_to_voxel_scale(self, device, dtype, return_sparse):
        vertices = torch.tensor([[[0, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]]], dtype=dtype, device=device)

        faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        origins = torch.zeros((1, 3), dtype=dtype, device=device)

        scale = torch.ones((1), dtype=dtype, device=device) * 2

        output = trianglemeshes_to_voxelgrids(
            vertices, faces, 3, origins, scale, return_sparse
        )

        expected = torch.tensor([[[[1., 1., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]],

                                  [[1., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]],

                                  [[0., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]]]], device=device, dtype=dtype)

        if return_sparse:
            output = output.to_dense()
        assert torch.equal(output, expected)

    def test_mesh_to_voxel_resolution_3(self, device, dtype, return_sparse):
        vertices = torch.tensor([[[0, 0, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]]], dtype=dtype, device=device)

        faces = torch.tensor([[0, 1, 2]], dtype=torch.long, device=device)

        origins = torch.zeros((1, 3), dtype=dtype, device=device)

        scale = torch.ones((1), dtype=dtype, device=device)

        output = trianglemeshes_to_voxelgrids(
            vertices, faces, 3, origins, scale, return_sparse
        )

        expected = torch.tensor([[[[1., 1., 1.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]],

                                  [[1., 1., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]],

                                  [[1., 0., 0.],
                                   [0., 0., 0.],
                                   [0., 0., 0.]]]], device=device, dtype=dtype)

        if return_sparse:
            output = output.to_dense()
        assert torch.equal(output, expected)
    
    def test_rectangle(self, device, dtype, return_sparse):
        vertices = torch.tensor([[0, 0, 0],
                                 [8, 0, 0],
                                 [0, 8, 0],
                                 [8, 8, 0],
                                 [0, 0, 12],
                                 [8, 0, 12],
                                 [0, 8, 12],
                                 [8, 8, 12]], dtype=dtype, device=device)

        faces = torch.tensor([[0, 3, 1],
                              [0, 2, 3],
                              [0, 1, 5],
                              [0, 5, 4],
                              [1, 7, 5],
                              [1, 3, 7],
                              [0, 6, 2],
                              [0, 4, 6],
                              [4, 7, 6],
                              [4, 5, 7],
                              [2, 7, 3],
                              [2, 6, 7]], dtype=torch.long, device=device)

        origin = torch.zeros((1, 3), dtype=dtype, device=device)

        origin[0][2] = 2

        scale = torch.ones((1), dtype=dtype, device=device) * 8

        output = trianglemeshes_to_voxelgrids(
            vertices.unsqueeze(0), faces, 4, origin, scale, return_sparse
        )

        expected = torch.tensor([[[[1., 1., 1., 1.],
                                   [1., 1., 1., 1.],
                                   [1., 1., 1., 1.],
                                   [1., 1., 1., 1.]],

                                  [[1., 1., 1., 1.],
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],
                                   [1., 1., 1., 1.]],

                                  [[1., 1., 1., 1.],
                                   [0., 0., 0., 0.],
                                   [0., 0., 0., 0.],
                                   [1., 1., 1., 1.]],

                                  [[1., 1., 1., 1.],
                                   [1., 1., 1., 1.],
                                   [1., 1., 1., 1.],
                                   [1., 1., 1., 1.]]]], device=device, dtype=dtype)

        if return_sparse:
            output = output.to_dense()
        assert torch.equal(output, expected)

@pytest.mark.parametrize('level', [3])
class TestUnbatchedMeshToSpc:
    @pytest.fixture(autouse=True)
    def faces(self):
        return torch.tensor([
            [0, 1, 2],
            [2, 1, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], device='cuda', dtype=torch.long)

    @pytest.fixture(autouse=True)
    def vertices(self):
        return torch.tensor([[
            [-0.4272,  0.0795,  0.3548],
            [-0.9217,  0.3106,  0.1516],
            [-0.2636,  0.3794, -0.7979],
            [ 0.1259,  0.9089,  0.7439],
            [ 0.0710, -0.6947, -0.0480],
            [ 0.6215,  0.2809, -0.0480],
            [ 0.4972,  0.3347,  0.4422],
            [-0.4374,  0.4967, -0.6047],
            [ 0.0397,  0.1230, -0.7417],
            [-0.3534,  0.9970, -0.4558]
        ]], device='cuda')

    @pytest.fixture(autouse=True)
    def face_vertices(self, vertices, faces):
        return kal.ops.mesh.index_vertices_by_faces(vertices, faces)

    @pytest.fixture(autouse=True)
    def expected_octree(self, level):
        if level == 1:
            return torch.tensor([], dtype=torch.uint8, device='cuda')
        elif level == 3:
            return torch.tensor([
                252, 242, 213, 10, 5, 35, 29, 232, 172, 79, 170, 55, 245, 48,
                7, 179, 81, 8, 162, 4, 209, 2, 32, 10, 176, 11, 4, 15
            ], dtype=torch.uint8, device='cuda')

    @pytest.fixture(autouse=True)
    def expected_face_idx(self, level):
        if level == 3:
            return torch.tensor([
                0, 0, 0, 0, 0, 0, 3, 1, 0, 0, 0, 0, 1, 3, 3, 3, 3, 1, 1, 3, 1, 1,
                0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2,
                2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2
            ], device='cuda', dtype=torch.long)
        elif level == 1:
            return torch.tensor([0], device='cuda', dtype=torch.long)

    @pytest.fixture(autouse=True)
    def expected_bary_w(self, level):
        return torch.tensor([
            [4.5012e-08, 7.7766e-01],
            [2.8764e-01, 4.0506e-01],
            [3.5860e-08, 4.7760e-01],
            [1.0753e-01, 5.5666e-01],
            [2.5024e-08, 3.0500e-04],
            [5.3537e-03, 1.7265e-01],
            [2.4690e-01, 7.5310e-01],
            [8.8672e-01, 4.2031e-09],
            [4.0263e-01, 1.9203e-05],
            [6.0202e-01, 3.6483e-08],
            [2.2252e-01, 1.5161e-01],
            [4.3968e-01, 1.3058e-01],
            [7.3768e-01, 2.0286e-02],
            [6.2631e-01, 8.5154e-02],
            [2.8269e-01, 2.0198e-02],
            [7.7711e-08, 4.6322e-01],
            [9.7272e-08, 2.4475e-01],
            [6.3429e-01, 1.7624e-01],
            [4.4455e-01, 2.6633e-01],
            [1.8181e-01, 2.8813e-08],
            [7.0239e-01, 3.2221e-08],
            [5.5041e-01, 2.5328e-02],
            [1.7266e-01, 8.1010e-01],
            [1.7232e-08, 9.5489e-01],
            [5.0481e-01, 3.8403e-01],
            [6.9277e-01, 3.0723e-01],
            [3.2469e-01, 5.3563e-01],
            [2.7070e-08, 7.3333e-01],
            [1.4894e-01, 5.9743e-01],
            [5.6993e-09, 6.5052e-01],
            [8.0139e-01, 4.1631e-08],
            [1.0000e+00, 0.0000e+00],
            [2.5233e-01, 4.4148e-01],
            [2.5480e-01, 3.5643e-01],
            [6.5063e-02, 4.4652e-01],
            [3.6067e-01, 1.1542e-01],
            [1.7093e-01, 2.0551e-01],
            [1.7340e-01, 1.2046e-01],
            [4.2470e-08, 4.2354e-01],
            [2.3278e-08, 2.7855e-01],
            [4.6319e-08, 1.9574e-01],
            [9.2212e-01, 7.7879e-02],
            [7.2775e-01, 2.7225e-01],
            [6.1808e-01, 3.8192e-01],
            [4.2371e-01, 5.7629e-01],
            [8.7880e-01, 2.5213e-08],
            [7.0510e-01, 8.7381e-09],
            [6.1462e-01, 1.1334e-01],
            [4.1944e-01, 2.4449e-01],
            [3.7678e-01, 3.8143e-08],
            [3.8031e-08, 9.9842e-01],
            [2.2935e-01, 7.7065e-01],
            [1.1967e-01, 8.8033e-01],
            [0.0000e+00, 1.0000e+00],
            [2.2426e-01, 3.7564e-01],
            [2.0308e-01, 1.5850e-08],
            [2.3061e-02, 3.8613e-02],
            [3.9331e-01, 1.1329e-08],
            [2.5610e-01, 1.6592e-08],
            [2.0898e-01, 1.9427e-08],
            [7.1771e-02, 1.6657e-08],
            [1.1603e-01, 5.9735e-01],
            [1.1001e-01, 1.2918e-01],
            [2.4004e-08, 6.5422e-01],
            [2.3279e-08, 1.8040e-01]
        ], device='cuda', dtype=torch.float)

    def test_octree(self, face_vertices, level, expected_octree, expected_face_idx, expected_bary_w):
        octree, face_idx, bary_w =  unbatched_mesh_to_spc(face_vertices.squeeze(0), level)
        assert torch.equal(octree, expected_octree)
        assert torch.equal(face_idx, expected_face_idx)
        assert torch.allclose(bary_w, expected_bary_w, atol=1e-3, rtol=1e-3)

