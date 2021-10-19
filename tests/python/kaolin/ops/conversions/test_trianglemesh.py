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

from kaolin.ops.conversions import trianglemeshes_to_voxelgrids
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
