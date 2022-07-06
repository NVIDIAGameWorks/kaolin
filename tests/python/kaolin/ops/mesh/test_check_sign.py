# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

import math
import pytest

import torch

from kaolin.utils.testing import FLOAT_TYPES
from kaolin.ops import mesh

@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
class TestCheckSign:

    @pytest.fixture(autouse=True)
    def verts(self, device, dtype):
        verts = torch.tensor(
            [[[1.,  0.,  0.],
              [1.,  0.,  1.],
              [1., -1., -1.],
              [1.,  1., -1.],

              [-1.,  0.,  0.],
              [-1.,  0., -4.],
              [-1., -4.,  4.],
              [-1.,  4.,  4.]]], device=device, dtype=dtype)
        # TODO(cfujitsang): fix cpu and extend test with torch.flip(verts, dims=(-1,))
        return torch.cat([verts, -verts], dim=0)


    @pytest.fixture(autouse=True)
    def faces(self, device):
        return torch.tensor(
            [[0, 1, 2],
             [0, 2, 3],
             [0, 3, 1],

             [1, 2, 6],
             [2, 3, 5],
             [3, 1, 7],
             
             [5, 6, 2],
             [6, 7, 1],
             [7, 5, 3],

             [4, 6, 5],
             [4, 7, 6],
             [4, 5, 7]], device=device, dtype=torch.long)

    @pytest.fixture(autouse=True)
    def points(self, device, dtype):
        points = torch.tensor([[
            # Inside
            [ 0.9,   0.,   0. ], # inside overlap with vertices 0 and 4
            [ 0.9,   0.,  -1. ], # inside overlap with edges 2-3, and 4-5 
            [ 0.9,   0.,  -0.9], # inside overlap with edge 4-5
            [ 0.9,   0.1, -1.0], # inside overlap with edge 2-3
            [ 0.9,   0.,   1. ], # inside overlap with vertice 1
            [ 0.9,  -1.,  -1. ], # inside overlap with vertice 2
            [ 0.9,   1.,  -1. ], # inside overlap with vertice 3
            [-0.99,  0.,  -3.9], # inside near vertice 5
            [-0.99, -3.9,  3.9], # inside near vertice 6
            [-0.99,  3.9,  3.9], # inside near vertice 7
            # Outside
            [ 0.9,   0.,  -4. ], # outside overlap with 5
            [ 0.9,  -4.,   4. ], # outside overlap with 6
            [ 0.9,   4.,   4. ], # outside overlap with 7
            [ 0.9,   0.,   4. ], # outside overlap with edge 6-7
            [-0.9,   0.,  -3.9], # outside aligned with edge 2-3 and overlap with 4-5
            [-0.9,  -3.9,  3.9], # outside near vertice 6
            [-0.9,   3.9,  3.9], # outside near vertice 7
            [ 0.5,   0.,   5. ], # outside aligned with edges 2-3 and 4-5
            [ 0.5,  -5.,   4. ], # outside aligned with edge 6-7
            [ 1.1,   0.,   0. ], # in front overlap with vertice 0 and 4
            [ 1.1,   0.,  -1. ], # in front overlap with edges 2-3 and 4-5
            [ 1.1,   0.,  -0.9], # in front overlap with edge 4-5
            [ 1.1,   0.1, -1.0], # in front overlap with edge 2-3
            [ 1.1,   0.,   1. ], # in front overlap with vertice 1
            [ 1.1,  -1.,  -1. ], # in front overlap with vertice 2
            [ 1.1,   1.,  -1. ], # in front overlap with vertice 3
            [-1.1,   0.,   0. ], # behind overlap with vertice 0 and 4
            [-1.1,   0.,  -1. ], # behind overlap with edges 2-3 and 4-5
            [-1.1,   0.,  -0.9], # behind overlap with edge 4-5
            [-1.1,   0.1, -1.0], # behind overlap with edge 2-3
            [-1.1,   0.,   1. ], # behind overlap with vertice 1
            [-1.1,  -1.,  -1. ], # behind overlap with vertice 2
            [-1.1,   1.,  -1. ], # behind overlap with vertice 3
        ]], device=device, dtype=dtype)
        return torch.cat([
            points, torch.flip(-points, dims=(1,))], dim=0)


    @pytest.fixture(autouse=True)
    def expected(self, device):
        expected = torch.tensor(
            [[True,  True,  True,  True,  True,  True,  True,  True,  True,  True,
              False, False, False, False, False, False, False, False, False, False,
              False, False, False, False, False, False, False, False, False, False,
              False, False, False]], device=device)
        return torch.cat([expected, torch.flip(expected, dims=(1,))], dim=0)

    def test_faces_type(self, verts, faces, points):
        with pytest.raises(TypeError,
                           match=r"Expected faces entries to be torch.int64 "
                                 r"but got torch.int32."):
            faces = faces.int()
            mesh.check_sign(verts, faces, points)

    def test_hash_resolution_type(self, verts, faces, points):
        with pytest.raises(TypeError,
                           match=r"Expected hash_resolution to be int "
                                 r"but got <class 'float'>."):
            mesh.check_sign(verts, faces, points, 512.0)

    def test_verts_ndim(self, verts, faces, points):
        with pytest.raises(ValueError,
                           match=r"Expected verts to have 3 dimensions "
                                 r"but got 4 dimensions."):
            verts = verts.unsqueeze(-1)
            mesh.check_sign(verts, faces, points)

    def test_faces_ndim(self, verts, faces, points):
        with pytest.raises(ValueError,
                           match=r"Expected faces to have 2 dimensions "
                                 r"but got 3 dimensions."):
            faces = faces.unsqueeze(-1)
            mesh.check_sign(verts, faces, points)

    def test_points_ndim(self, verts, faces, points):
        with pytest.raises(ValueError,
                           match=r"Expected points to have 3 dimensions "
                                 r"but got 4 dimensions."):
            points = points.unsqueeze(-1)
            mesh.check_sign(verts, faces, points)

    def test_verts_shape(self, verts, faces, points):
        with pytest.raises(ValueError,
                           match=r"Expected verts to have 3 coordinates "
                                 r"but got 2 coordinates."):
            verts = verts[...,:2]
            mesh.check_sign(verts, faces, points)

    def test_faces_shape(self, verts, faces, points):
        with pytest.raises(ValueError,
                           match=r"Expected faces to have 3 vertices "
                                 r"but got 2 vertices."):
            faces = faces[:,:2]
            mesh.check_sign(verts, faces, points)

    def test_points_shape(self, verts, faces, points):
        with pytest.raises(ValueError,
                           match=r"Expected points to have 3 coordinates "
                                 r"but got 2 coordinates."):
            points = points[...,:2]
            mesh.check_sign(verts, faces, points)

    def test_single_batch(self, verts, faces, points, expected):
        output = mesh.check_sign(verts[:1], faces, points[:1])
        diff_idxs = torch.where(output != expected[:1])
        assert(torch.equal(output, expected[:1]))

    def test_meshes(self, verts, faces, points, expected):
        output = mesh.check_sign(verts, faces, points)
        assert(torch.equal(output, expected))

    def test_faces_with_zero_area(self, verts, faces, points, expected):
        faces = torch.cat([faces, torch.tensor([[1, 1, 1],
                              [0, 0, 0],
                              [2, 2, 2],
                              [3, 3, 3]]).to(faces.device)])
        output = mesh.check_sign(verts, faces, points)
        assert(torch.equal(output, expected))

