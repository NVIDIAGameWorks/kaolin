# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.
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

import random
import pytest

import numpy as np
import torch
import math
import os

from kaolin.render.camera import perspective_camera, rotate_translate_points
from kaolin.render.mesh import deftet_sparse_render
from kaolin.render.mesh.deftet import _naive_deftet_sparse_render
import kaolin as kal
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, os.pardir, os.pardir, os.pardir, os.pardir, 'samples/')

@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
class TestSimpleDeftetSparseRender:
    @pytest.fixture(autouse=True)
    def face_vertices_image(self, device, dtype):
        # Mesh 0:
        # three faces: (no intersection)
        # - two fully overlapped on left side (not same normal)
        # - one only overlapping two corners on right side

        # Mesh 1:
        # three faces, fully overlapped (will have intersection)
        return torch.tensor(
            [[[[-1.,  0. ], [0., -1.], [ 0.,  1.]],
              [[-1.,  0. ], [0.,  1.], [ 0., -1.]],
              [[ 0., -1.],  [0.,  1.], [ 1.,  0.]]],
             [[[-1., -1.],  [1., -1.], [-1.,  1.]],
              [[-1., -1.],  [1., -1.], [-1.,  1.]],
              [[-1., -1.],  [1., -1.], [-1.,  1.]]]],
            device=device, dtype=dtype)

    @pytest.fixture(autouse=True)
    def face_vertices_z(self, device, dtype):
        # Mesh 0:
        # The face on the right side is in-between
        # the two faces on the left side

        # Mesh 1:
        # the three faces are intersecting
        return torch.tensor(
            [[[-2.,  -1., -1.],
              [-2.5, -3., -3.],
              [-2.,  -2., -2.]],
             [[-2., -1., -3.],
              [-2., -2., -2.],
              [-2., -3., -1.]]],
            device=device, dtype=dtype)

    @pytest.fixture(autouse=True)
    def face_features(self, device, dtype):
        features_per_face = torch.tensor(
            [[[[0.], [0.], [0.]],
              [[1.], [1.], [1.]],
              [[2.], [2.], [2.]]],
             [[[3.], [3.], [3.]],
              [[4.], [4.], [4.]],
              [[5.], [5.], [5.]]]],
            device=device, dtype=dtype)
        features_per_vertice = torch.tensor(
            [[[[0.], [1.], [2.]],
              [[3.], [4.], [5.]],
              [[6.], [7.], [8.]]],
             [[[9.], [10.], [11.]],
              [[12.], [13.], [14.]],
              [[15.], [16.], [17.]]]],
            device=device, dtype=dtype)
        return [features_per_face, features_per_vertice]

    @pytest.fixture(autouse=True)
    def pixel_coords(self, device, dtype):
        # slightly shifting coords to stay away from corner cases
        return torch.tensor(
            [[[-0.999, 0.], [-0.001, -0.998], [0.001, 0.998], [0.999, 0.], # corners
              [-0.45, 0.], [0.45, 0.], # centers
              [-0.999, -0.999]], # void
             [[-0.998, -0.999], [0.998, -0.999], [-0.999, 0.998], # corners
              [-0.001, -0.], [0., -0.999], [-0.999, 0.], # center of edges
              [0.001, 0.001]]], # void
            device=device, dtype=dtype)

    @pytest.mark.parametrize('cat_features', [False, True])
    @pytest.mark.parametrize('use_naive', [False, True])
    def test_full_render(self, pixel_coords, face_vertices_image, face_vertices_z,
                         face_features, device, dtype, use_naive, cat_features):
        render_ranges = torch.tensor([[[-4., 0.]]], device='cuda',
                                     dtype=dtype).repeat(2, 7, 1)
        if cat_features:
            face_features = torch.cat(face_features, dim=-1)
        if use_naive:
            interpolated_features, face_idx = _naive_deftet_sparse_render(
                pixel_coords, render_ranges, face_vertices_z,
                face_vertices_image, face_features, 5)
        else:
            interpolated_features, face_idx = deftet_sparse_render(
                pixel_coords, render_ranges, face_vertices_z,
                face_vertices_image, face_features, 5)
        gt_face_idx = torch.tensor(
            [[[0,  1, -1, -1, -1],
              [0,  1, -1, -1, -1],
              [2, -1, -1, -1, -1],
              [2, -1, -1, -1, -1],
              [0,  1, -1, -1, -1],
              [2, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1]],
             [[0, 1, 2, -1, -1],
              [0, 1, 2, -1, -1],
              [2, 1, 0, -1, -1],
              [2, 1, 0, -1, -1],
              [0, 1, 2, -1, -1],
              [2, 1, 0, -1, -1],
              [-1, -1, -1, -1, -1]]],
            device=device, dtype=torch.long)
        assert torch.equal(face_idx, gt_face_idx)
        gt_interpolated_features0 = (
            gt_face_idx + torch.arange(2, device=device).view(2, 1, 1) * face_vertices_image.shape[1]
        ).to(dtype).unsqueeze(-1)
        gt_interpolated_features0[gt_face_idx == -1] = 0.
        gt_interpolated_features1 = torch.tensor(
            [[[[0.],    [3.],    [0.],   [0.], [0.]],
              [[1.],    [5.],    [0.],   [0.], [0.]],
              [[7.],    [0.],    [0.],   [0.], [0.]],
              [[8.],    [0.],    [0.],   [0.], [0.]],
              [[0.825], [3.825], [0.],   [0.], [0.]],
              [[7.175], [0.],    [0.],   [0.], [0.]],
              [[0.],    [0.],    [0.],   [0.], [0.]]],
             [[[9.],    [12.],   [15.],  [0.], [0.]],
              [[10.],   [13.],   [16.],  [0.], [0.]],
              [[17.],   [14.],   [11.],  [0.], [0.]],
              [[16.5],  [13.5],  [10.5], [0.], [0.]],
              [[9.5],   [12.5],  [15.5], [0.], [0.]],
              [[16.],   [13.],   [10.],  [0.], [0.]],
              [[0.],    [0.],    [0.],   [0.], [0.]]]],
            device=device, dtype=dtype)

        if cat_features:
            gt_interpolated_features = torch.cat(
                [gt_interpolated_features0, gt_interpolated_features1], dim=-1)
            assert torch.allclose(interpolated_features, gt_interpolated_features,
                                  atol=3e-3, rtol=1e-5)
        else:
            assert torch.allclose(interpolated_features[0], gt_interpolated_features0)
            assert torch.allclose(interpolated_features[1], gt_interpolated_features1,
                                  atol=3e-3, rtol=1e-5)

    @pytest.mark.parametrize('cat_features', [False, True])
    @pytest.mark.parametrize('use_naive', [False, True])
    def test_restricted_range(self, pixel_coords, face_vertices_image, face_vertices_z,
                              face_features, device, dtype, use_naive, cat_features):
        render_ranges = torch.tensor([[[-2.1, 0.]]], device='cuda',
                                     dtype=dtype).repeat(2, 7, 1)
        if cat_features:
            face_features = torch.cat(face_features, dim=-1)
        if use_naive:
            interpolated_features, face_idx = _naive_deftet_sparse_render(
                pixel_coords, render_ranges, face_vertices_z,
                face_vertices_image, face_features, 5)
        else:
            interpolated_features, face_idx = deftet_sparse_render(
                pixel_coords, render_ranges, face_vertices_z,
                face_vertices_image, face_features, 5)

        gt_face_idx = torch.tensor(
            [[[0, -1, -1, -1, -1],
              [0, -1, -1, -1, -1],
              [2, -1, -1, -1, -1],
              [2, -1, -1, -1, -1],
              [0, -1, -1, -1, -1],
              [2, -1, -1, -1, -1],
              [-1, -1, -1, -1, -1]],
             [[0, 1, 2, -1, -1],
              [0, 1, -1, -1, -1],
              [2, 1, -1, -1, -1],
              [2, 1, 0, -1, -1],
              [0, 1, -1, -1, -1],
              [2, 1, -1, -1, -1],
              [-1, -1, -1, -1, -1]]],
            device=device, dtype=torch.long)
        assert torch.equal(face_idx, gt_face_idx)
        gt_interpolated_features0 = (
            gt_face_idx + torch.arange(2, device=device).view(2, 1, 1) * face_vertices_image.shape[1]
        ).to(dtype).unsqueeze(-1)
        gt_interpolated_features0[gt_face_idx == -1] = 0.
        gt_interpolated_features1 = torch.tensor(
            [[[[0.],    [0.],   [0.],   [0.], [0.]],
              [[1.],    [0.],   [0.],   [0.], [0.]],
              [[7.],    [0.],   [0.],   [0.], [0.]],
              [[8.],    [0.],   [0.],   [0.], [0.]],
              [[0.825], [0.],   [0.],   [0.], [0.]],
              [[7.175], [0.],   [0.],   [0.], [0.]],
              [[0.],    [0.],   [0.],   [0.], [0.]]],
             [[[9.],    [12.],  [15.],  [0.], [0.]],
              [[10.],   [13.],  [0.],   [0.], [0.]],
              [[17.],   [14.],  [0.],   [0.], [0.]],
              [[16.5],  [13.5], [10.5], [0.], [0.]],
              [[9.5],   [12.5], [0.],   [0.], [0.]],
              [[16.],   [13.],  [0.],   [0.], [0.]],
              [[0.],    [0.],   [0.],   [0.], [0.]]]],
            device=device, dtype=dtype)

        if cat_features:
            gt_interpolated_features = torch.cat(
                [gt_interpolated_features0, gt_interpolated_features1], dim=-1)
            assert torch.allclose(interpolated_features, gt_interpolated_features,
                                  atol=3e-3, rtol=1e-5)
        else:
            assert torch.allclose(interpolated_features[0], gt_interpolated_features0)
            assert torch.allclose(interpolated_features[1], gt_interpolated_features1,
                                  atol=3e-3, rtol=1e-5)

    @pytest.mark.parametrize('cat_features', [False, True])
    def test_only_closest(self, pixel_coords, face_vertices_image, face_vertices_z,
                          face_features, device, dtype, cat_features):
        """Equivalent to rasterization"""
        render_ranges = torch.tensor([[[-4., 0.]]], device='cuda',
                                     dtype=dtype).repeat(2, 7, 1)
        if cat_features:
            face_features = torch.cat(face_features, dim=-1)
        interpolated_features, face_idx = _naive_deftet_sparse_render(
            pixel_coords, render_ranges, face_vertices_z,
            face_vertices_image, face_features, 1)
        gt_face_idx = torch.tensor(
            [[[0], [0], [2], [2], [0], [2], [-1]],
             [[0], [0], [2], [2], [0], [2], [-1]]],
            device=device, dtype=torch.long)
        assert torch.equal(face_idx, gt_face_idx)
        gt_interpolated_features0 = (
            gt_face_idx + torch.arange(2, device=device).view(2, 1, 1) * face_vertices_image.shape[1]
        ).to(dtype).unsqueeze(-1)
        gt_interpolated_features0[gt_face_idx == -1] = 0.
        gt_interpolated_features1 = torch.tensor(
            [[[[0.]], [[1.]], [[7.]], [[8.]], [[0.825]], [[7.175]], [[0.]]],
             [[[9.]], [[10.]], [[17.]], [[16.5]], [[9.5]], [[16.]], [[0.]]]],
            device=device, dtype=dtype)

        if cat_features:
            gt_interpolated_features = torch.cat(
                [gt_interpolated_features0, gt_interpolated_features1], dim=-1)
            assert torch.allclose(interpolated_features, gt_interpolated_features,
                                  atol=3e-3, rtol=1e-5)
        else:
            assert torch.allclose(interpolated_features[0], gt_interpolated_features0)
            assert torch.allclose(interpolated_features[1], gt_interpolated_features1,
                                  atol=3e-3, rtol=1e-5)

    @pytest.mark.parametrize('cat_features', [False, True])
    def test_with_valid_faces(self, pixel_coords, face_vertices_image, face_vertices_z,
                              face_features, device, dtype, cat_features):
        render_ranges = torch.tensor([[[-4., 0.]]], device='cuda',
                                     dtype=dtype).repeat(2, 7, 1)
        valid_faces = torch.tensor([[False, True, True], [True, False, True]], device='cuda',
                                   dtype=torch.bool)
        if cat_features:
            face_features = torch.cat(face_features, dim=-1)
        interpolated_features, face_idx = _naive_deftet_sparse_render(
            pixel_coords, render_ranges, face_vertices_z,
            face_vertices_image, face_features, 3, valid_faces=valid_faces)

        gt_face_idx = torch.tensor(
            [[[1, -1, -1],
              [1, -1, -1],
              [2, -1, -1],
              [2, -1, -1],
              [1, -1, -1],
              [2, -1, -1],
              [-1, -1, -1]],
             [[0, 2, -1],
              [0, 2, -1],
              [2, 0, -1],
              [2, 0, -1],
              [0, 2, -1],
              [2, 0, -1],
              [-1, -1, -1]]],
            device=device, dtype=torch.long)
        assert torch.equal(face_idx, gt_face_idx)

        gt_interpolated_features0 = (
            gt_face_idx + torch.arange(2, device=device).view(2, 1, 1) * face_vertices_image.shape[1]
        ).to(dtype).unsqueeze(-1)
        gt_interpolated_features0[gt_face_idx == -1] = 0.

        gt_interpolated_features1 = torch.tensor(
            [[[[3.],    [0.],   [0.]],
              [[5.],    [0.],   [0.]],
              [[7.],    [0.],   [0.]],
              [[8.],    [0.],   [0.]],
              [[3.825], [0.],   [0.]],
              [[7.175], [0.],   [0.]],
              [[0.],    [0.],   [0.]]],
             [[[9.],    [15.],  [0.]],
              [[10.],   [16.],  [0.]],
              [[17.],   [11.],  [0.]],
              [[16.5],  [10.5], [0.]],
              [[9.5],   [15.5], [0.]],
              [[16.],   [10.],  [0.]],
              [[0.],    [0.],   [0.]]]],
            device=device, dtype=dtype)

        if cat_features:
            gt_interpolated_features = torch.cat(
                [gt_interpolated_features0, gt_interpolated_features1], dim=-1)
            assert torch.allclose(interpolated_features, gt_interpolated_features,
                                  atol=3e-3, rtol=1e-5)
        else:
            assert torch.allclose(interpolated_features[0], gt_interpolated_features0)
            assert torch.allclose(interpolated_features[1], gt_interpolated_features1,
                                  atol=3e-3, rtol=1e-5)


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("num_pixels", [1, 31, 1025])
@pytest.mark.parametrize("render_up_to_center", [True, False])
class TestDeftetSparseRender:
    @pytest.fixture(autouse=True)
    def mesh(self):
        mesh = kal.io.obj.import_mesh(os.path.join(MODEL_DIR, 'model.obj'),
                                      with_materials=True)
        return mesh

    @pytest.fixture(autouse=True)
    def faces(self, mesh):
        return mesh.faces.cuda()

    @pytest.fixture(autouse=True)
    def camera_pos(self, batch_size, dtype):
        return torch.tensor([[0.5, 0.5, 3.],
                             [2., 2., -2.],
                             [3., 0.5, 0.5]],
                            device='cuda', dtype=dtype)[:batch_size]

    @pytest.fixture(autouse=True)
    def look_at(self, batch_size, dtype):
        return torch.full((batch_size, 3), 0.5, device='cuda',
                          dtype=dtype)

    @pytest.fixture(autouse=True)
    def camera_up(self, batch_size, dtype):
        return torch.tensor([[0., 1., 0.]], device='cuda',
                            dtype=dtype).repeat(batch_size, 1)

    @pytest.fixture(autouse=True)
    def camera_proj(self, dtype):
        return kal.render.camera.generate_perspective_projection(
            fovyangle=math.pi / 4., dtype=dtype).cuda()

    @pytest.fixture(autouse=True)
    def vertices_camera(self, mesh, camera_pos, look_at, camera_up, dtype):
        vertices = mesh.vertices.to('cuda', dtype).unsqueeze(0)
        min_vertices = vertices.min(dim=1, keepdims=True)[0]
        max_vertices = vertices.max(dim=1, keepdims=True)[0]
        vertices = (vertices - min_vertices) / (max_vertices - min_vertices)
        camera_rot, camera_trans = kal.render.camera.generate_rotate_translate_matrices(
            camera_pos, look_at, camera_up)
        return kal.render.camera.rotate_translate_points(
            vertices, camera_rot, camera_trans)

    @pytest.fixture(autouse=True)
    def vertices_image(self, vertices_camera, camera_proj):
        return kal.render.camera.perspective_camera(
            vertices_camera, camera_proj)

    @pytest.fixture(autouse=True)
    def face_vertices_z(self, vertices_camera, faces):
        return kal.ops.mesh.index_vertices_by_faces(
            vertices_camera[:, :, -1:], faces).squeeze(-1)

    @pytest.fixture(autouse=True)
    def face_vertices_image(self, vertices_image, faces):
        return kal.ops.mesh.index_vertices_by_faces(
            vertices_image, faces)

    @pytest.fixture(autouse=True)
    def texture_map(self, mesh, dtype):
        return mesh.materials[0]['map_Kd'].to('cuda', dtype).permute(
            2, 0, 1).unsqueeze(0) / 255.

    @pytest.fixture(autouse=True)
    def face_uvs(self, mesh, batch_size, dtype):
        return kal.ops.mesh.index_vertices_by_faces(
            mesh.uvs.unsqueeze(0).to('cuda', dtype),
            mesh.face_uvs_idx.cuda()).repeat(batch_size, 1, 1, 1)

    @pytest.fixture(autouse=True)
    def pixel_coords(self, batch_size, num_pixels, dtype):
        return torch.rand((batch_size, num_pixels, 2), device='cuda',
                          dtype=dtype) * 2. - 1.

    @pytest.fixture(autouse=True)
    def render_ranges(self, vertices_camera, render_up_to_center, num_pixels):
        min_z = vertices_camera[:, :, -1].min(dim=1)[0]
        max_z = vertices_camera[:, :, -1].max(dim=1)[0]
        min_render_range = (min_z + max_z) / 2. if render_up_to_center else \
                           min_z
        render_range = torch.nn.functional.pad(min_render_range.unsqueeze(-1), (0, 1),
                                               value=0.)
        return render_range.unsqueeze(1).repeat(1, num_pixels, 1)

    @pytest.mark.parametrize('knum', [20, 30])
    def test_forward(self, pixel_coords, render_ranges, face_vertices_z,
                     face_vertices_image, face_uvs, knum):
        interpolated_features, face_idx = deftet_sparse_render(
            pixel_coords, render_ranges, face_vertices_z,
            face_vertices_image, face_uvs, knum)
        gt_interpolated_features, gt_face_idx = _naive_deftet_sparse_render(
            pixel_coords, render_ranges, face_vertices_z,
            face_vertices_image, face_uvs, knum)
        assert torch.equal(face_idx, gt_face_idx)
        assert torch.allclose(interpolated_features,
                              gt_interpolated_features,
                              rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize('knum', [20, 30])
    def test_forward_with_mask(self, pixel_coords, render_ranges,
                               face_vertices_z, face_vertices_image,
                               face_uvs, knum):
        face_mask = torch.ones_like(face_uvs[:, :, :, :1])
        interpolated_features, face_idx = deftet_sparse_render(
            pixel_coords, render_ranges, face_vertices_z,
            face_vertices_image, [face_uvs, face_mask], knum)
        gt_interpolated_features, gt_face_idx = _naive_deftet_sparse_render(
            pixel_coords, render_ranges, face_vertices_z,
            face_vertices_image, [face_uvs, face_mask], knum)
        assert torch.equal(face_idx, gt_face_idx)
        assert torch.allclose(interpolated_features[0],
                              gt_interpolated_features[0],
                              rtol=1e-4, atol=1e-4)
        assert torch.allclose(interpolated_features[0],
                              gt_interpolated_features[0],
                              rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize('knum', [20, 30])
    def test_backward(self, pixel_coords, render_ranges, face_vertices_z,
                      face_vertices_image, face_uvs, knum):
        pixel_coords = pixel_coords.detach()
        pixel_coords.requires_grad = True
        render_ranges = render_ranges.detach()
        render_ranges.requires_grad = True
        face_vertices_z = face_vertices_z.detach()
        face_vertices_z.requires_grad = True
        face_vertices_image = face_vertices_image.detach()
        face_vertices_image.requires_grad = True
        face_uvs = face_uvs.detach()
        face_uvs.requires_grad = True
        pixel_coords2 = pixel_coords.detach()
        pixel_coords2.requires_grad = True
        render_ranges2 = render_ranges.detach()
        render_ranges2.requires_grad = True
        face_vertices_z2 = face_vertices_z.detach()
        face_vertices_z2.requires_grad = True
        face_vertices_image2 = face_vertices_image.detach()
        face_vertices_image2.requires_grad = True
        face_uvs2 = face_uvs.detach()
        face_uvs2.requires_grad = True

        interpolated_features, face_idx = deftet_sparse_render(
            pixel_coords, render_ranges, face_vertices_z,
            face_vertices_image, face_uvs, knum)
        gt_interpolated_features, gt_face_idx = _naive_deftet_sparse_render(
            pixel_coords2, render_ranges2, face_vertices_z2,
            face_vertices_image2, face_uvs2, knum)

        grad_out = torch.rand_like(interpolated_features)
        interpolated_features.backward(grad_out)
        gt_interpolated_features.backward(grad_out)

        assert pixel_coords.grad is None or torch.all(pixel_coords.grad == 0.)
        assert render_ranges.grad is None or torch.all(render_ranges.grad == 0.)
        assert face_vertices_z.grad is None or torch.all(face_vertices_z.grad == 0.)
        assert torch.allclose(face_vertices_image.grad,
                              face_vertices_image2.grad,
                              rtol=5e-3, atol=5e-3)
        assert torch.allclose(face_uvs.grad,
                              face_uvs2.grad,
                              rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('knum', [20, 30])
    def test_backward_with_mask(self, pixel_coords, render_ranges,
                                face_vertices_z, face_vertices_image,
                                face_uvs, knum):
        pixel_coords = pixel_coords.detach()
        pixel_coords.requires_grad = True
        render_ranges = render_ranges.detach()
        render_ranges.requires_grad = True
        face_vertices_z = face_vertices_z.detach()
        face_vertices_z.requires_grad = True
        face_vertices_image = face_vertices_image.detach()
        face_vertices_image.requires_grad = True
        face_uvs = face_uvs.detach()
        face_uvs.requires_grad = True
        face_mask = torch.ones_like(face_uvs[:, :, :, -1:],
                                    requires_grad=True)
        pixel_coords2 = pixel_coords.detach()
        pixel_coords2.requires_grad = True
        render_ranges2 = render_ranges.detach()
        render_ranges2.requires_grad = True
        face_vertices_z2 = face_vertices_z.detach()
        face_vertices_z2.requires_grad = True
        face_vertices_image2 = face_vertices_image.detach()
        face_vertices_image2.requires_grad = True
        face_uvs2 = face_uvs.detach()
        face_uvs2.requires_grad = True
        face_mask2 = torch.ones_like(face_uvs2[:, :, :, -1:],
                                     requires_grad=True)

        interpolated_features, face_idx = deftet_sparse_render(
            pixel_coords, render_ranges, face_vertices_z,
            face_vertices_image, [face_uvs, face_mask], knum)
        gt_interpolated_features, gt_face_idx = _naive_deftet_sparse_render(
            pixel_coords2, render_ranges2, face_vertices_z2,
            face_vertices_image2, [face_uvs2, face_mask2], knum)

        interpolated_features = torch.cat(interpolated_features, dim=-1)
        gt_interpolated_features = torch.cat(gt_interpolated_features, dim=-1)
        grad_out = torch.rand_like(interpolated_features)
        interpolated_features.backward(grad_out)
        gt_interpolated_features.backward(grad_out)

        assert pixel_coords.grad is None or torch.all(pixel_coords.grad == 0.)
        assert render_ranges.grad is None or torch.all(render_ranges.grad == 0.)
        assert face_vertices_z.grad is None or torch.all(face_vertices_z.grad == 0.)
        assert torch.allclose(face_vertices_image.grad,
                              face_vertices_image2.grad,
                              rtol=5e-2, atol=5e-2)
        assert torch.allclose(face_uvs.grad,
                              face_uvs2.grad,
                              rtol=1e-3, atol=1e-3)
        assert torch.allclose(face_mask.grad,
                              face_mask2.grad,
                              rtol=1e-3, atol=1e-3)
