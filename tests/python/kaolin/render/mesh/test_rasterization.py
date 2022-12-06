# Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
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
from kaolin.render.mesh import rasterize
from kaolin.render.mesh.deftet import _naive_deftet_sparse_render
import kaolin as kal

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, os.pardir, os.pardir, os.pardir, os.pardir, 'samples/')

@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("height,width", [(35, 31)])
@pytest.mark.parametrize("flip", [False, True])
class TestRasterize:
    @pytest.fixture(autouse=True)
    def mesh(self):
        mesh = kal.io.obj.import_mesh(os.path.join(MODEL_DIR, 'model.obj'),
                                      with_materials=True)
        return mesh

    @pytest.fixture(autouse=True)
    def faces(self, mesh, flip):
        out = mesh.faces.cuda()
        if flip:
            out = torch.flip(out, dims=(-1,))
        return out

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
    def valid_faces(self, batch_size, face_vertices_z):
        min_z = face_vertices_z.reshape(batch_size, -1).min(dim=1, keepdims=True)[0]
        max_z = face_vertices_z.reshape(batch_size, -1).max(dim=1, keepdims=True)[0]
        middle_z = (min_z + max_z) / 2.
        return torch.all(face_vertices_z < middle_z.unsqueeze(-1), dim=-1)

    @pytest.fixture(autouse=True)
    def face_vertices_image(self, vertices_image, faces):
        return kal.ops.mesh.index_vertices_by_faces(
            vertices_image, faces)

    @pytest.fixture(autouse=True)
    def texture_map(self, mesh, dtype):
        return mesh.materials[0]['map_Kd'].to('cuda', dtype).permute(
            2, 0, 1).unsqueeze(0) / 255.

    @pytest.fixture(autouse=True)
    def face_uvs(self, mesh, batch_size, dtype, flip):
        face_uvs_idx = mesh.face_uvs_idx.cuda()
        if flip:
            face_uvs_idx = torch.flip(face_uvs_idx, dims=(-1,))
        return kal.ops.mesh.index_vertices_by_faces(
            mesh.uvs.unsqueeze(0).to('cuda', dtype),
            face_uvs_idx).repeat(batch_size, 1, 1, 1)

    @pytest.fixture(autouse=True)
    def pixel_coords(self, batch_size, height, width, dtype):
        x = (2 * torch.arange(width, device='cuda', dtype=dtype) + 1 - width) / width
        y = (height - 2 * torch.arange(height, device='cuda', dtype=dtype) - 1.) / height
        return torch.stack([
            x.reshape(1, 1, -1).repeat(batch_size, height, 1),
            y.reshape(1, -1, 1).repeat(batch_size, 1, width)
        ], dim=-1).reshape(batch_size, -1, 2)

    @pytest.fixture(autouse=True)
    def render_ranges(self, vertices_camera, height, width):
        min_z = vertices_camera[:, :, -1].min(dim=1)[0]
        max_z = vertices_camera[:, :, -1].max(dim=1)[0]
        render_range = torch.stack([min_z - 1e-2, max_z + 1e-2], dim=-1)

        return render_range.unsqueeze(1).repeat(1, height * width, 1)

    @pytest.mark.parametrize('with_valid_faces', [False, True])
    def test_cuda_forward(self, batch_size, height, width, pixel_coords,
                          render_ranges, face_vertices_z, face_vertices_image,
                          face_uvs, with_valid_faces, valid_faces):
        kwargs = {}
        if with_valid_faces:
            kwargs['valid_faces'] = valid_faces
        face_attr = face_uvs

        interpolated_features, face_idx = rasterize(
            height, width, face_vertices_z, face_vertices_image,
            face_attr, backend='cuda', **kwargs)
        gt_interpolated_features, gt_face_idx = _naive_deftet_sparse_render(
            pixel_coords, render_ranges, face_vertices_z,
            face_vertices_image, face_attr, 1, **kwargs)

        assert torch.equal(face_idx, gt_face_idx.reshape(batch_size, height, width))
        assert torch.allclose(
            interpolated_features,
            gt_interpolated_features.reshape(batch_size, height, width, face_uvs.shape[-1]),
            rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize('with_valid_faces', [False, True])
    def test_cuda_forward_with_list(
            self, batch_size, height, width, pixel_coords,
            render_ranges, face_vertices_z, face_vertices_image,
            face_uvs, with_valid_faces, valid_faces):
        """Test with list of tensors as features"""
        kwargs = {}
        if with_valid_faces:
            kwargs['valid_faces'] = valid_faces
        face_attr = [face_uvs, torch.ones_like(face_uvs[..., 1:])]

        (uvs_map, mask), face_idx = rasterize(
            height, width, face_vertices_z, face_vertices_image,
            face_attr, backend='cuda', **kwargs)
        (gt_uvs_map, gt_mask), gt_face_idx = _naive_deftet_sparse_render( 
            pixel_coords, render_ranges, face_vertices_z,
            face_vertices_image, face_attr, 1, **kwargs)

        assert torch.equal(face_idx, gt_face_idx.reshape(batch_size, height, width))
        assert torch.allclose(
            uvs_map,
            gt_uvs_map.reshape(batch_size, height, width, face_uvs.shape[-1]),
            rtol=1e-5, atol=1e-5
        )
        assert torch.allclose(
            mask,
            gt_mask.reshape(batch_size, height, width, 1),
            rtol=1e-5, atol=1e-5
        )

    @pytest.mark.parametrize('with_valid_faces', [False, True])
    def test_cuda_backward(self, batch_size, height, width, pixel_coords,
                           render_ranges, face_vertices_z, face_vertices_image,
                           face_uvs, with_valid_faces, valid_faces):
        kwargs = {}
        if with_valid_faces:
            kwargs['valid_faces'] = valid_faces
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

        interpolated_features, face_idx = rasterize(
            height, width, face_vertices_z, face_vertices_image,
            face_uvs, backend='cuda')
        gt_interpolated_features, gt_face_idx = _naive_deftet_sparse_render(
            pixel_coords2, render_ranges2, face_vertices_z2,
            face_vertices_image2, face_uvs2, 1)
        gt_interpolated_features = gt_interpolated_features.reshape(
            batch_size, height, width, face_uvs.shape[-1])

        grad_out = torch.rand_like(interpolated_features)
        interpolated_features.backward(grad_out)
        gt_interpolated_features.backward(grad_out)

        assert face_vertices_z.grad is None or torch.all(face_vertices_z.grad == 0.)
        assert torch.allclose(face_vertices_image.grad,
                              face_vertices_image2.grad,
                              rtol=1e-3, atol=1e-2)
        assert torch.allclose(face_uvs.grad,
                              face_uvs2.grad,
                              rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('with_valid_faces', [False, True])
    def test_cuda_backward_with_list(
            self, batch_size, height, width, pixel_coords,
            render_ranges, face_vertices_z, face_vertices_image,
            face_uvs, with_valid_faces, valid_faces):
        """Test with list of tensors as features"""
        kwargs = {}
        if with_valid_faces:
            kwargs['valid_faces'] = valid_faces
        face_vertices_z = face_vertices_z.detach()
        face_vertices_z.requires_grad = True
        face_vertices_image = face_vertices_image.detach()
        face_vertices_image.requires_grad = True
        face_uvs = face_uvs.detach()
        face_uvs.requires_grad = True
        face_mask = torch.ones_like(face_uvs[..., :1], requires_grad=True)
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
        face_mask2 = face_mask.detach()
        face_mask2.requires_grad = True

        interpolated_features, face_idx = rasterize(
            height, width, face_vertices_z, face_vertices_image,
            [face_uvs, face_mask], backend='cuda', **kwargs)
        interpolated_features = torch.cat(interpolated_features, dim=-1)

        gt_interpolated_features, gt_face_idx = _naive_deftet_sparse_render(
            pixel_coords2, render_ranges2, face_vertices_z2,
            face_vertices_image2, [face_uvs2, face_mask2], 1, **kwargs)
        gt_interpolated_features = torch.cat([
            feat.reshape(batch_size, height, width, -1) for feat in gt_interpolated_features
        ], dim=-1)

        grad_out = torch.rand_like(gt_interpolated_features)
        interpolated_features.backward(grad_out)
        gt_interpolated_features.backward(grad_out)

        assert face_vertices_z.grad is None or torch.all(face_vertices_z.grad == 0.)
        assert torch.allclose(face_vertices_image.grad,
                              face_vertices_image2.grad,
                              rtol=1e-3, atol=1e-2)
        assert torch.allclose(face_uvs.grad,
                              face_uvs2.grad,
                              rtol=1e-3, atol=1e-3)
        assert torch.allclose(face_mask.grad,
                              face_mask2.grad,
                              rtol=1e-3, atol=1e-3)

    @pytest.mark.parametrize('with_valid_faces', [False, True])
    def test_nvdiffrast_fwd_forward(
            self, batch_size, height, width, pixel_coords,
            render_ranges, face_vertices_z, face_vertices_image,
            face_uvs, with_valid_faces, valid_faces):
        if os.getenv('KAOLIN_TEST_NVDIFFRAST', '0') == '0':
            pytest.skip(f'test is ignored as KAOLIN_TEST_NVDIFFRAST is not set')
        if face_vertices_image.dtype == torch.double:
            pytest.skip("nvdiffrast not compatible with double")
        kwargs = {}
        if with_valid_faces:
            kwargs['valid_faces'] = valid_faces

        interpolated_features, face_idx = rasterize(
            height, width, face_vertices_z, face_vertices_image,
            face_uvs, backend='nvdiffrast_fwd', **kwargs)
        gt_interpolated_features, gt_face_idx = _naive_deftet_sparse_render(
            pixel_coords, render_ranges, face_vertices_z,
            face_vertices_image, face_uvs, 1, **kwargs)
        gt_interpolated_features = gt_interpolated_features.reshape(
            batch_size, height, width, face_uvs.shape[-1])
        gt_face_idx = gt_face_idx.reshape(batch_size, height, width)

        face_idx_same = face_idx == gt_face_idx
        # Numerical differences can lead to difference
        # face being rasterized, we assume about 98% similarity
        assert torch.sum(face_idx_same) / face_idx.numel() > 0.98
        mask_intersection = (face_idx >= 0) & (gt_face_idx >= 0)

        # Attribute can be quite different if the face getting rasterized is different
        assert torch.allclose(
            interpolated_features[face_idx_same],
            gt_interpolated_features[face_idx_same],
            rtol=1e-3, atol=1e-3
        )

    @pytest.mark.parametrize('with_valid_faces', [False, True])
    def test_nvdiffrast_fwd_forward_with_list(
            self, batch_size, height, width, pixel_coords,
            render_ranges, face_vertices_z, face_vertices_image,
            face_uvs, with_valid_faces, valid_faces, dtype):
        """Test with list of tensors as features"""
        if os.getenv('KAOLIN_TEST_NVDIFFRAST', '0') == '0':
            pytest.skip(f'test is ignored as KAOLIN_TEST_NVDIFFRAST is not set')
        if face_vertices_image.dtype == torch.double:
            pytest.skip("nvdiffrast not compatible with double")
        kwargs = {}
        if with_valid_faces:
            kwargs['valid_faces'] = valid_faces
        face_attr = [face_uvs, face_vertices_z.unsqueeze(-1)]

        (uvs_map, depth_map), face_idx = rasterize(
            height, width, face_vertices_z, face_vertices_image,
            face_attr, backend='nvdiffrast_fwd', **kwargs)
        (gt_uvs_map, gt_depth_map), gt_face_idx = _naive_deftet_sparse_render(
            pixel_coords, render_ranges, face_vertices_z,
            face_vertices_image, face_attr, 1, **kwargs)
        gt_uvs_map = gt_uvs_map.reshape(batch_size, height, width, face_uvs.shape[-1])
        gt_depth_map = gt_depth_map.reshape(batch_size, height, width, 1)
        gt_face_idx = gt_face_idx.reshape(batch_size, height, width)

        face_idx_same = face_idx == gt_face_idx
        # Numerical differences can lead to difference
        # face being rasterized, we assume about 98% similarity
        assert torch.sum(face_idx_same) / face_idx.numel() > 0.98
        mask_intersection = (face_idx >= 0) & (gt_face_idx >= 0)

        # On a smooth enough surface the depth maps should match
        # (exclusing border because of numerical difference)
        assert torch.allclose(
            depth_map[mask_intersection],
            gt_depth_map[mask_intersection],
            rtol=1e-3, atol=1e-3
        )
        # Attribute can be quite different if the face getting rasterized is different
        assert torch.allclose(
            uvs_map[face_idx_same],
            gt_uvs_map[face_idx_same],
            rtol=1e-3, atol=1e-3
        )

    @pytest.mark.parametrize('with_valid_faces', [False, True])
    def test_nvdiffrast_fwd_backward(
            self, batch_size, height, width, pixel_coords,
            render_ranges, face_vertices_z, face_vertices_image,
            face_uvs, with_valid_faces, valid_faces):
        if os.getenv('KAOLIN_TEST_NVDIFFRAST', '0') == '0':
            pytest.skip(f'test is ignored as KAOLIN_TEST_NVDIFFRAST is not set')
        if face_vertices_image.dtype == torch.double:
            pytest.skip("nvdiffrast not compatible with double")
        kwargs = {}
        if with_valid_faces:
            kwargs['valid_faces'] = valid_faces
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

        interpolated_features, face_idx = rasterize(
            height, width, face_vertices_z, face_vertices_image,
            face_uvs, backend='nvdiffrast_fwd', **kwargs)
        gt_interpolated_features, gt_face_idx = _naive_deftet_sparse_render(
            pixel_coords2, render_ranges2, face_vertices_z2,
            face_vertices_image2, face_uvs2, 1, **kwargs)
        gt_interpolated_features = gt_interpolated_features.reshape(
            batch_size, height, width, -1)
        gt_face_idx = gt_face_idx.reshape(batch_size, height, width)

        face_idx_diff = face_idx != gt_face_idx

        grad_out = torch.rand_like(gt_interpolated_features)
        grad_out[face_idx_diff] = 0.
        interpolated_features.backward(grad_out)
        gt_interpolated_features.backward(grad_out)

        assert face_vertices_z.grad is None or torch.all(face_vertices_z.grad == 0.)
        assert torch.allclose(face_vertices_image.grad,
                              face_vertices_image2.grad,
                              rtol=5e-2, atol=5e-2)
        assert torch.allclose(face_uvs.grad,
                              face_uvs2.grad,
                              rtol=1e-2, atol=5e-2)

    @pytest.mark.parametrize('with_valid_faces', [False, True])
    def test_nvdiffrast_fwd_backward_with_mask(
            self, batch_size, height, width, pixel_coords,
            render_ranges, face_vertices_z, face_vertices_image,
            face_uvs, with_valid_faces, valid_faces):
        if os.getenv('KAOLIN_TEST_NVDIFFRAST', '0') == '0':
            pytest.skip(f'test is ignored as KAOLIN_TEST_NVDIFFRAST is not set')
        if face_vertices_image.dtype == torch.double:
            pytest.skip("nvdiffrast not compatible with double")
        kwargs = {}
        if with_valid_faces:
            kwargs['valid_faces'] = valid_faces
        face_vertices_z = face_vertices_z.detach()
        face_vertices_z.requires_grad = True
        face_vertices_image = face_vertices_image.detach()
        face_vertices_image.requires_grad = True
        face_uvs = face_uvs.detach()
        face_uvs.requires_grad = True
        face_mask = torch.ones_like(face_uvs[..., :1], requires_grad=True)
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
        face_mask2 = face_mask.detach()
        face_mask2.requires_grad = True

        interpolated_features, face_idx = rasterize(
            height, width, face_vertices_z, face_vertices_image,
            [face_uvs, face_mask], backend='nvdiffrast_fwd', **kwargs)
        interpolated_features = torch.cat(interpolated_features, dim=-1)

        gt_interpolated_features, gt_face_idx = _naive_deftet_sparse_render(
            pixel_coords2, render_ranges2, face_vertices_z2,
            face_vertices_image2, [face_uvs2, face_mask2], 1, **kwargs)
        gt_interpolated_features = torch.cat([
            feat.reshape(batch_size, height, width, -1) for feat in gt_interpolated_features
        ], dim=-1)
        gt_face_idx = gt_face_idx.reshape(batch_size, height, width)

        face_idx_diff = face_idx != gt_face_idx

        grad_out = torch.rand_like(gt_interpolated_features)
        grad_out[face_idx_diff] = 0.
        interpolated_features.backward(grad_out)
        gt_interpolated_features.backward(grad_out)

        assert face_vertices_z.grad is None or torch.all(face_vertices_z.grad == 0.)
        assert torch.allclose(face_vertices_image.grad,
                              face_vertices_image2.grad,
                              rtol=5e-2, atol=5e-2)
        assert torch.allclose(face_uvs.grad,
                              face_uvs2.grad,
                              rtol=1e-2, atol=5e-2)
        assert torch.allclose(face_mask.grad,
                              face_mask2.grad,
                              rtol=1e-2, atol=5e-2)

    @pytest.mark.parametrize('with_valid_faces', [False, True])
    def test_nvdiffrast_forward(
            self, batch_size, height, width, face_vertices_z,
            face_vertices_image, face_uvs, with_valid_faces, valid_faces):
        if os.getenv('KAOLIN_TEST_NVDIFFRAST', '0') == '0':
            pytest.skip(f'test is ignored as KAOLIN_TEST_NVDIFFRAST is not set')
        if face_vertices_image.dtype == torch.double:
            pytest.skip("nvdiffrast not compatible with double")
        kwargs = {}
        if with_valid_faces:
            kwargs['valid_faces'] = valid_faces
        face_attr = face_uvs

        interpolated_features, face_idx = rasterize(
            height, width, face_vertices_z, face_vertices_image,
            face_attr, backend='nvdiffrast', **kwargs)
        # To simplify the test we use nvdiffrast_fwd for ground truth
        # already tested above
        gt_interpolated_features, gt_face_idx = rasterize(
            height, width, face_vertices_z, face_vertices_image,
            face_attr, backend='nvdiffrast_fwd', **kwargs)
        gt_interpolated_features = gt_interpolated_features.reshape(
            batch_size, height, width, face_uvs.shape[-1])
        gt_face_idx = gt_face_idx.reshape(batch_size, height, width)

        assert torch.equal(face_idx, gt_face_idx)
        assert torch.equal(interpolated_features, gt_interpolated_features)

    @pytest.mark.parametrize('with_valid_faces', [False, True])
    def test_nvdiffrast_forward_with_list(
            self, batch_size, height, width, face_vertices_z,
            face_vertices_image, face_uvs, with_valid_faces, valid_faces):
        """Test with list of tensors as features"""
        if os.getenv('KAOLIN_TEST_NVDIFFRAST', '0') == '0':
            pytest.skip(f'test is ignored as KAOLIN_TEST_NVDIFFRAST is not set')
        if face_vertices_image.dtype == torch.double:
            pytest.skip("nvdiffrast not compatible with double")
        kwargs = {}
        if with_valid_faces:
            kwargs['valid_faces'] = valid_faces
        face_attr = [face_uvs, torch.ones_like(face_uvs[..., :1])]

        (uvs_map, mask), face_idx = rasterize(
            height, width, face_vertices_z, face_vertices_image,
            face_attr, backend='nvdiffrast', **kwargs)
        # To simplify the test we use nvdiffrast_fwd for ground truth
        # already tested above
        (gt_uvs_map, gt_mask), gt_face_idx = rasterize(
            height, width, face_vertices_z, face_vertices_image,
            face_attr, backend='nvdiffrast_fwd', **kwargs)
        gt_uvs_map = gt_uvs_map.reshape(
            batch_size, height, width, face_uvs.shape[-1])
        gt_mask = gt_mask.reshape(batch_size, height, width, 1)
        gt_face_idx = gt_face_idx.reshape(batch_size, height, width)

        assert torch.equal(face_idx, gt_face_idx)
        assert torch.equal(uvs_map, gt_uvs_map)
        assert torch.equal(mask, gt_mask)

    @pytest.mark.parametrize('with_valid_faces', [False, True])
    def test_nvdiffrast_backward(
            self, batch_size, height, width, face_vertices_z,
            face_vertices_image, face_uvs, with_valid_faces, valid_faces):
        if os.getenv('KAOLIN_TEST_NVDIFFRAST', '0') == '0':
            pytest.skip(f'test is ignored as KAOLIN_TEST_NVDIFFRAST is not set')
        if face_vertices_image.dtype == torch.double:
            pytest.skip("nvdiffrast not compatible with double")
        kwargs = {}
        if with_valid_faces:
            kwargs['valid_faces'] = valid_faces

        face_vertices_z = face_vertices_z.detach()
        face_vertices_z.requires_grad = True
        face_vertices_image = face_vertices_image.detach()
        face_vertices_image.requires_grad = True
        face_uvs = face_uvs.detach()
        face_uvs.requires_grad = True
        face_vertices_z2 = face_vertices_z.detach()
        face_vertices_z2.requires_grad = True
        face_vertices_image2 = face_vertices_image.detach()
        face_vertices_image2.requires_grad = True
        face_uvs2 = face_uvs.detach()
        face_uvs2.requires_grad = True

        interpolated_features, face_idx = rasterize(
            height, width, face_vertices_z, face_vertices_image,
            face_uvs, backend='nvdiffrast', **kwargs)
        gt_interpolated_features, gt_face_idx = rasterize(
            height, width, face_vertices_z2, face_vertices_image2,
            face_uvs2, backend='nvdiffrast_fwd', **kwargs)
        gt_interpolated_features = gt_interpolated_features.reshape(
            batch_size, height, width, -1)
        gt_face_idx = gt_face_idx.reshape(batch_size, height, width)

        grad_out = torch.rand_like(gt_interpolated_features)
        interpolated_features.backward(grad_out)
        gt_interpolated_features.backward(grad_out)

        assert face_vertices_z.grad is None or torch.all(face_vertices_z.grad == 0.)
        assert torch.allclose(face_vertices_image.grad,
                              face_vertices_image2.grad,
                              rtol=5e-2, atol=5e-2)
        assert torch.allclose(face_uvs.grad,
                              face_uvs2.grad,
                              rtol=1e-2, atol=1e-2)
