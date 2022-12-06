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
import kaolin as kal

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, os.pardir, os.pardir,
                         os.pardir, os.pardir, 'samples/')
SIMPLE_GT_DIR = os.path.join(ROOT_DIR, os.pardir, os.pardir,
                             os.pardir, os.pardir, 'samples/dibr/simple/')
SPHERE_GT_DIR = os.path.join(ROOT_DIR, os.pardir, os.pardir,
                             os.pardir, os.pardir, 'samples/dibr/sphere/')

@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
@pytest.mark.parametrize('height,width', [(35, 31)])
@pytest.mark.parametrize('sigmainv', [7000, 70])
@pytest.mark.parametrize('boxlen', [0.02, 0.2])
class TestSimpleDibrSoftMask:
    @pytest.fixture(autouse=True)
    def face_vertices_image(self, device, dtype):
        return torch.tensor(
            [[[[-0.7,  0. ], [0. , -0.7], [ 0. ,  0.7]],
              [[-0.7,  0. ], [0. ,  0.7], [ 0. , -0.7]],
              [[ 0. , -0.7], [0. ,  0.7], [ 0.7,  0. ]]],
             [[[-0.7, -0.7], [0.7, -0.7], [-0.7,  0.7]],
              [[-0.7, -0.7], [0.7, -0.7], [-0.7,  0.7]],
              [[-0.7, -0.7], [0.7, -0.7], [-0.7,  0.7]]]],
            device=device, dtype=dtype)

    @pytest.fixture(autouse=True)
    def face_vertices_z(self, device, dtype):
        return torch.tensor(
            [[[-2. , -1., -1.],
              [-2.5, -3., -3.],
              [-2. , -2., -2.]],
             [[-2. , -1., -3.],
              [-2. , -2., -2.],
              [-2. , -3., -1.]]],
            device=device, dtype=dtype)

    @pytest.fixture(autouse=True)
    def selected_face_idx(self, height, width, face_vertices_image,
                          face_vertices_z):
        # this face_features is not really used
        # but we need it to run rasterize
        face_features = torch.zeros(face_vertices_z.shape + (1,),
                                    dtype=face_vertices_z.dtype,
                                    device=face_vertices_z.device)
        _, face_idx = kal.render.mesh.rasterize(
            height, width, face_vertices_z,
            face_vertices_image, face_features)
        return face_idx

    @pytest.fixture(autouse=True)
    def gt_soft_mask(self, height, width, sigmainv, boxlen, device, dtype):
        # From Kaolin V0.10.0
        return torch.load(os.path.join(
            SIMPLE_GT_DIR,
            f'soft_mask_{height}_{width}_{int(sigmainv)}_{boxlen}.pt'
        )).to(device=device, dtype=dtype)

    @pytest.fixture(autouse=True)
    def gt_close_face_idx(self, height, width, sigmainv, boxlen, device, dtype):
        # From Kaolin V0.10.0
        return torch.load(os.path.join(
            SIMPLE_GT_DIR,
            f'close_face_idx_{height}_{width}_{int(sigmainv)}_{boxlen}.pt'
        )).to(device=device, dtype=dtype).long() - 1

    @pytest.fixture(autouse=True)
    def gt_close_face_dist(self, height, width, sigmainv, boxlen, device, dtype):
        # From Kaolin V0.10.0
        return torch.load(os.path.join(
            SIMPLE_GT_DIR,
            f'close_face_dist_{height}_{width}_{int(sigmainv)}_{boxlen}.pt'
        )).to(device=device, dtype=dtype)

    @pytest.fixture(autouse=True)
    def gt_close_face_dist_type(self, height, width, sigmainv, boxlen, device, dtype):
        # From Kaolin V0.10.0
        return torch.load(os.path.join(
            SIMPLE_GT_DIR,
            f'close_face_dist_type_{height}_{width}_{int(sigmainv)}_{boxlen}.pt'
        )).to(device=device, dtype=torch.uint8)

    @pytest.mark.parametrize('multiplier', [1000, 100, 1])
    @pytest.mark.parametrize('knum', [30, 20])
    def test_C_dibr_soft_mask_forward(
            self, face_vertices_image, selected_face_idx, sigmainv, boxlen,
            knum, multiplier, gt_soft_mask, gt_close_face_idx,
            gt_close_face_dist, gt_close_face_dist_type):
        # This is testing the CUDA Op so we can also check for stored tensors
        face_vertices_image = face_vertices_image * multiplier
        points_min = torch.min(face_vertices_image, dim=-2)[0]
        points_max = torch.max(face_vertices_image, dim=-2)[0]
        face_large_bboxes = torch.cat([
            points_min - boxlen * multiplier,
            points_max + boxlen * multiplier
        ], dim=-1)
        soft_mask, close_face_dist, close_face_idx, close_face_dist_type = \
            kal._C.render.mesh.dibr_soft_mask_forward_cuda(
                face_vertices_image,
                face_large_bboxes,
                selected_face_idx,
                sigmainv,
                knum,
                multiplier)

        assert torch.allclose(
            soft_mask, gt_soft_mask, atol=1e-5, rtol=1e-5)
        assert torch.equal(
            close_face_idx, gt_close_face_idx[..., :knum])
        assert torch.allclose(
            close_face_dist, gt_close_face_dist[..., :knum],
            atol=1e-5, rtol=1e-5)
        assert torch.equal(
            close_face_dist_type, gt_close_face_dist_type[..., :knum])

    @pytest.mark.parametrize('multiplier', [1000, 100])
    @pytest.mark.parametrize('knum', [30, 20])
    def test_dibr_soft_mask_forward(self, face_vertices_image, selected_face_idx,
                                    sigmainv, boxlen, knum, multiplier, gt_soft_mask):
        soft_mask = kal.render.mesh.dibr_soft_mask(
            face_vertices_image,
            selected_face_idx,
            sigmainv,
            boxlen,
            knum,
            multiplier
        )

        assert torch.allclose(
            soft_mask, gt_soft_mask, atol=1e-5, rtol=1e-5)

    @pytest.fixture(autouse=True)
    def gt_grad_face_vertices_image(self, height, width, sigmainv,
                                    boxlen, device, dtype):
        # From Kaolin V0.10.0
        return torch.load(os.path.join(
            SIMPLE_GT_DIR,
            f'grad_face_vertices_image_{height}_{width}_{int(sigmainv)}_{boxlen}.pt'
        )).to(device=device, dtype=dtype)

    @pytest.mark.parametrize('multiplier', [1000, 100, 1])
    @pytest.mark.parametrize('knum', [30, 20])
    def test_dibr_soft_mask_backward(self, face_vertices_image, selected_face_idx,
                                     sigmainv, boxlen, knum, multiplier, 
                                     gt_grad_face_vertices_image):
        face_vertices_image = face_vertices_image.detach()
        face_vertices_image.requires_grad = True
        soft_mask = kal.render.mesh.dibr_soft_mask(
            face_vertices_image,
            selected_face_idx,
            sigmainv,
            boxlen,
            knum,
            multiplier
        )
        mask = selected_face_idx != -1
        shifted_mask = torch.nn.functional.pad(
            mask, (0, 5)
        )[..., 5:]
        loss = kal.metrics.render.mask_iou(soft_mask, shifted_mask)
        loss.backward()

        assert torch.allclose(
            face_vertices_image.grad, gt_grad_face_vertices_image,
            rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('device', ['cuda'])
@pytest.mark.parametrize('dtype', [torch.float, torch.double])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("height,width", [(35, 31)])
@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize('sigmainv', [7000, 70])
@pytest.mark.parametrize('boxlen', [0.02, 0.01])
class TestDibrSoftMask:
    @pytest.fixture(autouse=True)
    def mesh(self):
        mesh = kal.io.obj.import_mesh(os.path.join(MODEL_DIR, 'model.obj'),
                                      with_materials=False)
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
    def face_vertices_image(self, vertices_image, faces):
        return kal.ops.mesh.index_vertices_by_faces(
            vertices_image, faces)


    @pytest.fixture(autouse=True)
    def selected_face_idx(self, height, width, face_vertices_image,
                          face_vertices_z):
        # this face_features is not really used
        # but we need it to run rasterize
        face_features = torch.zeros(face_vertices_z.shape + (1,),
                                    dtype=face_vertices_z.dtype,
                                    device=face_vertices_z.device)
        _, face_idx = kal.render.mesh.rasterize(
            height, width, face_vertices_z,
            face_vertices_image, face_features)
        return face_idx

    @pytest.fixture(autouse=True)
    def gt_soft_mask(self, batch_size, height, width, sigmainv, boxlen, device, dtype):
        # From Kaolin V0.10.0
        return torch.load(os.path.join(
            SPHERE_GT_DIR,
            f'soft_mask_{height}_{width}_{int(sigmainv)}_{boxlen}.pt'
        )).to(device=device, dtype=dtype)[:batch_size]

    @pytest.fixture(autouse=True)
    def gt_close_face_idx(self, batch_size, height, width, sigmainv, boxlen, device, dtype):
        # From Kaolin V0.10.0.
        return torch.load(os.path.join(
            SPHERE_GT_DIR,
            f'close_face_idx_{height}_{width}_{int(sigmainv)}_{boxlen}.pt'
        )).to(device=device, dtype=dtype)[:batch_size].long() - 1

    @pytest.fixture(autouse=True)
    def gt_close_face_dist(self, batch_size, height, width, sigmainv, boxlen, device, dtype):
        # From Kaolin V0.10.0
        return torch.load(os.path.join(
            SPHERE_GT_DIR,
            f'close_face_dist_{height}_{width}_{int(sigmainv)}_{boxlen}.pt'
        )).to(device=device, dtype=dtype)[:batch_size]

    @pytest.fixture(autouse=True)
    def gt_close_face_dist_type(self, batch_size, height, width, sigmainv, boxlen, device, dtype):
        # From Kaolin V0.10.0
        return torch.load(os.path.join(
            SPHERE_GT_DIR,
            f'close_face_dist_type_{height}_{width}_{int(sigmainv)}_{boxlen}.pt'
        )).to(device=device, dtype=dtype)[:batch_size]

    @pytest.mark.parametrize('multiplier', [1000, 100])
    @pytest.mark.parametrize('knum', [30, 40])
    def test_C_dibr_soft_mask_forward(
            self, face_vertices_image, selected_face_idx, knum, multiplier,
            sigmainv, boxlen, gt_soft_mask, gt_close_face_idx,
            gt_close_face_dist, gt_close_face_dist_type):
        # This is testing the CUDA Op so we can also check for stored tensors
        face_vertices_image = face_vertices_image * multiplier
        points_min = torch.min(face_vertices_image, dim=-2)[0]
        points_max = torch.max(face_vertices_image, dim=-2)[0]
        face_large_bboxes = torch.cat([
            points_min - boxlen * multiplier,
            points_max + boxlen * multiplier
        ], dim=-1)
        soft_mask, close_face_dist, close_face_idx, close_face_dist_type = \
            kal._C.render.mesh.dibr_soft_mask_forward_cuda(
                face_vertices_image,
                face_large_bboxes,
                selected_face_idx,
                sigmainv,
                knum,
                multiplier
            )

        assert torch.allclose(
            soft_mask, gt_soft_mask, atol=1e-5, rtol=1e-5)
        assert torch.equal(close_face_idx, gt_close_face_idx[..., :knum])
        _, height, width = selected_face_idx.shape
        assert torch.allclose(
            close_face_dist, gt_close_face_dist[..., :knum],
            atol=1e-5, rtol=1e-5)
        same = close_face_dist_type != gt_close_face_dist_type[..., :knum]
        assert torch.sum(same) / same.numel() <= 0.01

    @pytest.mark.parametrize('multiplier', [1000, 100])
    @pytest.mark.parametrize('knum', [30, 40])
    def test_dibr_soft_mask_forward(self, face_vertices_image, selected_face_idx,
                                    sigmainv, boxlen, knum, multiplier, gt_soft_mask):
        soft_mask = kal.render.mesh.dibr_soft_mask(
            face_vertices_image,
            selected_face_idx,
            sigmainv,
            boxlen,
            knum,
            multiplier
        )

        assert torch.allclose(
            soft_mask, gt_soft_mask, atol=1e-5, rtol=1e-5)

    @pytest.fixture(autouse=True)
    def gt_grad_face_vertices_image(self, batch_size, height, width, sigmainv,
                                    boxlen, device, dtype):
        # From Kaolin V0.10.0
        return torch.load(os.path.join(
            SPHERE_GT_DIR,
            f'grad_face_vertices_image_{height}_{width}_{int(sigmainv)}_{boxlen}.pt'
        )).to(device=device, dtype=dtype)[:batch_size]

    @pytest.mark.parametrize('multiplier', [1000, 100, 1])
    @pytest.mark.parametrize('knum', [30, 40])
    def test_dibr_soft_mask_backward(self, face_vertices_image, selected_face_idx,
                                     sigmainv, boxlen, knum, multiplier,
                                     gt_grad_face_vertices_image):
        face_vertices_image = face_vertices_image.detach()
        face_vertices_image.requires_grad = True
        soft_mask = kal.render.mesh.dibr_soft_mask(
            face_vertices_image,
            selected_face_idx,
            sigmainv,
            boxlen,
            knum,
            multiplier
        )
        mask = selected_face_idx != -1
        shifted_mask = torch.nn.functional.pad(
            mask, (0, 5)
        )[..., 5:]
        loss = kal.metrics.render.mask_iou(soft_mask, shifted_mask)
        loss.backward()

        # rtol and atol must be high because numerical differences leads to different
        # distance types
        assert torch.allclose(
            face_vertices_image.grad, gt_grad_face_vertices_image,
            rtol=1e-1, atol=1e-1)


@pytest.mark.parametrize('dtype', [torch.float, torch.double])
@pytest.mark.parametrize("batch_size", [3, 1])
@pytest.mark.parametrize("height,width", [(35, 31)])
@pytest.mark.parametrize("flip", [False, True])
class TestDibrRasterization:
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
    def face_vertices_camera(self, vertices_camera, faces):
        return kal.ops.mesh.index_vertices_by_faces(
            vertices_camera, faces)

    @pytest.fixture(autouse=True)
    def face_vertices_z(self, face_vertices_camera):
        return face_vertices_camera[..., -1]

    @pytest.fixture(autouse=True)
    def face_vertices_image(self, vertices_image, faces):
        return kal.ops.mesh.index_vertices_by_faces(
            vertices_image, faces)

    @pytest.fixture(autouse=True)
    def face_normals_z(self, face_vertices_camera):
        return kal.ops.mesh.face_normals(
            face_vertices_camera, unit=True
        )[..., -1]

    @pytest.fixture(autouse=True)
    def face_uvs(self, mesh, batch_size, dtype, flip):
        face_uvs_idx = mesh.face_uvs_idx.cuda()
        if flip:
            face_uvs_idx = torch.flip(face_uvs_idx, dims=(-1,))
        return kal.ops.mesh.index_vertices_by_faces(
            mesh.uvs.unsqueeze(0).to('cuda', dtype),
            face_uvs_idx).repeat(batch_size, 1, 1, 1)

    @pytest.mark.parametrize('sigmainv', [7000, 70])
    @pytest.mark.parametrize('boxlen', [0.02, 0.01])
    @pytest.mark.parametrize('knum', [30, 40])
    @pytest.mark.parametrize('multiplier', [1000, 100])
    @pytest.mark.parametrize('rast_backend', ['cuda', 'nvdiffrast_fwd', 'nvdiffrast'])
    def test_dibr_rasterization(self, height, width, face_vertices_z,
                                face_vertices_image, face_uvs, face_normals_z,
                                sigmainv, boxlen, knum, multiplier, rast_backend):
        if rast_backend in {'nvdiffrast_fwd', 'nvdiffrast'}:
            if os.getenv('KAOLIN_TEST_NVDIFFRAST', '0') == '0':
                pytest.skip(f'test is ignored as KAOLIN_TEST_NVDIFFRAST is not set')
            if face_vertices_z.dtype == torch.double:
                pytest.skip("nvdiffrast not compatible with double")
        gt_interpolated_features, gt_face_idx = rasterize(
            height, width,
            face_vertices_z,
            face_vertices_image,
            face_uvs,
            face_normals_z >= 0.,
            multiplier,
            backend=rast_backend
        )
        _multiplier = 1000. if multiplier is None else multiplier
        gt_soft_mask = kal.render.mesh.dibr_soft_mask(
            face_vertices_image,
            gt_face_idx,
            sigmainv,
            boxlen,
            knum,
            _multiplier
        )

        interpolated_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
            height, width,
            face_vertices_z,
            face_vertices_image,
            face_uvs,
            face_normals_z,
            sigmainv,
            boxlen,
            knum,
            multiplier,
            rast_backend=rast_backend
        )

        assert torch.equal(interpolated_features, gt_interpolated_features)
        assert torch.equal(soft_mask, gt_soft_mask)
        assert torch.equal(face_idx, gt_face_idx)
