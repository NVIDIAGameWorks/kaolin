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

import random
import pytest

import numpy as np
import torch
import os

from kaolin.render.camera import perspective_camera, rotate_translate_points
from kaolin.render.mesh.rasterization import dibr_rasterization
from kaolin.render.mesh.utils import texture_mapping, spherical_harmonic_lighting
from kaolin.ops.mesh import index_vertices_by_faces, face_normals
from PIL import Image

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(ROOT_DIR, '../../../../samples/rasterization')


# TODO(cfujitsang): Add half support
@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("height", [256])
@pytest.mark.parametrize("width", [512])
class TestDIBR:
    @pytest.fixture(autouse=True)
    def vertices(self, dtype, device):
        # shape: (batch_size, num_vertices, 3)
        return torch.tensor(
            [[[-1, -1, -1], [1, -1, 0], [1, 1, -1], [-1, 1, 0]],
             [[-1, -1, 0], [1, -1, -1], [1, 1, 0], [-1, 1, -1]]],
            dtype=dtype,
            device=device)

    @pytest.fixture(autouse=True)
    def faces(self, device):
        # shape: (num_faces, 3)
        return torch.tensor([[0, 1, 2], [0, 2, 3]],
                            dtype=torch.long,
                            device=device)

    @pytest.fixture(autouse=True)
    def vertex_colors(self, dtype, device):
        # shape: (batch_size, num_vertices, 3)
        return torch.tensor([[[0.9, 0.1, 0.1],
                              [0.1, 0.9, 0.1],
                              [0.1, 0.1, 0.9],
                              [0.9, 0.9, 0.1]],
                             [[0.1, 0.9, 0.1],
                              [0.9, 0.1, 0.1],
                              [0.9, 0.9, 0.1],
                              [0.1, 0.1, 0.9]]],
                            dtype=dtype,
                            device=device)

    @pytest.fixture(autouse=True)
    def uvs(self, dtype, device):
        return torch.tensor([[[0.001, 0.001],
                              [1 - 0.001, 0.001],
                              [1 - 0.001, 1 - 0.001],
                              [0.001, 1 - 0.001]]],
                            dtype=dtype,
                            device=device).repeat(2, 1, 1)

    @pytest.fixture(autouse=True)
    def texture_maps(self, dtype, device):
        # shape: (batch_size, 3, h, w)
        # this is because when we do pytorch interpolation
        # the shape is (batch_size, features, height, width)
        # we we need to prepare an image with size as (batch_size, 3, h, w)
        jetcolor = torch.tensor([[[
            [0.0, 0.0, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, 1.0, 1.0],
            [0.5, 1.0, 0.5],
            [1.0, 1.0, 0.0],
            [1.0, 0.5, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.0, 0.0]
        ]]], dtype=dtype, device=device)

        return jetcolor.repeat(2, jetcolor.shape[2], 1, 1).permute(0, 3, 1, 2)

    @pytest.fixture(autouse=True)
    def lights(self, dtype, device):
        # shape: (batch_size, 9)
        return torch.tensor([[1, 1, 1, -1, 0, 0, 0, 0, 0]],
                            dtype=dtype,
                            device=device).repeat(2, 1)

    @pytest.fixture(autouse=True)
    def camera_rot(self, dtype, device):
        # shape: (batch_size, 3, 3)
        return torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                            dtype=dtype,
                            device=device).repeat(2, 1, 1)

    @pytest.fixture(autouse=True)
    def camera_trans(self, dtype, device):
        # shape: (batch_size, 3)
        return torch.tensor([[0, 0, 4]],
                            dtype=dtype,
                            device=device).repeat(2, 1)

    @pytest.fixture(autouse=True)
    def camera_proj(self, width, height, dtype, device):
        # shape: (3, 1)
        return torch.tensor([[2.5 / (width / height)], [2.5], [-1]],
                            dtype=dtype,
                            device=device)

    @pytest.fixture(autouse=True)
    def vertices_camera(self, vertices, camera_rot, camera_trans):
        return rotate_translate_points(vertices, camera_rot, camera_trans)

    @pytest.fixture(autouse=True)
    def vertices_image(self, vertices_camera, camera_proj):
        return perspective_camera(vertices_camera, camera_proj)

    @pytest.fixture(autouse=True)
    def face_vertices_camera(self, vertices_camera, faces):
        return index_vertices_by_faces(vertices_camera, faces)

    @pytest.fixture(autouse=True)
    def face_vertices_image(self, vertices_image, faces):
        return index_vertices_by_faces(vertices_image, faces)

    @pytest.fixture(autouse=True)
    def face_camera_normals_z(self, face_vertices_camera):
        face_normals_unit = face_normals(face_vertices_camera, unit=True)
        return face_normals_unit[:, :, 2]

    def test_render_vertex_colors(self, vertex_colors, faces,
                                  face_vertices_camera, face_vertices_image,
                                  face_camera_normals_z, height, width,
                                  dtype, device):
        batch_size = faces.shape[0]
        # face_vertex_colors
        attributes = vertex_colors
        face_attributes_idx = faces
        face_attributes = index_vertices_by_faces(attributes, face_attributes_idx)

        # imfeat is interpolated features
        # improb is the soft mask
        # imfaceidx is the face index map, which pixel is covered by which face
        # it starts from 1, 0 is void.
        imfeat, improb, imfaceidx = dibr_rasterization(height,
                                                       width,
                                                       face_vertices_camera[:, :, :, 2],
                                                       face_vertices_image,
                                                       face_attributes,
                                                       face_camera_normals_z)
        image = imfeat
        images_gt = [torch.from_numpy(np.array(Image.open(
                         os.path.join(SAMPLE_DIR, f'vertex_color_{bs}.png'))))
                     for bs in range(batch_size)]
        images_gt = torch.stack(images_gt, dim=0).to(device, dtype) / 255.
        # the rendered soft mask is only tested here
        images_prob_gt = [torch.from_numpy(np.array(Image.open(
                              os.path.join(SAMPLE_DIR, f"image_prob_{bs}.png"))))
                          for bs in range(batch_size)]
        images_prob_gt = torch.stack(images_prob_gt, dim=0).to(device, dtype) / 255.
        # the rendered face_idx is only tested here
        images_face_idx_gt = [torch.from_numpy(np.array(Image.open(
                                  os.path.join(SAMPLE_DIR, f"image_face_idx_{bs}.png"))))
                              for bs in range(batch_size)]
        images_face_idx_gt = \
            (torch.stack(images_face_idx_gt, dim=0).to(device, torch.long) // 100) - 1
        assert torch.allclose(image, images_gt, atol=1. / 255.)
        assert torch.allclose(improb, images_prob_gt, atol=1. / 255.0)

        if dtype == torch.double:
            num_pix_diff_tol = 4
        else:
            num_pix_diff_tol = 0

        num_pix_diff = torch.sum(~torch.isclose(imfaceidx,
                                                images_face_idx_gt,
                                                atol=1. / 255.))
        assert num_pix_diff <= num_pix_diff_tol

    def test_render_normal(self, face_vertices_camera, face_vertices_image,
                           face_camera_normals_z, height, width, dtype, device):
        batch_size = face_vertices_camera.shape[0]
        face_normals_unit = face_normals(face_vertices_camera, unit=True)
        face_attributes = face_normals_unit.unsqueeze(-2).repeat(1, 1, 3, 1)

        # imfeat is interpolated features
        # improb is the soft mask
        # imfaceidx is the face index map, which pixel is covered by which face
        # it starts from 1, 0 is void.
        imfeat, improb, imfaceidx = dibr_rasterization(height,
                                                       width,
                                                       face_vertices_camera[:, :, :, 2],
                                                       face_vertices_image,
                                                       face_attributes,
                                                       face_camera_normals_z)
        images = (imfeat + 1) / 2
        images_gt = [torch.from_numpy(np.array(Image.open(
                         os.path.join(SAMPLE_DIR, f'vertex_normal_{bs}.png'))))
                     for bs in range(batch_size)]
        images_gt = torch.stack(images_gt, dim=0).to(device, dtype) / 255.
        if dtype == torch.double:
            num_pix_diff_tol = 8
        else:
            num_pix_diff_tol = 0
        num_pix_diff = torch.sum(~torch.isclose(images, images_gt, atol=1. / 255.))
        assert num_pix_diff <= num_pix_diff_tol

    def test_render_depths(self, face_vertices_camera, face_vertices_image,
                           face_camera_normals_z, height, width, dtype, device):
        batch_size = face_vertices_camera.shape[0]
        # face_vertices_camera is of shape (num_batch, num_face, 9)
        num_batch, num_faces = face_vertices_camera.shape[:2]
        face_attributes = face_vertices_camera.reshape(num_batch, num_faces, 3, 3)[:, :, :, 2:3]

        # imfeat is interpolated features
        # improb is the soft mask
        # imfaceidx is the face index map, which pixel is covered by which face
        # it starts from 1, 0 is void.
        imfeat, improb, imfaceidx = dibr_rasterization(height,
                                                       width,
                                                       face_vertices_camera[:, :, :, 2],
                                                       face_vertices_image,
                                                       face_attributes,
                                                       face_camera_normals_z)

        image = imfeat
        image_valid_region = image < 0
        image_valid_values = image[image_valid_region]
        image_depth_norm = (image - image_valid_values.min()) / (
            image_valid_values.max() - image_valid_values.min())
        image = torch.where(image_valid_region, image_depth_norm, image)

        for bs in range(batch_size):
            image_gt = torch.from_numpy(np.array(Image.open(
                os.path.join(SAMPLE_DIR, f'depth_{bs}.png'))))
            image_gt = image_gt.to(device, dtype).unsqueeze(2) / 255.
            assert torch.allclose(image[bs], image_gt, atol=1 / 255.0)

    def test_render_texture(self, uvs, faces, texture_maps,
                            face_vertices_camera, face_vertices_image,
                            face_camera_normals_z, height, width, dtype, device):
        batch_size = faces.shape[0]
        # attributes with uvs
        #uvs_with_mask = torch.nn.functional.pad(uvs, pad=(1, 0), value=1)
        #attributes = uvs_with_mask
        face_uvs = index_vertices_by_faces(uvs, faces)
        face_attributes = [torch.ones((*face_uvs.shape[:-1], 1),
                                      device=device, dtype=dtype),
                           face_uvs]

        (texmask, texcoord), improb, imfaceidx = dibr_rasterization(height,
                                                                    width,
                                                                    face_vertices_camera[:, :, :, 2],
                                                                    face_vertices_image,
                                                                    face_attributes,
                                                                    face_camera_normals_z)

        texcolor = texture_mapping(texcoord, texture_maps, mode='bilinear')
        image = texcolor * texmask

        for bs in range(batch_size):
            image_gt = torch.from_numpy(np.array(Image.open(
                os.path.join(SAMPLE_DIR, f'texture_{bs}.png'))))
            image_gt = image_gt.to(device, dtype) / 255.
            assert torch.allclose(image[bs], image_gt, atol=1. / 255.0)

    def test_render_texture_with_light(self, uvs, faces, texture_maps, lights,
                                       face_vertices_camera, face_vertices_image,
                                       face_camera_normals_z, height, width, dtype, device):
        batch_size = faces.shape[0]
        # Note: in this example uv face is the same as mesh face
        # but they could be different
        face_uvs = index_vertices_by_faces(uvs, faces)

        # normal
        face_normals_unit = face_normals(face_vertices_camera, unit=True)
        face_normals_unit = face_normals_unit.unsqueeze(-2).repeat(1, 1, 3, 1)

        # merge them together
        face_attributes = [
            torch.ones((*face_uvs.shape[:-1], 1), device=device, dtype=dtype),
            face_uvs,
            face_normals_unit
        ]

        (texmask, texcoord, imnormal), improb, imidx = dibr_rasterization(height,
                                                                          width,
                                                                          face_vertices_camera[:, :, :, 2],
                                                                          face_vertices_image,
                                                                          face_attributes,
                                                                          face_camera_normals_z)

        texcolor = texture_mapping(texcoord, texture_maps, mode='nearest')
        coef = spherical_harmonic_lighting(imnormal, lights)
        images = torch.clamp(texmask * texcolor * coef.unsqueeze(-1), 0, 1)

        if dtype == torch.double:
            num_pix_diff_tol = 74  # (over 2 x 256 x 512 x 3 pixels)
        else:
            num_pix_diff_tol = 0

        images_gt = [torch.from_numpy(np.array(Image.open(
                         os.path.join(SAMPLE_DIR, f'texture_light_{bs}.png'))))
                     for bs in range(batch_size)]
        images_gt = torch.stack(images_gt, dim=0).to(device, dtype) / 255.

        num_pix_diff = torch.sum(~torch.isclose(images, images_gt, atol=1. / 255.))
        assert num_pix_diff <= num_pix_diff_tol

    ###################################
    #### TEST VERTICE OPTIMIZATION ####
    ###################################
    # Test that the vertex positions can be optimized
    # with rendered target image
    def test_optimize_vertex_position(self, vertices, faces, vertex_colors, vertices_image,
                                      camera_rot, camera_trans, camera_proj,
                                      height, width, dtype, device):
        batch_size = faces.shape[0]
        # face_vertex_colors
        camera_rot = camera_rot.to(device, dtype)
        camera_trans = camera_trans.to(device, dtype)
        camera_proj = camera_proj.to(device, dtype)
        face_attributes = index_vertices_by_faces(vertex_colors.to(device, dtype), faces)
        vertices = vertices.to(device, dtype).clone().detach()
        vertices.requires_grad = False
        moved_vertices = vertices.to(device, dtype).clone()
        moved_vertices[:,0,:2] += 0.4
        moved_vertices = moved_vertices.detach()
        moved_vertices.requires_grad = True

        images_gt = [torch.from_numpy(np.array(Image.open(
                        os.path.join(SAMPLE_DIR, f'vertex_color_{bs}.png'))))
                     for bs in range(batch_size)]
        images_gt = torch.stack(images_gt, dim=0).to(device, dtype) / 255.

        moved_vertices_camera = rotate_translate_points(moved_vertices, camera_rot, camera_trans)
        moved_vertices_image = perspective_camera(moved_vertices_camera, camera_proj)

        # test that the vertex are far enough to fail the test.
        assert not torch.allclose(moved_vertices_image, vertices_image, atol=1e-2, rtol=1e-2)

        with torch.no_grad():
            moved_vertices_camera = rotate_translate_points(moved_vertices, camera_rot, camera_trans)
            moved_vertices_image = perspective_camera(moved_vertices_camera, camera_proj)
            face_moved_vertices_camera = index_vertices_by_faces(moved_vertices_camera, faces)
            face_moved_vertices_image = index_vertices_by_faces(moved_vertices_image, faces)
            face_moved_normals_z = face_normals(face_moved_vertices_camera,
                                                     unit=True)[:, :, 2]
            imfeat, _, _ = dibr_rasterization(height,
                                              width,
                                              face_moved_vertices_camera[:, :, :, 2],
                                              face_moved_vertices_image,
                                              face_attributes,
                                              face_moved_normals_z)
            original_loss = torch.mean(torch.abs(imfeat - images_gt))

        # test that the loss is high enough
        assert original_loss > 0.01
        optimizer = torch.optim.Adam([moved_vertices], lr=5e-3)

        for i in range(100):
            optimizer.zero_grad()
            moved_vertices_camera = rotate_translate_points(moved_vertices, camera_rot, camera_trans)
            moved_vertices_image = perspective_camera(moved_vertices_camera, camera_proj)
            face_moved_vertices_camera = index_vertices_by_faces(moved_vertices_camera, faces)
            face_moved_vertices_image = index_vertices_by_faces(moved_vertices_image, faces)
            face_moved_normals_z = face_normals(face_moved_vertices_camera,
                                                     unit=True)[:, :, 2]
            imfeat, _, _ = dibr_rasterization(height,
                                              width,
                                              face_moved_vertices_camera[:, :, :, 2],
                                              face_moved_vertices_image,
                                              face_attributes,
                                              face_moved_normals_z)
            loss = torch.mean(torch.abs(imfeat - images_gt))
            loss.backward()

            optimizer.step()

        moved_vertices_camera = rotate_translate_points(moved_vertices, camera_rot, camera_trans)
        moved_vertices_image = perspective_camera(moved_vertices_camera, camera_proj)

        # test that the loss went down
        assert loss < 0.001
        # We only test on image plan since we don't change camera angle during training we don't expect depth to be correct.
        # We could probably fine-tune the test to have a lower tolerance (TODO: cfujitsang)
        assert torch.allclose(moved_vertices_image, vertices_image, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("height", [256])
@pytest.mark.parametrize("width", [512])
class TestDIBRGrad:
    @pytest.fixture(autouse=True)
    def vertices(self, device):
        # shape: (batch_size, num_vertices, 3)
        return torch.tensor(
            [[[-1, -1, -1], [1, -1, 0], [1, 1, -1], [-1, 1, 0]],
             [[-1, -1, 0], [1, -1, -1], [1, 1, 0], [-1, 1, -1]]],
            dtype=torch.double,
            device=device)

    @pytest.fixture(autouse=True)
    def faces(self, device):
        # shape: (num_faces, 3)
        return torch.tensor([[0, 1, 2], [0, 2, 3]],
                            dtype=torch.long,
                            device=device)

    @pytest.fixture(autouse=True)
    def vertex_colors(self, device):
        # shape: (batch_size, num_vertices, 3)
        return torch.tensor([[[0.9, 0.1, 0.1],
                              [0.1, 0.9, 0.1],
                              [0.1, 0.1, 0.9],
                              [0.9, 0.9, 0.1]],
                             [[0.1, 0.9, 0.1],
                              [0.9, 0.1, 0.1],
                              [0.9, 0.9, 0.1],
                              [0.1, 0.1, 0.9]]],
                            dtype=torch.double,
                            device=device)

    @pytest.fixture(autouse=True)
    def camera_trans(self, device):
        # shape: (batch_size, 3)
        return torch.tensor([[0, 0, 4]],
                            dtype=torch.double,
                            device=device).repeat(2, 1)

    @pytest.fixture(autouse=True)
    def camera_rot(self, device):
        # shape: (batch_size, 3, 3)
        return torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]],
                            dtype=torch.double,
                            device=device).repeat(2, 1, 1)

    @pytest.fixture(autouse=True)
    def camera_proj(self, width, height, device):
        # shape: (3, 1)
        return torch.tensor([[2.5 / (width / height)], [2.5], [-1]],
                            dtype=torch.double,
                            device=device)

    @pytest.fixture(autouse=True)
    def camera_proj_zoom(self, width, height, device):
        # very small field of view
        return torch.tensor([[10. / (width / height)], [10.], [-1]],
                            dtype=torch.double,
                            device=device)

    @pytest.fixture(autouse=True)
    def vertices_camera(self, vertices, camera_rot, camera_trans):
        return rotate_translate_points(vertices, camera_rot, camera_trans)

    @pytest.fixture(autouse=True)
    def vertices_image(self, vertices_camera, camera_proj):
        return perspective_camera(vertices_camera, camera_proj)

    @pytest.fixture(autouse=True)
    def face_vertices_camera(self, vertices_camera, faces):
        return index_vertices_by_faces(vertices_camera, faces)

    @pytest.fixture(autouse=True)
    def face_vertices_image(self, vertices_image, faces):
        return index_vertices_by_faces(vertices_image, faces)

    @pytest.fixture(autouse=True)
    def face_camera_normals_z(self, face_vertices_camera):
        face_normals_unit = face_normals(face_vertices_camera, unit=True)
        return face_normals_unit[:, :, 2]

    @pytest.fixture(autouse=True)
    def face_vertex_colors(self, vertex_colors, faces):
        face_vertex_colors = index_vertices_by_faces(vertex_colors, faces).detach()
        face_vertex_colors.requires_grad = True
        return face_vertex_colors
    #########################################
    #### TEST GRADIENTS ON FACE_FEATURES ####
    #########################################
    # This should always works regardless of the mesh
    def test_face_vertex_colors_gradcheck(self, height, width, face_vertices_camera,
                                          face_vertices_image, face_vertex_colors,
                                          face_camera_normals_z):
        torch.autograd.gradcheck(
            dibr_rasterization,
            inputs=(20, 30, face_vertices_camera[:, :, :, 2], face_vertices_image,
                    face_vertex_colors, face_camera_normals_z),
            raise_exception=True)

    @pytest.fixture(autouse=True)
    def target_face_vertex_colors_grad(self, height, width, face_vertices_camera,
                                       face_vertices_image, face_vertex_colors,
                                       face_camera_normals_z):
        _face_vertex_colors = face_vertex_colors.detach()
        _face_vertex_colors.requires_grad = True
        # This assume that test_vertex_colors_gradcheck pass
        outputs = dibr_rasterization(20, 30, face_vertices_camera[:, :, :, 2],
                                     face_vertices_image, _face_vertex_colors,
                                     face_camera_normals_z)
        # gradients will be provided through the two differentiable outputs
        output = sum([torch.sum(output) for output in outputs])
        output.backward()
        return _face_vertex_colors.grad.clone()

    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    def test_face_vertex_colors_grads(self, height, width, face_vertices_camera,
                                      face_vertices_image, face_vertex_colors,
                                      face_camera_normals_z, dtype,
                                      target_face_vertex_colors_grad):
        _face_vertex_colors = face_vertex_colors.detach()
        _face_vertex_colors.requires_grad = True
        outputs = dibr_rasterization(20,
                                     30,
                                     face_vertices_camera.to(dtype)[:, :, :, 2],
                                     face_vertices_image.to(dtype),
                                     _face_vertex_colors.to(dtype),
                                     face_camera_normals_z.to(dtype))
        # gradients will be provided through the two differentiable outputs
        output = sum([torch.sum(output) for output in outputs])
        output.backward()
        assert torch.allclose(_face_vertex_colors.grad, target_face_vertex_colors_grad)


    ######################################################
    #### TEST GRADIENTS ON FACE_FEATURES AND VERTICES ####
    ######################################################
    # We also test for gradients on vertices,
    # it only works if the mesh is filling the rendered image,
    # because the gradients generated on void is analytically wrong
    @pytest.fixture(autouse=True)
    def vertices_images_zoom(self, vertices_camera, camera_proj_zoom):
        # the two faces are fully covering the camera
        return perspective_camera(vertices_camera, camera_proj_zoom)

    @pytest.fixture(autouse=True)
    def face_vertices_image_zoom(self, vertices_images_zoom, faces):
        face_vertices_image_zoom = index_vertices_by_faces(vertices_images_zoom, faces)
        face_vertices_image_zoom.requires_grad = True
        return face_vertices_image_zoom

    def test_all_gradcheck(self, height, width, face_vertices_camera,
                           face_vertices_image_zoom, face_vertex_colors,
                           face_camera_normals_z):
        _face_vertex_colors = face_vertex_colors.detach()
        _face_vertex_colors.requires_grad = True
        _face_vertices_image_zoom = face_vertices_image_zoom.detach()
        _face_vertices_image_zoom.requires_grad = True

        torch.autograd.gradcheck(
            dibr_rasterization,
            inputs=(20, 30, face_vertices_camera[:, :, :, 2],
                    _face_vertices_image_zoom,
                    _face_vertex_colors, face_camera_normals_z),
            raise_exception=True)

    @pytest.fixture(autouse=True)
    def target_all_grads(self, height, width, face_vertices_camera,
                         face_vertices_image_zoom, face_vertex_colors,
                         face_camera_normals_z):
        _face_vertex_colors = face_vertex_colors.detach()
        _face_vertex_colors.requires_grad = True
        _face_vertices_image_zoom = face_vertices_image_zoom.detach()
        _face_vertices_image_zoom.requires_grad = True
        # This assume that test_vertex_colors_gradcheck pass
        outputs = dibr_rasterization(20, 30, face_vertices_camera[:, :, :, 2],
                                     _face_vertices_image_zoom, _face_vertex_colors,
                                     face_camera_normals_z)
        # gradients will be provided through the two differentiable outputs
        output = sum([torch.sum(output) for output in outputs])
        output.backward()
        return _face_vertices_image_zoom.grad.clone().detach(), _face_vertex_colors.grad.clone().detach()

    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    def test_all_grads(self, height, width, face_vertices_camera,
                       face_vertices_image_zoom, face_vertex_colors,
                       face_camera_normals_z, dtype,
                       target_all_grads):
        _face_vertex_colors = face_vertex_colors.detach()
        _face_vertex_colors.requires_grad = True
        _face_vertices_image_zoom = face_vertices_image_zoom.detach()
        _face_vertices_image_zoom.requires_grad = True
        outputs = dibr_rasterization(20,
                                     30,
                                     face_vertices_camera.to(dtype)[:, :, :, 2],
                                     _face_vertices_image_zoom.to(dtype),
                                     _face_vertex_colors.to(dtype),
                                     face_camera_normals_z.to(dtype))
        # gradients will be provided through the two differentiable outputs
        output = sum([torch.sum(output) for output in outputs])
        output.backward()
        if dtype == torch.float:
            assert torch.allclose(target_all_grads[0], _face_vertices_image_zoom.grad,
                                  rtol=1e-5, atol=1e-4)
        else:
            # TODO(cfujitsang): non-determinism?
            assert torch.allclose(target_all_grads[0], _face_vertices_image_zoom.grad)
        assert torch.allclose(target_all_grads[1], _face_vertex_colors.grad)
