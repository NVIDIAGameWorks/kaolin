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

import os
import math
import pytest
import numpy as np
import torch

from PIL import Image

import kaolin as kal

ROOT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.pardir, os.pardir, os.pardir, os.pardir, 'samples'
)

def _naive_sg_inner_product(intensity, direction, sharpness,
                            other_intensity, other_direction, other_sharpness):
    dm = math.sqrt(sum([(sharpness * direction[i] + other_sharpness * other_direction[i]) ** 2
                        for i in range(3)]))
    lm = sharpness + other_sharpness
    mul = math.exp(dm - lm)
    expo = [mul * intensity[i] * other_intensity[i] for i in range(3)]
    other = 1. - math.exp(-2. * dm)
    return [2. * math.pi * expo[i] * other / dm for i in range(3)]    
    
@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("num_sg", [1])
@pytest.mark.parametrize("num_other", [1])
class TestUnbatchedSgInnerProduct:
    @pytest.fixture(autouse=True)
    def intensity(self, num_sg, device, dtype):
        return torch.rand((num_sg, 3), device=device, dtype=dtype)

    @pytest.fixture(autouse=True)
    def direction(self, num_sg, device, dtype):
        return torch.rand((num_sg, 3), device=device, dtype=dtype)

    @pytest.fixture(autouse=True)
    def sharpness(self, num_sg, device, dtype):
        return torch.rand((num_sg), device=device, dtype=dtype)

    @pytest.fixture(autouse=True)
    def other_intensity(self, num_other, device, dtype):
        return torch.rand((num_other, 3), device=device, dtype=dtype)

    @pytest.fixture(autouse=True)
    def other_direction(self, num_other, device, dtype):
        return torch.rand((num_other, 3), device=device, dtype=dtype)

    @pytest.fixture(autouse=True)
    def other_sharpness(self, num_other, device, dtype):
        return torch.rand((num_other), device=device, dtype=dtype)

    def test_forward(self, intensity, direction, sharpness,
                     other_intensity, other_direction, other_sharpness):
        with torch.no_grad():
            expected_output = []
            for i in range(other_intensity.shape[0]):
                expected_output.append([])
                for j in range(intensity.shape[0]):
                    expected_output[-1].append(_naive_sg_inner_product(
                        intensity[j], direction[j], sharpness[j],
                        other_intensity[i], other_direction[i], other_sharpness[i]
                    ))
            expected_output = torch.tensor(expected_output,
                                           device=intensity.device,
                                           dtype=intensity.dtype)

            output = kal.render.lighting.sg.unbatched_sg_inner_product(
                intensity, direction, sharpness,
                other_intensity, other_direction, other_sharpness)
            assert torch.allclose(output, expected_output, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", ["cuda"])
@pytest.mark.parametrize("dtype", [torch.float])
@pytest.mark.parametrize("num_sg", [1, 17, 32, 511, 10000])
@pytest.mark.parametrize("num_other", [1, 17, 32, 511])
class TestUnbatchedReducedSgInnerProduct:
    @pytest.fixture(autouse=True)
    def intensity(self, num_sg, device, dtype):
        return torch.rand((num_sg, 3), device=device, dtype=dtype,
                          requires_grad=True)

    @pytest.fixture(autouse=True)
    def direction(self, num_sg, device, dtype):
        return torch.rand((num_sg, 3), device=device, dtype=dtype,
                          requires_grad=True)

    @pytest.fixture(autouse=True)
    def sharpness(self, num_sg, device, dtype):
        return torch.rand((num_sg), device=device, dtype=dtype,
                          requires_grad=True)

    @pytest.fixture(autouse=True)
    def other_intensity(self, num_other, device, dtype):
        return torch.rand((num_other, 3), device=device, dtype=dtype,
                          requires_grad=True)

    @pytest.fixture(autouse=True)
    def other_direction(self, num_other, device, dtype):
        return torch.rand((num_other, 3), device=device, dtype=dtype,
                          requires_grad=True)

    @pytest.fixture(autouse=True)
    def other_sharpness(self, num_other, device, dtype):
        return torch.rand((num_other), device=device, dtype=dtype,
                          requires_grad=True)

    def test_forward(self, intensity, direction, sharpness,
                     other_intensity, other_direction, other_sharpness):
        with torch.no_grad():
            expected_output = kal.render.lighting.sg.unbatched_sg_inner_product(
                intensity, direction, sharpness,
                other_intensity, other_direction, other_sharpness).sum(1)
            output = kal.render.lighting.sg.unbatched_reduced_sg_inner_product(
                intensity, direction, sharpness,
                other_intensity, other_direction, other_sharpness)
            assert torch.allclose(output, expected_output, rtol=1e-4, atol=1e-4)

    def test_backward(self, intensity, direction, sharpness,
                      other_intensity, other_direction, other_sharpness):
        gt_intensity = intensity.detach()
        gt_intensity.requires_grad = True
        gt_direction = direction.detach()
        gt_direction.requires_grad = True
        gt_sharpness = sharpness.detach()
        gt_sharpness.requires_grad = True
        gt_other_intensity = other_intensity.detach()
        gt_other_intensity.requires_grad = True
        gt_other_direction = other_direction.detach()
        gt_other_direction.requires_grad = True
        gt_other_sharpness = other_sharpness.detach()
        gt_other_sharpness.requires_grad = True
        gt_output = kal.render.lighting.sg.unbatched_sg_inner_product(
            gt_intensity, gt_direction, gt_sharpness,
            gt_other_intensity, gt_other_direction, gt_other_sharpness).sum(1)
        output = kal.render.lighting.sg.unbatched_reduced_sg_inner_product(
            intensity, direction, sharpness,
            other_intensity, other_direction, other_sharpness)
        grad_out = torch.rand_like(gt_output)
        gt_output.backward(grad_out)
        output.backward(grad_out)
        assert torch.allclose(intensity.grad, gt_intensity.grad,
                              rtol=1e-4, atol=1e-4)
        assert torch.allclose(direction.grad, gt_direction.grad,
                              rtol=1e-4, atol=1e-4)
        assert torch.allclose(sharpness.grad, gt_sharpness.grad,
                              rtol=1e-4, atol=1e-4)
        assert torch.allclose(other_intensity.grad, gt_other_intensity.grad,
                              rtol=1e-4, atol=1e-4)
        assert torch.allclose(other_direction.grad, gt_other_direction.grad,
                              rtol=1e-4, atol=1e-4)
        assert torch.allclose(other_sharpness.grad, gt_other_sharpness.grad,
                              rtol=1e-4, atol=1e-4)

def _generate_pinhole_rays_dir(camera, device='cuda'):
    """Ray direction generation function for pinhole cameras.

    This function assumes that the principal point (the pinhole location) is specified by a 
    displacement (camera.x0, camera.y0) in pixel coordinates from the center of the image. 
    The Kaolin camera class does not enforce a coordinate space for how the principal point is specified,
    so users will need to make sure that the correct principal point conventions are followed for 
    the cameras passed into this function.

    Args:
        height (int): The resolution height.
        width (int): The resolution width.
        camera (kaolin.render.camera.Camera): The camera instance, should be of batch == 1.
    Returns:
        (torch.Tensor): the rays directions, of shape (height, width, 3)
    """
    # Generate centered grid
    pixel_y, pixel_x = torch.meshgrid(
        torch.arange(camera.height, device=device),
        torch.arange(camera.width, device=device),
    )
    pixel_x = pixel_x + 0.5  # scale and add bias to pixel center
    pixel_y = pixel_y + 0.5  # scale and add bias to pixel center

    # Account for principal point (offsets from the center)
    pixel_x = pixel_x - camera.x0
    pixel_y = pixel_y + camera.y0

    # pixel values are now in range [-1, 1], both tensors are of shape res_y x res_x
    # Convert to NDC
    pixel_x = 2 * (pixel_x / camera.width) - 1.0
    pixel_y = 2 * (pixel_y / camera.height) - 1.0

    ray_dir = torch.stack((pixel_x * camera.tan_half_fov(kal.render.camera.intrinsics.CameraFOV.HORIZONTAL),
                           -pixel_y * camera.tan_half_fov(kal.render.camera.intrinsics.CameraFOV.VERTICAL),
                           -torch.ones_like(pixel_x)), dim=-1)

    ray_dir = ray_dir.reshape(-1, 3)    # Flatten grid rays to 1D array
    ray_orig = torch.zeros_like(ray_dir)

    # Transform from camera to world coordinates
    ray_orig, ray_dir = camera.extrinsics.inv_transform_rays(ray_orig, ray_dir)
    ray_dir /= torch.linalg.norm(ray_dir, dim=-1, keepdim=True)

    return ray_dir[0].reshape(camera.height, camera.width, 3)

@pytest.mark.parametrize('scene_idx,azimuth,elevation,amplitude,sharpness', [
    (0, torch.tensor([0., math.pi / 2.], device='cuda'), torch.tensor([0., 0.], device='cuda'),
     torch.tensor([[5., 2., 2.], [5., 10., 5.]], device='cuda'), torch.tensor([6., 20.], device='cuda')),
    (1, torch.tensor([0., 0.], device='cuda'), torch.tensor([-math.pi / 2., math.pi / 2.], device='cuda'),
     torch.tensor([[3., 3., 7.], [8., 8., 1.]], device='cuda'), torch.tensor([5., 40.], device='cuda'))
])
class TestRenderLighting:

    @pytest.fixture(autouse=True, scope='class')
    def rasterization_output(self):
        MODEL_PATH = os.path.join(ROOT_DIR, 'colored_sphere.obj')
        obj = kal.io.obj.import_mesh(MODEL_PATH, with_materials=True, with_normals=True)

        vertices = obj.vertices.cuda().unsqueeze(0)
        # Normalize vertices in [-0.5, 0.5] range
        vertices_max = vertices.max(dim=1, keepdim=True)[0]
        vertices_min = vertices.min(dim=1, keepdim=True)[0]
        vertices = ((vertices - vertices_min) / (vertices_max - vertices_min)) - 0.5
        
        faces = obj.faces.cuda()
        num_faces = faces.shape[0]
        num_vertices = vertices.shape[1]
        face_vertices = kal.ops.mesh.index_vertices_by_faces(vertices, faces)
        # Face normals w.r.t to the world coordinate system
        face_normals_idx = obj.face_normals_idx.cuda()
        normals = obj.normals.cuda().unsqueeze(0)
        face_world_normals = kal.ops.mesh.index_vertices_by_faces(normals, face_normals_idx)

        face_uvs_idx = obj.face_uvs_idx.cuda()
        uvs = obj.uvs.cuda().unsqueeze(0)
        face_uvs = kal.ops.mesh.index_vertices_by_faces(uvs, face_uvs_idx)
        # Take diffuse texture map component from materials
        diffuse_texture = obj.materials[0]['map_Kd'].cuda().float().permute(2, 0, 1).unsqueeze(0) / 255.
        cam_pos = torch.tensor([
            [0., 0., 1.],
            [0., -0.3, 0.9],
            [0., -1., 1.],
            [0., -0.999, 0.111],
            [0., 0.999, 0.111],
            [0.5, 0., 0.5]
        ], device='cuda')
        nb_views = cam_pos.shape[0]
        cam_pos = cam_pos / cam_pos.norm(dim=-1, keepdim=True)
        cams = kal.render.camera.Camera.from_args(
            eye=cam_pos,
            at=torch.tensor([[0., 0., 0.]], device='cuda').repeat(nb_views, 1),
            up=torch.tensor([[0., 1., 0.]], device='cuda').repeat(nb_views, 1),
            fov=70. * 2. * math.pi / 360,
            width=256, height=256, device='cuda'
        )
        vertices_camera = cams.extrinsics.transform(vertices)
        vertices_ndc = cams.intrinsics.transform(vertices_camera)
        face_vertices_camera = kal.ops.mesh.index_vertices_by_faces(vertices_camera, faces)
        face_vertices_image = kal.ops.mesh.index_vertices_by_faces(vertices_ndc[..., :2], faces)
        face_vertices_z = face_vertices_camera[..., -1]
        
        # Compute the rays
        rays_d = []
        for cam in cams:
            rays_d.append(_generate_pinhole_rays_dir(cam))
        # Rays must be toward the camera
        rays_d = -torch.stack(rays_d, dim=0)
        imsize = 256
        face_vertices = kal.ops.mesh.index_vertices_by_faces(vertices, faces)
        im_features, face_idx = kal.render.mesh.rasterize(
            imsize, imsize, face_vertices_camera[..., -1], face_vertices_image,
            [face_uvs.repeat(nb_views, 1, 1, 1), face_world_normals.repeat(nb_views, 1, 1, 1)]
        )
        hard_mask = face_idx != -1
        hard_mask = hard_mask
        uv_map = im_features[0]
        im_world_normal = im_features[1] / torch.sqrt(torch.sum(im_features[1] * im_features[1], dim=-1, keepdim=True))
        albedo = kal.render.mesh.texture_mapping(uv_map, diffuse_texture.repeat(nb_views, 1, 1, 1))
        albedo = torch.clamp(albedo * hard_mask.unsqueeze(-1), min=0., max=1.)
        return {
            'albedo': albedo,
            'im_world_normal': im_world_normal,
            'hard_mask': hard_mask,
            'roughness': hard_mask * 0.1,
            'rays_d': rays_d
        }

    @pytest.fixture(autouse=True, scope='class')
    def albedo(self, rasterization_output):
        return rasterization_output['albedo']

    @pytest.fixture(autouse=True, scope='class')
    def im_world_normal(self, rasterization_output):
        return rasterization_output['im_world_normal']

    @pytest.fixture(autouse=True, scope='class')
    def hard_mask(self, rasterization_output):
        return rasterization_output['hard_mask']

    @pytest.fixture(autouse=True, scope='class')
    def roughness(self, rasterization_output):
        return rasterization_output['roughness']

    @pytest.fixture(autouse=True, scope='class')
    def rays_d(self, rasterization_output):
        return rasterization_output['rays_d']

    def test_diffuse_inner_product(self, scene_idx, azimuth, elevation, amplitude, sharpness,
                                   albedo, im_world_normal, hard_mask):
        directions = torch.stack(kal.ops.coords.spherical2cartesian(azimuth, elevation), dim=-1).cuda()
        img = torch.zeros_like(im_world_normal)
        lighting_effect = kal.render.lighting.sg_diffuse_inner_product(
            amplitude, directions, sharpness,
            im_world_normal[hard_mask], albedo[hard_mask]
        )
        img[hard_mask] = lighting_effect

        gt = torch.stack([
            torch.from_numpy(np.array(Image.open(os.path.join(ROOT_DIR, 'render', 'sg', f'diffuse_inner_product_{scene_idx}_{j}.png'))))
            for j in range(6)
        ], dim=0).cuda().float() / 255.

        assert torch.allclose(torch.clamp(img, 0., 1.), gt, rtol=0., atol=1. / 255.)
    
    def test_diffuse_fitted(self, scene_idx, azimuth, elevation, amplitude, sharpness,
                            albedo, im_world_normal, hard_mask):
        directions = torch.stack(kal.ops.coords.spherical2cartesian(azimuth, elevation), dim=-1).cuda()
        img = torch.zeros_like(im_world_normal)
        lighting_effect = kal.render.lighting.sg_diffuse_fitted(
            amplitude, directions, sharpness,
            im_world_normal[hard_mask], albedo[hard_mask]
        )
        img[hard_mask] = lighting_effect

        gt = torch.stack([
            torch.from_numpy(np.array(Image.open(os.path.join(ROOT_DIR, 'render', 'sg', f'diffuse_fitted_{scene_idx}_{j}.png'))))
            for j in range(6)
        ], dim=0).cuda().float() / 255.

        assert torch.allclose(torch.clamp(img, 0., 1.), gt, rtol=0., atol=1. / 255.)

    def test_specular(self, scene_idx, azimuth, elevation, amplitude, sharpness,
                      albedo, im_world_normal, roughness, rays_d, hard_mask):
        directions = torch.stack(kal.ops.coords.spherical2cartesian(azimuth, elevation), dim=-1).cuda()
        img = torch.zeros_like(im_world_normal)
        lighting_effect = kal.render.lighting.sg_warp_specular_term(
            amplitude, directions, sharpness,
            im_world_normal[hard_mask], roughness[hard_mask], rays_d[hard_mask], albedo[hard_mask]
        )
        img[hard_mask] = lighting_effect

        gt = torch.stack([
            torch.from_numpy(np.array(Image.open(os.path.join(ROOT_DIR, 'render', 'sg', f'specular_{scene_idx}_{j}.png'))))
            for j in range(6)
        ], dim=0).cuda().float() / 255.

        assert torch.allclose(torch.clamp(img, 0., 1.), gt, rtol=0., atol=1. / 255.)
