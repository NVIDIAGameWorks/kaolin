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
import kaolin.io.utils
from kaolin.io.utils import read_image
from kaolin.utils.testing import assert_images_close

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
        # TODO: use Mesh API instead

        vertices = obj.vertices.cuda().unsqueeze(0)
        # Normalize vertices in [-0.5, 0.5] range
        vertices_max = vertices.max(dim=1, keepdim=True)[0]
        vertices_min = vertices.min(dim=1, keepdim=True)[0]
        vertices = ((vertices - vertices_min) / (vertices_max - vertices_min)) - 0.5
        
        faces = obj.faces.cuda()
        # Face normals w.r.t to the world coordinate system
        face_normals_idx = obj.face_normals_idx.cuda()
        normals = obj.normals.cuda().unsqueeze(0)
        face_world_normals = kal.ops.mesh.index_vertices_by_faces(normals, face_normals_idx)

        face_uvs_idx = obj.face_uvs_idx.cuda()
        uvs = obj.uvs.cuda().unsqueeze(0)
        uvs[..., 1] = 1 - uvs[..., 1]
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
        
        # Compute the rays
        rays_d = []
        for cam in cams:
            _, per_cam_ray_dirs = kal.render.camera.raygen.generate_pinhole_rays(cam)
            per_cam_ray_dirs = per_cam_ray_dirs.reshape(cam.height, cam.width, 3)
            rays_d.append(per_cam_ray_dirs)
        # Rays must be toward the camera
        rays_d = -torch.stack(rays_d, dim=0)
        imsize = 256
        im_features, face_idx = kal.render.mesh.rasterize(
            imsize, imsize, face_vertices_camera[..., -1], face_vertices_image,
            [face_uvs.repeat(nb_views, 1, 1, 1), face_world_normals.repeat(nb_views, 1, 1, 1)]
        )
        hard_mask = face_idx != -1
        hard_mask = hard_mask
        uv_map = im_features[0]
        im_world_normal = im_features[1] / torch.sqrt(
            torch.sum(im_features[1] * im_features[1], dim=-1, keepdim=True))
        albedo = kal.render.mesh.texture_mapping(uv_map, diffuse_texture.repeat(nb_views, 1, 1, 1))
        albedo = torch.clamp(albedo * hard_mask.unsqueeze(-1), min=0., max=1.)

        res = {
            'albedo': albedo,
            'im_world_normal': im_world_normal,
            'hard_mask': hard_mask,
            'roughness': hard_mask * 0.1,
            'rays_d': rays_d
        }
        return res

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


class TestSgLightingParameters:
    def test_basics(self):
        # Test can instantiate with default arguments
        default_light = kal.render.lighting.SgLightingParameters()
        assert default_light.amplitude is not None
        assert default_light.direction is not None
        assert default_light.sharpness is not None

        # Test can set amplitude as both number and tensor
        light = kal.render.lighting.SgLightingParameters(amplitude=4.0)
        assert torch.allclose(light.amplitude, torch.full((1, 3), 4.0, dtype=torch.float))

        light = kal.render.lighting.SgLightingParameters(amplitude=(1.0, 4.0, 3.0))
        assert torch.allclose(light.amplitude, torch.tensor([[1.0, 4.0, 3.0]], dtype=torch.float))

        light = kal.render.lighting.SgLightingParameters(amplitude=torch.tensor([1.0, 4.0, 3.0], dtype=torch.float))
        assert torch.allclose(light.amplitude, torch.tensor([[1.0, 4.0, 3.0]], dtype=torch.float))

        light = kal.render.lighting.SgLightingParameters(amplitude=4.0, direction=torch.rand((15, 3), dtype=torch.float))
        assert torch.allclose(light.amplitude, torch.full((15, 3), 4.0, dtype=torch.float))

        # Test set sharpness
        light = kal.render.lighting.SgLightingParameters(sharpness=4.0)
        assert torch.allclose(light.sharpness, torch.full((1,), 4.0, dtype=torch.float))

        light = kal.render.lighting.SgLightingParameters(sharpness=(1.0, 4.0, 3.0))
        assert torch.allclose(light.sharpness, torch.tensor([1.0, 4.0, 3.0], dtype=torch.float))

        light = kal.render.lighting.SgLightingParameters(sharpness=4.0,
                                                         direction=torch.rand((10, 3), dtype=torch.float))
        assert torch.allclose(light.sharpness, torch.full((10,), 4.0, dtype=torch.float))

        # Test set all
        ampl = torch.rand((6, 3), dtype=torch.float)
        directions = torch.rand((6, 3), dtype=torch.float)
        sharp = torch.rand((6,), dtype=torch.float)
        orig_device = ampl.device
        light = kal.render.lighting.SgLightingParameters(sharpness=sharp, amplitude=ampl, direction=directions)
        assert torch.allclose(light.amplitude, ampl)
        assert torch.allclose(light.sharpness, sharp)
        assert torch.allclose(light.direction, torch.nn.functional.normalize(directions))

        # Test to device conversions ---
        def _assert_device(obj, device):
            assert obj.amplitude.device.type == device.type
            assert obj.sharpness.device.type == device.type
            assert obj.direction.device.type == device.type

        light_cuda = light.cuda()
        _assert_device(light, orig_device)
        _assert_device(light_cuda, torch.device("cuda"))
        light_cuda = light.to("cuda")
        _assert_device(light_cuda, torch.device("cuda"))
        light_cpu = light_cuda.cpu()
        _assert_device(light_cpu, torch.device("cpu"))
        assert kal.utils.testing.contained_torch_equal(light, light_cpu, approximate=True, print_error_context="light")
        light_cpu = light_cuda.to("cpu")
        _assert_device(light_cpu, torch.device("cpu"))
        assert kal.utils.testing.contained_torch_equal(light, light_cpu, approximate=True, print_error_context="light")

    def test_from_sun(self):
        # default
        direction = torch.rand((5, 3), dtype=torch.float)
        light = kal.render.lighting.SgLightingParameters.from_sun(direction)
        assert kal.utils.testing.check_tensor(light.direction, shape=(5, 3))
        assert kal.utils.testing.check_tensor(light.amplitude, shape=(5, 3))
        assert kal.utils.testing.check_tensor(light.sharpness, shape=(5,))

        light = kal.render.lighting.SgLightingParameters.from_sun(direction, strength=(1, 2, 3, 4, 5))
        assert kal.utils.testing.check_tensor(light.direction, shape=(5, 3))
        assert kal.utils.testing.check_tensor(light.amplitude, shape=(5, 3))
        assert kal.utils.testing.check_tensor(light.sharpness, shape=(5,))

        strength = torch.rand((7,), dtype=torch.float) * 10
        direction = torch.rand((7, 3), dtype=torch.float)
        angle = torch.rand((7,), dtype=torch.float) * math.pi
        color = torch.rand((7, 3), dtype=torch.float)
        # without color
        light = kal.render.lighting.SgLightingParameters.from_sun(direction, strength=strength, angle=angle)
        assert torch.allclose(light.amplitude, strength.unsqueeze(-1).repeat(1,3))
        assert kal.utils.testing.check_tensor(light.sharpness, shape=(7,))
        assert torch.allclose(light.direction, torch.nn.functional.normalize(direction))

        # with color
        light = kal.render.lighting.SgLightingParameters.from_sun(direction, strength=strength, angle=angle, color=color)
        assert torch.allclose(light.amplitude, strength.unsqueeze(-1) * color)
        assert kal.utils.testing.check_tensor(light.sharpness, shape=(7,))
        assert torch.allclose(light.direction, torch.nn.functional.normalize(direction))


class TestUtilities:
    def test_sg_from_sun(self):
        strength = torch.rand((7,), dtype=torch.float) * 10 + 1.
        direction = torch.rand((7, 3), dtype=torch.float)
        angle = torch.rand((7,), dtype=torch.float) * math.pi
        color = torch.rand((7, 3), dtype=torch.float)

        amplitude, direction_out, sharpness = kal.render.lighting.sg_from_sun(direction, strength, angle, color)
        assert torch.allclose(amplitude, strength.unsqueeze(-1) * color)
        assert torch.allclose(direction, direction_out)
        assert torch.allclose(sharpness, torch.log(0.5 / strength) / (torch.cos(angle / 2) - 1))

        angle[:] = math.pi * 2
        amplitude, direction_out, sharpness = kal.render.lighting.sg_from_sun(direction, strength, angle, color)
        assert torch.all(torch.logical_not(torch.isnan(sharpness)))
        assert torch.all(torch.logical_not(torch.isinf(sharpness)))
        # assert torch.all(sharpness > 0)

        angle[:] = 0
        amplitude, direction_out, sharpness = kal.render.lighting.sg_from_sun(direction, strength, angle, color)
        assert torch.all(torch.logical_not(torch.isnan(sharpness)))
        # TODO(Clement): fix this one
        # assert torch.all(torch.logical_not(torch.isinf(sharpness)))  # Fails
        # assert torch.all(sharpness > 0)

        # Larger angle should reduce sharpness
        angle = torch.rand((7,), dtype=torch.float) * math.pi
        angle[0] = math.pi
        _, _, sharpness1 = kal.render.lighting.sg_from_sun(direction, strength, angle, color)

        angle2 = angle + math.pi * 0.9
        angle2[0] = math.pi * 1.5
        _, _, sharpness2 = kal.render.lighting.sg_from_sun(direction, strength, angle2, color)
        assert torch.all(sharpness2 < sharpness1)

    def test_sg_direction_from_azimuth_elevation(self):
        azimuth = math.pi / 2
        elevation = math.pi / 3
        directions = kal.render.lighting.sg_direction_from_azimuth_elevation(azimuth, elevation)
        assert kal.utils.testing.check_tensor(directions, shape=(1, 3))

        azimuth = torch.rand((6,), dtype=torch.float) * math.pi
        elevation = torch.rand((6,), dtype=torch.float) * math.pi
        directions = kal.render.lighting.sg_direction_from_azimuth_elevation(azimuth, elevation)
        assert kal.utils.testing.check_tensor(directions, shape=(6, 3))
