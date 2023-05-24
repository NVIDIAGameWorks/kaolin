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

@pytest.mark.parametrize('scene_idx,azimuth,elevation', [
    (0, torch.tensor([0.], device='cuda'), torch.tensor([0.], device='cuda')),
    (1, torch.tensor([math.pi / 4.], device='cuda'), torch.tensor([math.pi / 2.], device='cuda'))
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
        normals = obj.normals.cuda().unsqueeze(0)
        face_normals_idx = obj.face_normals_idx.cuda()
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

    def test_diffuse_sh(self, scene_idx, azimuth, elevation,
                        albedo, im_world_normal, hard_mask):
        directions = torch.cat(kal.ops.coords.spherical2cartesian(azimuth, elevation), dim=-1).cuda()
        img = torch.zeros_like(im_world_normal)
        lighting_effect = kal.render.lighting.sh9_diffuse(
            directions, im_world_normal[hard_mask], albedo[hard_mask]
        )
        img[hard_mask] = lighting_effect

        gt = torch.stack([
            torch.from_numpy(np.array(Image.open(os.path.join(
                ROOT_DIR, 'render', 'sh', f'diffuse_{scene_idx}_{j}.png'
            )))) for j in range(6)
        ], dim=0).cuda().float() / 255.

@pytest.mark.parametrize('shape', [(1025,)])
class TestSh9:
    # Those a simply regression tests
    @classmethod
    def _naive_project_onto_sh9(cls, directions):
        if isinstance(directions, torch.Tensor):
            assert directions.shape[-1] == 3
            x, y, z = torch.split(directions, 1, dim=-1)
            band0 = torch.full_like(x, 0.28209479177)
        elif isinstance(directions, list):
            assert len(directions) == 3
            x, y, z = directions
            band0 = 0.28209479177
        else:
            raise TypeError(f"direction is a {type(direction)}, "
                            "must be a list or a torch.Tensor")
        # Band 1
        band1_m1 = -0.4886025119 * y
        band1_0 = 0.4886025119 * z
        band1_p1 = -0.4886025119 * x

        # Band 2
        band2_m2 = 1.0925484305920792 * (x * y)
        band2_m1 = -1.0925484305920792 * (y * z)
        band2_0 = 0.94617469575 * (z * z) - 0.31539156525
        band2_p1 = -1.0925484305920792 * x * z
        band2_p2 = 0.5462742152960396 * (x * x - y * y)

        if isinstance(directions, torch.Tensor):
            return torch.cat([
                band0,
                band1_m1, band1_0, band1_p1,
                band2_m2, band2_m1, band2_0, band2_p1, band2_p2
            ], dim=-1)
        else:
            return torch.tensor([
                band0,
                band1_m1, band1_0, band1_p1,
                band2_m2, band2_m1, band2_0, band2_p1, band2_p2
            ])

    @classmethod
    def _naive_sh9_irradiance(cls, lights, normals):
        is_batched = lights.ndim == 3
        assert lights.shape[-1] == 9
        bands = cls._naive_project_onto_sh9(normals)
        if is_batched:
            assert lights.shape[0] == normals.shape[0]
            num_scenes = lights.shape[0]
            bands = bands.reshape(num_scenes, -1, 9)
        else:
            bands = bands.reshape(-1, 9)

        bands[..., 0] *= math.pi
        bands[..., 1:4] *= 2. * math.pi / 3.
        bands[..., 4:] *= math.pi / 4.

        return torch.sum(bands * lights.unsqueeze(-2), dim=-1).reshape(*normals.shape[:-1])

    @pytest.fixture(autouse=True)
    def point_directions(self, shape):
        directions = torch.rand((*shape, 3), device='cuda') - 0.5
        directions /= (directions ** 2).sum(-1, keepdim=True)
        return directions

    @pytest.fixture(autouse=True)
    def light_directions(self):
        directions = torch.rand((3), device='cuda') - 0.5
        directions /= (directions ** 2).sum(-1, keepdim=True)
        return directions

    @pytest.fixture(autouse=True)
    def albedo(self, shape):
        return torch.rand((*shape, 3), device='cuda')

    @pytest.fixture(autouse=True)
    def expected_lights_sh9(self, light_directions):
        return self._naive_project_onto_sh9(light_directions)

    def test_project_onto_sh9(self, light_directions, expected_lights_sh9):
        output = kal.render.lighting.project_onto_sh9(light_directions)
        assert torch.equal(expected_lights_sh9, output)

    @pytest.fixture(autouse=True)
    def expected_irradiance(self, expected_lights_sh9, point_directions):
        return self._naive_sh9_irradiance(expected_lights_sh9, point_directions)

    def test_sh9_irradiance(self, expected_lights_sh9, point_directions, expected_irradiance):
        output = kal.render.lighting.sh9_irradiance(expected_lights_sh9, point_directions)
        assert torch.equal(output, expected_irradiance)

    def test_sh9_diffuse(self, light_directions, point_directions, albedo, expected_irradiance):
        output = kal.render.lighting.sh9_diffuse(light_directions, point_directions, albedo)
        expected_diffuse = albedo * expected_irradiance.unsqueeze(-1)
        assert torch.equal(output, expected_diffuse)


