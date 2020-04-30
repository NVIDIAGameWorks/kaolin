# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import pytest
import torch
import kaolin


@pytest.mark.skipif("not torch.cuda.is_available()", reason="softras needs GPU")
def test_softras(device='cuda:0'):

    renderer = kaolin.graphics.SoftRenderer(camera_mode="look_at", device=device)
    filename_input = "tests/graphics/banana.obj"

    mesh = kaolin.rep.TriangleMesh.from_obj(filename_input)

    vertices = mesh.vertices
    faces = mesh.faces.int()
    face_textures = (faces).clone()
    vertices = vertices[None, :, :].cuda()  
    faces = faces[None, :, :].cuda()
    face_textures[None, :, :].cuda()
    vertices_max = vertices.max()
    vertices_min = vertices.min()
    vertices_middle = (vertices_max + vertices_min) / 2.
    vertices = vertices - vertices_middle
    coef = 5
    vertices = vertices * coef
    textures = torch.ones(1, faces.shape[1], 2, 3, dtype=torch.float32).cuda()
    renderer.set_eye_from_angles(2., 30., 0.)

    rgba = renderer.forward(vertices, faces, textures)
    assert rgba.shape == (1, 4, 256, 256)

    # Compare correctness with a reference softras output
    target = torch.load("tests/graphics/softras_reference_render.pt").to(device)
    avgmse = (rgba - target).abs().mean()
    assert avgmse <= 0.0002
