# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

def test_softras(device='cuda:0'):

    renderer = kaolin.graphics.SoftRenderer(camera_mode='look_at', device=device)
    filename_input = 'tests/model.obj'
    
    mesh = kaolin.rep.TriangleMesh.from_obj(filename_input)
    vertices = mesh.vertices.float()
    faces = mesh.faces.long()
    face_textures = faces.clone()
    vertices = vertices[None, :, :].cuda()
    faces = faces[None, :, :].cuda()
    face_textures = face_textures[None, :, :].cuda()
    textures = torch.ones(1, faces.shape[1], 2, 3, dtype=torch.float32).cuda()

    rgb, d, _, = renderer.forward(vertices, faces, textures)
    print(rgb.shape, d.shape)
    assert rgb.shape == (1, 3, 256, 256) and d.shape == (1, 1, 256, 256)
