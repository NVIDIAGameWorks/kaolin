# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import graphics
from graphics.utils.utils_perspective import lookatnp, perspectiveprojectionnp
from graphics.utils.utils_sphericalcoord import get_spherical_coords_x
from graphics.render.base import Render as Dib_Renderer
import os
import sys
import math

import torch
import numpy as np
import tqdm
import imageio
# from PIL import Image

import kaolin as kal
from kaolin.rep import TriangleMesh

sys.path.append(str(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../DIB-R')))


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
output_directory = os.path.join(data_dir, 'results')

output_directory_dib = os.path.join(output_directory, 'dib')
os.makedirs(output_directory_dib, exist_ok=True)


def main():
    filename_input = os.path.join(data_dir, 'banana.obj')
    filename_output = os.path.join(output_directory, 'example1.gif')

    ###########################
    # camera settings
    ###########################
    camera_distance = 2
    elevation = 30

    ###########################
    # load object
    ###########################
    mesh = TriangleMesh.from_obj(filename_input)
    vertices = mesh.vertices
    faces = mesh.faces.int()
    face_textures = (faces).clone()

    vertices = vertices[None, :, :].cuda()
    faces = faces[None, :, :].cuda()
    face_textures[None, :, :].cuda()

    ###########################
    # normalize verts
    ###########################
    vertices_max = vertices.max()
    vertices_min = vertices.min()
    vertices_middle = (vertices_max + vertices_min) / 2.
    vertices = vertices - vertices_middle

    coef = 5
    vertices = vertices * coef

    ###########################
    # DIB-Renderer
    ###########################
    renderer = Dib_Renderer(256, 256, mode='VertexColor')
    textures = torch.ones(1, vertices.shape[1], 3).cuda()
    loop = tqdm.tqdm(list(range(0, 360, 4)))
    loop.set_description('Drawing Dib_Renderer VertexColor')
    writer = imageio.get_writer(os.path.join(output_directory_dib, 'rotation_VertexColor.gif'), mode='I')
    for azimuth in loop:
        renderer.set_look_at_parameters([90 - azimuth], [elevation], [camera_distance])
        predictions, _, _ = renderer.forward(points=[vertices, faces[0].long()], colors=[textures])
        image = predictions.detach().cpu().numpy()[0]
        writer.append_data((image * 255).astype(np.uint8))
    writer.close()


if __name__ == '__main__':
    main()
