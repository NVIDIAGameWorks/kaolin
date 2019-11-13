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

import os
import math 

import torch
import numpy as np
import tqdm
import imageio
# from PIL import Image

import kaolin as kal
from kaolin.rep import TriangleMesh
import neural_renderer as nr


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
output_directory = os.path.join(data_dir, 'results')

output_directory_nmr = os.path.join(output_directory, 'nmr')
os.makedirs(output_directory_nmr, exist_ok=True)

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
    vertices_middle = (vertices_max + vertices_min)/2.
    vertices = vertices - vertices_middle
    
    coef = 5
    vertices = vertices * coef

    ###########################
    # NMR 
    ###########################
    textures = torch.ones(1, faces.shape[1], 2, 2, 2, 3, dtype=torch.float32).cuda()
    renderer = nr.Renderer(camera_mode='look_at')
    # loop = tqdm.tqdm(list(range(0, 360, 4)))
    # loop.set_description('Drawing NMR')
    # writer = imageio.get_writer(os.path.join(output_directory_nmr, 'rotation.gif'), mode='I')
    renderer.eye =  nr.get_points_from_angles(camera_distance, elevation, 0)
    images, _, _ = renderer(vertices, faces, textures)
    # for num, azimuth in enumerate(loop):
    #     renderer.eye =  nr.get_points_from_angles(camera_distance, elevation, azimuth)
    #     images, _, _ = renderer(vertices, faces, textures)
    #     image = images.detach().cpu().numpy()[0].transpose((1, 2, 0)) 
    #     writer.append_data((255*image).astype(np.uint8))
    # writer.close()


if __name__ == '__main__':
    main()
