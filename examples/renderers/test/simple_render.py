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

"""
Example 1. Drawing a teapot from multiple viewpoints.
"""
import os
import math 

import torch
import numpy as np
import tqdm
import imageio
from PIL import Image

import neural_renderer as nr
import soft_renderer as sr
import graphics 
from graphics.render.base import Render as Dib_Renderer
import kaolin as kal
from kaolin.rep import TriangleMesh 
from graphics.utils.utils_sphericalcoord import get_spherical_coords_x
from graphics.utils.utils_perspective import lookatnp, perspectiveprojectionnp



def obj_centened_camera_pos(dist, azimuth_deg, elevation_deg):
    phi = float(elevation_deg) / 180.0 * math.pi
    theta = float(azimuth_deg) / 180.0 * math.pi
    x = (dist * math.cos(theta) * math.cos(phi))
    y = (dist * math.sin(theta) * math.cos(phi))
    z = (dist * math.sin(phi))
    return (x, y, z)




current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
output_directory = os.path.join(data_dir, 'results')\

output_directory_nmr = os.path.join(output_directory, 'nmr')
os.makedirs(output_directory_nmr, exist_ok=True)

output_directory_sr = os.path.join(output_directory, 'sr')
os.makedirs(output_directory_sr, exist_ok=True)

output_directory_dib = os.path.join(output_directory, 'Dib')
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
    uvs = torch.FloatTensor(get_spherical_coords_x(vertices.data.numpy())) 
    face_textures = (faces).clone()
    
    vertices = vertices[None, :, :].cuda()  
    faces = faces[None, :, :].cuda() 
    uvs = uvs[None, :, :].cuda()
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
    loop = tqdm.tqdm(list(range(0, 360, 4)))
    loop.set_description('Drawing NMR')
    writer = imageio.get_writer(os.path.join(output_directory_nmr, 'rotation.gif'), mode='I')
    for num, azimuth in enumerate(loop):
        renderer.eye =  nr.get_points_from_angles(camera_distance, elevation, azimuth)
        images, _, _ = renderer(vertices, faces, textures)  
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0)) 
        writer.append_data((255*image).astype(np.uint8))
    writer.close()


    

    ###########################
    # Soft Rasterizer 
    ###########################
    textures = torch.ones(1, faces.shape[1], 2, 3, dtype=torch.float32).cuda()
    mesh = sr.Mesh(vertices, faces, textures)
    renderer = sr.SoftRenderer(camera_mode='look_at')
    loop = tqdm.tqdm(list(range(0, 360, 4)))
    loop.set_description('Drawing SR')
    writer = imageio.get_writer(os.path.join(output_directory_sr, 'rotation.gif'), mode='I')
    for azimuth in loop:
        mesh.reset_()
        renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
        images = renderer.render_mesh(mesh)
        image = images.detach().cpu().numpy()[0].transpose((1, 2, 0))
        writer.append_data((255*image).astype(np.uint8))
    writer.close()



    ################################
    # Dib-Renderer - Vertex Colours
    ################################ 
    renderer = Dib_Renderer(256, 256, mode = 'VertexColor')
    textures = torch.ones(1, vertices.shape[1], 3).cuda()
    loop = tqdm.tqdm(list(range(0, 360, 4)))
    loop.set_description('Drawing Dib_Renderer VertexColor')
    writer = imageio.get_writer(os.path.join(output_directory_dib, 'rotation_VertexColor.gif'), mode='I')
    for azimuth in loop:
        renderer.set_look_at_parameters([90-azimuth], [elevation], [camera_distance])
        predictions, _, _ = renderer.forward(points=[vertices, faces[0].long()], colors=[textures])
        image = predictions.detach().cpu().numpy()[0]
        writer.append_data((image*255).astype(np.uint8))
    writer.close()
 



    ################################
    # Dib-Renderer - Lambertian
    ################################  
    renderer = Dib_Renderer(256, 256, mode = 'Lambertian')
    textures = torch.ones(1, 3, 256, 256).cuda()
    loop = tqdm.tqdm(list(range(0, 360, 4)))
    loop.set_description('Drawing Dib_Renderer Lambertian')
    writer = imageio.get_writer(os.path.join(output_directory_dib, 'rotation_Lambertian.gif'), mode='I')
    for azimuth in loop:
        renderer.set_look_at_parameters([90-azimuth], [elevation], [camera_distance])
        predictions, _, _ = renderer.forward(points=[vertices, faces[0].long()], \
                                              colors=[uvs, face_textures.long(), textures])
        image = predictions.detach().cpu().numpy()[0]
        writer.append_data((image*255).astype(np.uint8))
    writer.close()


    ################################
    # Dib-Renderer - Phong
    ################################
    renderer = Dib_Renderer(256, 256, mode = 'Phong')
    textures = torch.ones(1, 3, 256, 256).cuda()

    ### Lighting info ###
    material = np.array([[0.1, 0.1, 0.1], 
                         [1.0, 1.0, 1.0],
                         [0.4, 0.4, 0.4]], dtype=np.float32).reshape(-1, 3, 3)
    material = torch.from_numpy(material).repeat(1, 1, 1).cuda()
    
    shininess = np.array([100], dtype=np.float32).reshape(-1, 1)
    shininess = torch.from_numpy(shininess).repeat(1, 1).cuda()

    lightdirect = 2 * np.random.rand(1, 3).astype(np.float32) - 1
    lightdirect[:, 2] += 2
    lightdirect = torch.from_numpy(lightdirect).cuda()
    
   
    loop = tqdm.tqdm(list(range(0, 360, 4)))
    loop.set_description('Drawing Dib_Renderer Phong')
    writer = imageio.get_writer(os.path.join(output_directory_dib, 'rotation_Phong.gif'), mode='I')
    for azimuth in loop:
        renderer.set_look_at_parameters([90-azimuth], [elevation], [camera_distance])
        predictions, _, _ = renderer.forward(points=[vertices, faces[0].long()], \
                                              colors=[uvs, face_textures.long(), textures],\
                                              light= lightdirect, \
                                              material=material, \
                                              shininess=shininess )
        image = predictions.detach().cpu().numpy()[0]
        writer.append_data((image*255).astype(np.uint8))
    writer.close()
 
   
    ################################
    # Dib-Renderer - SH
    ################################
    renderer = Dib_Renderer(256, 256, mode = 'SphericalHarmonics')
    textures = torch.ones(1, 3, 256, 256).cuda()
    
    ### Lighting info ###
    lightparam = np.random.rand(1, 9).astype(np.float32)
    lightparam[:, 0] += 2
    lightparam = torch.from_numpy(lightparam).cuda()
    
   
    loop = tqdm.tqdm(list(range(0, 360, 4)))
    loop.set_description('Drawing Dib_Renderer SH')
    writer = imageio.get_writer(os.path.join(output_directory_dib, 'rotation_SH.gif'), mode='I')
    for azimuth in loop:
        renderer.set_look_at_parameters([90-azimuth], [elevation], [camera_distance])
        predictions, _, _ = renderer.forward(points=[vertices, faces[0].long()], \
                                              colors=[uvs, face_textures.long(), textures],\
                                              light=lightparam)
        image = predictions.detach().cpu().numpy()[0]
        writer.append_data((image*255).astype(np.uint8))
    writer.close()

  






if __name__ == '__main__':
    main()
