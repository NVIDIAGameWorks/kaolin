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
Demo deform.
Deform template mesh based on input silhouettes and camera pose
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.io import imread, imsave
import os
import tqdm
import numpy as np
import imageio
import argparse

import neural_renderer as nr
import soft_renderer as sr
import kaolin as kal
from kaolin.rep import TriangleMesh 
import graphics 
from graphics.render.base import Render as Dib_Renderer
from graphics.utils.utils_sphericalcoord import get_spherical_coords_x
from graphics.utils.utils_perspective import lookatnp, perspectiveprojectionnp


current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(current_dir, 'data')
output_directory = os.path.join(data_dir, 'results')


output_directory_nmr = os.path.join(output_directory, 'nmr')
os.makedirs(output_directory_nmr, exist_ok=True)

output_directory_sr = os.path.join(output_directory, 'sr')
os.makedirs(output_directory_sr, exist_ok=True)

output_directory_dib = os.path.join(output_directory, 'Dib')
os.makedirs(output_directory_dib, exist_ok=True)

class Model(nn.Module):

    def __init__(self, vertices):
        super(Model, self).__init__()
        self.update = nn.Parameter(torch.rand(vertices.shape)*.001)
        self.verts = vertices
    def forward(self):
        return self.update + self.verts




def main():
    ###########################
    # camera settings
    ###########################
    camera_distance = 2.732
    elevation = 0
    azimuth = 0 
  
    ###########################
    # load object
    ###########################
    filename_input = os.path.join(data_dir, 'banana.obj')
    filename_ref = os.path.join(data_dir, 'example2_ref.png')
    image_gt = torch.from_numpy(imread(filename_ref).astype(np.float32).mean(-1) / 255.)[None, ::].cuda()


    mesh = TriangleMesh.from_obj(filename_input)
    vertices = mesh.vertices
    faces = mesh.faces.int()
    uvs = torch.FloatTensor(get_spherical_coords_x(vertices.data.numpy())) 
    face_textures = (faces).clone()


    pmax = vertices.max()
    pmin = vertices.min()
    pmiddle = (pmax + pmin) / 2
    vertices = vertices - pmiddle    
    coef = 10
    vertices = vertices * coef

    
    vertices = vertices[None, :, :].cuda()  
    faces = faces[None, :, :].cuda() 
    uvs = uvs[None, :, :].cuda()
    face_textures[None, :, :].cuda()

    ##########################
    # normalize verts
    ##########################
    vertices_max = vertices.max()
    vertices_min = vertices.min()
    vertices_middle = (vertices_max + vertices_min)/2.
    vertices = vertices - vertices_middle
    
    # coef = 5
    # vertices = vertices * coef



    ###########################
    # NMR 
    ###########################
    textures = torch.ones(1, faces.shape[1], 2,2,2, 3, dtype=torch.float32).cuda()
    model = Model(vertices.clone()).cuda()
    renderer = nr.Renderer(camera_mode='look_at')
    renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
    optimizer = torch.optim.Adam(model.parameters(), 0.001, betas=(0.5, 0.99))


    loop = tqdm.tqdm(range(2000))
    loop.set_description('Optimizing NMR')
    writer = imageio.get_writer(os.path.join(output_directory_nmr, 'deform.gif'), mode='I')
    for i in loop:
        optimizer.zero_grad()
        new_vertices = model() 
        image_pred  ,_, _= renderer(new_vertices, faces, textures)
        loss = torch.sum((image_pred - image_gt[None, :, :])**2)

        loss.backward()
        optimizer.step()
        loop.set_description('Loss: %.4f'%(loss.item()))
        if i % 20 == 0:
            image = image_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
            other_image = image_gt.detach().cpu().numpy().transpose((1, 2, 0))
           
            pass_image = image + other_image
            writer.append_data((128*pass_image).astype(np.uint8))


   

    ###########################
    # Soft Rasterizer 
    ###########################
    textures = torch.ones(1, faces.shape[1], 2, 3, dtype=torch.float32).cuda()
    model = Model(vertices.clone()).cuda()
    mesh = sr.Mesh(vertices, faces, textures)
    renderer = sr.SoftRenderer(image_size=256, sigma_val=3e-5, aggr_func_rgb='hard', 
                               camera_mode='look_at')
    renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
    optimizer = torch.optim.Adam(model.parameters(), 0.001, betas=(0.5, 0.99))


    loop = tqdm.tqdm(range(2000))
    loop.set_description('Optimizing SR')
    writer = imageio.get_writer(os.path.join(output_directory_sr, 'deform.gif'), mode='I')
    for i in loop:
        optimizer.zero_grad()
        new_vertices = model() 
        new_mesh = sr.Mesh(new_vertices, faces, textures)
        image_pred = renderer.render_mesh(new_mesh)
        loss = torch.sum((image_pred - image_gt[None, :, :])**2)
        loss.backward()
        optimizer.step()
        loop.set_description('Loss: %.4f'%(loss.item()))
        if i % 20 == 0:
            image = image_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
            other_image = image_gt.detach().cpu().numpy().transpose((1, 2, 0))
           
            pass_image = image + other_image
            writer.append_data((128*pass_image).astype(np.uint8))

    

    ################################
    # Dib-Renderer - Vertex Colours
    ################################
    model = Model(vertices.clone()).cuda()
    textures = torch.ones(1, vertices.shape[1], 3).cuda() 
    renderer = Dib_Renderer(256, 256, mode = 'VertexColor')
    renderer.set_look_at_parameters([90-azimuth], [elevation], [camera_distance])
    optimizer = torch.optim.Adam(model.parameters(), 0.001, betas=(0.5, 0.99))


    loop = tqdm.tqdm(range(2000))
    loop.set_description('Optimizing Dib_Renderer VertexColor')
    writer = imageio.get_writer(os.path.join(output_directory_dib, 'deform_VertexColor.gif'), mode='I')
    for i in loop:
        optimizer.zero_grad()
        new_vertices = model() 
        image_pred, alpha, _ = renderer.forward(points=[new_vertices, faces[0].long()], colors=[textures])

        image_pred = torch.cat((image_pred, alpha), dim = 3)
        image_pred = image_pred.permute(0,3,1,2)
        
        loss = torch.sum((image_pred - image_gt[None, :, :])**2) 
     
        loss.backward()
        optimizer.step()
       
        loop.set_description('Loss: %.4f'%(loss.item()))

        if i % 20 == 0:
            image = image_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
            other_image = image_gt.detach().cpu().numpy().transpose((1, 2, 0))
           
            pass_image = image + other_image
            writer.append_data((127*pass_image).astype(np.uint8))
    
    ################################
    # Dib-Renderer - Lambertian
    ################################
    model = Model(vertices.clone()).cuda()
    textures = torch.ones(1, 3, 256, 256).cuda()
    renderer = Dib_Renderer(256, 256, mode = 'Lambertian')
    renderer.set_look_at_parameters([90-azimuth], [elevation], [camera_distance])
    optimizer = torch.optim.Adam(model.parameters(), 0.001, betas=(0.5, 0.99))


    loop = tqdm.tqdm(range(2000))
    loop.set_description('Optimizing Dib_Renderer Lambertian')
    writer = imageio.get_writer(os.path.join(output_directory_dib, 'deform_Lambertian.gif'), mode='I')
    for i in loop:
        optimizer.zero_grad()
        new_vertices = model() 
        image_pred, alpha, _ = renderer.forward(points=[new_vertices, faces[0].long()], colors=[uvs, face_textures.long(), textures])
        image_pred = torch.cat((image_pred, alpha), dim = 3)
        image_pred = image_pred.permute(0,3,1,2)

        loss = torch.sum((image_pred - image_gt[None, :, :])**2) 
     
        loss.backward()
        optimizer.step()
       
        loop.set_description('Loss: %.4f'%(loss.item()))

        if i % 20 == 0:
            image = image_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
            other_image = image_gt.detach().cpu().numpy().transpose((1, 2, 0))
           
            pass_image = image + other_image
            writer.append_data((127*pass_image).astype(np.uint8))


    ################################
    # Dib-Renderer - Phong
    ################################
    model = Model(vertices.clone()).cuda()
    textures = torch.ones(1, 3, 256, 256).cuda() 
    renderer = Dib_Renderer(256, 256, mode = 'Phong')
    renderer.set_look_at_parameters([90-azimuth], [elevation], [camera_distance])
    optimizer = torch.optim.Adam(model.parameters(), 0.001, betas=(0.5, 0.99))

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

    loop = tqdm.tqdm(range(2000))
    loop.set_description('Optimizing Dib_Renderer Phong')
    writer = imageio.get_writer(os.path.join(output_directory_dib, 'deform_Phong.gif'), mode='I')
    for i in loop:
        optimizer.zero_grad()
        new_vertices = model() 
        image_pred, alpha, _ = renderer.forward(points=[new_vertices, faces[0].long()], \
                                              colors=[uvs, face_textures.long(), textures],\
                                              light= lightdirect, \
                                              material=material, \
                                              shininess=shininess)
        image_pred = torch.cat((image_pred, alpha), dim = 3)
        image_pred = image_pred.permute(0,3,1,2)

        loss = torch.sum((image_pred - image_gt[None, :, :])**2) 
     
        loss.backward()
        optimizer.step()
       
        loop.set_description('Loss: %.4f'%(loss.item()))

        if i % 20 == 0:
            image = image_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
            other_image = image_gt.detach().cpu().numpy().transpose((1, 2, 0))
           
            pass_image = image + other_image
            writer.append_data((127*pass_image).astype(np.uint8))


    ################################
    # Dib-Renderer - SphericalHarmonics
    ################################
    model = Model(vertices.clone()).cuda()
    textures = torch.ones(1, 3, 256, 256).cuda()
    renderer = Dib_Renderer(256, 256, mode = 'SphericalHarmonics')
    renderer.set_look_at_parameters([90-azimuth], [elevation], [camera_distance])
    optimizer = torch.optim.Adam(model.parameters(), 0.001, betas=(0.5, 0.99))

    lightparam = np.random.rand(1, 9).astype(np.float32)
    lightparam[:, 0] += 2
    lightparam = torch.from_numpy(lightparam).cuda()
    

    loop = tqdm.tqdm(range(2000))
    loop.set_description('Optimizing Dib_Renderer SH')
    writer = imageio.get_writer(os.path.join(output_directory_dib, 'deform_SH.gif'), mode='I')
    for i in loop:
        optimizer.zero_grad()
        new_vertices = model() 
        image_pred, alpha, _ = renderer.forward(points=[new_vertices, faces[0].long()],\
                colors=[uvs, face_textures.long(), textures], light =lightparam)
        image_pred = torch.cat((image_pred, alpha), dim = 3)

        image_pred = image_pred.permute(0,3,1,2)

        loss = torch.sum((image_pred - image_gt[None, :, :])**2) 
     
        loss.backward()
        optimizer.step()
       
        loop.set_description('Loss: %.4f'%(loss.item()))

        if i % 20 == 0:
            image = image_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
            other_image = image_gt.detach().cpu().numpy().transpose((1, 2, 0))
           
            pass_image = image + other_image
            writer.append_data((127*pass_image).astype(np.uint8))




if __name__ == '__main__':
    main()