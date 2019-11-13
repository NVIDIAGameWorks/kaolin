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

	def __init__(self, textures):
		super(Model, self).__init__()
		self.textures = nn.Parameter(textures)
	def forward(self):
		return torch.sigmoid(self.textures) 




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
	filename_ref = os.path.join(data_dir, 'example3_ref.png')
	
	mesh = TriangleMesh.from_obj(filename_input)
	vertices = mesh.vertices
	faces = mesh.faces.int()
	uvs = mesh.uvs
	face_textures = mesh.face_textures
		
	
	pmax = vertices.max()
	pmin = vertices.min()
	pmiddle = (pmax + pmin) / 2
	vertices = vertices - pmiddle    
	coef = 8
	vertices = vertices * coef

	
	vertices = vertices[None, :, :].cuda()  
	faces = faces[None, :, :].cuda() 
	uvs = uvs[None, :, :].cuda()
	face_textures[None, :, :].cuda()

	image_gt = torch.from_numpy(imread(filename_ref).astype('float32') / 255.).permute(2,0,1)[None, ::].cuda()


	##########################
	#NMR 
	##########################
	textures = torch.rand(1, faces.shape[1], 2, 2, 2, 3, dtype=torch.float32)
	model = Model( textures ).cuda()
	renderer = nr.Renderer(camera_mode='look_at')
	renderer.eye = nr.get_points_from_angles(camera_distance, elevation, azimuth)
	renderer.perspective = False
	renderer.light_intensity_directional = 0.0
	renderer.light_intensity_ambient = 1.0
	optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))


	loop = tqdm.tqdm(range(2000))
	loop.set_description('Optimizing NMR')
	writer = imageio.get_writer(os.path.join(output_directory_nmr, 'texture.gif'), mode='I')
	for i in loop:
		optimizer.zero_grad()
		new_texture = model() 
		image_pred  ,_, _= renderer(vertices, faces, new_texture)
		loss = torch.sum((image_pred - image_gt)**2)

		loss.backward()
		optimizer.step()
		loop.set_description('Loss: %.4f'%(loss.item()))
		if i % 20 == 0:
			image = image_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))
			writer.append_data((255*image).astype(np.uint8))


   

	##########################
	#Soft Rasterizer 
	##########################
	textures = torch.rand(1, faces.shape[1], 2, 3, dtype=torch.float32)
	model = Model( textures ).cuda()
	renderer = sr.SoftRenderer(image_size=256, sigma_val=3e-5, 
							   camera_mode='look_at', perspective = False,
							   light_intensity_directionals = 0.0,light_intensity_ambient = 1.0  )
	renderer.transform.set_eyes_from_angles(camera_distance, elevation, azimuth)
	optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))
	loop = tqdm.tqdm(range(2000))
	loop.set_description('Optimizing SR')
	writer = imageio.get_writer(os.path.join(output_directory_sr, 'texture.gif'), mode='I')
	for i in loop:
		optimizer.zero_grad()
		new_texture = model() 
		mesh = sr.Mesh(vertices, faces, new_texture)
		image_pred = renderer.render_mesh(mesh)

		loss = torch.sum(((image_pred[:,:3] - image_gt[None, :, :]))**2)
		loss.backward()
		optimizer.step()
		loop.set_description('Loss: %.4f'%(loss.item()))
		if i % 20 == 0:
			image = image_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))

			writer.append_data((255*image).astype(np.uint8))



	###########################
	# Dib-Renderer - Vertex Colours
	###########################
	textures = torch.rand(1, vertices.shape[1], 3).cuda()
	model = Model(textures).cuda()
	renderer = Dib_Renderer(256, 256, mode = 'VertexColor')
	renderer.set_look_at_parameters([90-azimuth], [elevation], [camera_distance])
	optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))
	loop = tqdm.tqdm(range(2000))
	loop.set_description('Optimizing VertexColor')
	writer = imageio.get_writer(os.path.join(output_directory_dib, 'texture_VertexColor.gif'), mode='I')
	for i in loop:
		optimizer.zero_grad()
		new_texture = model()         

		image_pred, alpha, _ = renderer.forward(points=[vertices, faces[0].long()], colors=[new_texture])
		image_pred = torch.cat((image_pred, alpha), dim = 3)
		image_pred = image_pred.permute(0,3,1,2)
		loss = torch.sum(((image_pred[:,:3] - image_gt[None, :, :]))**2)
		loss.backward()
		optimizer.step()
		loop.set_description('Loss: %.4f'%(loss.item()))
		if i % 20 == 0:
			image = image_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))

			writer.append_data((255*image).astype(np.uint8))

	###########################
	# Dib-Renderer - Lambertian
	###########################
	textures = torch.rand(1, 3, 256, 256).cuda()
	model = Model(textures).cuda()
	renderer = Dib_Renderer(256, 256, mode = 'Lambertian')
	renderer.set_look_at_parameters([90-azimuth], [elevation], [camera_distance])
	optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))
	loop = tqdm.tqdm(range(2000))
	loop.set_description('Optimizing Lambertian')
	writer = imageio.get_writer(os.path.join(output_directory_dib, 'texture_Lambertian.gif'), mode='I')
	for i in loop:
		optimizer.zero_grad()
		new_texture = model()         

		image_pred, alpha, _ = renderer.forward(points=[vertices, faces[0].long()], \
			colors=[uvs, face_textures.long(), new_texture])
		image_pred = torch.cat((image_pred, alpha), dim = 3)
		image_pred = image_pred.permute(0,3,1,2)
		loss = torch.sum(((image_pred[:,:3] - image_gt[None, :, :]))**2)
		loss.backward()
		optimizer.step()
		loop.set_description('Loss: %.4f'%(loss.item()))
		if i % 20 == 0:
			image = image_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))

			writer.append_data((255*image).astype(np.uint8))

	###########################
	# Dib-Renderer - Phong
	# ###########################
	textures = torch.rand(1, 3, 256, 256).cuda()
	model = Model(textures).cuda()
	renderer = Dib_Renderer(256, 256, mode = 'Phong')
	renderer.set_look_at_parameters([90-azimuth], [elevation], [camera_distance])
	optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))
	loop = tqdm.tqdm(range(2000))
	loop.set_description('Optimizing Phong')


	### Lighting info ###
	material = np.array([[0.3, 0.3, 0.3], 
						 [1.0, 1.0, 1.0],
						 [0.4, 0.4, 0.4]], dtype=np.float32).reshape(-1, 3, 3)
	material = torch.from_numpy(material).repeat(1, 1, 1).cuda()
	
	shininess = np.array([100], dtype=np.float32).reshape(-1, 1)
	shininess = torch.from_numpy(shininess).repeat(1, 1).cuda()

	lightdirect = 2 * np.random.rand(1, 3).astype(np.float32) - 1
	lightdirect[:, 2] += 2
	lightdirect = torch.from_numpy(lightdirect).cuda()



	writer = imageio.get_writer(os.path.join(output_directory_dib, 'texture_Phong.gif'), mode='I')
	for i in loop:
		optimizer.zero_grad()
		new_texture = model()         

		image_pred, alpha, _ = renderer.forward(points=[vertices, faces[0].long()], \
			colors=[uvs, face_textures.long(), new_texture],\
											  light= lightdirect, \
											  material=material, \
											  shininess=shininess)
		image_pred = torch.cat((image_pred, alpha), dim = 3)
		image_pred = image_pred.permute(0,3,1,2)


		loss = torch.sum(((image_pred[:,:3] - image_gt[None, :, :]))**2)
		loss.backward()
		optimizer.step()
		loop.set_description('Loss: %.4f'%(loss.item()))
		if i % 20 == 0:
			image = image_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))

			writer.append_data((255*image).astype(np.uint8))


	###########################
	# Dib-Renderer - SphericalHarmonics
	###########################
	textures = torch.rand(1, 3, 256, 256).cuda()
	model = Model(textures).cuda()
	renderer = Dib_Renderer(256, 256, mode = 'SphericalHarmonics')
	renderer.set_look_at_parameters([90-azimuth], [elevation], [camera_distance])
	optimizer = torch.optim.Adam(model.parameters(), 0.01, betas=(0.5, 0.99))
	

	lightparam = np.random.rand(1, 9).astype(np.float32)
	lightparam[:, 0] += 4
	lightparam = torch.from_numpy(lightparam).cuda()
	

	loop = tqdm.tqdm(range(2000))
	loop.set_description('Optimizing SH')
	writer = imageio.get_writer(os.path.join(output_directory_dib, 'texture_SH.gif'), mode='I')
	for i in loop:
		optimizer.zero_grad()
		new_texture = model()         

		image_pred, alpha, _ = renderer.forward(points=[vertices, faces[0].long()], \
			colors=[uvs, face_textures.long(), new_texture],\
			light=lightparam)
		image_pred = torch.cat((image_pred, alpha), dim = 3)
		image_pred = image_pred.permute(0,3,1,2)
		loss = torch.sum(( (image_pred[:,:3] - image_gt[None, :, :]))**2)
		loss.backward()
		optimizer.step()
		loop.set_description('Loss: %.4f'%(loss.item()))
		if i % 20 == 0:
			image = image_pred.detach().cpu().numpy()[0].transpose((1, 2, 0))

			writer.append_data((255*image).astype(np.uint8))


 

if __name__ == '__main__':
	main()