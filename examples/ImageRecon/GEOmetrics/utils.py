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

import torch 
from torchvision import transforms
from torchvision.transforms import Normalize as norm 
import trimesh
from sklearn.preprocessing import normalize
import kaolin as kal
from PIL import Image
from collections import defaultdict
import numpy as np
from kaolin.rep import TriangleMesh
import kaolin as kal




preprocess = transforms.Compose([
   transforms.Resize(224),
   transforms.ToTensor()
])


def collate_fn(data): 
	new_data = {}
	for k in data[0].keys():
		
		if k in ['points','norms', 'imgs', 'cam_mat', 'cam_pos']:
			new_info = tuple(d[k] for d in data)
			new_info = torch.stack(new_info, 0)
		elif k in ['adj']: 
			
			adj_values = tuple(d[k].coalesce().values() for d in data)
			adj_indices = tuple(d[k].coalesce().indices() for d in data)
			new_data['adj_values'] = adj_values
			new_data['adj_indices'] = adj_indices
			
		else: 
			new_info = tuple(d[k] for d in data)

		new_data[k] = new_info
	return new_data





def get_pooling_index( positions, cam_mat, cam_pos, dims):
	device = positions.device
	#project points into 2D
	positions = positions * .57  # accounting for recaling in 3Dr2n
	positions = positions - cam_pos 
	positions = torch.mm(positions,cam_mat.permute(1,0))
	positions_xs =  positions[:, 1] / positions[:, 2]
	positions_ys = -positions[:, 0] / positions[:, 2] 

	# do bilinear interpolation over pixel coordiantes
	data_meta = defaultdict(list)

	for dim in dims:
		focal_length = 250./224. * dim
		xs = positions_xs * focal_length + dim/2.
		ys = positions_ys * focal_length + dim/2.

		cur_xs = torch.clamp(xs , 0, dim - 1)
		cur_ys = torch.clamp(ys , 0, dim - 1)


		# img = np.zeros((dim,dim))
		# for x,y in zip (cur_xs, cur_ys): 
		# 	img[x.int(), y.int()] = 255
		# from PIL import Image
		# Image.fromarray(img).show()
		
		# exit()
		
		
		x1s, y1s, x2s, y2s = torch.floor(cur_xs), torch.floor(cur_ys), torch.ceil(cur_xs), torch.ceil(cur_ys)

		A = x2s - cur_xs
		B = cur_xs - x1s
		G = y2s - cur_ys
		H = cur_ys - y1s

		y1s = y1s + torch.arange(positions.shape[0]).float().to(device)*dim 
		y2s = y2s + torch.arange(positions.shape[0]).float().to(device)*dim 

		data_meta['A'].append(A.float().unsqueeze(0))
		data_meta['B'].append(B.float().unsqueeze(0))
		data_meta['G'].append(G.float().unsqueeze(0))
		data_meta['H'].append(H.float().unsqueeze(0))
		data_meta['x1s'].append(x1s.long().unsqueeze(0))
		data_meta['x2s'].append(x2s.long().unsqueeze(0))
		data_meta['y1s'].append(y1s.long().unsqueeze(0))
		data_meta['y2s'].append(y2s.long().unsqueeze(0))

	for key in data_meta:
		data_meta[key] = torch.cat(data_meta[key], dim=0)
	return data_meta





def pooling(blocks, pooling_indices, index):

	full_features = None 
	for i_block, block in enumerate(blocks):
		block = block[index]
		A = pooling_indices['A'][i_block]
		B = pooling_indices['B'][i_block]
		G = pooling_indices['G'][i_block]
		H = pooling_indices['H'][i_block]

		x1s = pooling_indices['x1s'][i_block]
		x2s = pooling_indices['x2s'][i_block]
		y1s = pooling_indices['y1s'][i_block]
		y2s = pooling_indices['y2s'][i_block]


		C =torch.index_select(block, 1, x1s).view(block.shape[0], -1 )
		C = torch.index_select(C, 1, y1s)
		D =torch.index_select(block, 1, x1s).view(block.shape[0], -1 )
		D = torch.index_select(D, 1, y2s)
		E =torch.index_select(block, 1, x2s).view(block.shape[0], -1 )
		E = torch.index_select(E, 1, y1s)
		F =torch.index_select(block, 1, x2s).view(block.shape[0], -1 )
		F = torch.index_select(F, 1, y2s)


		features = (A*C*G + H*D*A + G*E*B + B*F*H).permute(1,0)

		if full_features is None: full_features = features
		else: full_features = torch.cat((full_features, features), dim = 1)
 
	return full_features

norm_distance = kal.metrics.point.SidedDistance()

def chamfer_normal(pred_mesh, gt_points,gt_norms): 

	# find closest gt points
	gt_indices = norm_distance(pred_mesh.vertices.unsqueeze(0), gt_points.unsqueeze(0))[0]
	# select norms from closest points and exand to match edges lengths

	gt_norm_selections = gt_norms[gt_indices]
	new_dimensions = (gt_norm_selections.shape[0],pred_mesh.ve.shape[1], 3 )
	vertex_norms = gt_norm_selections.view(-1,1,3).expand(new_dimensions)


	
	# get all nieghbor positions
	neighbor_indecies = pred_mesh.vv.clone()
	empty_indecies = (neighbor_indecies <0)
	neighbor_indecies[empty_indecies] = 0 
	empty_indecies = ((empty_indecies -1) *-1).float().unsqueeze(-1)
	neighbor_indecies = neighbor_indecies.view(-1)
	vertex_neighbors  = pred_mesh.vertices[neighbor_indecies].view(new_dimensions)

	# mask both tensors
	vertex_norms = vertex_norms * empty_indecies
	vertex_norms = vertex_norms.contiguous().view(-1,3)
	vertex_neighbors = vertex_neighbors * empty_indecies 
	vertex_neighbors = vertex_neighbors.contiguous().view(-1,3)

	# calculate normal loss, devide by number of unmasked elements to get mean 
	normal_loss = (torch.abs(torch.sum(vertex_norms * vertex_neighbors, dim = 1))) 
	normal_loss = normal_loss.sum() / float(empty_indecies.sum())
	return normal_loss



def setup_meshes(filename='meshes/156.obj', device='cuda'): 
	mesh_1 = kal.rep.TriangleMesh.from_obj(filename, enable_adjacency=True)
	if device == "cuda":
		mesh_1.cuda()
	adj_1 = mesh_1.compute_adjacency_matrix_full().clone()
	adj_1 = normalize_adj(adj_1)
	mesh_1_i = kal.rep.TriangleMesh.from_tensors(mesh_1.vertices.clone(), mesh_1.faces.clone())
	face_list_1 = calc_face_list(mesh_1)

	initial_meshes = [mesh_1_i]
	updated_meshes = [mesh_1]
	adjs = [adj_1]
	
	face_lists = [face_list_1]

	mesh_info = {'init':initial_meshes, 'update':updated_meshes , 'adjs': adjs,\
		'face_lists': face_lists}
	
	return mesh_info



def calc_face_list(mesh): 
	
	face_list = np.zeros((len(mesh.faces), 3, 3))
	for e,f1 in enumerate(mesh.faces):
		for ee, index in enumerate(mesh.ff[e]):
			f2 = mesh.faces[index]
			f1_position = -1
			f2_position = -1
			if f1[0] in f2 and f1[1] in f2: 
				f1_position = 0
			elif f1[1] in f2 and f1[2] in f2: 
				f1_position = 1
			elif f1[0] in f2 and f1[2] in f2: 
				f1_position = 2
			if f1_position >= 0 : 
				if f2[0] in f1 and f2[1] in f1: 
					face_list[e][f1_position] = [index,0, e]
				elif f2[1] in f1 and f2[2] in f1: 
					face_list[e][f1_position] = [index,1, e]
				elif f2[0] in f1 and f2[2] in f1: 
					face_list[e][f1_position] = [index,2, e]
	
	return torch.LongTensor(face_list).to(mesh.faces.device)


def compute_splitting_faces( meshes,index, angle = 50, show = False): 
	eps = .00001

	# extract vertex coordinated for each vertex in face 
	faces = meshes['face_archive'][index]
	verts = meshes['update'][index].vertices
	face_list = meshes['face_lists'][index]
	p1 = torch.index_select(verts, 0,faces[:,1])
	p2 = torch.index_select(verts, 0,faces[:,0])
	p3 = torch.index_select(verts, 0,faces[:,2])
 
	# cauculate normals of each face 
	e1 = p2-p1
	e2 = p3-p1
	face_normals = torch.cross(e1, e2)
	qn = torch.norm(face_normals, p=2, dim=1).detach().view(-1,1)
	face_normals = face_normals.div(qn.expand_as(face_normals))
	main_face_normals = torch.index_select(face_normals, 0, face_list[:,0,2])

	# cauculate the curvature with the 3 nighbor faces 
	#1
	face_1_normals = torch.index_select(face_normals, 0, face_list[:,0,0])
	curvature_proxi_rad = torch.sum(main_face_normals*face_1_normals, dim = 1).clamp(-1.0 + eps, 1.0 - eps).acos()
	curvature_proxi_1 = (curvature_proxi_rad).view(-1,1)
	#2
	face_2_normals = torch.index_select(face_normals, 0, face_list[:,1,0])
	curvature_proxi_rad = torch.sum(main_face_normals*face_2_normals, dim = 1).clamp(-1.0 + eps, 1.0 - eps).acos()
	curvature_proxi_2 = (curvature_proxi_rad).view(-1,1)
	#3
	face_3_normals = torch.index_select(face_normals, 0, face_list[:,2,0])
	curvature_proxi_rad = torch.sum(main_face_normals*face_3_normals, dim = 1).clamp(-1.0 + eps, 1.0 - eps).acos()
	curvature_proxi_3 = (curvature_proxi_rad).view(-1,1)
	
	# get average over neighbors 
	curvature_proxi_full = torch.cat( (curvature_proxi_1, curvature_proxi_2, curvature_proxi_3), dim = 1)
	curvature_proxi = torch.mean(curvature_proxi_full, dim = 1)

	#select faces with high curvature and return their index
	splitting_faces  = np.where(curvature_proxi.cpu()*180/np.pi  > angle )[0]

	if splitting_faces.shape[0] <3:
		splitting_faces  = curvature_proxi.topk(3, sorted = False)[1] 
	else:
		splitting_faces  = torch.LongTensor(splitting_faces).to(faces.device)
	return splitting_faces






def split_meshes(meshes, features, index, angle = 70 ): 
	# compute faces to split
	faces_to_split = compute_splitting_faces(meshes, index, angle, show = ( index == 1))
	# split mesh with selected faces
	new_verts, new_faces, new_face_archive, new_face_list, new_features = split_info( meshes, faces_to_split, features, index)
	new_mesh = TriangleMesh.from_tensors(new_verts, new_faces)
	new_mesh_i = TriangleMesh.from_tensors(new_verts, new_faces)
	# compute new adj matrix
	new_adj = new_mesh.compute_adjacency_matrix_full().clone()
	new_adj = normalize_adj(new_adj)
	#update the meshes dictionary
	meshes['init'].append(new_mesh)
	meshes['update'].append(new_mesh_i)
	meshes['adjs'].append(new_adj)
	meshes['face_lists'].append(new_face_list)
	meshes['face_archive'].append(new_face_archive)

	return new_features

def normalize_adj(mx):
	rowsum = mx.sum(dim =1).view(-1)
	r_inv = 1./rowsum
	r_inv[r_inv!= r_inv] = 0.
	r_mat_inv = torch.eye(r_inv.shape[0]).to(mx.device)*r_inv
	mx = torch.mm(r_mat_inv,mx)
	return mx
	
	

def reset_meshes(meshes): 
	meshes['face_archive'] = [meshes['init'][0].faces.clone()]
	meshes['init'] = meshes['init'][:1]
	meshes['update'] = meshes['update'][:1]
	meshes['adjs'] = meshes['adjs'][:1]
	meshes['face_lists'] = meshes['face_lists'][:1]







def split_features(split_mx, features): 
	features = features.permute(1,0)
	new_features = torch.mm(features, split_mx)
	features = torch.cat((features, new_features), dim= 1 ).permute(1,0)
	return features

def loss_surf(meshes, tgt_points):	
	loss = kal.metrics.point.chamfer_distance(tgt_points, meshes['update'][0].sample(3000)[0])
	loss +=  kal.metrics.point.chamfer_distance(tgt_points, meshes['update'][1].sample(3000)[0])
	loss += kal.metrics.point.chamfer_distance(tgt_points, meshes['update'][2].sample(3000)[0])
	return loss

def loss_surf2(meshes, tgt_points):	
	loss = nvl.metrics.mesh.point_to_surface(tgt_points, meshes['update'][0])
	loss += nvl.metrics.point.directed_distance((meshes['update'][0].sample(3000)[0]), tgt_points)
	loss += nvl.metrics.mesh.point_to_surface(tgt_points, meshes['update'][1])
	loss += nvl.metrics.point.directed_distance((meshes['update'][1].sample(3000)[0]), tgt_points)
	loss += nvl.metrics.mesh.point_to_surface(tgt_points, meshes['update'][2])
	loss += nvl.metrics.point.directed_distance((meshes['update'][2].sample(3000)[0]), tgt_points)
	return loss


def loss_edge(meshes):	
	loss =  kal.metrics.mesh.edge_length(meshes['update'][0])
	loss += kal.metrics.mesh.edge_length(meshes['update'][1])
	loss += kal.metrics.mesh.edge_length(meshes['update'][2])
	return loss

def loss_lap(meshes): 
	loss =  .3* kal.metrics.mesh.laplacian_loss(meshes['init'][0],meshes['update'][0])
	loss += kal.metrics.mesh.laplacian_loss(meshes['init'][1],meshes['update'][1])
	loss += kal.metrics.mesh.laplacian_loss(meshes['init'][2],meshes['update'][2])

	loss += torch.sum((meshes['init'][1].vertices-meshes['update'][1].vertices)**2, 1).mean() * .0666
	loss += torch.sum((meshes['init'][2].vertices-meshes['update'][2].vertices)**2, 1).mean() * .0666
	
	return loss 





def split_info(meshes, split_faces, features, index ):
	
	device = meshes['init'][0].vertices.device
	faces_verts = meshes['face_archive'][index].clone() # vertex info of all faces made
	face_list = meshes['face_lists'][index].clone()


	splitting_face_list_values = torch.index_select(face_list, 0,  split_faces ) # neighbor info of faces to be split 
	splitting_face_list_len = splitting_face_list_values.shape[0]

	counter = torch.zeros((face_list.shape[0])).to(device)
	counter[ split_faces ] = 1
	unsplitting_faces_list_indecies  = np.where(counter.cpu() == 0 )[0]
	unsplitting_face_list_values = torch.index_select(face_list, 0, torch.LongTensor(unsplitting_faces_list_indecies).to(device))  # neighbor info of faces not split 
	

	splitting_faces_indecies = splitting_face_list_values[:,0,2] # indecies of faces being split in faces_verts
	unsplitting_faces_indecies = unsplitting_face_list_values[:,0,2] # indecies of faces not being split from face_verts 

	 
	# indecies of new faces being made in, in the unpdated faces_verts array 
	new_faces_indecies_1 = torch.arange(splitting_face_list_len).to(device).view(-1,1) + faces_verts.shape[0]
	new_faces_indecies_2 = new_faces_indecies_1 + splitting_face_list_len
	new_faces_indecies_3 = new_faces_indecies_2 + splitting_face_list_len
	splitting_new_faces_indecies = torch.cat((new_faces_indecies_1, new_faces_indecies_2, new_faces_indecies_3), dim = 1 )
	unsplitting_new_faces_indecies = torch.cat( (unsplitting_faces_indecies.view(-1,1), unsplitting_faces_indecies.view(-1,1), unsplitting_faces_indecies.view(-1,1)), dim = 1)

	# saving where each face will be held in the updated face_verts array, saved in this manner for quick selection
	new_positions = torch.zeros((faces_verts.shape[0], 3)).to(device).long()
	new_positions[splitting_faces_indecies] = splitting_new_faces_indecies
	new_positions[unsplitting_faces_indecies] = unsplitting_new_faces_indecies


	# adding unsplitting triangles to new face_list 
	#get location of 3 neighbors 
	unsplitting_connecting_face_1  = new_positions[unsplitting_face_list_values[:,0,0],unsplitting_face_list_values[:,0,1] ].view(-1,1,1)
	unsplitting_connecting_face_2  = new_positions[unsplitting_face_list_values[:,1,0],unsplitting_face_list_values[:,1,1] ].view(-1,1,1)
	unsplitting_connecting_face_3  = new_positions[unsplitting_face_list_values[:,2,0],unsplitting_face_list_values[:,2,1] ].view(-1,1,1)	
	# get the niegbors index in updated face_verts array 
	unsplitting_connecting_side_1 = unsplitting_face_list_values[:,0,1].view(-1,1,1)
	unsplitting_connecting_side_2 = unsplitting_face_list_values[:,1,1].view(-1,1,1)
	unsplitting_connecting_side_3 = unsplitting_face_list_values[:,2,1].view(-1,1,1)
	# make new face_list 
	unsplitting_face_number = unsplitting_faces_indecies.view(-1,1,1)
	new_unsplitting_face_list_1 = torch.cat((unsplitting_connecting_face_1,unsplitting_connecting_side_1, unsplitting_face_number ), dim = 2 )
	new_unsplitting_face_list_2 = torch.cat((unsplitting_connecting_face_2,unsplitting_connecting_side_2, unsplitting_face_number ), dim = 2 )
	new_unsplitting_face_list_3 = torch.cat((unsplitting_connecting_face_3,unsplitting_connecting_side_3, unsplitting_face_number ), dim = 2 )
	new_unsplitting_face_list = torch.cat((new_unsplitting_face_list_1, new_unsplitting_face_list_2, new_unsplitting_face_list_3), dim = 1)

	# adding splitting triangles to new face_list 
	# new triangle 1
	#get location of 3 neighbors
	splitting_connecting_face_1_1  = new_positions[splitting_face_list_values[:,0,0],splitting_face_list_values[:,0,1] ].view(-1,1,1) # one old face is its neigboors
	splitting_connecting_face_1_2  = new_faces_indecies_2.view(-1,1,1) # 2 new faces are its neighbor 
	splitting_connecting_face_1_3  = new_faces_indecies_3.view(-1,1,1)
	# get the nigbors index in updated face_verts array 
	splitting_connecting_side_1_1 = splitting_face_list_values[:,0,1].view(-1,1,1) # get old face's index in face_verts 
	splitting_connecting_side_1_2 = torch.zeros(splitting_face_list_len).view(-1,1,1).long().to(device)# use new faces' known indices 
	splitting_connecting_side_1_3 = torch.zeros(splitting_face_list_len).view(-1,1,1).long().to(device)
	# make new face_list 
	splitting_face_number_1 = new_faces_indecies_1.view(-1,1,1)
	new_splitting_face_list_1_1 = torch.cat((splitting_connecting_face_1_1,splitting_connecting_side_1_1, splitting_face_number_1 ), dim = 2 ) 
	new_splitting_face_list_1_2 = torch.cat((splitting_connecting_face_1_2,splitting_connecting_side_1_2, splitting_face_number_1 ), dim = 2 )
	new_splitting_face_list_1_3 = torch.cat((splitting_connecting_face_1_3,splitting_connecting_side_1_3, splitting_face_number_1 ), dim = 2 )
	new_splitting_face_list_1 = torch.cat((new_splitting_face_list_1_1, new_splitting_face_list_1_2, new_splitting_face_list_1_3), dim = 1)

	# new triangle 2
	splitting_connecting_face_2_1  = new_faces_indecies_1.view(-1,1,1)
	splitting_connecting_face_2_2  = new_positions[splitting_face_list_values[:,1,0],splitting_face_list_values[:,1,1] ].view(-1,1,1)
	splitting_connecting_face_2_3  = new_faces_indecies_3.view(-1,1,1)

	splitting_connecting_side_2_1 = torch.ones(splitting_face_list_len).view(-1,1,1).long().to(device)
	splitting_connecting_side_2_2 = splitting_face_list_values[:,1,1].view(-1,1,1)
	splitting_connecting_side_2_3 = torch.ones(splitting_face_list_len).view(-1,1,1).long().to(device)

	splitting_face_number_2 = new_faces_indecies_2.view(-1,1,1)
	new_splitting_face_list_2_1 = torch.cat((splitting_connecting_face_2_1,splitting_connecting_side_2_1, splitting_face_number_2 ), dim = 2 )
	new_splitting_face_list_2_2 = torch.cat((splitting_connecting_face_2_2,splitting_connecting_side_2_2, splitting_face_number_2 ), dim = 2 )
	new_splitting_face_list_2_3 = torch.cat((splitting_connecting_face_2_3,splitting_connecting_side_2_3, splitting_face_number_2 ), dim = 2 )
	new_splitting_face_list_2 = torch.cat((new_splitting_face_list_2_1, new_splitting_face_list_2_2, new_splitting_face_list_2_3), dim = 1)


	# new triangle 3
	splitting_connecting_face_3_1  = new_faces_indecies_1.view(-1,1,1)
	splitting_connecting_face_3_2  = new_faces_indecies_2.view(-1,1,1)
	splitting_connecting_face_3_3  = new_positions[splitting_face_list_values[:,2,0],splitting_face_list_values[:,2,1] ].view(-1,1,1)

	splitting_connecting_side_3_1 = torch.ones(splitting_face_list_len).view(-1,1,1).long().to(device)*2
	splitting_connecting_side_3_2 = torch.ones(splitting_face_list_len).view(-1,1,1).long().to(device)*2
	splitting_connecting_side_3_3 = splitting_face_list_values[:,2,1].view(-1,1,1)

	splitting_face_number_3 = new_faces_indecies_3.view(-1,1,1)
	new_splitting_face_list_3_1 = torch.cat((splitting_connecting_face_3_1,splitting_connecting_side_3_1, splitting_face_number_3 ), dim = 2 )
	new_splitting_face_list_3_2 = torch.cat((splitting_connecting_face_3_2,splitting_connecting_side_3_2, splitting_face_number_3 ), dim = 2 )
	new_splitting_face_list_3_3 = torch.cat((splitting_connecting_face_3_3,splitting_connecting_side_3_3, splitting_face_number_3 ), dim = 2 )
	new_splitting_face_list_3 = torch.cat((new_splitting_face_list_3_1, new_splitting_face_list_3_2, new_splitting_face_list_3_3), dim = 1)

	# conplete new face_list is made 

	new_splitting_face_list = torch.cat((new_unsplitting_face_list, new_splitting_face_list_1, new_splitting_face_list_2, new_splitting_face_list_3))
	split_faces = faces_verts[splitting_faces_indecies]

	# now to make the new vertex 
	vertex_count = meshes['update'][index].vertices.shape[0]
	new_len = split_faces.shape[0] + vertex_count

	# select the vertices of the faces to be split 
	x_f = split_faces[:,0]
	y_f = split_faces[:,1]
	z_f = split_faces[:,2]
	verts_and_features = torch.cat((meshes['update'][index].vertices, features), dim = 1) 
	x_v = verts_and_features[x_f] 
	y_v = verts_and_features[y_f]
	z_v = verts_and_features[z_f]
	#average the featurs and position 
	v1 = x_v/3  + y_v/3 + z_v/3 
	verts_and_features = torch.cat((verts_and_features, v1))
	verts = verts_and_features[:,:3]

	features = verts_and_features[:,3:]
	v1_inds = (vertex_count + torch.arange(split_faces.shape[0])).to(device).view(-1,1)
	x_f = x_f.view(-1,1)
	y_f = y_f.view(-1,1)
	z_f = z_f.view(-1,1)
	# define verts of new faces 
	new_face_1 = torch.cat((x_f ,y_f ,v1_inds) , dim =1)
	new_face_2 = torch.cat((v1_inds , y_f, z_f ) , dim =1)
	new_face_3 = torch.cat((x_f ,v1_inds,z_f) , dim =1)
	# name new face_verts array by appending then to old in order previously defined 
	face_archive = torch.cat( (faces_verts, new_face_1, new_face_2, new_face_3))

	faces = face_archive[new_splitting_face_list[:,0,2]]

	return verts, faces, face_archive, new_splitting_face_list, features


