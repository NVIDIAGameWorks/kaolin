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
Dataset classes
"""

import numpy as np
import os
import torch
from tqdm import tqdm 

import torch.utils.data as data
import scipy.io as sio

import kaolin as kal



class ModelNet_ODMS(object):
	r"""
	Dataloader for downloading and reading from ModelNet 

	Note: 
		Made to be passed to the torch dataloader 

	Args: 
		root (str): location the dataset should be downloaded to /loaded from 
		train (bool): if True loads training set, else loads test 
		download (bool): downloads the dataset if not found in root 
		object_class (str): object class to be loaded, if 'all' then all are loaded 
		single_view (bool): if true only on roation is used, if not all 12 views are loaded 

	Attributes: 
		list: list of all voxel locations to be loaded from 

	Examples:
		>>> data_set = ModelNet(root ='./datasets/')
		>>> train_loader = DataLoader(data_set, batch_size=10, n=True, num_workers=8) 

	"""

	def __init__(self, root='../datasets/', train=True, download=True, compute=True, categories=['chair'], single_view=True, voxels = True):
		voxel_set = kal.dataloader.ModelNet( root, train=train, download=download, categories=categories, single_view=single_view)
		odm_location = root + '/ModelNet/ODMs/'
		if voxels: 
			self.load_voxels = True 
			self.voxel_names = []

		self.names = []
		if not os.path.exists(odm_location) and compute:
				print ('ModelNet ODMS were not found at {0}, and compute is set to False'.format(odm_location))
			
		for n in tqdm(voxel_set.names): 
		
			example_location = odm_location + n.split('volumetric_data')[-1]
			example_length = len(example_location.split('/')[-1])
			example_folder = example_location[:-example_length]
			if not os.path.exists(example_folder):
				if compute:  
					os.makedirs(example_folder)
				else: 
					print ('ModelNet ODMS were not found at {0}, and compute is set to False'.format(example_location))
			if not os.path.exists(example_location):
				voxel = sio.loadmat(n)['instance']
				odms = kal.rep.voxel.extract_odms(voxel)
				odms = np.array(odms)
				sio.savemat(example_location, {'odm': odms})

							
			self.names.append(example_location)
			if self.load_voxels: 
				self.voxel_names.append(n)


						
	def __len__(self):
		""" 
		Returns:
			number of odms lists in active dataloader

		"""
		return len(self.names)

	def __getitem__(self, item):
		"""Gets a single example of a ModelNet voxel model 
		Args: 
			item (int): index of required model 

		return: 
			dictionary which contains a odm data

		"""
		data = {}
		odm_path = self.names[item]
		odms = sio.loadmat(odm_path)['odm']
		data['odms'] = torch.FloatTensor(odms.astype(float))
		if self.load_voxels: 
			voxels = sio.loadmat(self.voxel_names[item])['instance']
			data['voxels'] = torch.FloatTensor(voxels.astype(float))
		return data

