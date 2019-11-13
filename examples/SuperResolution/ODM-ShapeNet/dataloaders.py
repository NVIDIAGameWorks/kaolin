"""
Dataset classes
"""

import numpy as np
import os
import torch
from tqdm import tqdm 

import torch.utils.data as data
import scipy.io as sio
import scipy.sparse

import kaolin as kal



class ShapeNet_ODMS(object):
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

	def __init__(self, root='../datasets/', train=True, download=True, compute=True, high=128, low=32, categories=['chair'], single_view=True, voxels = True, split = .7):
		self.high = high
		self.low = low
		voxel_set = kal.dataloader.ShapeNet.Voxels( root, train=train, download=download, categories=categories, resolutions=[high,low], split = split)
		odm_location = root + '/ShapeNet/ODMs/'
		self.load_voxels = voxels
		if self.load_voxels :  
			self.voxel_names = {}

		self.names = {}
		if not os.path.exists(odm_location) and compute:
				print ('ShapeNet ODMS were not found at {0}, and compute is set to False'.format(odm_location))
	
		for res in [high, low]:
			self.names[res] = []
			if voxels: 
				self.voxel_names[res] = []
			print ("Computing ODMs from Shapenet Dataset classes {0} in resolution {1}".format(categories, res))
			
			for n in tqdm(voxel_set.names[res]): 
				example_location = odm_location + n.split('voxel')[-1][:-4] + '.mat'
				example_length = len(example_location.split('/')[-1])
				example_folder = example_location[:-example_length]
				if not os.path.exists(example_folder):
					if compute:  
						os.makedirs(example_folder)
					else: 
						print ('ModelNet ODMS were not found at {0}, and compute is set to False'.format(example_location))
				if not os.path.exists(example_location):
					voxel = scipy.sparse.load_npz(n)
					voxel = np.array((voxel.todense()))
					voxel_res = voxel.shape[0]
					voxel = voxel.reshape((voxel_res, voxel_res, voxel_res))
					odms = kal.rep.voxel.extract_odms(voxel)
					odms = np.array(odms)
					sio.savemat(example_location, {'odm': odms})

				self.names[res].append(example_location)
				if self.load_voxels: 
					self.voxel_names[res].append(n)


						
	def __len__(self):
		""" 
		Returns:
			number of odms lists in active dataloader

		"""
		return len(self.names[self.high])

	def __getitem__(self, item):
		"""Gets a single example of a ModelNet voxel model 
		Args: 
			item (int): index of required model 

		return: 
			dictionary which contains a odm data

		"""
		data = {}
		for res in [self.high, self.low]:
			odm_path = self.names[res][item]
			odms = sio.loadmat(odm_path)['odm']
			data['odms_{0}'.format(res)] = torch.FloatTensor(odms.astype(float))
			if self.load_voxels: 
				voxel = scipy.sparse.load_npz(self.voxel_names[res][item])
				voxel = np.array((voxel.todense()))
				voxel_res = voxel.shape[0]
				voxel = voxel.reshape((voxel_res, voxel_res, voxel_res))
				data['voxels_{0}'.format(res)] = torch.FloatTensor(voxel.astype(float))
		return data


