"""
Dataset classes
"""

import numpy as np
import os
import torch
import torch.utils.data as data


class Ellipsoids(data.Dataset):
	"""Dataset class for synthetic ellipsoids.

	Args:
		- datapath_input (str): path to the dir containing the input voxel grids
		- datapath_target (str): path to the dir containing the target voxel grids
		- mode (str, choice: ['train', 'val']): train vs val mode
		- num_train (int, default=5000): number of training samples
		- num_val (int, default=2000): number of val samples
	"""

	def __init__(self, datapath_input, datapath_target, mode='train', \
		num_train=5000, num_val=2000):
		assert mode in ['train', 'val'], 'Invalide mode specified. Must be train or val.'
		self.datapath_input = datapath_input
		self.datapath_target = datapath_target
		self.mode = mode
		self.num_train = num_train
		self.num_val = num_val


	def __len__(self):
		"""Returns the length of the dataset. """
		if self.mode == 'train':
			return self.num_train
		elif self.mode == 'val':
			return self.num_val


	def __getitem__(self, idx):
		"""Returns the element at index 'idx'. """
		if self.mode == 'val':
			idx += self.num_train
		filepath_input = os.path.join(self.datapath_input, str(idx).zfill(5) + '.npy')
		filepath_target = os.path.join(self.datapath_target, str(idx).zfill(5) + '.npy')
		inp = torch.from_numpy(np.load(filepath_input)).unsqueeze(0)
		tgt = torch.from_numpy(np.load(filepath_target)).unsqueeze(0)
		return inp, tgt


class EllipsoidsForNLL(data.Dataset):
	"""Dataset class for synthetic ellipsoids.

	Args:
		- datapath_input (str): path to the dir containing the input voxel grids
		- datapath_target (str): path to the dir containing the target voxel grids
		- mode (str, choice: ['train', 'val']): train vs val mode
		- num_train (int, default=5000): number of training samples
		- num_val (int, default=2000): number of val samples
	"""

	def __init__(self, datapath_input, datapath_target, mode='train', \
		num_train=5000, num_val=2000):
		assert mode in ['train', 'val'], 'Invalide mode specified. Must be train or val.'
		self.datapath_input = datapath_input
		self.datapath_target = datapath_target
		self.mode = mode
		self.num_train = num_train
		self.num_val = num_val


	def __len__(self):
		"""Returns the length of the dataset. """
		if self.mode == 'train':
			return self.num_train
		elif self.mode == 'val':
			return self.num_val


	def __getitem__(self, idx):
		"""Returns the element at index 'idx'. """
		if self.mode == 'val':
			idx += self.num_train
		filepath_input = os.path.join(self.datapath_input, str(idx).zfill(5) + '.npy')
		filepath_target = os.path.join(self.datapath_target, str(idx).zfill(5) + '.npy')
		inp_ = torch.from_numpy(np.load(filepath_input)).unsqueeze(0)
		inp = torch.zeros(2, inp_.shape[1], inp_.shape[2], inp_.shape[3])
		cond = (inp_[0] == 1).float()
		inp[1] = cond * inp_ + (1 - cond) * inp_

		tgt = torch.from_numpy(np.load(filepath_target)).long()
		# This is not needed for the target, when using NLLLoss.
		tgt_ = torch.from_numpy(np.load(filepath_target)).unsqueeze(0)
		# tgt = torch.zeros(2, tgt_.shape[1], tgt_.shape[2], tgt_.shape[3])
		# cond = (tgt_[0] == 1).float()
		# tgt[1] = cond * tgt_ + (1 - cond) * tgt_
		
		return inp, tgt
