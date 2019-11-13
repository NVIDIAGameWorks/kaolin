"""
Network architecture definitions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
	"""A simple encoder-decoder style voxel superresolution network"""


	def __init__(self):
		super(EncoderDecoder, self).__init__()

		self.conv1 = nn.Conv3d(1, 16, 3, stride = 2, padding=1)
		self.bn1 = nn.BatchNorm3d(16)
		self.conv2 = nn.Conv3d(16, 32, 3, stride = 2, padding=1)
		self.bn2 = nn.BatchNorm3d(32)
		self.deconv3 = nn.ConvTranspose3d(32, 16, 3, stride = 2,  padding=1)
		self.bn3 = nn.BatchNorm3d(16)
		self.deconv4 = nn.ConvTranspose3d(16, 8, 3, stride = 2 ,padding=0)
		self.deconv5 = nn.ConvTranspose3d(8, 1, 3,stride = 2, padding=0)

	
		


	def forward(self, x):
		

		# Encoder
		x = (F.relu(self.bn1(self.conv1(x))))
		x = (F.relu(self.bn2(self.conv2(x))))
		# Decoder
		x = F.relu(self.bn3(self.deconv3(x)))
		x = F.relu(self.deconv4(x))
	
		return F.sigmoid((self.deconv5(x)))[:,0, :30,:30,:30]



class EncoderDecoderForNLL(nn.Module):
	"""A simple encoder-decoder style voxel superresolution network, intended for 
	use with the NLL loss. (The major change here is in the shape of each voxel 
	grid batch. It is now B x 2 x N x N x N, where B is the batchsize, 2 denotes the 
	occupancy classes (occupied vs unoccupied), and N is the number of voxels along 
	each dimension.)
	"""

	def __init__(self):
		super(EncoderDecoderForNLL, self).__init__()

		self.conv1 = nn.Conv3d(1, 16, 3, stride = 2, padding=1)
		self.bn1 = nn.BatchNorm3d(16)
		self.conv2 = nn.Conv3d(16, 32, 3, stride = 2, padding=1)
		self.bn2 = nn.BatchNorm3d(32)
		self.deconv3 = nn.ConvTranspose3d(32, 16, 3, stride = 2,  padding=1)
		self.bn3 = nn.BatchNorm3d(16)
		self.deconv4 = nn.ConvTranspose3d(16, 8, 3, stride = 2 ,padding=0)
		self.deconv5 = nn.ConvTranspose3d(8, 2, 3,stride = 2, padding=0)

	
		self.log_softmax = nn.LogSoftmax(dim=1)


	def forward(self, x):
		

		# Encoder
		x = (F.relu(self.bn1(self.conv1(x))))
		x = (F.relu(self.bn2(self.conv2(x))))
		# Decoder
		x = F.relu(self.bn3(self.deconv3(x)))
		x = F.relu(self.deconv4(x))
	
		return torch.exp(self.log_softmax(self.deconv5(x)))[:,:, :30,:30,:30]



