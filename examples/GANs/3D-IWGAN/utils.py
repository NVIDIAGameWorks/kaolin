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
import numpy as np
from torchvision import transforms
from torchvision.transforms import Normalize as norm 


np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

def gradient_penalty(netD, real_data, fake_data):
	batch_size = real_data.shape[0]
	dim = real_data.shape[2]
	alpha = torch.rand(batch_size, 1)
	alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
	alpha = alpha.view(batch_size, 32, 32, 32)
	alpha = alpha.cuda()

	
	fake_data = fake_data.view(batch_size, 32, 32, 32)
	interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
	
	
	interpolates.requires_grad_(True)

	disc_interpolates = netD(interpolates)
	

	gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
							  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
							  create_graph=True, retain_graph=True)[0]
	

                            
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
	return gradient_penalty


def calculate_gradient_penalty(netD, real_images, fake_images):
		batch_size = real_images.shape[0]
		eta = torch.rand(batch_size, 1)
		eta = eta.expand(batch_size, int(real_images.nelement()/batch_size)).contiguous()
		eta = eta.view(batch_size, 32, 32, 32)
		eta = eta.cuda()
		
	  

		interpolated = eta * real_images + ((1 - eta) * fake_images)
		
		

		# define it to calculate gradient
		interpolated.requires_grad_(True)

		# calculate probability of interpolated examples
		prob_interpolated = netD(interpolated)
		

		# calculate gradients of probabilities with respect to examples
		gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
							   grad_outputs=torch.ones(
								   prob_interpolated.size()).cuda(),
							   create_graph=True, retain_graph=True)[0]
	
		grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
		return grad_penalty