# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch 

__all__ = [
    'precompute_fiber_matrix',
    'unbatched_muscle_energy',
    'unbatched_muscle_gradient',
    'unbatched_muscle_hessian'
]

def precompute_fiber_matrix(fiber_vecs):
	r""" Precompute the fiber matrix blocks. Muscle energy is E = 0.5 v'F'Fv where DOFs are F here, its rewritten as E = 0.5 flat(F)' B' B flat(F) where B encodes fibers

	Args:
		fiber_vecs (torch.Tensor): Muscle fiber direction, of shape :math:`(\text{num_samples}, 3)`

	Returns:
		torch.Tensor: Returns matrix that encodes fibers, of shape :math:`(\text{num_samples}, 3, 9)`
	"""
	# E = 0.5 * activation * v'F'Fv 
	#	== 0.5*activation * flat(F)'*fiber_mat'*fiber_mat*flat(F)
	def build_mat(fiber_vec):
		return torch.kron(torch.eye(3, device=fiber_vecs.device), fiber_vec)
	blocks = torch.vmap(build_mat, randomness="same")(fiber_vecs)
	# list_blocks = [blocks[i] for i in range(blocks.shape[0])]
	return blocks


def unbatched_muscle_energy(activation, fiber_mat_blocks, defo_grad):
	r""" Computes muscle energy

	Args:
		activation (float): Scalar muscle activation
		fiber_mat_blocks (torch.Tensor): Matrix encodes fiber directions, of shape :math:`(\text{num_samples}, 3, 9)`
		defo_grad (torch.Tensor): Flattened 3d deformation gradients, of shape :math:`(\text{batch_dim}*3*3, 1)`

	Returns:
		torch.Tensor: Vector of per-primitive energy values, of shape :math:`(\text{batch_dim}, 1)`
	"""
	# e = a*v'F'Fv
	defo_grad_batchwise_flat = defo_grad.reshape(-1,9)
	Bf = torch.bmm(fiber_mat_blocks, defo_grad_batchwise_flat.unsqueeze(-1))
	return 0.5*activation*torch.sum(Bf*Bf, dim=1)
	#Equivalently
	# B = fiber_mat_blocks
	# F = defo_grad
	# BB = torch.bmm(B.transpose(1,2), B)
	# BBF = torch.bmm(BB, F.reshape(-1,9).unsqueeze(-1))
	# FBBF = torch.bmm(F.reshape(-1,9).unsqueeze(1), BBF)
	# return 0.5*activation*FBBF.squeeze(-1)

def unbatched_muscle_gradient(activation, fiber_mat_blocks, defo_grad):
	r""" Computes muscle gradient

	Args:
		activation (float): Scalar muscle activation
		fiber_mat_blocks (torch.Tensor): Matrix encodes fiber directions, of shape :math:`(\text{num_samples}, 3, 9)`
		defo_grad (torch.Tensor): Flattened 3d deformation gradients, of shape :math:`(\text{batch_dim}*3*3, 1)`

	Returns:
		torch.Tensor: Tensor of per-primitive gradients, of shape :math:`(\text{batch_dim}, 9)`
	"""
	BB = torch.bmm(fiber_mat_blocks.transpose(1,2), fiber_mat_blocks) # TODO (Vismay): Can be precomputed and stored
	return activation*torch.bmm(BB, defo_grad.reshape(-1,9).unsqueeze(-1)).squeeze()


def unbatched_muscle_hessian(activation, fiber_mat_blocks, defo_grad):
	r""" Computes muscle hessian

	Args:
		activation (float): Scalar muscle activation
		fiber_mat_blocks (torch.Tensor): Matrix encodes fiber directions, of shape :math:`(\text{num_samples}, 3, 9)`
		defo_grad (torch.Tensor): Flattened 3d deformation gradients, of shape :math:`(\text{batch_dim}*3*3, 1)`

	Returns:
		torch.Tensor: Hessian blocks per-primitive, of shape :math:`(\text{batch_dim}, 9, 9)`
	"""
	return activation*torch.bmm(fiber_mat_blocks.transpose(1,2), fiber_mat_blocks)