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
import kaolin.physics.utils as utils 

__all__ = [
    'Gravity',
    'Floor',
    'Boundary'
]
class Gravity(utils.ForceWrapper): 
    r"""Gravity wrapper applies gravity potential energy, gradient and hessian."""
    
    def __init__(self, rhos, acceleration):
        r"""Sets up the wrapper for gravity energy, gradient and hessian

        Args:
            rhos (torch.Tensor): :math:`(\text{num_samples})` rhos per sample
            acceleration (torch.Tensor): :math:`(\text{num_samples}, 3)` accelaration due to gravity
        """
        dim = acceleration.shape[0]
        device = rhos.device
        dtype = rhos.dtype
        num_samples = rhos.shape[0]
        self.pt_wise_acc = acceleration.expand(num_samples, dim)
        self.rhos = rhos
        self.zero_hess = torch.zeros(num_samples, num_samples, dim, dim, device=device, dtype=dtype)
    
    def _energy(self, x):
        r"""Implements gravity potential energy

        Args:
            x (torch.Tensor): Points in R^3, of shape :math:`(\text{num_samples}, 3)`

        Returns:
            torch.Tensor: Per-primitive energy, of shape :math:`(\text{num_samples}, 1)`
        """
        return self.rhos*torch.sum(x*self.pt_wise_acc, dim=1, keepdim=True)

    def _gradient(self, x):
        r"""Implements gravity gradient

        Args:
            x (torch.Tensor): (Unused) Points in R^3, of shape :math:`(\text{num_samples}, 3)`

        Returns:
            torch.Tensor: Per-primitive gradient , of shape :math:`(\text{num_samples}, 3)`
        """
        return self.rhos.expand(-1,self.pt_wise_acc.shape[1])*self.pt_wise_acc

    def _hessian(self, x):
        r"""Implements gravity hessian

        Args:
                x (torch.Tensor): Points in R^3, of shape :math:`(\text{num_samples}, 3)`

            Returns:
                torch.Tensor: Per-primitive hessian blocks, of shape :math:`(\text{num_samples}, num_samples, 3, 3)`
        """
        return self.zero_hess
        
class Floor(utils.ForceWrapper):
    r"""Floor wrapper applies a floor penalty energy, gradient and hessian on object. Ensures the object stays above floor_height via a quadratic penalty."""
    
    def __init__(self, floor_height, floor_axis, flip_floor=False):
        r"""Initializes floor object at floor_height, along floor_axis and in the positive direction unless flip_floor is True.

        Args:
            floor_height (float): floor height
            floor_axis (int): axis, 0 is x, 1 is y, 2 is z
        """
        self.floor_height = floor_height 
        self.floor_axis = floor_axis
        self.flip_floor = flip_floor
        self.cached_hess = None
    
    def _energy(self, x):
        r"""Implements floor contact energy

        Args:
            x (torch.Tensor): Points in R^3, of shape :math:`(\text{num_samples}, 3)`

        Returns:
            torch.Tensor: Per-primitive energy, of shape :math:`(\text{num_samples}, 1)`
        """
        #### Floor
        column_vec = x[:, self.floor_axis]
        # Calculate the distance of each y-coordinate from floor height
        distances = torch.abs(column_vec  - self.floor_height)
        if(self.flip_floor):
            # Create a new list with values based on the sign of the distance
            result = torch.where(column_vec <= self.floor_height, torch.zeros_like(distances), distances)
        else:
            # Create a new list with values based on the sign of the distance
            result = torch.where(column_vec >= self.floor_height, torch.zeros_like(distances), distances)
        pt_wise_energy = result**2
        return pt_wise_energy

    def _gradient(self, x):
        r"""Implements floor gradient

        Args:
            x (torch.Tensor): Points in R^3, of shape :math:`(\text{num_samples}, 3)`

        Returns:
            torch.Tensor: Per-primitive gradient , of shape :math:`(\text{num_samples}, 3)`
        """
        return torch.autograd.functional.jacobian(lambda p: torch.sum(self.energy(p)), inputs=x)

    def _hessian(self, x):
        r"""Implements floor hessian

        Args:
                x (torch.Tensor): Points in R^3, of shape :math:`(\text{num_samples}, 3)`

            Returns:
                torch.Tensor: Per-primitive hessian blocks, of shape :math:`(\text{num_samples}, num_samples, 3, 3)`
        """
        if self.cached_hess == None:
            self.cached_hess = torch.zeros(x.shape[0], x.shape[0], x.shape[1], x.shape[1], device=x.device, dtype=x.dtype)

        self.cached_hess.zero_()
        column_vec = x[:, self.floor_axis]
        if(self.flip_floor):
            idx_below_floor = torch.nonzero(column_vec > self.floor_height)
            self.cached_hess[idx_below_floor,idx_below_floor,self.floor_axis,self.floor_axis] = 2 
        else:
            idx_below_floor = torch.nonzero(column_vec < self.floor_height)
            self.cached_hess[idx_below_floor,idx_below_floor,self.floor_axis,self.floor_axis] = 2 
        
        return self.cached_hess # big sparse tensor n x n x 3 x 3

class Boundary(utils.ForceWrapper):
    r"""Boundary wrapper applies a dirichlet boundary condition to a set of points. Initializes to an empty boundary, but when pinned vertices are set, it applies an quadratic penalty to enforce boundary conditions."""
    
    def __init__(self):
        r"""Initializes empty dirichlet boundary. Set it up by setting pinned vertices of object. Update them to move the bdry.
        """
        self.pinned_indices = None 
        self.pinned_vertices = None 
        self.cached_hess = None
    
    def set_pinned_verts(self, idx=None, pos=None):
        r""" Sets pinned points and indices

        Args:
            idx (torch.Tensor): Pinned vertex indices, of shape :math:`(\text{num_pinned})`
            pos (torch.Tensor): Pinned vertex positions, of shape :math:`(\text{num_pinned}, 3)` 

        """
        # If indices is None, pin nothing. 
        self.pinned_indices = idx 
        self.pinned_vertices = pos

    def update_pinned(self, pos):
        r""" Updates pinned points

        Args:
            pos (torch.Tensor): Pinned vertex positions, of shape :math:`(\text{num_pinned}, 3)` 

        """
        if self.pinned_indices is not None:
            self.pinned_vertices = pos
        
    def _energy(self, x):
        r"""Implements boundary energy

        Args:
            x (torch.Tensor): Points in R^3, of shape :math:`(\text{num_samples}, 3)`

        Returns:
            torch.Tensor: Per-primitive energy, of shape :math:`(\text{num_samples}, 1)`
        """
        pt_wise_en = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        if self.pinned_indices == None:
            return pt_wise_en
        pt_wise_en[self.pinned_indices] = torch.sum(torch.square(x[self.pinned_indices] - self.pinned_vertices), dim=1)
        return pt_wise_en
    
    def _gradient(self, x):
        r"""Implements boundary gradient

        Args:
            x (torch.Tensor): Points in R^3, of shape :math:`(\text{num_samples}, 3)`

        Returns:
            torch.Tensor: Per-primitive gradient , of shape :math:`(\text{num_samples}, 3)`
        """
        if self.pinned_indices == None:
            return torch.zeros_like(x)
        return torch.autograd.functional.jacobian(lambda p: torch.sum(self.energy(p)), inputs=x)
    
    def _hessian(self, x):
        r""" Boundary hessian

            Args:
                x (torch.Tensor): Points in R^3, of shape :math:`(\text{num_samples}, 3)`

            Returns:
                torch.Tensor: Per-primitive hessian blocks, of shape :math:`(\text{num_samples}, num_samples, 3, 3)`
        """
        if self.cached_hess == None:
            self.cached_hess = torch.zeros(x.shape[0], x.shape[0], x.shape[1], x.shape[1], device=x.device, dtype=x.dtype)
        self.cached_hess.zero_()
        
        if self.pinned_indices == None:
            return self.cached_hess
        self.cached_hess[self.pinned_indices,self.pinned_indices] = 2*torch.eye(x.shape[1], device=x.device, dtype=x.dtype) 
        return self.cached_hess # big sparse matrix