# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

import types
from typing import Callable

import torch
import torch.nn as nn


__all__ = [
    'SimplicitsMLP',
    'SkinningModule'
]

class SkinningModule(torch.nn.Module):
    r"""Base class for skinning weight modules.

    Handles bounding-box normalization and provides helpers to compute
    skinning weights (with a constant handle appended) and their spatial
    Jacobian for use in physics simulation.

    Args:
        bb_min (torch.Tensor or array-like, optional): Minimum corner of the
            bounding box used to normalize input points, of shape :math:`(3,)`.
            Defaults to ``torch.zeros(3)``.
        bb_max (torch.Tensor or array-like, optional): Maximum corner of the
            bounding box used to normalize input points, of shape :math:`(3,)`.
            Defaults to ``torch.ones(3)``.
    """
    def __init__(self, bb_min=None, bb_max=None):
        super().__init__()
        if bb_min is None:
            bb_min = torch.zeros(3, dtype=torch.float32)
        elif not torch.is_tensor(bb_min):
            bb_min = torch.tensor(bb_min, dtype=torch.float32)
        if bb_max is None:
            bb_max = torch.ones(3, dtype=torch.float32)
        elif not torch.is_tensor(bb_max):
            bb_max = torch.tensor(bb_max, dtype=torch.float32)
        assert (bb_min < bb_max).all(), f"bb_min must be greater than bb_max, received {bb_min} and {bb_max}."
        self.register_buffer('bb_min', bb_min)
        self.register_buffer('bb_max', bb_max)

    def _offset_scale(self, pts):
        return (pts - self.bb_min.to(pts.dtype)) / (self.bb_max.to(pts.dtype) - self.bb_min.to(pts.dtype))

    def compute_skinning_weights(self, pts):
        r"""
        Computes the skinning weights for the given points (including the normalization and padding with constant handle).

        Args:
            pts (torch.Tensor): The points to be skinned, of shape :math:`(N, 3)` (in :math:`m`).

        Returns:
            torch.Tensor: The skinning weights, of shape :math:`(N, \text{num_handles})`.
        """
        norm_pts = self._offset_scale(pts)
        return torch.cat([
            self(norm_pts),
            torch.ones((pts.shape[0], 1), device=pts.device, dtype=pts.dtype)
        ], dim=1)

    def compute_dwdx(self, pts):
        r"""
        Computes the Jacobian of the skinning weights with respect to the points.

        Args:
            pts (torch.Tensor): The points to be skinned, of shape :math:`(N, 3)` (in :math:`m`).

        Returns:
            torch.Tensor: The Jacobian of the skinning weights, of shape :math:`(N, \text{num_handles}, 3)`.
        """
        jac_single = torch.func.jacrev(lambda x: self.compute_skinning_weights(x[None])[0])
        return torch.vmap(jac_single)(pts)

    @staticmethod
    def from_function(function: Callable, bb_min=0, bb_max=1):
        r"""Create a SkinningModule from a function.

        Args:
            function (Callable): The function to be used to compute the skinning weights.
            bb_min (float or torch.Tensor or array-like, optional): Minimum corner of the bounding box. Defaults to 0.
            bb_max (float or torch.Tensor or array-like, optional): Maximum corner of the bounding box. Defaults to 1.

        Returns:
            SkinningModule: A SkinningModule whose ``forward`` delegates to ``function``.
        """
        skinning_mod = SkinningModule(bb_min=bb_min, bb_max=bb_max)
        def _forward(self, pts):
            return function(pts)
        skinning_mod.forward = types.MethodType(_forward, skinning_mod)
        return skinning_mod

class SimplicitsMLP(SkinningModule):
    r"""This implements an MLP with ELU activations

    Args:
          spatial_dimensions (int): 3 for 3D, 2 for 2D
          layer_width (int): layer width
          num_handles (int): number of handles (including the implicit constant handle)
          num_layers (int): number of layers in MLP
          bb_min (torch.Tensor or array-like, optional): Minimum corner of the bounding box,
              of shape :math:`(3,)`. Defaults to ``torch.zeros(3)``.
          bb_max (torch.Tensor or array-like, optional): Maximum corner of the bounding box,
              of shape :math:`(3,)`. Defaults to ``torch.ones(3)``.

    """
    def __init__(self, spatial_dimensions, layer_width, num_handles, num_layers, bb_min=None, bb_max=None):
        super().__init__(bb_min=bb_min, bb_max=bb_max)
        layers = []
        layers.append(nn.Linear(spatial_dimensions, layer_width))
        layers.append(nn.ELU())

        for n in range(0, num_layers):
            layers.append(nn.Linear(layer_width, layer_width))
            layers.append(nn.ELU())

        layers.append(nn.Linear(layer_width, num_handles - 1))

        self.linear_elu_stack = nn.Sequential(*layers)

    def forward(self, x):
        r"""Output of the network with the learned handles only (no rigid handle).

            Args:
                x (torch.Tensor): Tensor of spatial points in :math:`\mathbb{R}^dim`, of shape :math:`(\text{batch_dim}, \text{dim})`

            Returns:
                torch.Tensor: Skinning weights, of shape :math:`(\text{batch_dim}, \text{num_handles} - 1)`

        """
        output = self.linear_elu_stack(x)
        return output
