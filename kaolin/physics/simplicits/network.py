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
import torch.nn as nn

__all__ = [
    'SimplicitsMLP',
]


class SimplicitsMLP(nn.Module):
    r"""This implements an MLP with ELU activations

    Args:
          spatial_dimensions (int): 3 for 3D, 2 for 2D
          layer_width (int): layer width
          num_handles (int): number of handles
          num_layers (int): number of layers in MLP

    """

    def __init__(self, spatial_dimensions, layer_width, num_handles, num_layers):
        super(SimplicitsMLP, self).__init__()

        layers = []
        layers.append(nn.Linear(spatial_dimensions, layer_width))
        layers.append(nn.ELU())

        for n in range(0, num_layers):
            layers.append(nn.Linear(layer_width, layer_width))
            layers.append(nn.ELU())

        layers.append(nn.Linear(layer_width, num_handles))

        self.linear_elu_stack = nn.Sequential(*layers)

    def forward(self, x):
        r""" Calls the network

            Args:
                x (torch.Tensor): Tensor of spatial points in :math:`\mathbb{R}^dim`, of shape :math:`(\text{batch_dim}, \text{dim})`

            Returns:
                torch.Tensor: Skinning weights, of shape :math:`(\text{batch_dim}, \text{num_handles})`

        """
        output = self.linear_elu_stack(x)
        return output
