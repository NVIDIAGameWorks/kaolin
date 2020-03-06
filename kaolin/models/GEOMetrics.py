# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
from torch import nn 


class VoxelDecoder(nn.Module):
    r"""
    .. note::

        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::

            @InProceedings{smith19a,
                title = {{GEOM}etrics: Exploiting Geometric Structure for Graph-Encoded Objects},
                author = {Smith, Edward and Fujimoto, Scott and Romero, Adriana and Meger, David},
                booktitle = {Proceedings of the 36th International Conference on Machine Learning},
                pages = {5866--5876},
                year = {2019},
                volume = {97},
                series = {Proceedings of Machine Learning Research},
                publisher = {PMLR},
            }

    """
    def __init__(self, latent_length): 
        super(VoxelDecoder, self).__init__()
        self.fully = torch.nn.Sequential(
            torch.nn.Linear(latent_length, 512)
        )

        self.model = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 64, 4, stride=2, padding=(1, 1, 1)), 
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),

            torch.nn.ConvTranspose3d(64, 64, 4, stride=2, padding=(1, 1, 1)), 
            nn.BatchNorm3d(64),
            nn.ELU(inplace=True),

            torch.nn.ConvTranspose3d(64, 32, 4, stride=2, padding=(1, 1, 1)), 
            nn.BatchNorm3d(32),
            nn.ELU(inplace=True),

            torch.nn.ConvTranspose3d(32, 8, 4, stride=2, padding=(1, 1, 1)), 
            nn.BatchNorm3d(8),
            nn.ELU(inplace=True),

            nn.Conv3d(8, 1, (3, 3, 3), stride=1, padding=(1, 1, 1))
        )

    def forward(self, latent):
        decode = self.fully(latent).view(-1, 64, 2, 2, 2)
        decode = self.model(decode).reshape(-1, 32, 32, 32)
        voxels = torch.sigmoid(decode)
        return voxels
