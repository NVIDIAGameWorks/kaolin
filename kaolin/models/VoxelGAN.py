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
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """TODO: Add docstring.

    https://arxiv.org/abs/1610.07584

    Input shape: B x 200 (B -> batchsize, 200 -> latent code size)
    Output shape: B x 1 x 32 x 32 x 32


    .. note::

        If you use this code, please cite the original paper in addition to Kaolin.
        
        .. code-block::

            @inproceedings{3dgan,
              title={Learning a probabilistic latent space of object shapes via 3d generative-adversarial modeling},
              author={Wu, Jiajun and Zhang, Chengkai and Xue, Tianfan and Freeman, William T and Tenenbaum, Joshua B},
              booktitle={Advances in Neural Information Processing Systems},
              pages={82--90},
              year={2016}
            }

            
    """

    def __init__(self):
        super(Generator, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(200, 512, 4, 2, 0),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 256, 4, 2, 1),
            torch.nn.BatchNorm3d(256),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, 4, 2, 1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 1, 4, 2, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 200, 1, 1, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class Discriminator(nn.Module):
    """TODO: Add docstring.
    
    https://arxiv.org/abs/1610.07584

    Input shape: B x 1 x 32 x 32 x 32
    Output shape: B x 1 x 1 x 1 x 1
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 64, 4, 2, 1),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, 4, 2, 1),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, 4, 2, 1),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(256, 512, 4, 2, 1),
            torch.nn.BatchNorm3d(512),
            torch.nn.LeakyReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(512, 1, 2, 2, 0),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 1, 32, 32, 32)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x
