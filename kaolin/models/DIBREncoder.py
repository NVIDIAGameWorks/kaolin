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


class DIBREncoder(nn.Module):
    r"""Encoder architecture used for single-image based mesh prediction in
    the Neurips 2019 paper "Learning to Predict 3D Objects with an
    Interpolation-based Differentiable Renderer"
    
    .. note::

        If you use this code, please cite the original paper in addition to Kaolin.
        
        .. code-block::

            @inproceedings{chen2019dibrender,
                title={Learning to Predict 3D Objects with an Interpolation-based Differentiable Renderer},
                author={Wenzheng Chen and Jun Gao and Huan Ling and Edward Smith and Jaakko Lehtinen and Alec Jacobson and Sanja Fidler},
                booktitle={Advances In Neural Information Processing Systems},
                year={2019}
            }
            
    """

    def __init__(self, N_CHANNELS, N_KERNELS, \
                 BATCH_SIZE, IMG_DIM, VERTS):
        super(Encoder, self).__init__()
        
        block1 = self.convblock(N_CHANNELS, 32, N_KERNELS, stride=2, pad=2)
        block2 = self.convblock(32, 64, N_KERNELS, stride=2, pad=2)
        block3 = self.convblock(64, 128, N_KERNELS, stride=2, pad=2)
        block4 = self.convblock(128, 128, N_KERNELS, stride=2, pad=2)
        
        linear1 = self.linearblock(10368, 1024)
        linear2 = self.linearblock(1024, 1024)
        self.linear3 = nn.Linear(1024, 1024)
        
        linear4 = self.linearblock(1024, 1024)
        linear5 = self.linearblock(1024, 2048)
        self.linear6 = nn.Linear(2048, VERTS*6)
       
        #################################################
        all_blocks = block1 + block2 + block3 + block4
        self.encoder1 = nn.Sequential(*all_blocks)
        
        all_blocks = linear1 + linear2
        self.encoder2 = nn.Sequential(*all_blocks)
        
        all_blocks = linear4 + linear5
        self.decoder = nn.Sequential(*all_blocks)
        
        # Initialize with Xavier Glorot
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) \
            or isinstance(m, nn.Linear) \
            or isinstance(object, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.weight, mean=0, std=0.001)

        # Free some memory
        del all_blocks, block1, block2, block3, \
        linear1, linear2, linear4, linear5, \
    
    def convblock(self, indim, outdim, ker, stride, pad):
        block2 = [
            nn.Conv2d(indim, outdim, ker, stride, pad),
            nn.BatchNorm2d(outdim),
            nn.ReLU()
        ]
        return block2
    
    def linearblock(self, indim, outdim):
        block2 = [
            nn.Linear(indim, outdim),
            nn.BatchNorm1d(outdim),
            nn.ReLU()
        ]
        return block2
        
    def forward(self, x):
        
        for layer in self.encoder1:
            x = layer(x)
        
        bnum = x.shape[0] 
        x = x.view(bnum, -1) 
        for layer in self.encoder2:
            x = layer(x)
        x = self.linear3(x)
        
       
        for layer in self.decoder:
            x = layer(x)
        x = self.linear6(x).view(x.shape[0], -1,6)
        verts = x[:,:,:3]
        colors = x[:,:,3:]
        return verts, colors
