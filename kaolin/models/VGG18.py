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

import math

import torch 
from torch import nn 
import torch.nn.functional as F


class VGG18(nn.Module):
    r"""
    .. note::

        If you use this code, please cite the original paper in addition to Kaolin.
        
        .. code-block::

            @InProceedings{Simonyan15,
              author       = "Karen Simonyan and Andrew Zisserman",
              title        = "Very Deep Convolutional Networks for Large-Scale Image Recognition",
              booktitle    = "International Conference on Learning Representations",
              year         = "2015",
            }
    """
    
    def __init__(self):
        super(VGG, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1), 
            nn.BatchNorm2d(16), 
            nn.ReLU(inplace=True))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), 
            nn.ReLU(inplace=True))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride = 2),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True))

        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True))  

        self.layer5 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(inplace=True))

        self.layer6 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride = 2),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True))

        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True))

        self.layer8 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(inplace=True))

        self.layer9 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride = 2),
            nn.BatchNorm2d(128),    
            nn.ReLU(inplace=True))

        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),    
            nn.ReLU(inplace=True))

        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),    
            nn.ReLU(inplace=True))

        self.layer12 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride = 2),
            nn.BatchNorm2d(256),    
            nn.ReLU(inplace=True))

        self.layer13 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

    
        self.layer14 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.layer15 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride = 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.layer16 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.layer17 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))
        self.layer18 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

    def forward(self, tensor):
        x = self.layer1(tensor)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        A = x 
        x = self.layer9(x) 
        x = self.layer10(x)
        x = self.layer11(x)
        B = x 
        x = self.layer12(x)
        x = self.layer13(x)
        x = self.layer14(x)
        C = x
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        D = self.layer18(x)

        return [A,B,C,D]