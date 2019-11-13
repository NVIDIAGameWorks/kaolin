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
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import torch
from torch.nn import Parameter

from .SimpleGCN import SimpleGCN


class GraphResNet(nn.Module):
    r"""An enhanced version of the MeshEncoder; used residual connections
    across graph convolution layers.

    """

    def __init__(self, input_features, hidden = 192, output_features = 3):
        super(G_Res_Net, self).__init__()
        self.gc1 = SimpleGCN(input_features, hidden)
        self.gc2 = SimpleGCN(hidden, hidden)
        self.gc3 = SimpleGCN(hidden , hidden)
        self.gc4 = SimpleGCN(hidden, hidden)
        self.gc5 = SimpleGCN(hidden , hidden)
        self.gc6 = SimpleGCN(hidden, hidden)
        self.gc7 = SimpleGCN(hidden , hidden)
        self.gc8 = SimpleGCN(hidden, hidden)
        self.gc9 = SimpleGCN(hidden , hidden)
        self.gc10 = SimpleGCN(hidden, hidden)
        self.gc11 = SimpleGCN(hidden , hidden)
        self.gc12 = SimpleGCN(hidden, hidden)
        self.gc13 = SimpleGCN(hidden , hidden)
        self.gc14 = SimpleGCN(hidden,  output_features)
        self.hidden = hidden

    def forward(self, features, adj):

        x = (F.relu(self.gc1(features, adj)))
        x = (F.relu(self.gc2(x, adj)))
        features = features[..., :self.hidden] + x
        features /= 2.
        # 2
        x = (F.relu(self.gc3(features, adj)))
        x = (F.relu(self.gc4(x, adj)))
        features = features + x
        features /= 2.
        # 3
        x = (F.relu(self.gc5(features, adj)))
        x = (F.relu(self.gc6(x, adj)))
        features = features + x
        features /= 2.

        # 4
        x = (F.relu(self.gc7(features, adj)))
        x = (F.relu(self.gc8(x, adj)))
        features = features + x
        features /= 2.

        # 5
        x = (F.relu(self.gc9(features, adj)))
        x = (F.relu(self.gc10(x, adj)))
        features = features + x
        features /= 2.

        # 6
        x = (F.relu(self.gc11(features, adj)))
        x = (F.relu(self.gc12(x, adj)))
        features = features + x
        features /= 2.

        # 7
        x = (F.relu(self.gc13(features, adj)))

        features = features + x
        features /= 2.

        coords = (self.gc14(features, adj))
        return coords,features
