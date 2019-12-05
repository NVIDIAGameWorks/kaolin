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

import math

import torch
from torch import nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class VGG(nn.Module):
    def __init__(self, channels=4):
        super(VGG, self).__init__()

        self.layer0_1 = nn.Conv2d(channels, 16, 3, stride=1, padding=1)
        self.layer0_2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.layer1_1 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.layer1_2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.layer1_3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)

        self.layer2_1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.layer2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.layer2_3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)

        self.layer3_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.layer3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.layer3_3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.layer4_1 = nn.Conv2d(128, 256, 5, stride=2, padding=2)
        self.layer4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.layer4_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1)

        self.layer5_1 = nn.Conv2d(256, 512, 5, stride=2, padding=2)
        self.layer5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.layer5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1)
        self.layer5_4 = nn.Conv2d(512, 512, 3, stride=1, padding=1)

    def forward(self, img):
        img = F.relu(self.layer0_1(img))
        img = F.relu(self.layer0_2(img))

        img = F.relu(self.layer1_1(img))
        img = F.relu(self.layer1_2(img))
        img = F.relu(self.layer1_3(img))

        img = F.relu(self.layer2_1(img))
        img = F.relu(self.layer2_2(img))
        img = F.relu(self.layer2_3(img))
        A = torch.squeeze(img)

        img = F.relu(self.layer3_1(img))
        img = F.relu(self.layer3_2(img))
        img = F.relu(self.layer3_3(img))
        B = torch.squeeze(img)

        img = F.relu(self.layer4_1(img))
        img = F.relu(self.layer4_2(img))
        img = F.relu(self.layer4_3(img))
        C = torch.squeeze(img)

        img = F.relu(self.layer5_1(img))
        img = F.relu(self.layer5_2(img))
        img = F.relu(self.layer5_3(img))
        img = F.relu(self.layer5_4(img))
        D = torch.squeeze(img)

        return [A, B, C, D]


class G_Res_Net(nn.Module):
    def __init__(self, input_features, hidden=128, output_features=3):
        super(G_Res_Net, self).__init__()
        self.gc1 = GCN(input_features, hidden)
        self.gc2 = GCN(hidden, hidden)
        self.gc3 = GCN(hidden, hidden)
        self.gc4 = GCN(hidden, hidden)
        self.gc5 = GCN(hidden, hidden)
        self.gc6 = GCN(hidden, hidden)
        self.gc7 = GCN(hidden, hidden)
        self.gc8 = GCN(hidden, hidden)
        self.gc9 = GCN(hidden, hidden)
        self.gc10 = GCN(hidden, hidden)
        self.gc11 = GCN(hidden, hidden)
        self.gc12 = GCN(hidden, hidden)
        self.gc13 = GCN(hidden, hidden)
        self.gc14 = GCN(hidden, output_features)
        self.hidden = hidden

    def forward(self, features, adj):
        features = features.unsqueeze(0)

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
        return coords.squeeze(0), features.squeeze(0)

class GCN(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = .6 / math.sqrt((self.weight.size(1) + self.weight.size(0)))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-.1, .1)

    def forward(self, input, adj):

        support = torch.bmm(input, self.weight.unsqueeze(0).expand(input.shape[0], -1, -1))

        output = torch.bmm(adj.unsqueeze(0).expand(input.shape[0], -1, -1), support)

        output = output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
