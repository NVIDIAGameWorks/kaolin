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


# from torch_geometric.nn import GCNConv

class VGG(nn.Module):
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
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
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
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
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
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2),
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
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2),
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

        return [A, B, C, D]


class G_Res_Net(nn.Module):
    def __init__(self, input_features, hidden=192, output_features=3):
        super(G_Res_Net, self).__init__()
        self.gc1 = ZNGCN(input_features, hidden)
        self.gc2 = ZNGCN(hidden, hidden)
        self.gc3 = ZNGCN(hidden , hidden)
        self.gc4 = ZNGCN(hidden, hidden)
        self.gc5 = ZNGCN(hidden , hidden)
        self.gc6 = ZNGCN(hidden, hidden)
        self.gc7 = ZNGCN(hidden , hidden)
        self.gc8 = ZNGCN(hidden, hidden)
        self.gc9 = ZNGCN(hidden , hidden)
        self.gc10 = ZNGCN(hidden, hidden)
        self.gc11 = ZNGCN(hidden , hidden)
        self.gc12 = ZNGCN(hidden, hidden)
        self.gc13 = ZNGCN(hidden , hidden)
        self.gc14 = ZNGCN(hidden, output_features)
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
        return coords, features


class ZNGCN(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(ZNGCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight1 = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 6. / math.sqrt((self.weight1.size(1) + self.weight1.size(0)))
        stdv *= .6


        # stdv = math.sqrt(6. / (self.weight1.size(1) + self.weight1.size(0)))
        # stdv*= .2
        self.weight1.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-.1, .1)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight1)
        side_len = max(support.shape[1] // 3, 2)
        if adj.type() == 'torch.cuda.sparse.FloatTensor': 

            norm = torch.sparse.mm(adj, torch.ones((support.shape[0], 1)).cuda())
            normalized_support = support[:, :side_len] / norm
            side_1 = torch.sparse.mm(adj, normalized_support)
        else: 
            side_1 = torch.mm(adj, support[:, :side_len])

        side_2 = support[:, side_len:]
        output = torch.cat((side_1, side_2), dim=1)

        if self.bias is not None:
            output = output + self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class MeshEncoder(nn.Module):
    def __init__(self, latent_length):
        super(MeshEncoder, self).__init__()
        self.h1 = ZNGCN(3, 60)
        self.h21 = ZNGCN(60, 60)
        self.h22 = ZNGCN(60, 60)
        self.h23 = ZNGCN(60, 60)
        self.h24 = ZNGCN(60, 120)
        self.h3 = ZNGCN(120, 120)
        self.h4 = ZNGCN(120, 120)
        self.h41 = ZNGCN(120, 150)
        self.h5 = ZNGCN(150, 200)
        self.h6 = ZNGCN(200, 210)
        self.h7 = ZNGCN(210, 250)
        self.h8 = ZNGCN(250, 300)
        self.h81 = ZNGCN(300, 300)
        self.h9 = ZNGCN(300, 300)
        self.h10 = ZNGCN(300, 300)
        self.h11 = ZNGCN(300, 300)
        self.reduce = ZNGCN(300, latent_length) 

    def resnet(self, features, res):
        temp = features[:, :res.shape[1]]
        temp = temp + res
        features = torch.cat((temp, features[:, res.shape[1]:]), dim=1)
        return features, features

    def forward(self, positions, adj):
        res = positions
        features = F.elu(self.h1(positions, adj))
        features = F.elu(self.h21(features, adj))
        features = F.elu(self.h22(features, adj))
        features = F.elu(self.h23(features, adj))
        features = F.elu(self.h24(features, adj))
        features = F.elu(self.h3(features, adj))
        features = F.elu(self.h4(features, adj))
        features = F.elu(self.h41(features, adj))
        features = F.elu(self.h5(features, adj))
        features = F.elu(self.h6(features, adj))
        features = F.elu(self.h7(features, adj))
        features = F.elu(self.h8(features, adj))
        features = F.elu(self.h81(features, adj))
        features = F.elu(self.h9(features, adj))
        features = F.elu(self.h10(features, adj))
        features = F.elu(self.h11(features, adj))

        latent = F.elu(self.reduce(features , adj))  
        latent = (torch.max(latent, dim=0)[0])      
        return latent


class VoxelDecoder(nn.Module): 
    def __init__(self, latent_length): 
        super(VoxelDecoder, self).__init__()
        self.fully = torch.nn.Sequential(
            torch.nn.Linear(latent_length, 512)
        )

        self.model = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 64, 4, stride=2, padding=(1, 1, 1), ), 
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
        voxels = F.sigmoid(decode)
        return voxels
