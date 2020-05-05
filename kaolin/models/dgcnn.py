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
#
#
# pytorch_DGCNN
#
# Copyright (c) 2018 Muhan Zhang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


def conv(
    type,
    input_dim,
    output_dim,
    batch_norm=True,
    leaky_relu=True,
    dropout=False
):
    layer = nn.Sequential()
    if type="Conv1d":
        layer.add_module(nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=False))
    elif type="Conv2d":
        layer.add_module(nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False))

    if batch_norm:
        layer.add_module(nn.BatchNorm2d(output_dim))
    if leaky_relu:
        layer.add_module(nn.LeakyReLU(negative_slope=0.2))
    if dropout:
        layer.add_module(nn.Dropout(p=0.5))
    return layer


def fc(
    input_dim,
    output_dim,
    batch_norm=False,
    leaky_relu=False,
    dropout=False
):
    layer = nn.Sequential(
        nn.Linear(input_dim, output_dim, bias=False)
    )
    if batch_norm:
        layer.add_module(nn.BatchNorm1d(output_dim))
    if leaky_relu:
        layer.add_module(nn.LeakyReLU(negative_slope=0.2))
    if dropout:
        layer.add_module(nn.Dropout(p=0.5))
    return layer


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(self, x, k=20, idx=None):
    """Compute Graph feature.

    Args:
        x (torch.tensor): torch tensor
        k (int): number of nearest neighbors
    """
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    if self.use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(
        -1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous(
    )  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class DGCNN_CLS(nn.Module):
    """Implementation of the DGCNN for pointcloud classificaiton.
    
    Args:
        input_dim (int): number of features per point. Default: ``3`` (xyz point coordinates)
        conv_dims (list): list of output feature dimensions of the convolutional layers. Default: ``[64,64,128,256]`` (as preoposed in original implementation).
        emb_dims (int): dimensionality of the intermediate embedding.
        fc_dims (list): list of output feature dimensions of the fully connected layers. Default: ``[512, 256]`` (as preoposed in original implementation).
        output_channels (int): number of output channels. Default: ``64``.
        dropout (float): dropout probability (applied to fully connected layers only). Default: ``0.5``.
        k (int): number of nearest neighbors.
        use_cuda (bool): if ``True`` will move the model to GPU

    .. note::

        If you use this code, please cite the original paper in addition to Kaolin.
        
        .. code-block::

            @article{dgcnn,
                title={Dynamic Graph CNN for Learning on Point Clouds},
                author={Wang, Yue and Sun, Yongbin and Liu, Ziwei and Sarma, Sanjay E. and Bronstein, Michael M. and Solomon, Justin M.},
                journal={ACM Transactions on Graphics (TOG)},
                year={2019}
            }

    """
    def __init__(
            self,
            input_dim=3,
            conv_dims=[64, 64, 128, 256],
            emb_dims=1024,  # dimension of embeddings
            fc_dims=[512, 256],
            output_channels=64,
            dropout=0.5,  # dropout probability
            k=20,  # number of nearest neighbors
            use_cuda=True,  # use CUDA or not
    ):
        super(DGCNN, self).__init__()
        self.k = k
        emb_input_dim = sum(conv_dims)

        self.conv_dims = [input_dim] + conv_dims

        for it in range(len(self.conv_dims) - 1):
            # self.conv_layers.append(
            self.__setattr__(
                f'conv_layers_{it}',
                get_layer(
                    conv(
                        "Conv2d",
                        self.conv_dims[it] * 2,
                        self.conv_dims[it + 1],
                    )
                )
            )
        
        # create intermediate embedding
        
        self.embedding_layer = get_layer(
            conv(
                "Conv1d",
                emb_input_dim,
                emb_dims
            )
        )

        # fully connected layers
        self.fc_dims = [emb_dims * 2] + fc_dims
        for it in range(len(self.fc_dims) - 1):
            self.__setattr__(f'fc_layers_{it}',
                get_layer(
                    fc(
                        self.fc_dims[it], 
                        self.fc_dims[it + 1],
                        batch_norm=True,
                        leaky_relu=True,
                        dropout=True
                    )
                )
            )

        # final output projection
        self.final_layer = fc(self.fc_dims[-1], output_channels)

        if use_cuda:
            self.cuda()

    def forward(self, x):
        """Forward pass of the DGCNN model.

        Args:
            x (torch.tensor): input to the network in the format: [B, N_feat, N_points]
        """
        batch_size = x.size(0)
        x_list = []

        # convolutional layers
        for it in range(len(self.conv_dims) - 1):
            x = self.get_graph_feature(x, k=self.k)
            x = self.__getattr__(f'conv_layers_{it}')(x)
            x = x.max(dim=-1, keepdim=False)[0]
            x_list.append(x)

        # embedding layer
        x = self.embedding_layer(torch.cat(x_list, dim=1))

        # prepare for FC layer input
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        # fully connected layers
        for it in range(len(self.fc_dims) - 1):
            x = self.__getattr__(f'fc_layers_{it}')(x)

        # final layer
        x = self.final_layer(x)

        return x


class Transform_Net(nn.Module):
    def __init__(
        self,
        input_dim=3,
        conv_2d_dims=[64, 128],
        conv_1d_dims=[1024],
        fc_dims=[512, 256],
        output_channels=9,
        use_cuda=True
    ):
        super(Transform_Net, self).__init__()

        self.conv1 = conv("Conv2d", input_dim*2, conv_2d_dims[0])
        self.conv2 = conv("Conv2d", conv_2d_dims[0], conv_2d_dims[1])
        self.conv3 = conv("Conv1d", conv_2d_dims[-1], conv_1d_dims[0])
        self.fc1 = fc(conv_1d_dims[-1], fc[0], batch_norm=True, leaky_relu=True) 
        self.fc2 = fc(fc[0], fc[1], batch_norm=True, leaky_relu=True)
        self.transform = fc(fc_dims[-1],output_channels)
        
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))
        if use_cuda:
            self.cuda()
    
    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(self.conv2(x))
        x = x.max(dim=-1, keepdim=False)[0]

        x = self.conv3(x)
        x = x.max(dim=-1, keepdim=False)[0] 

        x = self.fc2(self.fc1(x))
        x = self.transform(x)

        return x.view(batch_size, 3, 3)


class DGCNN_PARTSEG(nn.Module):
    def __init__(
        self,
        input_dim=3,
        emb_dims=
        output_dim=None,
        k=20,
        use_cuda=True
    ):
        super(DGCNN_PARTSEG, self).__init__()
        self.transform_net = Transform_Net()
        self.k = k
        
        self.conv1 = conv("Conv2d", input_dim*2, 64)
        self.conv2 = conv("Conv2d", 64, 64)
        self.conv3 = conv("Conv2d", 64*2, 64)
        self.conv4 = conv("Conv2d", 64, 64)
        self.conv5 = conv("Conv2d", 64*2, 64)
        self.conv6 = conv("Conv1d", 192, emb_dims)
        self.conv7 = conv("Conv1d", 16, 64)
        self.conv8 = conv("Conv1d", 1280, 256, dropout=True)
        self.conv9 = conv("Conv1d", 256, 256, dropout=True)
        self.conv10 = conv("Conv1d", 256, 128)
        self.conv11 = conv(
            "Conv1d", 128, output_dim,
            batch_norm=False, leaky_relu=False
        )
        if use_cuda:
            self.cuda()

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)
        t = self.transform_net(x0)  
        x = x.transpose(2, 1) 
        x = torch.bmm(x, t) 
        x = x.transpose(2, 1)

        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x)
        x = x.max(dim=-1, keepdim=True)[0]

        l = l.view(batch_size, -1, 1)
        l = self.conv7(l)

        x = torch.cat((x, l), dim=1)
        x = x.repeat(1, 1, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
    

class DGCNN_SEMSEG(nn.Module):
    def __init__(
        self,
        input_dim=3,
        emb_dims=
        output_dim=None,
        k=20,
        use_cuda=True
    ):
        self.k = k
        self.conv1 = conv("Conv2d", 18, 64)
        self.conv2 = conv("Conv2d", 64, 64)
        self.conv3 = conv("Conv2d", 64*2, 64)
        self.conv4 = conv("Conv2d", 64, 64)
        self.conv5 = conv("Conv2d", 64*2, 64)
        self.conv6 = conv("Conv1d", 192, emb_dims)
        self.conv7 = conv("Conv1d", 1216, 512)
        self.conv8 = conv("Conv1d", 512, 256, dropout=True)
        self.conv9 = conv(
            "Conv1d", 256, 13,
            batch_norm=False, leaky_relu=False, dropout=False
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k, dim9=True)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv5(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3), dim=1)

        x = self.conv6(x)
        x = x.max(dim=-1, keepdim=True)[0]

        x = x.repeat(1, 1, num_points)
        x = torch.cat((x, x1, x2, x3), dim=1)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)

        return x