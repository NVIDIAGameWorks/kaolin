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
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


class DGCNN(nn.Module):
    """Implementation of the DGCNNfor pointcloud classificaiton.
    
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
        self.use_cuda = use_cuda

        emb_input_dim = sum(conv_dims)
        self.conv_dims = [input_dim] + conv_dims

        for it in range(len(self.conv_dims) - 1):
            # self.conv_layers.append(
            self.__setattr__(f'conv_layers_{it}',
                self.get_layer(
                    nn.Sequential(
                        nn.Conv2d(self.conv_dims[it] * 2,
                                  self.conv_dims[it + 1],
                                  kernel_size=1,
                                  bias=False),
                        nn.BatchNorm2d(self.conv_dims[it + 1]),
                        nn.LeakyReLU(negative_slope=0.2))))
        
        # create intermediate embedding
        self.embedding_layer = self.get_layer(
            nn.Sequential(
                nn.Conv1d(emb_input_dim, emb_dims, kernel_size=1, bias=False),
                nn.BatchNorm1d(emb_dims), nn.LeakyReLU(negative_slope=0.2)))

        # fully connected layers
        self.fc_dims = [emb_dims * 2] + fc_dims
        for it in range(len(self.fc_dims) - 1):
            self.__setattr__(f'fc_layers_{it}',
                self.get_layer(
                    nn.Sequential(
                        nn.Linear(self.fc_dims[it], self.fc_dims[it + 1], bias=False),
                        nn.BatchNorm1d(self.fc_dims[it + 1]),
                        nn.LeakyReLU(negative_slope=0.2),
                        nn.Dropout(p=dropout))))

        # final output projection
        self.final_layer = self.get_layer(
            nn.Linear(self.fc_dims[-1], output_channels))

    def get_layer(self, layer):
        """Convert the layer to cuda if needed.

        Args:
            layer: torch.nn layer
        """
        if self.use_cuda:
            return layer.cuda()
        else:
            return layer

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
