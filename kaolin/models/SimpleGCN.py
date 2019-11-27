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


class SimpleGCN(nn.Module):
    r"""A simple graph convolution layer, similar to the one defined in
    Kipf et al. https://arxiv.org/abs/1609.02907

    .. note::

        If you use this code, please cite the original paper in addition to Kaolin.

        .. code-block::

            @article{kipf2016semi,
              title={Semi-Supervised Classification with Graph Convolutional Networks},
              author={Kipf, Thomas N and Welling, Max},
              journal={arXiv preprint arXiv:1609.02907},
              year={2016}
            }

    """

    def __init__(self, in_features, out_features, bias=True):
        super(SimpleGCN, self).__init__()
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
