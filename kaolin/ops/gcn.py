# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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


def sparse_bmm(sparse_matrix, dense_matrix_batch):
    """
    Perform torch.bmm on an unbatched sparse matrix and a batched dense matrix.

    Args:
        sparse_matrix (torch.sparse.FloatTensor): Shape = (m, n)
        dense_matrix_batch (torch.FloatTensor): Shape = (b, n, p)

    Returns:
        (torch.FloatTensor):
            Result of the batched matrix multiplication. Shape = (b, n, p)
    """
    m = sparse_matrix.shape[0]
    b, n, p = dense_matrix_batch.shape
    # Stack the matrix batch into columns. (b, n, p) -> (n, b * p)
    dense_matrix = dense_matrix_batch.transpose(0, 1).reshape(n, b * p)
    result = torch.sparse.mm(sparse_matrix, dense_matrix)
    # Reverse the reshaping. (m, b * p) -> (b, m, p)
    return result.reshape(m, b, p).transpose(0, 1)


def normalize_adj(adj):
    """
    Normalize the adjacency matrix with shape = (num_nodes, num_nodes) such that
    the sum of each row is 1.

    This operation is slow, so it should be done only once for a graph and then
    reused.

    This supports both sparse tensor and regular tensor. The return type will be
    the same as the input type. For example, if the input is a sparse tensor,
    the normalized matrix will also be a sparse tensor.

    Args:
        adj (torch.sparse.FloatTensor or torch.FloatTensor):
            Shape = (num_nodes, num_nodes)
            The adjacency matrix.

    Returns:
        (torch.sparse.FloatTensor or torch.FloatTensor):
            A new adjacency matrix with the same connectivity as the input, but
            with the sum of each row normalized to 1.
    """
    if adj.type().endswith('sparse.FloatTensor'):
        norm = torch.sparse.mm(adj, torch.ones((adj.shape[0], 1),
                                               device=adj.device)).squeeze(1)
        indices = adj._indices()
        values = adj._values() / norm.gather(dim=0, index=indices[0, :])
        return torch.sparse.FloatTensor(
            indices, values, adj.shape).to(adj.device)

    else:
        norm = torch.matmul(adj, torch.ones((adj.shape[0], 1),
                                            device=adj.device))
        return adj / norm

class GraphConv(nn.Module):
    """A simple graph convolution layer, similar to the one defined by *Kipf et al.* in
    `Semi-Supervised Classification with Graph Convolutional Networks`_ ICLR 2017

    This operation with self_layer=False is equivalent to :math:`(A H W)` where:
    - :math:`H` is the node features with shape (batch_size, num_nodes, input_dim)
    - :math:`W` is a weight matrix of shape (input_dim, output_dim)
    - :math:`A` is the adjacency matrix of shape (num_nodes, num_nodes).
    It can include self-loop.

    With normalize_adj=True, it is equivalent to :math:`(D^{-1} A H W)`, where:
    - :math:`D` is a diagonal matrix with :math:`D_{ii}` = the sum of the i-th row of A.
    In other words, :math:`D` is the incoming degree of each node.

    With self_layer=True, it is equivalent to the above plus :math:`(H W_{\\text{self}})`, where:
    - :math:`W_{\\text{self}}` is a separate weight matrix to filter each node's self features.

    Note that when self_layer is True, A should not include self-loop.

    Args:
        input_dim (int): The number of features in each input node.
        output_dim (int): The number of features in each output node.
        bias (bool): Whether to add bias after the node-wise linear layer.

    Example:
        >>> node_feat = torch.rand(1, 3, 5)
        >>> i = torch.LongTensor(
        ...     [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
        >>> v = torch.FloatTensor([1, 1, 1, 1, 1, 1])
        >>> adj = torch.sparse.FloatTensor(i, v, torch.Size([3, 3]))
        >>> model = GraphConv(5, 10)
        >>> output = model(node_feat, adj)
        >>> # pre-normalize adj
        >>> adj = normalize_adj(adj)
        >>> output = model(node_feat, adj, normalize_adj=False)

    .. _Semi-Supervised Classification with Graph Convolutional Networks:
        https://arxiv.org/abs/1609.02907
    """
    def __init__(self, input_dim, output_dim, self_layer=True, bias=True):
        super(GraphConv, self).__init__()

        self.self_layer = self_layer

        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

        if self_layer:
            self.linear_self = nn.Linear(input_dim, output_dim, bias=bias)

        else:
            self.linear_self = None

        self.initialize()

    def initialize(self):
        nn.init.xavier_uniform_(self.linear.weight.data)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-1.0, 1.0)

        if self.self_layer:
            nn.init.xavier_uniform_(self.linear_self.weight.data)
            if self.linear_self.bias is not None:
                self.linear_self.bias.data.uniform_(-1.0, 1.0)

    def forward(self, node_feat, adj, normalize_adj=True):
        """
        Args:
            node_feat (torch.FloatTensor):
                Shape = (batch_size, num_nodes, input_dim)
                The input features of each node.
            adj (torch.sparse.FloatTensor or torch.FloatTensor):
                Shape = (num_nodes, num_nodes)
                The adjacency matrix. adj[i, j] is non-zero if there's an
                incoming edge from j to i. Should not include self-loop if
                self_layer is True.
            normalize_adj (bool):
                Set this to true to apply normalization to adjacency; that is,
                each output feature will be divided by the number of incoming
                neighbors. If normalization is not desired, or if the adjacency
                matrix is pre-normalized, set this to False to improve
                performance.

        Returns:
            (torch.FloatTensor):
                The output features of each node.
                Shape = (batch_size, num_nodes, output_dim)
        """

        if adj.type().endswith('sparse.FloatTensor'):
            if normalize_adj:
                norm = torch.sparse.mm(adj, torch.ones((adj.shape[0], 1),
                                                       device=node_feat.device))
                result = sparse_bmm(adj, self.linear(node_feat)) / norm

            else:
                result = sparse_bmm(adj, self.linear(node_feat))

        else:
            if normalize_adj:
                norm = torch.matmul(adj, torch.ones((adj.shape[0], 1),
                                                    device=node_feat.device))

                result = torch.matmul(adj, self.linear(node_feat)) / norm

            else:
                result = torch.matmul(adj, self.linear(node_feat))

        if self.self_layer:
            result += self.linear_self(node_feat)

        return result
