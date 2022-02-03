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

import pytest
import torch
import os

from kaolin.ops.gcn import sparse_bmm, normalize_adj, GraphConv
from kaolin.utils.testing import ALL_DEVICES

os.environ['NVIDIA_TF32_OVERRIDE'] = '0'

@pytest.mark.parametrize('device', ALL_DEVICES)
def test_sparse_bmm(device):
    i = torch.LongTensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
    v = torch.FloatTensor([1, 2, 3, 4, 5, 6])
    sparse = torch.sparse.FloatTensor(i, v, torch.Size([4, 3])).to(device)
    dense = torch.tensor(
        [[[0.47605860, 0.97254932, 0.93103176],
          [0.56519330, 0.03351519, 0.02914280],
          [0.16332115, 0.24698994, 0.17907326]],

         [[0.57908791, 0.72093546, 0.19004048],
          [0.51033562, 0.15572953, 0.24628967],
          [0.41850159, 0.87904519, 0.06477704]],

         [[0.42210183, 0.37572026, 0.62902039],
          [0.03129875, 0.26592126, 0.95092678],
          [0.87077409, 0.28091857, 0.12425283]]],
        device=device)
    result = sparse_bmm(sparse, dense)
    expected = torch.tensor(
        [[[1.54512024, 1.51545477, 1.10358238],
          [1.44208074, 2.68606853, 2.39928341],
          [4.64106607, 4.99680758, 4.77173042],
          [0.0, 0.0, 0.0]],

         [[3.02134514, 5.43000031, 0.63495189],
          [2.41368055, 4.07900620, 0.57441211],
          [4.93678188, 4.22759533, 1.93536115],
          [0.0, 0.0, 0.0]],

         [[5.25594330, 1.95143259, 1.69644380],
          [3.45652604, 1.59419620, 1.63079929],
          [2.23570418, 2.94228649, 6.94880915],
          [0.0, 0.0, 0.0]]],
        device=device)
    assert torch.allclose(result, expected)


@pytest.mark.parametrize('device', ALL_DEVICES)
class TestNormalizeAdj(object):

    @pytest.fixture(autouse=True)
    def adj(self, device):
        i = torch.LongTensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
        v = torch.FloatTensor([1, 2, 3, 4, 5, 6])
        return torch.sparse.FloatTensor(i, v, torch.Size([3, 3])).to(device)

    def test_normalize_adj_sparse(self, device, adj):
        result = normalize_adj(adj)

        norm = torch.sparse.mm(adj, torch.ones((adj.shape[0], 1),
                                               device=device))
        expected = torch.sparse.mm(adj, torch.eye(3, device=device)) / norm

        assert torch.allclose(
            torch.sparse.mm(result, torch.eye(3, device=device)),
            expected)

    def test_normalize_adj_dense(self, device, adj):
        dense_adj = torch.sparse.mm(adj, torch.eye(3, device=device))
        result = normalize_adj(dense_adj)

        expected = torch.tensor(
            [[0.0, 0.14285714, 0.85714285],
             [0.4, 0.0, 0.6],
             [0.55555555, 0.44444444, 0.0]],
            device=device)

        assert torch.allclose(result, expected)


@pytest.mark.parametrize('device', ALL_DEVICES)
@pytest.mark.parametrize('self_layer', [True, False])
class TestGraphConv(object):

    @pytest.fixture(autouse=True)
    def gcn(self, device, self_layer):
        model = GraphConv(3, 5, self_layer=self_layer)
        model.to(device)
        model.linear.weight.data.copy_(torch.tensor(
            [[-0.61831456, 0.57409757, -0.14574467],
             [0.00189979, 0.77582508, 0.36306566],
             [-0.27461752, -0.69267106, 0.61524123],
             [-0.46579394, -0.00121037, 0.72196031],
             [0.54187351, -0.42773548, 0.59835148]],
            device=device))
        model.linear.bias.data.copy_(torch.tensor(
            [0.40155911, -0.45286083, -0.19249618, 0.21454012, -0.17628896],
            device=device))

        if self_layer:
            model.linear_self.weight.data.copy_(torch.tensor(
                [[0.81866288, 0.24061465, 0.55818230],
                 [0.37344468, 0.07631248, 0.34876764],
                 [0.51045960, 0.73214161, 0.15645593],
                 [0.01274079, 0.44412971, 0.59611768],
                 [0.31227762, 0.13015020, 0.77652276]],
                device=device))
            model.linear_self.bias.data.copy_(torch.tensor(
                [0.54663211, 0.38193095, 0.71667391, 0.14995629, 0.27089202],
                device=device))

        return model

    @pytest.fixture(autouse=True)
    def adj(self, device):
        i = torch.LongTensor(
            [[0, 1, 1, 2, 2, 0, 0, 1, 2], [1, 0, 2, 1, 0, 2, 0, 1, 2]])
        v = torch.FloatTensor([1, 1, 1, 1, 1, 1, 1, 1, 1])
        return torch.sparse.FloatTensor(i, v, torch.Size([3, 3])).to(device)

    @pytest.fixture(autouse=True)
    def node_feat_in(self, device):
        return torch.tensor(
            [[[0.17502755, 0.01767362, 0.43572336],
              [0.84568930, 0.50088108, 0.65273631],
              [0.18389270, 0.30413085, 0.71014285]]],
            device=device)

    @pytest.fixture(autouse=True)
    def expected(self, device, self_layer):
        result = torch.tensor(
            [[[0.22333825, -0.02167436, -0.12385714, 0.46001482, 0.28272793],
              [0.22333825, -0.02167436, -0.12385714, 0.46001482, 0.28272793],
              [0.22333825, -0.02167436, -0.12385714, 0.46001482, 0.28272793]]],
            device=device)

        if self_layer:
            result += torch.tensor(
                [[0.93738627, 0.60060900, 0.88712949, 0.41977805, 0.66619855],
                 [1.72383165, 0.96362591, 1.61720443, 0.77229488, 1.10703623],
                 [1.16674566, 0.72148854, 1.14431667, 0.71070147, 0.91934240]],
                device=device)

        return result

    def test_gcn_sparse(self, device, gcn, adj, node_feat_in, expected):
        node_feat_out = gcn(node_feat_in, adj, normalize_adj=True)
        assert torch.allclose(node_feat_out, expected, rtol=1e-3, atol=1e-3)
        adj = normalize_adj(adj)
        node_feat_out_2 = gcn(node_feat_in, adj, normalize_adj=False)
        assert torch.allclose(node_feat_out, node_feat_out_2, rtol=1e-4, atol=1e-4)

    def test_gcn_dense(self, device, gcn, adj, node_feat_in, expected):
        dense_adj = torch.sparse.mm(adj, torch.eye(3, device=device))
        node_feat_out = gcn(node_feat_in, dense_adj, normalize_adj=True)
        assert torch.allclose(node_feat_out, expected, rtol=1e-3, atol=1e-3)
        dense_adj = normalize_adj(dense_adj)
        node_feat_out_2 = gcn(node_feat_in, dense_adj, normalize_adj=False)
        assert torch.allclose(node_feat_out, node_feat_out_2, rtol=1e-4, atol=1e-4)
