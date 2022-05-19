# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
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

import kaolin.render.spc as spc_render

class TestRaytrace:
    @pytest.fixture(autouse=True)
    def feats(self):
        feats = torch.tensor([
            [1,1],[1,1],[1,1],[2,2],[3,3],[5,5]
            ],
            device='cuda', dtype=torch.float)
        return feats
    
    @pytest.fixture(autouse=True)
    def feats_big(self):
        feats = torch.rand([10000, 100, 32], device='cuda', dtype=torch.float)
        return feats

    @pytest.fixture(autouse=True)
    def boundaries_big(self):
        boundary = torch.zeros([10000, 100], device='cuda', dtype=torch.bool)
        boundary[:, 0] = True
        return boundary.reshape(-1)
    
    @pytest.fixture(autouse=True)
    def tau(self):
        feats = torch.tensor([
            [0],[0],[0],[1],[0],[1]
            ],
            device='cuda', dtype=torch.float)
        return feats

    @pytest.fixture(autouse=True)
    def boundaries(self):
        boundary = torch.tensor([1,0,1,0,0,1], device='cuda', dtype=torch.bool)
        return boundary

    def test_mark_pack_boundaries(self):
        ridx = torch.tensor([1,1,1,1,2,2,3,3,3], device='cuda', dtype=torch.int)
        
        expected_boundary = torch.tensor([1,0,0,0,1,0,1,0,0], device='cuda', dtype=torch.bool)

        output = spc_render.mark_pack_boundaries(ridx)

        assert torch.equal(output, expected_boundary)

    def test_diff(self, feats, boundaries):
        diff = spc_render.diff(feats, boundaries)
        expected = torch.tensor([[0,0], [0,0], [1,1], [1,1], [0,0], [0,0]], device='cuda', dtype=torch.float)
        assert torch.equal(diff, expected)

    def test_sum_reduce(self, feats, boundaries):
        sum_reduce = spc_render.sum_reduce(feats, boundaries)
        expected = torch.tensor([[2,2], [6,6], [5,5]], device='cuda', dtype=torch.float)
        assert torch.equal(sum_reduce, expected)

    def test_sum_reduce_big(self, feats_big, boundaries_big):
        fdim = feats_big.shape[-1]
        sum_reduce = spc_render.sum_reduce(feats_big.reshape(-1, fdim), boundaries_big)
        expected = feats_big.sum(1)
        assert torch.allclose(sum_reduce, expected, atol=1e-5)
    
    def test_sum_reduce_big_backward(self, feats_big, boundaries_big):

        feats_big.requires_grad = True
        fdim = feats_big.shape[-1]

        if feats_big.grad is not None:
            feats_big.grad.detach_()
            feats_big.grad.zero_()
        sum_reduce = spc_render.sum_reduce(feats_big.reshape(-1, fdim), boundaries_big)
        loss = sum_reduce.sum()
        loss.backward()
        grad0 = feats_big.grad.clone()

        if feats_big.grad is not None:
            feats_big.grad.detach_()
            feats_big.grad.zero_()
        expected = feats_big.sum(1)
        loss = expected.sum()
        loss.backward()
        grad1 = feats_big.grad.clone()

        assert torch.allclose(grad0, grad1, atol=1e-5)

    def test_cumsum(self, feats, boundaries):
        cumsum = spc_render.cumsum(feats, boundaries)
        expected = torch.tensor([[1,1], [2,2], [1,1], [3,3], [6,6], [5,5]], device='cuda', dtype=torch.float)
        assert torch.equal(cumsum, expected)
    
    def test_cumsum_big(self, feats_big, boundaries_big):
        fdim = feats_big.shape[-1]
        cumsum = spc_render.cumsum(feats_big.reshape(-1, fdim), boundaries_big)
        expected = torch.cumsum(feats_big, dim=1).reshape(-1, fdim)
        assert torch.allclose(cumsum, expected, atol=1e-5)

    def test_cumsum_big_backward(self, feats_big, boundaries_big):

        feats_big.requires_grad = True
        fdim = feats_big.shape[-1]

        if feats_big.grad is not None:
            feats_big.grad.detach_()
            feats_big.grad.zero_()
        cumsum = spc_render.cumsum(feats_big.reshape(-1, fdim), boundaries_big)
        loss = cumsum.sum()
        loss.backward()
        grad0 = feats_big.grad.clone()

        if feats_big.grad is not None:
            feats_big.grad.detach_()
            feats_big.grad.zero_()
        expected = torch.cumsum(feats_big, dim=1)
        loss = expected.sum()
        loss.backward()
        grad1 = feats_big.grad.clone()

        assert torch.allclose(grad0, grad1, atol=1e-4)

    def test_cumsum_reverse(self, feats, boundaries):
        cumsum = spc_render.cumsum(feats, boundaries, reverse=True)
        expected = torch.tensor([[2,2], [1,1], [6,6], [5,5], [3,3], [5,5]], device='cuda', dtype=torch.float)
        assert torch.equal(cumsum, expected)
    
    def test_cumsum_exclusive(self, feats, boundaries):
        cumsum = spc_render.cumsum(feats, boundaries, reverse=False, exclusive=True)
        expected = torch.tensor([[0,0], [1,1], [0,0], [1,1], [3,3], [0,0]], device='cuda', dtype=torch.float)
        assert torch.equal(cumsum, expected)
    
    def test_cumsum_exclusive_reverse(self, feats, boundaries):
        cumsum = spc_render.cumsum(feats, boundaries, reverse=True, exclusive=True)
        expected = torch.tensor([[1,1], [0,0], [5,5], [3,3], [0,0], [0,0]], device='cuda', dtype=torch.float)
        assert torch.equal(cumsum, expected)
       
    def test_cumprod(self, feats, boundaries):
        cumprod = spc_render.cumprod(feats, boundaries)
        expected = torch.tensor([[1,1], [1,1], [1,1], [2,2], [6,6], [5,5]], device='cuda', dtype=torch.float)
        assert torch.equal(cumprod, expected)
    
    def test_cumprod_big(self, feats_big, boundaries_big):
        fdim = feats_big.shape[-1]
        cumprod = spc_render.cumprod(feats_big.reshape(-1, fdim), boundaries_big)
        expected = torch.cumprod(feats_big, dim=1).reshape(-1, fdim)
        assert torch.allclose(cumprod, expected, atol=1e-4)
    
    def test_cumprod_big_backward(self, feats_big, boundaries_big):

        feats_big += 1e-3
        feats_big.requires_grad = True
        fdim = feats_big.shape[-1]

        if feats_big.grad is not None:
            feats_big.grad.detach_()
            feats_big.grad.zero_()
        cumprod = spc_render.cumprod(feats_big.reshape(-1, fdim), boundaries_big)
        loss = cumprod.sum()
        loss.backward()
        grad0 = feats_big.grad.clone()

        if feats_big.grad is not None:
            feats_big.grad.detach_()
            feats_big.grad.zero_()
        expected = torch.cumprod(feats_big, dim=1)
        loss = expected.sum()
        loss.backward()
        grad1 = feats_big.grad.clone()
    
        assert torch.allclose(grad0, grad1, atol=1e-2)

    def test_cumprod_reverse(self, feats, boundaries):
        cumprod = spc_render.cumprod(feats, boundaries, reverse=True)
        expected = torch.tensor([[1,1], [1,1], [6,6], [6,6], [3,3], [5,5]], device='cuda', dtype=torch.float)
        assert torch.equal(cumprod, expected)
    
    def test_cumprod_exclusive(self, feats, boundaries):
        cumprod = spc_render.cumprod(feats, boundaries, reverse=False, exclusive=True)
        expected = torch.tensor([[1,1], [1,1], [1,1], [1,1], [2,2], [1,1]], device='cuda', dtype=torch.float)
        assert torch.equal(cumprod, expected)
    
    def test_cumprod_exclusive_reverse(self, feats, boundaries):
        cumprod = spc_render.cumprod(feats, boundaries, reverse=True, exclusive=True)
        expected = torch.tensor([[1,1], [1,1], [6,6], [3,3], [1,1], [1,1]], device='cuda', dtype=torch.float)
        assert torch.equal(cumprod, expected)
       
    def test_exponential_integration(self, feats, tau, boundaries):
        integrated_feats, transmittance = spc_render.exponential_integration(feats, tau, boundaries, exclusive=False)
        expected_feats = torch.tensor([[0,0], [0.4651,0.4651], [1.1627, 1.1627]], device='cuda', dtype=torch.float)
        expected_transmittance = torch.tensor([[0.0],[0.0],[0.0],[0.2325],[0.0],[0.2325]], device='cuda', dtype=torch.float)
        assert torch.allclose(integrated_feats, expected_feats, atol=1e-4)
        assert torch.allclose(transmittance, expected_transmittance, atol=1e-4)


