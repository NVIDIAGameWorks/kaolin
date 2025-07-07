# Copyright (c) 2021,2025 NVIDIA CORPORATION & AFFILIATES.
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
from functools import partial
import torch

import kaolin.render.spc as spc_render
from kaolin.utils.testing import check_allclose

def _naive_diff(t):
    output = torch.zeros_like(t)
    output[:-1] = torch.diff(t, dim=0)
    return output

def _cumsum(t, reverse, exclusive):
    if reverse:
        inp = torch.flip(t, dims=(0,))
    else:
        inp = t
    output = torch.cumsum(inp, dim=0)
    if exclusive:
        output = torch.roll(output, shifts=1, dims=(0,))
        output[0] = 0.
    if reverse:
        return torch.flip(output, dims=(0,))
    else:
        return output

def _cumprod(t, reverse, exclusive):
    if reverse:
        inp = torch.flip(t, dims=(0,))
    else:
        inp = t
    output = torch.cumprod(inp, dim=0)
    if exclusive:
        output = torch.roll(output, shifts=1, dims=(0,))
        output[0] = 1.
    if reverse:
        return torch.flip(output, dims=(0,))
    else:
        return output
    
def _naive_reduce(feats, boundaries, func):
    indices = torch.cumsum(boundaries, dim=0) - 1
    num_outputs = int(indices[-1]) + 1
    output = torch.zeros((num_outputs, feats.shape[-1]),
                         device=feats.device, dtype=feats.dtype)
    for i in range(num_outputs):
        output[i] = func(feats[indices == i])
    return output

def _naive_process(feats, boundaries, func):
    indices = torch.cumsum(boundaries, dim=0) - 1
    num_steps = int(indices[-1]) + 1
    output = torch.zeros_like(feats)
    for i in range(num_steps):
        mask = indices == i
        output[mask] = func(feats[mask])
    return output

class TestRaytrace:
    @pytest.fixture(autouse=True)
    def feats(self):
        return torch.rand(1000, 2, device='cuda', dtype=torch.float)

    @pytest.fixture(autouse=True)
    def boundaries(self):
        boundaries = torch.rand(1000, device='cuda', dtype=torch.float) < 0.2
        boundaries[0] = True
        return boundaries

    @pytest.fixture(autouse=True)
    def tau(self):
        return torch.rand(1000, 1, device='cuda', dtype=torch.float)

    def test_mark_pack_boundaries(self):
        ridx = torch.tensor([1,1,1,1,2,2,3,3,3], device='cuda', dtype=torch.int)
        
        expected_boundary = torch.tensor([1,0,0,0,1,0,1,0,0], device='cuda', dtype=torch.bool)

        output = spc_render.mark_pack_boundaries(ridx)

        assert torch.equal(output, expected_boundary)

    def test_diff(self, feats, boundaries):
        diff = spc_render.diff(feats, boundaries)
        expected = _naive_process(feats, boundaries, _naive_diff)
        assert torch.equal(diff, expected)
    
    def test_sum_reduce(self, feats, boundaries):
        sum_reduce = spc_render.sum_reduce(feats, boundaries)
        expected = _naive_reduce(feats, boundaries, partial(torch.sum, dim=0))
        check_allclose(sum_reduce, expected)

    def test_sum_reduce_backward(self, feats, boundaries):
        feats1 = feats.detach()
        feats1.requires_grad = True
        feats2 = feats.detach()
        feats2.requires_grad = True
        sum_reduce = spc_render.sum_reduce(feats1, boundaries)
        expected = _naive_reduce(feats2, boundaries, partial(torch.sum, dim=0))
        grad_out = torch.rand_like(expected)
        sum_reduce.backward(grad_out)
        expected.backward(grad_out)
        check_allclose(feats1.grad, feats2.grad, rtol=1e-4, atol=1e-4)

    def test_prod_reduce(self, feats, boundaries):
        prod_reduce = spc_render.prod_reduce(feats, boundaries)
        expected = _naive_reduce(feats, boundaries, partial(torch.prod, dim=0))
        check_allclose(prod_reduce, expected)

    @pytest.mark.parametrize('reverse', [False, True])
    @pytest.mark.parametrize('exclusive', [False, True])
    def test_cumsum(self, feats, boundaries, reverse, exclusive):
        cumsum = spc_render.cumsum(feats, boundaries, reverse=reverse, exclusive=exclusive)
        expected = _naive_process(feats, boundaries, partial(_cumsum, reverse=reverse, exclusive=exclusive))
        assert torch.equal(cumsum, expected)

    @pytest.mark.parametrize('reverse', [False, True])
    @pytest.mark.parametrize('exclusive', [False, True])
    def test_cumsum_backward(self, feats, boundaries, reverse, exclusive):
        feats1 = feats.detach()
        feats1.requires_grad = True
        feats2 = feats.detach()
        feats2.requires_grad = True
        cumsum = spc_render.cumsum(feats1, boundaries, reverse=reverse, exclusive=exclusive)
        expected = _naive_process(feats2, boundaries, partial(_cumsum, reverse=reverse, exclusive=exclusive))
        grad_out = torch.rand_like(expected)
        cumsum.backward(grad_out)
        expected.backward(grad_out)
        check_allclose(feats1.grad, feats2.grad, rtol=1e-4, atol=1e-4)
 
    @pytest.mark.parametrize('reverse', [False, True])
    @pytest.mark.parametrize('exclusive', [False, True])   
    def test_cumprod(self, feats, boundaries, reverse, exclusive):
        cumprod = spc_render.cumprod(feats, boundaries, reverse=reverse, exclusive=exclusive)
        expected = _naive_process(feats, boundaries, partial(_cumprod, reverse=reverse, exclusive=exclusive))
        check_allclose(cumprod, expected)

    @pytest.mark.parametrize('reverse', [False, True])
    @pytest.mark.parametrize('exclusive', [False, True])
    def test_cumprod_backward(self, feats, boundaries, reverse, exclusive):
        feats1 = feats.detach()
        feats1.requires_grad = True
        feats2 = feats.detach()
        feats2.requires_grad = True
        cumprod = spc_render.cumprod(feats1, boundaries, reverse=reverse, exclusive=exclusive)
        expected = _naive_process(feats2, boundaries, partial(_cumprod, reverse=reverse, exclusive=exclusive))
        grad_out = torch.ones_like(expected)
        cumprod.backward(grad_out)
        expected.backward(grad_out)
        check_allclose(feats1.grad, feats2.grad, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize('exclusive', [False, True])
    def test_exponential_integration(self, feats, tau, boundaries, exclusive):
        integrated_feats, transmittance = spc_render.exponential_integration(feats, tau, boundaries, exclusive=exclusive)
        alpha = 1.0 - torch.exp(-tau.contiguous())
        expected_transmittance = torch.exp(-1.0 * _naive_process(tau, boundaries, partial(_cumsum, reverse=False, exclusive=exclusive)))
        expected_transmittance = expected_transmittance * alpha
        check_allclose(transmittance, expected_transmittance)
        expected_integrated_feats = _naive_reduce(expected_transmittance * feats, boundaries, partial(torch.sum, dim=0))
        check_allclose(integrated_feats, expected_integrated_feats)
