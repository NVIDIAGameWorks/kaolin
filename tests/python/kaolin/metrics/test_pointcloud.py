# Copyright (c) 2019,20-21 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import torch

from kaolin.metrics import pointcloud as pc
from kaolin.utils.testing import FLOAT_DTYPES, with_seed


@pytest.mark.parametrize('dtype', FLOAT_DTYPES)
@pytest.mark.parametrize('device', ['cuda'])
class TestSidedDistance:
    @pytest.fixture(autouse=True)
    def get_tol(self, device, dtype):
        if dtype == torch.half:
            return 1e-3, 1e-3
        elif dtype == torch.float:
            return 1e-5, 1e-4
        elif dtype == torch.double:
            return 1e-6, 1e-5

    @with_seed(torch_seed=0)
    @pytest.fixture(autouse=True)
    def input_double_p1(self, device, dtype):
        return torch.randn((5, 20, 3), requires_grad=True, device='cuda', dtype=torch.double)

    @with_seed(torch_seed=0)
    @pytest.fixture(autouse=True)
    def input_double_p2(self, device, dtype):
        return torch.randn((5, 15, 3), requires_grad=True, device='cuda', dtype=torch.double)

    @pytest.fixture(autouse=True)
    def get_input(self, device, dtype):
        p1 = torch.tensor([[[8.8977, 4.1709, 1.2839],
                            [8.5640, 7.7767, 9.4214]],
                            [[0.5431, 6.4495, 11.4914],
                            [3.2126, 8.0865, 3.1018]]], dtype=dtype, device=device)

        p2 = torch.tensor([[[6.9340, 6.1152, 3.4435],
                            [0.1032, 9.8181, 11.3350]],
                           [[11.4006, 2.2154, 7.9589],
                            [4.2586, 1.4133, 7.2606]]], dtype=dtype, device=device)

        return p1, p2

    @with_seed(torch_seed=0)
    @pytest.fixture(autouse=True)
    def get_large_input(self, device, dtype):
        N = 100
        B = 3
        M = 50

        p1 = torch.randint(0, 100, (B, N, 3), dtype=dtype, device=device)
        
        p2 = torch.randint(0, 100, (B, M, 3), dtype=dtype, device=device)
        return p1, p2

    @pytest.fixture(autouse=True)
    def target_grad_double(self, input_double_p1, input_double_p2):
        # if test_gradcheck passed the gradient using torch.double inputs is trustable
        outputs = torch.sum(pc.sided_distance(input_double_p1, input_double_p2)[0])
        outputs.backward()
        return input_double_p1.grad.clone(), input_double_p2.grad.clone()

    @pytest.fixture(autouse=True)
    def target_grad_double_2(self, get_input):
        # if test_gradcheck passed the gradient using torch.double inputs is trustable
        p1, p2 = get_input
        p1 = p1.detach()
        p2 = p2.detach()
        p1.requires_grad = True
        p2.requires_grad = True

        outputs = torch.sum(pc.sided_distance(p1, p2)[0])
        outputs.backward()
        return p1.grad.clone(), p2.grad.clone()
    
    @pytest.fixture(autouse=True)
    def target_grad_double_large(self, get_large_input):
        # if test_gradcheck passed the gradient using torch.double inputs is trustable
        p1, p2 = get_large_input
        p1 = p1.detach()
        p2 = p2.detach()
        p1.requires_grad = True
        p2.requires_grad = True

        outputs = torch.sum(pc.sided_distance(p1, p2)[0])
        outputs.backward()
        return p1.grad.clone(), p2.grad.clone()

    def test_sided_distance(self, device, dtype, get_input, get_tol):
        p1, p2 = get_input
        output_p1, output_idx_p1 = pc.sided_distance(p1, p2)
        expected_p1 = torch.tensor([[12.3003, 41.1528], [57.0679, 62.9213]], device=device, dtype=dtype)
        expected_idx_p1 = torch.tensor([[0, 0], [1, 1]], device=device, dtype=torch.long)

        atol, rtol = get_tol
        assert torch.allclose(output_p1, expected_p1, atol=atol, rtol=rtol)
        assert torch.equal(output_idx_p1, expected_idx_p1)

    def test_sided_distance_large_input(self, device, dtype, get_large_input, get_tol):
        p1, p2 = get_large_input
        output_p1, output_idx_p1 = pc.sided_distance(p1, p2)

        expected_p1 = pc._sided_distance(p1, p2)

        atol, rtol = get_tol
        assert torch.allclose(output_p1, expected_p1, atol=atol, rtol=rtol)

    @with_seed(torch_seed=0)
    def test_directed_distance_batch_size(self, device, dtype):

        with pytest.raises(RuntimeError,
                match=r"Expected tensor of size \[3, 3, 3\], but got tensor "
                      r"of size \[2, 3, 3\] for argument #2 'p2' "
                      r"\(while checking arguments for sided_distance_forward_cuda\)"):
            p1 = torch.randint(0, 10, (3, 2, 3), dtype=dtype, device=device)
            p2 = torch.randint(0, 10, (2, 3, 3), dtype=dtype, device=device)
            pc.sided_distance(p1, p2)

    @with_seed(torch_seed=0)
    def test_directed_distance_dims(self, device, dtype):

        with pytest.raises(RuntimeError,
                           match="Expected 3-dimensional tensor, but got "
                                 "4-dimensional tensor for argument #1 'p1' "
                                 r"\(while checking arguments for sided_distance_forward_cuda\)"):
            p1 = torch.randint(0, 10, (3, 2, 3, 4), dtype=dtype, device=device)
            p2 = torch.randint(0, 10, (2, 3, 3), dtype=dtype, device=device)
            pc.sided_distance(p1, p2)

        with pytest.raises(RuntimeError,
                           match=r"Expected tensor of size \[2, 2, 3\], but got "
                                 r"tensor of size \[2, 2, 2\] for argument #1 'p1' "
                                 r"\(while checking arguments for sided_distance_forward_cuda\)"):
            p1 = torch.randint(0, 10, (2, 2, 2), dtype=dtype, device=device)
            p2 = torch.randint(0, 10, (2, 3, 3), dtype=dtype, device=device)
            pc.sided_distance(p1, p2)

    def test_grad_check(self, device, dtype, input_double_p1, input_double_p2):
        if dtype != torch.double:
            pytest.skip("Gradient check only works in double.")

        input_points = (input_double_p1, input_double_p2)

        grad_result = torch.autograd.gradcheck(pc.sided_distance, input_points, eps=1e-6, atol=1e-6)

        assert grad_result

    def test_grad_check_2(self, device, dtype, get_input):
        # Test for gradient accumulation w.r.t p2
        if dtype != torch.double:
            pytest.skip("Gradient check only works in double.")

        p1, p2 = get_input
        p1.requires_grad = True
        p2.requires_grad = True

        grad_result = torch.autograd.gradcheck(pc.sided_distance, (p1, p2), eps=1e-6, atol=1e-6)

        assert grad_result

    def test_grad_check_large(self, device, dtype, get_large_input):
        # Test for gradient accumulation w.r.t p2
        if dtype != torch.double:
            pytest.skip("Gradient check only works in double.")

        p1, p2 = get_large_input
        p1.requires_grad = True
        p2.requires_grad = True

        grad_result = torch.autograd.gradcheck(pc.sided_distance, (p1, p2), eps=1e-6, atol=1e-6)

        assert grad_result

    def test_grad_check_other_type(self, device, dtype, input_double_p1, input_double_p2, target_grad_double):
        if dtype == torch.double:
            pytest.skip("Gradient check for double already tested.")
        
        p1 = input_double_p1.to(dtype).detach()
        p2 = input_double_p2.to(dtype).detach()
        p1.requires_grad = True
        p2.requires_grad = True

        output = pc.sided_distance(p1, p2)[0]
        torch.sum(output).backward()
        target_grad_p1, target_grad_p2 = target_grad_double
        target_grad_p1 = target_grad_p1.to(dtype)
        target_grad_p2 = target_grad_p2.to(dtype)

        assert torch.allclose(p1.grad, target_grad_p1, rtol=1e-2, atol=1e-2)
        assert torch.allclose(p2.grad, target_grad_p2, rtol=1e-2, atol=1e-2)

    def test_grad_check_other_type_2(self, device, dtype, get_input, target_grad_double_2):
        if dtype == torch.double:
            pytest.skip("Gradient check for double already tested.")
        
        p1, p2 = get_input
        p1.requires_grad = True
        p2.requires_grad = True

        output = pc.sided_distance(p1, p2)[0]
        torch.sum(output).backward()
        target_grad_p1, target_grad_p2 = target_grad_double_2
        target_grad_p1 = target_grad_p1.to(dtype)
        target_grad_p2 = target_grad_p2.to(dtype)

        assert torch.allclose(p1.grad, target_grad_p1, rtol=1e-2, atol=1e-2)
        assert torch.allclose(p2.grad, target_grad_p2, rtol=1e-2, atol=1e-2)

    def test_grad_check_other_type_large(self, device, dtype, get_large_input, target_grad_double_large):
        if dtype == torch.double:
            pytest.skip("Gradient check for double already tested.")
        
        p1, p2 = get_large_input
        p1.requires_grad = True
        p2.requires_grad = True

        output = pc.sided_distance(p1, p2)[0]
        torch.sum(output).backward()
        target_grad_p1, target_grad_p2 = target_grad_double_large
        target_grad_p1 = target_grad_p1.to(dtype)
        target_grad_p2 = target_grad_p2.to(dtype)

        assert torch.allclose(p1.grad, target_grad_p1, rtol=1e-2, atol=1e-2)
        assert torch.allclose(p2.grad, target_grad_p2, rtol=1e-2, atol=1e-2)
    
    
@pytest.mark.parametrize('dtype', FLOAT_DTYPES)
@pytest.mark.parametrize('device', ['cuda'])
class TestChamferDistance:
    @pytest.fixture(autouse=True)
    def tolerances(self, device, dtype):
        if dtype == torch.half:
            return 1e-3, 1e-3
        elif dtype == torch.float:
            return 1e-5, 1e-4
        elif dtype == torch.double:
            return 1e-6, 1e-5

    @pytest.fixture(autouse=True)
    def p1(self, device, dtype):
        return torch.tensor([[[8.8977, 4.1709, 1.2839],
                              [8.5640, 7.7767, 9.4214]],
                             [[0.5431, 6.4495, 11.4914],
                              [3.2126, 8.0865, 3.1018]]],
                            dtype=dtype, device=device)

    @pytest.fixture(autouse=True)
    def p2(self, device, dtype):
        return torch.tensor([[[6.9340, 6.1152, 3.4435],
                              [0.1032, 9.8181, 11.3350]],
                             [[11.4006, 2.2154, 7.9589],
                              [4.2586, 1.4133, 7.2606]]],
                            dtype=dtype, device=device)

    def test_chamfer_distance(self, device, dtype, p1, p2, tolerances):
        output = pc.chamfer_distance(p1, p2)

        expected = torch.tensor([72.5838, 151.0809], dtype=dtype, device=device)

        atol, rtol = tolerances
        assert torch.allclose(output, expected, atol=atol, rtol=rtol)

    def test_weighted_chamfer_distance(self, device, dtype, p1, p2, tolerances):
        output = pc.chamfer_distance(p1, p2, w1=1.3, w2=0.8)
        expected = torch.tensor([71.4303, 150.8620], dtype=dtype, device=device)

        atol, rtol = tolerances
        assert torch.allclose(output, expected, atol=atol, rtol=rtol)

    def test_chamfer_distance_not_squared(self, device, dtype, p1, p2, tolerances):
        output = pc.chamfer_distance(p1, p2, squared=False)
        expected = torch.tensor([11.1704, 17.1130], dtype=dtype, device=device)
        
        atol, rtol = tolerances
        assert torch.allclose(output, expected, atol=atol, rtol=rtol)

@pytest.mark.parametrize('dtype', FLOAT_DTYPES)
@pytest.mark.parametrize('device', ['cuda'])
class TestFScore:
    @pytest.fixture(autouse=True)
    def get_tol(self, device, dtype):
        if dtype == torch.half:
            return 1e-3, 1e-3
        elif dtype == torch.float:
            return 1e-5, 1e-4
        elif dtype == torch.double:
            return 1e-6, 1e-5

    def test_FScore(self, device, dtype, get_tol):

        gt_points = torch.tensor([[[8.8977, 4.1709, 1.2839],
                                   [8.5640, 7.7767, 9.4214]],
                                  [[0.5431, 6.4495, 11.4914],
                                   [3.2126, 8.0865, 3.1018]]], dtype=dtype, device=device)

        pred_points = torch.tensor([[[8.8914, 4.1788, 1.2176],
                                     [8.5291, 7.5513, 9.5412]],
                                    [[0.4010, 6.4602, 11.5183],
                                     [3.2977, 8.0325, 3.1180]]], dtype=dtype, device=device)
        output1 = pc.f_score(gt_points, pred_points, radius=0.2)
        output2 = pc.f_score(gt_points, pred_points, radius=0.12)

        expected1 = torch.tensor([0.5, 1], device=device, dtype=dtype)
        expected2 = torch.tensor([0.5, 0.5], device=device, dtype=dtype)

        atol, rtol = get_tol
        assert torch.allclose(output1, expected1, atol=atol, rtol=rtol)
        assert torch.allclose(output2, expected2, atol=atol, rtol=rtol)

    def test_FScore_heterogeneous(self, device, dtype, get_tol):
        gt_points = torch.tensor([[[8.8977, 4.1709, 1.2839],
                                   [8.5640, 7.7767, 9.4214]],
                                  [[0.5431, 6.4495, 11.4914],
                                   [3.2126, 8.0865, 3.1018]]], dtype=dtype, device=device)

        pred_points = torch.tensor([[[8.8914, 4.1788, 1.2176],
                                     [8.5291, 7.5513, 9.5412],
                                     [3.7831, 6.0182, 4.1208]],
                                    [[0.4010, 6.4602, 11.5183],
                                     [3.2977, 8.0325, 3.1180],
                                     [2.4987, 5.8763, 3.1987]]], dtype=dtype, device=device)
        output1 = pc.f_score(gt_points, pred_points, radius=0.2)
        output2 = pc.f_score(gt_points, pred_points, radius=0.12)

        expected1 = torch.tensor([0.4, 0.8], device=device, dtype=dtype)
        expected2 = torch.tensor([0.4, 0.4], device=device, dtype=dtype)

        atol, rtol = get_tol
        assert torch.allclose(output1, expected1, atol=atol, rtol=rtol)
        assert torch.allclose(output2, expected2, atol=atol, rtol=rtol)

