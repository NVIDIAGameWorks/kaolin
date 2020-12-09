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

from kaolin.ops import batch, reduction
from kaolin.ops.random import random_shape_per_tensor, random_tensor

TEST_TYPES = [('cuda', dtype) for dtype in [torch.half, torch.float, torch.double, torch.bool, torch.int, torch.long]] + \
             [('cpu', dtype) for dtype in [torch.float, torch.double, torch.bool, torch.int, torch.long]]


# Same naive implementation than packed_simple_sum for cpu input
# as it's pretty straightforward
def _torch_packed_simple_sum(inputs, numel_per_tensor):
    outputs = []
    last_id = 0
    for i, numel in enumerate(numel_per_tensor):
        first_id = last_id
        last_id += int(numel)
        outputs.append(torch.sum(inputs[first_id:last_id]))
    return torch.stack(outputs, dim=0)


@pytest.mark.parametrize("numel_per_tensor",
                         [torch.LongTensor([1]),
                          torch.LongTensor([1, 100000]),
                          torch.arange(257, dtype=torch.long)])
class Test_PackedSimpleSumCuda:
    @pytest.fixture(autouse=True)
    def total_numel(self, numel_per_tensor):
        return torch.sum(numel_per_tensor)

    @pytest.fixture(autouse=True)
    def inputs_double(self, total_numel):
        return torch.rand((total_numel, 1), dtype=torch.double, device='cuda',
                          requires_grad=True)

    @pytest.fixture(autouse=True)
    def target_output_double(self, inputs_double, numel_per_tensor):
        return _torch_packed_simple_sum(inputs_double, numel_per_tensor)

    @pytest.fixture(autouse=True)
    def target_grad_double(self, inputs_double, numel_per_tensor):
        # if test_gradcheck passed the gradient using torch.double inputs is trustable
        outputs = torch.sum(reduction._PackedSimpleSumCuda.apply(inputs_double,
                                                                numel_per_tensor))
        outputs.backward()
        return inputs_double.grad.clone()

    @pytest.fixture(autouse=True)
    def inputs_long(self, total_numel):
        return torch.randint(0, 33, size=(total_numel, 1), dtype=torch.long, device='cuda')

    @pytest.fixture(autouse=True)
    def target_output_long(self, inputs_long, numel_per_tensor):
        return _torch_packed_simple_sum(inputs_long, numel_per_tensor)

    def test_gradcheck(self, numel_per_tensor, total_numel):
        # gradcheck only for double
        inputs = torch.rand((total_numel, 1), dtype=torch.double, device='cuda',
                            requires_grad=True)
        torch.autograd.gradcheck(reduction._PackedSimpleSumCuda.apply,
                                 (inputs, numel_per_tensor))

    @pytest.mark.parametrize("dtype", [torch.double, torch.float, torch.half])
    def test_float_types(self, inputs_double, numel_per_tensor, dtype,
                         target_output_double, target_grad_double):
        inputs = inputs_double.type(dtype).detach()
        inputs.requires_grad = True
        output = reduction._PackedSimpleSumCuda.apply(inputs, numel_per_tensor)
        target_output = target_output_double.to(dtype)
        assert torch.allclose(output, target_output, rtol=1e-3, atol=1e-4)
        torch.sum(output).backward()
        target_grad = target_grad_double.to(dtype)
        assert torch.allclose(inputs.grad, target_grad, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.long, torch.int])
    def test_int_types(self, inputs_long, numel_per_tensor, dtype, target_output_long):
        inputs = inputs_long.type(dtype)
        output = reduction._PackedSimpleSumCuda.apply(inputs, numel_per_tensor)
        target_output = target_output_long
        assert torch.equal(output, target_output)

    def test_bool_type(self, total_numel, numel_per_tensor):
        inputs = torch.randint(0, 2, size=(total_numel, 1), dtype=torch.bool, device='cuda')
        target_outputs = _torch_packed_simple_sum(inputs, numel_per_tensor)
        outputs = reduction._PackedSimpleSumCuda.apply(inputs, numel_per_tensor)
        torch.equal(outputs, target_outputs)

    def test_cpu_fail(self, inputs_double, numel_per_tensor):
        inputs = inputs_double.cpu()
        with pytest.raises(RuntimeError,
                           match="packed_tensor must be a CUDA tensor"):
            reduction._PackedSimpleSumCuda.apply(inputs, numel_per_tensor)


@pytest.mark.parametrize("device,dtype", TEST_TYPES)
@pytest.mark.parametrize("numel_per_tensor",
                         [torch.LongTensor([1]),
                          torch.LongTensor([1, 100000]),
                          torch.arange(257, dtype=torch.long)])
class TestPackedSimpleSum:
    @pytest.fixture(autouse=True)
    def total_numel(self, numel_per_tensor):
        return torch.sum(numel_per_tensor)

    @pytest.fixture(autouse=True)
    def high_val(self, dtype):
        if dtype.is_floating_point or dtype == torch.bool:
            return 1
        else:
            return 32

    @pytest.fixture(autouse=True)
    def inputs(self, high_val, total_numel, dtype, device):
        return random_tensor(0, high_val, shape=(total_numel, 1), dtype=dtype,
                             device=device)

    def test_packed_simple_sum(self, inputs, numel_per_tensor, device, dtype):
        sum_tensor = reduction.packed_simple_sum(inputs, numel_per_tensor)
        target = _torch_packed_simple_sum(inputs, numel_per_tensor)
        assert torch.allclose(sum_tensor, target)
