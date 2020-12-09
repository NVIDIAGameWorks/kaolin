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

from kaolin.ops import batch
from kaolin.ops.random import random_shape_per_tensor, random_tensor
from kaolin.utils.testing import FLOAT_DTYPES, INT_DTYPES, ALL_TYPES, NUM_TYPES, \
    check_packed_tensor, check_padded_tensor


# Same naive implementation than packed_simple_sum for cpu input
# as it's pretty straightforward
def _torch_tile_to_packed(values, numel_per_tensor):
    return torch.cat(
        [torch.full((int(numel),), fill_value=value.item(), dtype=values.dtype,
                    device=values.device)
         for value, numel in zip(values, numel_per_tensor)], dim=0).unsqueeze(
        -1)


@pytest.mark.parametrize("numel_per_tensor",
                         [torch.LongTensor([1]),
                          torch.LongTensor([1, 100000]),
                          torch.arange(257, dtype=torch.long)])
class Test_TileToPackedCuda:
    @pytest.fixture(autouse=True)
    def total_numel(self, numel_per_tensor):
        return torch.sum(numel_per_tensor)

    @pytest.fixture(autouse=True)
    def inputs_double(self, numel_per_tensor):
        return torch.rand((numel_per_tensor.shape[0]), dtype=torch.double,
                          device='cuda',
                          requires_grad=True)

    @pytest.fixture(autouse=True)
    def target_output_double(self, inputs_double, numel_per_tensor):
        return _torch_tile_to_packed(inputs_double, numel_per_tensor)

    @pytest.fixture(autouse=True)
    def target_grad_double(self, inputs_double, numel_per_tensor, total_numel):
        # if test_gradcheck passed the gradient using torch.double inputs is trustable
        outputs = torch.sum(
            batch._TileToPackedCuda.apply(inputs_double, numel_per_tensor,
                                         total_numel))
        outputs.backward()
        return inputs_double.grad.clone()

    @pytest.fixture(autouse=True)
    def inputs_long(self, numel_per_tensor):
        return torch.randint(0, 32, size=(numel_per_tensor.shape[0],),
                             dtype=torch.long, device='cuda')

    @pytest.fixture(autouse=True)
    def target_output_long(self, inputs_long, numel_per_tensor):
        return _torch_tile_to_packed(inputs_long, numel_per_tensor)

    def test_gradcheck(self, numel_per_tensor, total_numel):
        # gradcheck only for double
        inputs = torch.rand((numel_per_tensor.shape[0],), dtype=torch.double,
                            device='cuda', requires_grad=True)
        torch.autograd.gradcheck(batch._TileToPackedCuda.apply,
                                 (inputs, numel_per_tensor, total_numel))

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_float_types(self, inputs_double, numel_per_tensor, total_numel,
                         dtype,
                         target_output_double, target_grad_double):
        inputs = inputs_double.type(dtype).detach()
        inputs.requires_grad = True
        output = batch._TileToPackedCuda.apply(inputs, numel_per_tensor,
                                              total_numel)
        target_output = target_output_double.to(dtype)
        assert torch.equal(output, target_output)
        torch.sum(output).backward()
        target_grad = target_grad_double.to(dtype)
        assert torch.allclose(inputs.grad, target_grad, rtol=1e-2, atol=1e-2)

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_int_types(self, inputs_long, numel_per_tensor, total_numel, dtype,
                       target_output_long):
        inputs = inputs_long.type(dtype)
        output = batch._TileToPackedCuda.apply(inputs, numel_per_tensor,
                                              total_numel)
        target_output = target_output_long.to(dtype)
        assert torch.equal(output, target_output)

    def test_cpu_fail(self, inputs_double, numel_per_tensor, total_numel):
        inputs = inputs_double.cpu()
        with pytest.raises(RuntimeError,
                           match="values_tensor must be a CUDA tensor"):
            batch._TileToPackedCuda.apply(inputs, numel_per_tensor, total_numel)


@pytest.mark.parametrize("device,dtype", NUM_TYPES)
@pytest.mark.parametrize("numel_per_tensor",
                         [torch.LongTensor([1]),
                          torch.LongTensor([1, 100000]),
                          torch.arange(257, dtype=torch.long)])
class TestTileToPacked:
    @pytest.fixture(autouse=True)
    def total_numel(self, numel_per_tensor):
        return torch.sum(numel_per_tensor)

    @pytest.fixture(autouse=True)
    def high_val(self, dtype):
        if dtype.is_floating_point:
            return 1
        else:
            return 32

    @pytest.fixture(autouse=True)
    def inputs(self, high_val, numel_per_tensor, dtype, device):
        return random_tensor(0, high_val, shape=(numel_per_tensor.shape[0],),
                             dtype=dtype, device=device)

    def test_packed_simple_sum(self, inputs, numel_per_tensor, device, dtype):
        tiled_tensor = batch.tile_to_packed(inputs, numel_per_tensor)
        target = _torch_tile_to_packed(inputs, numel_per_tensor)
        assert torch.allclose(tiled_tensor, target)


class TestBatching:
    @pytest.fixture(autouse=True)
    def batch_size(self):
        return 1

    @pytest.fixture(autouse=True)
    def min_shape(self):
        return None

    @pytest.fixture(autouse=True)
    def max_shape(self):
        return (3, 3, 3)

    @pytest.fixture(autouse=True)
    def dtype(self):
        return torch.float

    @pytest.fixture(autouse=True)
    def device(self):
        return 'cpu'

    @pytest.fixture(autouse=True)
    def last_dim(self):
        return 1

    @pytest.fixture(autouse=True)
    def shape_per_tensor(self, batch_size, min_shape, max_shape):
        return random_shape_per_tensor(batch_size, min_shape=min_shape,
                                       max_shape=max_shape)

    @pytest.fixture(autouse=True)
    def high_val(self, dtype):
        return 1 if dtype == torch.bool else 32

    @pytest.fixture(autouse=True)
    def tensor_list(self, dtype, device, high_val, shape_per_tensor, last_dim):
        return [random_tensor(0, high_val, shape=tuple(shape) + (last_dim,),
                              dtype=dtype, device=device)
                for shape in shape_per_tensor]

    @pytest.fixture(autouse=True)
    def numel_per_tensor(self, shape_per_tensor):
        if shape_per_tensor.shape[1] == 1:
            output = shape_per_tensor.squeeze(1)
        else:
            output = torch.prod(shape_per_tensor, dim=1)
        return output

    @pytest.fixture(autouse=True)
    def padding_value(self, dtype):
        if dtype == torch.bool:
            val = False
        elif dtype == torch.uint8:
            val = 0
        else:
            val = -1
        return val

    @pytest.fixture(autouse=True)
    def first_idx(self, numel_per_tensor):
        return batch.get_first_idx(numel_per_tensor)

    @pytest.mark.parametrize("device,dtype", ALL_TYPES)
    @pytest.mark.parametrize("batch_size", [1, 8])
    @pytest.mark.parametrize("min_shape,max_shape",
                             [(None, (2,)), ((2, 2, 2), (3, 3, 3))])
    @pytest.mark.parametrize("last_dim", [1, 8])
    def test_get_shape_per_tensor(self, tensor_list, shape_per_tensor):
        output_shape_per_tensor = batch.get_shape_per_tensor(tensor_list)
        assert torch.equal(output_shape_per_tensor, shape_per_tensor)

    def test_get_shape_per_tensor_fail(self):
        tensor_list = [
            random_tensor(0, 32, shape=(2, 2, 2)),
            random_tensor(0, 32, shape=(3, 3, 3, 3))
        ]
        with pytest.raises(ValueError,
                           match='Expected all tensors to have 3 dimensions but got 4 at index 1'):
            output_shape_per_tensor = batch.get_shape_per_tensor(tensor_list)

    @pytest.mark.parametrize("shape_per_tensor", [
        torch.LongTensor([[1, 1, 1, 4], [1, 2, 1, 1], [1, 1, 3, 1]])])
    @pytest.mark.parametrize("partial_max_shape,expected_max_shape",
                             [(None, (1, 2, 3, 4)),
                              ((-1, -1, -1, 6), (1, 2, 3, 6))])
    def test_fill_max_shape(self, shape_per_tensor, partial_max_shape,
                            expected_max_shape):
        expected_max_shape = torch.LongTensor(expected_max_shape)
        max_shape = batch.fill_max_shape(shape_per_tensor, partial_max_shape)
        assert torch.equal(max_shape, expected_max_shape)

    @pytest.mark.parametrize("numel_per_tensor",
                             [torch.LongTensor([1, 5, 2, 8, 9, 2])])
    def test_get_first_idx(self, numel_per_tensor, first_idx, device):
        first_idx = batch.get_first_idx(numel_per_tensor)
        assert first_idx.device.type == device
        assert first_idx[0] == 0
        for i, numel in enumerate(numel_per_tensor):
            assert (first_idx[i + 1] - first_idx[i]) == numel

    @pytest.mark.parametrize("device,dtype", ALL_TYPES)
    @pytest.mark.parametrize("batch_size", [1, 8])
    @pytest.mark.parametrize("last_dim", [1, 8])
    @pytest.mark.parametrize("min_shape,max_shape",
                             [((1,), (1,)), ((1, 1, 1), (1, 1, 1)),
                              ((2, 6), (5, 10))])
    def test_list_to_packed_to_list(self, tensor_list, shape_per_tensor,
                                    first_idx,
                                    last_dim, dtype, device):
        packed_tensor, output_shape_per_tensor = batch.list_to_packed(
            tensor_list)
        assert torch.equal(output_shape_per_tensor, shape_per_tensor)
        check_packed_tensor(packed_tensor, total_numel=first_idx[-1],
                            last_dim=last_dim,
                            dtype=dtype, device=device)
        for i, tensor in enumerate(tensor_list):
            assert torch.equal(packed_tensor[first_idx[i]:first_idx[i + 1]],
                               tensor.reshape(-1, last_dim))
        output_tensor_list = batch.packed_to_list(packed_tensor,
                                                  shape_per_tensor, first_idx)
        for output_tensor, expected_tensor in zip(output_tensor_list,
                                                  tensor_list):
            assert torch.equal(output_tensor, expected_tensor)

    def test_list_to_packed_fail1(self):
        tensor_list = [
            random_tensor(0, 32, shape=(2, 2, 2)),
            random_tensor(0, 32, shape=(3, 3, 3))
        ]
        with pytest.raises(ValueError,
                           match='Expected all tensor to have last dimension 2 but '
                                 'got 3 at index 1'):
            _ = batch.list_to_packed(tensor_list)

    def test_list_to_packed_fail2(self):
        tensor_list = [
            random_tensor(0, 32, shape=(2, 2, 2), dtype=torch.long,
                          device='cpu'),
            random_tensor(0, 32, shape=(2, 2, 2), dtype=torch.float,
                          device='cuda')
        ]
        with pytest.raises(ValueError,
                           match='Expected all tensor to have type torch.LongTensor but '
                                 'got torch.cuda.FloatTensor at index 1'):
            _ = batch.list_to_packed(tensor_list)

    @pytest.mark.parametrize("device,dtype", ALL_TYPES)
    @pytest.mark.parametrize("batch_size", [1, 8])
    @pytest.mark.parametrize("last_dim", [1, 8])
    @pytest.mark.parametrize("min_shape,max_shape",
                             [((1,), (1,)), ((1, 1, 1), (1, 1, 1)),
                              ((2, 6), (5, 10))])
    def test_list_to_padded_to_list(self, tensor_list, batch_size,
                                    padding_value,
                                    shape_per_tensor, max_shape, last_dim,
                                    dtype, device):
        padded_tensor, output_shape_per_tensor = batch.list_to_padded(
            tensor_list, padding_value, max_shape)

        assert torch.equal(output_shape_per_tensor, shape_per_tensor)
        check_padded_tensor(padded_tensor, batch_size=batch_size,
                            shape_per_tensor=shape_per_tensor,
                            padding_value=padding_value, max_shape=max_shape,
                            last_dim=last_dim,
                            dtype=dtype, device=device)
        for i, shape in enumerate(shape_per_tensor):
            assert torch.equal(
                padded_tensor[[i] + [slice(dim) for dim in shape]],
                tensor_list[i])
        output_tensor_list = batch.padded_to_list(padded_tensor,
                                                  shape_per_tensor)
        for output_tensor, expected_tensor in zip(output_tensor_list,
                                                  tensor_list):
            assert torch.equal(output_tensor, expected_tensor)

    @pytest.mark.parametrize("device,dtype", ALL_TYPES)
    @pytest.mark.parametrize("batch_size", [1, 8])
    @pytest.mark.parametrize("last_dim", [1, 8])
    @pytest.mark.parametrize("min_shape,max_shape",
                             [((1,), (1,)), ((1, 1, 1), (1, 1, 1)),
                              ((2, 6), (5, 10))])
    def test_packed_to_padded_packed(self, tensor_list, batch_size,
                                     padding_value,
                                     shape_per_tensor, first_idx, max_shape,
                                     last_dim,
                                     dtype, device):
        padded_tensor, _ = batch.list_to_padded(tensor_list, padding_value,
                                                max_shape)
        packed_tensor, output_shape_per_tensor = batch.list_to_packed(
            tensor_list)
        _padded_tensor = batch.packed_to_padded(packed_tensor,
                                                output_shape_per_tensor,
                                                first_idx, padding_value,
                                                max_shape)
        assert torch.equal(padded_tensor, _padded_tensor)
        _packed_tensor = batch.padded_to_packed(padded_tensor,
                                                output_shape_per_tensor)
        assert torch.equal(packed_tensor, _packed_tensor)
