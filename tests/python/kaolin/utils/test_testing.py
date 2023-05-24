# Copyright (c) 2019,20-21 NVIDIA CORPORATION & AFFILIATES.
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

import copy
from collections import namedtuple
import logging
import numpy as np
import pytest
import random
import torch

from kaolin.ops.random import random_tensor
from kaolin.ops.spc.uint8 import bits_to_uint8

from kaolin.utils import testing


logger = logging.getLogger(__name__)

sample_tuple = namedtuple('sample_tuple', ['A', 'B', 'C', 'D'])


class TestCheckTensor:
    @pytest.fixture(autouse=True)
    def shape(self):
        return (4, 4)

    @pytest.fixture(autouse=True)
    def dtype(self):
        return torch.float

    @pytest.fixture(autouse=True)
    def device(self):
        return 'cpu'

    @pytest.fixture(autouse=True)
    def tensor(self, shape, dtype, device):
        return random_tensor(0, 256, shape=shape, dtype=dtype, device=device)

    def test_tensor_success(self, tensor, shape, dtype, device):
        assert testing.check_tensor(tensor, shape, dtype, device)

    @pytest.mark.parametrize("partial_shape", [(4, None), (None, 4)])
    def test_tensor_partial_shape_success(self, tensor, partial_shape, dtype,
                                          device):
        assert testing.check_tensor(tensor, partial_shape, dtype, device)

    def test_tensor_default_success(self, tensor):
        assert testing.check_tensor(tensor)

    @pytest.mark.parametrize("wrong_shape", [(3, 3)])
    def test_tensor_fail1(self, tensor, wrong_shape, dtype, device):
        with pytest.raises(ValueError,
                           match=r"tensor shape is torch.Size\(\[4, 4\]\), should be \(3, 3\)"):
            testing.check_tensor(tensor, wrong_shape, dtype, device)
        assert not testing.check_tensor(tensor, wrong_shape, dtype, device,
                                        throw=False)

    @pytest.mark.parametrize("wrong_dtype", [torch.long])
    def test_tensor_fail2(self, tensor, shape, wrong_dtype, device):
        with pytest.raises(TypeError,
                           match="tensor dtype is torch.float32, should be torch.int64"):
            testing.check_tensor(tensor, shape, wrong_dtype, device)
        assert not testing.check_tensor(tensor, shape, wrong_dtype, device,
                                        throw=False)

    @pytest.mark.parametrize("wrong_device", ['cuda'])
    def test_tensor_fail3(self, tensor, shape, dtype, wrong_device):
        with pytest.raises(TypeError,
                           match="tensor device is cpu, should be cuda"):
            testing.check_tensor(tensor, shape, dtype, wrong_device)
        assert not testing.check_tensor(tensor, shape, dtype, wrong_device,
                                        throw=False)


class TestCheckBatchedTensor:
    @pytest.fixture(autouse=True)
    def shape_per_tensor(self):
        return torch.LongTensor([[1, 1], [2, 2]])

    @pytest.fixture(autouse=True)
    def batch_size(self, shape_per_tensor):
        return shape_per_tensor.shape[0]

    @pytest.fixture(autouse=True)
    def max_shape(self):
        return (4, 4)

    @pytest.fixture(autouse=True)
    def last_dim(self):
        return 2

    @pytest.fixture(autouse=True)
    def dtype(self):
        return torch.float

    @pytest.fixture(autouse=True)
    def device(self):
        return 'cpu'

    @pytest.fixture(autouse=True)
    def padding_value(self):
        return -1.

    @pytest.fixture(autouse=True)
    def total_numel(self, shape_per_tensor):
        return torch.sum(torch.prod(shape_per_tensor, dim=1))

    @pytest.fixture(autouse=True)
    def packed_tensor(self, total_numel, last_dim, dtype, device):
        return random_tensor(0, 256, shape=(total_numel, last_dim),
                             dtype=dtype, device=device)

    @pytest.fixture(autouse=True)
    def padded_tensor(self, batch_size, max_shape, last_dim, padding_value,
                      dtype, device):
        output = torch.full((batch_size, *max_shape, last_dim),
                            fill_value=padding_value,
                            dtype=dtype, device=device)
        output[0, :1, :1] = 0
        output[1, :2, :2] = 0
        return output

    def test_packed_success(self, packed_tensor, total_numel, last_dim, dtype,
                            device):
        assert testing.check_packed_tensor(packed_tensor, total_numel, last_dim,
                                           dtype, device)

    def test_packed_default_success(self, packed_tensor):
        assert testing.check_packed_tensor(packed_tensor)

    @pytest.mark.parametrize("wrong_total_numel", [6])
    def test_packed_fail1(self, packed_tensor, wrong_total_numel, last_dim,
                          dtype, device):
        with pytest.raises(ValueError,
                           match='tensor total number of elements is 5, should be 6'):
            testing.check_packed_tensor(packed_tensor, wrong_total_numel,
                                        last_dim, dtype, device)
        assert not testing.check_packed_tensor(packed_tensor, wrong_total_numel,
                                               last_dim,
                                               dtype, device, throw=False)

    @pytest.mark.parametrize("wrong_last_dim", [3])
    def test_packed_fail2(self, packed_tensor, total_numel, wrong_last_dim,
                          dtype, device):
        with pytest.raises(ValueError,
                           match='tensor last_dim is 2, should be 3'):
            testing.check_packed_tensor(packed_tensor, total_numel,
                                        wrong_last_dim, dtype, device)
        assert not testing.check_packed_tensor(packed_tensor, total_numel,
                                               wrong_last_dim,
                                               dtype, device, throw=False)

    @pytest.mark.parametrize("wrong_dtype", [torch.long])
    def test_packed_fail3(self, packed_tensor, total_numel, last_dim,
                          wrong_dtype, device):
        with pytest.raises(TypeError,
                           match='tensor dtype is torch.float32, should be torch.int64'):
            testing.check_packed_tensor(packed_tensor, total_numel, last_dim,
                                        wrong_dtype, device)
        assert not testing.check_packed_tensor(packed_tensor, total_numel,
                                               last_dim,
                                               wrong_dtype, device, throw=False)

    @pytest.mark.parametrize("wrong_device", ['cuda'])
    def test_packed_fail4(self, packed_tensor, total_numel, last_dim, dtype,
                          wrong_device):
        with pytest.raises(TypeError,
                           match='tensor device is cpu, should be cuda'):
            testing.check_packed_tensor(packed_tensor, total_numel, last_dim,
                                        dtype, wrong_device)
        assert not testing.check_packed_tensor(packed_tensor, total_numel,
                                               last_dim,
                                               dtype, wrong_device, throw=False)

    def test_padded_success(self, padded_tensor, padding_value,
                            shape_per_tensor,
                            batch_size, max_shape, last_dim, dtype, device):
        assert testing.check_padded_tensor(padded_tensor, padding_value,
                                           shape_per_tensor,
                                           batch_size, max_shape, last_dim,
                                           dtype, device)

    @pytest.mark.parametrize("partial_max_shape", [(4, None), (None, 4)])
    def test_padded_partial_shape_success(self, padded_tensor, padding_value,
                                          shape_per_tensor,
                                          batch_size, partial_max_shape,
                                          last_dim, dtype, device):
        assert testing.check_padded_tensor(padded_tensor, padding_value,
                                           shape_per_tensor,
                                           batch_size, partial_max_shape,
                                           last_dim, dtype, device)

    def test_padded_default_success(self, padded_tensor):
        assert testing.check_padded_tensor(padded_tensor)

    @pytest.mark.parametrize("wrong_padding_value", [-2])
    def test_padded_fail1(self, padded_tensor, wrong_padding_value,
                          shape_per_tensor,
                          batch_size, max_shape, last_dim, dtype, device):
        with pytest.raises(ValueError,
                           match=r'tensor padding at \(0, 0, 1, 0\) is -1.0, should be -2'):
            testing.check_padded_tensor(padded_tensor, wrong_padding_value,
                                        shape_per_tensor,
                                        batch_size, max_shape, last_dim, dtype,
                                        device)
        assert not testing.check_padded_tensor(padded_tensor,
                                               wrong_padding_value,
                                               shape_per_tensor, batch_size,
                                               max_shape,
                                               last_dim, dtype, device,
                                               throw=False)

    @pytest.mark.parametrize("wrong_shape_per_tensor",
                             [torch.LongTensor([[1, 1], [1, 1]])])
    def test_padded_fail2(self, padded_tensor, padding_value,
                          wrong_shape_per_tensor,
                          batch_size, max_shape, last_dim, dtype, device):
        with pytest.raises(ValueError,
                           match=r'tensor padding at \(1, 0, 1, 0\) is 0.0, should be -1.0'):
            testing.check_padded_tensor(padded_tensor, padding_value,
                                        wrong_shape_per_tensor,
                                        batch_size, max_shape, last_dim, dtype,
                                        device)
        assert not testing.check_padded_tensor(padded_tensor, padding_value,
                                               wrong_shape_per_tensor,
                                               batch_size, max_shape,
                                               last_dim, dtype, device,
                                               throw=False)

    @pytest.mark.parametrize("wrong_batch_size", [3])
    def test_padded_fail3(self, padded_tensor, padding_value, shape_per_tensor,
                          wrong_batch_size, max_shape, last_dim, dtype, device):
        with pytest.raises(ValueError,
                           match='batch_size is 3, but there are 2 shapes in shape_per_tensor'):
            testing.check_padded_tensor(padded_tensor, padding_value,
                                        shape_per_tensor,
                                        wrong_batch_size, max_shape, last_dim,
                                        dtype, device)
        assert not testing.check_padded_tensor(padded_tensor, padding_value,
                                               shape_per_tensor,
                                               wrong_batch_size, max_shape,
                                               last_dim, dtype, device,
                                               throw=False)

    @pytest.mark.parametrize("wrong_batch_size", [3])
    def test_padded_fail4(self, padded_tensor, padding_value,
                          wrong_batch_size, max_shape, last_dim, dtype, device):
        with pytest.raises(ValueError,
                           match='tensor batch size is 2, should be 3'):
            testing.check_padded_tensor(padded_tensor, padding_value,
                                        batch_size=wrong_batch_size,
                                        max_shape=max_shape, last_dim=last_dim,
                                        dtype=dtype,
                                        device=device)
        assert not testing.check_padded_tensor(padded_tensor, padding_value,
                                               batch_size=wrong_batch_size,
                                               max_shape=max_shape,
                                               last_dim=last_dim, dtype=dtype,
                                               device=device,
                                               throw=False)

    @pytest.mark.parametrize("wrong_max_shape", [(4, 4, 4)])
    def test_padded_fail5(self, padded_tensor, padding_value, shape_per_tensor,
                          batch_size, wrong_max_shape, last_dim, dtype, device):
        with pytest.raises(ValueError,
                           match=r'tensor max_shape is torch.Size\(\[4, 4\]\), should be \(4, 4, 4\)'):
            testing.check_padded_tensor(padded_tensor, padding_value,
                                        shape_per_tensor,
                                        batch_size, wrong_max_shape, last_dim,
                                        dtype, device)
        assert not testing.check_padded_tensor(padded_tensor, padding_value,
                                               shape_per_tensor, batch_size,
                                               wrong_max_shape,
                                               last_dim, dtype, device,
                                               throw=False)

    @pytest.mark.parametrize("wrong_last_dim", [3])
    def test_padded_fail6(self, padded_tensor, padding_value, shape_per_tensor,
                          batch_size, max_shape, wrong_last_dim, dtype, device):
        with pytest.raises(ValueError,
                           match='tensor last_dim is 2, should be 3'):
            testing.check_padded_tensor(padded_tensor, padding_value,
                                        shape_per_tensor,
                                        batch_size, max_shape, wrong_last_dim,
                                        dtype, device)
        assert not testing.check_padded_tensor(padded_tensor, padding_value,
                                               shape_per_tensor, batch_size,
                                               max_shape,
                                               wrong_last_dim, dtype, device,
                                               throw=False)

    @pytest.mark.parametrize("wrong_dtype", [torch.long])
    def test_padded_fail7(self, padded_tensor, padding_value, shape_per_tensor,
                          batch_size, max_shape, last_dim, wrong_dtype, device):
        with pytest.raises(TypeError,
                           match='tensor dtype is torch.float32, should be torch.int64'):
            testing.check_padded_tensor(padded_tensor, padding_value,
                                        shape_per_tensor,
                                        batch_size, max_shape, last_dim,
                                        wrong_dtype, device)
        assert not testing.check_padded_tensor(padded_tensor, padding_value,
                                               shape_per_tensor, batch_size,
                                               max_shape,
                                               last_dim, wrong_dtype, device,
                                               throw=False)

    @pytest.mark.parametrize("wrong_device", ['cuda'])
    def test_padded_fail8(self, padded_tensor, padding_value, shape_per_tensor,
                          batch_size, max_shape, last_dim, dtype, wrong_device):
        with pytest.raises(TypeError,
                           match='tensor device is cpu, should be cuda'):
            testing.check_padded_tensor(padded_tensor, padding_value,
                                        shape_per_tensor,
                                        batch_size, max_shape, last_dim, dtype,
                                        wrong_device)
        assert not testing.check_padded_tensor(padded_tensor, padding_value,
                                               shape_per_tensor, batch_size,
                                               max_shape,
                                               last_dim, dtype, wrong_device,
                                               throw=False)

    def test_padded_fail9(self, padded_tensor, padding_value, batch_size,
                          max_shape,
                          last_dim, dtype, device):
        with pytest.raises(ValueError,
                           match='shape_per_tensor should not be None if padding_value is set'):
            testing.check_padded_tensor(padded_tensor, padding_value,
                                        batch_size=batch_size,
                                        max_shape=max_shape, last_dim=last_dim,
                                        dtype=dtype,
                                        device=device)

class TestCheckSpcOctrees:
    @pytest.fixture(autouse=True)
    def device(self):
        return 'cuda'

    @pytest.fixture(autouse=True)
    def octrees(self, device):
        bits_t = torch.flip(torch.tensor(
            [[0, 0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0],

             [1, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 0, 1, 1], [0, 1, 0, 0, 0, 0, 0, 0]
            ], dtype=torch.bool, device=device), dims=(-1,))
        return bits_to_uint8(bits_t)

    @pytest.fixture(autouse=True)
    def lengths(self):
        return torch.tensor([4, 5], dtype=torch.int)

    @pytest.fixture(autouse=True)
    def batch_size(self):
        return 2

    @pytest.fixture(autouse=True)
    def level(self):
        return 3

    def test_spc_success(self, octrees, lengths, batch_size, level, device):
        assert testing.check_spc_octrees(octrees, lengths, batch_size, level, device)

    def test_spc_default_success(self, octrees, lengths):
        assert testing.check_spc_octrees(octrees, lengths)

    @pytest.mark.parametrize('wrong_device', ['cpu'])
    def test_spc_wrong_device(self, octrees, lengths, wrong_device):
        with pytest.raises(ValueError,
                           match='octrees is on cuda:0, should be on cpu'):
            testing.check_spc_octrees(octrees, lengths, device=wrong_device)

    @pytest.mark.parametrize('wrong_lengths',
                             [torch.tensor([3, 5], dtype=torch.int)])
    def test_spc_wrong_lengths(self, octrees, wrong_lengths):
        with pytest.raises(ValueError,
                           match='lengths at 0 is 3, but level 3 ends at length 4'):
            testing.check_spc_octrees(octrees, wrong_lengths)

    @pytest.mark.parametrize('wrong_batch_size', [3])
    def test_spc_wrong_batch_size(self, octrees, lengths, wrong_batch_size):
        with pytest.raises(ValueError,
                           match=r'lengths is of shape torch.Size\(\[2\]\), '
                                 'but batch_size should be 3'):
            testing.check_spc_octrees(octrees, lengths, batch_size=wrong_batch_size)

    @pytest.mark.parametrize('wrong_level', [4])
    def test_spc_wrong_level(self, octrees, lengths, wrong_level):
        with pytest.raises(ValueError,
                           match='octree 0 ends at level 3, should end at 4'):
            testing.check_spc_octrees(octrees, lengths, level=wrong_level)

class TestSeedDecorator:

    @pytest.fixture(autouse=True)
    def get_fix_input(self):
        return torch.tensor([1, 2, 3])

    @pytest.fixture(autouse=True)
    def expected_seed1(self):
        torch_expected = torch.tensor([[0.6614, 0.2669, 0.0617]])
        np_expected = 0.5507979025745755
        random_expected = 978

        return torch_expected, np_expected, random_expected

    @pytest.fixture(autouse=True)
    def expected_seed2(self):
        torch_expected = torch.tensor([[-0.4868, -0.6038, -0.5581]])
        np_expected = 0.010374153885699955
        random_expected = 585

        return torch_expected, np_expected, random_expected

    @testing.with_seed(torch_seed=1, numpy_seed=2, random_seed=3)
    def test_seed1(self, expected_seed1):
        torch_result = torch.randn((1, 3), dtype=torch.float)
        np_result = np.random.random_sample()
        random_result = random.randint(0, 1000)

        torch_expected, np_expected, random_expected = expected_seed1

        assert torch.allclose(torch_result, torch_expected, atol=1e-4, rtol=1e-4)
        assert np_result == np_expected
        assert random_result == random_expected

    @testing.with_seed(torch_seed=5, numpy_seed=10, random_seed=9)
    def test_seed2(self, expected_seed2):
        torch_result = torch.randn((1, 3), dtype=torch.float)
        np_result = np.random.random_sample()
        random_result = random.randint(0, 1000)

        torch_expected, np_expected, random_expected = expected_seed2

        assert torch.allclose(torch_result, torch_expected, atol=1e-4, rtol=1e-4)
        assert np_result == np_expected
        assert random_result == random_expected

    @testing.with_seed(torch_seed=1, numpy_seed=2, random_seed=3)
    def test_nested_decorator(self, expected_seed1, expected_seed2):
        # The decorator should have no effect on other functions
        torch_expected1, np_expected1, random_expected1 = expected_seed1

        torch_expected2, np_expected2, random_expected2 = expected_seed2

        @testing.with_seed(torch_seed=5, numpy_seed=10, random_seed=9)
        def subtest_seed():
            torch_result = torch.randn((1, 3), dtype=torch.float)
            np_result = np.random.random_sample()
            random_result = random.randint(0, 1000)

            assert torch.allclose(torch_result, torch_expected2, atol=1e-4, rtol=1e-4)
            assert np_result == np_expected2
            assert random_result == random_expected2

        subtest_seed()

        torch_result = torch.randn((1, 3), dtype=torch.float)
        np_result = np.random.random_sample()
        random_result = random.randint(0, 1000)

        assert torch.allclose(torch_result, torch_expected1, atol=1e-4, rtol=1e-4)
        assert np_result == np_expected1
        assert random_result == random_expected1


    @testing.with_seed(torch_seed=1, numpy_seed=2, random_seed=3)
    def test_with_fixture(self, get_fix_input):
        # Test the seed decorator with pytest fixture works.
        fix_input = get_fix_input

        assert torch.equal(fix_input, torch.tensor([1, 2, 3]))

        torch_result = torch.randn((1, 3), dtype=torch.float)
        np_result = np.random.random_sample()
        random_result = random.randint(0, 1000)

        expected_torch = torch.tensor([[0.6614, 0.2669, 0.0617]], dtype=torch.float)
        expected_np = 0.5507979025745755
        expected_random = 978

        assert torch.allclose(torch_result, expected_torch, atol=1e-4, rtol=1e-4)
        assert np_result == expected_np
        assert random_result == expected_random

    @testing.with_seed(torch_seed=1, numpy_seed=2, random_seed=3)
    @pytest.mark.parametrize("device", ["cpu"])
    def test_with_other_decorator(self, device):
        # Test the seed decorator works with other decorator

        assert device == "cpu"

        torch_result = torch.randn((1, 3), dtype=torch.float, device=device)
        np_result = np.random.random_sample()
        random_result = random.randint(0, 1000)

        expected_torch = torch.tensor([[0.6614, 0.2669, 0.0617]], dtype=torch.float, device=device)
        expected_np = 0.5507979025745755
        expected_random = 978

        assert torch.allclose(torch_result, expected_torch, atol=1e-4, rtol=1e-4)
        assert np_result == expected_np
        assert random_result == expected_random


class TestTensorInfo:
    @pytest.mark.parametrize('dtype', [torch.uint8, torch.float32])
    @pytest.mark.parametrize('shape', [[1], [10, 30, 40]])
    @pytest.mark.parametrize('print_stats', [True, False])
    @pytest.mark.parametrize('detailed', [True, False])
    def test_torch_tensor(self, dtype, shape, print_stats, detailed):
        t = (torch.rand(shape) * 100).to(dtype)
        tensor_name = 'random_tensor'
        str = testing.tensor_info(t, tensor_name, print_stats=print_stats, detailed=detailed)
        logger.debug(str)
        assert len(str) > len(tensor_name)  # Just check that runs and produces output

    @pytest.mark.parametrize('dtype', [np.uint8, np.float32])
    @pytest.mark.parametrize('shape', [[1], [10, 30, 40]])
    @pytest.mark.parametrize('print_stats', [True, False])
    @pytest.mark.parametrize('detailed', [True, False])
    def test_numpy_array(self, dtype, shape, print_stats, detailed):
        t = (np.random.rand(*shape) * 10).astype(dtype)
        tensor_name = 'random_numpy_array'
        str = testing.tensor_info(t, tensor_name, print_stats=print_stats, detailed=detailed)
        logger.debug(str)
        assert len(str) > len(tensor_name)  # Just check that runs and produces output

class TestContainedTorchEqual:
    def test_true(self):
        elem = [1, 'a', {'b': torch.rand(3, 3), 'c': 0.1}]
        other = copy.deepcopy(elem)
        assert testing.contained_torch_equal(elem, other)

        # Also try on a tuple
        elem = sample_tuple('hello', torch.rand(3, 3), (torch.rand(10, 3) * 10).to(torch.int32), {'a': torch.rand(5)})
        other = copy.deepcopy(elem)
        assert testing.contained_torch_equal(elem, other)

    def test_false(self):
        elem = [1, 'a', {'b': torch.rand(3, 3), 'c': 0.1}]
        other = copy.deepcopy(elem)
        other[2]['b'][1, 1] += 1.
        assert not testing.contained_torch_equal(elem, other)

        # Also try on a tuple
        elem = sample_tuple('hello', torch.rand(3, 3), (torch.rand(10, 3) * 10).to(torch.int32), {'a': torch.rand(5)})
        other = copy.deepcopy(elem)
        other.B[0, 0] += 0.001
        assert not testing.contained_torch_equal(elem, other)

    def test_approximate(self):
        elem = [1, 'a', {'b': torch.rand(3, 3), 'c': 0.1}]
        other = copy.deepcopy(elem)
        eps = 0.0001
        other[2]['b'][1, 1] += eps
        other[2]['c'] += eps
        assert not testing.contained_torch_equal(elem, other)
        assert testing.contained_torch_equal(elem, other, approximate=True, atol=eps*2)

class TestCheckTensorAttributeShapes:
    @pytest.mark.parametrize("throw", [True, False])
    def test_checks_pass(self, throw):
        container = {'cat': torch.rand((1, 5, 6)), 'dog': torch.rand((5, 5, 6)), 'colors': torch.rand((100, 3))}
        assert testing.check_tensor_attribute_shapes(container, throw=throw, cat=(1, 5, 6), colors=(None, 3))

        container = sample_tuple('Hello', torch.rand((3, 4, 5)), torch.rand((5, 1, 6)), {})
        assert testing.check_tensor_attribute_shapes(container, throw=throw, B=(3, None, 5), C=[5, 1, 6])

    def test_checks_fail(self):
        container = {'cat': torch.rand((1, 5, 6)), 'dog': torch.rand((5, 5, 6)), 'colors': torch.rand((100, 3))}
        with pytest.raises(ValueError):
            assert testing.check_tensor_attribute_shapes(container, throw=True, cat=(1, 5, 6), colors=(59, 3))
        assert not testing.check_tensor_attribute_shapes(container, throw=False, cat=(1, 50, 6), colors=(59, 3))

class TestPrintDiagnostics:
    def test_print_namedtuple_attributes(self, capsys):
        sample1 = sample_tuple('My Name', [1, 2, 3], torch.zeros((5, 5, 5)), {'a': torch.rand(5)})

        testing.print_namedtuple_attributes(sample1)
        out1, err = capsys.readouterr()
        assert len(out1) > 10

        testing.print_namedtuple_attributes(sample1, detailed=True)
        out1_detailed, err = capsys.readouterr()
        assert len(out1) < len(out1_detailed)


