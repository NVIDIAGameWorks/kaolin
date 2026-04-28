# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Tests for :class:`kaolin.rep.TensorContainerBase`."""

import copy

import pytest
import torch

from kaolin.rep import TensorContainerBase
from kaolin.utils.testing import contained_torch_equal


class MockContainer(TensorContainerBase):
    """Minimal subclass exercising every code path of :class:`TensorContainerBase`.

    Has one required tensor attribute (``data``), one optional dict-of-tensors attribute
    (``extras``), and one non-tensor attribute (``label``).
    """

    @classmethod
    def class_tensor_attributes(cls):
        return ['data', 'extras']

    @classmethod
    def class_other_attributes(cls):
        return ['label']

    def __init__(self, data, extras=None, label='default'):
        self.data = data
        self.extras = extras
        self.label = label

    def check_tensor_attribute(self, attr, log_error=False):
        value = getattr(self, attr)
        if attr == 'data':
            return torch.is_tensor(value)
        if attr == 'extras':
            if value is None:
                return True
            if not isinstance(value, dict):
                return False
            return all(torch.is_tensor(v) for v in value.values())
        return False

    def check_other_attribute(self, attr, log_error=False):
        if attr == 'label':
            return isinstance(self.label, str)
        return False


class MockContainerNoOverride(TensorContainerBase):
    """Mock that does NOT override :meth:`check_other_attribute` despite having other attrs."""

    @classmethod
    def class_tensor_attributes(cls):
        return ['data']

    @classmethod
    def class_other_attributes(cls):
        return ['label']

    def __init__(self, data, label='default'):
        self.data = data
        self.label = label

    def check_tensor_attribute(self, attr, log_error=False):
        return torch.is_tensor(getattr(self, attr))


def _make_instance(label='default', with_extras=True):
    data = torch.randn(4, 3)
    extras = None
    if with_extras:
        extras = {'a': torch.randn(4, 2), 'b': torch.randn(4)}
    return MockContainer(data=data, extras=extras, label=label)


class TestTensorContainerBase:
    def test_assert_supported_pass_and_raise(self):
        instance = _make_instance()
        instance.assert_supported('data')
        instance.assert_supported('extras')
        instance.assert_supported('label')
        with pytest.raises(AttributeError):
            instance.assert_supported('not_an_attribute')

    def test_get_attributes(self):
        instance = _make_instance()
        attrs = instance.get_attributes()
        assert set(attrs) == {'data', 'extras', 'label'}

        only_tensors = instance.get_attributes(only_tensors=True)
        assert set(only_tensors) == {'data', 'extras'}

        instance_no_extras = _make_instance(with_extras=False)
        assert set(instance_no_extras.get_attributes()) == {'data', 'label'}
        assert set(instance_no_extras.get_attributes(only_tensors=True)) == {'data'}

    def test_as_dict_roundtrip(self):
        instance = _make_instance(label='my-label')
        d = instance.as_dict()
        assert set(d.keys()) == {'data', 'extras', 'label'}
        assert d['label'] == 'my-label'

        rebuilt = MockContainer(**d)
        assert contained_torch_equal(rebuilt.as_dict(), d, approximate=True)

        d_only_tensors = instance.as_dict(only_tensors=True)
        assert set(d_only_tensors.keys()) == {'data', 'extras'}

    def test_construct_apply_shallow_copy(self):
        instance = _make_instance()
        result = instance._construct_apply(lambda t: t * 2)

        assert result is not instance
        assert result.label == instance.label

        assert torch.allclose(result.data, instance.data * 2)
        for k in instance.extras:
            assert torch.allclose(result.extras[k], instance.extras[k] * 2)

    def test_to_dtype(self):
        instance = _make_instance()
        moved = instance.to(dtype=torch.float64)
        assert moved.data.dtype == torch.float64
        for v in moved.extras.values():
            assert v.dtype == torch.float64
        assert instance.data.dtype == torch.float32

    def test_to_attributes_filter(self):
        instance = _make_instance()
        moved = instance.to(dtype=torch.float64, attributes=['data'])
        assert moved.data.dtype == torch.float64
        for v in moved.extras.values():
            assert v.dtype == torch.float32

    def test_cpu(self):
        instance = _make_instance()
        result = instance.cpu()
        assert result.data.device.type == 'cpu'
        for v in result.extras.values():
            assert v.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
    def test_cuda_cpu_roundtrip(self):
        instance = _make_instance()
        on_cuda = instance.cuda()
        assert on_cuda.data.device.type == 'cuda'
        for v in on_cuda.extras.values():
            assert v.device.type == 'cuda'

        back_on_cpu = on_cuda.cpu()
        assert back_on_cpu.data.device.type == 'cpu'

    def test_detach(self):
        data = torch.randn(4, 3, requires_grad=True)
        extras = {'a': torch.randn(4, 2, requires_grad=True)}
        instance = MockContainer(data=data, extras=extras)

        detached = instance.detach()
        assert not detached.data.requires_grad
        for v in detached.extras.values():
            assert not v.requires_grad
        assert instance.data.requires_grad

    def test_describe_attribute_tensor_and_dict(self):
        instance = _make_instance()
        tensor_desc = instance.describe_attribute('data')
        assert isinstance(tensor_desc, str)
        assert 'data' in tensor_desc

        dict_desc = instance.describe_attribute('extras')
        assert isinstance(dict_desc, str)
        for k in instance.extras:
            assert f'extras[{k}]' in dict_desc

        with_stats = instance.describe_attribute('data', print_stats=True)
        assert len(with_stats) >= len(tensor_desc)

    def test_describe_attribute_unsupported_raises(self):
        instance = _make_instance()
        with pytest.raises(AttributeError):
            instance.describe_attribute('does_not_exist')

    def test_check_sanity_pass(self):
        instance = _make_instance()
        assert instance.check_sanity()
        assert instance.check_sanity(log_error=False)

    def test_check_sanity_fails_on_tensor(self):
        instance = _make_instance()
        instance.data = 'not a tensor'
        assert not instance.check_sanity(log_error=False)

    def test_check_sanity_fails_on_other(self):
        instance = _make_instance()
        instance.label = 12345
        assert not instance.check_sanity(log_error=False)

    def test_default_check_other_attribute_raises(self):
        instance = MockContainerNoOverride(data=torch.randn(3, 2), label='x')
        with pytest.raises(NotImplementedError):
            instance.check_other_attribute('label')

    def test_to_string_and_dunders(self):
        instance = _make_instance(label='hello')
        s = instance.to_string()
        assert isinstance(s, str)
        assert 'MockContainer' in s
        assert 'data' in s
        assert 'extras' in s

        s_long = instance.to_string(print_stats=True, detailed=True)
        assert isinstance(s_long, str)
        assert len(s_long) >= len(s)

        assert str(instance) == instance.to_string()
        assert repr(instance) == instance.to_string()
