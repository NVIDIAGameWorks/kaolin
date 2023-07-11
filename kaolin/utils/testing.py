# Copyright (c) 2019,20-21-22 NVIDIA CORPORATION & AFFILIATES.
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
import collections
import functools
import logging
import numpy as np
import torch

import kaolin.ops.random as random
from kaolin.ops.spc.uint8 import uint8_bits_sum

BOOL_DTYPES = [torch.bool]
INT_DTYPES = [torch.uint8, torch.short, torch.int, torch.long]
FLOAT_DTYPES = [torch.half, torch.float, torch.double]

NUM_DTYPES = INT_DTYPES + FLOAT_DTYPES
ALL_DTYPES = NUM_DTYPES + BOOL_DTYPES

ALL_DEVICES = ['cpu', 'cuda']

BOOL_TYPES = [('cuda', torch.bool), ('cpu', torch.bool)]
INT_TYPES = [(device, dtype) for device in ALL_DEVICES for dtype in INT_DTYPES]
FLOAT_TYPES = [('cuda', dtype) for dtype in FLOAT_DTYPES] + \
              [('cpu', dtype) for dtype in FLOAT_DTYPES if dtype != torch.half]

CUDA_FLOAT_TYPES = [('cuda', dtype) for dtype in FLOAT_DTYPES]

NUM_TYPES = INT_TYPES + FLOAT_TYPES
ALL_TYPES = NUM_TYPES + BOOL_TYPES

def with_seed(torch_seed=0, numpy_seed=None, random_seed=None):
    """Decorator to fix the seed of a function.

    Args:
        torch_seed (int): The desired seed for torch module.
        random_seed (int): The desired seed for random module. Default: torch_seed value.
        numpy_seed (int): The desired seed for numpy module. Default: torch_seed value.
    """
    def decorator(orig_test):
        @functools.wraps(orig_test)
        def orig_test_wrapper(*args, **kwargs):
            torch_state, random_state, np_state = random.get_state()
            random.manual_seed(torch_seed, numpy_seed, random_seed)
            output = orig_test(*args, **kwargs)
            random.set_state(torch_state, random_state, np_state)
            return output
        return orig_test_wrapper
    return decorator

def check_tensor(tensor, shape=None, dtype=None, device=None, throw=True):
    """Check if :class:`torch.Tensor` is valid given set of criteria.

    Args:
        tensor (torch.Tensor): the tensor to be tested.
        shape (list or tuple of int, optional): the expected shape,
            if a dimension is set at ``None`` then it's not verified.
        dtype (torch.dtype, optional): the expected dtype.
        device (torch.device, optional): the expected device.
        throw (bool): if true (default), will throw if checks fail

    Return:
        (bool) True if checks pass
    """
    if shape is not None:
        if len(shape) != tensor.ndim:
            if throw:
                raise ValueError(f"tensor have {tensor.ndim} ndim, should have {len(shape)}")
            return False
        for i, dim in enumerate(shape):
            if dim is not None and tensor.shape[i] != dim:
                if throw:
                    raise ValueError(f"tensor shape is {tensor.shape}, should be {shape}")
                return False
    if dtype is not None and dtype != tensor.dtype:
        if throw:
            raise TypeError(f"tensor dtype is {tensor.dtype}, should be {dtype}")
        return False
    if device is not None and device != tensor.device.type:
        if throw:
            raise TypeError(f"tensor device is {tensor.device.type}, should be {device}")
        return False
    return True

def check_packed_tensor(tensor, total_numel=None, last_dim=None, dtype=None, device=None,
                        throw=True):
    """Check if :ref:`packed tensor<packed>` is valid given set of criteria.

    Args:
        tensor (torch.Tensor): the packed tensor to be tested.
        total_numel (int, optional): the expected number of elements.
        last_dim (int, optional): the expected last dimension size.
        dtype (torch.dtype, optional): the expected dtype.
        device (torch.device, optional): the expected device.
        throw (bool): if True the check will raise an error if failing.

    Return:
        (bool): status of the check.
    """
    if not check_tensor(tensor, shape=(None, None), dtype=dtype, device=device, throw=throw):
        return False
    if total_numel is not None and tensor.shape[0] != total_numel:
        if throw:
            raise ValueError(f"tensor total number of elements is {tensor.shape[0]}, "
                             f"should be {total_numel}")
        return False
    if last_dim is not None and last_dim != tensor.shape[-1]:
        if throw:
            raise ValueError(f"tensor last_dim is {tensor.shape[-1]}, should be {last_dim}")
        return False
    return True

def check_padded_tensor(tensor, padding_value=None, shape_per_tensor=None,
                        batch_size=None, max_shape=None, last_dim=None,
                        dtype=None, device=None, throw=True):
    """Check if :ref:`padded tensor<padded>` is valid given set of criteria.

    Args:
        tensor (torch.Tensor): the padded tensor to be tested.
        padding_value (int, optional): the expected number of elements,
            :attr:`shape_per_tensor` must be provided with padding_value.
        shape_per_tensor (torch.LongTensor, optional): the expected :attr:`shape_per_tensor`.
        batch_size (int, optional): the expected batch size.
        last_dim (int, optional): the expected last dimension size.
        dtype (torch.dtype, optional): the expected dtype.
        device (torch.device, optional): the expected device.
        throw (bool): if True the check will raise an error if failing.

    Return:
        (bool): status of the check.
    """
    if not check_tensor(tensor, dtype=dtype, device=device, throw=throw):
        return False
    if shape_per_tensor is not None:
        if batch_size is not None and batch_size != shape_per_tensor.shape[0]:
            if throw:
                raise ValueError(f"batch_size is {batch_size}, "
                                 f"but there are {shape_per_tensor.shape[0]} "
                                 f"shapes in shape_per_tensor")
            return False
        batch_size = shape_per_tensor.shape[0]
    if batch_size is not None and batch_size != tensor.shape[0]:
        if throw:
            raise ValueError(f"tensor batch size is {tensor.shape[0]}, should be {batch_size}")
        return False
    if max_shape is not None:
        for i, dim in enumerate(max_shape, 1):
            if dim is not None and dim != tensor.shape[i]:
                if throw:
                    raise ValueError(f"tensor max_shape is {tensor.shape[1:-1]}, should be {max_shape}")
                return False
    if last_dim is not None and last_dim != tensor.shape[-1]:
        if throw:
            raise ValueError(f"tensor last_dim is {tensor.shape[-1]}, should be {last_dim}")
        return False
    if padding_value is not None:
        if shape_per_tensor is None:
            raise ValueError("shape_per_tensor should not be None if padding_value is set")
        mask = torch.ones(tensor.shape, dtype=torch.bool, device=tensor.device)
        for i, shape in enumerate(shape_per_tensor):
            mask[[i] + [slice(dim) for dim in shape]] = False
        if any(tensor[mask] != padding_value):
            if throw:
                first_false_coord = tuple(
                    int(l[0]) for l in torch.where((tensor != padding_value) & mask))
                raise ValueError(f"tensor padding at {first_false_coord} is {tensor[first_false_coord]}, "
                                 f"should be {padding_value}")
            return False
    return True

def check_spc_octrees(octrees, lengths, batch_size=None, level=None,
                      device=None, throw=True):
    if batch_size is not None and (batch_size,) != lengths.shape:
        if throw:
            raise ValueError(f"lengths is of shape {lengths.shape}, "
                             f"but batch_size should be {batch_size}")
        return False

    if device is not None and device != octrees.device.type:
        if throw:
            raise ValueError(f"octrees is on {octrees.device}, "
                             f"should be on {device}.")
        return False

    octree_start_idx = 0
    for i, length in enumerate(lengths):
        cur_node_idx = 0
        cur_num_nodes = 1
        cur_level = 0
        octree = octrees[octree_start_idx:octree_start_idx + length]
        while cur_node_idx < length:
            cur_level += 1
            cur_level_nodes = octree[cur_node_idx:cur_node_idx + cur_num_nodes]
            cur_node_idx += cur_num_nodes
            cur_num_nodes = int(torch.sum(uint8_bits_sum(cur_level_nodes).long()))
        if cur_node_idx > length:
            if throw:
                raise ValueError(f"lengths at {i} is {length}, "
                                 f"but level {cur_level} ends at length {cur_node_idx}")
            return False
        if level is not None and level != cur_level:
            if throw:
                raise ValueError(f"octree {i} ends at level {cur_level}, "
                                 f"should end at {level}")
            return False
        octree_start_idx += length
    return True

def tensor_info(t, name='', print_stats=False, detailed=False):
    """
    Convenience method to format diagnostic tensor information, including
    shape, type, and optional attributes if specified as string.
    This information can then be logged as:
    logger.debug(tensor_info(my_tensor, 'my tensor'))

    Log output:
    my_tensor: [10, 2, 100, 100] (torch.float32)

    Args:
        t: input pytorch tensor or numpy array or None
        name: human readable name of the tensor (optional)
        print_stats: if True, includes mean/max/min statistics (takes compute time)
        detailed: if True, includes details about tensor properties

    Returns:
        (String) formatted string

    Examples:
        >>> t = torch.Tensor([0., 2., 3.])
        >>> tensor_info(t, 'mytensor', True, True)
        'mytensor: torch.Size([3]) (torch.float32)  - [min 0.0000, max 3.0000, mean 1.6667]  - req_grad=False, is_leaf=True, device=cpu, layout=torch.strided'
    """
    def _get_stats_str():
        if torch.is_tensor(t):
            return ' - [min %0.4f, max %0.4f, mean %0.4f]' % \
                   (torch.min(t).item(),
                    torch.max(t).item(),
                    torch.mean(t.to(torch.float32)).item())
        elif type(t) == np.ndarray:
            return ' - [min %0.4f, max %0.4f, mean %0.4f]' % (np.min(t), np.max(t), np.mean(t))
        else:
            raise RuntimeError('Not implemented for {}'.format(type(t)))

    def _get_details_str():
        if torch.is_tensor(t):
            return ' - req_grad={}, is_leaf={}, layout={}'.format(
                t.requires_grad, t.is_leaf, t.layout)

    if t is None:
        return '%s: None' % name

    shape_str = ''
    if hasattr(t, 'shape'):
        shape_str = '%s ' % str(list(t.shape))

    if hasattr(t, 'dtype'):
        type_str = '%s' % str(t.dtype)
    else:
        type_str = '{}'.format(type(t))

    device_str = ''
    if hasattr(t, 'device'):
        device_str = '[{}]'.format(t.device)

    name_str = ''
    if name is not None and len(name) > 0:
        name_str = '%s: ' % name

    return ('%s%s(%s)%s %s %s' %
            (name_str, shape_str, type_str, device_str,
             (_get_stats_str() if print_stats else ''),
             (_get_details_str() if detailed else '')))

def contained_torch_equal(elem, other, approximate=False, print_error_context=None, **allclose_args):
    """Check for equality (or allclose if approximate) of two objects potentially containing tensors.

    :func:`torch.equal` do not support data structure like dictionary / arrays
    and `==` is ambiguous on :class:`torch.Tensor`.
    This class will try to apply recursion through :class:`collections.abc.Mapping`,
    :class:`collections.abc.Sequence`, :func:`torch.equal` if the objects are `torch.Tensor`,
    of else `==` operator.

    Args:
        elem (object, dict, list, tuple): The first object
        other (object, dict, list, tuple): The other object to compare to ``elem``
        approximate (bool): if requested will use allclose for comparison instead (default=False)
        print_error_context (str): set to any string value to print the context for the first nested failed match
        allclose_args: arguments to `torch.allclose` if approximate comparison requested

    Return (bool): the comparison result
    """
    def _maybe_print(val, extra_context='', prefix_string='Failed match for '):
        if not val and print_error_context is not None:  # match failed
            print(f'{prefix_string}{print_error_context}{extra_context}')
        return val

    elem_type = type(elem)
    if elem_type != type(other):
        return _maybe_print(False)

    def _tensor_compare(a, b):
        if not approximate:
            return torch.equal(a, b)
        else:
            return torch.allclose(a, b, **allclose_args)

    def _number_compare(a, b):
        return _tensor_compare(torch.tensor([a]), torch.tensor([b]))

    def _attrs_to_dict(a, attrs):
        return {k : getattr(a, k) for k in attrs if hasattr(a, k)}

    def _recursive_error_context(append_context):
        if print_error_context is None:
            return None
        return f'{print_error_context}{append_context}'

    recursive_args = copy.copy(allclose_args)
    recursive_args['approximate'] = approximate

    if isinstance(elem, torch.Tensor):
        return _maybe_print(_tensor_compare(elem, other))
    elif isinstance(elem, str):
        return _maybe_print(elem == other, extra_context=f': {elem} vs {other}')
    elif isinstance(elem, float):
        return _maybe_print(_number_compare(elem, other), extra_context=f': {elem} vs {other}')
    elif isinstance(elem, collections.abc.Mapping):
        if elem.keys() != other.keys():
            return _maybe_print(False, f': {elem.keys()} vs {other.keys()}', 'Different keys for ')
        return all(contained_torch_equal(
            elem[key], other[key],
            print_error_context=_recursive_error_context(f'[{key}]'), **recursive_args) for key in elem)
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        if set(elem._fields) != set(other._fields):
            return _maybe_print(False, f': {elem._fields} vs {other._fields}', 'Different fields for ')
        return all(contained_torch_equal(
            getattr(elem, f), getattr(other, f),
            print_error_context=_recursive_error_context(f'.{f}'), **recursive_args) for f in elem._fields)
    elif isinstance(elem, collections.abc.Sequence):
        if len(elem) != len(other):
            return _maybe_print(False, ': {len(elem)} vs {len(other)}', 'Different length for ')
        return all(contained_torch_equal(
            a, b, print_error_context=_recursive_error_context(f'[{i}]'), **recursive_args)
                   for i, (a, b) in enumerate(zip(elem, other)))
    elif hasattr(elem, '__slots__'):
        return contained_torch_equal(_attrs_to_dict(elem, elem.__slots__), _attrs_to_dict(other, other.__slots__),
                                     print_error_context=print_error_context, **recursive_args)
    else:
        return _maybe_print(elem == other)

def check_allclose(tensor, other, rtol=1e-5, atol=1e-8, equal_nan=False):
    if not torch.allclose(tensor, other, atol=atol, rtol=rtol, equal_nan=equal_nan):
        diff_idx = torch.where(~torch.isclose(tensor, other, atol=atol, rtol=rtol, equal_nan=equal_nan))
        raise ValueError(f"Tensors are not close on indices {diff_idx}:",
                         f"Example values: {tensor[diff_idx][:10]} vs {other[diff_idx][:10]}.")

def check_tensor_attribute_shapes(container, throw=True, **attribute_info):
    """Checks shape on all specified attributes of the container.

    Args:
        container (dict, tuple, object): container with named attributes to be tested
        throw (bool): if true (default), will throw error on first check that fails
        attribute_info: named attribute=shape values, where shape can be list or tuple (see `check_tensor`)

    Return:
        (bool) True if checks pass
    """
    def _get_item(container, attr):
        if isinstance(container, collections.abc.Mapping):
            return container[attr]
        else:
            return getattr(container, attr)

    success = True
    for k, shape in attribute_info.items():
        val = _get_item(container, k)
        if not check_tensor(val, shape=shape, throw=False):
            success = False
            message = f'Attribute {k} has shape {val.shape} (expected {shape})'
            if throw:
                raise ValueError(message)
            else:
                logging.error(message)
    return success


def print_namedtuple_attributes(ntuple, name='', prefix='', **tensor_info_kwargs):
    r"""Same as :func:`print_dict_attributes`, but with named tuple input. """
    print_dict_attributes(ntuple._asdict(), name=name, prefix=prefix, **tensor_info_kwargs)


def print_dict_attributes(in_dict, name='', prefix='', **tensor_info_kwargs):
    r"""Convenience function to print all attributes of a dict with names. Extra info
    will be provided for tensors.

    Args:
        in_dict (dict): input dictionary containing any values
        name (str): name of the input dictionary (especially useful if printing nested dictionaries)
        prefix (str): prefix to provide to every printed attributes (especially useful if printing nested dictionaries)
        **tensor_info_kwargs: extra arguments to pass to :func:`tensor_info`
    """
    if len(name) > 0:
        print(f'\nAttributes of {name}:')
    for k, v in in_dict.items():
        recurse = False
        if torch.is_tensor(v):
            tinfo = tensor_info(v, **tensor_info_kwargs)
        elif isinstance(v, (str, int, float)):
            tinfo = v
        elif isinstance(v, collections.abc.Sequence):
            tinfo = f'{type(v)} of length {len(v)}'
        elif isinstance(v, collections.abc.Mapping):
            tinfo = f'{type(v)} of length {len(v)}'
            recurse = True
        else:
            tinfo = type(v)
        print(f'   {prefix}{k}: {tinfo}')
        if recurse:
            print_dict_attributes(v, prefix=f'  ')
