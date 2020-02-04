# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

"""
Several helper functions, for internal use in Kaolin.
"""
import torch
import hashlib
from pathlib import Path
from typing import Callable
import numpy as np


def _composedecorator(*decs):
    """Returns a composition of several decorators.

    Source: https://stackoverflow.com/a/5409569

    Usage::

            @composedec(decorator1, decorator2)
            def func_that_needs_decoration(args):
                pass

        is equavalent to::

            @decorator1
            @decorator2
            def func_that_needs_decoration(args):
                pass

    """

    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f
    return deco


def _normalize_zerosafe(matrix: torch.Tensor):
    """Normalizes each row of a matrix in a 'division by zero'-safe way.

    Args:
        matrix (torch.tensor): Matrix where each row contains a vector
            to be normalized

    """

    assert matrix.dim() == 2, 'Need matrix to contain exactly 2 dimensions'
    magnitude = torch.sqrt(torch.sum(torch.pow(matrix, 2), dim=1))
    valid_inds = magnitude > 0
    matrix[valid_inds] = torch.div(
        matrix[valid_inds], magnitude[valid_inds].unsqueeze(1))
    return matrix


def _assert_tensor(inp):
    """Asserts that the input is of type torch.Tensor. """
    if not torch.is_tensor(inp):
        raise TypeError('Expected input to be of type torch.Tensor.'
                        ' Got {0} instead'.format(type(inp)))


def _assert_dim_gt(inp, tgt):
    """Asserts that the number of dims in inp is greater than the
    value sepecified in tgt. 

    Args:
        inp (torch.Tensor): Input tensor, whose number of dimensions is
            to be compared.
        tgt (int): Value which the number of dims of inp should exceed.
    """
    if inp.dim() <= tgt:
        raise ValueError('Expected input to contain more than {0} dims. '
                         'Got {1} instead.'.format(tgt, inp.dim()))


def _assert_dim_lt(inp, tgt):
    """Asserts that the number of dims in inp is less than the
    value sepecified in tgt. 

    Args:
        inp (torch.Tensor): Input tensor, whose number of dimensions is
            to be compared.
        tgt (int): Value which the number of dims of inp should be less than.
    """
    if not inp.dim() >= tgt:
        raise ValueError('Expected input to contain less than {0} dims. '
                         'Got {1} instead.'.format(tgt, inp.dim()))

def _assert_dim_ge(inp, tgt):
    """Asserts that the number of dims in inp is greater than or equal to the
    value sepecified in tgt. 

    Args:
        inp (torch.Tensor): Input tensor, whose number of dimensions is
            to be compared.
        tgt (int): Value which the number of dims of inp should exceed.
    """
    if inp.dim() < tgt:
        raise ValueError('Expected input to contain at least {0} dims. '
                         'Got {1} instead.'.format(tgt, inp.dim()))


def _assert_dim_le(inp, tgt):
    """Asserts that the number of dims in inp is less than or equal to the
    value sepecified in tgt. 

    Args:
        inp (torch.Tensor): Input tensor, whose number of dimensions is
            to be compared.
        tgt (int): Value which the number of dims of inp should not exceed.
    """
    if inp.dim() > tgt:
        raise ValueError('Expected input to contain at most {0} dims. '
                         'Got {1} instead.'.format(tgt, inp.dim()))


def _assert_dim_eq(inp, tgt):
    """Asserts that the number of dims in inp is exactly equal to the
    value sepecified in tgt. 

    Args:
        inp (torch.Tensor): Input tensor, whose number of dimensions is
            to be compared.
        tgt (int): Value which the number of dims of inp should equal.
    """
    if inp.dim() != tgt:
        raise ValueError('Expected input to contain exactly {0} dims. '
                         'Got {1} instead.'.format(tgt, inp.dim()))


def _assert_shape_eq(inp, tgt_shape, dim=None):
    """Asserts that the shape of tensor `inp` is equal to the tuple `tgt_shape`
    along dimension `dim`. If `dim` is None, shapes along all dimensions must
    be equal.
    """
    if dim is None:
        if inp.shape != tgt_shape:
            raise ValueError('Size mismatch. Input and target have different '
                             'shapes: {0} vs {1}.'.format(inp.shape,
                                tgt_shape))
    else:
        if inp.shape[dim] != tgt_shape[dim]:
            raise ValueError('Size mismatch. Input and target have different '
                             'shapes at dimension {2}: {0} vs {1}.'.format(
                                inp.shape[dim], tgt_shape[dim], dim))


def _assert_gt(inp, val):
    """Asserts that all elements in tensor `inp` are greater than value `val`.
    """
    if not (inp > val).all():
        raise ValueError('Each element of input must be greater '
                         'than {0}.'.format(val))


def _get_hash(x):
    """Generate a hash from a string, or dictionary.
    """
    if isinstance(x, dict):
        x = tuple(sorted(pair for pair in x.items()))

    return hashlib.md5(bytes(str(x), 'utf-8')).hexdigest()


class Cache(object):
    """Caches the results of a function to disk.
    If already cached, data is returned from disk, otherwise,
    the function is executed. Output tensors are always on CPU device.

        Args:
            transforms (Iterable): List of transforms to compose.
            cache_dir (str): Directory where objects will be cached. Default
                             to 'cache'.
    """

    def __init__(self, func: Callable, cache_dir: [str, Path], cache_key: str):
        self.func = func
        self.cache_dir = Path(cache_dir) / str(cache_key)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cached_ids = [p.stem for p in self.cache_dir.glob('*')]

    def __call__(self, unique_id: str, **kwargs):
        """Execute self.func if not cached, otherwise, read data from disk.

            Args:
                unique_id (str): The unique id with which to name the cached file.
                **kwargs: The arguments to be passed to self.func.

            Returns:
                dict of {str: torch.Tensor}: Dictionary of tensors.
        """

        fpath = self.cache_dir / f'{unique_id}.p'

        if not fpath.exists():
            output = self.func(**kwargs)
            self._write(output, fpath)
            self.cached_ids.append(unique_id)
        else:
            output = self._read(fpath)

        # Read file to move tensors to CPU.
        return self._read(fpath)

    def _write(self, x, fpath):
        torch.save(x, fpath)

    def _read(self, fpath):
        return torch.load(fpath, map_location='cpu')
