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


from itertools import product
import torch

__all__ = [
    'uint8_to_bits',
    'uint8_bits_sum',
    'bits_to_uint8'
]

global _uint8_to_bits_luts
_uint8_to_bits_luts = {}

def uint8_to_bits(uint8_t):
    """Convert uint8 ByteTensor to binary BoolTensor.

    Args:
        uint8_t (torch.ByteTensor): Tensor to convert.

    Returns:
        (BoolTensor):
            Converted tensor of same shape + last dimension 8
            and device than `uint8_t`.

    Examples:
        >>> uint8_t = torch.ByteTensor([[3, 5], [16, 2]])
        >>> uint8_to_bits(uint8_t)
        tensor([[[ True,  True, False, False, False, False, False, False],
                 [ True, False,  True, False, False, False, False, False]],
        <BLANKLINE>
                [[False, False, False, False,  True, False, False, False],
                 [False,  True, False, False, False, False, False, False]]])
    """
    # TODO(cfujitsang): This is a naive implementation
    global _uint8_to_bits_luts
    device = uint8_t.device
    if device not in _uint8_to_bits_luts:
        # flip is for converting to left-to-right binary
        lut = torch.flip(
            torch.tensor(list(product([False, True], repeat=8)),
                         dtype=torch.bool, device=device),
            dims=(1,)).contiguous()
        _uint8_to_bits_luts[device] = lut
    else:
        lut = _uint8_to_bits_luts[device]
    return lut[uint8_t.long()]

global _uint8_bits_sum_luts
_uint8_bits_sum_luts = {}

def uint8_bits_sum(uint8_t):
    """Compute the bits sums for each byte in ByteTensor.

    Args:
        uint8_t (torch.ByteTensor): Tensor to process.

    Return:
        (torch.LongTensor): Output of same shape and device than `uint8_t`.

    Examples:
        >>> uint8_t = torch.ByteTensor([[255, 2], [3, 40]])
        >>> uint8_bits_sum(uint8_t)
        tensor([[8, 1],
                [2, 2]])
    """
    global _uint8_bits_sum_luts
    device = uint8_t.device
    if device not in _uint8_bits_sum_luts:
        base_bits = torch.tensor(list(product([False, True], repeat=8)),
                                 dtype=torch.bool, device=device)
        lut = torch.sum(base_bits, dim=1)
        _uint8_bits_sum_luts[device] = lut
    else:
        lut = _uint8_bits_sum_luts[device]
    return lut[uint8_t.long()]

global _bool_to_uint8_w
_bool_to_uint8_w = {}

def bits_to_uint8(bool_t):
    """Convert uint8 ByteTensor to binary BoolTensor.

    Args:
        bool_t (torch.BoolTensor): Tensor to convert, of last dimension 8.

    Return:
        (torch.LongTensor):
            Converted tensor of same shape[:-1] and device than `bool_t`.

    Examples:
        >>> bool_t = torch.tensor(
        ... [[[1, 1, 0, 0, 0, 0, 0, 0],
        ...   [1, 0, 1, 0, 0, 0, 0, 0]],
        ...  [[0, 0, 0, 0, 1, 0, 0, 0],
        ...   [0, 1, 0, 0, 0, 0, 0, 0]]])
        >>> bits_to_uint8(bool_t)
        tensor([[ 3,  5],
                [16,  2]], dtype=torch.uint8)
    """
    # TODO(cfujitsang): This is a naive implementation
    global _bool_to_uint8_w
    device = bool_t.device
    if device not in _bool_to_uint8_w:
        weights = 2 ** torch.arange(8, device=device,
                                    dtype=torch.long)
        _bool_to_uint8_w[device] = weights
    else:
        weights = _bool_to_uint8_w[device]
    return torch.sum(bool_t * weights.reshape(*([1] * (bool_t.dim() - 1)), 8),
                     dim=-1).byte()
