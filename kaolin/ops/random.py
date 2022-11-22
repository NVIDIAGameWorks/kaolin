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

import random
import math

import numpy as np
import torch
from .spc.uint8 import uint8_to_bits

def manual_seed(torch_seed, random_seed=None, numpy_seed=None):
    """Set the seed for random and torch modules.

    Args:
        torch_seed (int): The desired seed for torch module.
        random_seed (int): The desired seed for random module. Default: ``torch_seed`` value.
        numpy_seed (int): The desired seed for numpy module. Default: ``torch_seed`` value.
    """
    if random_seed is None:
        random_seed = torch_seed
    if numpy_seed is None:
        numpy_seed = torch_seed
    random.seed(random_seed)
    torch.manual_seed(torch_seed)
    np.random.seed(numpy_seed)

def set_state(torch_state, random_state, numpy_state):
    """Set the generator states for generating random numbers.

    Mostly used in pair with :func:`get_state`

    Args:
        torch_state (torch.ByteTensor): the state of torch module.
        random_state (tuple): the state of random module.
        numpy_state (tuple): the state of numpy module.

    Example:
        >>> torch_state, random_state, numpy_state = get_state()
        >>> s = torch.randn((1, 3))
        >>> set_state(torch_state, random_state, numpy_state)
    """
    torch.set_rng_state(torch_state)
    random.setstate(random_state)
    np.random.set_state(numpy_state)

def get_state():
    """Returns the generator states for generating random numbers.

    Mostly used in pair with :func:`set_state`.

    See also:

    * https://pytorch.org/docs/stable/generated/torch.get_rng_state.html#torch.get_rng_state
    * https://docs.python.org/3/library/random.html#random.getstate
    * https://numpy.org/doc/stable/reference/random/generated/numpy.random.set_state.html#numpy.random.set_state

    Returns:
       (torch.ByteTensor, tuple, tuple):
           the states for the corresponding modules (torch, random, numpy).

    Example:
        >>> torch_state, random_state, numpy_state = get_state()
        >>> s = torch.randn((1, 3))
        >>> set_state(torch_state, random_state, numpy_state)
    """
    return torch.get_rng_state(), random.getstate(), np.random.get_state()

def random_shape_per_tensor(batch_size, min_shape=None, max_shape=None):
    """Generate random :attr:`shape_per_tensor`.

    Args:
        batch_size (int): Batch size (first dimension) of the generated tensor.
        min_shape (list, tuple or torch.LongTensor):
            Minimum values for each dimension of generated shapes.
            Default: 1 for each dimensions.
        max_shape (list, tuple or torch.LongTensor):
            maximum values for each dimension of generated shapes.

    Return:
        (torch.LongTensor): A shape_per_tensor (2D).

    Example:
        >>> _ = torch.random.manual_seed(1)
        >>> random_shape_per_tensor(3, min_shape=(4, 4), max_shape=(10, 10))
        tensor([[ 4,  7],
                [ 7,  7],
                [ 8, 10]])
    """
    if min_shape is None:
        min_shape = [1] * len(max_shape)
    output = torch.cat([torch.randint(low_dim, high_dim + 1, size=(batch_size, 1))
                        for low_dim, high_dim in zip(min_shape, max_shape)], dim=1)
    return output

def random_tensor(low, high, shape, dtype=torch.float, device='cpu'):
    """Generate a random tensor.

    Args:
        low (float): the lowest value to be drawn from the distribution.
        high (float): the highest value to be drawn from the distribution.
        shape (list, tuple or torch.LongTensor): the desired output shape.
        dtype (torch.dtype): the desired output dtype. Default: ``torch.float``.
        device (torch.device): the desired output device. Default: 'cpu'

    Return:
        (torch.Tensor): a random generated tensor.

    Example:
        >>> _ = torch.random.manual_seed(1)
        >>> random_tensor(4., 5., (3, 3), dtype=torch.float, device='cpu')
        tensor([[4.7576, 4.2793, 4.4031],
                [4.7347, 4.0293, 4.7999],
                [4.3971, 4.7544, 4.5695]])
    """
    if dtype in (torch.half, torch.float, torch.double):
        output = torch.rand(shape, dtype=dtype, device=device)
        if (low != 0.) or (high != 1.):
            output = output * (high - low) + low
    elif dtype == torch.bool:
        assert (low is None) or (low == 0)
        assert (high is None) or (high == 1)
        output = torch.randint(0, 2, size=shape, dtype=dtype, device=device)
    else:
        output = torch.randint(low, high + 1, size=shape, dtype=dtype, device=device)
    return output

def random_spc_octrees(batch_size, max_level, device='cpu'):
    """Generate random SPC octrees.

    Args:
        batch_size (int): The desired number of octrees.
        max_level (int): The desired max level of the octrees.
        device (torch.device): The desired output device. Default: 'cpu'.

    Return:
        (torch.ByteTensor, torch.IntTensor):

            - A batch of randomly generated octrees.
            - The length of each octree.

    Example:
        >>> _ = torch.random.manual_seed(1)
        >>> random_spc_octrees(2, 3, device='cpu')
        (tensor([ 71, 180, 220,   9, 134,  59,  42, 102, 210, 193, 204, 190, 107,  24,
                104, 151,  13,   7,  18, 107,  16, 154,  57, 110,  19,  22, 230,  48,
                135,  65,  69, 147, 148, 184, 203, 229, 114, 232,  18, 231, 241, 195],
               dtype=torch.uint8), tensor([19, 23], dtype=torch.int32))
    """
    octrees = []
    lengths = []
    for bs in range(batch_size):
        octree_length = 0
        cur_num_nodes = 1
        for i in range(max_level):
            cur_nodes = torch.randint(1, 256, size=(cur_num_nodes,),
                                      dtype=torch.uint8, device=device)
            cur_num_nodes = torch.sum(uint8_to_bits(cur_nodes))
            octrees.append(cur_nodes)
            octree_length += cur_nodes.shape[0]
        lengths.append(octree_length)
    return torch.cat(octrees, dim=0), torch.tensor(lengths, dtype=torch.torch.int32)

def sample_spherical_coords(shape,
                            azimuth_low=0., azimuth_high=math.pi * 2.,
                            elevation_low=0., elevation_high=math.pi * 0.5,
                            device='cpu', dtype=torch.float):
    """Sample spherical coordinates with a uniform distribution.

    Args:
        shape (Sequence): shape of outputs.
        azimuth_low (float, optional): lower bound for azimuth, in radian. Default: 0.
        azimuth_high (float, optional): higher bound for azimuth, in radian. Default: 2 * pi.
        elevation_low (float, optional): lower bound for elevation, in radian. Default: 0.
        elevation_high (float, optional): higher bound for elevation, in radian. Default: pi / 2.
        device (torch.device, optional): device of the output tensor. Default: 'cpu'.
        dtype (torch.dtype, optional): dtype of the output tensor. Default: torch.float.

    Returns:
        (torch.Tensor, torch.Tensor): the azimuth and elevation, both of desired ``shape``.
    """
    low = torch.tensor([
        [azimuth_low], [math.sin(elevation_low)]
    ], device=device, dtype=dtype).reshape(2, *[1 for _ in shape])
    high = torch.tensor([
        [azimuth_high], [math.sin(elevation_high)]
    ], device=device, dtype=dtype).reshape(2, *[1 for _ in shape])

    rand = torch.rand([2, *shape], dtype=dtype, device=device)
    inter_samples = low + rand * (high - low)
    azimuth = inter_samples[0]
    elevation = torch.asin(inter_samples[1])
    return azimuth, elevation
