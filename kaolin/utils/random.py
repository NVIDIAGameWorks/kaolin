# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

import torch
import numpy as np
import random

__all__ = [
    'manual_seed',
    'set_state',
    'get_state'
]

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
