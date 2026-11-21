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
import functools as _functools
import warnings as _warnings
import torch as _torch

__all__ = ['maybe_compile']

# torch.compile raises on unsupported setups (e.g. torch 2.9 with Python 3.14).
try:
    _torch.compile()
    _compile_error = None
except RuntimeError as e:
    _compile_error = e


def maybe_compile(func=None, **kwargs):
    r"""Decorator applying :func:`torch.compile` when it is supported, falling back to eager otherwise.

    In the fallback case, a warning is emitted the first time each decorated function is called.

    Args:
        func (callable, optional): The function to compile. If None, returns a decorator.
        **kwargs: Arguments forwarded to :func:`torch.compile` (e.g. ``fullgraph=True``).

    Returns:
        (callable): The compiled function, or the eager function if compilation is unsupported.

    Examples:
        >>> @maybe_compile(fullgraph=True)
        ... def f(x):
        ...     return torch.exp(-x ** 2)
    """
    if func is None:
        return _functools.partial(maybe_compile, **kwargs)
    if _compile_error is None:
        return _torch.compile(func, **kwargs)

    @_functools.wraps(func)
    def wrapper(*args, **kw):
        # Shown once per function under the default warnings filter.
        _warnings.warn(f"torch.compile is unavailable ({_compile_error}), "
                       f"{func.__module__}.{func.__qualname__} will run in eager mode and may be slower.")
        return func(*args, **kw)

    return wrapper
