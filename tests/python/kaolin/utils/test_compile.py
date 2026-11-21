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

import warnings

import pytest
import torch

from kaolin.utils import compile as kal_compile


def _func(x):
    return torch.exp(-x ** 2)


def _eager_warnings(records):
    return [r for r in records if 'eager mode' in str(r.message)]


@pytest.fixture
def compile_unavailable(monkeypatch):
    monkeypatch.setattr(kal_compile, '_compile_error', RuntimeError('unsupported'))


@pytest.mark.parametrize('with_kwargs', [False, True])
def test_fallback_warns_once(compile_unavailable, with_kwargs):
    if with_kwargs:
        f = kal_compile.maybe_compile(fullgraph=True)(_func)
    else:
        f = kal_compile.maybe_compile(_func)
    x = torch.rand(10)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('default')
        out1 = f(x)
        out2 = f(x)
    assert torch.equal(out1, _func(x))
    assert torch.equal(out2, _func(x))
    eager_warnings = _eager_warnings(records)
    assert len(eager_warnings) == 1
    assert 'unsupported' in str(eager_warnings[0].message)
    assert '_func' in str(eager_warnings[0].message)


def test_fallback_warns_once_per_function(compile_unavailable):
    def _other(x):
        return x + 1
    f = kal_compile.maybe_compile(_func)
    g = kal_compile.maybe_compile(_other)
    x = torch.rand(10)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('default')
        f(x)
        g(x)
        f(x)
        g(x)
    assert len(_eager_warnings(records)) == 2


def test_no_warning_when_compile_available(monkeypatch):
    monkeypatch.setattr(kal_compile, '_compile_error', None)
    monkeypatch.setattr(torch, 'compile', lambda func, **kwargs: func)
    f = kal_compile.maybe_compile(fullgraph=True)(_func)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter('default')
        f(torch.rand(10))
    assert f is _func
    assert len(_eager_warnings(records)) == 0
