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

"""Tests for kaolin.visualize.dash.builtins."""

import pytest

from kaolin.visualize.dash.builtins import BehaviorLibrary
from kaolin.visualize.dash.option import BehaviorMeta, OptionSpec


@pytest.fixture(autouse=True)
def isolated_library(monkeypatch):
    monkeypatch.setattr(BehaviorLibrary, '_user_directories', [])
    monkeypatch.setattr(BehaviorLibrary, '_user_cache', {})
    monkeypatch.setattr(BehaviorLibrary, '_user_cache_invalid', True)


class TestBehaviorLibrary:
    def test_builtin_lookup(self):
        names = BehaviorLibrary.names()
        assert names == sorted(names) and 'drawing' in names
        meta = BehaviorLibrary.meta('drawing')
        assert isinstance(meta, BehaviorMeta)
        assert isinstance(meta.options['color'], OptionSpec)
        assert BehaviorLibrary.is_builtin_behavior('drawing')
        assert not BehaviorLibrary.is_user_behavior('drawing')
        assert BehaviorLibrary.all_meta()['drawing'] is meta
        assert BehaviorLibrary.meta('missing') is None
        assert 'drawing' in str(BehaviorLibrary)
        assert 'color' in BehaviorLibrary.behavior_to_string('drawing', detailed=True)
        with pytest.raises(ValueError, match='not a known behavior'):
            BehaviorLibrary.behavior_to_string('missing')

    def test_user_directory(self, tmp_path):
        asset = tmp_path / 'custom.js'
        asset.write_text('BehaviorRegister.register("custom", Custom);')
        BehaviorLibrary.register_user_directory(tmp_path)
        BehaviorLibrary.register_user_directory(tmp_path)
        assert BehaviorLibrary.names().count('custom') == 1
        assert BehaviorLibrary.is_user_behavior('custom')
        assert not BehaviorLibrary.is_builtin_behavior('custom')
        assert BehaviorLibrary.meta('custom').source_path == str(asset)
        assert 'custom.js' in BehaviorLibrary.behavior_to_string('custom')
        (tmp_path / 'another.js').write_text('BehaviorRegister.register("another", Custom);')
        BehaviorLibrary.invalidate_user_cache()
        assert 'another' in BehaviorLibrary.all_meta()
        assert 'custom' in BehaviorLibrary.to_string(detailed=True)

    @pytest.mark.parametrize('builtin_collision', [True, False])
    def test_duplicate_names(self, tmp_path, builtin_collision):
        first, second = tmp_path / 'first', tmp_path / 'second'
        first.mkdir()
        second.mkdir()
        name = 'drawing' if builtin_collision else 'custom'
        if not builtin_collision:
            (first / 'a.js').write_text(f'BehaviorRegister.register("{name}", Custom);')
            BehaviorLibrary.register_user_directory(first)
        (second / 'b.js').write_text(f'BehaviorRegister.register("{name}", Custom);')
        with pytest.raises(ValueError, match=name):
            BehaviorLibrary.register_user_directory(second)


# TODO: test is_builtin_layer validation when layer metadata is implemented.
