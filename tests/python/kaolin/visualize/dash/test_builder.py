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

"""Simple configuration tests for kaolin.visualize.dash.builder."""

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from dash import dcc

from kaolin.visualize.dash import builder as builder_module
from kaolin.visualize.dash.builder import SessionRegistry, WebappBuilder
from kaolin.visualize.web.sockets import TuidCallable


@pytest.fixture(autouse=True)
def isolated_sessions(monkeypatch):
    monkeypatch.setattr(SessionRegistry, '_entries', {})


class TestSessionRegistry:
    def test_session_lifecycle(self):
        defaults = {'values': [1]}
        first = SessionRegistry.get_or_create('first', defaults)
        second = SessionRegistry.get_or_create('second', defaults)
        first['values'].append(2)
        assert defaults == second == {'values': [1]}
        assert SessionRegistry.get_or_create('first', {}) is first
        updated = SessionRegistry.get_or_create_and_update(
            'first', defaults, lambda session: session.update(color='red'))
        assert updated is first and first['color'] == 'red'
        assert SessionRegistry.get('first') is first
        assert SessionRegistry.get_or_create_and_update('second', defaults) is second
        SessionRegistry.remove('first')
        SessionRegistry.remove('first')
        assert SessionRegistry.get('first') is None


class TestWebappBuilder:
    def test_configuration(self):
        builder = WebappBuilder(debug=False, default_theme='theme.css', set_favicon=False)
        builder.add_extra_stylesheets(['custom.css'])
        builder.add_extra_scripts(['custom.js'])
        builder.add_raw_body_html('<div>Loading</div>')
        builder.add_importmap_item('renderer', '/renderer.js', 'renderer')
        helper = SimpleNamespace(stylesheets=lambda: ['layout.css'])
        builder.set_layout_helper(helper)
        assert not builder.debug and not builder.set_favicon
        assert builder.stylesheets() == ['theme.css', 'custom.css', 'layout.css']
        assert builder.external_scripts() == ['custom.js']
        assert builder.extra_raw_html == ['<div>Loading</div>']
        assert builder.importmap['renderer'].url == '/renderer.js'
        assert builder.importmap['renderer'].import_as_global == 'renderer'
        assert builder.layout_helper is helper
        assert builder.app is None and builder.server is None

    def test_static_file_mapping(self, tmp_path):
        builder = WebappBuilder(debug=False)
        first, second = tmp_path / 'first.txt', tmp_path / 'second.txt'
        first.write_text('first')
        second.write_text('second')
        url = builder.add_static_file(str(first), 'data.txt')
        assert url == 'kaolin_custom/data.txt'
        assert builder.served_files[url] == str(first.resolve())
        assert builder.add_static_file(str(first), 'data.txt') == url
        with pytest.raises(ValueError, match='already mapped'):
            builder.add_static_file(str(second), 'data.txt')
        with pytest.raises(AssertionError, match='Not a file'):
            builder.add_static_file(str(tmp_path / 'missing'))

    def test_user_settings(self, monkeypatch):
        # Record callback declarations without starting Dash or a web server.
        register = Mock(return_value=lambda function: function)
        monkeypatch.setattr(builder_module, 'callback', register)

        @dataclass
        class Settings:
            size: int = 3

        builder = WebappBuilder(debug=False)
        components = builder.add_user_settings(Settings(), 'brush')
        assert len(components) == 2
        store = components[-1]
        assert isinstance(store, dcc.Store)
        assert store.data == {'size': 3} and store.storage_type == 'session'
        assert register.call_count == 2
        assert isinstance(builder.get_user_settings_func('brush'), TuidCallable)
        with pytest.raises(ValueError, match='identical name'):
            builder.add_user_settings(Settings(), 'brush')
        with pytest.raises(ValueError, match='have not been registered'):
            builder.get_user_settings_func('missing')


# TODO: regression-test build(), static HTTP routes, importmap execution and
# settings synchronization in a running app. Check add_static_dir's reserved-path
# condition and route prefixes before asserting the intended directory behavior.
