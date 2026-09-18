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

"""Tests for WebappBuilder error-propagation infrastructure."""

import pytest
from unittest.mock import MagicMock, patch

from dash.exceptions import PreventUpdate

from kaolin.visualize.dash.builder import WebappBuilder


def _make_builder(debug=False, write_errors=False, tmp_path=None):
    """Instantiate a WebappBuilder with patched log setup and Dash callback registry."""
    log_dir = str(tmp_path) if tmp_path is not None else None
    with patch('kaolin.visualize.dash.builder.callback',
               side_effect=lambda *a, **kw: (lambda f: f)):
        builder = WebappBuilder(debug=debug, log_dir=log_dir)
    builder._write_errors_to_client = write_errors
    return builder


class TestErrorPropagatingCallback:

    def test_normal_return_passes_through(self, tmp_path):
        """Successful callback returns its value unchanged."""
        builder = _make_builder(tmp_path=tmp_path)
        with patch('kaolin.visualize.dash.builder.callback',
                   side_effect=lambda *a, **kw: (lambda f: f)):
            @builder.callback()
            def fn():
                return 'ok'

        assert fn(None) == 'ok'  # None = tab_uuid injected as last arg

    def test_non_debug_sends_to_tab_and_raises_prevent_update(self, tmp_path):
        """Non-debug: exception → targeted WS error + PreventUpdate."""
        builder = _make_builder(debug=False, tmp_path=tmp_path)
        mock_manager = MagicMock()
        with patch('kaolin.visualize.dash.builder.callback',
                   side_effect=lambda *a, **kw: (lambda f: f)), \
             patch('kaolin.visualize.web.sockets.GlobalWebSocketConnectionManager.instance',
                   return_value=mock_manager):
            @builder.callback()
            def failing():
                raise ValueError('error')

            with pytest.raises(PreventUpdate):
                failing('tab-abc')

        mock_manager.send_error_to_tab.assert_called_once()
        args, _ = mock_manager.send_error_to_tab.call_args
        assert args[0] == 'tab-abc'

    def test_no_ws_send_without_tab_uuid(self, tmp_path):
        """When tab_uuid is None, WS send is skipped."""
        builder = _make_builder(debug=False, tmp_path=tmp_path)
        mock_manager = MagicMock()
        with patch('kaolin.visualize.dash.builder.callback',
                   side_effect=lambda *a, **kw: (lambda f: f)), \
             patch('kaolin.visualize.web.sockets.GlobalWebSocketConnectionManager.instance',
                   return_value=mock_manager):
            @builder.callback()
            def failing():
                raise ValueError('error')

            with pytest.raises(PreventUpdate):
                failing(None)

        mock_manager.send_error_to_tab.assert_not_called()

    def test_debug_reraises_without_ws_relay(self, tmp_path):
        """Debug: re-raises for Dash debug pane; WS relay skipped to avoid duplicate."""
        builder = _make_builder(debug=True, tmp_path=tmp_path)
        mock_manager = MagicMock()
        with patch('kaolin.visualize.dash.builder.callback',
                   side_effect=lambda *a, **kw: (lambda f: f)), \
             patch('kaolin.visualize.web.sockets.GlobalWebSocketConnectionManager.instance',
                   return_value=mock_manager):
            @builder.callback()
            def failing():
                raise ValueError('debug error')

            with pytest.raises(ValueError, match='debug error'):
                failing('tab-debug')

        mock_manager.send_error_to_tab.assert_not_called()

    def test_write_errors_forwarded(self, tmp_path):
        """write_errors_to_client flag is forwarded to send_error_to_tab."""
        builder = _make_builder(debug=False, write_errors=True, tmp_path=tmp_path)
        mock_manager = MagicMock()
        with patch('kaolin.visualize.dash.builder.callback',
                   side_effect=lambda *a, **kw: (lambda f: f)), \
             patch('kaolin.visualize.web.sockets.GlobalWebSocketConnectionManager.instance',
                   return_value=mock_manager):
            @builder.callback()
            def failing():
                raise RuntimeError('detail')

            with pytest.raises(PreventUpdate):
                failing('tab-write')

        _, kwargs = mock_manager.send_error_to_tab.call_args
        assert kwargs.get('write_errors_to_client') is True


# ---------------------------------------------------------------------------
# unsafe_* API contracts
# ---------------------------------------------------------------------------

class TestUsafeAPI:

    def test_unsafe_write_websocket_errors_requires_debug(self, tmp_path):
        builder = WebappBuilder(debug=False, log_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match='requires debug=True'):
            builder.unsafe_write_websocket_errors()

    def test_unsafe_write_websocket_errors_sets_flag(self, tmp_path):
        with patch('kaolin.visualize.dash.builder.callback',
                   side_effect=lambda *a, **kw: (lambda f: f)):
            builder = WebappBuilder(debug=True, log_dir=str(tmp_path))
        builder.unsafe_write_websocket_errors()
        assert builder._write_errors_to_client is True

    def test_unsafe_enable_log_download_requires_debug(self, tmp_path):
        builder = WebappBuilder(debug=False, log_dir=str(tmp_path))
        with pytest.raises(RuntimeError, match='requires debug=True'):
            builder.unsafe_enable_log_download()

    def test_unsafe_enable_log_download_sets_config(self, tmp_path):
        with patch('kaolin.visualize.dash.builder.callback',
                   side_effect=lambda *a, **kw: (lambda f: f)):
            builder = WebappBuilder(debug=True, log_dir=str(tmp_path))
        builder.unsafe_enable_log_download(url='/my-logs', auto_add_download_button=True)
        assert builder._log_download_config == ('/my-logs', True)


# ---------------------------------------------------------------------------
# Builder wiring — overlay injection and log-download button
# ---------------------------------------------------------------------------

class TestBuilderWiring:

    def _build_minimal_app(self, tmp_path, debug=False):
        """Create a Dash app via _make_builder and attach a trivial layout."""
        from dash import Dash, html
        import dash_bootstrap_components as dbc
        from kaolin.visualize.dash.layout import StandardLayoutHelper

        with patch('kaolin.visualize.dash.builder.callback',
                   side_effect=lambda *a, **kw: (lambda f: f)):
            builder = WebappBuilder(debug=debug, log_dir=str(tmp_path))

        app = Dash(external_stylesheets=[dbc.themes.LUX])
        title = html.Span('Test')
        layout_helper = StandardLayoutHelper(title)
        builder.set_layout_helper(layout_helper)
        app.layout = layout_helper.layout()
        return builder, app

    def test_setup_error_overlay_injects_component(self, tmp_path):
        """_setup_error_overlay must add a KaolinErrorOverlay to the layout."""
        from kaolin.visualize.dash.components.autogen.KaolinErrorOverlay import KaolinErrorOverlay

        builder, app = self._build_minimal_app(tmp_path)
        with patch('kaolin.visualize.dash.builder.callback',
                   side_effect=lambda *a, **kw: (lambda f: f)):
            builder._setup_error_overlay(app)

        # Walk the layout tree (which may be a list at the root) to find the component.
        def find_component(node, cls):
            if isinstance(node, list):
                for item in node:
                    result = find_component(item, cls)
                    if result is not None:
                        return result
                return None
            if isinstance(node, cls):
                return node
            children = getattr(node, 'children', None) or []
            if not isinstance(children, list):
                children = [children]
            for child in children:
                result = find_component(child, cls)
                if result is not None:
                    return result
            return None

        overlay = find_component(app.layout, KaolinErrorOverlay)
        assert overlay is not None, 'KaolinErrorOverlay not found in layout'
        assert overlay.id == '_kaolin-error-overlay'
        assert overlay.debug == builder.debug

    def test_relay_callback_raises_with_error_info(self, tmp_path):
        """The WS→Dash debug-pane relay callback must re-raise with error_type and message."""
        from dash import Dash, html
        import dash_bootstrap_components as dbc
        from kaolin.visualize.dash.layout import StandardLayoutHelper

        with patch('kaolin.visualize.dash.builder.callback',
                   side_effect=lambda *a, **kw: (lambda f: f)):
            builder = WebappBuilder(debug=True, log_dir=str(tmp_path))

        app = Dash(external_stylesheets=[dbc.themes.LUX])
        lh = StandardLayoutHelper(html.Span('T'))
        builder.set_layout_helper(lh)
        app.layout = lh.layout()

        # Capture the relay callback by intercepting @callback registration.
        relay_fn = None
        def _capture_callback(*args, **kwargs):
            def decorator(fn):
                nonlocal relay_fn
                relay_fn = fn
                return fn
            return decorator

        with patch('kaolin.visualize.dash.builder.callback', side_effect=_capture_callback):
            builder._setup_ws_error_debug_relay(app)

        assert relay_fn is not None, 'relay callback was not registered'
        with pytest.raises(RuntimeError, match=r'\[WS handler\] ValueError: boom'):
            relay_fn({'error_type': 'ValueError', 'message': 'boom', 'traceback': ''})

    def test_relay_callback_includes_traceback(self, tmp_path):
        """Relay exception message must embed the server-side traceback string."""
        from dash import Dash, html
        import dash_bootstrap_components as dbc
        from kaolin.visualize.dash.layout import StandardLayoutHelper

        with patch('kaolin.visualize.dash.builder.callback',
                   side_effect=lambda *a, **kw: (lambda f: f)):
            builder = WebappBuilder(debug=True, log_dir=str(tmp_path))

        app = Dash(external_stylesheets=[dbc.themes.LUX])
        builder.set_layout_helper(StandardLayoutHelper(html.Span('T')))
        app.layout = builder.layout_helper.layout()

        relay_fn = None
        def _capture(* a, **kw):
            def d(fn):
                nonlocal relay_fn
                relay_fn = fn
                return fn
            return d

        with patch('kaolin.visualize.dash.builder.callback', side_effect=_capture):
            builder._setup_ws_error_debug_relay(app)

        tb = 'Traceback (most recent call last):\n  File "handler.py", line 42\nRuntimeError: boom'
        with pytest.raises(RuntimeError) as exc_info:
            relay_fn({'error_type': 'RuntimeError', 'message': 'boom', 'traceback': tb})

        assert 'Server traceback' in str(exc_info.value)
        assert 'handler.py' in str(exc_info.value)

    def test_relay_callback_raises_prevent_update_on_empty_data(self, tmp_path):
        """Relay must silently no-op (PreventUpdate) when data is None or empty."""
        from dash import Dash, html
        import dash_bootstrap_components as dbc
        from kaolin.visualize.dash.layout import StandardLayoutHelper

        with patch('kaolin.visualize.dash.builder.callback',
                   side_effect=lambda *a, **kw: (lambda f: f)):
            builder = WebappBuilder(debug=True, log_dir=str(tmp_path))

        app = Dash(external_stylesheets=[dbc.themes.LUX])
        builder.set_layout_helper(StandardLayoutHelper(html.Span('T')))
        app.layout = builder.layout_helper.layout()

        relay_fn = None
        def _capture(*a, **kw):
            def d(fn):
                nonlocal relay_fn
                relay_fn = fn
                return fn
            return d

        with patch('kaolin.visualize.dash.builder.callback', side_effect=_capture):
            builder._setup_ws_error_debug_relay(app)

        with pytest.raises(PreventUpdate):
            relay_fn(None)

    def test_add_log_download_button_adds_to_navbar(self, tmp_path):
        """add_log_download_button must inject an anchor into navbar_content."""
        from dash import html
        from kaolin.visualize.dash.layout import StandardLayoutHelper

        layout_helper = StandardLayoutHelper(html.Span('Test'))
        initial_count = len(layout_helper.navbar_content.children)
        layout_helper.add_log_download_button('/_my_logs')

        new_children = layout_helper.navbar_content.children
        assert len(new_children) == initial_count + 1
        btn = new_children[-1]
        assert isinstance(btn, html.A)
        assert btn.href == '/_my_logs'
