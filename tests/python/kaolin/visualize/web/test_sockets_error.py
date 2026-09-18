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

"""Tests for WebSocket error-forwarding logic in kaolin.visualize.web.sockets.

Covers:
 - WebSocketHandlerManager._send_error_to_client   (payload gating)
 - WebSocketHandlerManager._on_handler_task_done   (async task error surfacing)
 - GlobalWebSocketConnectionManager.send_error_to_tab  (fan-out to tab handlers)
"""

import asyncio
from collections import defaultdict
import json
import pytest
from unittest.mock import MagicMock, patch

from kaolin.visualize.web.sockets import (
    WebSocketHandlerManager,
    GlobalWebSocketConnectionManager,
)


def _make_handler(write_errors_to_client=False, connection_id='test-conn'):
    """Minimal mock of a WebSocketHandlerManager instance."""
    h = MagicMock()
    h._connection_id = connection_id
    h.write_errors_to_client = write_errors_to_client
    h.write_message_safe = MagicMock()
    return h


# ---------------------------------------------------------------------------
# _send_error_to_client
# ---------------------------------------------------------------------------

class TestSendErrorToClient:

    def test_no_message_when_connection_closed(self):
        h = _make_handler(connection_id=None)
        WebSocketHandlerManager._send_error_to_client(h, RuntimeError('boom'))
        h.write_message_safe.assert_not_called()

    def test_message_sent_when_connection_open(self):
        h = _make_handler()
        WebSocketHandlerManager._send_error_to_client(h, RuntimeError('x'))
        h.write_message_safe.assert_called_once()

    def test_generic_payload_without_write_errors(self):
        h = _make_handler(write_errors_to_client=False)
        WebSocketHandlerManager._send_error_to_client(h, RuntimeError('secret details'))
        raw = h.write_message_safe.call_args[0][0]
        msg = json.loads(raw)
        assert msg.get('tag') == 'kaolin_error'
        content = msg.get('msg', {})
        assert 'error_type' not in content
        assert 'message' not in content

    def test_detailed_payload_with_write_errors(self):
        h = _make_handler(write_errors_to_client=True)
        exc = ValueError('exposed detail')
        WebSocketHandlerManager._send_error_to_client(h, exc)
        raw = h.write_message_safe.call_args[0][0]
        content = json.loads(raw).get('msg', {})
        assert content.get('error_type') == 'ValueError'
        assert content.get('message') == 'exposed detail'

    def test_explicit_override_enables_details(self):
        """write_errors_to_client=True kwarg overrides instance flag=False."""
        h = _make_handler(write_errors_to_client=False)
        exc = RuntimeError('override')
        WebSocketHandlerManager._send_error_to_client(
            h, exc, write_errors_to_client=True)
        content = json.loads(h.write_message_safe.call_args[0][0]).get('msg', {})
        assert 'error_type' in content
        assert content['message'] == 'override'

    def test_explicit_override_suppresses_details(self):
        """write_errors_to_client=False kwarg overrides instance flag=True."""
        h = _make_handler(write_errors_to_client=True)
        WebSocketHandlerManager._send_error_to_client(
            h, RuntimeError('secret'), write_errors_to_client=False)
        content = json.loads(h.write_message_safe.call_args[0][0]).get('msg', {})
        assert 'error_type' not in content

    def test_tag_is_not_in_payload(self):
        """tag is for server-side logging only and must not appear in the WS payload."""
        h = _make_handler()
        WebSocketHandlerManager._send_error_to_client(
            h, RuntimeError('x'), tag='render')
        content = json.loads(h.write_message_safe.call_args[0][0]).get('msg', {})
        assert 'tag' not in content

    # --- Traceback field ---

    def test_traceback_not_in_payload_without_write_errors(self):
        """No traceback is sent in non-debug (write_errors=False) mode."""
        h = _make_handler(write_errors_to_client=False)
        WebSocketHandlerManager._send_error_to_client(h, RuntimeError('x'))
        content = json.loads(h.write_message_safe.call_args[0][0]).get('msg', {})
        assert 'traceback' not in content

    def test_traceback_in_payload_with_write_errors(self):
        """Traceback string is included in the payload when write_errors=True."""
        h = _make_handler(write_errors_to_client=True)
        try:
            raise ValueError('tb test')
        except ValueError as exc:
            WebSocketHandlerManager._send_error_to_client(h, exc)
        content = json.loads(h.write_message_safe.call_args[0][0]).get('msg', {})
        assert 'traceback' in content
        tb = content['traceback']
        assert isinstance(tb, str)
        assert 'ValueError' in tb
        assert 'tb test' in tb

    def test_traceback_contains_source_location(self):
        """Traceback must reference the file where the exception was raised."""
        h = _make_handler(write_errors_to_client=True)
        try:
            raise RuntimeError('loc test')
        except RuntimeError as exc:
            WebSocketHandlerManager._send_error_to_client(h, exc)
        tb = json.loads(h.write_message_safe.call_args[0][0])['msg']['traceback']
        # The traceback should name this test file as the raise site.
        assert 'test_sockets_error' in tb


# ---------------------------------------------------------------------------
# _on_handler_task_done
# ---------------------------------------------------------------------------

class TestOnHandlerTaskDone:

    def _make_task(self, cancelled=False, exception=None):
        task = MagicMock()
        task.cancelled.return_value = cancelled
        task.exception.return_value = exception
        return task

    def test_cancelled_task_ignored(self):
        h = MagicMock()
        task = self._make_task(cancelled=True)
        WebSocketHandlerManager._on_handler_task_done(h, task)
        h._send_error_to_client.assert_not_called()

    def test_successful_task_does_nothing(self):
        h = MagicMock()
        task = self._make_task(cancelled=False, exception=None)
        WebSocketHandlerManager._on_handler_task_done(h, task)
        h._send_error_to_client.assert_not_called()

    def test_failed_task_sends_error(self):
        h = MagicMock()
        exc = RuntimeError('async boom')
        task = self._make_task(cancelled=False, exception=exc)
        WebSocketHandlerManager._on_handler_task_done(h, task, tag='some_tag')
        h._send_error_to_client.assert_called_once_with(exc, 'some_tag')

    def test_failed_task_without_tag(self):
        h = MagicMock()
        exc = ValueError('no tag')
        task = self._make_task(cancelled=False, exception=exc)
        WebSocketHandlerManager._on_handler_task_done(h, task)
        h._send_error_to_client.assert_called_once_with(exc, None)


# ---------------------------------------------------------------------------
# GlobalWebSocketConnectionManager.send_error_to_tab
# ---------------------------------------------------------------------------

class TestSendErrorToTab:

    def test_fans_out_to_all_open_handlers(self):
        manager = GlobalWebSocketConnectionManager.instance()
        exc = RuntimeError('fan out')
        h1 = MagicMock()
        h1._connection_id = 'conn-1'
        h2 = MagicMock()
        h2._connection_id = 'conn-2'

        with patch.object(manager, 'get_handlers_by_tab', return_value=[h1, h2]):
            manager.send_error_to_tab('tab-abc', exc, write_errors_to_client=True)

        h1._send_error_to_client.assert_called_once_with(exc, write_errors_to_client=True)
        h2._send_error_to_client.assert_called_once_with(exc, write_errors_to_client=True)

    def test_skips_handlers_with_closed_connection(self):
        manager = GlobalWebSocketConnectionManager.instance()
        exc = RuntimeError('closed')
        closed = MagicMock()
        closed._connection_id = None

        with patch.object(manager, 'get_handlers_by_tab', return_value=[closed]):
            manager.send_error_to_tab('tab-xyz', exc)

        closed._send_error_to_client.assert_not_called()

    def test_write_errors_defaults_to_false(self):
        """Default write_errors_to_client=False must be forwarded to each handler."""
        manager = GlobalWebSocketConnectionManager.instance()
        h = MagicMock()
        h._connection_id = 'conn'

        with patch.object(manager, 'get_handlers_by_tab', return_value=[h]):
            manager.send_error_to_tab('tab', RuntimeError('x'))

        _, kwargs = h._send_error_to_client.call_args
        assert kwargs.get('write_errors_to_client') is False


# ---------------------------------------------------------------------------
# _apply_handlers — synchronous and asynchronous exception paths
# ---------------------------------------------------------------------------

class TestApplyHandlers:
    """Tests for WebSocketHandlerManager._apply_handlers, focusing on the
    exception-capture paths introduced in this MR."""

    def _make_manager(self, handlers_map=None):
        """Create a minimal mock manager that supports _apply_handlers."""
        m = MagicMock()
        m.handlers = handlers_map or {}
        m.tab_uuid = 'tab-test'
        m._send_error_to_client = MagicMock()
        # Wire _on_handler_task_done to the real implementation so the async
        # done-callback path is exercised end-to-end.
        def real_done_callback(task, tag=None):
            return WebSocketHandlerManager._on_handler_task_done(m, task, tag=tag)
        m._on_handler_task_done = real_done_callback
        return m

    def test_sync_handler_exception_calls_send_error(self):
        """Synchronous handler exceptions must be caught and sent to the client."""


        mock_handler = MagicMock()
        mock_handler.on_message.side_effect = RuntimeError('sync boom')

        handlers = defaultdict(list)
        handlers['my_tag'] = [mock_handler]
        m = self._make_manager(handlers)

        asyncio.run(WebSocketHandlerManager._apply_handlers(
            m, {'tag': 'my_tag', 'msg': {}}))

        m._send_error_to_client.assert_called_once()
        exc = m._send_error_to_client.call_args[0][0]
        assert isinstance(exc, RuntimeError)
        assert str(exc) == 'sync boom'

    def test_async_handler_exception_surfaced_via_done_callback(self):
        """Async handler exceptions must trigger _on_handler_task_done → _send_error_to_client."""


        async def failing(tag, content, write_fn):
            raise RuntimeError('async boom')

        mock_handler = MagicMock()
        mock_handler.on_message = failing

        handlers = defaultdict(list)
        handlers['my_tag'] = [mock_handler]
        m = self._make_manager(handlers)

        async def _run():
            await WebSocketHandlerManager._apply_handlers(
                m, {'tag': 'my_tag', 'msg': {}})
            await asyncio.sleep(0)  # let the created task complete

        asyncio.run(_run())

        m._send_error_to_client.assert_called_once()
        exc = m._send_error_to_client.call_args[0][0]
        assert isinstance(exc, RuntimeError)
        assert str(exc) == 'async boom'

    def test_successful_handler_does_not_send_error(self):


        mock_handler = MagicMock()
        mock_handler.on_message.return_value = None

        handlers = defaultdict(list)
        handlers['ok_tag'] = [mock_handler]
        m = self._make_manager(handlers)

        asyncio.run(WebSocketHandlerManager._apply_handlers(
            m, {'tag': 'ok_tag', 'msg': {}}))

        m._send_error_to_client.assert_not_called()

    def test_unknown_tag_does_not_send_error(self):
        """Unknown tags are logged as warnings but must not trigger an error message."""

        m = self._make_manager({})  # no handlers registered

        asyncio.run(WebSocketHandlerManager._apply_handlers(
            m, {'tag': 'unknown', 'msg': {}}))

        m._send_error_to_client.assert_not_called()
