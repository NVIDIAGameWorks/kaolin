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

"""Tests for kaolin.visualize.web.sockets."""

import copy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import pytest
import torch

from kaolin.visualize.web import sockets
from kaolin.visualize.web.sockets import (
    AnyRendererMessageHandler, RemoteRenderingOptions, RenderTimeFps,
    SlidingWindowFps,
)


class TestInteractiveQuality(unittest.TestCase):
    def setUp(self):
        self.now = 10.0
        self.clock = lambda: self.now
        self.clock_patch = patch.object(sockets.time, 'monotonic', self.clock)
        self.clock_patch.start()
        self.addCleanup(self.clock_patch.stop)
        self.normal = RemoteRenderingOptions('jpeg', 80, 1000, {'samples': 8})
        self.low = RemoteRenderingOptions('jpeg', 40, 400, {'samples': 2})
        self.handler = AnyRendererMessageHandler(
            Mock(), device='cpu', rendering_options=self.normal,
            low_rendering_options=self.low)
        self.handler._request_fps = SlidingWindowFps(now=self.clock)
        self.handler._rendering_fps = RenderTimeFps(now=self.clock)
        self.handler._interactive_rendering_fps = RenderTimeFps(now=self.clock)
        self.handler._note_render_request()
        self.assertIs(self.handler._interactive_rendering_options(), self.normal)

    def sample(self, duration, advance=1.1):
        """Supply a fresh window of 10 requests/s and a completed render."""
        self.now += advance
        for offset in (0.2, 0.1, 0.0):
            self.handler._request_fps.event(self.now - offset)
        if duration is not None:
            self.handler._interactive_rendering_fps.record(duration)
        return self.handler._interactive_rendering_options()

    def test_no_low_preset_or_no_timings_keeps_defaults(self):
        self.assertIs(self.sample(None), self.normal)
        self.handler.low_rendering_options = None
        for _ in range(3):
            self.assertIs(self.sample(0.2), self.normal)
        self.assertEqual(self.handler.stats()['low_quality_weight'], 0.0)

    def test_overload_blends_to_low_without_mutating_presets(self):
        presets = copy.deepcopy((self.normal, self.low))
        with self.assertLogs(sockets.logger, level='INFO') as logs:
            options = self.sample(0.2)
        self.assertEqual(options.max_resolution, 850)
        self.assertEqual(options.jpeg_quality, 70)
        self.assertEqual(options.kwargs, self.normal.kwargs)
        self.assertIn('low-quality weight 0.25', logs.output[0])
        self.assertAlmostEqual(self.handler.stats()['interactive_rendering_fps'], 5.0)
        options = self.sample(0.2)
        self.assertEqual(options.kwargs, self.low.kwargs)
        for _ in range(4):
            options = self.sample(0.2)
        self.assertEqual(options, self.low)
        self.assertEqual(self.handler.stats()['low_quality_weight'], 1.0)
        self.assertEqual((self.normal, self.low), presets)

    def test_rate_limit_headroom_and_gradual_recovery(self):
        self.sample(0.2, advance=0.2)
        self.assertEqual(self.handler.stats()['low_quality_weight'], 0.0)
        self.sample(0.2)
        self.sample(0.2, advance=0.2)
        self.assertEqual(self.handler.stats()['low_quality_weight'], 0.25)
        # 11 FPS for 10 requests/s is within the hold band.
        self.sample(1.0 / 11.0)
        self.assertEqual(self.handler.stats()['low_quality_weight'], 0.25)
        self.sample(0.05)
        self.assertAlmostEqual(self.handler.stats()['low_quality_weight'], 0.15)
        for _ in range(3):
            options = self.sample(0.05)
        self.assertIs(options, self.normal)
        self.assertEqual(self.handler.stats()['low_quality_weight'], 0.0)

    def test_new_interaction_resets_quality_and_old_timings(self):
        self.sample(0.2)
        self.now += 0.6
        self.handler._note_render_request()
        self.assertIs(self.handler._interactive_rendering_options(), self.normal)
        self.assertEqual(self.handler.stats()['low_quality_weight'], 0.0)
        self.assertIsNone(self.handler.stats()['interactive_rendering_fps'])

    def test_uncapped_and_discrete_fields_keep_interpolation_semantics(self):
        self.normal.max_resolution = None
        self.low.encode_format = 'png'
        options = self.sample(0.2)
        self.assertIsNone(options.max_resolution)
        self.assertEqual(options.encode_format, 'jpeg')
        options = self.sample(0.2)
        self.assertEqual(options.max_resolution, 400)
        self.assertEqual(options.encode_format, 'png')


class TestRenderAdaptationIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.normal = RemoteRenderingOptions('jpeg', 80, 1000, {'samples': 8})
        self.low = RemoteRenderingOptions('jpeg', 40, 400, {'samples': 2})
        self.high = RemoteRenderingOptions('png', 100, 2000, {'samples': 16})
        self.render = Mock(return_value=torch.zeros((1, 1, 4)))
        self.handler = AnyRendererMessageHandler(
            self.render, device='cpu', tag_suffix='_test',
            rendering_options=self.normal, low_rendering_options=self.low,
            high_rendering_options=self.high)
        self.handler.encode_rendering = Mock(return_value=b'frame')
        self.camera = SimpleNamespace(width=2000, height=1000)
        self.handler.camera = self.camera
        self.write = Mock()

    async def test_requests_use_selected_options_and_track_interactive_time(self):
        selected = RemoteRenderingOptions.interpolate(self.normal, self.low, 0.5)
        with patch.object(self.handler, '_interactive_rendering_options',
                          return_value=selected), patch.object(
                              self.handler, '_schedule_high_res_check') as schedule:
            await self.handler.on_message('render_test', {}, self.write)
        camera = self.render.call_args.args[0]
        self.assertEqual((camera.width, camera.height), (700, 350))
        self.assertEqual(self.render.call_args.kwargs, {'samples': 2})
        self.assertEqual(self.camera.width, 2000)
        self.assertEqual(self.handler.encode_rendering.call_args.args[1], selected)
        self.assertIsNotNone(self.handler.stats()['interactive_rendering_fps'])
        self.write.assert_called_once_with(b'frame', True)
        schedule.assert_called_once_with(self.write)

    async def test_final_frame_uses_high_preset_without_affecting_adaptation(self):
        self.handler._low_quality_weight = 0.75
        await self.handler._maybe_render_and_send(
            self.write, self.low, interactive=True)
        timings = self.handler._interactive_rendering_fps._samples.copy()
        self.handler._request_seq = 3
        with patch.object(sockets.asyncio, 'sleep', new_callable=AsyncMock):
            await self.handler._high_res_after_gap(2, self.write)
            self.assertEqual(self.render.call_count, 1)
            await self.handler._high_res_after_gap(3, self.write)
            await self.handler._high_res_after_gap(3, self.write)
        self.assertEqual(self.render.call_count, 2)
        self.assertIs(self.handler.encode_rendering.call_args.args[1], self.high)
        self.assertEqual(self.render.call_args.args[0].width, 2000)
        self.assertEqual(self.render.call_args.kwargs, {'samples': 16})
        self.assertEqual(self.handler._interactive_rendering_fps._samples, timings)
        self.assertEqual(self.handler.stats()['low_quality_weight'], 0.75)

    async def test_initial_and_direct_frames_keep_defaults_and_skip_adaptation(self):
        self.handler.camera = None
        self.handler._low_quality_weight = 1.0
        camera = Mock()
        camera.to.return_value = self.camera
        with patch.object(sockets.kaolin.render.camera.Camera, 'from_dict',
                          return_value=camera):
            await self.handler.on_message('set_camera_test', {}, self.write)
        self.handler.render_and_send(self.write)
        self.assertEqual(self.render.call_count, 3)  # Includes the sanity render.
        for call in self.render.call_args_list:
            self.assertEqual(call.args[0].width, 1000)
            self.assertEqual(call.kwargs, {'samples': 8})
        for call in self.handler.encode_rendering.call_args_list:
            self.assertIs(call.args[1], self.normal)
        self.assertIsNone(self.handler.stats()['interactive_rendering_fps'])
        self.assertIsNotNone(self.handler.stats()['rendering_fps'])


class TestSlidingWindowFps:
    def test_rate_idle_and_reset(self):
        now = [0.0]
        tracker = SlidingWindowFps(now=lambda: now[0])
        assert tracker.fps() is None and tracker.is_idle()
        tracker.event()
        assert tracker.fps() is None and not tracker.is_idle()
        now[0] = 0.25
        tracker.event()
        assert tracker.fps() == 4.0
        now[0] = 0.8
        assert tracker.is_idle() and tracker.fps() is None
        tracker.event()
        now[0] = 1.05
        tracker.event()
        assert tracker.fps() == pytest.approx(4.0)
        tracker.reset()
        assert tracker.fps() is None and tracker.is_idle()

    def test_window_and_duplicate_timestamps(self):
        now = [1.0]
        tracker = SlidingWindowFps(window_s=1, idle_threshold_s=2, now=lambda: now[0])
        for timestamp in (0, 0.25, 1):
            tracker.event(timestamp)
        assert tracker.fps() == 2.0
        now[0] = 1.1
        assert tracker.fps() == pytest.approx(1 / 0.75)
        now[0] = 3.0
        assert tracker.fps() is None
        tracker.event()
        tracker.event()
        assert tracker.fps() is None


class TestRenderTimeFps:
    def test_durations_window_and_reset(self):
        now = [0.0]
        tracker = RenderTimeFps(window_s=1, now=lambda: now[0])
        assert tracker.fps() is None and tracker.last_duration() is None
        tracker.record(0.1)
        now[0] = 0.5
        tracker.record(0.3)
        assert tracker.fps() == pytest.approx(5.0)
        assert tracker.last_duration() == 0.3
        now[0] = 1.1
        assert tracker.fps() == pytest.approx(1 / 0.3)
        now[0] = 2.0
        assert tracker.fps() is None and tracker.last_duration() is None
        tracker.record(0, timestamp=now[0])
        assert tracker.fps() is None
        tracker.reset()
        assert tracker.fps() is None and tracker.last_duration() is None


class TestGlobalWebSocketConnectionManager:
    def test_connection_lifecycle(self):
        # TODO: use fake connections to check register/get/unregister and snapshots.
        pytest.skip('TODO: implement connection registry checks')

    def test_tab_isolation(self):
        # TODO: register connections for two tabs and check tab-specific lookup.
        pytest.skip('TODO: implement tab isolation checks')


class TestWebSocketHandlerManager:
    def test_handler_factories(self):
        # TODO: use fake factories and TuidCallable arguments to check per-tab instances.
        pytest.skip('TODO: implement handler factory checks')

    def test_message_dispatch(self):
        # TODO: test sync/async handlers with decoded JSON/binary, unknown tags and bad input.
        pytest.skip('TODO: implement message dispatch checks')

    def test_connection_callbacks(self):
        # TODO: check open/close once per handler, safe writes and programmatic set_camera.
        pytest.skip('TODO: implement connection callback checks')


class TestMeshGeometryStreamer:
    def test_mesh_payload(self):
        # TODO: decode a small CPU mesh with normals, UVs and a material/texture.
        pytest.skip('TODO: implement mesh payload checks')

    def test_initial_and_requested_mesh(self):
        # TODO: capture initial/get_mesh writes and ensure unrelated tags are ignored.
        pytest.skip('TODO: implement mesh request checks')


# TODO: extend AnyRendererMessageHandler tests with actual idle-task cancellation,
# real Camera resolution caps, encoding metadata and worker exception handling.
