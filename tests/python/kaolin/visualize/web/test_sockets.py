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

import asyncio
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, call, patch

import numpy as np
import pytest
import torch

from tornado.httputil import HTTPServerRequest
from tornado.web import Application

from kaolin.render.camera import Camera
from kaolin.render.materials import PBRMaterial
from kaolin.rep import SurfaceMesh
from kaolin.visualize.web import io, sockets
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


def make_camera(width=80, height=40):
    return Camera.from_args(view_matrix=torch.eye(4), width=width, height=height,
                            fov=1.0, device='cpu')


def make_manager(specs=(), tab='first'):
    request = HTTPServerRequest(uri=f'/?tab={tab}', connection=Mock())
    manager = sockets.WebSocketHandlerManager(Application(), request,
                                              handler_specs=list(specs))
    manager.write_message = Mock()
    return manager


class TestGlobalWebSocketConnectionManager:
    def test_connection_lifecycle(self):
        registry = sockets.GlobalWebSocketConnectionManager()
        handler = Mock(tab_uuid=None)
        key = registry.register(handler)
        assert key in registry and len(registry) == 1
        assert registry.get(key) is handler
        assert registry.connection_ids() == [key]
        assert registry.items() == [(key, handler)]
        snapshot = registry.connections()
        snapshot.clear()
        assert registry.connections() == [handler]
        registry.unregister(key)
        registry.unregister(key)
        assert registry.get(key) is None and key not in registry
        assert registry.items() == [] and len(registry) == 0

    def test_tab_isolation(self):
        registry = sockets.GlobalWebSocketConnectionManager()
        first, second, other = [Mock(tab_uuid=tab) for tab in ('a', 'a', 'b')]
        keys = [registry.register(handler) for handler in (first, second, other)]
        assert set(registry.get_handlers_by_tab('a')) == {first, second}
        assert registry.get_handlers_by_tab('b') == [other]
        snapshot = registry.get_handlers_by_tab('a')
        registry.unregister(keys[0])
        assert set(snapshot) == {first, second}
        assert registry.get_handlers_by_tab('a') == [second]
        registry.unregister(keys[1])
        assert registry.get_handlers_by_tab('a') == []
        assert registry.get_handlers_by_tab('missing') == []


class TestWebSocketHandlerManager(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = sockets.GlobalWebSocketConnectionManager()
        patcher = patch.object(sockets.GlobalWebSocketConnectionManager,
                               '_singleton', self.registry)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_handler_factories(self):
        factory = Mock(side_effect=lambda *args, **kwargs: Mock(
            accepted_message_tags=lambda: ['first', 'second']))
        spec = (factory, (sockets.TuidCallable(lambda tab: f'arg-{tab}'),),
                {'value': sockets.TuidCallable(lambda tab: f'kw-{tab}')})
        first, second = make_manager([spec], 'a'), make_manager([spec], 'b')
        first.open()
        second.open()
        assert factory.call_args_list == [
            call('arg-a', value='kw-a'),
            call('arg-b', value='kw-b')]
        assert first.handlers['first'][0] is not second.handlers['first'][0]
        assert first.handlers['first'][0] is first.handlers['second'][0]
        first.handlers['first'][0].on_connection_open.assert_called_once()
        assert self.registry.get_handlers_by_tab('a') == [first]
        first.on_close()
        second.on_close()
        assert len(self.registry) == 0

        manager = make_manager([(factory, ())])
        manager.open()
        factory.assert_called_with()
        manager.on_close()
        with self.assertRaisesRegex(ValueError, 'Handler spec'):
            make_manager([(factory,)])._handlers_lazy_initialize()

    async def test_message_dispatch(self):
        manager = make_manager()
        sync_handler = Mock(accepted_message_tags=lambda: ['example'])
        received = asyncio.Event()

        async def receive(*args):
            received.set()

        async_handler = Mock(accepted_message_tags=lambda: ['example'],
                             on_message=AsyncMock(side_effect=receive))
        manager.register_message_handler(sync_handler)
        manager.register_message_handler(async_handler)
        for binary in (False, True):
            received.clear()
            await manager.on_message(io.encode_message('example', {'value': 3}, binary=binary))
            await asyncio.wait_for(received.wait(), 1)
            sync_handler.on_message.assert_called_with('example', {'value': 3}, manager.write_message_safe)
            async_handler.on_message.assert_awaited_with('example', {'value': 3}, manager.write_message_safe)
        with self.assertLogs(sockets.logger, level='WARNING'):
            await manager.on_message(io.encode_message('unknown', {}))
        for malformed in ('not JSON', b''):
            with self.assertLogs(sockets.logger, level='ERROR'):
                await manager.on_message(malformed)
        with self.assertRaisesRegex(ValueError, 'does not contain'):
            await manager.on_message('{}')
        assert sync_handler.on_message.call_count == 2

    async def test_connection_callbacks(self):
        manager = make_manager()
        handler = Mock(accepted_message_tags=lambda: ['set_camera', 'render'])
        manager.register_message_handler(handler)
        manager.open()
        handler.on_connection_open.assert_called_once_with(manager.write_message_safe)
        assert self.registry.get(manager.connection_id) is manager
        manager.write_message.assert_called_once_with({'status': 'ok'}, binary=False)

        received = asyncio.Event()
        handler.on_message.side_effect = lambda *args: received.set()
        camera = make_camera()
        manager.set_camera(camera)
        await asyncio.wait_for(received.wait(), 1)
        tag, content, write = handler.on_message.call_args.args
        assert tag == 'set_camera' and write == manager.write_message_safe
        restored = Camera.from_dict(content)
        torch.testing.assert_close(restored.extrinsics.view_matrix(), camera.extrinsics.view_matrix())
        assert (restored.width, restored.height) == (80, 40)

        manager._io_loop = Mock()
        manager.write_message.reset_mock()
        manager.write_message_safe(b'frame', True)
        manager.write_message.assert_not_called()
        manager._io_loop.add_callback.call_args.args[0]()
        manager.write_message.assert_called_once_with(b'frame', True)
        manager.on_close()
        manager.on_close()
        assert manager.connection_id is None and len(self.registry) == 0


class TestMeshGeometryStreamer:
    @staticmethod
    def mesh():
        texture = torch.zeros((8, 8, 3))
        texture[..., 0] = 1
        return SurfaceMesh(
            vertices=torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]),
            faces=torch.tensor([[0, 1, 2]]),
            face_normals=torch.tensor([[[0., 0., 1.]] * 3]),
            face_uvs=torch.tensor([[[0., 0.], [1., 0.], [0., 1.]]]),
            material_assignments=torch.tensor([0]),
            materials=[PBRMaterial(diffuse_color=(1., 0., 0.), diffuse_texture=texture)])

    def test_mesh_payload(self):
        mesh = self.mesh()
        decoded = io.from_binary(sockets.MeshGeometryStreamer(mesh).encode_mesh())
        assert decoded['tag'] == 'set_mesh'
        content = decoded[io.MESSAGE_CONTENT_KEY]
        for name in ('face_vertices', 'face_normals', 'face_uvs', 'material_assignments'):
            np.testing.assert_array_equal(content[name], getattr(mesh, name).numpy())
        material = content['materials'][0]
        np.testing.assert_array_equal(material['diffuse_color'], [1., 0., 0.])
        assert material['diffuse_texture'].shape == (8, 8, 4)
        np.testing.assert_array_equal(material['diffuse_texture'][0, 0], [255, 0, 0, 255])

    def test_initial_and_requested_mesh(self):
        handler = sockets.MeshGeometryStreamer(self.mesh())
        write = Mock()
        assert handler.accepted_message_tags() == ['get_mesh']
        handler.on_connection_open(write)
        handler.on_message('unknown', {}, write)
        assert write.call_count == 1
        handler.on_message('get_mesh', {}, write)
        assert write.call_count == 2
        for call in write.call_args_list:
            assert call.args[1] is True
            assert io.from_binary(call.args[0])['tag'] == 'set_mesh'


class TestRendererOutput:
    @pytest.mark.parametrize('width,height,cap,expected', [
        (80, 40, 20, (20, 10)), (40, 80, 20, (10, 20)),
        (8, 4, 20, (8, 4)), (80, 40, None, (80, 40)),
    ])
    def test_camera_resolution_cap(self, width, height, cap, expected):
        camera = make_camera(width, height)
        original = copy.deepcopy(camera)
        handler = AnyRendererMessageHandler(Mock(), device='cpu')
        resized = handler._scale_camera(camera, RemoteRenderingOptions(max_resolution=cap))
        assert (resized.width, resized.height) == expected
        assert (camera.width, camera.height) == (width, height)
        torch.testing.assert_close(camera.intrinsics.params, original.intrinsics.params)
        torch.testing.assert_close(resized.extrinsics.view_matrix(), original.extrinsics.view_matrix())

    @pytest.mark.parametrize('image_format', ['raw', 'png', 'jpeg'])
    def test_encoded_frame_and_metadata(self, image_format):
        handler = AnyRendererMessageHandler(Mock(), tag_suffix='_test', device='cpu')
        frame = torch.full((8, 12, 3), 128, dtype=torch.uint8)
        options = RemoteRenderingOptions(encode_format=image_format)
        write = Mock()
        handler.send_render_message({'img': frame, 'frame_id': 7}, write, options)
        encoded, binary = write.call_args.args
        assert binary is True
        decoded = io.from_binary(encoded)
        assert decoded['tag'] == 'render_test' and decoded[io.MESSAGE_CONTENT_KEY]['frame_id'] == 7
        image = decoded[io.MESSAGE_CONTENT_KEY]['img']
        assert image.shape[:2] == (8, 12)
        np.testing.assert_allclose(image[..., :3], frame.numpy(), atol=2)


class TestRendererScheduling(unittest.IsolatedAsyncioTestCase):
    async def test_new_request_suppresses_old_idle_frame(self):
        handler = AnyRendererMessageHandler(Mock(), device='cpu',
            high_rendering_options=RemoteRenderingOptions(max_resolution=200))
        handler.camera = make_camera()
        handler._maybe_render_and_send = AsyncMock()
        entered = asyncio.Queue()
        release = asyncio.Event()

        async def wait_gap(delay):
            assert delay == handler._high_res_gap_s
            entered.put_nowait(None)
            await release.wait()

        tasks = []
        create_task = asyncio.create_task

        def schedule(coroutine):
            task = create_task(coroutine)
            tasks.append(task)
            return task

        with patch.object(sockets.asyncio, 'sleep', side_effect=wait_gap), patch.object(
                sockets.asyncio, 'create_task', side_effect=schedule):
            await handler.on_message('render', {}, Mock())
            await asyncio.wait_for(entered.get(), 1)
            await handler.on_message('render', {}, Mock())
            await asyncio.wait_for(entered.get(), 1)
            assert len(tasks) == 2
            release.set()
            await asyncio.wait_for(asyncio.gather(*tasks), 1)
        calls = handler._maybe_render_and_send.call_args_list
        assert len(calls) == 3  # Two interactive frames, one final frame.
        assert calls[-1].args[1] is handler.high_rendering_options

        handler.high_rendering_options = None
        with patch.object(sockets.asyncio, 'create_task') as create_task:
            await handler.on_message('render', {}, Mock())
            create_task.assert_not_called()

    async def test_worker_failure_releases_queue_and_next_frame_succeeds(self):
        render = Mock(side_effect=[RuntimeError('render failed'), torch.zeros((8, 8, 4))])
        handler = AnyRendererMessageHandler(render, device='cpu',
            rendering_options=RemoteRenderingOptions(encode_format='raw'))
        handler.camera = make_camera()
        write = Mock()
        with self.assertRaisesRegex(RuntimeError, 'render failed'):
            await handler._maybe_render_and_send(write, handler.rendering_options)
        assert handler._queue.empty() and handler._queue.unfinished_tasks == 0
        write.assert_not_called()
        await handler._maybe_render_and_send(write, handler.rendering_options)
        assert handler._queue.empty() and handler._queue.unfinished_tasks == 0
        assert io.from_binary(write.call_args.args[0])['tag'] == 'render'
        handler._execute_rendering_task(write)  # Empty queue is a no-op.
        write.assert_called_once()
        for _ in range(2):
            handler._queue.put((handler.rendering_options, False))
        await handler._maybe_render_and_send(write, handler.rendering_options)
        assert handler._queue.qsize() == 2
        assert render.call_count == 2
        for _ in range(2):
            handler._queue.get_nowait()
            handler._queue.task_done()
