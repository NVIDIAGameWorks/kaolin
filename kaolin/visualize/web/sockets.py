from __future__ import annotations, print_function

import argparse
import asyncio
from collections import defaultdict
import copy
from dataclasses import dataclass
from datetime import datetime
import functools
import logging
import queue
import os
import sys
import time
import warnings
import io

import torch
import threading
import flask
from flask import Flask, render_template, Response
from tornado.wsgi import WSGIContainer
from tornado.web import Application, FallbackHandler
from tornado.ioloop import IOLoop
import PIL
import numpy as np
import re
import random
import json

import tornado.websocket
import tornado.gen
from abc import ABC, abstractmethod

import kaolin
import kaolin.visualize.web.io
# Bind the constant via the import machinery (not an attribute walk through the
# `kaolin` package) so the dataclass default below resolves even when this
# module is imported while the top-level `kaolin` package is still initializing.
from kaolin.visualize.web.io import _DEFAULT_JPEG_QUALITY
from kaolin.visualize.web.naming import UniqueIdGenerator

logger = logging.getLogger(__name__)

from typing import Protocol, Dict, Any, Optional, Union, runtime_checkable, Callable

class TuidCallable:
    """Wrap arguments to handlers in this to return a fresh per-client instance
    (handled by WebAppBuilder).
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, tuid):
        return self.func(tuid)


@runtime_checkable
class MessageHandlerProtocol(Protocol):
    """Protocol for WebSocket message handlers."""

    def accepted_message_tags(self) -> list[str]:
        """Return list of message tags this handler responds to."""
        ...

    def on_connection_open(self, write_message_fn: Callable[[Union[bytes, str, Dict[str, Any]], bool], Any]):
        """Called when WebSocket connection opens."""
        ...

@runtime_checkable
class SyncMessageHandlerProtocol(MessageHandlerProtocol, Protocol):
    """Requires synchronous message handling."""

    def on_message(self, message_tag: str, message_content: Dict,
                   write_message_fn: Callable[[Union[bytes, str, Dict[str, Any]], bool], Any]):
        """Handle incoming message.

        Args:

            message_tag:
            message_content: The message content to handle (bytes or string)
            write_message_fn:
        """
        ...

@runtime_checkable
class AsyncMessageHandlerProtocol(MessageHandlerProtocol, Protocol):
    """Requires asynchronous message handling."""

    async def on_message(self, message_tag: str, message_content: Dict,
                         write_message_fn: Callable[[Union[bytes, str, Dict[str, Any]], bool], Any]):
        ...


class MeshGeometryStreamer(SyncMessageHandlerProtocol):
    """Streams mesh geometry to connected WebSocket clients.
    
    Sends mesh data (vertices, normals, UVs, materials) on connection open
    and on 'get_mesh' requests.
    
    Args:
        mesh: The SurfaceMesh to stream.
    """

    def __init__(self, mesh: kaolin.rep.SurfaceMesh):
        self.mesh = mesh
        self.write_msg = None

    # TODO: this is clunky; needs iteration
    def accepted_message_tags(self) -> list[str]:
        return ['get_mesh']

    def on_connection_open(self, write_message_fn: Callable[[Union[bytes, str, Dict[str, Any]], bool], Any]):
        self.write_msg = write_message_fn
        self.write_msg(self.encode_mesh(), True)

    # TODO: this is clunky; needs iteration
    def on_message(self, message_tag: str, message_content: Dict,
                   write_message_fn: Callable[[Union[bytes, str, Dict[str, Any]], bool], Any]):
        if message_tag == 'get_mesh':
            self.write_msg(self.encode_mesh(), True)

    # TODO: move to separate functions
    def convert_material(self, m):
        """Convert PBRMaterial to dict suitable for WebSocket transmission."""
        # TODO: this is brittle with basically no support for anything interesting --> fix and test
        raw_dict = m.hwc().as_dict()
        kaolin.utils.log.log_tensor(raw_dict, 'mat dict orig:', logger, print_stats=True)
        res = {'classname': raw_dict['classname']}
        for k in kaolin.render.materials.PBRMaterial.supported_tensor_attributes():
            val = getattr(m, k, None)
            if val is not None:
                if k in kaolin.render.materials.PBRMaterial.supported_texture_attributes():
                    res[k] = (val * 255).clip(0,255).to(torch.uint8)
                    if res[k].shape[-1] == 3:
                        res[k] = torch.cat([res[k], torch.ones_like(res[k][..., :1]) * 255], dim=-1)  # must be rgba due to webgpu
                else:
                    res[k] = val

                # HACK
                # if 'diffuse' not in k:
                #     del res[k]
                # END OF HACK
        kaolin.utils.log.log_tensor(res, 'mat dict for encoding:', logger, print_stats=True)
        return res

    def encode_mesh(self):
        """Encode mesh as binary message for transmission."""
        #mesh_properties = {'vertices': self.mesh.vertices, 'faces': self.mesh.faces.to(torch.uint32)}
        # sending unindexed geometry, b/c threejs seems to use global index for all of faces, uvs, normals, which
        # is not the case for typical formats read from disk or generated.
        mesh_properties = {'face_vertices': self.mesh.face_vertices}
        if self.mesh.has_attribute('normals') or self.mesh.has_attribute('face_normals'):
            if self.mesh.has_or_can_compute_attribute('face_normals'):
                mesh_properties['face_normals'] = self.mesh.face_normals

        if self.mesh.has_or_can_compute_attribute('face_uvs'):
            mesh_properties['face_uvs'] = self.mesh.face_uvs

        if self.mesh.material_assignments is not None:
            mesh_properties['material_assignments'] = self.mesh.material_assignments.to(torch.int8)  # Note: assuming don't need much precision

        if self.mesh.materials:  # excludes None and empty list
            mesh_properties['materials'] = [self.convert_material(m) for m in self.mesh.materials]
        # TODO: add materials too
        return kaolin.visualize.web.io.encode_message(f'set_mesh', mesh_properties, binary=True)


# Gap (seconds) between consecutive render requests that marks the end of an
# interactive burst. Mirrors the client-side ``idleThresholdMs`` default in
# kaolin/visualize/dash/components/src/ts/interact/fps.tsx (500 ms): once no
# render request arrives for this long, the interaction is considered to have
# settled and (if configured) a final high-resolution frame is sent.
_DEFAULT_IDLE_THRESHOLD_S = 0.5
# Sliding window (seconds) over which render-request rate and render time are
# averaged. Matches the client-side ``windowMs`` default (1000 ms).
_DEFAULT_FPS_WINDOW_S = 1.0


class SlidingWindowFps:
    """Sliding-window event-rate tracker (renders-per-second of arrivals).

    Python port of the client-side ``InteractiveFps``
    (kaolin/visualize/dash/components/src/ts/interact/fps.tsx). Interactive
    viewers render on demand, so frame/request arrivals come in bursts
    separated by arbitrarily long idle gaps; a naive moving average would drag
    the reported rate toward zero during those gaps. This measures the rate
    only over the most recent ``window_s`` of events and treats any gap longer
    than ``idle_threshold_s`` as the end of a burst (earlier samples are
    discarded). :meth:`fps` returns ``None`` when no meaningful rate is
    available (fewer than two samples, or currently idle). Thread-safe.
    """

    def __init__(self, window_s: float = _DEFAULT_FPS_WINDOW_S,
                 idle_threshold_s: float = _DEFAULT_IDLE_THRESHOLD_S,
                 now: Callable[[], float] = time.monotonic):
        self._window_s = window_s
        self._idle_threshold_s = idle_threshold_s
        self._now = now
        self._events: list[float] = []
        self._lock = threading.Lock()

    def event(self, timestamp: Optional[float] = None) -> None:
        """Record that an event (e.g. a render request) just occurred."""
        t = self._now() if timestamp is None else timestamp
        with self._lock:
            prev = self._events[-1] if self._events else None
            if prev is not None and t - prev > self._idle_threshold_s:
                self._events.clear()
            self._events.append(t)
            self._trim(t)

    def fps(self) -> Optional[float]:
        """Current rate (events/second) over the window, or ``None`` if idle."""
        t = self._now()
        with self._lock:
            self._trim(t)
            if len(self._events) < 2:
                return None
            last = self._events[-1]
            if t - last > self._idle_threshold_s:
                return None
            elapsed = last - self._events[0]
            if elapsed <= 0:
                return None
            return (len(self._events) - 1) / elapsed

    def is_idle(self) -> bool:
        """True iff no event has arrived within ``idle_threshold_s``."""
        with self._lock:
            if not self._events:
                return True
            return self._now() - self._events[-1] > self._idle_threshold_s

    def reset(self) -> None:
        with self._lock:
            self._events.clear()

    def _trim(self, t: float) -> None:
        cutoff = t - self._window_s
        drop = 0
        while drop < len(self._events) and self._events[drop] < cutoff:
            drop += 1
        if drop:
            del self._events[:drop]


class RenderTimeFps:
    """Tracks render+encode durations and reports the implied throughput.

    Unlike :class:`SlidingWindowFps` (which measures how often requests
    *arrive*), this measures how fast frames could be produced back-to-back:
    ``fps = n / sum(durations)`` over the most recent ``window_s`` (i.e. the
    reciprocal of the mean per-frame render+encode time). This is the
    response-time bookkeeping used to decide whether rendering can keep up.
    Thread-safe (durations are recorded from the render worker thread).
    """

    def __init__(self, window_s: float = _DEFAULT_FPS_WINDOW_S,
                 now: Callable[[], float] = time.monotonic):
        self._window_s = window_s
        self._now = now
        self._samples: list[tuple[float, float]] = []  # (end_time, duration_s)
        self._lock = threading.Lock()

    def record(self, duration_s: float, timestamp: Optional[float] = None) -> None:
        """Record the time taken to render and encode one frame (seconds)."""
        t = self._now() if timestamp is None else timestamp
        with self._lock:
            self._samples.append((t, duration_s))
            self._trim(t)

    def fps(self) -> Optional[float]:
        """Implied renders/second from recent durations, or ``None`` if no samples."""
        t = self._now()
        with self._lock:
            self._trim(t)
            if not self._samples:
                return None
            total = sum(d for _, d in self._samples)
            if total <= 0:
                return None
            return len(self._samples) / total

    def last_duration(self) -> Optional[float]:
        """Most recent render+encode duration in seconds, or ``None``."""
        with self._lock:
            return self._samples[-1][1] if self._samples else None

    def reset(self) -> None:
        with self._lock:
            self._samples.clear()

    def _trim(self, t: float) -> None:
        cutoff = t - self._window_s
        self._samples = [s for s in self._samples if s[0] >= cutoff]


@dataclass
class RemoteRenderingOptions:
    """Options controlling how a remote frame is rendered and encoded.

    A single set of these is applied per render; :class:`AnyRendererMessageHandler`
    holds a default set plus optional ``low``/``high`` sets so quality can be
    traded off against responsiveness.

    Args:
        encode_format: How the rendered image is encoded on the wire: one of
            ``'raw'``, ``'png'`` or ``'jpeg'`` (see
            :class:`kaolin.visualize.web.io.ImageFormat`). Defaults to
            ``'png'``.
        jpeg_quality: 0..100 quality used only when ``encode_format`` is
            ``'jpeg'``. Defaults to :data:`~kaolin.visualize.web.io._DEFAULT_JPEG_QUALITY`.
        max_resolution: If set (and > 0), the camera is scaled so its largest
            image dimension equals this many pixels before rendering (aspect
            ratio preserved); ``None`` renders at the camera's native
            resolution.
        kwargs: Extra keyword arguments forwarded to the render function, or
            ``None`` to call it with the camera only.
    """
    encode_format: str = 'png'
    jpeg_quality: int = _DEFAULT_JPEG_QUALITY
    max_resolution: Optional[int] = None
    kwargs: Optional[Dict[str, Any]] = None

    @classmethod
    def interpolate(cls, a: RemoteRenderingOptions, b: RemoteRenderingOptions,
                    alpha: float = 0.5) -> RemoteRenderingOptions:
        """Blend two option sets. ``alpha=0`` returns ``a``, ``alpha=1`` returns ``b``.

        Numeric fields (``jpeg_quality``, ``max_resolution``) are linearly
        interpolated and rounded when both endpoints define them; otherwise
        (and for the non-numeric ``encode_format`` / ``kwargs``) the nearer
        endpoint is used (``a`` for ``alpha < 0.5``, else ``b``).
        """
        alpha = min(max(float(alpha), 0.0), 1.0)
        nearer = a if alpha < 0.5 else b

        def _lerp(x, y):
            if x is None or y is None:
                return x if alpha < 0.5 else y
            return int(round(x * (1.0 - alpha) + y * alpha))

        return cls(
            encode_format=nearer.encode_format,
            jpeg_quality=_lerp(a.jpeg_quality, b.jpeg_quality),
            max_resolution=_lerp(a.max_resolution, b.max_resolution),
            kwargs=nearer.kwargs,
        )


class AnyRendererMessageHandler(AsyncMessageHandlerProtocol):
    """Async handler that renders images on camera updates and render requests.

    Handles 'set_camera' and 'render' messages (with optional tag_suffix).
    Runs rendering in a separate thread to avoid blocking the event loop.

    Bookkeeping: tracks the rate of incoming render requests (``request_fps``)
    and the throughput implied by render+encode time (``rendering_fps``); see
    :meth:`stats`. These are recorded but not yet used to adapt quality.

    When ``high_rendering_options`` is configured, a final high-resolution
    frame is sent once render requests stop (a gap longer than
    :data:`_DEFAULT_IDLE_THRESHOLD_S`); if it is ``None`` no such frame is sent.

    Args:
        render_func: Callable(camera, \\*\\*kwargs) -> RGBA tensor (H, W, 4) or
            dict of outputs.
        device: Torch device the camera is moved to before rendering.
        tag_suffix: Optional suffix for message tags (e.g., '_secondary').
        rendering_options: Default :class:`RemoteRenderingOptions` used for
            requests. Defaults to a fresh :class:`RemoteRenderingOptions`.
        low_rendering_options: If configured, can interpolate from
            ``rendering_options`` toward these faster options to keep up
            response speed (not yet used).
        high_rendering_options: If configured, the final frame after a gap in
            requests is rendered with these options.
    """

    def __init__(self, render_func, device='cuda', tag_suffix='',
                 rendering_options: Optional[RemoteRenderingOptions] = None,
                 low_rendering_options: Optional[RemoteRenderingOptions] = None,
                 high_rendering_options: Optional[RemoteRenderingOptions] = None):
        super().__init__()
        self.camera = None
        self.device = device
        self.render_func = render_func
        self.tag_suffix = tag_suffix
        self.rendering_options = rendering_options if rendering_options is not None \
            else RemoteRenderingOptions()
        self.low_rendering_options = low_rendering_options
        self.high_rendering_options = high_rendering_options

        self._dirty = True
        self._queue = queue.Queue(2)
        self._cam_lock = threading.Lock()

        # Response-time bookkeeping.
        self._request_fps = SlidingWindowFps()
        self._rendering_fps = RenderTimeFps()

        # Gap detection for the high-res follow-up render. Each render request
        # bumps ``_request_seq``; a scheduled check fires after the idle gap and
        # only proceeds if no newer request arrived (seq unchanged), sending at
        # most one high-res frame per idle period.
        self._high_res_gap_s = _DEFAULT_IDLE_THRESHOLD_S
        self._request_seq = 0
        self._high_res_sent_seq = -1

    def accepted_message_tags(self) -> list[str]:
        return [f'set_camera{self.tag_suffix}', f'render{self.tag_suffix}']

    def on_connection_open(self, write_message_fn: Callable[[Union[bytes, str, Dict[str, Any]], bool], Any]):
        pass
        # Note: we don't send a rendering, as we expect the camera to arrive on the connection as well, for consistency

    def stats(self) -> Dict[str, Optional[float]]:
        """Snapshot of bookkeeping metrics.

        Returns a dict with ``request_fps`` (rate of incoming render requests),
        ``rendering_fps`` (throughput implied by render+encode time) and
        ``last_render_s`` (most recent render+encode duration). Values may be
        ``None`` before enough samples are collected.
        """
        return {
            'request_fps': self._request_fps.fps(),
            'rendering_fps': self._rendering_fps.fps(),
            'last_render_s': self._rendering_fps.last_duration(),
        }

    def _sanity_check(self):
        res = self._invoke_render(self.camera, self.rendering_options)
        if res.shape[-1] not in [4, 3]:
            raise ValueError(f'Rendering should be channel last')
        encoded = self.encode_rendering(res, self.rendering_options)

    async def on_message(self, message_tag: str, message_content: Dict,
                   write_message_fn: Callable[[Union[bytes, str, Dict[str, Any]], bool], Any]):
        """Handle incoming message."""
        if message_tag == f'set_camera{self.tag_suffix}':
            with self._cam_lock:
                first_cam = self.camera is None
                self.camera = kaolin.render.camera.Camera.from_dict(message_content).to(self.device)
            if first_cam:
                self._sanity_check()
                await self._maybe_render_and_send(write_message_fn, self.rendering_options)
        elif message_tag == f'render{self.tag_suffix}':
            if self.camera is not None:
                self._note_render_request()
                await self._maybe_render_and_send(write_message_fn, self.rendering_options)
                self._schedule_high_res_check(write_message_fn)

    def _note_render_request(self) -> None:
        """Record a render-request arrival (request-rate bookkeeping + gap seq)."""
        self._request_fps.event()
        self._request_seq += 1

    def _schedule_high_res_check(self, write_message_fn) -> None:
        """Arrange to send a high-res frame if requests go quiet (no-op if unconfigured)."""
        if self.high_rendering_options is None:
            return
        seq = self._request_seq
        asyncio.create_task(self._high_res_after_gap(seq, write_message_fn))

    async def _high_res_after_gap(self, seq: int, write_message_fn) -> None:
        await asyncio.sleep(self._high_res_gap_s)
        # A newer request arrived: that request scheduled its own check, so bail.
        if seq != self._request_seq:
            return
        # Already sent a high-res frame for this idle period.
        if self._high_res_sent_seq == seq:
            return
        self._high_res_sent_seq = seq
        await self._maybe_render_and_send(write_message_fn, self.high_rendering_options)

    async def _maybe_render_and_send(self, write_message_fn, options: RemoteRenderingOptions):
        try:
            # The queue carries the desired rendering options for this frame.
            self._queue.put(options, block=False)
            await asyncio.to_thread(self._execute_rendering_task, write_message_fn)
        except queue.Full:
            pass

    def _execute_rendering_task(self, write_message_fn):
        try:
            options = self._queue.get(block=False)
        except queue.Empty:
            return
        try:
            with self._cam_lock:
                camera = copy.deepcopy(self.camera)
            self.render_and_send(write_message_fn, camera, options)
        finally:
            self._queue.task_done()

    def render_and_send(self, write_message_fn, cam=None,
                        options: Optional[RemoteRenderingOptions] = None):
        """Render with given camera/options, recording render+encode time, and send."""
        if options is None:
            options = self.rendering_options
        if cam is None:
            cam = self.camera
        start = time.monotonic()
        res = self._invoke_render(cam, options)
        encoded = self.encode_rendering(res, options)
        self._rendering_fps.record(time.monotonic() - start)
        write_message_fn(encoded, True)

    def _invoke_render(self, cam, options: RemoteRenderingOptions):
        """Scale the camera per ``max_resolution`` and call the render function."""
        logger.info(f'Rendering ({cam.width}x{cam.height}) with {options.jpeg_quality} quality')
        cam = self._scale_camera(cam, options)
        if options.kwargs:
            return self.render_func(cam, **options.kwargs)
        return self.render_func(cam)

    def _scale_camera(self, cam, options: RemoteRenderingOptions):
        """Return a camera scaled to ``options.max_resolution`` (copy), or ``cam`` as-is."""
        if options.max_resolution is None or options.max_resolution <= 0 or \
                (options.max_resolution > cam.width and options.max_resolution > cam.height):
            return cam
        width, height = kaolin.render.camera.dimensions_to_max_resolution(
            cam.width, cam.height, options.max_resolution)
        cam = copy.deepcopy(cam)
        cam.width = width
        cam.height = height
        return cam

    def send_render_message(self, render_res,
                            write_message_fn: Callable[[Union[bytes, str, Dict[str, Any]], bool], Any],
                            options: Optional[RemoteRenderingOptions] = None):
        """Encode and send render result as binary message."""
        encoded = self.encode_rendering(render_res, options)
        write_message_fn(encoded, True)

    def encode_rendering(self, render_res, options: Optional[RemoteRenderingOptions] = None):
        """Encode render result (tensor or dict) as binary message."""
        if options is None:
            options = self.rendering_options
        if isinstance(render_res, dict):
            output_dict = render_res
        elif isinstance(render_res, torch.Tensor):
            output_dict = {'img': render_res}
        else:
            raise TypeError(f"render function output type ({type(render_res)}) unsupported")
        return kaolin.visualize.web.io.encode_message(
            f'render{self.tag_suffix}', output_dict, binary=True,
            image_format=options.encode_format, jpeg_quality=options.jpeg_quality)



class GlobalWebSocketConnectionManager:
    """Process-wide registry of active WebSocket connections.

    Tracks live :class:`WebSocketHandlerManager` instances (one per connected
    client) so server-side code can iterate over them, broadcast updates, or
    look up a specific connection independent of which client initiated an
    action.

    This is a thread-safe singleton; access it via :meth:`instance`.
    Connections register themselves automatically from
    :meth:`WebSocketHandlerManager.open` and unregister from
    :meth:`WebSocketHandlerManager.on_close`, so application code typically
    only needs to *read* this registry.
    """

    _singleton: Optional[GlobalWebSocketConnectionManager] = None
    _singleton_lock = threading.Lock()

    @classmethod
    def instance(cls) -> GlobalWebSocketConnectionManager:
        """Return the process-wide singleton, creating it on first call."""
        if cls._singleton is None:
            with cls._singleton_lock:
                if cls._singleton is None:
                    cls._singleton = cls()
        return cls._singleton

    def __init__(self):
        self._lock = threading.Lock()
        self._connections: Dict[str, WebSocketHandlerManager] = {}
        self._by_tab: Dict[str, set[WebSocketHandlerManager]] = defaultdict(set)

    def register(self, handler: WebSocketHandlerManager) -> str:
        """Record a newly-opened connection and return its assigned id.

        The id is allocated via :class:`~kaolin.visualize.web.naming.UniqueIdGenerator`
        with the ``websocket`` prefix so it is unique across all kaolin-issued
        ids in the process.

        If ``handler.tab_uuid`` is set (typically populated by
        :meth:`WebSocketHandlerManager.open` from the ``tab`` query argument),
        the handler is also indexed under that uuid so :meth:`get_handlers_by_tab`
        can fan out updates to every WS handler belonging to a single browser tab.
        """
        conn_id = UniqueIdGenerator.get_unique_id(prefix='websocket')
        tab_uuid = getattr(handler, 'tab_uuid', None)
        with self._lock:
            self._connections[conn_id] = handler
            if tab_uuid is not None:
                self._by_tab[tab_uuid].add(handler)
            count = len(self._connections)
        logger.info(
            f'Registered WebSocket connection {conn_id} '
            f'(tab {tab_uuid}, live connections: {count})')
        return conn_id

    def unregister(self, conn_id: str) -> None:
        """Remove a closed connection. Idempotent: unknown ids are silently ignored."""
        with self._lock:
            removed = self._connections.pop(conn_id, None)
            if removed is not None:
                tab_uuid = getattr(removed, 'tab_uuid', None)
                if tab_uuid is not None and tab_uuid in self._by_tab:
                    self._by_tab[tab_uuid].discard(removed)
                    if not self._by_tab[tab_uuid]:
                        del self._by_tab[tab_uuid]
            count = len(self._connections)
        if removed is not None:
            logger.info(f'Unregistered WebSocket connection {conn_id} (live connections: {count})')

    def get(self, conn_id: str) -> Optional[WebSocketHandlerManager]:
        """Return the handler for ``conn_id`` if it is still connected, else None."""
        with self._lock:
            return self._connections.get(conn_id)

    def get_handlers_by_tab(self, tab_uuid: str) -> list[WebSocketHandlerManager]:
        """Snapshot list of all live handlers that belong to the given browser tab.

        The list may be empty if the tab has no open WS connection (e.g. mid-reload).
        Handlers register themselves under their ``tab_uuid`` automatically when
        :meth:`WebSocketHandlerManager.open` parses the ``tab`` query argument.
        """
        with self._lock:
            return list(self._by_tab.get(tab_uuid, ()))

    def connections(self) -> list[WebSocketHandlerManager]:
        """Snapshot list of all currently-active connection handlers."""
        with self._lock:
            return list(self._connections.values())

    def broadcast(self, raw_message):
        logger.error(f'Broadcast of segmentation to other clients not implemented')
        pass

    def connection_ids(self) -> list[str]:
        """Snapshot list of all currently-active connection ids."""
        with self._lock:
            return list(self._connections.keys())

    def items(self) -> list[tuple[str, WebSocketHandlerManager]]:
        """Snapshot list of ``(conn_id, handler)`` pairs for active connections."""
        with self._lock:
            return list(self._connections.items())

    def __len__(self) -> int:
        with self._lock:
            return len(self._connections)

    def __contains__(self, conn_id: object) -> bool:
        with self._lock:
            return conn_id in self._connections


class WebSocketHandlerManager(tornado.websocket.WebSocketHandler):
    """Tornado WebSocket handler that dispatches messages to registered handlers.

    Routes incoming messages by tag to appropriate sync/async handlers.
    Provides thread-safe message writing via write_message_safe().

    Handlers are constructed per-connection from the handler specs passed at
    application setup time; each WebSocket connection gets its own handler
    instances so that per-connection state (e.g. cameras, render queues) is
    not shared across clients.

    Each live connection is also registered in
    :class:`GlobalWebSocketConnectionManager` for the duration of its open
    socket so other server-side code can enumerate or broadcast to clients.
    """

    def initialize(self, handler_specs: list[tuple]):
        """Construct per-connection message handlers from specs.

        Args:
            handler_specs: list of ``(handler_class, args)`` or
                ``(handler_class, args, kwargs)`` tuples, where ``args`` is a
                tuple/list of positional arguments and ``kwargs`` is a dict of
                keyword arguments to pass to the handler constructor. A fresh
                handler instance is built for each WebSocket connection.
        """
        logger.info("Initializing per-connection WebSocket handlers")
        self.handlers = defaultdict(list)
        self._io_loop = tornado.ioloop.IOLoop.current()
        self._connection_id: Optional[str] = None
        self.tab_uuid: Optional[str] = None
        self.handler_specs = handler_specs
        """Per-browser-tab uuid, parsed from the ``tab`` query argument on
        :meth:`open`. Use this to correlate the WS handler instance with a
        :class:`dash.dcc.Store` value or a :class:`SessionRegistry` entry."""

    def _handlers_lazy_initialize(self):
        logger.debug(f'Initializing custom handlers using {len(self.handler_specs)} specs tab {self.tab_uuid}')

        def _func_to_arg(v):    
            if type(v) == TuidCallable:
                return v(self.tab_uuid)
            else:
                return v

        for spec in self.handler_specs:
            if len(spec) == 2:
                handler_cls, args = spec
                kwargs = {}
            elif len(spec) == 3:
                handler_cls, args, kwargs = spec
            else:
                raise ValueError(
                    f'Handler spec must be a (class, args) or (class, args, kwargs) '
                    f'tuple, got {spec!r}')
            args = [_func_to_arg(x) for x in args]
            kwargs = {k: _func_to_arg(v) for k, v in kwargs.items()}
            handler = handler_cls(*args, **kwargs)
            self.register_message_handler(handler)


    def set_camera(self, camera: kaolin.render.camera.Camera):
        """Programmatically set camera (triggers handlers)."""
        message = kaolin.visualize.web.io.encode_message('set_camera', camera.as_dict(), binary=False)
        self._apply_handlers(message)

    def register_message_handler(self, handler: SyncMessageHandlerProtocol | AsyncMessageHandlerProtocol):
        """Register a handler for its accepted message tags."""
        # TODO: maybe split into sync and async here?
        for tag in handler.accepted_message_tags():
            self.handlers[tag].append(handler)

    def check_origin(self, origin):
        """Allow connections from any origin"""
        return True

    @property
    def connection_id(self) -> Optional[str]:
        """Id assigned by :class:`GlobalWebSocketConnectionManager` while open."""
        return self._connection_id

    def open(self):
        """ Open socket connection and send test message."""
        self.tab_uuid = self.get_query_argument('tab', default=None)
        self._connection_id = GlobalWebSocketConnectionManager.instance().register(self)
        logger.debug(
            f"Socket opened (connection {self._connection_id}, tab {self.tab_uuid}).")

        if len(self.handlers) == 0:
            self._handlers_lazy_initialize()

        message = {"status": f"ok"}
        self.write_message(message, binary=False)

        for k, handler_list in self.handlers.items():
            for h in handler_list:
                h.on_connection_open(self.write_message_safe)

    def send_sample_json_message(self):
        message = {"type": "test",
                   "data": {"message": "Nice to meet you!"}}
        self.write_message(message, binary=False)

    async def on_message(self, raw_message):
        """ Handles new messages on the socket."""
        logger.debug('Received message of type {}'.format(type(raw_message)))  # , message))

        try:
            if type(raw_message) == bytes:
                #logger.debug('Decoding binary message')
                message = kaolin.visualize.web.io.from_binary(raw_message)
            else:
                #logger.debug('Decoding string message')
                message = json.loads(raw_message)
        except Exception as e:
            logger.error('Failed to decode incoming message: {}'.format(e))
            return

        await self._apply_handlers(message)

    #@tornado.gen.coroutine
    async def _apply_handlers(self, message: dict):
        tag = message.get(kaolin.visualize.web.io.MESSAGE_TAG_KEY, None)
        if tag is None:
            raise ValueError(f'Message does not contain {kaolin.visualize.web.io.MESSAGE_TAG_KEY} among keys {message.keys()}')

        content = message.get(kaolin.visualize.web.io.MESSAGE_CONTENT_KEY, None)
        if content is None and len(list(message.keys())) > 1:
            logger.warning(f'Expected message content to be keyed by "{kaolin.visualize.web.io.MESSAGE_CONTENT_KEY}", '
                           f'but found these keys instead: {message.keys()}')

        handlers = self.handlers.get(tag, None)
        if handlers is None:
            logger.warning(f'No handler assigned for message with tag {tag}; message {kaolin.utils.testing.tensor_info(message)}')
            return

        print(f'WebSocketHandlerManager.Applying Handlers ({message["tag"]}) {self.tab_uuid}')
        for h in handlers:
            if asyncio.iscoroutinefunction(h.on_message):
                asyncio.create_task(h.on_message(tag, content, self.write_message_safe))
            else:
                h.on_message(tag, content, self.write_message_safe)

    def on_close(self):
        if self._connection_id is not None:
            GlobalWebSocketConnectionManager.instance().unregister(self._connection_id)
            logger.info(f"Socket closed (connection {self._connection_id}).")
            self._connection_id = None
        else:
            logger.info("Socket closed (never fully opened).")

    def write_message_safe(self, message, binary=False):
        """Thread-safe message write via IOLoop callback."""
        # with self.write_lock:
        callback = functools.partial(self.write_message, message, binary)
        # Schedule the callback to be run safely by the main Tornado IOLoop.
        logger.info("Scheduling response")
        self._io_loop.add_callback(callback)









# def start_tornado(app, port):
#     app.listen(port)
#     try:
#         IOLoop.current().start()
#     except Exception as e:
#         print('Loop running')
#         pass
#
#
# def start_server_thread(server, port):
#     """ Starts a stub webserver in a separate thread. """
#     #ClientManager.init()
#     thread = threading.Thread(target=lambda: start_tornado(server, port))
#     thread.daemon = True
#     thread.start()
#     return thread
#

# def create_server(renderer):
#     """ Creates HTTP & websocket helper. """
#
#     # Flask for HTTP
#     # _base_dir = os.path.dirname(__file__)
#     # _template_dir = os.path.join(_base_dir, 'templates')
#     # _static_dir = os.path.join(_base_dir, 'static')
#     # app = Flask('simx',
#     #             template_folder=_template_dir,
#     #             static_url_path='/static',
#     #             static_folder=_static_dir)
#     # app.config["TEMPLATES_AUTO_RELOAD"] = True
#
#     # @app.route('/')
#     # def index():
#     #     # flask.request.args.get('canvas', 2000)
#     #     return render_template('home.html',
#     #                            canvas_width=1000)
#
#     # @app.route('/brush/<library_name>/<brush_name>.jpg')
#     # def brush_image(library_name, brush_name):
#     #     # TODO: this gotta be async
#     #     if library_name in libraries:
#     #         image = libraries[library_name].get_style_icon(brush_name)
#     #     else:
#     #         image = np.zeros((128, 128, 3), dtype=np.uint8)
#     #     image = PIL.Image.fromarray(image)
#     #     byte_io = io.BytesIO()
#     #     image.save(byte_io, format="JPEG")
#     #     jpg_buffer = byte_io.getvalue()
#     #     byte_io.close()
#     #     response = Response(jpg_buffer, mimetype='image/jpeg')
#     #     return response
#
#     def no_action(*args, **kwargs):
#         logger.warning(f'warn')
#
#     # Tornado server to handle websockets
#     #container = WSGIContainer(app)
#     server = Application([
#         (r'/websocket/',
#          SimpleWebSocketHandler, dict(renderer=renderer))])
#
#     #     (r'.*',
#     #      FallbackHandler, dict(fallback=no_action))
#     # ])
#     return server
#





# class ClientManager:
#     """
#     Placeholder. Managers concurrent connections from multiple clients to the same live scene.
#     """
#     __singleton = None
#     __lock = threading.Lock()
#
#     @staticmethod
#     def singleton():
#         if ClientManager.__singleton is None:
#             ClientManager.__singleton = ClientManager()
#         return ClientManager.__singleton
#
#     @staticmethod
#     def init():
#         ClientManager.singleton()  # instantiate to avoid deadlock
#
#     def __init__(self):
#         self.rb = None
#         self._handlers = {}
#
#     @staticmethod
#     def add_ws_handler(handler):
#         with ClientManager.__lock:
#             ClientManager.singleton()._add_ws_handler(handler)
#
#     def _add_ws_handler(self, handler):
#         name = None
#         while name is None or name in self._handlers:
#             name = f'connection{random.randint(0, 2000)}'
#         self._handlers[name] = handler
#         handler.name = name
#
#     @staticmethod
#     def remove_ws_handler(handler):
#         with ClientManager.__lock:
#             ClientManager.singleton()._remove_ws_handler(handler)
#
#     def _remove_ws_handler(self, handler):
#         del self._handlers[handler.name]
#
#     @staticmethod
#     def num_handlers():
#         with ClientManager.__lock:
#             return ClientManager.singleton()._num_handlers()
#
#     def _num_handlers(self):
#         return len(self._handlers)
#
#     @staticmethod
#     def broadcast_render(render):
#         with ClientManager.__lock:
#             ClientManager.singleton()._broadcast_render(render)
#
#     def _broadcast_render(self, render):
#         self.rb = render
#
#         logger.info(f'Broadcasting render to {self._num_handlers()} handlers')
#         for handler in self._handlers.values():
#             handler.send_updated_render(render)
#
#     @staticmethod
#     def get_current_frame():
#         with ClientManager.__lock:
#             return ClientManager.singleton().rb  # TODO

# class SimpleWebSocketHandler(WebSocketHandler):
#     """ Handles websocket communication with the JS client.
#     """
#
#     def initialize(self, get_current_frame=ClientManager.get_current_frame):
#         """ Takes TBD helper type that can actually run the model."""
#         # Note: this is correct, __init__ method should not be written for this
#         self.get_current_frame = get_current_frame
#         self.name = None
#
#     def open(self):
#         """ Open socket connection and send information about available geometry."""
#         ClientManager.add_ws_handler(self)
#         logger.debug("Socket opened.")
#         message = {"status": f"ok {self.name} {ClientManager.num_handlers()}"}
#         self.write_message(message, binary=False)
#         self.send_current_frame()
#
#     @tornado.gen.coroutine
#     def send_current_frame(self):
#         self.send_updated_render(self.get_current_frame())
#
#
#     # @tornado.gen.coroutine
#     # def send_current_brush_info(self):
#     #     message = {"type": "brushinfo",
#     #                "data": {"style_id": "%s" % str(self.helper.brush_options.style_id),
#     #                         "library_id": "%s" % self.helper.brush_options.library_id,
#     #                         "colors": "%s" % self.helper.engine.uvs_mapper.get_colors(self.helper.brush_options)}}
#     #     self.write_message(message, binary=False)
#
#
#     @tornado.gen.coroutine
#     def on_message(self, message):
#         """ Handles new messages on the socket."""
#         logger.debug('Received message of type {}'.format(type(message)))  #, message))
#
#         try:
#             if type(message) == bytes:
#                 self._handle_binary_request(message)
#             else:
#                 self._handle_json_request(message)
#         except Exception as e:
#             logger.error('Failed to decode incoming message: {}'.format(e))
#
#     def _encode_type_render(self):
#         return int32_to_binary(0)
#
#     @tornado.gen.coroutine
#     def send_updated_render(self, rb):
#         if rb is not None:
#             bin_str = self._encode_type_render() + image_to_binary(rb.numpy())
#             self.write_message(bin_str, binary=True)
#
#     # @tornado.gen.coroutine
#     # def _handle_image_request(self, meta, bg_img, fg_img):
#     #     brush_options = self.helper.default_brush_options()
#     #     for colorinfo in meta['colors']:
#     #         cidx = colorinfo[0]
#     #         brush_options.set_color(cidx, colorinfo[1:])
#     #     brush_options.debug = meta['debug']
#     #     if self.use_positions:
#     #         brush_options.set_position(int(meta['x']), int(meta['y']))
#     #     else:
#     #         brush_options.position = None
#     #
#     #     brush_options.enable_uvs_mapping = self.uvs_mapping
#     #
#     #     res_img, debug_img, meta_out = self.helper.render_stroke(bg_img, fg_img, brush_options, meta)
#     #     bin_str = self._encode_type_render(meta['extra_data']) + image_patch_to_binary(res_img, meta_out['x'], meta_out['y'])
#     #     self.write_message(bin_str, binary=True)
#     #
#     #     # Also send debug image
#     #     if debug_img is not None:
#     #         bin_str = self._encode_type_debug_img() + image_patch_to_binary(debug_img, 0, 0)
#     #         self.write_message(bin_str, binary=True)
#
#     @tornado.gen.coroutine
#     def _handle_binary_request(self, raw_message):
#         logger.debug('Decoding binary message')
#         # meta, read_offset = decode_render_request_metadata(raw_message)
#         # # logger.debug(f'Decoded meta {meta}')
#         # patch_meta, img_stroke, img_canvas = binary_to_image_patches(raw_message, read_offset)
#         # # logger.debug(f'Decoded patch meta {patch_meta}')
#         # meta.update(patch_meta)
#         # # log_tensor(img_stroke, 'decoded image (stroke)', logger)
#         # self._handle_image_request(meta, img_stroke, img_canvas)
#
#     def _handle_set_option(self, msg):
#         if msg.get('option') == 'positions':
#             self.use_positions = msg.get('value')
#             logger.info(f'Set use_positions to {self.use_positions}')
#         elif msg.get('option') == 'uvs_mapping':
#             self.uvs_mapping = msg.get('value')
#
#     @tornado.gen.coroutine
#     def _handle_json_request(self, raw_message):
#         logger.debug('Decoding string message')
#         # msg = json.loads(raw_message)
#         #
#         # if msg.get('type') == 'set_brush':
#         #     if msg.get('style_id') and msg.get('library_id'):
#         #         library_id = msg.get('library_id')
#         #         style_id = msg.get('style_id')
#         #         if library_id in self.libraries and style_id in self.libraries[library_id].get_style_ids():
#         #             self.libraries[library_id].set_style(style_id, self.helper.brush_options)
#         #             self.helper.brush_options.library_id = library_id
#         #     else:
#         #         self.helper.set_new_brush(msg.get('seed'))
#         #     self.send_current_brush_info()
#         # elif msg.get('type') == 'save_brush':
#         #     self.save_current_brush()
#         # elif msg.get('type') == 'set_option':
#         #     self._handle_set_option(msg)
#         # elif msg.get('type') == 'set_render_mode':
#         #     self.helper.set_render_mode(msg.get('mode'))
#         # elif msg.get('type') == 'new_canvas':
#         #     print(msg)
#         #     self.helper.make_new_canvas(int(msg.get('rows')), int(msg.get('cols')),
#         #                                 feature_blending=int(msg.get('feature_blending')))
#         # else:
#         #     logger.warning('Received unknown json message of type {}: {}'.format(
#         #         msg.get('type'), msg))
#
#     def on_close(self):
#         logger.info("Socket closed.")
#         ClientManager.remove_ws_handler(self)





