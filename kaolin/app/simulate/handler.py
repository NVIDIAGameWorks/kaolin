# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import asyncio
import logging
import math
import pathlib
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Union

import torch

import kaolin
import kaolin.io.gaussians
import kaolin.render.camera
import kaolin.visualize.web.io as _ws_io
from kaolin.visualize.web.sockets import AnyRendererMessageHandler, AsyncMessageHandlerProtocol

logger = logging.getLogger(__name__)

try:
    from kaolin.io.usd.gaussians import import_gaussianclouds as _import_gaussianclouds_usd
    _HAS_USD_MULTI = True
except ImportError:
    _HAS_USD_MULTI = False
    logger.warning('kaolin.io.usd.gaussians.import_gaussianclouds not available; multi-Gaussian USD selection disabled.')

try:
    from kaolin.io.usd.subset import import_subsets as _import_subsets_usd
    _HAS_USD_SUBSET = True
except ImportError:
    _HAS_USD_SUBSET = False

try:
    from kaolin.io.usd import get_skinned_physics as _get_skinned_physics
    _HAS_USD_PHYSICS = True
except ImportError:
    _HAS_USD_PHYSICS = False


def _strip_common_prefix(names: list) -> list:
    """Strip the longest shared USD path prefix from a list of prim paths."""
    if not names:
        return names
    parts = [n.lstrip('/').split('/') for n in names]
    common_depth = 0
    for i in range(min(len(p) for p in parts)):
        if all(p[i] == parts[0][i] for p in parts):
            common_depth = i + 1
        else:
            break
    result = []
    for p in parts:
        remaining = p[common_depth:]
        result.append('/'.join(remaining) if remaining else p[-1])
    return result


def _build_subset_entries(path: str, prim_path: str) -> list:
    """Return a list of available_gaussians entries for each non-background GeomSubset in prim_path.

    Returns an empty list if there are no subsets (caller should fall back to loading the whole prim).
    """
    try:
        subsets = _import_subsets_usd(path, prim_path, family_name='part')
    except Exception:
        logger.debug('import_subsets failed for %s / %s', path, prim_path, exc_info=True)
        return []
    if not subsets:
        return []
    # Strip the 'background' segment — it's the complement and rarely useful as a separate object.
    non_bg = {k: v for k, v in subsets.items() if k.split('/')[-1] != 'background'}
    if not non_bg:
        return []
    stripped_names = _strip_common_prefix(list(non_bg.keys()))
    return [
        {'prim_path': sub_path, 'name': name, 'is_subset': True, 'parent_prim': prim_path}
        for sub_path, name in zip(non_bg.keys(), stripped_names)
    ]


def _slice_gaussians(gs, indices):
    """Return a new GaussianSplatModel containing only the splats at the given indices."""
    import kaolin.rep.gaussians as _grep
    idx = indices.long()
    return _grep.GaussianSplatModel(
        positions=gs.positions[idx],
        orientations=gs.orientations[idx],
        scales=gs.scales[idx],
        opacities=gs.opacities[idx],
        sh_coeff=gs.sh_coeff[idx],
        sh_degree=gs.sh_degree,
        strict_checks=False,
    )


def _load_baked_physics(path: str, prim_path: str):
    """Try to load pre-baked SkinnedPhysicsPoints from a USD prim. Returns None on failure."""
    if not _HAS_USD_PHYSICS:
        return None
    try:
        baked = _get_skinned_physics(path, prim_path)
        if baked is not None:
            logger.info(f'Loaded pre-baked SkinnedPhysicsPoints from {prim_path}')
        return baked
    except Exception:
        logger.debug(f'No skinned physics data found at {prim_path}', exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def _euler_deg_to_rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> torch.Tensor:
    """ZYX Euler angles (degrees) → 4×4 float32 rotation matrix."""
    cx = math.cos(math.radians(rx_deg)); sx = math.sin(math.radians(rx_deg))
    cy = math.cos(math.radians(ry_deg)); sy = math.sin(math.radians(ry_deg))
    cz = math.cos(math.radians(rz_deg)); sz = math.sin(math.radians(rz_deg))
    Rx = torch.tensor([[1, 0, 0, 0], [0, cx, -sx, 0], [0, sx, cx, 0], [0, 0, 0, 1]], dtype=torch.float32)
    Ry = torch.tensor([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype=torch.float32)
    Rz = torch.tensor([[cz, -sz, 0, 0], [sz, cz, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.float32)
    return Rz @ Ry @ Rx


def make_transform_matrix(translation, rotation_euler_deg) -> torch.Tensor:
    """Build a 4×4 float32 matrix from [tx, ty, tz] and [rx, ry, rz] Euler degrees."""
    T = torch.eye(4, dtype=torch.float32)
    T[0, 3], T[1, 3], T[2, 3] = float(translation[0]), float(translation[1]), float(translation[2])
    return T @ _euler_deg_to_rotation_matrix(*rotation_euler_deg)


# ---------------------------------------------------------------------------
# State dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ObjectState:
    obj_id: int
    name: str
    gaussians: object                                       # GaussianSplatModel (rest pose, no transform)
    translation: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation_euler_deg: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    baked_physics: object = None                           # SkinnedPhysicsPoints loaded from USD, or None
    centroid: list = field(default_factory=lambda: [0.0, 0.0, 0.0], init=False, repr=False)

    def __post_init__(self):
        c = self.gaussians.positions.float().mean(dim=0).cpu().tolist()
        self.centroid = c

    def transform_matrix(self) -> torch.Tensor:
        """Rotate around the object's centroid, then apply user translation.

        Order: T_user @ T_c @ R @ T_neg_c
          - T_neg_c  : shift object so its centroid sits at the world origin
          - R        : rotate around that origin (= object centre)
          - T_c      : shift centroid back to its rest-pose position
          - T_user   : apply user translation
        """
        cx, cy, cz = self.centroid
        tx, ty, tz = [float(v) for v in self.translation]

        T_user = torch.eye(4, dtype=torch.float32)
        T_user[0, 3], T_user[1, 3], T_user[2, 3] = tx, ty, tz

        T_c = torch.eye(4, dtype=torch.float32)
        T_c[0, 3], T_c[1, 3], T_c[2, 3] = cx, cy, cz

        T_neg_c = torch.eye(4, dtype=torch.float32)
        T_neg_c[0, 3], T_neg_c[1, 3], T_neg_c[2, 3] = -cx, -cy, -cz

        R = _euler_deg_to_rotation_matrix(*[float(v) for v in self.rotation_euler_deg])
        return T_user @ T_c @ R @ T_neg_c

    def transformed_gaussians(self):
        """Return a GaussianSplatModel with the current transform applied."""
        import kaolin.rep.gaussians as _grep
        gs = self.gaussians
        mat = self.transform_matrix().to(gs.positions.device)
        copy = _grep.GaussianSplatModel(
            positions=gs.positions.clone(),
            orientations=gs.orientations.clone(),
            scales=gs.scales.clone(),
            opacities=gs.opacities.clone(),
            sh_coeff=gs.sh_coeff.clone(),
            sh_degree=gs.sh_degree,
            transform=mat,
        )
        return copy.as_transformed()

    def to_dict(self) -> dict:
        return {
            'obj_id': self.obj_id,
            'name': self.name,
            'translation': list(self.translation),
            'rotation_euler_deg': list(self.rotation_euler_deg),
        }


@dataclass
class AppState:
    device: str = 'cuda'
    objects: list = field(default_factory=list)
    _next_id: int = field(default=0, repr=False)

    def add_object(self, name: str, gaussians) -> 'ObjectState':
        obj = ObjectState(obj_id=self._next_id, name=name, gaussians=gaussians)
        self.objects.append(obj)
        self._next_id += 1
        return obj

    def remove_object(self, obj_id: int) -> None:
        self.objects = [o for o in self.objects if o.obj_id != obj_id]

    def get_object(self, obj_id: int) -> Optional['ObjectState']:
        for o in self.objects:
            if o.obj_id == obj_id:
                return o
        return None


# ---------------------------------------------------------------------------
# WebSocket handler
# ---------------------------------------------------------------------------

class SimulateHandler(AsyncMessageHandlerProtocol):
    """Per-client WebSocket handler for the Kaolin Simulate app.

    Implements AsyncMessageHandlerProtocol (on_connection_open + on_message).
    Shared AppState is passed in at construction time; per-client state
    (camera, sim thread) lives on the instance.
    """

    def __init__(self, app_state: AppState):
        self._state = app_state
        self._write_fn: Optional[Callable] = None
        self._sim_stop_event = threading.Event()
        self._sim_thread: Optional[threading.Thread] = None
        self._physics_runner = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Delegate camera management + standard render/set_camera messages here.
        self._render_handler = AnyRendererMessageHandler(
            render_func=self._render_scene,
            device=app_state.device,
        )

    # ------------------------------------------------------------------
    # AsyncMessageHandlerProtocol
    # ------------------------------------------------------------------

    def accepted_message_tags(self) -> list:
        return [
            'render', 'set_camera',
            'load_file', 'select_objects', 'set_transform', 'remove_object',
            'start_sim', 'stop_sim', 'reset_sim',
            'path_complete',
        ]

    def on_connection_open(self, write_message_fn: Callable[[Union[bytes, str, Dict[str, Any]], bool], Any]):
        self._write_fn = write_message_fn
        self._render_handler.on_connection_open(write_message_fn)
        self._loop = asyncio.get_event_loop()
        self._send_sync('object_list', {'objects': [o.to_dict() for o in self._state.objects]})

    async def on_message(self, message_tag: str, message_content: Dict,
                         write_message_fn: Callable[[Union[bytes, str, Dict[str, Any]], bool], Any]):
        # Camera + render cycle is delegated to AnyRendererMessageHandler.
        if message_tag in ('render', 'set_camera'):
            await self._render_handler.on_message(message_tag, message_content, write_message_fn)
            return

        handlers = {
            'load_file': self._handle_load_file,
            'select_objects': self._handle_select_objects,
            'set_transform': self._handle_set_transform,
            'remove_object': self._handle_remove_object,
            'start_sim': self._handle_start_sim,
            'stop_sim': self._handle_stop_sim,
            'reset_sim': self._handle_reset_sim,
            'path_complete': self._handle_path_complete,
        }
        fn = handlers.get(message_tag)
        if fn:
            await fn(message_content, write_message_fn)
        else:
            logger.debug(f'Unhandled WS tag: {message_tag}')

    # ------------------------------------------------------------------
    # Message handlers
    # ------------------------------------------------------------------

    async def _handle_load_file(self, content: dict, write_fn: Callable):
        path = (content.get('path') or '').strip()
        if not path or not pathlib.Path(path).exists():
            self._send_sync('error', {'message': f'File not found: {path}'}, write_fn)
            return

        suffix = pathlib.Path(path).suffix.lower()
        try:
            if suffix in ('.usd', '.usda', '.usdc', '.usdz') and _HAS_USD_MULTI:
                clouds = _import_gaussianclouds_usd(path, return_list=False)
                if not clouds:
                    self._send_sync('error', {'message': 'No Gaussian clouds found in USD file.'}, write_fn)
                    return
                if len(clouds) == 1:
                    prim_path, gs = next(iter(clouds.items()))
                    # Check for GeomSubset segments inside this prim.
                    entries = _build_subset_entries(path, prim_path) if _HAS_USD_SUBSET else []
                    if entries:
                        self._send_sync('available_gaussians', {
                            'path': path,
                            'entries': entries,
                        }, write_fn)
                    else:
                        obj = self._state.add_object(prim_path.split('/')[-1], gs.to(self._state.device))
                        obj.baked_physics = _load_baked_physics(path, prim_path)
                        self._send_object_list(write_fn)
                        self._render_handler.render_and_send(write_fn)
                else:
                    # Multiple Gaussian prims — let user pick.
                    self._send_sync('available_gaussians', {
                        'path': path,
                        'entries': [{'prim_path': p, 'name': p.split('/')[-1],
                                     'is_subset': False, 'parent_prim': ''} for p in clouds.keys()],
                    }, write_fn)
            else:
                gs = kaolin.io.gaussians.import_gaussiancloud(path)
                if gs is None:
                    self._send_sync('error', {'message': 'No Gaussian cloud found in file.'}, write_fn)
                    return
                self._state.add_object(pathlib.Path(path).stem, gs.to(self._state.device))
                self._send_object_list(write_fn)
                self._render_handler.render_and_send(write_fn)
        except Exception as e:
            logger.exception(f'Error loading {path}')
            self._send_sync('error', {'message': str(e)}, write_fn)

    async def _handle_select_objects(self, content: dict, write_fn: Callable):
        path = content.get('path', '')
        # Support both the new 'selected_entries' (list of dicts) and the old 'selected_paths'.
        entries = content.get('selected_entries') or []
        if not entries:
            old_paths = content.get('selected_paths') or []
            entries = [{'prim_path': p, 'is_subset': False, 'parent_prim': ''} for p in old_paths]
        if not entries:
            return

        try:
            # Cache loaded Gaussian clouds keyed by prim_path to avoid re-reading the USD.
            _cloud_cache: dict = {}

            def _get_cloud(prim_path: str):
                if prim_path not in _cloud_cache:
                    clouds = _import_gaussianclouds_usd(path, [prim_path])
                    _cloud_cache[prim_path] = clouds.get(prim_path)
                return _cloud_cache[prim_path]

            for entry in entries:
                prim_path = entry.get('prim_path', '')
                name = entry.get('name') or prim_path.split('/')[-1]
                is_subset = entry.get('is_subset', False)
                parent_prim = entry.get('parent_prim', '')

                if is_subset and _HAS_USD_SUBSET and parent_prim:
                    gs_full = _get_cloud(parent_prim)
                    if gs_full is None:
                        continue
                    all_subsets = _import_subsets_usd(path, parent_prim, family_name='part')
                    if prim_path not in all_subsets:
                        continue
                    indices = all_subsets[prim_path]['indices']
                    gs = _slice_gaussians(gs_full.to(self._state.device), indices)
                    obj = self._state.add_object(name, gs)
                    # Physics is stored on the subset prim itself (VoMP convention).
                    obj.baked_physics = _load_baked_physics(path, prim_path)
                elif _HAS_USD_MULTI:
                    gs = _get_cloud(prim_path)
                    if gs is None:
                        continue
                    obj = self._state.add_object(name, gs.to(self._state.device))
                    obj.baked_physics = _load_baked_physics(path, prim_path)

        except Exception as e:
            logger.exception(f'Error selecting objects from {path}')
            self._send_sync('error', {'message': str(e)}, write_fn)
            return
        self._send_object_list(write_fn)
        self._render_handler.render_and_send(write_fn)

    async def _handle_set_transform(self, content: dict, write_fn: Callable):
        obj = self._state.get_object(content.get('obj_id'))
        if obj is None:
            return
        if 'translation' in content:
            # Decode from numpy scalars (binary WS payload) to plain Python floats.
            obj.translation = [float(v) for v in content['translation']]
        if 'rotation_euler_deg' in content:
            obj.rotation_euler_deg = [float(v) for v in content['rotation_euler_deg']]
        self._render_handler.render_and_send(write_fn)

    async def _handle_remove_object(self, content: dict, write_fn: Callable):
        self._state.remove_object(content.get('obj_id'))
        self._send_object_list(write_fn)
        self._render_handler.render_and_send(write_fn)

    async def _handle_path_complete(self, content: dict, write_fn: Callable):
        import glob
        import os
        partial = (content.get('path') or '').strip()
        try:
            matches = sorted(glob.glob(partial + '*'))[:30]
            completions = [p + ('/' if os.path.isdir(p) else '') for p in matches]
        except Exception:
            completions = []
        self._send_sync('path_completions', {'completions': completions}, write_fn)

    async def _handle_start_sim(self, content: dict, write_fn: Callable):
        if not self._state.objects:
            self._send_sync('error', {'message': 'No objects loaded.'}, write_fn)
            return
        missing = [o.name for o in self._state.objects if o.baked_physics is None]
        if missing:
            self._send_sync('error', {
                'message': 'The following objects have no physics data — load them from a USD file '
                           'that was processed by VoMP: ' + ', '.join(missing)
            }, write_fn)
            return
        self._stop_sim()
        self._sim_stop_event.clear()

        from kaolin.app.simulate.physics import PhysicsRunner
        runner = PhysicsRunner(
            objects=list(self._state.objects),
            timestep=content.get('timestep', 0.03),
            newton_steps=content.get('newton_steps', 3),
            enable_collisions=bool(content.get('enable_collisions', False)),
            device=self._state.device,
        )
        self._physics_runner = runner

        # Capture current write_fn and loop for use in the background thread.
        _write_fn = write_fn
        _loop = self._loop

        def _run():
            self._schedule(_loop, lambda: self._send_sync('sim_status', {'message': 'Training physics models...'}))
            try:
                runner.setup()
            except Exception as e:
                logger.exception('Physics setup failed')
                self._schedule(_loop, lambda: self._send_sync('sim_status', {'message': f'Error: {e}'}))
                return

            self._schedule(_loop, lambda: self._send_sync('sim_status', {'message': 'Simulating...'}))
            while not self._sim_stop_event.is_set():
                runner.step()
                gaussians_list = runner.get_current_gaussians()
                if gaussians_list and self._render_handler.camera is not None:
                    from kaolin.app.simulate.renderer import render_objects
                    try:
                        img = render_objects(gaussians_list, self._render_handler.camera)
                        encoded = _ws_io.encode_message('render', {'img': img}, binary=True)
                        _write_fn(encoded, True)
                    except Exception:
                        logger.debug('Sim render failed', exc_info=True)

            self._schedule(_loop, lambda: self._send_sync('sim_status', {'message': 'Idle'}))

        self._sim_thread = threading.Thread(target=_run, daemon=True)
        self._sim_thread.start()

    async def _handle_stop_sim(self, content: dict, write_fn: Callable):
        self._stop_sim()
        self._send_sync('sim_status', {'message': 'Idle'}, write_fn)

    async def _handle_reset_sim(self, content: dict, write_fn: Callable):
        self._stop_sim()
        self._send_sync('sim_status', {'message': 'Idle'}, write_fn)
        self._render_handler.render_and_send(write_fn)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stop_sim(self):
        self._sim_stop_event.set()
        if self._sim_thread and self._sim_thread.is_alive():
            self._sim_thread.join(timeout=3.0)
        self._sim_thread = None
        self._physics_runner = None

    def _render_scene(self, camera):
        """Render function passed to AnyRendererMessageHandler."""
        if not self._state.objects:
            import torch
            h, w = int(camera.height), int(camera.width)
            return torch.zeros(h, w, 3, dtype=torch.uint8)
        from kaolin.app.simulate.renderer import render_objects
        return render_objects([o.transformed_gaussians() for o in self._state.objects], camera)

    def _send_object_list(self, write_fn: Optional[Callable] = None):
        self._send_sync('object_list', {'objects': [o.to_dict() for o in self._state.objects]}, write_fn)

    def _send_sync(self, tag: str, content: dict, write_fn: Optional[Callable] = None):
        """Send a JSON-serialisable dict synchronously via the provided (or stored) write_fn."""
        fn = write_fn or self._write_fn
        if fn is None:
            return
        try:
            encoded = _ws_io.encode_message(tag, content, binary=False)
            fn(encoded, False)
        except Exception:
            logger.debug(f'WS send failed for tag={tag}', exc_info=True)

    @staticmethod
    def _schedule(loop: Optional[asyncio.AbstractEventLoop], fn):
        """Schedule a zero-arg callable on the tornado event loop from any thread."""
        if loop is not None:
            loop.call_soon_threadsafe(fn)
