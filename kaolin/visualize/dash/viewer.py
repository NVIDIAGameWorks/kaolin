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

from __future__ import annotations
import copy
import inspect
import logging
import math
from enum import unique
import re
from typing import Any, Callable, Dict, Iterable, Protocol, Union, runtime_checkable, get_args

from dash import dcc, html, clientside_callback, Input, Output
from dash.development.base_component import Component

import kaolin.render.easy_render
from kaolin.render.camera import Camera
from kaolin.render.easy_render import default_camera
from kaolin.visualize.dash.components.autogen.KaolinViewerInternal import KaolinViewerInternal
import kaolin.visualize.dash.auto_ui as auto_ui
from kaolin.visualize.dash.builtins import BehaviorLibrary, is_builtin_layer
from kaolin.visualize.dash.option import OptionKind, OptionSpec
from kaolin.visualize.web.naming import SequenceWithUniqueGlobalIds, UniqueIdGenerator

logger = logging.getLogger(__name__)

__all__ = [
    'ViewerBuilder',
    'KaolinViewer',
]


def _warn_fixed_aspect_size(width: int, height: int) -> None:
    """Warn when camera dimensions make exact-aspect resizing impractical."""
    if width <= 0 or height <= 0:
        logger.warning(
            'fixed_aspect: camera %dx%d must have positive width and height.',
            width, height)
        return
    divisor = math.gcd(width, height)
    min_width = width // divisor
    min_height = height // divisor
    if min_width > 50 or min_height > 50:
        logger.warning(
            'fixed_aspect: camera %dx%d has a smallest exact-aspect '
            'size of %dx%d pixels; resizing may be coarse or exceed '
            'small containers.',
            width, height, min_width, min_height)


def _read_valid_resize_modes() -> tuple[str, ...] | None:
    """Best-effort extraction of the valid ``resize_mode`` values from the
    auto-generated :class:`KaolinViewerInternal` component.

    The clientside ``resize_mode`` prop is typed as the ``ViewportResizeMode``
    TS enum, which Dash compiles into a Python ``Literal[...]`` annotation on
    the generated component's ``__init__``. Reading it here keeps the
    Python-side validation in lockstep with the clientside enum (single source
    of truth). Returns ``None`` if the annotation cannot be parsed so callers
    can skip validation rather than break.
    """
    try:
        # ``_explicitize_args`` wraps __init__, but the original annotations are
        # preserved on the signature. The annotation is Optional[Literal[...]],
        # i.e. Union[Literal[...], None]; unwrap one level, then the Literal.
        annotation = inspect.signature(KaolinViewerInternal.__init__).parameters['resize_mode'].annotation
        modes = get_args(get_args(annotation)[0])
        if not modes or not all(isinstance(m, str) for m in modes):
            raise ValueError(f'unexpected resize_mode annotation: {annotation!r}')
        return modes
    except Exception as e:
        logger.warning(f'Could not read valid resize modes from KaolinViewerInternal; '
                       f'skipping viewport_resize_mode validation ({e}).')
        return None


# Computed once at import; None when the annotation could not be read.
_VALID_RESIZE_MODES = _read_valid_resize_modes()


class _BehaviorOptionInfo:
    """Per-behavior bookkeeping for :meth:`ViewerBuilder.add_user_behavior_options`.
    """
    def __init__(self, behavior_id: str,
                 field_specs: list[auto_ui.FieldSpec],
                 triggers: list[Component] | None):
        self.behavior_id = behavior_id
        self.field_specs = field_specs
        self.triggers = triggers

class _ModeHooksInfo:
    def __init__(self, mode_name, clear_layer_ids=None, reset_behavior_ids=None):
        self.name = mode_name
        self.clear_layer_ids = clear_layer_ids if clear_layer_ids is not None else []
        self.reset_behavior_ids = reset_behavior_ids if reset_behavior_ids is not None else []


class ViewerBuilder:
    """Configure layers, behaviors, camera controls and connections for a viewer."""
    _VIEWER_ID_PLACEHOLDER = '<VIEWER_ID>'

    def __init__(self, camera: Camera = kaolin.render.easy_render.default_camera(500),
                 width: int | None = None,
                 height: int | None = None):
        """Construct and configure the client-side Kaolin Viewer.

        Viewport resizing defaults to ``'fixed'``. Use :meth:`set_resize_mode`
        to configure the resize mode and any associated cleanup together.

        Args:
            camera: Initial camera; the builder stores a copy.
            width: Optional override for the camera width.
            height: Optional override for the camera height.
        """
        self._camera = copy.deepcopy(camera)
        if width is not None:
            self._camera.width = width
        if height is not None:
            self._camera.height = height
        self._viewport_resize_mode = 'fixed'

        # The input ID might not be unique globally, so we ensure that class user
        # can use either their ID or unique-ified ID to refer to layers/behaviors
        self._layers = SequenceWithUniqueGlobalIds(ViewerBuilder._VIEWER_ID_PLACEHOLDER)
        self._behaviors = SequenceWithUniqueGlobalIds()
        self._ws_addresses = []
        self._camera_listeners = []
        # Behavior ids registered as camera controllers (only one active at a time).
        self._camera_controllers = []
        # Option broadcast rules: list of
        # (option_name, source_behavior_id|None, [target_behavior_id, ...]|None,
        #  [target_option_name, ...]|None), passed to the React viewer via the
        # `broadcast_behavior_options` prop. See :meth:`broadcast_behavior_option`.
        self._broadcast_behavior_options: list[tuple[str, str | None, list[str] | None, list[str] | None]] = []
        # Named modes: insertion-ordered list of
        # (mode_name, [unique_behavior_id, ...], ui_icon, description). The 4-tuple
        # is passed verbatim to the React viewer via the `modes` prop; the React
        # side auto-assigns digit shortcuts 1..9 in registration order.
        self._modes: list[tuple[str, list[str], str | None, str | None]] = []
        self._mode_hooks : dict[str, _ModeHooksInfo] = {}
        self._resize_hooks: dict[str, Any] | None = None
        # behavior_id -> _BehaviorOptionInfo, populated by add_user_behavior_options.
        self._behavior_option_infos: dict[str, _BehaviorOptionInfo] = {}
        # WebSocket message tag whose arrivals count as "frames" for the
        # in-viewer FPS overlay; None disables the overlay. Set by
        # :meth:`show_remote_fps`.
        self._remote_fps_source: str | None = None
        # Optional suffix for the FPS overlay (e.g. target rate from
        # :meth:`add_remote_rendering`).
        self._remote_fps_label: str | None = None

    @property
    def width(self):
        """Configured camera width before browser DPR adjustment."""
        return self._camera.width

    @property
    def height(self):
        """Configured camera height before browser DPR adjustment."""
        return self._camera.height

    def add_remote_rendering(self, active_layer_id, connection_id, cam_update_period = 100,
                            render_update_period = 200, tag_suffix = ''):
        """Wire up server-side remote rendering for a layer.

        The viewer sends camera updates (``set_camera{tag_suffix}``) and render
        requests (``render{tag_suffix}``) to the server, and draws the frames the
        server pushes back (tagged ``render{tag_suffix}``).

        Args:
            active_layer_id: Canvas layer identifier receiving rendered images.
            connection_id: Connection registered with :meth:`add_websocket_connection`.
            cam_update_period: Minimum interval between camera updates, in milliseconds.
            render_update_period: Minimum interval between render requests, in
                milliseconds; must be positive.
            tag_suffix: Suffix appended to the ``set_camera``/``render`` message
                tags. Leave empty for a single rendered viewer. Use a distinct
                suffix (e.g. ``'_result'``) when a page has more than one
                server-rendered viewer sharing the same WebSocket, so each
                viewer's camera/render stream is routed to its own
                :class:`~kaolin.visualize.web.sockets.AnyRendererMessageHandler`
                (constructed with the matching ``tag_suffix``).

        Returns:
            Tuple of behavior ids for image drawing, camera updates and render requests.
        """
        remote_draw_behavior = self.add_behavior(
            'draw_remote_image', active_layer_id=active_layer_id,
            options={'messageTag': f'render{tag_suffix}'})
        self.show_remote_fps(message_tag=f'render{tag_suffix}',
                             label='(target %0.1f)' % (1000 / render_update_period))
        send_cam_behavior_id = self.add_behavior(
            'send_camera',
            options={'connectionId': connection_id, 'minUpdateMs': cam_update_period,
                     'messageTag': f'set_camera{tag_suffix}'})
        request_render_behavior = self.add_behavior(
            'send_value',
            options={'connectionId': connection_id, 'minUpdateMs': render_update_period,
                     'messageTag': f'render{tag_suffix}', 'toSerializableFunctionName': 'kaolin.util.types.toEmptyString'})

        self.add_camera_listener(send_cam_behavior_id)
        self.add_camera_listener(request_render_behavior, setter='setValue')
        return remote_draw_behavior, send_cam_behavior_id, request_render_behavior

    def show_remote_fps(self, message_tag: str = 'render', label: str = '') -> None:
        """Enable a small in-viewer overlay showing the rate of incoming
        WebSocket messages tagged ``message_tag``.

        Intended for remote-rendering setups where the server pushes one
        message per rendered frame (default tag ``'render'``, matching
        :meth:`add_remote_rendering`). The React side maintains a
        sliding-window FPS tracker that treats long idle gaps as the end of a
        render burst, so the displayed rate reflects interactive responsiveness
        rather than being dragged toward zero when nothing is happening.

        Calling this more than once replaces the tag and label.

        Args:
            message_tag: Incoming message tag to count as rendered frames.
            label: Optional text displayed alongside the frame rate.
        """
        self._remote_fps_source = message_tag
        self._remote_fps_label = label

    def add_layer(self, tag='canvas', identifier=None, options=None):
        """ Adds a visible layer to the viewer. All layers are stacked over each other
        and take advantage of opacity. Viewer takes care of sizing and styling the layers,
        but extra options can be passed in, for example:
        { 'extraClasses': 'custom-css-class' }

        🔔 Returns **builder-level** layer identifier, which can only be used with
        the `ViewerBuilder` API, not for DOM manipulation -- as unique DOM layer ID will
        be created for every _built_ layer after calling `build`. To get the actual
        DOM element ID for a specific viewer (when no explicit `id` is supplied):
        ```
        builder_layer_id = builder.add_layer()
        viewer = builder.build()
        dom_layer_id = viewer.resolve_id(builder_layer_id)
        ```

        ☝

        Args:
            tag: Registered layer type, such as ``'canvas'``, ``'svg'`` or ``'div'``.
            identifier: Optional name unique within this builder.
            options: Layer properties. Built-in layers accept ``extraClasses``
                (additional CSS classes) and ``id`` (an explicit DOM id);
                canvas and SVG layers also accept initial ``width`` and ``height``.
                Prefer generated ids when building multiple viewers.

        Returns:
            A builder-level layer identifier.
        """
        is_builtin_layer(tag, warn=True)
        unique_id = self._layers.add(identifier, None, name_hint=tag)
        if options is None:
            options = {}

        if tag == 'canvas' and 'width' not in options and 'height' not in options:
            options['width'] = '%dpx' % self.width
            options['height'] = '%dpx' % self.height
        self._layers[unique_id] = (unique_id, tag, options)
        return unique_id

    @staticmethod
    def check_behavior_semantics(name, options=None):
        """Check a behavior name and its option names against registered metadata.

        Args:
            name: Registered behavior name.
            options: Optional mapping of option names to values.

        Returns:
            False, with a logged warning, for an unknown behavior or option name
            absent from a nonempty schema; True otherwise. Values are not validated.
        """
        meta = BehaviorLibrary.meta(name)
        if meta is None:
            logger.warning(f'Python BehaviorLibrary does not know about behavior {name}; could be missing from clientside code')
            logger.info(f'Registered behaviors: {BehaviorLibrary.to_string()}')
            return False

        expected_options = meta.options
        if options and expected_options:
            unexpected_options = set(options.keys()).difference(set(expected_options.keys()))
            if len(unexpected_options) > 0:
                logger.warning(f'Behavior {name} configured with unexpected options: {unexpected_options}')
                logger.info(f'Registered behavior spec: {BehaviorLibrary.behavior_to_string(name, detailed=True)}')
                return False
        return True

    def _behavior_control_options(self, name, initial_options, user_options) -> list[OptionSpec]:
        meta = BehaviorLibrary.meta(name)
        default_options = meta.options if meta else {}  # dict[str, OptionSpec]

        def _warn_unknown(k):
            if len(default_options) > 0:
                logger.warning(f'Registered behavior {name} does not have a registered option name {k}; proceeding anyway.')
                logger.info(f'Registered behavior spec: {BehaviorLibrary.behavior_to_string(name, detailed=True)}')

        final_options: list[OptionSpec] = []
        if user_options is None:
            if len(default_options) == 0:
                logger.warning(f'Behavior {name} does not have a registered schema; must supply manual option annotations')
            for k, spec in default_options.items():
                if spec.ui_bound:
                    final_options.append(spec.merged())
        else:
            for user_option in user_options:
                if isinstance(user_option, OptionSpec):
                    # A full spec; its own name is the key (it may override a
                    # manifest option of the same name or define a new one).
                    if user_option.name not in default_options:
                        _warn_unknown(user_option.name)
                    final_options.append(user_option)
                else:
                    # A bare option name: reuse the manifest spec, or fall back to
                    # an unconstrained `OptionKind.ANY` control if there is none.
                    k = user_option
                    if k in default_options:
                        final_options.append(default_options[k].merged())
                    else:
                        _warn_unknown(k)
                        final_options.append(OptionSpec(name=k, kind=OptionKind.ANY))

        out: list[OptionSpec] = []
        for spec in final_options:
            if spec.name in initial_options:
                spec = spec.merged({'default': initial_options[spec.name]})
            out.append(spec)
        return out

    def user_behavior_options_spec(
            self, behavior_id,
            options: Iterable[str | OptionSpec] | None = None) -> list[OptionSpec]:
        """Return option specifications for a registered behavior.

        Args:
            behavior_id: Original or unique id of a behavior added to this builder.
            options: Option names or explicit :class:`OptionSpec` instances, in
                display order. None selects the registered UI-bound options.
                Unknown names receive an unconstrained specification.

        Returns:
            Option specifications with defaults from the behavior's initial options.

        Raises:
            KeyError: If the behavior id is unknown.
        """
        _, name, _, initial_options, _ = self._behaviors[behavior_id]
        return self._behavior_control_options(name, initial_options, options)

    def add_user_behavior_options(
            self,
            behavior_id: str,
            options: Iterable[str | OptionSpec] | None = None):
        """Generate auto-UI controls for a behavior's options and persist them.

        Returns a list of Dash components (controls plus hidden callback targets)
        that the caller is responsible for inserting into the app layout. The
        controls use per-tab session persistence for the user's edits:

        - On every control change, a clientside callback calls
          ``kaolin.core.event.requestBehaviorSetOption(behavior_id, name, value, viewer_id)``,
          so the JS behavior updates without a server roundtrip.
        - On tab reload, Dash restores the controls from sessionStorage; the
          initial clientside callbacks replay the restored values to the JS
          behavior via ``requestBehaviorSetOption``.

        Call :meth:`build` to bind the controls to the viewer. Initial behavior
        options supply the control defaults.

        Args:
            behavior_id: id returned by :meth:`add_behavior`.
            options: which options to expose, as a list whose entries are either a
                bare option name (``str``) -- reusing the behavior's manifest spec
                -- or a full :class:`~kaolin.visualize.dash.option.OptionSpec`
                (keyed by its own ``name``, to override a manifest option or define
                one for a *user* behavior with no manifest). ``None`` exposes all
                ``uiBound`` manifest options.

        Returns:
            Dash components to insert together into the layout, or an empty list
            when no options are selected.

        Raises:
            KeyError: If the behavior id is unknown.
        """
        name = self._behaviors[behavior_id][1]
        specs = self.user_behavior_options_spec(behavior_id, options)

        if len(specs) == 0:
            logger.warning(f'Behavior {behavior_id} ({name}) does not have uiBound options')
            return []

        controls, field_specs = auto_ui.make_controls(
            specs, id_prefix=f'beh-{behavior_id}', persistence_type='session')

        # store_id = UniqueIdGenerator.get_unique_id(f'options-{behavior_id}')
        # store = dcc.Store(id=store_id, data={v[0]: v[1] for v in final_options}, storage_type='session')
        # Dummy output to anchor the JS execution
        triggers = [html.Div(
                id=UniqueIdGenerator.get_unique_id(f'js-trigger-{behavior_id}-{fs.name}'),
                style={'display': 'none'}) for fs in field_specs]

        self._behavior_option_infos[behavior_id] = _BehaviorOptionInfo(behavior_id, field_specs, triggers)
        return [*controls, *triggers]

    def add_behavior(self, name, identifier=None, active_layer_id=None, options=None, is_active=True):
        """Add a registered client-side behavior to the viewer.

        Args:
            name: Behavior name registered by the library or application.
            identifier: Optional name unique within this builder.
            active_layer_id: Layer identifier to bind, ``'EVENT_CANVAS'`` for the
                viewer's input surface, or None for an unbound behavior.
            options: Initial behavior options. Unknown names produce a warning.
            is_active: Whether the behavior starts active.

        Returns:
            The unique behavior identifier.

        Raises:
            ValueError: If the layer identifier is unknown.
            RuntimeError: If the behavior identifier is already used.
        """
        self.check_behavior_semantics(name, options)
        unique_id = self._behaviors.add(identifier, None, name_hint=name)

        unique_layer_id = None
        if active_layer_id == 'EVENT_CANVAS':
            unique_layer_id = active_layer_id
        elif active_layer_id is not None:
            unique_layer_id = self._layers.get_unique_id_or_raise(active_layer_id)

        if options is None:
            options = {}

        self._behaviors[unique_id] = (unique_id, name, unique_layer_id, options, is_active)
        return unique_id

    def add_websocket_connection(self, address, connection_id):
        """Add a WebSocket connection available to viewer behaviors.

        Args:
            address: WebSocket URL or server-relative endpoint path.
            connection_id: Identifier used by behaviors to select this connection.
        """
        self._ws_addresses.append((address, connection_id))

    def add_camera_controller(self, controller_behavior_name='threejs_camera_orbit', options=None,
                              is_active=True):
        """Configure the viewer with a camera and add a camera-controller behavior.

        The controller owns the camera client-side (in its native format) and
        implements the camera-controller API. May be called more than once to
        register multiple controllers; only one is active at a time and the
        active controller is dispatched to camera listeners.

        Args:
            controller_behavior_name: registered controller behavior name.
            options: optional dict of behavior options for the controller, passed
                through to :meth:`add_behavior` (unknown option names are warned
                about when a schema is available). For the default
                ``threejs_camera_orbit`` this exposes the three.js
                ``OrbitControls`` parameters (e.g. ``enableDamping``,
                ``rotateSpeed``, ``screenSpacePanning``) as well as the world
                ``up`` direction as ``[x, y, z]`` (defaults to ``[0, 0, 1]``).
            is_active: whether this controller starts active.

        Returns:
            The unique behavior id of the added controller.
        """
        cam_behavior_id = self.add_behavior(controller_behavior_name, active_layer_id='EVENT_CANVAS',
                                            options=options, is_active=is_active)
        self._camera_controllers.append(cam_behavior_id)
        return cam_behavior_id


    def add_camera_listener(self, behavior_id, setter='setCamera', use_raw_object=False):
        """Send active-camera updates to an active behavior.

        Args:
            behavior_id: Original or unique id of a registered behavior.
            setter: Behavior method receiving the camera on each update.
            use_raw_object: If True, pass the controller's native camera object;
                otherwise pass serialized Kaolin camera parameters.

        Raises:
            ValueError: If the behavior id is unknown.
        """
        unique_behavior_id = self._behaviors.get_unique_id_or_raise(behavior_id)
        self._camera_listeners.append((unique_behavior_id, setter, use_raw_object))

    def broadcast_behavior_option(self, option_name: str, source_behavior_id: str | None = None,
                                  target_behavior_ids: str | list[str] | None = None,
                                  target_option_names: str | list[str] | None = None):
        """Register a rule that mirrors an option change on one behavior onto others.

        Whenever a behavior receives ``setOption(option_name, value)`` in the
        browser, the value is also written to the broadcast targets. This powers,
        for example, a single UI control that drives the same option across
        several behaviors (e.g. one color picker setting the color of every
        drawing behavior).

        Args:
            option_name: the option whose change triggers the broadcast, or
                ``'*'`` to match every option change (the changed option's name is
                reused on each target, so ``target_option_names`` must be ``None``).
            source_behavior_id: if given, only changes originating from this
                behavior are broadcast; if ``None``, a change on any behavior is.
                Resolved to a unique behavior id (raises if unknown).
            target_behavior_ids: behavior id or list of ids that receive the
                value. If ``None`` (the default), the value is broadcast to all
                *other* behaviors, and each write is best-effort (a target that
                does not accept the option is skipped rather than erroring).
                When ids are listed explicitly, a failed write surfaces as an
                error.
            target_option_names: option name, or list aligned with
                ``target_behavior_ids``, to set on each target. If ``None``,
                ``option_name`` is reused on every target. Requires
                ``target_behavior_ids`` to be set.

        Returns:
            None.
        """
        # Option names come from a behavior's option schema, so they are plain
        # identifiers. Reject anything else (other than the '*' wildcard) so users
        # don't assume glob/regex patterns are supported.
        if option_name != '*' and not re.fullmatch(r'[A-Za-z_$][A-Za-z0-9_$]*', option_name):
            raise ValueError(
                f'broadcast_behavior_option: option_name {option_name!r} is not a valid option name; '
                "pass a single schema option name, or '*' to match every option "
                '(glob/regex patterns are not supported).')

        if source_behavior_id is not None:
            source_behavior_id = self._behaviors.get_unique_id_or_raise(source_behavior_id)

        if target_option_names is not None and target_behavior_ids is None:
            raise ValueError(
                'broadcast_behavior_option: target_option_names requires target_behavior_ids to be set.')

        if option_name == '*' and target_option_names is not None:
            raise ValueError(
                "broadcast_behavior_option: target_option_names is not allowed with the '*' wildcard "
                '(the changed option name is reused on each target).')

        if isinstance(target_behavior_ids, str):
            target_behavior_ids = [target_behavior_ids]
        if target_behavior_ids is not None:
            target_behavior_ids = self._behaviors.get_unique_ids_or_raise(target_behavior_ids)

        if isinstance(target_option_names, str):
            target_option_names = [target_option_names for _ in target_behavior_ids]
        if target_option_names is not None and len(target_option_names) != len(target_behavior_ids):
            raise ValueError(
                'broadcast_behavior_option: target_option_names and target_behavior_ids must have the same '
                f'length (got {len(target_option_names)} and {len(target_behavior_ids)}).')

        self._broadcast_behavior_options.append(
            (option_name, source_behavior_id, target_behavior_ids, target_option_names))

    def add_mode(self, name: str, active_behavior_ids,
                 ui_icon: str | None = None,
                 description: str | None = None,
                 clear_layers: list | None = None,
                 reset_behaviors: list | None = None) -> str:
        """Register a named mode: the subset of behaviors to activate together.

        Every entry in ``active_behavior_ids`` must already be a registered behavior
        (added via :meth:`add_behavior`); the ids may be the original input ids or
        the unique ids returned by :meth:`add_behavior`. Each is resolved to its
        unique id before being stored.

        Apps switch between registered modes at runtime by dispatching
        ``kaolin.core.event.requestMode(name)`` from clientside callbacks.
        Every registered behavior outside the active mode is deactivated,
        including behaviors that are not assigned to any mode.

        Args:
            name: unique mode name (raises ``ValueError`` on duplicate).
            active_behavior_ids: iterable of behavior ids to activate in this mode.
            ui_icon: optional Bootstrap Icons class such as ``'bi-binoculars'``.
                When supplied, the viewer renders a tiny button for this mode in
                its top-left overlay; the button is highlighted in NVIDIA green
                while this mode is active.
            description: optional human-readable label used as the tooltip on the
                overlay button. The auto-assigned digit shortcut (if any) is
                appended to the tooltip automatically.
            clear_layers: optional iterable of layer ids whose contents should be
                cleared every time this mode becomes active. Currently only
                canvas layers are supported; other layer types are skipped
                with a logged message.
            reset_behaviors: optional iterable of behavior ids whose ``reset()``
                method should be invoked every time this mode becomes active.
                Equivalent to ``kaolin.core.event.requestBehaviorReset(id)``.

        The first nine modes registered (in add order) are bound to the digit
        keys ``1``..``9`` on desktop; the shortcut fires only while the mouse is
        over this viewer, so multiple viewers on the same page can coexist with
        independent shortcuts.

        Returns:
            The mode ``name`` (for chaining).
        """
        if any(entry[0] == name for entry in self._modes):
            raise ValueError(f'Mode "{name}" is already registered')
        unique_ids = self._behaviors.get_unique_ids_or_raise(active_behavior_ids)
        self._modes.append((name, unique_ids, ui_icon, description))

        if clear_layers is not None or reset_behaviors is not None:
            clear_layers = self._layers.get_unique_ids_or_raise(clear_layers) if clear_layers is not None else []
            reset_behaviors = self._behaviors.get_unique_ids_or_raise(reset_behaviors) if reset_behaviors is not None else []
            self._mode_hooks[name] = _ModeHooksInfo(name, clear_layers, reset_behaviors)
        return name

    def set_resize_mode(self, viewport_resize_mode: str,
                        clear_layers: list | None = None,
                        reset_behaviors: list | None = None,
                        switch_mode: str = "") -> None:
        """Configure viewport resizing and optional cleanup after size changes.

        Cleanup runs after layers and behaviors receive the new size, before
        the camera update. Initial sizing and unchanged sizes do not trigger
        cleanup. Register referenced layers, behaviors and modes before calling.
        Calling again replaces both the resize mode and its cleanup hooks.

        Args:
            viewport_resize_mode: One of ``'fixed'``, ``'adaptive'``, or
                ``'fixed_aspect'``; see :ref:`rubric viewport resize modes`.
            clear_layers: Canvas layer ids to clear, or None to clear none.
            reset_behaviors: Behavior ids whose reset() method is called.
            switch_mode: Mode to activate after cleanup; "" keeps the mode.

        Raises:
            ValueError: For an unsupported resize mode, unknown layer/behavior
                ids, an unknown switch mode, or a non-canvas clear target.

        .. _rubric viewport resize modes:

        .. rubric:: Viewport Resize Modes

        Controls how the viewport and camera are resized with the viewport container size changes. The behavior
        of the viewport container itself is **not** controlled by this configuration, but by the overall
        app layout. What this setting controls is how viewport and camera respond to the available space.

        Note: When using
        server-side rendering, care must be taken to configure `AnyRendererMessageHandler` to properly
        adjust actual rendering resolution, to avoid extreme latency on larger screens.

        ``'fixed'`` (default)
        Original aspect ratio comes from the configured Camera.
        The canvas display box scales to fit within the available space while keeping its aspect ratio fixed.
        The canvas buffer and camera pixel dimensions are the configured Camera dimensions
        multiplied by the browser device pixel ratio (DPR), rounded down to integers.
        These DPR-adjusted dimensions remain fixed as the container resizes.

        ``'fixed_aspect'``
        Original aspect ratio comes from the configured Camera.
        Sizes use whole multiples of its reduced integer ratio. A minimum of one
        multiple may exceed small containers. A warning is logged if either
        reduced dimension exceeds 50 pixels or a camera dimension is nonpositive.
        The canvas display box scales to fit within the available space while keeping its aspect ratio fixed.
        The canvas buffer and camera dimensions follow the display box size,
        adjusted by DPR and rounded down to integers.

        ``'adaptive'``
        The canvas display box scales to fit within the available space.
        The canvas buffer and camera dimensions follow the display box size,
        adjusted by DPR and rounded down to integers.
        """
        # Validate against the generated component unless its annotation could
        # not be read (see _read_valid_resize_modes).
        if _VALID_RESIZE_MODES is not None and viewport_resize_mode not in _VALID_RESIZE_MODES:
            raise ValueError(
                f'Unsupported viewport_resize_mode {viewport_resize_mode!r}; '
                f'expected one of {_VALID_RESIZE_MODES}.')
        layer_ids = self._layers.get_unique_ids_or_raise(clear_layers or [])
        behavior_ids = self._behaviors.get_unique_ids_or_raise(reset_behaviors or [])
        for layer_id in layer_ids:
            if self._layers[layer_id][1] != 'canvas':
                raise ValueError('set_resize_mode clear_layers supports only canvas layers; '
                                 'use reset_behaviors for other layer types.')
        if switch_mode and not any(mode[0] == switch_mode for mode in self._modes):
            raise ValueError(f'Unknown resize switch_mode {switch_mode!r}.')
        if viewport_resize_mode == 'fixed_aspect':
            _warn_fixed_aspect_size(self.width, self.height)
        self._viewport_resize_mode = viewport_resize_mode
        self._resize_hooks = dict(clear_layers=layer_ids,
                                  reset_behaviors=behavior_ids,
                                  switch_mode=switch_mode)

    def set_active_mode(self, name: str) -> None:
        """Set the initial active mode for the viewer.

        This is a build-time convenience that rewrites the ``is_active`` flag on
        every registered behavior tuple so that the per-behavior props passed to
        the React viewer reflect the requested mode. Every registered behavior is
        touched: those listed in ``name`` are flagged ``True``, all others
        ``False`` — matching the runtime semantics of ``requestMode`` on the
        client side.

        Args:
            name: a mode name previously registered via :meth:`add_mode`.

        Raises:
            ValueError: If the mode is unknown.
        """
        target = None
        for entry in self._modes:
            mode_name, ids = entry[0], entry[1]
            if mode_name == name:
                target = set(ids)
                break
        if target is None:
            raise ValueError(f'Mode "{name}" is not registered. Known modes: '
                             f'{[entry[0] for entry in self._modes]}')
        for unique_id in self._behaviors.unique_ids:
            uid, behavior_name, layer_id, options, _is_active = self._behaviors[unique_id]
            self._behaviors[unique_id] = (uid, behavior_name, layer_id, options, unique_id in target)

    @staticmethod
    def resolve_id_for_viewer(builder_level_value, viewer_id):
       """Resolve a builder-level layer reference for a particular viewer.

       Args:
           builder_level_value: Layer reference or other value to resolve.
           viewer_id: Id of the built viewer.

       Returns:
           The viewer-specific reference, or the original value if no resolution
           is needed. Explicit custom DOM ids are returned unchanged.
       """
       if type(builder_level_value) is str and ViewerBuilder._VIEWER_ID_PLACEHOLDER in builder_level_value:
           return re.sub(ViewerBuilder._VIEWER_ID_PLACEHOLDER, viewer_id, builder_level_value)
       else:
           return builder_level_value

    def _generate_mode_hooks_js(self, viewer_id: str) -> str:
        """Generate the body of a clientside callback that runs the per-mode
        cleanup hooks registered via :meth:`add_mode`'s ``clear_layers`` /
        ``reset_behaviors`` arguments.

        The returned string is the inner body of a JS function whose single
        argument ``activeMode`` is the new value of the viewer's ``active_mode``
        prop. Returns an empty string when no hooks are registered or every
        registered hook is empty (e.g. only non-canvas layers were specified).
        """
        per_mode_blocks: list[str] = []
        for mode_name, hook_info in self._mode_hooks.items():
            stmts: list[str] = []
            for layer_uid in hook_info.clear_layer_ids:
                unique_id, tag, _ = self._layers[layer_uid]
                if tag == "canvas":
                    elem_id = self.resolve_id_for_viewer(unique_id, viewer_id)
                    stmts.append(
                        f'kaolin.util.canvas.clearCanvas("{elem_id}");')
                else:
                    logger.error(f'Clearing layer of type {tag} not implemented; create manual mode hook')
            for behavior_uid in hook_info.reset_behavior_ids:
                stmts.append(
                    f'kaolin.core.event.requestBehaviorReset('
                    f'"{behavior_uid}", undefined, "{viewer_id}");')
            if not stmts:
                continue
            indented = '\n            '.join(stmts)
            per_mode_blocks.append(
                f'if (activeMode === "{mode_name}") {{\n            {indented}\n        }}')
        return '\n        else '.join(per_mode_blocks)

    def _layer_spec_for_viewer(self, layer_spec, viewer_id):
        unique_id, tag, options = layer_spec
        options = copy.copy(options)
        if 'id' not in options:
            options['id'] = ViewerBuilder.resolve_id_for_viewer(unique_id, viewer_id)

        if 'width' not in options:
            options['width'] = self.width
        if 'height' not in options:
            options['height'] = self.height
        return unique_id, tag, options

    def _behavior_spec_for_viewer(self, behavior_spec, viewer_id):
        unique_id, name, unique_layer_id, options, is_active = behavior_spec
        options = copy.copy(options)
        for k, v in options.items():
            if type(v) is str and ViewerBuilder._VIEWER_ID_PLACEHOLDER in v:
                options[k] = ViewerBuilder.resolve_id_for_viewer(v, viewer_id)

        return unique_id, name, unique_layer_id, options, is_active

    def build(self) -> KaolinViewer:
        """Create a viewer with the configured layers, behaviors and connections.

        Each call creates a new viewer id. Add the returned component and any
        components from :meth:`add_user_behavior_options` to the app layout.

        Returns:
            A :class:`KaolinViewer` Dash component.
        """
        my_id = UniqueIdGenerator.get_unique_id('viewer', prefix='kaolin')

        # # Bind any present options to behavior instances of each built viewer
        for behavior_id, options_info in self._behavior_option_infos.items():
            auto_ui.bind_controls_to_behavior_clientside(
                options_info.field_specs, behavior_id=behavior_id, viewer_id=my_id, triggers=options_info.triggers)

        # Per-mode cleanup hooks: emit a clientside callback that listens to
        # the viewer's `active_mode` prop and runs the registered clear/reset
        # actions for whichever mode is now active. We write back to
        # `active_mode` with `dash_clientside.no_update` purely to satisfy
        # Dash's "every callback needs an Output" rule without requiring the
        # caller to add an extra hidden Div to their layout.
        mode_hooks_body = self._generate_mode_hooks_js(my_id)
        if mode_hooks_body:
            clientside_callback(
                f"""
                function(activeMode) {{
                    {mode_hooks_body}
                    return window.dash_clientside.no_update;
                }}
                """,
                Output(my_id, 'active_mode', allow_duplicate=True),
                Input(my_id, 'active_mode'),
                prevent_initial_call=True,
            )

        kwargs: dict[str, Any] = dict(
            websocket_addresses=self._ws_addresses,
            id=my_id,
            camera_parameters=self._camera.as_dict(),
            layers=[self._layer_spec_for_viewer(x, my_id) for x in self._layers.items],
            behaviors=[self._behavior_spec_for_viewer(x, my_id) for x in self._behaviors.items],
            camera_listeners=self._camera_listeners,
            camera_controllers=self._camera_controllers,
            resize_mode=self._viewport_resize_mode,
            modes=self._modes)
        if self._resize_hooks is not None:
            kwargs['resize_hooks'] = self._resize_hooks
        if self._remote_fps_source is not None:
            kwargs['remote_fps_source'] = self._remote_fps_source
        if self._remote_fps_label is not None:
            kwargs['remote_fps_label'] = self._remote_fps_label
        if self._broadcast_behavior_options:
            kwargs['broadcast_behavior_options'] = self._broadcast_behavior_options
        return KaolinViewer(**kwargs)


class KaolinViewer(KaolinViewerInternal):
    """Dash viewer component with layer-id resolution.

    Use :class:`ViewerBuilder` to configure layers and behaviors. Direct
    construction accepts the underlying Dash component properties and requires
    an ``id`` unique within the app layout.
    """
    def __init__(self, *args, **kwargs):
        """Create a viewer from Dash component properties.

        Args:
            *args: Positional component properties.
            **kwargs: Component properties, including a required unique ``id``.
        """
        super().__init__(*args, **kwargs)
        self.id = kwargs.get('id', None)
        assert self.id is not None, f'ID cannot be None'

    def resolve_id(self, layer_identifier):
        """Return the DOM id for a builder-level layer identifier.

        Args:
            layer_identifier: Identifier returned by :meth:`ViewerBuilder.add_layer`.

        Returns:
            The layer's id for this viewer. If the layer has an explicit ``id``
            option, use that custom id directly instead.
        """
        return ViewerBuilder.resolve_id_for_viewer(layer_identifier, self.id)


