from __future__ import annotations
import copy
import inspect
import logging
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
    _VIEWER_ID_PLACEHOLDER = '<VIEWER_ID>'

    def __init__(self, camera: Camera = kaolin.render.easy_render.default_camera(500),
                 width: int | None = None,
                 height: int | None = None,
                 viewport_resize_mode: str='fixed'):
        self._camera = copy.deepcopy(camera)
        if width is not None:
            self._camera.width = width
        if height is not None:
            self._camera.height = height
        # How the viewport canvas resizes within its container; one of
        # 'fixed', 'adaptive', 'fix_aspect' (see ViewportResizeMode clientside).
        # Validated against the values compiled into the generated component,
        # unless that annotation could not be read (see _read_valid_resize_modes).
        if _VALID_RESIZE_MODES is not None and viewport_resize_mode not in _VALID_RESIZE_MODES:
            raise ValueError(
                f'Unsupported viewport_resize_mode {viewport_resize_mode!r}; '
                f'expected one of {_VALID_RESIZE_MODES}.')
        self._viewport_resize_mode = viewport_resize_mode

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
        return self._camera.width

    @property
    def height(self):
        return self._camera.height

    def add_remote_rendering(self, active_layer_id, connection_id, cam_update_period = 100,
                            render_update_period = 200, tag_suffix = ''):
        """Wire up server-side remote rendering for a layer.

        The viewer sends camera updates (``set_camera{tag_suffix}``) and render
        requests (``render{tag_suffix}``) to the server, and draws the frames the
        server pushes back (tagged ``render{tag_suffix}``).

        Args:
            tag_suffix: Suffix appended to the ``set_camera``/``render`` message
                tags. Leave empty for a single rendered viewer. Use a distinct
                suffix (e.g. ``'_result'``) when a page has more than one
                server-rendered viewer sharing the same WebSocket, so each
                viewer's camera/render stream is routed to its own
                :class:`~kaolin.visualize.web.sockets.AnyRendererMessageHandler`
                (constructed with the matching ``tag_suffix``).
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

        Calling this more than once simply overwrites the tag.
        """
        self._remote_fps_source = message_tag
        self._remote_fps_label = label

    def add_layer(self, tag='canvas', identifier=None, options=None):
        """ Adds a visible layer to the viewer. All layers are stacked over each other
        and take advantage of opacity. Viewer takes care of sizing and styling the layers,
        but extra options can be passed in, currently limited to:
        { 'extraClasses': 'custom-css-class' }

        🔔 Returns **builder-level** layer identifier, which can only be used with
        the `ViewerBuilder` API, not for DOM manipulation -- as unique DOM layer ID will
        be created for every _built_ layer after calling `build`. To get the actual
        DOM element ID for a specific viewer:
        ```
        builder_layer_id = builder.add_layer()
        viewer = builder.build()
        dom_layer_id = viewer.resolve_id(builder_layer_id)
        ```

        # TODO: document what options can be passed in
        # TODO: register documentation or warn, like for behaviors

        ☝

        Args:
            tag:
            identifier:
            options:

        Returns:

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
        _, name, _, initial_options, _ = self._behaviors[behavior_id]
        return self._behavior_control_options(name, initial_options, options)

    def add_user_behavior_options(
            self,
            behavior_id: str,
            options: Iterable[str | OptionSpec] | None = None):
        """Generate auto-UI controls for a behavior's options and persist them.

            Returns a list of Dash components (controls plus a hidden
            :class:`dash.dcc.Store(storage_type='session')`) that the caller is
            responsible for inserting into the app layout. The Store backs the
            per-tab persistence of the user's edits:

            - On every control change, a clientside callback merges the new value
              into the Store *and* calls
              ``kaolin.core.event.requestBehaviorSetOption(behavior_id, name, value, viewer_id)``,
              so the JS behavior updates without a server roundtrip.
            - On tab reload, the Store hydrates from sessionStorage; a one-shot
              hydration callback restores the matching UI controls via
              ``window.dash_clientside.set_props`` and replays each cached option
              to the JS behavior via ``requestBehaviorSetOption``.

              # TODO: finish doc and fix its formatting

        Args:
            behavior_id: id returned by :meth:`add_behavior`.
            options: which options to expose, as a list whose entries are either a
                bare option name (``str``) -- reusing the behavior's manifest spec
                -- or a full :class:`~kaolin.visualize.dash.option.OptionSpec`
                (keyed by its own ``name``, to override a manifest option or define
                one for a *user* behavior with no manifest). ``None`` exposes all
                ``uiBound`` manifest options.
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
                through to :meth:`add_behavior` (and validated against the
                behavior's registered option schema). For the default
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
        """

        Args:
            behavior_id:
            setter:
            use_raw_object:

        Returns:

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
        Behaviors not listed in the active mode are deactivated; behaviors not
        registered with any mode are still only affected by the explicit mode
        switch (i.e. any registered behavior outside the target mode becomes
        inactive) — manage them with :meth:`add_behavior` ``is_active=False`` if
        they should not be touched by mode changes.

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
                canvas layers are supported (cleared via
                ``kaolin.util.canvas.clearCanvas``); other layer types
                are skipped with a warning at build time.
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
        if self._remote_fps_source is not None:
            kwargs['remote_fps_source'] = self._remote_fps_source
        if self._remote_fps_label is not None:
            kwargs['remote_fps_label'] = self._remote_fps_label
        if self._broadcast_behavior_options:
            kwargs['broadcast_behavior_options'] = self._broadcast_behavior_options
        return KaolinViewer(**kwargs)


class KaolinViewer(KaolinViewerInternal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.id = kwargs.get('id', None)
        assert self.id is not None, f'ID cannot be None'

    def resolve_id(self, layer_identifier):
        return ViewerBuilder.resolve_id_for_viewer(layer_identifier, self.id)





# class KaolinViewer:
#     r"""
#     This 3D Viewer class can be used with Plotly Dash <https://dash.plotly.com/> to create
#     complex interactive 3D or 2D Web applications with complicated client-server communication.
#     This class is extensively configurable and can handle any custom client-server
#     communication through WebSockets or (TODO) other protocols. The primary logic of this
#     component resides on the client side as a React component, but it can be configred in
#     many different ways using this python API.
#
#     .. _rubric overview:
#
#     .. rubric:: Overview
#
#     The primary purpose of `KaolinViewer` is to provide an interactive 2D or 3D space.
#     The viewer UI is composed of several aligned and stacked **layers**, which can include
#     HTML canvas elements, SVGs, or others. This space accepts any touch or mouse
#     events, key events, and modifier keys. Separately, the viewer is configured with
#     **behaviors**, which map user interactions or events to actions, such as layer manipulations.
#     All layers and behaviors _run on the client side_, however, the behaviors can
#     facilitate _complex custom client-server communication_. See examples below.
#
#     .. _rubric:: Programming Client Side Behavior:
#
#     We support defining custom behaviors and client-side callbacks. When launching a dash
#     app using `KaolinViewer`, the kaolin _client-side_ utility functions in Javascript will be
#     exposed under `kaolin` namespace inside the browser. Review the Kaolin Library
#     javascript documentation to see a variety of useful functions. Notably, we can encode
#     decode, and compress custom binary messages, and support various canvas and svg
#     manipulations.
#
#     .. _rubric examples:
#
#     .. rubric:: Examples
#
#     Example 1: Server-side rendering with a custom camera control behavior.
#
#     .. code-block:: python
#
#         from kaolin.visualize.dash import KaolinViewer
#         viewer = KaolinViewer(id='my-viewer', socket_handlers=[], components=[], camera=camera)
#         viewer.show()
#
#
#     """
#     def __init__(self, id, socket_handlers, components, camera):
#         self.id = id
#         self.socket_handlers = [x for x in socket_handlers]
#         self.components = [x for x in components]
#         self.camera = camera
#
#     @staticmethod
#     def convenience_init_v0(camera: Camera = None, ws_address=None):
#         my_id ="kaolin-viewer"
#         if camera is None:
#             camera = kaolin.render.easy_render.default_camera()
#
#         component = KaolinViewerInternal(
#             websocket_addresses=[(ws_address, 'main-ws')],
#             id=my_id,
#             camera_parameters=camera.as_dict(),
#             layers=[
#                 ('gsplats-board', 'canvas', {"width": "500px", "height": "500px"}),
#                 #('gsplats-board2', 'canvas', {"width": "500px", "height": "500px"}),
#                 ('draw-board1', 'canvas', {"width": "500px", "height": "500px"}),
#                 ('draw-board2', 'canvas', {"width": "500px", "height": "500px"}),
#                 ('svg-board', 'svg', {'width': '1000', 'height': '1000'}),
#                 #('play-board', 'canvas', {"width": "500px", "height": "500px"})
#                 ],
#             behaviors=[
#                 ('draw-1', 'drawing', 'draw-board1', {"color": "#ffcc00"}),
#                 ('draw-2', 'drawing', 'draw-board2', {"thickness": 4}),
#                 ('svg', 'svg_annotation', 'svg-board', {}),
#                 ('gsplat-render', 'draw_remote_image', 'gsplats-board', {}),
#                 ('send-cam', 'send_camera', None, {'connectionId': 'main-ws', 'minUpdateMs': 100})
#                 #('image-draw-component', 'draw_canvas_image_react', 'gsplat-board2', {})
#                 #('play', 'playcanvas_splat_js', 'play-board', {})
#                 ]
#         )
#         return KaolinViewer(my_id, [], [component], camera=camera)
#
#     @staticmethod
#     def convenience_init(camera: Camera = None, ws_address=None):
#         my_id = "kaolin-viewer"
#         if camera is None:
#             camera = kaolin.render.easy_render.default_camera()
#
#         component = KaolinViewerInternal(
#             websocket_addresses=[ws_address],
#             id=my_id,
#             camera_parameters=camera.as_dict(),
#             layers=[
#                 ('gsplats-board', 'canvas', {"width": "500px", "height": "500px"}),
#                 # ('gsplats-board2', 'canvas', {"width": "500px", "height": "500px"}),
#                 ('mesh-board', 'canvas', {"width": "500px", "height": "500px"}),
#                 ('draw-board1', 'canvas', {"width": "500px", "height": "500px"}),
#                 ('draw-board2', 'canvas', {"width": "500px", "height": "500px"}),
#                 ('svg-board', 'svg', {'width': '1000', 'height': '1000'}),
#                 ('play-board', 'canvas', {"width": "500px", "height": "500px"})
#             ],
#             behaviors=[
#                 ('draw-1', 'drawing', 'draw-board1', {"color": "#ffcc00"}),
#                 ('draw-2', 'drawing', 'draw-board2', {"thickness": 4}),
#                 ('svg', 'svg_annotation', 'svg-board', {}),
#                 ('gsplat-render', 'draw_remote_image', 'gsplats-board', {}),
#                 ('three-render', 'three_render',  'mesh-board', {}),
#                 ('receive-mesh', 'receive_message', None,
#                     {"tag": "set_mesh",
#                      "msgProcessFunctionName": "kaolin.graphics.threejs.meshFromMessage",
#                      "alertBehaviors": [("three-render", "setMesh")]}),
#                 # ('image-draw-component', 'draw_canvas_image_react', 'gsplat-board2', {})
#                 ('play', 'playcanvas_splat', 'play-board', {})
#             ]
#         )
#         return KaolinViewer(my_id, [], [component], camera=camera)
#
#
#     def make_server_renderer(self):
#         pass
    
    # def __call__(self, *args, **kwargs):
    #     """Make the wrapper callable like the original component."""
    #     return self.component(*args, **kwargs)
    
    # def __getattr__(self, name):
    #     """Delegate attribute access to the underlying component."""
    #     return getattr(self.component, name)
    
    # def __repr__(self):
    #     """String representation of the wrapper."""
    #     return f"KaolinViewer(websocket_address={getattr(self.component, 'websocket_address', None)})"




