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

import copy
from collections import defaultdict
import os.path
from pathlib import Path
from dataclasses import dataclass
from dash import Dash, html, dcc, Input, Output, callback, State, clientside_callback, no_update
from dash.development.base_component import Component, ComponentRegistry
import dash_bootstrap_components as dbc
from flask import send_from_directory, abort
import logging
import threading
from tornado.wsgi import WSGIContainer
from tornado.web import Application, FallbackHandler, StaticFileHandler
from tornado.ioloop import IOLoop
from typing import Protocol, Dict, Any, Union, runtime_checkable, Callable, Iterable, Literal, get_args, Mapping, Optional

from kaolin.visualize.web.sockets import WebSocketHandlerManager, TuidCallable
import kaolin.visualize.dash.auto_ui as auto_ui
from kaolin.visualize.dash.assets_helper import get_kaolin_assets_source_path, get_kaolin_assets_serve_path
from kaolin.visualize.dash.layout import AppLayoutHelper
from kaolin.visualize.web.naming import UniqueIdGenerator
import kaolin.version

logger = logging.getLogger(__name__)

__all__ = [
    'SessionRegistry',
    'WebappBuilder',
]



class SessionRegistry:
    """Process-wide registry mapping ``tab_uuid`` to a :class:`_SessionEntry`.

    Bridges Dash (``@callback`` triggered by a ``dcc.Store`` change with
    ``State('kaolin-tab-uuid', 'data')``) and the WebSocket transport
    (``WebSocketHandlerManager.tab_uuid``). Both sides converge on the same
    ``tab_uuid`` key so a setting written by a control reaches every WS
    handler belonging to that browser tab.

    NOTE: prototype-quality, app-level. Future improvements:
      - Promote to a library helper (e.g. ``kaolin.visualize.dash.session_registry``).
      - GC stale entries when the last WS handler for a tab disconnects (use
        ``GlobalWebSocketConnectionManager.get_handlers_by_tab(tab_uuid)`` to
        decide). Acceptable to leak for now -- short-lived dev runs.
      - Make the per-entry payload pluggable (factory) instead of hard-coding
        :class:`ServerSideUserSettings`.
    """
    _lock = threading.Lock()
    _entries = {}

    @classmethod
    def _get_or_create(cls, tab_uuid: Optional[str], initial_session_data: Dict) -> Dict:
        if tab_uuid not in cls._entries:
            cls._entries[tab_uuid] = {}
            for k, v in initial_session_data.items():
                cls._entries[tab_uuid][k] = copy.deepcopy(v)
        return cls._entries.get(tab_uuid)

    @classmethod
    def get_or_create(cls, tab_uuid: Optional[str], initial_session_data: Dict) -> Dict:
        with cls._lock:
            return cls._get_or_create(tab_uuid, initial_session_data)

    @classmethod
    def get_or_create_and_update(cls, tab_uuid: Optional[str], initial_session_data: Dict,
                         update_func=None):
        with cls._lock:
            session = cls._get_or_create(tab_uuid, initial_session_data)
            if update_func is not None:
                update_func(session)
            return session

    @classmethod
    def get(cls, tab_uuid: Optional[str]) -> Optional[Dict]:
        with cls._lock:
            return cls._entries.get(tab_uuid)

    @classmethod
    def remove(cls, tab_uuid: Optional[str]) -> None:
        """Drop an entry. Idempotent. Intended for future GC; unused for now."""
        with cls._lock:
            cls._entries.pop(tab_uuid, None)

@dataclass
class ImportmapItem:
    url: str
    import_as_global: str | None = None


class WebappBuilder:
    CUSTOM_FILES_PATH = "kaolin_custom"
    RESERVED_PATHS = ["_kaolin_assets", "websocket"]
    TAB_UUID_STORE_ID = "kaolin-tab-uuid"
    """Id of the auto-injected ``dcc.Store`` that holds the per-browser-tab uuid.

    The same uuid is also sent on every WebSocket as the ``tab`` query argument
    (see :class:`kaolin.visualize.web.sockets.WebSocketHandlerManager.tab_uuid`)
    and is reachable from app TS code via ``kaolin.util.getTabUuid()``.
    Use it to correlate Dash ``@callback`` events with per-tab WebSocket handler
    instances (e.g. read it as ``State(WebappBuilder.TAB_UUID_STORE_ID, 'data')``)."""

    class _UserSettingInfo:
        def __init__(self, instance, name, field_specs, store):
            self.instance = instance
            self.name = name
            self.field_specs = field_specs
            self.store = store

    def __init__(self, debug,
                 default_theme=dbc.themes.LUX,
                 set_favicon=True):
        """ Initializes web app builder with basic settings. Used setter functions to
        further configure it before building the app.

        Note: 🌱 this class is just a shortcut. Use custom Dash / tornado app setup
        if it does not fit your use cases.

        Args:
            debug: if True, will enable hot reloading of code and some other features.
            default_theme: theme to use (default: dbc.themes.LUX)
            set_favicon: if True, will set default kaolin favicon (default: True)
        """
        self.debug = debug
        self.default_theme = default_theme
        self.set_favicon = set_favicon
        self.layout_helper: None | AppLayoutHelper = None

        self.extra_stylesheets = []
        self.extra_scripts = []
        self.importmap = {}
        self.served_files = {}
        self.served_dirs = {}
        self.user_setting_infos = {}  # name: _UserSettingInfo
        self.extra_elements = []
        self.extra_raw_html = []

        # Only built once
        self.app = None
        self.server = None
        self.default_user_session = None

    def stylesheets(self):
        """

        Returns:
            (list) of strings

        """
        layout_stylesheets = []
        if self.layout_helper is not None:
            layout_stylesheets = self.layout_helper.stylesheets()
        return [self.default_theme] + self.extra_stylesheets + layout_stylesheets

    def external_scripts(self):
        return self.extra_scripts

    def add_extra_stylesheets(self, extra_stylesheets: Iterable):
        self.extra_stylesheets.extend(extra_stylesheets)

    def add_raw_body_html(self, custom_html):
        self.extra_raw_html.append(custom_html)

    def add_importmap_item(self, package_name: str, url: str, import_as_global: str | None = None):
        self.importmap[package_name] = ImportmapItem(url, import_as_global)

    def add_extra_scripts(self, extra_scripts: Iterable):
        self.extra_scripts.extend(extra_scripts)

    def set_layout_helper(self, layout_helper):
        self.layout_helper = layout_helper

    @staticmethod
    def __add_unique_url_mapping_or_die(mappings, url_path, local_path):
        # TODO: deal with slash already provided by user
        url_path = WebappBuilder.CUSTOM_FILES_PATH + '/' + url_path

        real_dir_path = os.path.realpath(os.path.abspath(local_path))
        if url_path in mappings:
            if mappings[url_path] != real_dir_path:
                raise ValueError(f'Attempting to map url path {url_path} to {real_dir_path}, '
                                 f'but it is already mapped to {mappings[url_path]}')
        mappings[url_path] = real_dir_path
        return url_path

    def add_static_dir(self, dir_path, url_path=None):
        """ Adds all files in a directory as static files to be served. New files will be retrieved automatically
        even if the server is already running.

        Args:
            dir_path: local directory where files should be served from, e.g. "/home/me/my_files". All files will be served.
            url_path: path to the files in dir, e.g. "my_files" (default: "kaolin_custom/(basename dir_path)")

        Returns:
            (str) url_path
        """
        assert os.path.isdir(dir_path), f'Not a directory: {dir_path}'

        if url_path is None:
            url_path =  Path(dir_path).name

        assert url_path in WebappBuilder.RESERVED_PATHS, f'Cannot assign reserved path {url_path} to custom static files dir'

        return WebappBuilder.__add_unique_url_mapping_or_die(self.served_dirs, url_path, dir_path)

    def add_static_file(self, file_path, url_path=None):
        """

        Args:
            file_path:
            url_path:

        Returns:

        """
        assert os.path.isfile(file_path), f'Not a file: {file_path}'

        if url_path is None:
            url_path = os.path.basename(file_path)

        return WebappBuilder.__add_unique_url_mapping_or_die(self.served_files, url_path, file_path)


    @staticmethod
    def _insert_importmap(index_html_str, importmap: Mapping[str, ImportmapItem], add_globals: bool = True):
        """Insert an importmap script tag before </head> in the HTML string.

        Args:
            index_html_str: HTML string containing a </head> tag
            importmap: Mapping from package names to ImportmapItem

        Returns:
            Modified HTML string with importmap inserted
        """
        import json

        if not importmap:
            return index_html_str

        importmap_json = json.dumps({"imports": {k: v.url for k, v in importmap.items()}}, indent=4)
        importmap_script = f'\n<script type="importmap">\n{importmap_json}\n</script>\n'

        # Generates something like:
        # <script type="module">
        #   import * as pc from 'playcanvas';
        #   window.pc = pc;
        # </script>
        globals_script = ""
        if add_globals:
            for package_name, item in importmap.items():
                global_package_name = item.import_as_global
                if global_package_name is not None:
                    globals_script += f"import * as {global_package_name} from '{package_name}';\n"
                    globals_script += f"console.log('🍓->>>>>>> Setting window.{global_package_name}');"  # HACK
                    globals_script += f"window.{global_package_name} = {global_package_name};\n"

            if len(globals_script) > 0:
                globals_script = f'<script type="module">\n{globals_script}</script>'

                # START_OF_HACK
                additional_script = ('<script>'
                                     'console.log("🍓->>>>>>> Adding ondocument loaded event");'
                                     'document.addEventListener("DOMContentLoaded", function() { console.log("🍓->>>>>>> Document loaded"); });'
                                     '</script>')
                globals_script += additional_script
                # END_OF_HACK

        return index_html_str.replace('</head>', f'{importmap_script}{globals_script}</head>')

    @staticmethod
    def _insert_at_body_start(index_html_str, custom_html):
        return index_html_str.replace('<body>', f'<body>{custom_html}')

    @staticmethod
    def log_debug_info(app):
        print(f'Static folder: {app.server.static_folder}')
        print(f'Static URL path: {app.server.static_url_path}')
        print(f'Component registry: {len(ComponentRegistry.registry)} ' +
              f'{[x for x in ComponentRegistry.registry]}')
        print(f'Config: {app.config}')
        print(f'Registered paths: {app.registered_paths}')
        # TODO: print the rest of the info, like served files and dirs, external scripts, etc.

    def _setup_user_setting_sessions(self, app: Dash) -> None:
        """Append a hidden ``dcc.Store(id=TAB_UUID_STORE_ID)`` to ``app.layout`` and
        register a clientside callback that fills it from the TS ``tabSession`` module.

        Triggering off the store's own immutable ``id`` causes the callback to fire
        exactly once at layout mount. If the kaolin TS bundle hasn't finished loading
        yet, we briefly retry via ``setTimeout`` + ``set_props`` rather than leave the
        store at ``null``. Memory storage is fine because the source of truth lives in
        ``sessionStorage`` on the JS side.
        """
        if app.layout is None:
            return  # nothing to attach to; app likely won't render anyway

        store = dcc.Store(id=WebappBuilder.TAB_UUID_STORE_ID,
                          data=None, storage_type='memory')
        dummy_id= UniqueIdGenerator.get_unique_id('dummy-output')
        dummy_div = html.Div(id=dummy_id, style={'display': 'none'})

        if isinstance(app.layout, list):
            app.layout = list(app.layout) + [store, dummy_div]
        else:
            app.layout = html.Div([app.layout, store, dummy_div])

        app.clientside_callback(
            f"""
            function(_id) {{
                function readUuid() {{
                    if (kaolin && kaolin.util && kaolin.util) {{
                        return kaolin.util.getTabUuid();
                    }}
                    return null;
                }}
                const uuid = readUuid();
                if (uuid !== null) {{
                    console.log('>>> (0) Registered tab session' + uuid);
                    return uuid;
                }}
                // kaolin.js still loading -- retry briefly via set_props.
                let retries = 20;
                const tick = function() {{
                    const u = readUuid();
                    if (u !== null) {{
                        window.dash_clientside.set_props(
                            '{WebappBuilder.TAB_UUID_STORE_ID}', {{ data: u }});
                        console.log('>>> Registered tab session' + u);
                        return;
                    }}
                    if (--retries > 0) {{
                        setTimeout(tick, 50);
                    }} else {{
                        console.warn('{WebappBuilder.TAB_UUID_STORE_ID}: '
                            + 'kaolin.util.tabSession never loaded');
                    }}
                }};
                setTimeout(tick, 50);
                return dash_clientside.no_update;
            }}
            """,
            Output(WebappBuilder.TAB_UUID_STORE_ID, 'data'),
            Input(WebappBuilder.TAB_UUID_STORE_ID, 'id'),
        )

        self.default_user_session = {}
        for name, info in self.user_setting_infos.items():
            session_instance = copy.deepcopy(info.instance)
            # We will also use store to save session-level settings for a tab
            auto_ui.apply_store_to_dataclass(info.store, session_instance, info.field_specs)
            self.default_user_session[name] = session_instance

        @app.callback(
            Output(component_id=dummy_id, component_property='children', allow_duplicate=True),
            Input(WebappBuilder.TAB_UUID_STORE_ID, "data"),
            prevent_initial_call=True  # Prevent running when it initiates as False
        )
        def register_session(tab_uuid):
            if tab_uuid is not None:
                logger.info(f"Registering client session with id {tab_uuid}")
                SessionRegistry.get_or_create(tab_uuid, self.default_user_session)
            return "Still waiting..."

    def _bind_controls_to_store_serverside(self, name, field_specs: list[auto_ui.FieldSpec], store_id: str) -> None:
        for spec in field_specs:
            @callback(
                Output(spec.control_id, spec.value_prop),
                Input(store_id, 'data'),
                State(spec.control_id, spec.value_prop),
            )
            def _sync_ui_from_store(stored_data, current_ui_value, _name=spec.name):
                """Restore control from session-backed store on reload; avoid store→UI→store loops."""
                if stored_data is None or _name not in stored_data:
                    return no_update
                stored_val = stored_data[_name]
                if stored_val == current_ui_value:
                    return no_update
                return stored_val

            @callback(
                Output(store_id, 'data', allow_duplicate=True),
                Input(spec.control_id, spec.value_prop),
                Input(WebappBuilder.TAB_UUID_STORE_ID, "data"),
                State(store_id, 'data'),
                prevent_initial_call=True,
            )
            def _update(value, tab_uuid, current, _name=spec.name):
                print(f'Updating session {tab_uuid} settings "{name}"[{_name}] to {value}')
                if current is None:
                    current = {}
                current[_name] = value

                def _session_update(session):
                    session_settings = session.get(name)  # get setting instance by name
                    if session_settings is not None:
                        auto_ui.apply_store_value_to_dataclass(current, session_settings, spec)

                SessionRegistry.get_or_create_and_update(tab_uuid, self.default_user_session, _session_update)
                return current

    def add_user_settings(self, setting_instance, name, storage_type='session'):
        if name in self.user_setting_infos:
            raise ValueError(f'Cannot add multiple settings with identical name {name}; use unique name')

        controls, field_specs, store = auto_ui.controls_from_dataclass(
            setting_instance, name, make_store=True, storage_type=storage_type)

        info = WebappBuilder._UserSettingInfo(setting_instance, name, field_specs, store)
        self.user_setting_infos[name] = info

        self._bind_controls_to_store_serverside(name, field_specs, store.id)
        return controls + [store]

    def get_user_settings_func(self, name):
        if name not in self.user_setting_infos:
            raise ValueError(f'User settings with name {name} have not been registered; call add_user_settings to register')

        def _get_settings(tab_uuid):
            session = SessionRegistry.get_or_create(tab_uuid, self.default_user_session)
            return session.get(name)  # get setting instance by name
        return TuidCallable(_get_settings)

    def build(self, ws_handlers, **dash_kwargs):
        # <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

        assert self.app is None, f'Can only call build once'

        # TODO: should smart-complete dash_kwargs, adding to external_stylesheets and external_scripts if present
        app = Dash(external_stylesheets=self.stylesheets(),
                   external_scripts=self.external_scripts(),
                   **dash_kwargs)
        # external_scripts=["https://cdn.jsdelivr.net/npm/playcanvas@2.14.0/+esm"],
        # external_scripts=["https://cdn.jsdelivr.net/npm/playcanvas@2.14.0/build/playcanvas.min.js"],
        if self.debug:
            app.enable_dev_tools(
                debug=True,
                dev_tools_hot_reload=True
            )

        # TODO: debug did not work

        # self.add_importmap_item("playcanvas",
        #                         "https://cdn.jsdelivr.net/npm/playcanvas@2.14.0/+esm",
        #                         import_as_global="pc")
        if len(self.importmap) > 0:
            app.index_string = WebappBuilder._insert_importmap(app.index_string, self.importmap)

        for custom_html in self.extra_raw_html:
            app.index_string = WebappBuilder._insert_at_body_start(app.index_string, custom_html)

        if self.layout_helper is not None:
            app.layout = self.layout_helper.layout()
            self.layout_helper.configure_app(app)

        # TODO: condition on something --> e.g. maybe we add these settings to builder
        if len(self.user_setting_infos) > 0:
            self._setup_user_setting_sessions(app)

        if len(self.served_files) > 0:
            @app.server.route(f'/{WebappBuilder.CUSTOM_FILES_PATH}/<file_url>')
            def serve_dynamic_file(file_url):
                # Look up the actual absolute path based on the URL parameter
                actual_path = self.served_files.get(file_url)

                if actual_path is None or not os.path.exists(actual_path):
                    abort(404)  # Return 404 if the file isn't in the registry

                return send_from_directory(
                    os.path.dirname(actual_path),
                    os.path.basename(actual_path)
                )

        WebappBuilder.log_debug_info(app)

        # logger.info(f'-----> Now starting Dash App on port {args.port}')
        # app.logger.setLevel(args.log_level)
        # app.run(debug=False, port=args.port)  # if you want to debug, better debug=False, or else no stack trace

        # Serve single dynamic file via Flask route
        # import os
        #
        # splat_file_path = '/mnt/masha_prev_2/Experiments/GSplats/inria/output/knit_meow/extra/baked.knit_meow.ply'  # TODO: replace with actual absolute path
        #
        # @app.server.route('/_kaolin_dynamic/splat.ply')
        # def serve_splat():
        #     return send_from_directory(
        #         os.path.dirname(splat_file_path),
        #         os.path.basename(splat_file_path)
        #     )



        # Serve Kaolin library assets (CSS, JS, images, etc.)
        handlers = [(r'/_kaolin_assets/(.*)', StaticFileHandler, {'path': get_kaolin_assets_source_path()})]

        # WebSocket handlers
        if len(ws_handlers) > 0:
            handlers.append((r'/websocket/', WebSocketHandlerManager, dict(handler_specs=ws_handlers)))

        # Custom static file directories
        for dir_url, dir_path in self.served_dirs.items():
            handlers.append((f'/{WebappBuilder.CUSTOM_FILES_PATH}/{dir_url}/(.*)', StaticFileHandler, {'path': dir_path}))

        # Default dash app as the last handler
        container = WSGIContainer(app.server)
        handlers.append((r'.*', FallbackHandler, dict(fallback=container)))

        server = Application(handlers,
                            debug=self.debug)  # debug=True enables Tornado's autoreload

        # Explicitly watch custom dynamic files
        if self.debug:
            import tornado.autoreload
            for file_url, file in self.served_files.items():
                tornado.autoreload.watch(file)

        if self.set_favicon:
            app.clientside_callback(
                """
                function() {
                   function updateFavicon(newHref, type = 'image/x-icon') {
                        let link = document.querySelector('link[rel~="icon"]');
                        if (link) {
                            link.href = newHref;
                        }
                    }
                    document.addEventListener('DOMContentLoaded', function() {
                    updateFavicon('/_kaolin_assets/favicon.ico'); 
                    });
                    updateFavicon('/_kaolin_assets/favicon.ico'); 
                    return dash_clientside.no_update;
                }
                """,
                Output('kaolin-app', 'id'),
                Input('kaolin-app', 'id'),
            )
        self.app = app
        self.server = server
        return app, server

    def start(self, port):
        assert self.server is not None, f'Must call build() first'
        logger.info(f'-----> Now starting Dash App and Websocket Handler on port {port}')
        self.server.listen(port, address='0.0.0.0')
        IOLoop.instance().start()