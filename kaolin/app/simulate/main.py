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

import logging
import math
import os

import simple_parsing
import torch
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, clientside_callback

import kaolin.render.camera
from kaolin.visualize.dash import WebappBuilder, ViewerBuilder, StandardLayoutHelper, BehaviorLibrary

FILE_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)


def _default_camera():
    return kaolin.render.camera.Camera.from_args(
        eye=torch.tensor([0.0, -3.0, 1.5]),
        at=torch.tensor([0.0, 0.0, 0.0]),
        up=torch.tensor([0.0, 0.0, 1.0]),
        fov=math.pi * 50 / 180,
        height=512, width=512,
        device='cpu',
    )


def build_app(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ------------------------------------------------------------------
    # Register custom JS behaviors before building the viewer so the
    # BehaviorLibrary can resolve 'simulate_controller' by name.
    # ------------------------------------------------------------------
    BehaviorLibrary.register_user_directory(os.path.join(FILE_DIR, 'assets'))

    # ------------------------------------------------------------------
    # Viewer
    # ------------------------------------------------------------------
    app_builder = WebappBuilder(debug=False)
    viewer_builder = ViewerBuilder(camera=_default_camera(), viewport_resize_mode='fixed')
    viewer_builder.add_websocket_connection('ws://WINDOW_LOCATION/websocket/', 'main-ws')

    render_layer_id = viewer_builder.add_layer()
    cam_controller_id = viewer_builder.add_camera_controller(options={'up': [0.0, 0.0, 1.0]})

    draw_render_id, send_cam_id, request_render_id = viewer_builder.add_remote_rendering(
        active_layer_id=render_layer_id,
        connection_id='main-ws',
        cam_update_period=50,
        render_update_period=150,
    )

    sim_controller_id = viewer_builder.add_behavior(
        'simulate_controller',
        options={'listContainerId': 'object-list-container', 'connectionId': 'main-ws'},
        is_active=True,
    )

    viewer_builder.add_mode('view', active_behavior_ids=[
        draw_render_id, send_cam_id, request_render_id, cam_controller_id, sim_controller_id,
    ])
    viewer_builder.set_active_mode('view')
    viewer = viewer_builder.build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    layout_helper = StandardLayoutHelper(title='Kaolin Simulate', max_width='lg')
    layout_helper.add_sidebar(width=22, min_width_px=320)

    # Section: Objects
    layout_helper.add_sidebar_section('Objects', collapsible=False, bootstrap_icon='bi-boxes')
    layout_helper.add_sidebar_components([
        dbc.Button('Load Object', id='load-object-btn', color='primary', size='sm', className='mb-2 w-100'),
        html.Div(id='object-list-container'),
    ], section_name='Objects')

    # Section: Simulation
    layout_helper.add_sidebar_section('Simulation', collapsible=True,
                                      bootstrap_icon='bi-play-circle', collapsed_on_load=False)
    layout_helper.add_sidebar_components([
        dbc.Row([
            dbc.Col(dbc.Button('Simulate', id='start-sim-btn', color='success', size='sm'), width=4),
            dbc.Col(dbc.Button('Stop', id='stop-sim-btn', color='warning', size='sm'), width=4),
            dbc.Col(dbc.Button('Reset', id='reset-sim-btn', color='secondary', size='sm'), width=4),
        ], className='mb-2'),
        html.Div(id='sim-status-display', children='Idle', className='mb-2 small text-muted'),
        html.Hr(),
        html.Small('Timestep (s)'),
        dcc.Slider(id='timestep-slider', min=0.005, max=0.1, step=0.005, value=0.03,
                   marks={0.01: '0.01', 0.05: '0.05', 0.1: '0.1'}, className='mb-2'),
        html.Small('Newton Steps'),
        dcc.Slider(id='newton-steps-slider', min=1, max=10, step=1, value=3,
                   marks={1: '1', 5: '5', 10: '10'}, className='mb-2'),
        dbc.Checklist(
            id='collisions-checklist',
            options=[{'label': 'Enable Collisions', 'value': 'enable'}],
            value=[],
            className='mb-2',
        ),
    ], section_name='Simulation')

    # Modal: Load Object
    load_modal_content = html.Div([
        html.H5('Load Gaussian Object'),
        # Path input — shown by default, hidden while the Gaussian-selector is active.
        html.Div(id='modal-path-section', children=[
            dbc.InputGroup([
                dbc.InputGroupText('Path'),
                dbc.Input(id='load-path-input', placeholder='/path/to/file.ply or .usd',
                          type='text', autocomplete='off'),
            ], className='mb-1'),
            html.Div(id='path-completions-dropdown',
                     style={'display': 'none', 'maxHeight': '180px', 'overflowY': 'auto',
                            'border': '1px solid #dee2e6', 'borderRadius': '4px',
                            'marginBottom': '8px', 'background': '#fff'}),
            dbc.Button('Load', id='do-load-btn', color='primary', className='mt-1'),
        ]),
        # Gaussian-prim selector — populated by JS when multiple prims are found in a USD.
        html.Div(id='modal-gaussian-selector', style={'display': 'none'}),
        html.Div(id='load-error-msg', className='text-danger mt-1'),
    ])
    load_modal = layout_helper.add_modal(load_modal_content, title='Load Object')
    # Register the sidebar button as an open-trigger so configure_app wires open+close together.
    load_modal['trigger_ids'].append('load-object-btn')

    # Hidden element so custom.js can read the modal close-button ID and close the modal
    # programmatically after the user confirms the Gaussian-prim selection.
    layout_helper.add_main_content([
        viewer,
        html.Div(id='load-modal-meta',
                 **{'data-close-id': load_modal['close_id']},
                 style={'display': 'none'}),
    ])
    app_builder.set_layout_helper(layout_helper)

    noop_out = layout_helper.noop_callback_output()

    # ------------------------------------------------------------------
    # Clientside callbacks
    # ------------------------------------------------------------------

    # Reset the modal to its path-input state whenever it is opened (re-open after cancel).
    clientside_callback(
        '''function(n) {
            if (!n) return window.dash_clientside.no_update;
            var sel = document.getElementById('modal-gaussian-selector');
            var path = document.getElementById('modal-path-section');
            if (sel)  { sel.style.display = 'none'; sel.innerHTML = ''; }
            if (path) path.style.display = 'block';
            return window.dash_clientside.no_update;
        }''',
        layout_helper.noop_callback_output(),
        Input('load-object-btn', 'n_clicks'),
        prevent_initial_call=True,
    )

    # Send load_file message when "Load" is clicked in the modal
    clientside_callback(
        '''function(n, path) {
            if (!n || !path) return window.dash_clientside.no_update;
            var conn = kaolin.core.WebSocketConnectionsManager.getOpenConnection('main-ws');
            if (!conn) return 'Not connected — wait for the viewer to load and try again.';
            kaolin.io.encodeMessage('load_file', {path: path}).then(function(enc) { conn.send(enc); });
            return '';
        }''',
        Output('load-error-msg', 'children'),
        Input('do-load-btn', 'n_clicks'),
        State('load-path-input', 'value'),
        prevent_initial_call=True,
    )

    # Simulate / Stop / Reset buttons
    clientside_callback(
        '''function(ns, ns2, ns3, ts, nsteps, coll) {
            var ctx = window.dash_clientside.callback_context;
            if (!ctx || !ctx.triggered || !ctx.triggered[0]) return window.dash_clientside.no_update;
            var tid = ctx.triggered[0].prop_id;
            var conn = kaolin.core.WebSocketConnectionsManager.getOpenConnection('main-ws');
            if (!conn) return window.dash_clientside.no_update;
            var send = function(tag, content) {
                kaolin.io.encodeMessage(tag, content).then(function(enc) { conn.send(enc); });
            };
            if (tid.indexOf('start-sim-btn') !== -1) {
                send('start_sim', {timestep: ts, newton_steps: nsteps,
                                   enable_collisions: coll && coll.indexOf('enable') !== -1});
            } else if (tid.indexOf('stop-sim-btn') !== -1) {
                send('stop_sim', {});
            } else if (tid.indexOf('reset-sim-btn') !== -1) {
                send('reset_sim', {});
            }
            return window.dash_clientside.no_update;
        }''',
        noop_out,
        Input('start-sim-btn', 'n_clicks'),
        Input('stop-sim-btn', 'n_clicks'),
        Input('reset-sim-btn', 'n_clicks'),
        State('timestep-slider', 'value'),
        State('newton-steps-slider', 'value'),
        State('collisions-checklist', 'value'),
        prevent_initial_call=True,
    )

    # ------------------------------------------------------------------
    # Build and return
    # ------------------------------------------------------------------
    from kaolin.app.simulate.handler import AppState, SimulateHandler
    app_state = AppState(device=device)
    ws_handlers = [(SimulateHandler, (app_state,))]
    app, server = app_builder.build(ws_handlers)
    return app, server, args.port


def main():
    logging.basicConfig(level=logging.INFO)
    parser = simple_parsing.ArgumentParser('Kaolin Simulate App')
    parser.add_argument('--port', default=8002, type=int)
    args = parser.parse_args()

    app, server, port = build_app(args)
    from tornado.ioloop import IOLoop
    server.listen(port, address='0.0.0.0')
    print(f'Simulate app running at http://localhost:{port}')
    IOLoop.instance().start()


if __name__ == '__main__':
    main()
