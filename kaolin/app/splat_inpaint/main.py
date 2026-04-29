import logging
import math
import os

import simple_parsing
from dash import html, dcc, Input, clientside_callback
import dash_bootstrap_components as dbc

import kaolin
import kaolin.io.gaussians
import kaolin.render.camera.gsplats_nerfstudio
import kaolin.visualize.dash
import kaolin.visualize.web.sockets
from kaolin.utils.log import default_log_setup, add_log_level_flag
from kaolin.visualize.dash import StandardLayoutHelper, WebappBuilder, ViewerBuilder
from kaolin.visualize.web.sockets import AnyRendererMessageHandler
from kaolin.visualize.dash.builtins import BehaviorLibrary

import gsplat
import torch

from .handlers import InpaintImageMessageHandler, TrainGaussiansMessageHandler


logger = logging.getLogger(__name__)

FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


class ShellRenderer:
    """Thin closure around `gsplat.rasterization` so we can hand a plain
    `kal_cam -> RGBA tensor` callable to the websocket handlers."""

    def __init__(self, gsmodel):
        self.gsmodel = gsmodel

    def render(self, kal_cam):
        gsplat_cam_params = kaolin.render.camera.gsplats_nerfstudio.kaolin_camera_to_gsplat_nerfstudio(kal_cam)
        render_colors, render_alphas, info = gsplat.rendering.rasterization(
            self.gsmodel.positions,
            self.gsmodel.orientations,
            self.gsmodel.scales,
            self.gsmodel.opacities,
            self.gsmodel.sh_coeff,
            sh_degree=self.gsmodel.sh_degree,
            **gsplat_cam_params)
        render_colors = (render_colors.clip(0, 1) * 255).to(torch.uint8)
        render_colors = torch.cat([render_colors, torch.ones_like(render_colors[..., :1]) * 255], dim=-1)
        return render_colors.squeeze(0)


def default_camera(device):
    return kaolin.render.camera.Camera.from_args(
        eye=torch.tensor([0.3, -0.5, 0.6]), at=torch.tensor([0.0, 0.0, 0.3]), up=torch.tensor([0., 0, 1.]),
        fov=math.pi * 50 / 180, height=512, width=512).to(device)


if __name__ == '__main__':
    parser = simple_parsing.ArgumentParser('Toy Gaussian Splat Inpainter App')
    parser.add_argument('--port', default=8000, type=int)
    parser.add_argument('--input_scene', type=str, #required=True,
                        default='/mnt/masha_prev_2/Experiments/GSplats/inria/output/knit_meow/extra/baked.knit_meow.ply',
                        help='Input PLY or USD file (Gaussian splat).')
    add_log_level_flag(parser)
    args = parser.parse_args()
    device = "cuda"

    # Make sure logging is configured
    default_log_setup(level=args.log_level)

    # ------------------------------------------------------------------------------------------------------------------
    # Project:
    # load the scene (a Gaussian splat cloud) and create an initial camera. Right now state is global across all
    # clients; a future refactor could move this into a `Project` class like `kaolin/app/act_splat`.
    # ------------------------------------------------------------------------------------------------------------------
    gsmodel = kaolin.io.gaussians.import_gaussiancloud(args.input_scene).to(device)
    print(gsmodel.to_string(print_stats=True))

    camera = default_camera(device)

    # ------------------------------------------------------------------------------------------------------------------
    # WebappBuilder: helps us create the whole app - layout, websockets, UI, etc.
    # ------------------------------------------------------------------------------------------------------------------
    app_builder = WebappBuilder(debug=False)

    # WebSocket handlers ----------
    #   - Custom handlers subclass:
    #      - kaolin.visualize.web.sockets.SyncMessageHandlerProtocol or
    #      - kaolin.visualize.web.sockets.AsyncMessageHandlerProtocol
    #   - Lifecycle: each handler persists for an individual client connection
    #   - Each handler whitelists the message tags it serves via `accepted_message_tags()`
    ws_handlers = []

    # Default server-side rendering handler (handles 'set_camera' + 'render' messages).
    shell_renderer = ShellRenderer(gsmodel)
    ws_handlers.append((AnyRendererMessageHandler, (shell_renderer.render,)))

    # Diffusion-based 2D inpainting: receives image+mask+prompt, returns inpainted RGBA.
    ws_handlers.append((InpaintImageMessageHandler, ()))

    # Per-Gaussian color optimization: bakes an inpainted 2D edit back into 3D
    # by optimizing the splat colors inside the painted mask.
    ws_handlers.append((TrainGaussiansMessageHandler, (gsmodel,)))

    # ------------------------------------------------------------------------------------------------------------------
    # ViewerBuilder: helps us build the interactive viewer (on the webpage)
    # ------------------------------------------------------------------------------------------------------------------
    viewer_builder = ViewerBuilder(camera=camera, viewport_resize_mode='fixed')

    # WebSocket Connections ----------
    # Add a named socket connection (this will be used to connect to all our handlers).
    #   - WINDOW_LOCATION means we'll serve these requests from the same webserver as the UI
    viewer_builder.add_websocket_connection("ws://WINDOW_LOCATION/websocket/", 'main-ws')

    # Layers -------------------------
    # Stacked HTML elements (canvas/div/svg) we attach behaviors to. Identifiers
    # are passed through to the DOM so they're easy to locate in devtools.
    render_layer_id = viewer_builder.add_layer(identifier='render')   # server-side rendering goes here
    mask_layer_id = viewer_builder.add_layer(identifier='mask')       # user paints the inpaint mask here
    inpaint_layer_id = viewer_builder.add_layer(identifier='inpaint') # inpainted RGBA returned by the server

    # Behaviors ------------------------
    # Behaviors are defined in JavaScript and add interactive or message-receiving behavior to a layer.
    # See assets/custom.js for this app's custom helpers; `BehaviorLibrary` lets us register a user dir
    # of custom behaviors so warnings stay quiet at config time.
    BehaviorLibrary.register_user_directory(os.path.join(FILE_DIR, 'assets'))

    # Camera control + remote rendering (built-in shortcuts). This scene is
    # +Z up, so tell the orbit controller which way is up.
    cam_controller_id = viewer_builder.add_camera_controller(options={'up': [0.0, 1.0, 0.0]})
    draw_render_id, send_cam_id, request_render_id = viewer_builder.add_remote_rendering(
        active_layer_id=render_layer_id, connection_id='main-ws')

    # Mask painting behavior on the mask layer.
    mask_behavior_id = viewer_builder.add_behavior(
        'drawing', active_layer_id=mask_layer_id,
        options={"color": "#ffffff", "thickness": 25},
        is_active=False)

    # Behavior that draws server-pushed images (the inpaint result) into the inpaint layer.
    draw_inpaint_id = viewer_builder.add_behavior(
        'draw_remote_image', active_layer_id=inpaint_layer_id,
        options={'messageTag': 'inpaint'})

    # Drawing behavior on the inpaint layer for touch-up edits (erase mode by default).
    edit_behavior_id = viewer_builder.add_behavior(
        'drawing', active_layer_id=inpaint_layer_id,
        options={"mode": "erase"},
        is_active=False)

    # Modes ----------------------------
    # Group behaviors into modes; this auto-creates icons + 1..9 keyboard shortcuts in the viewer.
    # `requestMode(name)` from clientside JS switches modes at runtime.
    viewer_builder.add_mode(
        'view',
        [draw_render_id, send_cam_id, request_render_id, cam_controller_id, draw_inpaint_id],
        ui_icon='bi-binoculars', description='mode for interactive camera control')
    viewer_builder.add_mode(
        'mask',
        [draw_render_id, draw_inpaint_id, mask_behavior_id],
        ui_icon='bi-mask', description='paint a mask to inpaint')
    viewer_builder.add_mode(
        'edit',
        [draw_render_id, draw_inpaint_id, mask_behavior_id, edit_behavior_id],
        ui_icon='bi-pencil-square', description='touch up the inpainted result')
    viewer_builder.set_active_mode('view')

    # Building ----------------------
    viewer = viewer_builder.build()

    # ------------------------------------------------------------------------------------------------------------------
    # LayoutHelper: combines viewer + sidebar controls into a mobile-friendly webpage
    # ------------------------------------------------------------------------------------------------------------------
    title = html.Span([html.Span("Toy Gaussian Splat Inpainter"), html.I(className="p-2 bi bi-boxes me-2")])
    layout_helper = StandardLayoutHelper(title)
    layout_helper.add_sidebar(disappearing=False)

    # Painting ---------------------------
    # Brush mode toggle: switches the active drawing behavior between 'draw' and 'erase'.
    layout_helper.add_sidebar_section('Painting')
    brush_mode_toggle = layout_helper.add_sidebar_component(
        dbc.Switch(id='brush-mode-toggle', label='Erase mode', value=False),
        section_name='Painting')
    clientside_callback(
        f"""
        function(is_erase) {{
            set_brush_mode(is_erase ? 'erase' : 'draw', "{mask_behavior_id}", "{edit_behavior_id}");
            return dash_clientside.no_update;
        }}
        """,
        layout_helper.noop_callback_output(),
        Input('brush-mode-toggle', 'value'),
        prevent_initial_call=True)

    # Inpainting -------------------------
    # Prompt + Inpaint/Clear buttons. The inpaint button packages the rendered RGB
    # and the painted mask, ships them over the websocket as a tagged binary message,
    # and the server-side InpaintImageMessageHandler returns the inpainted RGBA
    # (which the `draw_remote_image` behavior paints into the inpaint layer).
    layout_helper.add_sidebar_section('Inpainting')
    promptarea = layout_helper.add_sidebar_component(
        dcc.Textarea(
            id='prompt-text',
            value=('two realistic cat eyes, hazel with yellow and orange highlights, shining in bright sunlight, '
                   'highly detailed, macro photography, 8k'),
            style={'width': '100%', 'height': 100}),
        section_name='Inpainting')
    inpaint_button = layout_helper.add_sidebar_component(
        html.Button('Inpaint', id='inpaint-btn', className='btn btn-secondary w-100'),
        section_name='Inpainting')
    clear_button = layout_helper.add_sidebar_component(
        html.Button('Clear', id='clear-btn', className='btn btn-secondary w-100 mt-2'),
        section_name='Inpainting')
    clientside_callback(
        f"""
        function(n_clicks) {{
            send_inpaint_request("{viewer.resolve_id(render_layer_id)}", "{viewer.resolve_id(mask_layer_id)}", "prompt-text");
            return dash_clientside.no_update;
        }}
        """,
        layout_helper.noop_callback_output(),
        Input('inpaint-btn', 'n_clicks'),
        prevent_initial_call=True)
    clientside_callback(
        f"""
        function(n_clicks) {{
            kaolin.util.canvas.clearCanvas("{viewer.resolve_id(inpaint_layer_id)}");
            kaolin.util.canvas.clearCanvas("{viewer.resolve_id(mask_layer_id)}");
            return dash_clientside.no_update;
        }}
        """,
        layout_helper.noop_callback_output(),
        Input('clear-btn', 'n_clicks'),
        prevent_initial_call=True)

    # Gaussian Training ------------------
    # "Optimize in Mask" runs the per-Gaussian color optimization that bakes
    # the inpainted 2D pixels back into 3D. On completion we switch the viewer
    # to View mode so the user sees the freshly-baked splats and clear the canvases.
    layout_helper.add_sidebar_section('Gaussian Training')
    opt_button = layout_helper.add_sidebar_component(
        html.Button('Optimize in Mask', id='opt-btn', className='btn btn-primary w-100'),
        section_name='Gaussian Training')
    clientside_callback(
        f"""
        function(n_clicks) {{
            send_opt_request("{viewer.resolve_id(inpaint_layer_id)}", "{viewer.resolve_id(mask_layer_id)}");
            kaolin.core.event.requestMode('view');
            document.getElementById('clear-btn').click();
            return dash_clientside.no_update;
        }}
        """,
        layout_helper.noop_callback_output(),
        Input('opt-btn', 'n_clicks'),
        prevent_initial_call=True)

    # Add the viewer -----------------------
    layout_helper.add_main_content([viewer])

    # ------------------------------------------------------------------------------------------------------------------
    # Build and run the app
    # ------------------------------------------------------------------------------------------------------------------
    app_builder.set_layout_helper(layout_helper)
    app, server = app_builder.build(ws_handlers)

    app_builder.start(args.port)

    # ------------------------------------------------------------------------------------------------------------------
    # Using the app
    # ------------------------------------------------------------------------------------------------------------------
    # Now navigate to localhost:{args.port}/
    # To access the website from another machine on the same network, either:
    # A. Make sure port is accessible, e.g. on Ubuntu: sudo ufw allow {args.port} and navigate to ip_addr:{args.port}
    # B. From another machine do port forwarding, e.g on Ubuntu: ssh -N -f -L $PORT:localhost:$PORT $HOST
    #    (where HOST is youruser@machine_address), then access from localhost:{args.port}/
