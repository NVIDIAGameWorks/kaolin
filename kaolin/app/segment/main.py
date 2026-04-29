from dash import html
import logging
import os
import simple_parsing


# Kaolin Utilities
#import kaolin
import kaolin.io.gaussians
#import kaolin.visualize.dash.auto_ui
#import kaolin.render.camera.gsplats_nerfstudio
from kaolin.utils.log import default_log_setup, add_log_level_flag
from kaolin.visualize.dash import StandardLayoutHelper, WebappBuilder, ViewerBuilder
from kaolin.visualize.dash.option import OptionKind, OptionSpec
from kaolin.visualize.dash.builtins import BehaviorLibrary
#import kaolin.visualize.dash
from kaolin.visualize.web.naming import UniqueIdGenerator
from kaolin.visualize.web.sockets import RemoteRenderingOptions
#import kaolin.visualize.web.sockets

# App-specific utilities
from .sam import GlobalSegmentAnything
from .handler import ServerSideUserSettings, ServerApplicationState, InteractiveCloudSelector, default_camera
from .input_abstraction import GaussianSplatInput
from .ui_util import SVG_FILTER, make_konva_action_controls, make_project_mask_buttons, hook_sidebar_sections_to_modes, \
    make_aggregate_ui

logger = logging.getLogger(__name__)

FILE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))


def read_cloud(fname, device, rep):
    if rep == 'splat':
        gsmodel = kaolin.io.gaussians.import_gaussiancloud(fname).to(device)
        print(gsmodel.to_string(print_stats=True))
        return GaussianSplatInput(gsmodel)
    else:
        raise NotImplementedError(f'The segmentation API for representation {rep} not implemented')


def default_render_kwargs():
    """Progressive rendering args accepted by AnyRendererMessageHandler."""
    return {
        "rendering_options": RemoteRenderingOptions(encode_format='jpeg', max_resolution=1000, jpeg_quality=80),
        "low_rendering_options": RemoteRenderingOptions(encode_format='jpeg', max_resolution=400, jpeg_quality=50),
        "high_rendering_options": RemoteRenderingOptions(encode_format='jpeg', max_resolution=2000, jpeg_quality=90)
    }

if __name__ == '__main__':
    """ Interactive segmentation web prototype. Current model design:
    - only one application state lives on the server
    - all application state from the server should be synced to any connecting client
    - no handling of competing actions from different clients (this is V0 prototype)
    """
    parser = simple_parsing.ArgumentParser('Segmentation App')
    parser.add_argument('--port', default=8001, type=int)
    parser.add_argument('--rep', type=str, default='splat')
    parser.add_argument('--input_scene', type=str, #required=True,
                        default='/mnt/masha_prev_2/Experiments/GSplats/inria/output/knit_meow/extra/baked.knit_meow.ply',
                        help='Input PLY or USD file (Gaussian splat)')
    parser.add_argument('--input_cameras', type=str, default=None,
                        help='Location of the cameras')
    parser.add_argument('--sam_model_id', type=str, default='facebook/sam2-hiera-large',
                        help='HuggingFace model ID for SAM2')
    parser.add_arguments(ServerSideUserSettings, dest="settings")
    add_log_level_flag(parser)
    args = parser.parse_args()
    device = "cuda"

    # Make sure logging is configured
    default_log_setup(level=args.log_level)

    # Configure global SAM2 model
    logger.info(f'Registering SAM2 model: {args.sam_model_id}')
    GlobalSegmentAnything.set_default_model_id(args.sam_model_id)

    # ------------------------------------------------------------------------------------------------------------------
    # Read Input:
    # TODO: should also figure out how to save; should user profide file? should we do save as?
    cloud = read_cloud(args.input_scene, device, args.rep)
    camera = default_camera().to(device)

    # Global server-side state, shared between clients
    application_state = ServerApplicationState(cloud, [])

    # ------------------------------------------------------------------------------------------------------------------
    # WebappBuilder: helps us create the whole app - layout, websockets, UI, etc.
    # ------------------------------------------------------------------------------------------------------------------
    app_builder = WebappBuilder(debug=False)
    app_builder.add_raw_body_html(SVG_FILTER)  # Styling for the mask layer

    # Setting controls ------------
    # Easily surface your settings as both flags (see parser.add_argument(ServerSideUserSettings)) and UI controls
    # these can control your server-side handlers, etc.
    server_controls = app_builder.add_user_settings(args.settings, 'server-settings')

    # WebSocket handlers ----------
    #   - You can create any handler for any message by overriding:
    #      - kaolin.visualize.web.sockets.SyncMessageHandlerProtocol or
    #      - kaolin.visualize.web.sockets.AsyncMessageHandlerProtocol
    #   - Lifecycle: each handler persists for an individual client connection
    #   - Broadcasting: app has access to all connections through GlobalWebSocketConnectionManager
    ws_handlers = []
    # We define a single handler to take care of rendering and all client requests
    ws_handlers.append(
        (InteractiveCloudSelector,
         (application_state, app_builder.get_user_settings_func('server-settings'), default_render_kwargs())))

    # ------------------------------------------------------------------------------------------------------------------
    # ViewerBuilder: helps us build the interactive viewer (on the webpage)
    # ------------------------------------------------------------------------------------------------------------------
    viewer_builder = ViewerBuilder(camera=camera, viewport_resize_mode='fixed')
    # Note: Masha is still handling last bugs in adaptive viewport

    # WebSocket Connections ----------
    # Add a named socket connection (this will be used to connect to all our handlers)
    #   - WINDOW_LOCATION means we'll serve these requests from the same webserver as the UI
    viewer_builder.add_websocket_connection("ws://WINDOW_LOCATION/websocket/", 'main-ws')

    # Layers -------------------------
    # Add visible or utility layers (there are stacked HTML elements)
    #   - Layers can be canvas, div, svg, etc.
    #   - We will add interaction to these layers through various Behaviors
    #   - Once we add a layer, we get a unique identifier we can use later
    #      - Important! these are not DOM Ids, but builder-level Ids, which need to be resolved for each built viewer
    #      (pass your own identifier to easily see which layer is which when debugging in-browser)

    # Where we draw remote rendering
    render_layer_id = viewer_builder.add_layer()

    # Where we draw current 2D mask
    baked_selection_layer_id = viewer_builder.add_layer(options={'extraClasses': 'mask-canvas'})

    # Where we allow interactive 2D selection drawing
    konva_layer_id = viewer_builder.add_layer('div')

    # Where we show SAM clicks
    svg_layer_id = viewer_builder.add_layer('svg', options={'width': 1000, 'height': 1000})

    # Behaviors ------------------------
    # Behaviors are defined in JavaScript and allow you to add any interactive or message-receiving behavior to a layer.

    # Kaolin is bundled with a number of behaviors.

    # You can define and register your own behavior in any js file; these can be React or class-based.
    #   - See kaolin/visualize/dash/components/src/ts/core/behavior/base.ts for behavior interface
    #   - To make behavior available you only have to register it in the js file -- it is created on page load
    #   - Behaviors can react to mouse motion, send and receive WS messages, and manipulate their active element
    #   - While python does not need to know about your behaviors at config time, you can avoid warnings
    BehaviorLibrary.register_user_directory(os.path.join(FILE_DIR, 'assets'))

    # List bundled and registered behaviors:
    print(BehaviorLibrary.to_string())

    # Shortcut behavior config for camera and remote rendering (see methods for what's under the hood)
    cam_controller_id = viewer_builder.add_camera_controller()
    draw_render_id, send_cam_id, request_render_id = viewer_builder.add_remote_rendering(
        active_layer_id=render_layer_id, connection_id='main-ws', cam_update_period=50, render_update_period=100)

    # Let's configure a kaolin-bundled selection behavior using konva library
    konva_behavior_id = viewer_builder.add_behavior(
        'konva_selection', active_layer_id=konva_layer_id,
        options={"mode": "box", "action": "new", "color": "#00ff88",
            "compositeCanvasId": baked_selection_layer_id,
        })

    # Let's configure a custom behavior (defined under assets/) for sam points, based on bundled kaolin behavior
    sam_behavior_id = viewer_builder.add_behavior(
        'sam_point', active_layer_id=svg_layer_id,
        options={'connectionId': 'main-ws', 'isPositive': True})
    # Let's configure a behavior to show the sam mask received from the server
    draw_sam_response_id = viewer_builder.add_behavior(
        'draw_remote_image', active_layer_id=baked_selection_layer_id,
        options={'messageTag': 'sam_result'})  # <-- our WS handler will tag its response

    # Let's configure our custom behavior (defined under assets/) for managing client-server segments list and visibility
    # (this behavior can send/receive server requests and works in tandem with our WS handler)
    seg_div = html.Div(id=UniqueIdGenerator.get_unique_id('seg_container'))
    seg_controller_id = viewer_builder.add_behavior(
        'seg_controller', options={"listContainerId": seg_div.id, 'switchToModeWhenDoneWithMask': 'view'})

    # Controlling behaviors ----------------

    # We can use a shortcut to hook up a UI control to a behavior.
    selection_controls = viewer_builder.add_user_behavior_options(konva_behavior_id, options=['mode'])

    # For *user* behaviors (declared in this app's assets/) supply an `OptionSpec` to provide
    # information for every controllable option, for example:
    sam_controls = viewer_builder.add_user_behavior_options(
        sam_behavior_id,
        options=[OptionSpec(name='isPositive', kind=OptionKind.BOOL, default=True)])

    # Manual behavior controls (see function in ui_util.py)
    selection_controls = selection_controls + make_konva_action_controls(konva_behavior_id)

    # Modes ----------------------------
    # We can combine behaviors into modes, which automatically creates little icons and shortcuts 1, 2, 3... to switch.
    # Manually, this can also be done with client-side call to: kaolin.core.event.requestBehaviorActiveStatus

    mode_info = {
        'view': ('bi-binoculars', 'View Settings'),
        'select': ('bi-brush', 'Manual 2D Masking'),
        'sam': ('bi-magic', 'SAM 2D Masking')
    }
    viewer_builder.add_mode(
        'view', [draw_render_id, seg_controller_id, send_cam_id, request_render_id, cam_controller_id],
        ui_icon=mode_info['view'][0], description='mode for interactive camera control',
        clear_layers=[baked_selection_layer_id], reset_behaviors=[sam_behavior_id, konva_behavior_id])
    viewer_builder.add_mode(
        'select', [draw_render_id, seg_controller_id, konva_behavior_id],
        ui_icon=mode_info['select'][0], description='mode for interactive 2D selection',
        reset_behaviors=[sam_behavior_id])
    viewer_builder.add_mode(
        'sam', [draw_render_id, seg_controller_id, draw_sam_response_id, sam_behavior_id],
        ui_icon=mode_info['sam'][0], description='mode for interactive 2D selection',
        clear_layers=[baked_selection_layer_id], reset_behaviors=[konva_behavior_id])
    viewer_builder.set_active_mode('view')

    # Building ----------------------
    # Now we construct actual viewer (eventually we'll support multiple viewports).
    viewer = viewer_builder.build()

    # ------------------------------------------------------------------------------------------------------------------
    # LayoutHelper: this optional utility helps us combine all the components and controls into a webpage
    # ------------------------------------------------------------------------------------------------------------------

    title = html.Span([html.Span("Gaussian Splat Segmentation"), html.I(className="p-2 bi bi-scissors me-2")])
    layout_helper = StandardLayoutHelper(title, max_width='lg')

    # Menus ---------------------------
    layout_helper.add_navbar_dropdown('File')
    save_item, save_as_item = layout_helper.add_navbar_dropdown_items('File', ['Save', 'Save As'])
    # TODO(Clement): figure out how to save

    # Help -----------------------------
    # Add a Help page (rendered from USAGE.md) reachable from the navbar.
    usage_path = os.path.join(os.path.dirname(__file__), 'USAGE.md')  # TODO(Clement): fill in page
    layout_helper.add_help_page(usage_path, add_to_navbar=True)

    # Convenience side bar ---------------------
    layout_helper.add_sidebar(disappearing=False)

    # Add link to our project ---
    lnk_help = layout_helper.add_sidebar_link('Project Page', bootstrap_icon='bi-house')
    lnk_help.href = 'https://research.nvidia.com/labs/sil/projects/ArtisanGS/'
    lnk_help.target = '_blank'  # in new tab

    # Add collapsible sections to sidebar ---
    # Create UI sections for each mode, with active one un-collapsed at load
    sidebar_sections = {}
    active_mode = 'view'
    for mode, info in mode_info.items():
        sidebar_sections[mode] = layout_helper.add_sidebar_section(
            info[1], bootstrap_icon=info[0], collapsed_on_load=(mode != active_mode))

    # Include the controls we created earlier for selection and SAM modes
    layout_helper.add_sidebar_components(selection_controls, mode_info['select'][1])
    layout_helper.add_sidebar_components(sam_controls, mode_info['sam'][1])

    # Expand the section corresponding to the active mode and collapse the others whenever the
    # viewer publishes a new `active_mode` (overlay button, keyboard 1/2/3, requestMode, etc.).
    mode_order = list(mode_info.keys())
    hook_sidebar_sections_to_modes(
        viewer.id, mode_order, [layout_helper.collapsible_section_ids[mode_info[mode][1]] for mode in mode_order])

    # We will also add sections not corresponding to modes
    # Project mask ---
    project_section_name = 'Project Mask to 3D'
    sidebar_sections['project'] = layout_helper.add_sidebar_section(
            project_section_name, bootstrap_icon='bi-box', collapsible=False)
    layout_helper.add_sidebar_components(
        make_project_mask_buttons(viewer.resolve_id(baked_selection_layer_id)),
        section_name = project_section_name)

    # Aggregate ---
    aggregate_section_name = 'Aggregate Masks to 3D'
    sidebar_sections['agg'] = layout_helper.add_sidebar_section(
        aggregate_section_name, bootstrap_icon='bi-box', collapsible=False)
    layout_helper.add_sidebar_components(
        make_aggregate_ui(),
        section_name=aggregate_section_name)
    # TODO(Clement): add UI here
    # Add mask
    # Clear masks
    # Aggregate

    # Segments ---
    segments_section_name = 'Segments'
    sidebar_sections['segments'] = layout_helper.add_sidebar_section(
        segments_section_name, bootstrap_icon='cloud-arrow-up-fill')
    layout_helper.add_sidebar_component(seg_div, section_name=segments_section_name)

    # Server Settings ----
    # Add the automatic server-side setting controls we created earlier. These will persist per clientside tab.
    layout_helper.add_sidebar_components(server_controls, 'Server Settings')

    # Add the viewer -----------------------
    layout_helper.add_main_content([viewer])

    # ------------------------------------------------------------------------------------------------------------------
    # Build and run the app
    # ------------------------------------------------------------------------------------------------------------------
    app_builder.set_layout_helper(layout_helper)
    app, server = app_builder.build(ws_handlers)

    # Any other configuration of the app can go here

    app_builder.start(args.port)

    # ------------------------------------------------------------------------------------------------------------------
    # Using the app
    # ------------------------------------------------------------------------------------------------------------------
    # Now navigate to localhost:{args.port}/
    # To access the website from another machine on the same network, either:
    # A. Make sure port is accessible, e.g. on Ubuntu: sudo ufw allow {args.port} and navigate to ip_addr:{args.port}
    # B. From another machine do port forwarding, e.g on Ubuntu: ssh -N -f -L $PORT:localhost:$PORT $HOST
    #    (where HOST is youruser@machine_address), the access from localhost:{args.port}/


# Debugging notes (if Kaolin component is not loading)
# from dash.development.base_component import ComponentRegistry
# print(f'Component registry initial: {len(ComponentRegistry.registry)} components')