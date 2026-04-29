#!/usr/bin/env python

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

"""2D interactive-behavior playground and test-case generator.

Creates one viewer mode per 2D interactive behavior (``drawing``,
``konva_drawing``, ``konva_selection``, ``svg_annotation``), auto-generates a UI
control panel for each, and attaches a ``record_interactions`` behavior that is
active in *every* mode. The recorder captures the exact pointer/click stream each behavior
receives, so a session here doubles as a fixture generator: drive a behavior by
hand, then use the *File -> Save Interactions As* menu to download the captured
interactions as JSON for use in unit tests.

Run:

    python -m examples.tutorial.app.interact2d_main --port 8000

then open http://localhost:8000/ and switch modes with the top-left icons (or
the digit keys 1..3 while hovering the viewer).
"""

import argparse
import logging

from dash import html, dcc, Input, Output, clientside_callback

from kaolin.utils.log import default_log_setup, add_log_level_flag
from kaolin.visualize.dash import StandardLayoutHelper, WebappBuilder, ViewerBuilder
from kaolin.visualize.dash.option import OptionKind, OptionSpec

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('2D interactive-behavior playground / test-case generator')
    parser.add_argument('--port', default=8000, type=int)
    parser.add_argument('--size', default=600, type=int, help='Square viewer size in pixels.')
    add_log_level_flag(parser)
    args = parser.parse_args()
    default_log_setup(level=args.log_level)

    # ------------------------------------------------------------------------------------------------------------------
    # Viewer: stacked 2D layers + one behavior per interaction style + a shared recorder.
    # ------------------------------------------------------------------------------------------------------------------
    viewer_builder = ViewerBuilder(width=args.size, height=args.size, viewport_resize_mode='fixed')

    # Layers (stacked bottom -> top). Pointer input is always captured by the
    # viewer's top "event canvas" and dispatched to every active behavior, so
    # these layers only determine where each behavior *draws*.
    selection_canvas_id = viewer_builder.add_layer('canvas', identifier='selection')  # konva_selection composite output
    draw_canvas_id = viewer_builder.add_layer('canvas', identifier='draw')            # drawing brush output
    konva_div_id = viewer_builder.add_layer('div', identifier='konva')                # konva_drawing shapes
    selection_div_id = viewer_builder.add_layer('div', identifier='selection-stage')  # konva_selection draw surface
    svg_layer_id = viewer_builder.add_layer('svg', identifier='svg',                  # svg_annotation markers
                                            options={'width': args.size, 'height': args.size})

    # One behavior per 2D interaction style. All start inactive; modes turn them on.
    drawing_id = viewer_builder.add_behavior(
        'drawing', active_layer_id=draw_canvas_id,
        options={'color': '#ff3344', 'thickness': 9}, is_active=False)
    konva_drawing_id = viewer_builder.add_behavior(
        'konva_drawing', active_layer_id=konva_div_id,
        options={'mode': 'box', 'color': '#00ff88', 'opacity': 0.2}, is_active=False)
    konva_selection_id = viewer_builder.add_behavior(
        'konva_selection', active_layer_id=selection_div_id,
        options={'action': 'new', 'mode': 'box', 'color': '#3399ff', 'opacity': 0.35,
                 # Resolved to the selection canvas's DOM id at build time.
                 'compositeCanvasId': selection_canvas_id},
        is_active=False)
    # Stamps marker assets onto the SVG layer at each click. Two filled assets
    # (red circle, green square) defined here, app-side; the createSvgGroup
    # defaults supply size/stroke-width, so we only override fill + outline color.
    # `activeAsset` selects which one a click stamps; it's driven by a dropdown
    # (see add_user_behavior_options below) whose choices are these asset names.
    svg_assets = {
        'red_circle': {'elements': [{'name': 'circle', 'options': {'fill': '#e23b3b', 'stroke': '#7a1010'}}]},
        'green_square': {'elements': [{'name': 'rect', 'options': {'fill': '#3bbf57', 'stroke': '#176b2a'}}]},
    }
    svg_annotation_id = viewer_builder.add_behavior(
        'svg_annotation', active_layer_id=svg_layer_id,
        options={'assets': svg_assets, 'activeAsset': 'red_circle'}, is_active=False)

    # The recorder logs the pointer/click stream of whatever mode is active.
    # Bound to the event canvas so its coordinates match the interaction surface.
    record_id = viewer_builder.add_behavior(
        'record_interactions', active_layer_id='EVENT_CANVAS',
        options={'record': 'all', 'elementType': 'canvas'}, is_active=True)

    # Modes: one per behavior, each also keeps the recorder active. Switching
    # modes (icons / digit keys) is wired up automatically by the viewer.
    all_behaviors = [drawing_id, konva_drawing_id, konva_selection_id, svg_annotation_id, record_id]
    all_canvases = [draw_canvas_id, selection_canvas_id]
    # SVG markers aren't a canvas, so they're cleared via the behavior reset (in
    # all_behaviors), not clear_layers.
    mode_kwargs = {'clear_layers': all_canvases, 'reset_behaviors': all_behaviors}
    viewer_builder.add_mode('draw', [drawing_id, record_id],
                            ui_icon='bi-brush', description='freehand brush drawing', **mode_kwargs)
    viewer_builder.add_mode('konva_draw', [konva_drawing_id, record_id],
                            ui_icon='bi-pentagon', description='konva box / polygon / freeform drawing', **mode_kwargs)
    viewer_builder.add_mode('select', [konva_selection_id, record_id],
                            ui_icon='bi-bounding-box', description='konva selection compositing', **mode_kwargs)
    viewer_builder.add_mode('svg_annotate', [svg_annotation_id, record_id],
                            ui_icon='bi-geo-alt', description='svg marker stamping', **mode_kwargs)
    viewer_builder.set_active_mode('draw')

    # Auto-UI controls for each behavior's user-facing (uiBound) options. These
    # must be requested BEFORE build(), which wires their client-side bindings.
    drawing_controls = viewer_builder.add_user_behavior_options(drawing_id)
    konva_drawing_controls = viewer_builder.add_user_behavior_options(konva_drawing_id)
    selection_controls = viewer_builder.add_user_behavior_options(konva_selection_id)
    # activeAsset is uiBound:false in the schema (its valid values are instance-
    # specific), so instrument it here as an enum dropdown whose choices are the
    # asset names defined above -- dynamic from the single svg_assets source.
    svg_controls = viewer_builder.add_user_behavior_options(
        svg_annotation_id,
        options=[OptionSpec(name='activeAsset', kind=OptionKind.ENUM, values=list(svg_assets.keys()))])

    # Mirror every behavior option change onto the recorder (the '*' wildcard), so
    # option edits (brush color/thickness, konva mode/opacity, etc.) are captured
    # in the recorded interaction stream alongside the pointer/click events.
    viewer_builder.broadcast_behavior_option('*', target_behavior_ids=record_id)

    viewer = viewer_builder.build()

    # ------------------------------------------------------------------------------------------------------------------
    # Layout: sidebar with the auto-generated per-behavior controls + a File menu.
    # ------------------------------------------------------------------------------------------------------------------
    title = html.Span([html.I(className='p-2 bi bi-vector-pen me-2'), html.Span('2D Interaction Playground')])
    layout_helper = StandardLayoutHelper(title, max_width='flex')
    layout_helper.add_sidebar(disappearing=False)

    layout_helper.add_sidebar_section('Brush (drawing)')
    layout_helper.add_sidebar_components(drawing_controls, section_name='Brush (drawing)')

    layout_helper.add_sidebar_section('Konva Drawing')
    layout_helper.add_sidebar_components(konva_drawing_controls, section_name='Konva Drawing')

    layout_helper.add_sidebar_section('Selection')
    layout_helper.add_sidebar_components(selection_controls, section_name='Selection')

    layout_helper.add_sidebar_section('SVG Annotation')
    layout_helper.add_sidebar_components(svg_controls, section_name='SVG Annotation')

    # Recording status + reset. The live count is read from the recorder via the
    # same `reportInteractions` callback used to save (see the File menu below).
    layout_helper.add_sidebar_section('Recording')
    layout_helper.add_sidebar_component(
        html.Div('Recorded interactions: 0', id='record-count', className='text-muted small'),
        section_name='Recording')
    layout_helper.add_sidebar_component(dcc.Interval(id='record-poll', interval=500), section_name='Recording')
    layout_helper.add_sidebar_component(
        html.Button('Reset Interactions', id='reset-interactions-btn', className='btn btn-secondary w-100 mt-2'),
        section_name='Recording')

    clientside_callback(
        f"""
        function(n) {{
            let count = 0;
            kaolin.core.event.requestBehaviorEdit(
                "{record_id}", "reportInteractions",
                function(records) {{ count = records.length; }}, "{viewer.id}");
            return "Recorded interactions: " + count;
        }}
        """,
        Output('record-count', 'children'),
        Input('record-poll', 'n_intervals'),
    )

    clientside_callback(
        f"""
        function(n_clicks) {{
            if (!n_clicks) return dash_clientside.no_update;
            kaolin.core.event.requestBehaviorReset("{record_id}", undefined, "{viewer.id}");
            return dash_clientside.no_update;
        }}
        """,
        layout_helper.noop_callback_output(),
        Input('reset-interactions-btn', 'n_clicks'),
        prevent_initial_call=True,
    )

    # File menu: download a layer's rendered pixels as a PNG (the plain brush /
    # selection canvases, or the Konva stage canvas), or save the recorded
    # interaction log as JSON. Layer ids resolve to their built DOM ids so the
    # client-side helpers can find the actual <canvas> elements.
    layout_helper.add_navbar_dropdown('File')
    (download_drawing_item, download_konva_item, download_selection_item, download_svg_item,
     save_item) = layout_helper.add_navbar_dropdown_items(
        'File', ['Download Drawing', 'Download Konva Drawing', 'Download Selection', 'Download SVG',
                 'Save Interactions As'])
    download_drawing_item.id = 'download-drawing-btn'
    download_konva_item.id = 'download-konva-drawing-btn'
    download_selection_item.id = 'download-selection-btn'
    download_svg_item.id = 'download-svg-btn'
    save_item.id = 'save-interactions-btn'

    # "Download Drawing" / "Download Selection": encode the target layer canvas as
    # a PNG (blobFromCanvas reads its pixels by DOM id) and save it via downloadBlob.
    def add_download_canvas_callback(button_id, canvas_dom_id, default_filename):
        clientside_callback(
            f"""
            function(n_clicks) {{
                if (!n_clicks) return dash_clientside.no_update;
                const filename = window.prompt("Download as:", "{default_filename}");
                if (!filename) return dash_clientside.no_update;
                kaolin.util.canvas.blobFromCanvas("{canvas_dom_id}", "png").then(function(blob) {{
                    if (blob) kaolin.util.file.downloadBlob(filename, blob);
                }});
                return dash_clientside.no_update;
            }}
            """,
            layout_helper.noop_callback_output(),
            Input(button_id, 'n_clicks'),
            prevent_initial_call=True,
        )

    add_download_canvas_callback('download-drawing-btn', viewer.resolve_id(draw_canvas_id), 'drawing.png')
    add_download_canvas_callback('download-selection-btn', viewer.resolve_id(selection_canvas_id), 'selection.png')

    # "Download Konva Drawing": konva_drawing renders into a Konva stage hosted in
    # the `konva` div layer (not a plain layer canvas), so there is no top-level
    # canvas DOM id to hand to blobFromCanvas. Resolve the layer div's DOM id and
    # grab the <canvas> Konva renders inside it (`.konvajs-content canvas`); the
    # element is passed straight to blobFromCanvas (which accepts a canvas, not
    # just an id) and downloaded.
    konva_div_dom_id = viewer.resolve_id(konva_div_id)
    clientside_callback(
        f"""
        function(n_clicks) {{
            if (!n_clicks) return dash_clientside.no_update;
            const container = document.getElementById("{konva_div_dom_id}");
            const canvas = container ? container.querySelector("canvas") : null;
            if (!canvas) {{
                console.warn("Download Konva Drawing: no canvas found in '{konva_div_dom_id}'.");
                return dash_clientside.no_update;
            }}
            const filename = window.prompt("Download as:", "konva_drawing.png");
            if (!filename) return dash_clientside.no_update;
            kaolin.util.canvas.blobFromCanvas(canvas, "png").then(function(blob) {{
                if (blob) kaolin.util.file.downloadBlob(filename, blob);
            }});
            return dash_clientside.no_update;
        }}
        """,
        layout_helper.noop_callback_output(),
        Input('download-konva-drawing-btn', 'n_clicks'),
        prevent_initial_call=True,
    )

    # "Download SVG": svg_annotation stamps marker nodes into the SVG layer, so
    # there are no pixels to read back. svgToBlob serializes the whole <svg>
    # element (root + namespace, a valid standalone document) -- this is the SVG
    # golden the svg_annotation replay test compares against.
    svg_dom_id = viewer.resolve_id(svg_layer_id)
    clientside_callback(
        f"""
        function(n_clicks) {{
            if (!n_clicks) return dash_clientside.no_update;
            const svg = document.getElementById("{svg_dom_id}");
            if (!svg) {{
                console.warn("Download SVG: no svg layer found ('{svg_dom_id}').");
                return dash_clientside.no_update;
            }}
            const filename = window.prompt("Download as:", "svg_annotation.svg");
            if (!filename) return dash_clientside.no_update;
            kaolin.util.file.downloadBlob(filename, kaolin.util.svg.svgToBlob(svg));
            return dash_clientside.no_update;
        }}
        """,
        layout_helper.noop_callback_output(),
        Input('download-svg-btn', 'n_clicks'),
        prevent_initial_call=True,
    )

    # "Save Interactions As" pulls the recorded log out of the recorder
    # (synchronously, via requestBehaviorEdit -> reportInteractions) and downloads
    # it as JSON with a user-chosen filename.
    clientside_callback(
        f"""
        function(n_clicks) {{
            if (!n_clicks) return dash_clientside.no_update;
            let records = [];
            kaolin.core.event.requestBehaviorEdit(
                "{record_id}", "reportInteractions",
                function(r) {{ records = r || []; }}, "{viewer.id}");
            const payload = JSON.stringify(records, null, 2);
            const filename = window.prompt("Save interactions as:", "interactions.json");
            if (!filename) return dash_clientside.no_update;
            kaolin.util.file.downloadTextFile(filename, payload, "application/json");
            return dash_clientside.no_update;
        }}
        """,
        layout_helper.noop_callback_output(),
        Input('save-interactions-btn', 'n_clicks'),
        prevent_initial_call=True,
    )

    layout_helper.add_viewer_grid([viewer], labels=['2D Interaction Playground'])

    # ------------------------------------------------------------------------------------------------------------------
    # Build and run (no server-side websockets needed: everything is client-side).
    # ------------------------------------------------------------------------------------------------------------------
    app_builder = WebappBuilder(debug=False)
    app_builder.set_layout_helper(layout_helper)
    app, server = app_builder.build(ws_handlers=[])
    logger.info(f'Open http://localhost:{args.port}/ to interact and record.')
    app_builder.start(args.port)
