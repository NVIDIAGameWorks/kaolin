from dash import html, clientside_callback, Input, Output, State, dcc
import dash_bootstrap_components as dbc

import kaolin.visualize.dash.auto_ui as auto_ui
from kaolin.visualize.web.naming import UniqueIdGenerator

# Corresponds to 2D segmentation overlay styling with color: #24B7BC
SVG_FILTER = """
<svg width="0" height="0" style="position: absolute; width: 0; height: 0; pointer-events: none;">
<filter id="colorize-mask" color-interpolation-filters="sRGB">
<feColorMatrix id="color-matrix" type="matrix" values="
0 0 0 0 0.141
0 0 0 0 0.718
0 0 0 0 0.737
0 0 0 0.75 0
" />
</filter>
</svg>
"""

def make_konva_action_controls(konva_behavior_id):
    actions = ['new', 'add', 'intersect', 'subtract']
    action_icons = ['bi-square-fill', 'bi-union', 'bi-intersect', 'bi-subtract']
    action_btns, action_tooltips = auto_ui.make_icon_buttons(action_icons, actions)
    controls = [
        html.Div(dbc.ButtonGroup(action_btns, className='btn-group-toggle'), className='d-flex justify-content-center my-2'),
    ] + action_tooltips

    # Wire the action toggle group: clicking a button makes it the single active one and requests the
    # konva_selection behavior's `action` option client-side (no server roundtrip). The default active
    # button ('new', active_idx=0) matches the behavior's default `action` option, so no initial sync.
    action_btn_ids = [btn.id for btn in action_btns]
    action_toggle_outputs = []
    for _bid in action_btn_ids:
        action_toggle_outputs.append(Output(_bid, 'active'))
        action_toggle_outputs.append(Output(_bid, 'outline'))
    clientside_callback(
        f"""
        function(...n_clicks) {{
            const ids = {action_btn_ids};
            const actions = {actions};
            const ctx = dash_clientside.callback_context;
            const noUpdate = ids.flatMap(() => [dash_clientside.no_update, dash_clientside.no_update]);
            if (!ctx.triggered || ctx.triggered.length === 0) {{ return noUpdate; }}
            const triggeredId = ctx.triggered[0].prop_id.split('.')[0];
            const idx = ids.indexOf(triggeredId);
            if (idx < 0) {{ return noUpdate; }}
            kaolin.core.event.requestBehaviorSetOption("{konva_behavior_id}", "action", actions[idx]);
            const result = [];
            for (let i = 0; i < ids.length; i++) {{
                const isActive = (i === idx);
                result.push(isActive);
                result.push(!isActive);
            }}
            return result;
        }}
        """,
        action_toggle_outputs,
        [Input(_bid, 'n_clicks') for _bid in action_btn_ids],
        prevent_initial_call=True,
    )
    return controls

def make_dummy():
    # Dash requires output even for no-op callbacks; this is a shortcut to make one
    dummy_id = UniqueIdGenerator.get_unique_id('dummy-output')
    return html.Div(id=dummy_id, style={'display': 'none'}), \
        Output(component_id=dummy_id, component_property='children', allow_duplicate=True)

def make_project_mask_buttons(mask_canvas_id):
    # A row of icon buttons to project the current 2D selection onto the 3D selection, keyed by operation.
    project_ops = [
        ('new', 'bi-square-fill', 'New: project 2D selection to new 3D selection'),
        ('add', 'bi-union', 'Add: project 2D selection to add frustum to 3D selection'),
        ('intersect', 'bi-intersect', 'Intersect: project 2D selection to intersect frustum with 3D selection'),
        ('subtract', 'bi-subtract', 'Subtract: project 2D selection to subtract frustum from 3D selection')
    ]
    project_btns = {}
    project_buttons = []
    project_tooltips = []
    for op, icon, explanation in project_ops:
        btn_id = f'project-{op}-btn'
        btn = dbc.Button(html.I(className=f'bi {icon}'), id=btn_id, color='success', outline=False,
                         className='rounded-pill px-1')
        project_btns[op] = btn
        project_buttons.append(btn)
        project_tooltips.append(dbc.Tooltip(explanation, target=btn_id))
    components = [dbc.ButtonGroup(project_buttons, className='gap-1')] + project_tooltips

    # Wire each project button: grab the mask layer and send a 'mask_project' request with the action.
    # The op keys match the lowercase SelectAction values, so they are sent directly.
    dummy_div, dummy_output = make_dummy()
    components.append(dummy_div)
    for op in project_btns:
        clientside_callback(
            f"""
                function(n_clicks) {{
                    send_mask_project_request("{mask_canvas_id}", "{op}");
                    return window.dash_clientside.no_update;
                }}
                """,
            dummy_output,
            Input(f'project-{op}-btn', 'n_clicks'),
            prevent_initial_call=True)

    return components

def hook_sidebar_sections_to_modes(viewer_id, modes, collapsible_section_ids):
    # Expand the section corresponding to the active mode and collapse the others whenever the
    # viewer publishes a new `active_mode` (overlay button, keyboard 1/2/3, requestMode, etc.).

    # TODO: this is a hack, assumes too much about the way collapsible sections are implemented
    section_outputs = []
    for mode, ids in zip(modes, collapsible_section_ids):
        section_outputs.append(Output(ids['body_id'], 'is_open', allow_duplicate=True))
        section_outputs.append(Output(ids['caret_id'], 'className', allow_duplicate=True))

    # TODO: this should not assume styling on the section
    clientside_callback(
        f"""
        function(activeMode) {{
            const modes = {modes};
            const result = [];
            for (const m of modes) {{
                const open = (m === activeMode);
                result.push(open);
                result.push(open
                    ? "bi bi-chevron-down kaolin-section-caret ms-auto"
                    : "bi bi-chevron-right kaolin-section-caret ms-auto");
            }}
            return result;
        }}
        """,
        section_outputs,
        Input(viewer_id, 'active_mode'),
        prevent_initial_call=True,
    )

def make_aggregate_ui():
    tooltips = []
    add_mask_btn = dbc.Button("Add Mask", id=UniqueIdGenerator.get_unique_id('agg'), color='primary', outline=True,
                     className='rounded-pill px-1')
    tooltips.append(dbc.Tooltip('Add current mask as reference', target=add_mask_btn.id))
    todo_btn = dbc.Button("TODO :)", id=UniqueIdGenerator.get_unique_id('agg'), color='success', outline=False,
                              className='rounded-pill px-1')
    tooltips.append(dbc.Tooltip('TBD', target=todo_btn.id))

    buttons = dbc.ButtonGroup([add_mask_btn, todo_btn], className='gap-1')

    # TODO: hook up UI to events as well

    return [buttons] + tooltips