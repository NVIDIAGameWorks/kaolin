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
import os.path
from pathlib import Path
from dataclasses import dataclass
from dash import Dash, html, dcc, Input, Output, callback, State
from dash.development.base_component import Component, ComponentRegistry
import dash_bootstrap_components as dbc
import logging
from typing import Protocol, Dict, Any, Union, runtime_checkable, Callable, Iterable, Literal, get_args, Mapping

#from docutils.nodes import sidebar

from tornado.wsgi import WSGIContainer
from tornado.web import Application, FallbackHandler, StaticFileHandler
from tornado.ioloop import IOLoop

from kaolin.visualize.web.sockets import WebSocketHandlerManager
from kaolin.visualize.dash.assets_helper import get_kaolin_assets_source_path, get_kaolin_assets_serve_path
from kaolin.visualize.web.naming import UniqueIdGenerator
import kaolin.version

logger = logging.getLogger(__name__)

__all__ = [
    'AppLayoutHelper',
    'StandardLayoutHelper',
]

@runtime_checkable
class AppLayoutHelper(Protocol):
    def layout(self) -> Iterable | Component:
        ...

    def stylesheets(self) -> Iterable:
        ...

    def configure_app(self, app: Dash):
        ...



BsBreakpoints = Literal['sm', 'md', 'lg', 'xl', 'xxl', 'flex']

def _max_width_to_bs_container_class(max_width: BsBreakpoints | None) -> str:
    if max_width is None or len(max_width) == 0:
        return ''
    allowable_values = get_args(BsBreakpoints)
    if max_width not in allowable_values:
        raise ValueError(f"max_width must be one of {allowable_values}")
    return f'container-{max_width}'

def _width_to_class(width: int):
    supported_width = min(100, max(5, int((width // 5) * 5)))
    return f'w-{supported_width}'


class StandardLayoutHelper(AppLayoutHelper):
    def __init__(self,
                 title='Kaolin App',
                 max_width: BsBreakpoints | None ='lg'):
        self.navbar_items = {}
        self.navbar_content, self.navbar = StandardLayoutHelper.make_navbar(title)
        self.main_container_class = _max_width_to_bs_container_class(max_width)
        self.help_page = None

        self.main_content = html.Div(children=[],
                                     className='kaolin-main-container')

        self.sidebar = None
        self.sidebar_sections = {}
        self.collapsible_sections = []
        self.collapsible_section_ids = {}
        self.modals = []
        self.sidebar_content = []
        self.sidebar_hide_button = None
        self.sidebar_show_button = None
        self.sidebar_extra_styles = {}
        self.dummy_id = UniqueIdGenerator.get_unique_id('dummy-output')

    def noop_callback_output(self):
        return Output(component_id=self.dummy_id, component_property='children', allow_duplicate=True)

    def layout(self):
        sidebar = [self.sidebar] if self.sidebar is not None else []
        sidebar_show = [self.sidebar_show_button] if self.sidebar_show_button is not None else []
        main_content_area = html.Main(
            className=f"{self.main_container_class} kaolin-primary-container",
            children=sidebar + sidebar_show + [self.main_content])

        modal_components = [m['component'] for m in self.modals]
        layout = [html.Div(id='kaolin-app', children=[
            self.navbar,
            main_content_area,
            *modal_components,
            html.Div(id=self.dummy_id, style={'display': 'none'}),
        ])]
        return layout

    def stylesheets(self):
        return [
            "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css",
            f'{get_kaolin_assets_serve_path()}/standard_layout.css']

    def add_sidebar(self, disappearing=False, width=20, min_width_px=300):
        """
        Adds sidebar component to the layout.
        Args:
            disappearing:

        Returns:
        """
        if self.sidebar is not None:
            logger.error(f'Sidebar exists already')
            return

        extra_classes = _width_to_class(width) if width is not None else ''

        if min_width_px is not None:
            self.sidebar_extra_styles['min-width'] = f'{min_width_px}px'

        self.sidebar_content = html.Div(children=[], className='kaolin-sidebar-content')
        children = [self.sidebar_content]
        if disappearing:
            extra_classes += ' kaolin-sliding-sidebar'
            self.sidebar_hide_button = html.Div(
                [html.I(className=f"bi bi-box-arrow-left kaolin-button-like")],
                id=UniqueIdGenerator.get_unique_id('sidebar-hide'),
                className='kaolin-sidebar-hide kaolin-button-like h4')
            self.sidebar_show_button = html.Div(
                [html.I(className="bi bi-tools")],
                id=UniqueIdGenerator.get_unique_id('show-sidebar'),
                className='kaolin-sidebar-show kaolin-button-like h4')
            children = [self.sidebar_hide_button] + children

        self.sidebar = html.Div(
            className=f"{extra_classes} kaolin-sidebar",
            children=children,
            id=UniqueIdGenerator.get_unique_id('sidebar'),
            style=self.sidebar_extra_styles
        )

    def _ensure_sidebar(self):
        if self.sidebar is None:
            logger.warning(f'Sidebar not initialized with add_sidebar; creating automatically')
            self.add_sidebar()

    def _ensure_sidebar_section(self, name):
        self._ensure_sidebar()
        if name not in self.sidebar_sections:
            self.add_sidebar_section(name)

    def add_sidebar_section(self, name, collapsible=True, bootstrap_icon=None,
                            collapsed_on_load=False):
        """Add a titled section to the sidebar.

        Args:
            name: Section title displayed in the header.
            collapsible: If True (default), the section header is clickable to collapse/expand its content,
                with a dropdown caret indicator shown in the header.
            bootstrap_icon: Optional Bootstrap icon name (e.g. ``"bi-boxes"``) shown before the section title.
            collapsed_on_load: If True, a collapsible section starts collapsed on page load (default False, expanded).

        Returns:
            The section container component.
        """
        self._ensure_sidebar()

        section_content = dbc.Nav(children=[],
                                  vertical=True,
                                  pills=True)
        self.sidebar_sections[name] = section_content

        section_parts = []
        if len(self.sidebar_content) > 0:
            section_parts.append(html.Hr(className="my-3"))

        title_children = []
        if bootstrap_icon is not None:
            title_children.append(html.I(className=f"bi {bootstrap_icon} me-2"))

        if collapsible:
            header_children = title_children + [html.Span(name)]
            caret_id = UniqueIdGenerator.get_unique_id(f'sec-caret-{name}')
            caret_dir = "bi-chevron-right" if collapsed_on_load else "bi-chevron-down"
            header_children.append(
                html.I(className=f"bi {caret_dir} kaolin-section-caret ms-auto", id=caret_id))
            header_id = UniqueIdGenerator.get_unique_id(f'sec-hd-{name}')
            section_title = html.H6(
                header_children,
                id=header_id,
                className="text-uppercase text-muted mb-3 kaolin-section-header d-flex align-items-center")
            body_id = UniqueIdGenerator.get_unique_id(f'sec-body-{name}')
            section_body = dbc.Collapse(section_content, id=body_id, is_open=not collapsed_on_load)
            section_parts.extend([section_title, section_body])
            self.collapsible_sections.append((header_id, body_id, caret_id))
            self.collapsible_section_ids[name] = {
                'header_id': header_id, 'body_id': body_id, 'caret_id': caret_id}
        else:
            section_title = html.H6(title_children + [html.Span(name)],
                                    className="text-uppercase text-muted mb-3")
            section_parts.extend([section_title, section_content])

        sec_id = UniqueIdGenerator.get_unique_id(f'sec-{name}')
        section_div = html.Div(section_parts, id=sec_id)
        self.add_sidebar_component(section_div)
        return section_div

    def add_sidebar_link(self, name, section_name=None, bootstrap_icon="bi-boxes"):
        if bootstrap_icon is not None:
            children = [html.I(className=f"bi {bootstrap_icon} me-2"), f" {name}"]
        else:
            children = [name]
        link_id = UniqueIdGenerator.get_unique_id(f'lnk-{name}')
        res = dbc.NavLink(children, id=link_id, href="#", active=False, className="py-2")

        return self.add_sidebar_component(res, section_name=section_name)

    def add_sidebar_links(self, names, section_name=None, bootstrap_icons=None):
        if bootstrap_icons is not None:
            if len(bootstrap_icons) != len(names):
                logger.warning(f'Different number of icons provided for {names}; ignoring')
                bootstrap_icons = None

        icons = bootstrap_icons if bootstrap_icons is not None else [None for _ in names]
        return [self.add_sidebar_link(n, section_name=section_name, bootstrap_icon=lnk) for n, lnk in zip(names, icons)]

    def add_sidebar_component(self, item: Component, section_name=None):
        if section_name is not None:
            self._ensure_sidebar_section(section_name)
            self.sidebar_sections[section_name].children.append(item)
        else:
            self._ensure_sidebar()
            self.sidebar_content.children.append(item)
        return item

    def add_sidebar_components(self, items: Iterable[Component], section_name=None):
        for item in items:
            self.add_sidebar_component(item, section_name=section_name)

    def add_navbar_component(self, item: Component):
        self.navbar_content.children.insert(0, item)
        return item

    def add_navbar_link(self, name):
        result = dbc.NavItem(dbc.NavLink(name, href="#"))
        return self.add_navbar_component(result)

    def add_navbar_dropdown(self, name):
        result = dbc.DropdownMenu(
                children=[],
                label=name,
                nav=True,
                in_navbar=True,
                className="me-2"
            )
        self.navbar_items[name] = result
        return self.add_navbar_component(result)

    def add_navbar_dropdown_item(self, dropdown_name, item: str | Component):
        if isinstance(item, str):
            if len(item) > 0:
                item = dbc.DropdownMenuItem(item, href="#")
            else:
                item = dbc.DropdownMenuItem(divider=True)
        self.navbar_items[dropdown_name].children.append(item)
        return item

    def add_navbar_dropdown_items(self, dropdown_name, items: Iterable):
        return [self.add_navbar_dropdown_item(dropdown_name, x) for x in items]

    def add_main_content(self, children):
        self.main_content.children.extend(children)
        print(self.main_content.children)

    def add_viewer_grid(self, viewers, columns=None, labels=None):
        """Lay out a flat list of viewers in a CSS grid in the main content area.

        Cells are equal ``1fr`` tracks at full height; grid items stretch to fill
        their cell, which gives each viewer a definite, non-zero size (viewers are
        width:100%/height:100% and size their canvas from the cell's pixel
        dimensions). Wraps the grid in a div and forwards it to
        :meth:`add_main_content`.

        Args:
            viewers: flat list of components, placed left-to-right, top-to-bottom.
            columns: number of columns. Defaults to a single row (all viewers side
                by side). E.g. ``columns=2`` with four viewers yields a 2x2 grid.
            labels: optional list of strings, one per viewer. When provided, each
                viewer is wrapped in a column with a text header above it. Must be
                the same length as ``viewers``.
        """
        viewers = list(viewers)
        if labels is not None:
            labels = list(labels)
            if len(labels) != len(viewers):
                raise ValueError(
                    f'labels length ({len(labels)}) must match number of viewers ({len(viewers)})'
                )
            viewers = [
                html.Div(
                    children=[
                        html.Div(label, style={
                            'textAlign': 'center',
                            'fontWeight': 'bold',
                            'padding': '4px 0',
                            'flexShrink': '0',
                        }),
                        html.Div(viewer, style={'flex': '1', 'minHeight': '0'}),
                    ],
                    style={'display': 'flex', 'flexDirection': 'column', 'height': '100%'},
                )
                for viewer, label in zip(viewers, labels)
            ]

        ncols = max(1, columns if columns is not None else len(viewers))
        nrows = max(1, (len(viewers) + ncols - 1) // ncols)  # ceil division

        # Inline styles (not a stylesheet class) so the layout always applies even
        # if a cached/older CSS is being served.
        grid = html.Div(
            children=viewers,
            style={
                'display': 'grid',
                'gridTemplateColumns': f'repeat({ncols}, 1fr)',
                'gridTemplateRows': f'repeat({nrows}, 1fr)',
                'gap': '10px',
                'width': '100%',
                'height': '100%',
            },
        )
        self.add_main_content([grid])

    @classmethod
    def make_navbar(cls, title):
        content = dbc.Nav(
            children=[],
            className="w-100",
            navbar=True
        )

        # For more consistent mobile performance with longer titles, put this at the top
        # For larger screens, this will have absolute position and not affect top bar thickness
        title_bar = html.Div(
            dbc.NavbarBrand(title, href="#", class_name="fw-bold kaolin-navbar-title"),
            className='kaolin-title-container'
        )
        # For more consistent mobile performance, put this at the bottom
        # For larger screens, this will have absolute position and not affect top bar thickness
        branding_bar = html.Div(
            [html.Div("Powered by", className='pw-by'),
             html.Img(src='/_kaolin_assets/logo.svg'),
             dcc.Link(html.Div([html.Div("Kaolin"),
                                html.Div(f"{kaolin.version.__version__}", className='br-version')],
                               className='br-kaolin'),
                          href="https://github.com/NVIDIAGameWorks/kaolin"),
             ],
            className='kaolin-branding-container'
        )

        # html.Button(
        #     html.I(className="bi bi-list"),
        #     className="btn btn-outline-light d-md-none me-2 mobile-menu-trigger",
        #     id="open-offcanvas"  # ID used by the callback below
        # ),

        controls = html.Div(
            # Mobile Sidebar Toggle Button (uses an ID for the Offcanvas trigger)
            children=[
                dbc.NavbarToggler(id="navbar-toggler-mobile"),

                # The Main Menu Links (File, Edit, Reports, User)
                dbc.Collapse(
                    content,
                    id="navbar-collapse",  # ID used by dbc.Navbar for collapse
                    is_open=False,
                    navbar=True,
                )],
            className="container-lg kaolin-menu-container"
        )

        full_navbar = dbc.Navbar(
            dbc.Container(
                fluid=True,  # want class "container"
                children=[title_bar, controls, branding_bar],
                class_name="kaolin-navbar-container"
            ),
            color="primary",
            dark=True,
            expand="lg",  # Controls when the main menu collapses (medium screen and up)
        )

        return content, full_navbar

    @staticmethod
    def make_markdown(md_file_or_string=None):
        markdown_text = ''
        if md_file_or_string is not None:
            is_file = False
            if isinstance(md_file_or_string, (str, Path)):
                try:
                    is_file = os.path.isfile(md_file_or_string)
                except (OSError, ValueError):
                    is_file = False
            if is_file:
                with open(md_file_or_string, "r", encoding="utf-8") as file:
                    markdown_text = file.read()
            else:
                markdown_text = md_file_or_string

        # Allow raw HTML so Bootstrap-icon markup embedded in markdown renders.
        content = dcc.Markdown(children=markdown_text, dangerously_allow_html=True)
        return content

    def add_modal(self, content, title=None):
        """Create a centered overlay modal with a close (X) button.

        Args:
            content: A Dash component (or list of components) to display inside the modal body.
            title: Optional title shown at the top of the modal.

        Returns:
            A dict describing the modal with keys ``component``, ``overlay_id``, ``close_id`` and
            ``trigger_ids`` (a list of ids whose clicks open the modal; append to it to add triggers).
        """
        overlay_id = UniqueIdGenerator.get_unique_id('modal')
        close_id = UniqueIdGenerator.get_unique_id('modal-close')

        close_button = html.Div(
            html.I(className="bi bi-x-lg"),
            id=close_id,
            className="kaolin-modal-close kaolin-button-like h5")

        panel_children = [close_button]
        if title is not None:
            panel_children.append(html.H5(title, className="kaolin-modal-title mb-3"))
        panel_children.append(html.Div(content, className="kaolin-modal-body"))

        panel = html.Div(panel_children, className="kaolin-modal-panel")
        overlay = html.Div(panel, id=overlay_id, className="kaolin-modal-overlay", style={'display': 'none'})

        modal = {'component': overlay, 'overlay_id': overlay_id, 'close_id': close_id, 'trigger_ids': []}
        self.modals.append(modal)
        return modal

    def add_help_page(self, md_file_or_string=None, add_to_navbar=True, bootstrap_icon='bi-info-circle-fill'):
        assert self.help_page is None, f'Only one help page supported currently'

        content = self.make_markdown(md_file_or_string)
        modal = self.add_modal(content)
        self.help_page = modal

        if add_to_navbar:
            trigger_id = UniqueIdGenerator.get_unique_id('help-trigger')
            trigger_children = [html.I(className=f"bi {bootstrap_icon} me-1"), "Help"]
            trigger = dbc.NavItem(
                dbc.NavLink(trigger_children, href="#", id=trigger_id, className="kaolin-help-trigger"))
            self.add_navbar_component(trigger)
            modal['trigger_ids'].append(trigger_id)

        return modal

    def register_help_trigger(self, trigger):
        """Register an existing component (or its id) as an extra trigger that opens the help page.

        Must be called after :meth:`add_help_page` and before the app is built.
        """
        assert self.help_page is not None, 'Call add_help_page before registering a trigger'
        trigger_id = trigger if isinstance(trigger, str) else trigger.id
        self.help_page['trigger_ids'].append(trigger_id)
        return trigger

    def configure_app(self, app):
        # Clientside Callback 2: Controls the main navbar collapse on mobile screens
        app.clientside_callback(
            """
            function(n_clicks, is_open) {
                // Only toggle if button has been clicked (n_clicks is not null/undefined)
                if (n_clicks && n_clicks > 0) {
                    return !is_open;
                }
                return is_open;
            }
            """,
            Output("navbar-collapse", "is_open"),
            Input("navbar-toggler-mobile", "n_clicks"),
            State("navbar-collapse", "is_open"),
        )

        if self.sidebar_hide_button is not None and self.sidebar is not None:
            new_styles = copy.deepcopy(self.sidebar_extra_styles)
            new_styles['left'] = '-120%'
            app.clientside_callback(
                f"""
                function(n_clicks) {{
                    return {new_styles};
                }}
                """,
                Output(self.sidebar.id, "style", allow_duplicate=True),
                Input(self.sidebar_hide_button.id, "n_clicks"),
                prevent_initial_call=True
            )

        if self.sidebar_show_button is not None and self.sidebar is not None:
            new_styles = copy.deepcopy(self.sidebar_extra_styles)
            new_styles['left'] = '0'
            app.clientside_callback(
                f"function(n_clicks) {{ return {new_styles}; }}",
                Output(self.sidebar.id, "style", allow_duplicate=True),
                Input(self.sidebar_show_button.id, "n_clicks"),
                prevent_initial_call=True
            )

        for header_id, body_id, caret_id in self.collapsible_sections:
            app.clientside_callback(
                """
                function(n_clicks, is_open) {
                    const new_open = n_clicks ? !is_open : is_open;
                    const caret_class = new_open
                        ? "bi bi-chevron-down kaolin-section-caret ms-auto"
                        : "bi bi-chevron-right kaolin-section-caret ms-auto";
                    return [new_open, caret_class];
                }
                """,
                [Output(body_id, "is_open"), Output(caret_id, "className")],
                Input(header_id, "n_clicks"),
                State(body_id, "is_open"),
                prevent_initial_call=True
            )

        for modal in self.modals:
            # Open when any trigger is clicked; close when the X is clicked.
            inputs = [Input(modal['close_id'], 'n_clicks')]
            inputs += [Input(trigger_id, 'n_clicks') for trigger_id in modal['trigger_ids']]
            app.clientside_callback(
                f"""
                function() {{
                    const ctx = window.dash_clientside.callback_context;
                    if (!ctx.triggered || ctx.triggered.length === 0) {{
                        return window.dash_clientside.no_update;
                    }}
                    const trig = ctx.triggered[0].prop_id.split('.')[0];
                    if (trig === "{modal['close_id']}") {{
                        return {{'display': 'none'}};
                    }}
                    return {{'display': 'flex'}};
                }}
                """,
                Output(modal['overlay_id'], 'style'),
                inputs,
                prevent_initial_call=True
            )