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

"""Tests for kaolin.visualize.dash.layout."""

from unittest.mock import Mock

import pytest
from dash import html

from kaolin.visualize.dash.layout import StandardLayoutHelper


class TestStandardLayoutHelper:
    @pytest.mark.parametrize('max_width', [None, 'sm', 'md', 'lg', 'xl', 'xxl', 'fluid'])
    def test_layout(self, max_width):
        helper = StandardLayoutHelper(title='Example', max_width=max_width)
        content = html.Div('Content')
        helper.add_main_content([content])
        root = helper.layout()[0]
        assert root.id == 'kaolin-app'
        assert root.children[0] is helper.navbar
        main = root.children[1]
        assert main.children == [helper.main_content]
        assert helper.main_content.children == [content]
        if max_width:
            assert f'container-{max_width}' in main.className
        assert helper.stylesheets()[-1] == '/_kaolin_assets/standard_layout.css'
        with pytest.raises(ValueError, match='max_width'):
            StandardLayoutHelper(max_width='invalid')

    @pytest.mark.parametrize('count,columns,rows', [(0, None, 1), (1, None, 1), (5, 2, 3)])
    def test_viewer_grid(self, count, columns, rows):
        helper = StandardLayoutHelper()
        viewers = [html.Div(id=f'viewer-{i}') for i in range(count)]
        helper.add_viewer_grid(viewers, columns=columns)
        grid = helper.main_content.children[0]
        assert grid.children == viewers
        assert grid.style['gridTemplateColumns'] == f'repeat({columns or max(1, count)}, 1fr)'
        assert grid.style['gridTemplateRows'] == f'repeat({rows}, 1fr)'
        assert grid.style['height'] == '100%'

    def test_viewer_labels(self):
        helper = StandardLayoutHelper()
        viewer = html.Div(id='viewer')
        helper.add_viewer_grid([viewer], labels=['Camera'])
        cell = helper.main_content.children[0].children[0]
        assert cell.children[0].children == 'Camera'
        assert cell.children[1].children is viewer
        assert cell.style['display'] == 'flex'
        assert cell.children[1].style == {'flex': '1', 'minHeight': '0'}
        with pytest.raises(ValueError, match='labels length'):
            helper.add_viewer_grid([viewer], labels=[])

    def test_sidebar_and_modals(self):
        helper = StandardLayoutHelper()
        helper.add_sidebar(disappearing=True, width=25, min_width_px=200)
        section = helper.add_sidebar_section('Tools', collapsed_on_load=True)
        link = helper.add_sidebar_link('Draw', section_name='Tools')
        modal = helper.add_help_page('# Help')
        helper.register_help_trigger('extra-help')
        assert helper.sidebar_sections['Tools'].children == [link]
        assert section.children[-1].is_open is False
        assert helper.sidebar.style['minWidth'] == '200px'
        assert modal['component'].style == {'display': 'none'}
        assert modal['component'] in helper.layout()[0].children
        assert helper.sidebar in helper.layout()[0].children[1].children

        app = Mock()
        helper.configure_app(app)
        calls = app.clientside_callback.call_args_list
        outputs = [call.args[1] for call in calls]
        assert outputs[0].component_id == 'navbar-collapse'
        assert outputs[1].component_id == helper.sidebar.id
        assert outputs[2].component_id == helper.sidebar.id
        assert outputs[3][0].component_id == helper.collapsible_section_ids['Tools']['body_id']
        assert outputs[4].component_id == modal['overlay_id']
        assert [item.component_id for item in calls[4].args[2]] == [
            modal['close_id'], *modal['trigger_ids']]
