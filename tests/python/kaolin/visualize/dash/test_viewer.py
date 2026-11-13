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

"""Tests for kaolin.visualize.dash.viewer."""

import logging
from unittest.mock import Mock

import pytest
from dash import dcc

from kaolin.render.easy_render import default_camera
from kaolin.visualize.dash import auto_ui, viewer
from kaolin.visualize.dash.option import OptionKind, OptionSpec
from kaolin.visualize.dash.viewer import KaolinViewer, ViewerBuilder


@pytest.fixture
def builder(monkeypatch):
    # Callback registration is checked without changing Dash's global registry.
    monkeypatch.setattr(viewer, 'clientside_callback', Mock())
    monkeypatch.setattr(auto_ui, 'clientside_callback', Mock())
    return ViewerBuilder(width=640, height=480)


class TestViewerBuilder:
    def test_camera_and_build(self):
        camera = default_camera(100)
        builder = ViewerBuilder(camera, width=640, height=480)
        camera.width = 200
        first, second = builder.build(), builder.build()
        assert (builder.width, builder.height) == (640, 480)
        assert isinstance(first, KaolinViewer)
        assert first.id != second.id
        assert first.camera_parameters['intrinsics']['width'] == 640
        assert first.camera_parameters['intrinsics']['height'] == 480
        assert first.resize_mode == 'fixed'
        assert 'resize_hooks' not in first.to_plotly_json()['props']

    def test_layers_and_behaviors(self, builder):
        layer = builder.add_layer('canvas', 'paint', {'extraClasses': 'overlay'})
        behavior = builder.add_behavior('drawing', 'brush', 'paint', {'color': '#123456'})
        first, second = builder.build(), builder.build()
        assert first.layers[0][:2] == (layer, 'canvas')
        assert first.layers[0][2]['extraClasses'] == 'overlay'
        assert first.layers[0][2]['id'] == first.resolve_id(layer)
        assert first.resolve_id(layer) != second.resolve_id(layer)
        assert first.behaviors == [(behavior, 'drawing', layer, {'color': '#123456'}, True)]
        with pytest.raises(RuntimeError, match='already used'):
            builder.add_layer(identifier='paint')
        with pytest.raises(ValueError, match='not found'):
            builder.add_behavior('drawing', active_layer_id='missing')

    def test_behavior_semantics(self, caplog):
        assert ViewerBuilder.check_behavior_semantics('drawing', {'color': '#123456'})
        assert not ViewerBuilder.check_behavior_semantics('drawing', {'unknown': 1})
        assert not ViewerBuilder.check_behavior_semantics('not_registered')
        assert 'unexpected options' in caplog.text
        assert 'does not know' in caplog.text

    def test_user_behavior_options(self, builder):
        behavior = builder.add_behavior('drawing', options={'color': '#123456'})
        specs = builder.user_behavior_options_spec(behavior)
        assert {s.name for s in specs} == {'color', 'mode', 'thickness'}
        selected = builder.user_behavior_options_spec(behavior, [
            'color', OptionSpec('thickness', OptionKind.FLOAT, default=4)])
        assert [(s.name, s.default) for s in selected] == [('color', '#123456'), ('thickness', 4)]
        components = builder.add_user_behavior_options(behavior, ['color'])
        assert len(components) == 2
        control, trigger = components[0].children[-1], components[1]
        assert isinstance(control, dcc.Input)
        assert control.value == '#123456'
        assert control.persistence_type == 'session'
        assert trigger.style == {'display': 'none'}
        built = builder.build()
        js, output, input_ = auto_ui.clientside_callback.call_args.args
        assert behavior in js and built.id in js and '"color"' in js
        assert (output.component_id, input_.component_id) == (trigger.id, control.id)
        assert builder.add_user_behavior_options(behavior, []) == []
        with pytest.raises(KeyError):
            builder.user_behavior_options_spec('missing')

    def test_camera_and_remote_rendering(self, builder):
        layer = builder.add_layer()
        controller = builder.add_camera_controller(options={'enablePan': False})
        builder.add_websocket_connection('/websocket', 'remote')
        draw, camera, render = builder.add_remote_rendering(
            layer, 'remote', cam_update_period=50, render_update_period=250, tag_suffix='_left')
        builder.add_camera_listener(draw, setter='setNativeCamera', use_raw_object=True)
        built = builder.build()
        behaviors = {b[0]: b for b in built.behaviors}
        assert built.websocket_addresses == [('/websocket', 'remote')]
        assert built.camera_controllers == [controller]
        assert behaviors[controller][2] == 'EVENT_CANVAS'
        assert behaviors[controller][3] == {'enablePan': False}
        assert behaviors[draw][2:4] == (layer, {'messageTag': 'render_left'})
        assert behaviors[camera][3] == {
            'connectionId': 'remote', 'minUpdateMs': 50, 'messageTag': 'set_camera_left'}
        assert behaviors[render][3]['minUpdateMs'] == 250
        assert built.camera_listeners == [
            (camera, 'setCamera', False), (render, 'setValue', False), (draw, 'setNativeCamera', True)]
        assert (built.remote_fps_source, built.remote_fps_label) == ('render_left', '(target 4.0)')
        builder.show_remote_fps('custom', 'frames')
        assert builder.build().remote_fps_source == 'custom'

    def test_modes(self, builder):
        layer = builder.add_layer(identifier='paint')
        drawing = builder.add_behavior('drawing', 'brush', layer)
        camera = builder.add_camera_controller()
        assert builder.add_mode('draw', ['brush'], ui_icon='bi-pencil', description='Draw',
                                clear_layers=['paint'], reset_behaviors=['brush']) == 'draw'
        builder.add_mode('view', [camera])
        builder.set_active_mode('view')
        built = builder.build()
        assert built.modes[0] == ('draw', [drawing], 'bi-pencil', 'Draw')
        assert {b[0]: b[4] for b in built.behaviors} == {drawing: False, camera: True}
        assert viewer.clientside_callback.call_count == 1
        with pytest.raises(ValueError, match='already registered'):
            builder.add_mode('draw', [])
        with pytest.raises(ValueError, match='not registered'):
            builder.set_active_mode('missing')

    def test_broadcast_options(self, builder):
        source = builder.add_behavior('drawing', 'source')
        target = builder.add_behavior('drawing', 'target')
        builder.broadcast_behavior_option('color', 'source', 'target', 'color')
        builder.broadcast_behavior_option('*')
        assert builder.build().broadcast_behavior_options == [
            ('color', source, [target], ['color']), ('*', None, None, None)]
        for args in [('bad-name',), ('*', None, target, 'color'),
                     ('color', None, None, 'color'), ('color', None, target, [])]:
            with pytest.raises(ValueError):
                builder.broadcast_behavior_option(*args)

    @pytest.mark.parametrize('mode', ['fixed', 'adaptive', 'fixed_aspect'])
    def test_resize_mode(self, builder, mode):
        layer = builder.add_layer(identifier='paint')
        behavior = builder.add_behavior('drawing', 'brush', layer)
        builder.add_mode('draw', [behavior])
        builder.set_resize_mode(mode, ['paint'], ['brush'], 'draw')
        hooks = {'clear_layers': [layer], 'reset_behaviors': [behavior], 'switch_mode': 'draw'}
        for built in (builder.build(), builder.build()):
            assert built.resize_mode == mode
            assert built.resize_hooks == hooks
            assert built.resolve_id(layer) == built.layers[0][2]['id']
        builder.set_resize_mode('fixed')
        assert builder.build().resize_hooks == {
            'clear_layers': [], 'reset_behaviors': [], 'switch_mode': ''}

    def test_invalid_resize_configuration(self, builder):
        layer = builder.add_layer(identifier='paint')
        svg = builder.add_layer('svg')
        behavior = builder.add_behavior('drawing', 'brush', layer)
        builder.add_mode('draw', [behavior])
        builder.set_resize_mode('adaptive', [layer], [behavior], 'draw')
        original = builder.build().resize_hooks.copy()
        invalid = [dict(viewport_resize_mode=mode) for mode in ('unknown', '', None)]
        invalid += [dict(clear_layers=['missing']), dict(clear_layers=[svg]),
                    dict(reset_behaviors=['missing']), dict(switch_mode='missing')]
        for options in invalid:
            with pytest.raises(ValueError):
                builder.set_resize_mode(**{'viewport_resize_mode': 'fixed', **options})
            assert builder.build().resize_mode == 'adaptive'
            assert builder.build().resize_hooks == original
        with pytest.raises(TypeError):
            ViewerBuilder(viewport_resize_mode='adaptive')

    @pytest.mark.parametrize('width,height,warns', [
        (100, 98, False), (98, 100, False), (102, 100, True), (100, 102, True),
        (641, 479, True), (0, 100, True), (100, 0, True), (-1, 100, True)])
    def test_fixed_aspect_warning(self, caplog, width, height, warns):
        builder = ViewerBuilder(width=width, height=height)
        with caplog.at_level(logging.WARNING, logger=viewer.__name__):
            builder.set_resize_mode('fixed_aspect')
        assert bool(caplog.records) is warns
        caplog.clear()
        builder.set_resize_mode('fixed')
        builder.set_resize_mode('adaptive')
        assert not caplog.records


class TestKaolinViewer:
    def test_resolve_id(self):
        viewer = KaolinViewer(id='left')
        assert viewer.resolve_id('<VIEWER_ID>-paint') == 'left-paint'
        assert viewer.resolve_id('custom') == 'custom'
        assert ViewerBuilder.resolve_id_for_viewer(7, 'left') == 7
        with pytest.raises(AssertionError, match='ID cannot be None'):
            KaolinViewer()
