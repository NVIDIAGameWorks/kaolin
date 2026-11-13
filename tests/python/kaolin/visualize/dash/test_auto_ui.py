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

"""Tests for kaolin.visualize.dash.auto_ui."""

import enum
from dataclasses import dataclass, field
from typing import Annotated, Literal
from unittest.mock import Mock

import dash_daq as daq
import pytest
from dash import dcc, html

from kaolin.visualize.dash import auto_ui
from kaolin.visualize.dash.option import OptionKind, OptionSpec


class Mode(enum.Enum):
    DRAW = 'draw'
    ERASE = 'erase'


@dataclass
class Brush:
    size: int = 3


@dataclass
class Settings:
    enabled: bool = True
    mode: Mode = Mode.DRAW
    brush: Brush = field(default_factory=Brush)


def _control(components):
    return components[0].children[-1]


class TestMakeControl:
    @pytest.mark.parametrize('kind,default,control_type,value_prop,expected', [
        (OptionKind.BOOL, True, daq.BooleanSwitch, 'on', True),
        (OptionKind.ENUM, 'draw', dcc.Dropdown, 'value', 'draw'),
        (OptionKind.INT, 3, dcc.Input, 'value', 3),
        (OptionKind.FLOAT, 0.5, dcc.Input, 'value', 0.5),
        (OptionKind.STRING, None, dcc.Input, 'value', ''),
        (OptionKind.COLOR, None, dcc.Input, 'value', '#000000'),
        (OptionKind.ANY, {'a': 1}, dcc.Input, 'value', "{'a': 1}")])
    def test_scalar_controls(self, kind, default, control_type, value_prop, expected):
        spec = OptionSpec('value', kind, default=default,
                          values=['draw', 'erase'] if kind == OptionKind.ENUM else None,
                          description='Choose a value')
        components, fields = auto_ui.make_control(spec, 'settings', persistence_type='session')
        control = _control(components)
        assert isinstance(control, control_type)
        assert getattr(control, value_prop) == expected
        assert control.persistence and control.persistence_type == 'session'
        assert components[0].title == 'Choose a value'
        assert len(fields) == 1
        assert (fields[0].name, fields[0].control_id, fields[0].value_prop) == (
            'value', control.id, value_prop)
        assert fields[0].json_to_py is spec.json_to_py
        if kind == OptionKind.ENUM:
            assert control.options == [{'label': x, 'value': x} for x in ('draw', 'erase')]
        elif isinstance(control, dcc.Input):
            expected_type = 'number' if kind in (OptionKind.INT, OptionKind.FLOAT) else (
                'color' if kind == OptionKind.COLOR else 'text')
            assert control.type == expected_type

    @pytest.mark.parametrize('kind,default,step,value', [
        (OptionKind.INT, None, None, 5), (OptionKind.FLOAT, None, None, 5.0),
        (OptionKind.FLOAT, 2.5, 0.5, 2.5)])
    def test_slider(self, kind, default, step, value):
        spec = OptionSpec('size', kind, default=default, min=0, max=10, step=step)
        components, _ = auto_ui.make_control(spec, 'brush')
        slider = _control(components)
        assert isinstance(slider, dcc.Slider)
        assert (slider.min, slider.max, slider.value) == (0, 10, value)
        assert slider.step == (step if step is not None else (1 if kind == OptionKind.INT else 0.1))

    def test_group_order_and_unique_ids(self):
        tip = OptionSpec('tip', OptionKind.GROUP)
        tip.add_child(OptionSpec('soft', OptionKind.BOOL))
        group = OptionSpec('brush', OptionKind.GROUP)
        group.add_child(OptionSpec('size', OptionKind.INT))
        group.add_child(tip)
        specs = [OptionSpec('name', OptionKind.STRING), group]
        controls, fields = auto_ui.make_controls(specs, 'settings')
        assert len(controls) == 2 and isinstance(controls[1], html.Div)
        assert [f.name for f in fields] == ['name', 'brush.size', 'brush.tip.soft']
        assert len({f.control_id for f in fields}) == 3
        _, repeated = auto_ui.make_controls(specs, 'settings')
        assert not {f.control_id for f in fields} & {f.control_id for f in repeated}


class TestControlsFromDataclass:
    def test_controls_and_store(self):
        settings = Settings(brush=Brush(7))
        controls, fields, store = auto_ui.controls_from_dataclass(settings, 'settings')
        assert len(controls) == 3 and store is None
        assert [f.name for f in fields] == ['enabled', 'mode', 'brush.size']
        assert fields[1].json_to_py('erase') is Mode.ERASE
        _, fields, store = auto_ui.controls_from_dataclass(
            settings, 'persisted', make_store=True, storage_type='local')
        assert isinstance(store, dcc.Store) and store.storage_type == 'local'
        assert store.data == {'enabled': True, 'mode': 'draw', 'brush.size': 7}
        restored = Settings()
        auto_ui.apply_store_to_dataclass(store.data, restored, fields)
        assert restored == settings
        auto_ui.apply_store_to_dataclass({'mode': 'erase', 'brush.size': '9'}, restored, fields)
        assert restored == Settings(mode=Mode.ERASE, brush=Brush(9))
        auto_ui.apply_store_value_to_dataclass({'mode': None}, restored, fields[1])
        assert restored.mode is None
        auto_ui.apply_store_to_dataclass(None, restored, fields)
        assert restored.brush.size == 9
        _, _, class_store = auto_ui.controls_from_dataclass(Settings, 'defaults', make_store=True)
        assert class_store.data['brush.size'] == 3

    def test_names_and_annotations(self):
        controls, fields, _ = auto_ui.controls_from_dataclass(
            Brush(), 'brush', names=['size'],
            annotations={'size': Annotated[int, {'min': 1, 'max': 10}]})
        assert [f.name for f in fields] == ['size']
        assert isinstance(_control(controls), dcc.Slider)
        with pytest.raises(ValueError):
            auto_ui.controls_from_dataclass(Brush(), 'brush', names=['missing'])


class TestControlsFromFunction:
    def test_parameters(self):
        def configure(size: int, mode: Literal['draw', 'erase'] = 'draw', *args, **kwargs):
            pass

        controls, fields = auto_ui.controls_from_function(configure, 'configure')
        assert [f.name for f in fields] == ['size', 'mode']
        assert controls[0].children[-1].value is None
        assert controls[1].children[-1].value == 'draw'
        assert fields[0].json_to_py('5') == 5


def test_bind_controls_to_behavior_clientside(monkeypatch):
    register = Mock()
    monkeypatch.setattr(auto_ui, 'clientside_callback', register)
    fields = [auto_ui.FieldSpec('color', 'color-control', 'value'),
              auto_ui.FieldSpec('enabled', 'enabled-control', 'on')]
    triggers = [html.Div(id='color-trigger'), html.Div(id='enabled-trigger')]
    auto_ui.bind_controls_to_behavior_clientside(fields, 'brush', 'viewer', triggers)
    assert register.call_count == 2
    for call, spec, trigger in zip(register.call_args_list, fields, triggers):
        js, output, input_ = call.args
        assert 'requestBehaviorSetOption' in js
        assert f'"brush", "{spec.name}", value, "viewer"' in js
        assert (output.component_id, output.component_property) == (trigger.id, 'children')
        assert (input_.component_id, input_.component_property) == (spec.control_id, spec.value_prop)
        assert call.kwargs == {'prevent_initial_call': False}
    with pytest.raises(AssertionError):
        auto_ui.bind_controls_to_behavior_clientside(fields, 'brush', 'viewer', [])


def test_make_icon_buttons():
    assert auto_ui.make_icon_buttons([], []) == []
    buttons, tooltips = auto_ui.make_icon_buttons(['bi-pencil', 'bi-eye'], ['Draw', 'View'], active_idx=1)
    assert [button.active for button in buttons] == [False, True]
    assert [button.outline for button in buttons] == [True, False]
    assert [tooltip.target for tooltip in tooltips] == [button.id for button in buttons]
    assert [tooltip.children for tooltip in tooltips] == ['Draw', 'View']
