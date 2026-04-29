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

"""Auto-generation of Dash controls from typed sources.

Controls are built from :class:`~kaolin.visualize.dash.option.OptionSpec` objects,
which can be obtained from several sources:

* a dataclass *instance* via :func:`controls_from_dataclass` (defaults are the
  instance's current attribute values);
* a callable's signature via :func:`controls_from_function` (defaults are the
  parameter defaults, or ``None``);
* directly, e.g. from a behavior manifest entry, then rendered with
  :func:`make_control` / :func:`make_controls`.

Generated controls have a deterministic id of the form
``f"{id_prefix}-{field_name}"`` so they can be wired to a :class:`dash.dcc.Store`
or to callbacks without manual id juggling.

:func:`controls_from_dataclass` can also build a backing :class:`dash.dcc.Store`
seeded from the instance (pass ``make_store=True``); :func:`apply_store_to_dataclass`
restores store data back onto a typed instance.
:func:`bind_controls_to_behavior_clientside` pipes control changes to a JS
``InteractiveBehavior`` via ``kaolin.core.event.requestBehaviorSetOption``.
"""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import dash_daq as daq
import dash_bootstrap_components as dbc
from dash import Input, Output, clientside_callback, dcc, html
from dash.development.base_component import Component

from kaolin.visualize.web.naming import UniqueIdGenerator

from kaolin.visualize.dash.option import (
    OptionKind, OptionSpec, specs_from_dataclass, specs_from_function)

logger = logging.getLogger(__name__)

__all__ = [
    'FieldSpec',
    'make_control',
    'make_controls',
    'controls_from_dataclass',
    'controls_from_function',
    'bind_controls_to_behavior_clientside',
    'apply_store_to_dataclass',
    'apply_store_value_to_dataclass',
]

_NVIDIA_GREEN = '#76b900'


# NEW VERSION ------------------------------------------

@dataclass(frozen=True)
class FieldSpec:
    """Describes how a generated control wires into a :class:`dash.dcc.Store`
    (or any other sink).

    One :class:`FieldSpec` is produced alongside every control. The
    :class:`FieldSpec` tells a binder (a) which Dash prop carries the value, and
    (b) how to convert the JSON-friendly value back into the original Python
    type when reconstructing an underlying instance on the server.

    Attributes:
        name: original field/parameter/option name as written in the source
            (e.g. ``selection_brush_radius``). Used as the dict key inside the
            store. For fields produced by recursing into a nested dataclass,
            ``name`` is a dotted path such as ``"outer.inner"``.
        control_id: deterministic Dash component id of the generated control
            (e.g. ``"usr-selection_brush_radius"``); use as
            ``Input(control_id, value_prop)`` or ``getElementById(control_id)``.
        value_prop: name of the Dash prop that carries the field's current
            value. Differs by control type: ``"on"`` for
            :class:`dash_daq.BooleanSwitch`, ``"value"`` for :class:`dcc.Slider`,
            :class:`dcc.Dropdown`, :class:`dcc.Input`.
        json_to_py: ``Callable[[Any], Any]`` that converts the raw JSON-friendly
            value emitted by the control (``str`` / ``int`` / ``float`` /
            ``bool``) back into the field's Python type. Sourced from
            :attr:`OptionSpec.json_to_py` (the type itself for primitives, the
            ``Enum`` subclass for enums, identity otherwise). Defaults to the
            identity.
    """
    name: str
    control_id: str
    value_prop: str
    json_to_py: Callable[[Any], Any] = field(default=lambda x: x)


def make_control(spec: OptionSpec,
                 id_prefix: str,
                 persistence_type: str | None = None,
                 name_prefix: str = '') -> tuple[list[Component], list[FieldSpec]]:
    """Render one :class:`OptionSpec` into Dash components plus matching field specs.

    The spec's :attr:`OptionKind` selects the control: ``BOOL`` ->
    :class:`dash_daq.BooleanSwitch`, ``ENUM`` -> :class:`dash.dcc.Dropdown`,
    ``INT`` / ``FLOAT`` with both bounds -> :class:`dash.dcc.Slider` (otherwise a
    numeric ``dcc.Input``), ``STRING`` -> text input, ``COLOR`` -> native color
    input, ``GROUP`` -> a recursively rendered section, and ``ANY`` (or anything
    unexpected) -> a text-input fallback. Each generated control gets a
    deterministic id ``f"{id_prefix}-{name_prefix}{spec.name}"``.

    Args:
        spec (OptionSpec): the option to render.
        id_prefix (str): prefix for the generated component id.
        persistence_type (Optional[str]): if set, enables Dash persistence on the
            control with this ``persistence_type`` (e.g. ``'session'``).
        name_prefix (str): dotted prefix prepended to the spec name for nested
            (group) children, so field-spec names read like ``"outer.inner"``.

    Return:
        (tuple) ``(components, field_specs)`` -- a one-element component list (a
        wrapped control, or a section ``Div`` for a group) and one
        :class:`FieldSpec` per rendered leaf control.
    """
    full_name = f'{name_prefix}{spec.name}'
    component_id = UniqueIdGenerator.get_unique_id(f'{id_prefix}-{full_name}')
    label_text = spec.name
    json_to_py = spec.json_to_py

    common_kwargs: dict[str, Any] = {'id': component_id}
    if persistence_type is not None:
        common_kwargs['persistence_type'] = persistence_type
        common_kwargs['persistence'] = True

    def _leaf(component: Component, value_prop: str, **wrap_kwargs) -> tuple[list[Component], list[FieldSpec]]:
        ctrl = _wrap(component, **wrap_kwargs)
        return [ctrl], [FieldSpec(name=full_name, control_id=component_id,
                                  value_prop=value_prop, json_to_py=json_to_py)]

    if spec.kind == OptionKind.GROUP:
        nested_controls: list[Component] = []
        nested_specs: list[FieldSpec] = []
        for child in spec.children or []:
            cs, ss = make_control(child, id_prefix, persistence_type=persistence_type,
                                  name_prefix=f'{full_name}.')
            nested_controls.extend(cs)
            nested_specs.extend(ss)
        heading = html.H6(spec.name, className='text-uppercase text-muted small mt-2')
        section = html.Div(children=[heading, *nested_controls], className='kaolin-auto-ui-section')
        return [section], nested_specs

    if spec.kind == OptionKind.BOOL:
        return _leaf(daq.BooleanSwitch(**common_kwargs,
                                       on=bool(spec.default) if spec.default is not None else False,
                                       color=_NVIDIA_GREEN, label=label_text),
                     'on', tooltip=spec.description)

    if spec.kind == OptionKind.ENUM:
        options = [{'label': str(v), 'value': v} for v in (spec.values or [])]
        return _leaf(dcc.Dropdown(**common_kwargs, options=options, value=spec.default,
                                  clearable=False, searchable=False),
                     'value', label=label_text, target_id=component_id, tooltip=spec.description)

    if spec.kind in (OptionKind.INT, OptionKind.FLOAT):
        rng = _numeric_range_from_spec(spec)
        if rng is not None:
            return _leaf(dcc.Slider(**common_kwargs, min=rng.min, max=rng.max, step=rng.step,
                                    value=_slider_value_from_spec(spec, rng), marks=None,
                                    tooltip={'placement': 'bottom', 'always_visible': False,
                                             'template': spec.description}),
                         'value', label=label_text, target_id=component_id)
        return _leaf(dcc.Input(**common_kwargs, type='number', value=spec.default),
                     'value', label=label_text, target_id=component_id, tooltip=spec.description)

    if spec.kind == OptionKind.STRING:
        return _leaf(dcc.Input(**common_kwargs, type='text',
                               value='' if spec.default is None else spec.default),
                     'value', label=label_text, target_id=component_id, tooltip=spec.description)

    if spec.kind == OptionKind.COLOR:
        # dcc.Input(type='color') renders the native HTML5 color picker; the value
        # comes back as a hex string, exactly what an OptionKind.COLOR expects.
        return _leaf(dcc.Input(**common_kwargs, type='color',
                               value=spec.default if spec.default is not None else '#000000'),
                     'value', label=label_text, target_id=component_id, tooltip=spec.description)

    # OptionKind.ANY (and any unexpected kind): text-input fallback.
    return _leaf(dcc.Input(**common_kwargs, type='text',
                           value='' if spec.default is None else str(spec.default)),
                 'value', label=label_text, target_id=component_id, tooltip=spec.description)


def make_controls(specs: Iterable[OptionSpec],
                  id_prefix: str,
                  persistence_type: str | None = None) -> tuple[list[Component], list[FieldSpec]]:
    """Render a list of :class:`OptionSpec` objects into Dash components + field specs.

    The single rendering pass shared by the OptionSpec ingestion entry points;
    each spec is delegated to :func:`make_control`.

    Args:
        specs (Iterable[OptionSpec]): the options to render, in order.
        id_prefix (str): prefix applied to every generated component id.
        persistence_type (Optional[str]): if set, enables Dash persistence on each
            control with this ``persistence_type`` (e.g. ``'session'``).

    Return:
        (tuple) ``(components, field_specs)`` -- two lists; a ``GROUP`` spec
        contributes a single section ``Div`` but one :class:`FieldSpec` per leaf
        (with dotted names).
    """
    controls: list[Component] = []
    specs_out: list[FieldSpec] = []
    for spec in specs:
        c, s = make_control(spec, id_prefix, persistence_type=persistence_type)
        controls.extend(c)
        specs_out.extend(s)
    return controls, specs_out


def controls_from_dataclass(instance: Any, id_prefix: str,
                            names: Iterable[str] | None = None,
                            annotations: dict[str, Any] | None = None,
                            make_store: bool = False,
                            storage_type: str = 'session',
                            ) -> tuple[list[Component], list[FieldSpec], dcc.Store | None]:
    """Build Dash controls + field specs (and optionally a backing store) for a dataclass.

    Derives an :class:`OptionSpec` list via
    :func:`kaolin.visualize.dash.option.specs_from_dataclass` and renders it with
    :func:`make_controls`. When ``make_store`` is set, also returns a
    :class:`dash.dcc.Store` seeded with the instance's current values (keyed to
    match the field specs, with dotted keys for nested-dataclass fields).

    Args:
        instance (Any): a dataclass instance (or a class instantiable with
            defaults).
        id_prefix (str): prefix for generated component ids and the store id.
        names (Optional[Iterable[str]]): field names to include and order;
            defaults to all fields in declaration order.
        annotations (Optional[dict]): ``{field_name: annotation}`` overrides.
        make_store (bool): if True, build and return a backing ``dcc.Store``.
        storage_type (str): ``dcc.Store`` storage type when ``make_store`` is set.

    Return:
        (tuple) ``(controls, field_specs, store)`` where ``store`` is ``None``
        unless ``make_store`` is True.
    """
    specs = specs_from_dataclass(instance, names=names, annotations=annotations)
    controls, field_specs = make_controls(specs, id_prefix)
    store = None
    if make_store:
        if isinstance(instance, type):
            instance = instance()
        store = dcc.Store(id=UniqueIdGenerator.get_unique_id(id_prefix),
                          data=_make_default_store_data(instance, field_specs),
                          storage_type=storage_type)
    return controls, field_specs, store


def controls_from_function(fnc: Callable, id_prefix: str,
                           names: Iterable[str] | None = None,
                           annotations: dict[str, Any] | None = None,
                           ) -> tuple[list[Component], list[FieldSpec]]:
    """Build Dash controls + field specs for a callable's parameters.

    Behaves like :func:`controls_from_dataclass` but introspects a function
    signature via :func:`kaolin.visualize.dash.option.specs_from_function`
    (``*args`` / ``**kwargs`` are skipped, parameters without a default map to
    ``default=None``). No backing store is produced, as there is no instance to
    persist.

    Args:
        fnc (Callable): callable to introspect (bound methods are supported).
        id_prefix (str): prefix for generated component ids.
        names (Optional[Iterable[str]]): parameter names to include and order.
        annotations (Optional[dict]): ``{param_name: annotation}`` overrides.

    Return:
        (tuple) ``(controls, field_specs)``.
    """
    specs = specs_from_function(fnc, names=names, annotations=annotations)
    return make_controls(specs, id_prefix)


def _wrap(component: Component,
          label: str | None = None,
          target_id: str | None = None,
          tooltip: str | None = None) -> Component:
    """Wrap a control in a Div, with an optional html.Label tied to the control via htmlFor.

    ``tooltip`` is rendered as the wrapping ``Div``'s ``title`` attribute and
    surfaces as a native browser tooltip on hover. It is independent of
    ``label`` -- a control may have only a tooltip (e.g. a boolean switch with
    a built-in label) or both.
    """
    children: list[Component] = []
    if label is not None:
        children.append(html.Label(label, htmlFor=target_id, className='form-label small'))
    children.append(component)
    div_kwargs: dict[str, Any] = {'className': 'kaolin-auto-ui-field mb-3'}
    if tooltip:
        div_kwargs['title'] = tooltip
    return html.Div(children=children, **div_kwargs)


@dataclass(frozen=True)
class NumericRange:
    """A resolved slider range (bounds + concrete step) for a numeric option.

    Internal helper type produced by :func:`_numeric_range_from_spec`. Unlike
    :class:`~kaolin.visualize.dash.option.OptionSpec` (whose ``step`` is
    optional), this always carries a concrete ``step`` so the renderer does not
    have to re-derive one.
    """
    min: float
    max: float
    step: float


def _numeric_range_from_spec(spec: OptionSpec) -> NumericRange | None:
    """Best-effort slider range for an ``INT`` / ``FLOAT`` :class:`OptionSpec`.

    Returns ``None`` when the spec lacks both bounds (the caller then falls back
    to a free numeric input, since a slider needs a finite range). When both
    bounds are present, the step is taken from the spec or guessed: ``1`` for
    ``INT`` and ``(max - min) / 100`` for ``FLOAT``.

    Args:
        spec (OptionSpec): an ``INT`` or ``FLOAT`` option.

    Return:
        (Optional[NumericRange]) the resolved range, or ``None`` if not sliderable.
    """
    if spec.min is None or spec.max is None:
        return None
    is_int = spec.kind == OptionKind.INT
    step = spec.step if spec.step is not None else (1 if is_int else (spec.max - spec.min) / 100.0)
    return NumericRange(min=spec.min, max=spec.max, step=step)


def _slider_value_from_spec(spec: OptionSpec, rng: NumericRange) -> float | int:
    """Initial slider value for ``spec``: its default, or the range midpoint as a
    guess when no default is given (coerced to ``int`` for ``INT`` options).
    """
    value = spec.default if spec.default is not None else (rng.min + rng.max) / 2
    return int(value) if spec.kind == OptionKind.INT else value


# ----------------------------------------------------------------------------------------
# Wiring helpers: generated controls <-> dcc.Store <-> typed dataclass instance
# ----------------------------------------------------------------------------------------

def _make_default_store_data(instance: Any, field_specs: list[FieldSpec]) -> dict:
    """Build a flat, JSON-friendly dict suitable for ``dcc.Store(data=...)``.

    The dict's keys are ``spec.name`` for every spec (with dotted keys for
    nested-dataclass fields), matching what :func:`apply_store_to_dataclass`
    reads back. :class:`enum.Enum` defaults are unwrapped to their ``.value`` so
    the dict is JSON-serializable.
    """
    result: dict[str, Any] = {}
    for spec in field_specs:
        cur: Any = instance
        for part in spec.name.split('.'):
            cur = getattr(cur, part)
        if isinstance(cur, enum.Enum):
            cur = cur.value
        result[spec.name] = cur
    return result


def apply_store_to_dataclass(store_data: dict, instance: Any,
                             field_specs: list[FieldSpec]) -> None:
    """Mutate ``instance`` (and any nested dataclasses it owns) in place from store data.

    Inverse of :func:`_make_default_store_data`. For every spec, reads
    ``store_data[spec.name]``, applies ``spec.json_to_py``, and writes it onto the
    appropriate nested attribute. Missing keys are skipped silently; ``None``
    values are written through without casting.
    """
    if store_data is None:
        return
    for spec in field_specs:
        apply_store_value_to_dataclass(store_data, instance, spec)


def apply_store_value_to_dataclass(store_data: dict, instance: Any, spec: FieldSpec) -> None:
    if store_data is None or spec.name not in store_data:
        return
    raw = store_data[spec.name]
    value = spec.json_to_py(raw) if raw is not None else None
    cur = instance
    parts = spec.name.split('.')
    for part in parts[:-1]:
        cur = getattr(cur, part)
    setattr(cur, parts[-1], value)


# ----------------------------------------------------------------------------------------
# Wiring helper: generated controls -> JS InteractiveBehavior (via requestBehaviorSetOption)
# ----------------------------------------------------------------------------------------

def bind_controls_to_behavior_clientside(
        field_specs: list[FieldSpec],
        behavior_id: str,
        viewer_id: str,
        triggers: list[Component]) -> None:
    """Pipe generated controls to a JS ``InteractiveBehavior`` on every change.

    Requires a dummy trigger component.
    """
    assert len(field_specs) == len(triggers)

    for spec, trigger in zip(field_specs, triggers):
        clientside_callback(
            f'function(value) {{\n'
            f'    if (kaolin && kaolin.core && kaolin.core.event &&\n'
            f'        typeof kaolin.core.event.requestBehaviorSetOption === "function") {{\n'
            f'        kaolin.core.event.requestBehaviorSetOption(\n'
            f'            {json.dumps(behavior_id)}, {json.dumps(spec.name)}, value, {json.dumps(viewer_id)});\n'
            f'    }} else {{\n'
            f'        console.warn("kaolin.core.event.requestBehaviorSetOption is not loaded; "\n'
            f'                     + "cannot deliver option " + {json.dumps(spec.name)} + " to behavior "\n'
            f'                     + {json.dumps(behavior_id)});\n'
            f'    }}\n'
            f'    return window.dash_clientside.no_update;\n'
            f'}}',
            Output(trigger.id, 'children'),
            Input(spec.control_id, spec.value_prop),
            prevent_initial_call=False
        )


def make_icon_buttons(icons, descriptions, active_idx=0, color='secondary'):
    if len(icons) == 0:
        return []

    buttons = []
    tooltips = []
    for i, (icon, label) in enumerate(zip(icons, descriptions)):
        btn_id = UniqueIdGenerator.get_unique_id(icon)
        is_active = (i == active_idx)
        btn = dbc.Button(
            html.I(className=f'bi {icon}'),
            id=btn_id, color=color, outline=not is_active, active=is_active)
        buttons.append(btn)
        tooltips.append(dbc.Tooltip(label, target=btn_id))
    return buttons, tooltips
