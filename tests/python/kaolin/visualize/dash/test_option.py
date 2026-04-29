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

"""Tests for kaolin.visualize.dash.option."""

import enum
from dataclasses import dataclass, field
from typing import Annotated, Literal

import pytest

from kaolin.utils.testing import contained_torch_equal
from kaolin.visualize.dash.option import (
    OptionKind, OptionSpec, UIBound, specs_from_dataclass, specs_from_function)


def _check_dict(spec, expected, label):
    """Assert ``spec.as_dict()`` equals ``expected``."""
    actual = spec.as_dict()
    assert contained_torch_equal(actual, expected, approximate=True), \
        f'{label}: as_dict() = {actual!r}, expected {expected!r}'


def _check_round_trip(spec, expected, label):
    """Assert constructor, ``as_dict``, and ``from_dict`` all agree on ``expected``."""
    _check_dict(spec, expected, f'{label} constructor')
    _check_dict(OptionSpec.from_dict(expected), expected, f'{label} from_dict')
    _check_dict(OptionSpec.from_dict(spec.as_dict()), expected, f'{label} round-trip')


def _check_no_children(spec, label):
    """Assert a non-group spec rejects ``add_child``."""
    assert spec.children is None, f'{label}: non-group spec has no children list'
    with pytest.raises(ValueError, match='non-group'):
        spec.add_child(OptionSpec(name='child', kind=OptionKind.STRING))


def test_option_kind_values():
    cases = [(OptionKind.INT, 'int'), (OptionKind.FLOAT, 'float'), (OptionKind.STRING, 'string'),
             (OptionKind.COLOR, 'color'), (OptionKind.BOOL, 'bool'), (OptionKind.ENUM, 'enum'),
             (OptionKind.GROUP, 'group'), (OptionKind.ANY, 'any')]
    for member, value in cases:
        assert member == value, f'{member!r} should compare equal to its string value {value!r}'
        assert OptionKind(value) is member, f'OptionKind({value!r}) should resolve to {member!r}'


class TestOptionSpec:
    """Tests for kaolin.visualize.dash.option.OptionSpec."""

    def test_valid_int(self):
        expected = {'name': 'r', 'kind': 'int', 'default': 5, 'description': 'radius',
                    'min': 0, 'max': 10, 'step': 2, 'uiBound': True}
        spec = OptionSpec(name='r', kind=OptionKind.INT, default=5, description='radius',
                          min=0, max=10, step=2, ui_bound=True)
        _check_round_trip(spec, expected, 'int')

        _check_dict(OptionSpec.from_annotated_field('r', int, default=5),
                    {'name': 'r', 'kind': 'int', 'default': 5}, 'int bare annotation')
        _check_dict(OptionSpec.from_annotated_field('r', Annotated[int, {'min': 0, 'max': 10, 'step': 2}], default=5),
                    {'name': 'r', 'kind': 'int', 'default': 5, 'min': 0, 'max': 10, 'step': 2},
                    'int annotated bounds')

        with pytest.raises(ValueError, match='min'):
            OptionSpec(name='r', kind=OptionKind.INT, min=10, max=0)
        with pytest.raises(ValueError, match='int or float'):
            OptionSpec(name='r', kind=OptionKind.INT, step=True)
        with pytest.raises(ValueError, match='enum'):
            OptionSpec(name='r', kind=OptionKind.INT, values=[1, 2])
        _check_no_children(spec, 'int')

    def test_valid_float(self):
        expected = {'name': 'f', 'kind': 'float', 'default': 0.5, 'min': 0.0, 'max': 1.0, 'step': 0.1}
        spec = OptionSpec(name='f', kind=OptionKind.FLOAT, default=0.5, min=0.0, max=1.0, step=0.1)
        _check_round_trip(spec, expected, 'float')

        _check_dict(OptionSpec.from_annotated_field('f', float, default=0.5),
                    {'name': 'f', 'kind': 'float', 'default': 0.5}, 'float bare annotation')
        _check_dict(OptionSpec.from_annotated_field('f', Annotated[float, {'min': 0.0, 'max': 1.0, 'step': 0.1}],
                                                    default=0.5),
                    expected, 'float annotated bounds')

        with pytest.raises(ValueError, match='min'):
            OptionSpec(name='f', kind=OptionKind.FLOAT, min=1.0, max=0.0)
        _check_no_children(spec, 'float')

    def test_valid_string(self):
        expected = {'name': 's', 'kind': 'string', 'default': 'hi', 'uiBound': False}
        spec = OptionSpec(name='s', kind=OptionKind.STRING, default='hi', ui_bound=False)
        _check_round_trip(spec, expected, 'string')

        _check_dict(OptionSpec.from_annotated_field('s', str, default='hi'),
                    {'name': 's', 'kind': 'string', 'default': 'hi'}, 'string annotation')

        with pytest.raises(ValueError, match='numeric'):
            OptionSpec(name='s', kind=OptionKind.STRING, max=5)
        _check_no_children(spec, 'string')

    def test_valid_color(self):
        # No annotation maps to COLOR, so from_annotated_field is not exercised here.
        expected = {'name': 'c', 'kind': 'color', 'default': '#fff', 'description': 'tint'}
        spec = OptionSpec(name='c', kind=OptionKind.COLOR, default='#fff', description='tint')
        _check_round_trip(spec, expected, 'color')

        with pytest.raises(ValueError, match='numeric'):
            OptionSpec(name='c', kind=OptionKind.COLOR, min=0)
        _check_no_children(spec, 'color')

    def test_valid_bool(self):
        expected = {'name': 'b', 'kind': 'bool', 'default': True, 'uiBound': False}
        spec = OptionSpec(name='b', kind=OptionKind.BOOL, default=True, ui_bound=False)
        _check_round_trip(spec, expected, 'bool')

        _check_dict(OptionSpec.from_annotated_field('b', bool, default=True),
                    {'name': 'b', 'kind': 'bool', 'default': True}, 'bool annotation')

        with pytest.raises(ValueError, match='numeric'):
            OptionSpec(name='b', kind=OptionKind.BOOL, step=1)
        _check_no_children(spec, 'bool')

    def test_valid_enum(self):
        expected = {'name': 'm', 'kind': 'enum', 'default': 'a', 'values': ['a', 'b', 'c']}
        spec = OptionSpec(name='m', kind=OptionKind.ENUM, default='a', values=['a', 'b', 'c'])
        _check_round_trip(spec, expected, 'enum')

        _check_dict(OptionSpec.from_annotated_field('m', Literal['a', 'b', 'c'], default='a'),
                    expected, 'enum from Literal')

        class Mode(enum.Enum):
            A = 'a'
            B = 'b'
        from_enum = OptionSpec.from_annotated_field('m', Mode, default=Mode.A)
        _check_dict(from_enum, {'name': 'm', 'kind': 'enum', 'default': 'a', 'values': ['a', 'b']},
                    'enum from Enum subclass')
        # from_annotated_field must register a member<->value caster for a real enum.
        assert from_enum._json_to_py('b') is Mode.B, 'enum-subclass json_to_py rebuilds the member'
        assert from_enum._py_to_json(Mode.B) == 'b', 'enum-subclass py_to_json unwraps the member'
        # A Literal enum has plain JSON values, so it keeps the identity caster.
        from_literal = OptionSpec.from_annotated_field('m', Literal['a', 'b', 'c'], default='a')
        assert from_literal._json_to_py('a') == 'a' and not from_literal._custom_casters, \
            'Literal enum keeps the default identity caster'

        with pytest.raises(ValueError, match='enum'):
            OptionSpec(name='m', kind=OptionKind.ENUM)
        _check_no_children(spec, 'enum')

    def test_valid_group(self):
        grp = OptionSpec(name='g', kind=OptionKind.GROUP, description='grp')
        assert grp.children == [], 'group kind seeds an empty children list'
        grp.add_child(OptionSpec(name='r', kind=OptionKind.INT, default=1, min=0, max=5))
        sub = OptionSpec(name='sub', kind=OptionKind.GROUP)
        sub.add_child(OptionSpec(name='flag', kind=OptionKind.BOOL, default=True, ui_bound=False))
        grp.add_child(sub)

        expected = {
            'name': 'g', 'kind': 'group', 'description': 'grp',
            'children': [
                {'name': 'r', 'kind': 'int', 'default': 1, 'min': 0, 'max': 5},
                {'name': 'sub', 'kind': 'group',
                 'children': [{'name': 'flag', 'kind': 'bool', 'default': True, 'uiBound': False}]},
            ],
        }
        _check_round_trip(grp, expected, 'group')

    def test_valid_any(self):
        expected = {'name': 'x', 'kind': 'any', 'default': 'whatever'}
        spec = OptionSpec(name='x', kind=OptionKind.ANY, default='whatever')
        _check_round_trip(spec, expected, 'any')

        # Unsupported base type falls back to ANY.
        _check_dict(OptionSpec.from_annotated_field('x', dict),
                    {'name': 'x', 'kind': 'any'}, 'any from unsupported annotation')
        _check_no_children(spec, 'any')

    def test_default_validation(self):
        # Enum default must be one of values; the caster is not consulted for enums.
        with pytest.raises(ValueError, match='not one of values'):
            OptionSpec(name='m', kind=OptionKind.ENUM, default='z', values=['a', 'b'])
        # Other kinds reject a default only when the kind's caster cannot handle it.
        with pytest.raises(ValueError, match='not valid'):
            OptionSpec(name='r', kind=OptionKind.INT, default='nope')
        with pytest.raises(ValueError, match='not valid'):
            OptionSpec(name='f', kind=OptionKind.FLOAT, default='x')
        # Coercible / loose defaults pass: 1.5 for int, int for float, 0/1 for bool, anything for ANY.
        for kind, default in [(OptionKind.INT, 1.5), (OptionKind.FLOAT, 1), (OptionKind.BOOL, 0),
                              (OptionKind.BOOL, 1), (OptionKind.STRING, 5), (OptionKind.ANY, {'k': 'v'})]:
            OptionSpec(name='x', kind=kind, default=default)

    def test_ui_bound(self):
        # A UIBound marker sets ui_bound; the bare marker defaults to True.
        for marker, expected in [(UIBound(True), True), (UIBound(False), False), (UIBound(), True)]:
            spec = OptionSpec.from_annotated_field('age', Annotated[int, marker], default=1)
            assert spec.ui_bound is expected, f'UIBound({marker.is_bound!r}) -> ui_bound {spec.ui_bound!r}'

        # Both dict spellings are accepted.
        assert OptionSpec.from_annotated_field('age', Annotated[int, {'uiBound': True}]).ui_bound is True, \
            "'uiBound' dict key sets ui_bound"
        assert OptionSpec.from_annotated_field('age', Annotated[int, {'ui_bound': False}]).ui_bound is False, \
            "'ui_bound' dict key sets ui_bound"

        # Absent -> None, and the marker coexists with other metadata and surfaces in as_dict.
        assert OptionSpec.from_annotated_field('age', int).ui_bound is None, 'no marker -> ui_bound None'
        spec = OptionSpec.from_annotated_field('age', Annotated[int, {'min': 0, 'max': 9}, UIBound(False)], default=1)
        _check_dict(spec, {'name': 'age', 'kind': 'int', 'default': 1, 'min': 0, 'max': 9, 'uiBound': False},
                    'ui_bound alongside numeric bounds')

    def test_type_casters(self, caplog):
        # Default casters are seeded from the kind and coerce values across the wire.
        assert OptionSpec(name='r', kind=OptionKind.INT)._json_to_py('5') == 5, 'int seeds an int caster'
        assert OptionSpec(name='s', kind=OptionKind.STRING)._py_to_json(3) == '3', 'string seeds a str caster'

        class Mode(enum.Enum):
            A = 'a'
            B = 'b'
        spec = OptionSpec(name='m', kind=OptionKind.ENUM, default='a', values=['a', 'b'])
        assert spec._json_to_py('a') == 'a' and not spec._custom_casters, 'enum seeds an identity caster'

        spec.add_type_cast(py_to_json=lambda m: m.value, json_to_py=Mode)
        assert spec._custom_casters, 'add_type_cast marks the spec as having custom casters'
        assert spec._json_to_py('a') is Mode.A, 'custom json_to_py rebuilds the enum member'
        assert spec._py_to_json(Mode.B) == 'b', 'custom py_to_json unwraps the enum member'

        # as_dict warns about, but never serializes, custom casters.
        caplog.clear()
        with caplog.at_level('WARNING'):
            out = spec.as_dict()
        assert 'caster' in caplog.text.lower(), f'as_dict should warn about custom casters, got {caplog.text!r}'
        assert out == {'name': 'm', 'kind': 'enum', 'default': 'a', 'values': ['a', 'b']}, \
            f'casters must not leak into as_dict() output, got {out!r}'

    def test_caster_properties_and_merged(self):
        spec = OptionSpec(name='r', kind=OptionKind.INT, default=5, min=0, max=10, step=2, description='d')
        # Public caster properties mirror the seeded private casters.
        assert spec.json_to_py('7') == 7 and spec.py_to_json(7) == 7, 'int caster properties coerce via int'

        # merged() copies and overrides only the given fields, re-validating, without mutating the original.
        m = spec.merged({'default': 8, 'description': 'd2'})
        _check_dict(m, {'name': 'r', 'kind': 'int', 'default': 8, 'min': 0, 'max': 10, 'step': 2,
                        'description': 'd2'}, 'merged overrides default/description')
        assert spec.default == 5 and spec.description == 'd', 'merged leaves the original untouched'

        # uiBound alias is honored and unknown keys are rejected (by the constructor).
        assert spec.merged({'uiBound': False}).ui_bound is False, 'merged accepts the uiBound alias'
        with pytest.raises(TypeError):
            spec.merged({'nope': 1})

        # Custom casters and group children survive a merge.
        class Mode(enum.Enum):
            A = 'a'
        assert OptionSpec.from_annotated_field('m', Mode, default=Mode.A).merged({'description': 'x'}) \
            ._json_to_py('a') is Mode.A, 'merged carries custom casters'
        grp = OptionSpec(name='g', kind=OptionKind.GROUP)
        grp.add_child(OptionSpec(name='c', kind=OptionKind.INT, default=1))
        assert [c.name for c in grp.merged().children] == ['c'], 'merged copies group children'

    def test_from_dict_aliases(self):
        assert OptionSpec.from_dict({'name': 'a', 'kind': 'int', 'uiBound': False}).ui_bound is False, \
            "'uiBound' key populates ui_bound"
        assert OptionSpec.from_dict({'name': 'a', 'kind': 'int', 'ui_bound': True}).ui_bound is True, \
            "'ui_bound' key populates ui_bound"
        with pytest.raises(AssertionError):
            OptionSpec.from_dict({'kind': 'int'})
        with pytest.raises(AssertionError):
            OptionSpec.from_dict({'name': 'a'})


class TestDataclassSpecs:
    """Tests for kaolin.visualize.dash.option.specs_from_dataclass."""

    def test_basics(self):
        @dataclass
        class Sub:
            flag: bool = True
            label: str = 'x'

        @dataclass
        class Cfg:
            count: Annotated[int, {'min': 0, 'max': 9}] = 3
            mode: Literal['a', 'b'] = 'a'
            sub: Sub = field(default_factory=Sub)

        specs = specs_from_dataclass(Cfg())
        assert [s.name for s in specs] == ['count', 'mode', 'sub'], 'fields map in declaration order'
        _check_dict(specs[0], {'name': 'count', 'kind': 'int', 'default': 3, 'min': 0, 'max': 9},
                    'annotated int field')
        _check_dict(specs[1], {'name': 'mode', 'kind': 'enum', 'default': 'a', 'values': ['a', 'b']},
                    'literal field')
        # A nested dataclass becomes a GROUP whose children are the nested fields.
        _check_dict(specs[2], {'name': 'sub', 'kind': 'group',
                               'children': [{'name': 'flag', 'kind': 'bool', 'default': True},
                                            {'name': 'label', 'kind': 'string', 'default': 'x'}]},
                    'nested dataclass field')

    def test_custom_annotations(self, caplog):
        @dataclass
        class Cfg:
            count: int = 3
            label: str = 'hi'

        # An override annotation is honored over the field's declared type.
        specs = specs_from_dataclass(Cfg(), annotations={'count': Annotated[int, {'min': 1, 'max': 9, 'step': 2}]})
        _check_dict(specs[0], {'name': 'count', 'kind': 'int', 'default': 3, 'min': 1, 'max': 9, 'step': 2},
                    'annotation override is respected')

        # A malformed / unsupported override falls back to ANY and warns.
        caplog.clear()
        with caplog.at_level('WARNING'):
            specs = specs_from_dataclass(Cfg(), annotations={'label': 123})
        _check_dict(specs[1], {'name': 'label', 'kind': 'any', 'default': 'hi'}, 'malformed annotation -> any')
        assert 'unsupported annotation' in caplog.text, f'malformed annotation should warn, got {caplog.text!r}'

    def test_unsupported_fields(self, caplog):
        class Opaque:
            pass

        @dataclass
        class Cfg:
            blob: Opaque = None

        with caplog.at_level('WARNING'):
            specs = specs_from_dataclass(Cfg())
        _check_dict(specs[0], {'name': 'blob', 'kind': 'any'}, 'unsupported field type -> any')
        assert 'unsupported annotation' in caplog.text, f'unsupported field should warn, got {caplog.text!r}'

    def test_unsupported_input(self, caplog):
        @dataclass
        class WithDefaults:
            x: int = 1

        @dataclass
        class NeedsArg:
            x: int  # no default, so NeedsArg() cannot be constructed

        # A class instantiable with defaults works just like its instance.
        assert [s.name for s in specs_from_dataclass(WithDefaults)] == ['x'], 'class with defaults is accepted'

        # A class that needs args is rejected with a logged, informative error.
        caplog.clear()
        with caplog.at_level('ERROR'):
            with pytest.raises(TypeError, match='cannot be instantiated'):
                specs_from_dataclass(NeedsArg)
        assert 'cannot be instantiated' in caplog.text, 'the non-instantiable-class error is logged'

        # A non-dataclass is rejected outright.
        with pytest.raises(TypeError, match='dataclass'):
            specs_from_dataclass(42)

    def test_names(self):
        @dataclass
        class Cfg:
            a: int = 1
            b: int = 2
            c: int = 3

        assert [s.name for s in specs_from_dataclass(Cfg(), names=['c', 'a'])] == ['c', 'a'], \
            'names selects a subset and honors its order'
        with pytest.raises(ValueError, match='unknown name'):
            specs_from_dataclass(Cfg(), names=['a', 'nope'])


class TestFunctionSpecs:
    """Tests for kaolin.visualize.dash.option.specs_from_function."""

    def test_basics(self):
        def fn(steps: int = 5, label: Annotated[str, UIBound(False)] = 'hi', untyped=1, *args, **kwargs):
            pass
        specs = specs_from_function(fn)
        # Only typed positional/keyword params are kept; *args/**kwargs and untyped params are skipped.
        assert [s.name for s in specs] == ['steps', 'label'], 'untyped params and *args/**kwargs are skipped'
        _check_dict(specs[0], {'name': 'steps', 'kind': 'int', 'default': 5}, 'int param')
        assert specs[1].ui_bound is False, 'UIBound annotation flows into ui_bound'

        def needs_arg(x: int):
            pass
        assert specs_from_function(needs_arg)[0].default is None, 'a param without a default maps to default None'

    def test_custom_annotations(self, caplog):
        def fn(count: int = 3, label: str = 'hi'):
            pass

        # An override annotation is honored over the parameter's declared type.
        specs = specs_from_function(fn, annotations={'count': Annotated[int, {'min': 1, 'max': 9, 'step': 2}]})
        _check_dict(specs[0], {'name': 'count', 'kind': 'int', 'default': 3, 'min': 1, 'max': 9, 'step': 2},
                    'annotation override is respected')

        # A malformed / unsupported override falls back to ANY and warns.
        caplog.clear()
        with caplog.at_level('WARNING'):
            specs = specs_from_function(fn, annotations={'label': 123})
        _check_dict(specs[1], {'name': 'label', 'kind': 'any', 'default': 'hi'}, 'malformed annotation -> any')
        assert 'unsupported annotation' in caplog.text, f'malformed annotation should warn, got {caplog.text!r}'

    def test_unsupported_fields(self, caplog):
        class Opaque:
            pass

        def fn(blob: Opaque = None):
            pass

        with caplog.at_level('WARNING'):
            specs = specs_from_function(fn)
        _check_dict(specs[0], {'name': 'blob', 'kind': 'any'}, 'unsupported param type -> any')
        assert 'unsupported annotation' in caplog.text, f'unsupported param should warn, got {caplog.text!r}'

    def test_unsupported_input(self):
        with pytest.raises(TypeError):
            specs_from_function(42)

    def test_names(self):
        def fn(a: int = 1, b: int = 2, c: int = 3):
            pass

        assert [s.name for s in specs_from_function(fn, names=['c', 'a'])] == ['c', 'a'], \
            'names selects a subset and honors its order'
        with pytest.raises(ValueError, match='unknown name'):
            specs_from_function(fn, names=['a', 'nope'])
