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

"""Typed description of a single behavior / UI option.

Defines the :class:`OptionKind` enum and :class:`OptionSpec`, the Python-side
mirror of the TypeScript ``OptionKind`` / ``OptionSpec`` in
``kaolin/visualize/dash/components/src/ts/core/behavior/option.ts``. Field names
are intentionally kept identical across the two sides so specs round-trip through
JSON (see :meth:`OptionSpec.as_dict` / :meth:`OptionSpec.from_dict`).

An :class:`OptionSpec` can also be derived from a typed Python source via
:meth:`OptionSpec.from_annotated_field`, which reads numeric bounds, step, and
allowed-value sets from PEP 593 ``Annotated[...]`` metadata.
"""

from __future__ import annotations

import enum
import inspect
import logging
from dataclasses import dataclass, fields, is_dataclass
from typing import Annotated, Any, Callable, Iterable, Literal, get_args, get_origin, get_type_hints

logger = logging.getLogger(__name__)

# Soft-support for the `annotated_types` PEP 593 vocabulary
# (https://github.com/annotated-types/annotated-types) -- the de-facto standard
# used by pydantic, msgspec, typer, etc. When available we read
# `Annotated[float, Ge(0), Le(1)]`-style constraints. Not a hard dependency: if
# the package is missing we recognize only the plain dict / Literal forms.
try:
    import annotated_types as _annotated_types
except ImportError:  # pragma: no cover - optional dep
    _annotated_types = None

__all__ = [
    'BehaviorMeta',
    'OptionKind',
    'OptionSpec',
    'UIBound',
    'specs_from_dataclass',
    'specs_from_function',
]


def specs_from_dataclass(instance: Any,
                         names: Iterable[str] | None = None,
                         annotations: dict[str, Any] | None = None,
                         ) -> list[OptionSpec]:
    """Build an :class:`OptionSpec` list from a dataclass instance or class.

    Each field is mapped to a spec via :meth:`OptionSpec.from_annotated_field`,
    reading its default from the instance's current attribute value. A nested
    dataclass field becomes an :attr:`OptionKind.GROUP` spec whose ``children``
    are the nested fields (recursively); a nested dataclass without a default
    instance to recurse into is skipped with a warning.

    A dataclass *class* is also accepted, provided it can be instantiated with
    no arguments (i.e. every field has a default); the defaults of that throwaway
    instance are used.

    Args:
        instance (Any): a dataclass instance, or a dataclass class instantiable
            with defaults.
        names (Optional[Iterable[str]]): field names to include and order;
            defaults to all fields in declaration order.
        annotations (Optional[dict]): ``{field_name: annotation}`` overrides that
            supplement or replace individual field annotations.

    Return:
        (list) the constructed :class:`OptionSpec` objects, in the resolved field order.

    Raises:
        TypeError: if ``instance`` is not a dataclass, or is a dataclass class
            that cannot be instantiated with defaults.
        ValueError: if ``names`` contains a field not present on the dataclass.
    """
    if isinstance(instance, type):
        if not is_dataclass(instance):
            raise TypeError(f'specs_from_dataclass: expected a dataclass, got class {instance.__name__}')
        try:
            instance = instance()
        except Exception as e:
            msg = (f'specs_from_dataclass: dataclass {instance.__name__} was passed as a class but cannot '
                   f'be instantiated with defaults ({e}); pass an instance instead')
            logger.error(msg)
            raise TypeError(msg) from e
    elif not is_dataclass(instance):
        raise TypeError(f'specs_from_dataclass: expected a dataclass instance or class, '
                        f'got {type(instance).__name__}')

    type_hints = get_type_hints(type(instance), include_extras=True)
    if annotations:
        type_hints = {**type_hints, **annotations}

    selected = _resolve_names([f.name for f in fields(instance)], names)

    out: list[OptionSpec] = []
    for n in selected:
        if n not in type_hints:
            logger.warning(f'option: skipping field {n!r}: no type annotation')
            continue
        spec = _field_to_spec(n, getattr(instance, n), type_hints[n])
        if spec is not None:
            out.append(spec)
    return out


def specs_from_function(fnc: Callable,
                        names: Iterable[str] | None = None,
                        annotations: dict[str, Any] | None = None,
                        ) -> list[OptionSpec]:
    """Build an :class:`OptionSpec` list from a callable's signature.

    Behaves like :func:`specs_from_dataclass` but introspects a function
    signature: ``*args`` / ``**kwargs`` are skipped, a parameter without a
    default is treated as ``default=None``, and parameters lacking a type
    annotation are skipped with a warning.

    Args:
        fnc (Callable): callable to introspect (bound methods are supported).
        names (Optional[Iterable[str]]): parameter names to include and order;
            defaults to all accepted parameters in declaration order.
        annotations (Optional[dict]): ``{param_name: annotation}`` overrides that
            supplement or replace individual parameter annotations.

    Return:
        (list) the constructed :class:`OptionSpec` objects, in the resolved param order.

    Raises:
        ValueError: if ``names`` contains a parameter not present on ``fnc``.
    """
    sig = inspect.signature(fnc)
    type_hints = get_type_hints(fnc, include_extras=True)
    if annotations:
        type_hints = {**type_hints, **annotations}

    accepted_kinds = {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
        inspect.Parameter.POSITIONAL_ONLY,
    }
    available = [n for n, p in sig.parameters.items() if p.kind in accepted_kinds]
    selected = _resolve_names(available, names)

    out: list[OptionSpec] = []
    for n in selected:
        if n not in type_hints:
            logger.warning(f'option: skipping parameter {n!r}: no type annotation')
            continue
        param = sig.parameters[n]
        default = None if param.default is inspect.Parameter.empty else param.default
        spec = _field_to_spec(n, default, type_hints[n])
        if spec is not None:
            out.append(spec)
    return out


@dataclass(frozen=True)
class UIBound:
    """PEP 593 annotation marking whether an option is user-accessible through UI.

    Use inside an ``Annotated[...]`` to set :attr:`OptionSpec.ui_bound` from a typed
    Python source, e.g. ``Annotated[int, Gt(0), UIBound(True)]``. The plain dict
    forms ``{'ui_bound': ...}`` / ``{'uiBound': ...}`` are accepted as alternatives.

    Args:
        is_bound (bool): whether the option is exposed in the auto-generated UI.
    """
    is_bound: bool = True


class OptionKind(str, enum.Enum):
    """Supported behavior-option value kinds.

    Mirrors the TS-side ``OptionKind`` enum in
    ``kaolin/visualize/dash/components/src/ts/core/behavior/option.ts``. The enum
    is ``str``-mixed so members compare equal to their string values
    (``OptionKind.INT == 'int'``) and round-trip cleanly through JSON / dict-shaped
    option specs.
    """
    INT = 'int'
    FLOAT = 'float'
    STRING = 'string'
    COLOR = 'color'
    BOOL = 'bool'
    ENUM = 'enum'
    GROUP = 'group'  # Note: not supported on the TS side yet
    ANY = 'any'


def _identity(value: Any) -> Any:
    return value


# Default (py_to_json, json_to_py) caster pair per kind. Scalar kinds coerce
# through the matching builtin so a value read off the wire lands in the right
# Python type; kinds without a single Python type (enum/group/any) pass through
# unchanged and can be overridden via :meth:`OptionSpec.add_type_cast`.
_DEFAULT_CASTERS: dict[OptionKind, tuple[Callable[[Any], Any], Callable[[Any], Any]]] = {
    OptionKind.INT: (int, int),
    OptionKind.FLOAT: (float, float),
    OptionKind.BOOL: (bool, bool),
    OptionKind.STRING: (str, str),
    OptionKind.COLOR: (str, str),
}


class OptionSpec:
    """Declarative description of a single behavior / UI option.

    Mirrors the TS-side ``OptionSpec`` interface in
    ``kaolin/visualize/dash/components/src/ts/core/behavior/option.ts``. Field
    names are intentionally kept identical to the TS side (including the camelCase
    ``uiBound`` emitted by :meth:`as_dict`) so the two definitions stay trivially
    comparable.

    A spec is typically flat (one scalar value), but a :attr:`OptionKind.GROUP`
    spec may hold ordered ``children`` to describe nested option groups.

    Args:
        name (str): option name; matches its key within the behavior schema.
        kind (OptionKind): value kind, drives which auto-UI control is generated.
        default (Any): default value used when the caller omits this key. Stored
            in its JSON-serializable form, *not* the rich Python form: for an
            enum option pass the member's value (e.g. ``'a'``), not the member
            itself. Use :meth:`add_type_cast` to register Python<->JSON casters.
        description (Optional[str]): human-readable label / tooltip for the auto-UI.
        min (Optional[int | float]): numeric lower bound (numeric kinds only).
        max (Optional[int | float]): numeric upper bound (numeric kinds only).
        step (Optional[int | float]): numeric step (numeric kinds only).
        values (Optional[list]): allowed value set (enum kind only).
        ui_bound (Optional[bool]): whether the option is user-accessible through UI.
    """

    def __init__(self,
                 name: str,
                 kind: OptionKind,
                 default: Any = None,
                 description: str | None = None,
                 min: int | float | None = None,
                 max: int | float | None = None,
                 step: int | float | None = None,
                 values: list | None = None,
                 ui_bound: bool | None = None):
        self.name = name
        self.kind = kind
        self.default = default  # kept in JSON-serializable form (see class docstring)
        self.description = description
        self.ui_bound = ui_bound

        # type-specific, but keeping flat for simplicity and consistency with TS
        self.min = min
        self.max = max
        self.step = step
        self.values = values
        self.children: list[OptionSpec] | None = [] if self.kind == OptionKind.GROUP else None

        # Private JSON<->Python value casters, seeded from the kind and optionally
        # overridden via add_type_cast(); never serialized by as_dict().
        self._py_to_json, self._json_to_py = _DEFAULT_CASTERS.get(self.kind, (_identity, _identity))
        self._custom_casters = False

        self.validate()

    def add_child(self, child: OptionSpec) -> None:
        """Append a child spec; only valid when this spec is a group kind.

        Args:
            child (OptionSpec): the nested option to add.

        Raises:
            ValueError: if this spec's :attr:`kind` is not :attr:`OptionKind.GROUP`.
        """
        if self.kind != OptionKind.GROUP:
            raise ValueError(f'OptionSpec "{self.name}": cannot add child to non-group kind {self.kind.value}')
        self.children.append(child)

    def add_type_cast(self, py_to_json: Callable[[Any], Any], json_to_py: Callable[[Any], Any]) -> None:
        """Register custom JSON<->Python value casters for this option.

        Casters convert between an option's rich Python value and its
        JSON-serializable form -- e.g. an enum member and its ``.value``. They
        are intentionally *not* constructor arguments and are private to the
        instance: they are never serialized, and :meth:`as_dict` warns if any
        custom caster is set (since it cannot round-trip a callable).

        Args:
            py_to_json (Callable): converts a Python value to its JSON form.
            json_to_py (Callable): converts a JSON value back to its Python form.
        """
        self._py_to_json = py_to_json
        self._json_to_py = json_to_py
        self._custom_casters = True

    @property
    def json_to_py(self) -> Callable[[Any], Any]:
        """Caster turning this option's JSON-friendly value into its Python value."""
        return self._json_to_py

    @property
    def py_to_json(self) -> Callable[[Any], Any]:
        """Caster turning this option's Python value into its JSON-friendly form."""
        return self._py_to_json

    def validate(self) -> None:
        """Check field consistency, raising :class:`ValueError` on a bad spec.

        Enforces the option contract: numeric bounds (``min`` / ``max`` / ``step``)
        only apply to :attr:`OptionKind.INT` / :attr:`OptionKind.FLOAT`, ``values``
        only applies to :attr:`OptionKind.ENUM`, and ``min <= max`` when both given.
        Also runs a loose check on ``default`` (when set): an enum default must be
        one of ``values``, and any other default must survive the kind's caster.

        Raises:
            ValueError: if any field is inconsistent with :attr:`kind`.
        """
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f'OptionSpec.name must be a non-empty string, got {self.name!r}')
        if not isinstance(self.kind, OptionKind):
            raise ValueError(f'OptionSpec.kind must be an OptionKind, got {self.kind!r}')

        numeric = self.kind in (OptionKind.INT, OptionKind.FLOAT)
        for attr in ('min', 'max', 'step'):
            value = getattr(self, attr)
            if value is None:
                continue
            if not numeric:
                raise ValueError(
                    f'OptionSpec "{self.name}": {attr} is only valid for numeric kinds '
                    f'(int/float), not {self.kind.value}')
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f'OptionSpec "{self.name}": {attr} must be int or float, got {value!r}')
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(
                f'OptionSpec "{self.name}": min ({self.min}) must be <= max ({self.max})')

        if self.kind == OptionKind.ENUM:
            if not self.values:
                raise ValueError(f'OptionSpec "{self.name}": enum kind requires a non-empty "values" list')
        elif self.values is not None:
            raise ValueError(
                f'OptionSpec "{self.name}": "values" is only valid for enum kind, not {self.kind.value}')

        self._validate_default()

    def _validate_default(self) -> None:
        """Loosely check ``default`` (stored as JSON).

        For an enum the default must be one of ``values``. Otherwise the check is
        intentionally permissive: a default is accepted as long as this option's
        JSON->Python caster can handle it (e.g. ``int('5')`` is fine, ``int('x')``
        is not), so coercible values like ``0``/``1`` for a bool are allowed.
        """
        if self.default is None:
            return
        if self.kind == OptionKind.ENUM:
            if self.default not in (self.values or []):
                raise ValueError(
                    f'OptionSpec "{self.name}": default {self.default!r} is not one of values {self.values!r}')
            return
        try:
            self._json_to_py(self.default)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f'OptionSpec "{self.name}": default {self.default!r} is not valid for kind '
                f'{self.kind.value}: {e}')

    def as_dict(self) -> dict:
        """Serialize this spec to a JSON-friendly dict (TS-compatible keys).

        ``kind`` is emitted as its string value and ``None`` fields are omitted,
        so the result round-trips through :meth:`from_dict`. Custom casters set via
        :meth:`add_type_cast` are not serialized; a warning is logged if any are set.

        Return:
            (dict) a dict with ``name`` and ``kind`` always present, plus any set
            optional fields (``default``, ``description``, ``min``, ``max``,
            ``step``, ``values``, ``uiBound``) and, for group kinds, a recursively
            serialized ``children`` list.
        """
        if self._custom_casters:
            logger.warning(
                'OptionSpec "%s": custom type casters are not serialized by as_dict()', self.name)
        return self._field_dict()

    def _field_dict(self) -> dict:
        """Serialize fields to a JSON-friendly dict without the custom-caster warning.

        Shared by :meth:`as_dict` (the public, warning-emitting entry point) and
        :meth:`merged` (which re-attaches casters, so the warning would be
        misleading). This is the single place that enumerates serializable fields.

        Return:
            (dict) same shape as :meth:`as_dict`.
        """
        out: dict = {'name': self.name, 'kind': self.kind.value}
        for key in ('default', 'description', 'min', 'max', 'step', 'values'):
            value = getattr(self, key)
            if value is not None:
                out[key] = value
        if self.ui_bound is not None:
            out['uiBound'] = self.ui_bound
        if self.children:
            out['children'] = [child._field_dict() for child in self.children]
        return out

    def as_string(self) -> str:
        """One-line, human-readable summary of this option (kind + range/values
        + default + description), suitable for listing under a behavior.

        Return:
            (str) the summary line.
        """
        bits = [f'{self.name}: {self.kind.value}']
        if self.kind == OptionKind.ENUM and self.values:
            bits.append(f'[{", ".join(str(v) for v in self.values)}]')
        elif self.kind in (OptionKind.INT, OptionKind.FLOAT):
            if self.min is not None or self.max is not None:
                bits.append(f'[{self.min if self.min is not None else "-inf"}..'
                            f'{self.max if self.max is not None else "inf"}]')
        if self.default is not None:
            bits.append(f'(default={self.default!r})')
        head = ' '.join(bits)
        return f'{head} — {self.description}' if self.description else head

    def merged(self, overrides: dict | None = None) -> OptionSpec:
        """Return a new spec copied from this one with ``overrides`` applied.

        Implemented on top of :meth:`as_dict` / :meth:`from_dict` so the set of
        serializable fields (and the ``children`` handling) lives in exactly one
        place. ``overrides`` keys are therefore the :meth:`as_dict` field names
        (``default``, ``min``, ``max``, ``step``, ``values``, ``description``,
        ``name``, ``kind``, ``uiBound``); ``ui_bound`` is accepted as an alias for
        ``uiBound`` and an :class:`OptionKind` is accepted for ``kind``. The copy
        is re-validated and custom casters are carried over. Passing no overrides
        returns a validated copy.

        Args:
            overrides (Optional[dict]): field overrides to apply over this spec.

        Return:
            (OptionSpec) the merged, validated copy.

        Raises:
            TypeError: if ``overrides`` contains a key that is not an OptionSpec field.
            ValueError: if the resulting spec is invalid (via :meth:`from_dict`).
        """
        overrides = dict(overrides or {})
        if 'ui_bound' in overrides:
            overrides['uiBound'] = overrides.pop('ui_bound')
        if isinstance(overrides.get('kind'), OptionKind):
            overrides['kind'] = overrides['kind'].value
        data = self._field_dict()
        data.update(overrides)
        new = OptionSpec.from_dict(data)
        if self._custom_casters:
            new.add_type_cast(self._py_to_json, self._json_to_py)
        return new

    @classmethod
    def from_dict(cls, data: dict) -> OptionSpec:
        """Build an :class:`OptionSpec` from a JSON-friendly dict.

        Inverse of :meth:`as_dict`; ``kind`` may be an :class:`OptionKind` or its
        string value, ``uiBound`` is accepted as an alias for ``ui_bound``, and a
        ``children`` list is reconstructed recursively (only valid for group kinds,
        enforced by :meth:`add_child`).

        Args:
            data (dict): dict with at least ``name`` and ``kind`` keys.

        Return:
            (OptionSpec) the constructed (and validated) spec.
        """
        args = {k: v for k, v in data.items() if k not in ('children')}
        assert 'name' in args and 'kind' in args, f'OptionSpec requires "name" and "kind" keys, got {args.keys()}'
        args['kind'] = OptionKind(args['kind'])  # validate?
        if 'uiBound' in args:
            args['ui_bound'] = args['uiBound']
            del args['uiBound']
        spec = cls(**args)
        for child in data.get('children') or []:
            spec.add_child(cls.from_dict(child))
        return spec

    @classmethod
    def from_annotated_field(cls, name, annotation: Annotated[Any, Any], default: Any = None) -> OptionSpec:
        """Build an :class:`OptionSpec` from a ``(name, annotation, default)`` triple.

        Reads numeric bounds / step / allowed-values out of PEP 593 metadata via
        :func:`_min_max_from_annotation_metadata`, :func:`_step_from_annotation_metadata`,
        and :func:`_values_from_annotation_metadata`, then maps the base type to an
        :class:`OptionKind`. When the base type is an :class:`enum.Enum` subclass a
        member <-> value caster is registered via :meth:`add_type_cast`.

        Examples:
            Supported annotation forms::

                from typing import Annotated, Literal
                from annotated_types import Ge, Gt, Interval, MultipleOf

                # Bare scalar types -> int / float / bool / string options
                OptionSpec.from_annotated_field('count', int, default=1)
                OptionSpec.from_annotated_field('name', str, default='cube')
                OptionSpec.from_annotated_field('visible', bool, default=True)

                # Numeric bounds / step, via annotated_types or a plain dict
                OptionSpec.from_annotated_field('age', Annotated[int, Gt(0)], default=1)
                OptionSpec.from_annotated_field('ratio', Annotated[float, Interval(ge=0, le=1)], default=0.5)
                OptionSpec.from_annotated_field('size', Annotated[int, MultipleOf(2)], default=2)
                OptionSpec.from_annotated_field('size', Annotated[int, {'min': 0, 'max': 10, 'step': 2}], default=2)

                # Enum options: a Literal, an explicit value list, or an enum.Enum subclass
                OptionSpec.from_annotated_field('mode', Literal['a', 'b', 'c'], default='a')
                OptionSpec.from_annotated_field('mode', Annotated[str, {'values': ['a', 'b']}], default='a')
                OptionSpec.from_annotated_field('mode', Annotated[str, ['a', 'b']], default='a')
                OptionSpec.from_annotated_field('mode', Color, default=Color.RED)  # Color(enum.Enum)

                # Mark an option as UI-bound via a UIBound marker or a dict
                OptionSpec.from_annotated_field('age', Annotated[int, Gt(0), UIBound(True)], default=1)
                OptionSpec.from_annotated_field('age', Annotated[int, Gt(0), {'uiBound': True}], default=1)

        Args:
            name (str): option name.
            annotation (Annotated): the (possibly ``Annotated[...]``) type annotation.
            default (Any): default value for the option.

        Return:
            (OptionSpec) the constructed (and validated) spec.
        """
        base_type, metadata = _strip_annotated(annotation)

        mn, mx = _min_max_from_annotation_metadata(metadata)
        step = _step_from_annotation_metadata(metadata)
        values = _values_from_annotation_metadata(metadata)
        ui_bound = _ui_bound_from_annotation_metadata(metadata)

        if get_origin(base_type) is Literal:
            return cls(name=name, kind=OptionKind.ENUM, default=default,
                       values=list(get_args(base_type)), ui_bound=ui_bound)

        if isinstance(base_type, type) and issubclass(base_type, enum.Enum):
            members = [m.value for m in base_type]
            value = default.value if isinstance(default, enum.Enum) else default
            spec = cls(name=name, kind=OptionKind.ENUM, default=value, values=members, ui_bound=ui_bound)
            # The enum class is the only thing that knows how to map a wire value
            # back to a member, so register a caster for it (identity won't do).
            spec.add_type_cast(py_to_json=lambda m: m.value if isinstance(m, enum.Enum) else m,
                               json_to_py=base_type)
            return spec

        if values is not None:
            return cls(name=name, kind=OptionKind.ENUM, default=default, values=values, ui_bound=ui_bound)

        if base_type is bool:
            return cls(name=name, kind=OptionKind.BOOL, default=default, ui_bound=ui_bound)

        if base_type is int:
            return cls(name=name, kind=OptionKind.INT, default=default, min=mn, max=mx, step=step,
                       ui_bound=ui_bound)

        if base_type is float:
            return cls(name=name, kind=OptionKind.FLOAT, default=default, min=mn, max=mx, step=step,
                       ui_bound=ui_bound)

        if base_type is str:
            return cls(name=name, kind=OptionKind.STRING, default=default, ui_bound=ui_bound)

        logger.warning(f'option: unsupported annotation {base_type!r} for {name!r}; using kind=any')
        return cls(name=name, kind=OptionKind.ANY, default=default, ui_bound=ui_bound)


# ----------------------------------------------------------------------------------------
# Internal helpers: build OptionSpecs from dataclass fields / function parameters
# ----------------------------------------------------------------------------------------

def _resolve_names(available: list[str], names: Iterable[str] | None) -> list[str]:
    """Return ``names`` in order, validated against ``available``; ``None`` selects all.

    Raises:
        ValueError: if any requested name is not in ``available``.
    """
    if names is None:
        return list(available)
    names = list(names)
    unknown = [n for n in names if n not in set(available)]
    if unknown:
        raise ValueError(f'option: unknown name(s) {unknown}; available names are {available}')
    return names


def _field_to_spec(name: str, default: Any, annotation: Any) -> OptionSpec | None:
    """Map one ``(name, default, annotation)`` triple to an :class:`OptionSpec`.

    Delegates scalar / enum fields to :meth:`OptionSpec.from_annotated_field`. A
    nested dataclass becomes an :attr:`OptionKind.GROUP` whose ``children`` are the
    nested fields; a nested dataclass without a default instance returns ``None``
    (the caller logs and skips it).
    """
    base_type, _ = _strip_annotated(annotation)
    if isinstance(base_type, type) and is_dataclass(base_type):
        if default is None:
            logger.warning(f'option: skipping nested dataclass {name!r}: no default instance')
            return None
        group = OptionSpec(name=name, kind=OptionKind.GROUP)
        for child in specs_from_dataclass(default):
            group.add_child(child)
        return group
    return OptionSpec.from_annotated_field(name, annotation, default)


# ----------------------------------------------------------------------------------------
# Internal helpers: read option metadata out of PEP 593 Annotated[...] annotations
# ----------------------------------------------------------------------------------------

def _strip_annotated(annotation: Any) -> tuple[Any, tuple]:
    """Return ``(base_type, metadata_tuple)``; strips a single Annotated[...] wrapper."""
    if get_origin(annotation) is Annotated:
        args = get_args(annotation)
        return args[0], args[1:]
    return annotation, ()


def _find_in_metadata(metadata: tuple, cls: type):
    """Return the first metadata item that is an instance of ``cls``, else ``None``."""
    for m in metadata:
        if isinstance(m, cls):
            return m
    return None


def _min_max_from_annotation_metadata(metadata: tuple) -> tuple[float | None, float | None]:
    """Extract ``(min, max)`` numeric bounds from PEP 593 metadata.

    Recognizes, in order of increasing precedence: ``annotated_types`` constraints
    (``Ge`` / ``Gt`` / ``Le`` / ``Lt`` / ``Interval``), and a plain
    ``{'min': ..., 'max': ...}`` dict. ``Gt`` / ``Lt`` are folded into the
    min / max identically to ``Ge`` / ``Le``.

    Args:
        metadata (tuple): the metadata tuple from :func:`_strip_annotated`.

    Return:
        (tuple) ``(min, max)``; each element is ``None`` when that bound is absent.
    """
    mn: Any = None
    mx: Any = None

    if _annotated_types is not None:
        for m in metadata:
            if isinstance(m, _annotated_types.Interval):
                if m.ge is not None:
                    mn = m.ge
                elif m.gt is not None:
                    mn = m.gt
                if m.le is not None:
                    mx = m.le
                elif m.lt is not None:
                    mx = m.lt
            elif isinstance(m, _annotated_types.Ge):
                mn = m.ge
            elif isinstance(m, _annotated_types.Gt):
                mn = m.gt
            elif isinstance(m, _annotated_types.Le):
                mx = m.le
            elif isinstance(m, _annotated_types.Lt):
                mx = m.lt

    for m in metadata:
        if isinstance(m, dict):
            if m.get('min') is not None:
                mn = m['min']
            if m.get('max') is not None:
                mx = m['max']

    return mn, mx


def _step_from_annotation_metadata(metadata: tuple) -> float | None:
    """Extract a numeric ``step`` from PEP 593 metadata.

    Recognizes ``annotated_types.MultipleOf`` and a plain ``{'step': ...}`` dict
    (in order of increasing precedence).

    Args:
        metadata (tuple): the metadata tuple from :func:`_strip_annotated`.

    Return:
        (float) the step value, or ``None`` if no step is present.
    """
    step: Any = None

    if _annotated_types is not None:
        multiple_of = _find_in_metadata(metadata, _annotated_types.MultipleOf)
        if multiple_of is not None:
            step = multiple_of.multiple_of

    for m in metadata:
        if isinstance(m, dict) and m.get('step') is not None:
            step = m['step']

    return step


def _values_from_annotation_metadata(metadata: tuple) -> list[Any] | None:
    """Extract an allowed-value set from PEP 593 metadata.

    Recognizes a :data:`typing.Literal` (``Annotated[int, Literal[1, 2, 3]]``),
    a plain ``{'values': [...]}`` dict, and a bare ``list`` / ``tuple`` / ``set``
    present in the metadata (in that order of precedence).

    Args:
        metadata (tuple): the metadata tuple from :func:`_strip_annotated`.

    Return:
        (list) the list of allowed values, or ``None`` if none is present.
    """
    for m in metadata:
        if get_origin(m) is Literal:
            return list(get_args(m))
    for m in metadata:
        if isinstance(m, dict) and m.get('values') is not None:
            return list(m['values'])
    for m in metadata:
        if isinstance(m, (list, tuple, set)):
            return list(m)
    return None


def _ui_bound_from_annotation_metadata(metadata: tuple) -> bool | None:
    """Extract whether an option is UI-bound from PEP 593 metadata.

    Recognizes a :class:`UIBound` marker and a plain ``{'ui_bound': ...}`` /
    ``{'uiBound': ...}`` dict (in that order of precedence).

    Args:
        metadata (tuple): the metadata tuple from :func:`_strip_annotated`.

    Return:
        (bool) the UI-bound flag, or ``None`` if it is not specified.
    """
    marker = _find_in_metadata(metadata, UIBound)
    if marker is not None:
        return marker.is_bound
    for m in metadata:
        if isinstance(m, dict):
            if m.get('ui_bound') is not None:
                return bool(m['ui_bound'])
            if m.get('uiBound') is not None:
                return bool(m['uiBound'])
    return None


class BehaviorMeta:
    """Metadata for a single registered behavior: a description plus its option specs.

    Python-side mirror of the TypeScript ``BehaviorMeta`` in
    ``kaolin/visualize/dash/components/src/ts/core/behavior/option.ts``, and the
    decoded form of one entry in the behavior manifest emitted by
    ``dump_behavior_manifest.ts``. Use :meth:`from_dict` to read a manifest
    entry into typed :class:`OptionSpec` instances and :meth:`as_dict` for the
    inverse (the JSON round-trip is symmetric with the TS ``toJSON`` /
    ``fromJson``).

    ``source_path`` is Python-only provenance (the user-asset file a behavior was
    discovered in); it has no TypeScript counterpart and is omitted from
    :meth:`as_dict` when unset.

    TODO: extend both this class and the TypeScript ``BehaviorMeta`` with the
    behavior's interface flags once the dumper can emit them, e.g.
    ``is_element_bound`` / ``is_message_handler`` / ``is_camera_controller``
    (camelCase ``isElementBound`` / ``isMessageHandler`` / ``isCameraController``
    on the wire).
    """

    #: Keys accepted by :meth:`from_dict` (``options`` is consumed separately).
    _KNOWN_KEYS = frozenset({'description', 'options', 'sourcePath', 'source_path'})

    def __init__(self,
                 description: str | None = None,
                 options: dict[str, OptionSpec] | None = None,
                 source_path: str | None = None):
        """Create behavior metadata.

        Args:
            description (str | None): human-readable behavior description, or
                ``None`` if none.
            options (dict[str, OptionSpec] | None): option specs keyed by name;
                ``None`` is treated as an empty mapping.
            source_path (str | None): provenance for a user-scanned behavior
                (the file it was registered in), or ``None`` for library
                behaviors.
        """
        self.description = description
        self.options: dict[str, OptionSpec] = dict(options or {})
        self.source_path = source_path

    def as_dict(self) -> dict:
        """Serialize to a JSON-friendly dict (manifest entry form).

        Each option is serialized via :meth:`OptionSpec.as_dict`, so the result
        round-trips through :meth:`from_dict`. ``sourcePath`` is included only
        when set.

        Return:
            (dict) a dict with ``description`` and an ``options`` mapping of
            ``name -> OptionSpec.as_dict()`` (plus ``sourcePath`` if present).
        """
        out: dict = {
            'description': self.description,
            'options': {name: spec.as_dict() for name, spec in self.options.items()},
        }
        if self.source_path is not None:
            out['sourcePath'] = self.source_path
        return out

    @classmethod
    def from_dict(cls, data: dict) -> BehaviorMeta:
        """Build a :class:`BehaviorMeta` from a manifest entry dict.

        Inverse of :meth:`as_dict`. Each option dict is rebuilt via
        :meth:`OptionSpec.from_dict` (falling back to the mapping key for a
        spec's ``name`` if absent). ``sourcePath`` (or ``source_path``) is read
        as provenance. Unknown top-level keys raise :class:`ValueError`.

        Args:
            data (dict): a manifest entry with optional ``description``,
                ``options``, and ``sourcePath`` keys.

        Return:
            (BehaviorMeta) the decoded metadata.
        """
        unknown = set(data) - cls._KNOWN_KEYS
        if unknown:
            raise ValueError(f'BehaviorMeta.from_dict: unknown keys {sorted(unknown)}')
        options = {}
        for name, spec in (data.get('options') or {}).items():
            options[name] = OptionSpec.from_dict({'name': name, **spec})
        return cls(description=data.get('description'),
                   options=options,
                   source_path=data.get('source_path', data.get('sourcePath')))

    def as_string(self, name: str, detailed: bool = False) -> str:
        """Render this behavior as human-readable, indented text.

        Args:
            name (str): the behavior's registered name (kept as the registry key,
                not stored on the instance).
            detailed (bool): if True, include description, provenance, and the
                per-option summary; otherwise a single ``- name`` line.

        Return:
            (str) the rendered (possibly multi-line) summary.
        """
        if not detailed:
            suffix = f'  (from {self.source_path})' if self.source_path else ''
            return f'    - {name}{suffix}'

        lines = [f'    - {name}']
        if self.description:
            lines.append(f'        description: {self.description}')
        # TODO: render element/message/camera interface flags once available
        #       (see class-level TODO).
        if self.source_path:
            lines.append(f'        source: {self.source_path}')
        if self.options:
            lines.append('        options:')
            for opt_name in sorted(self.options):
                lines.append('          ' + self.options[opt_name].as_string())
        else:
            lines.append('        options: (none)')
        return '\n'.join(lines)
