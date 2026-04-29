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

"""Built-in / known behavior discovery for the Dash viewer.

:class:`BehaviorLibrary` is the single Python entry point downstream code
(auto-UI, builder, sidebar generators) should use to enumerate behaviors.
It merges:

* the build-time library manifest (see
  :mod:`kaolin.visualize.dash.behavior_manifest`), and
* runtime scans of user-supplied asset directories (see
  :mod:`kaolin.visualize.dash.user_behavior_scan`).

Both sources expose :class:`kaolin.visualize.dash.option.BehaviorMeta`
instances, so downstream code does not need to care which origin a
behavior came from.

Singleton style mirrors :class:`kaolin.visualize.dash.builder.SessionRegistry`:
all public methods are class methods working on class-level state, and
there is no module-level instance.

Conflict policy: a user-registered name that collides with a library name
(or with another user dir) raises :class:`ValueError`. This makes silent
shadowing impossible — rename the conflicting user behavior instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

from kaolin.visualize.dash import behavior_manifest as _manifest_module
from kaolin.visualize.dash import user_behavior_scan as _scan_module
from kaolin.visualize.dash.option import BehaviorMeta

logger = logging.getLogger(__name__)

__all__ = [
    'BehaviorLibrary',
    'is_builtin_layer',
]


class _BehaviorLibraryMeta(type):
    """Metaclass so ``print(BehaviorLibrary)`` / ``str(BehaviorLibrary)`` produce
    the human-readable summary instead of the default ``<class ...>`` repr.

    Implemented as a metaclass rather than a regular ``__str__`` because the
    public API is class-method-only — ``BehaviorLibrary`` is intentionally
    never instantiated.
    """

    def __str__(cls) -> str:
        return cls.to_string(detailed=False)  # type: ignore[attr-defined]

    def __repr__(cls) -> str:
        return cls.to_string(detailed=False)  # type: ignore[attr-defined]


class BehaviorLibrary(metaclass=_BehaviorLibraryMeta):
    """Process-wide registry of behavior names and their metadata.

    Library behaviors are loaded once from the build-time manifest. User
    directories are opt-in: callers (e.g. :class:`WebappBuilder` /
    :class:`ViewerBuilder`) call :meth:`register_user_directory` once per
    directory; subsequent :meth:`names` / :meth:`meta` /
    :meth:`is_builtin_behavior` calls reflect the merged view.

    All state is class-level; do not instantiate. The class is intended to
    be used directly:

    .. code-block:: python

        from kaolin.visualize.dash.builtins import BehaviorLibrary

        BehaviorLibrary.register_user_directory('kaolin/app/segment/assets')
        print(BehaviorLibrary)                          # summary
        print(BehaviorLibrary.to_string(True))          # detailed
        print(BehaviorLibrary.behavior_to_string('drawing', True))
        BehaviorLibrary.is_builtin_behavior('drawing')  # True
        BehaviorLibrary.is_user_behavior('drawing')     # False
        BehaviorLibrary.meta('drawing')

    Not thread-safe; assume single-threaded build/server initialization
    (matches Dash's threading model for layout assembly).
    """

    _user_directories: list[Path] = []
    _user_cache: dict[str, BehaviorMeta] = {}
    _user_cache_invalid: bool = True

    # ----------------------------------------------------------------------
    # Library side (immutable per process)
    # ----------------------------------------------------------------------

    @classmethod
    def _library(cls) -> dict[str, BehaviorMeta]:
        return _manifest_module.get_behavior_manifest()

    @classmethod
    def is_builtin_behavior(cls, name: str) -> bool:
        """``True`` iff ``name`` appears in the build-time library manifest."""
        return name in cls._library()

    # ----------------------------------------------------------------------
    # User side (mutable, opt-in)
    # ----------------------------------------------------------------------

    @classmethod
    def register_user_directory(cls, path: str | Path) -> None:
        """Register a directory whose ``.js``/``.ts``/``.tsx`` files should be
        scanned for ``BehaviorRegister.register(...)`` calls.

        Duplicate registrations are ignored. The clash check (against the
        library and against previously registered user dirs) runs eagerly
        here so the offending call site is the one that raises.
        """
        resolved = Path(path).resolve()
        if resolved in cls._user_directories:
            return
        cls._user_directories.append(resolved)
        cls._user_cache_invalid = True
        cls._user()

    @classmethod
    def invalidate_user_cache(cls) -> None:
        """Force a re-scan of user directories on the next lookup."""
        cls._user_cache_invalid = True

    @classmethod
    def is_user_behavior(cls, name: str) -> bool:
        """``True`` iff ``name`` was discovered in a registered user directory."""
        return name in cls._user()

    @classmethod
    def _user(cls) -> dict[str, BehaviorMeta]:
        if not cls._user_cache_invalid:
            return cls._user_cache

        library = cls._library()
        merged: dict[str, BehaviorMeta] = {}
        for directory in cls._user_directories:
            entries = _scan_module.scan_user_behavior_names([directory])
            for name, entry in entries.items():
                if name in library:
                    raise ValueError(
                        f'BehaviorLibrary: user behavior {name!r} (from '
                        f'{entry.source_path}) clashes with a library '
                        f'behavior of the same name. Rename the user behavior.')
                if name in merged:
                    raise ValueError(
                        f'BehaviorLibrary: user behavior {name!r} is registered '
                        f'by both {merged[name].source_path} and '
                        f'{entry.source_path}. Rename one of them.')
                merged[name] = entry

        cls._user_cache = merged
        cls._user_cache_invalid = False
        return cls._user_cache

    # ----------------------------------------------------------------------
    # Unified view
    # ----------------------------------------------------------------------

    @classmethod
    def names(cls) -> list[str]:
        """All known behavior names (library + user), sorted."""
        return sorted(set(cls._library()) | set(cls._user()))

    @classmethod
    def meta(cls, name: str) -> BehaviorMeta | None:
        """Return the :class:`BehaviorMeta` for ``name``, or ``None`` if unknown.

        Name collisions are impossible by construction (they raise at
        :meth:`register_user_directory` time), so this is an unambiguous lookup.
        """
        lib = cls._library().get(name)
        if lib is not None:
            return lib
        return cls._user().get(name)

    @classmethod
    def all_meta(cls) -> dict[str, BehaviorMeta]:
        """Return a merged ``{name: BehaviorMeta}`` dict (library + user)."""
        merged = dict(cls._user())
        merged.update(cls._library())
        return merged

    # ----------------------------------------------------------------------
    # Pretty printing
    # ----------------------------------------------------------------------

    @classmethod
    def to_string(cls, detailed: bool = False) -> str:
        """Human-readable summary of the registry, one behavior per line.

        Args:
            detailed: if True, include the per-behavior element binding and
                its options schema (a behavior's effective input surface).

        Built-in and user behaviors are listed under separate headers.
        """
        library = cls._library()
        user = cls._user()
        lines: list[str] = [
            f'BehaviorLibrary ({len(library)} built-in, {len(user)} user):'
        ]

        lines.append('')
        lines.append(f'  Built-in ({len(library)}):')
        if not library:
            lines.append('    (none — run `npm run build:dash:manifest`)')
        else:
            for name in sorted(library):
                lines.append(library[name].as_string(name, detailed))

        lines.append('')
        lines.append(f'  User ({len(user)}):')
        if not user:
            lines.append('    (none)')
        else:
            for name in sorted(user):
                lines.append(user[name].as_string(name, detailed))

        return '\n'.join(lines)

    @classmethod
    def behavior_to_string(cls, name: str, detailed: bool = False) -> str:
        """Same semantics as :meth:`to_string`, but for a single behavior.

        Args:
            name: behavior name to render.
            detailed: if True, include description, element binding and
                option schema (matching :meth:`to_string` detailed mode).

        Raises:
            ValueError: if ``name`` is neither a library nor a user behavior.
        """
        library = cls._library()
        if name in library:
            return library[name].as_string(name, detailed)
        user = cls._user()
        if name in user:
            return user[name].as_string(name, detailed)
        raise ValueError(
            f'BehaviorLibrary: {name!r} is not a known behavior. '
            f'Known: {cls.names()}')

    # ----------------------------------------------------------------------
    # Test / dev hook
    # ----------------------------------------------------------------------

    @classmethod
    def _reset(cls) -> None:
        """Wipe all user-side state. Intended for tests; clears registered
        directories and the cached scan so each test starts from a clean
        process-wide singleton.
        """
        cls._user_directories = []
        cls._user_cache = {}
        cls._user_cache_invalid = True


def is_builtin_layer(tag, warn=False):
    # TODO: parity with `BehaviorLibrary.is_builtin_behavior` — emit a layer
    #       manifest at build time so this stops being best-effort.
    if warn:
        logger.warning(f'Layer tag {tag} is not checked -- implement')
    return True
