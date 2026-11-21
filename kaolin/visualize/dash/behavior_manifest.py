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

"""Python lookup for the library-side behavior manifest.

The manifest is emitted at build time by
``kaolin/visualize/dash/components/scripts/dump_behavior_manifest.ts`` (wired
into ``npm run build:dash``) and ships with the wheel under
``kaolin/visualize/dash/components/autogen/behavior_manifest.json``.

Use :func:`get_behavior_manifest` for the full dict and
:func:`get_behavior_meta` for a single entry. The unified accessor that also
merges user-side scans lives in :mod:`kaolin.visualize.dash.builtins`.
"""

from __future__ import annotations

import json
import logging
from importlib import resources
from typing import Any

from kaolin.visualize.dash.option import BehaviorMeta

logger = logging.getLogger(__name__)

__all__ = [
    'BEHAVIOR_MANIFEST_RESOURCE',
    'get_behavior_manifest',
    'get_behavior_meta',
]

#: Resource name of the manifest JSON inside the ``autogen`` package.
BEHAVIOR_MANIFEST_RESOURCE = 'behavior_manifest.json'

_AUTOGEN_PACKAGE = 'kaolin.visualize.dash.components.autogen'

# Module-level cache. ``_loaded`` distinguishes "not yet attempted" from
# "loaded but missing" (in which case ``_cache`` stays ``{}`` and we don't
# spam the warning on every call). ``_meta_cache`` holds the decoded
# :class:`BehaviorMeta` view, built lazily from ``_cache``.
_cache: dict[str, dict[str, Any]] = {}
_loaded: bool = False
_meta_cache: dict[str, BehaviorMeta] | None = None


def _load_manifest() -> dict[str, dict[str, Any]]:
    global _cache, _loaded
    if _loaded:
        return _cache
    _loaded = True
    try:
        text = (resources.files(_AUTOGEN_PACKAGE)
                .joinpath(BEHAVIOR_MANIFEST_RESOURCE)
                .read_text(encoding='utf-8'))
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        logger.warning(
            'behavior_manifest.json not found in %s (%s); '
            "run 'npm run build:dash:manifest' to generate it. "
            "Library behaviors will not be discoverable from Python.",
            _AUTOGEN_PACKAGE, exc)
        _cache = {}
        return _cache
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error('behavior_manifest.json is not valid JSON: %s', exc)
        _cache = {}
        return _cache
    if not isinstance(parsed, dict):
        logger.error('behavior_manifest.json must be an object, got %s', type(parsed).__name__)
        _cache = {}
        return _cache
    _cache = parsed
    return _cache


def _behavior_metas() -> dict[str, BehaviorMeta]:
    """Decode the raw manifest into :class:`BehaviorMeta` instances (cached).

    Entries that fail to decode are skipped with a logged error rather than
    failing the whole load.
    """
    global _meta_cache
    if _meta_cache is not None:
        return _meta_cache
    metas: dict[str, BehaviorMeta] = {}
    for name, entry in _load_manifest().items():
        try:
            metas[name] = BehaviorMeta.from_dict(entry)
        except (ValueError, TypeError, KeyError) as exc:
            logger.error('behavior_manifest: cannot decode entry %r: %s', name, exc)
    _meta_cache = metas
    return _meta_cache


def get_behavior_manifest() -> dict[str, BehaviorMeta]:
    """Return the full library behavior manifest as ``{name: BehaviorMeta}`` (cached).

    Each value is a :class:`kaolin.visualize.dash.option.BehaviorMeta` with a
    ``description`` and ``options`` (a ``{name: OptionSpec}`` mapping). Returns
    an empty dict if the manifest has not been generated yet (a warning is
    logged once).
    """
    return _behavior_metas()


def get_behavior_meta(name: str) -> BehaviorMeta | None:
    """Return one library behavior's :class:`BehaviorMeta`, or ``None`` if unknown.

    Args:
        name: behavior name as passed to ``BehaviorRegister.register`` on the
            JS side (e.g. ``'drawing'``, ``'konva_drawing'``).
    """
    return _behavior_metas().get(name)
