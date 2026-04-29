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

"""Runtime regex scanner for user-side behavior assets.

V0 just discovers behavior *names*: every call to ``BehaviorRegister.register``
in a user-provided directory's ``.js``/``.ts``/``.tsx`` files is matched
(anchored on the dotted suffix so namespaced calls like
``window.kaolin.interact.behavior_register.BehaviorRegister.register("foo", ...)``
match). The shape returned per entry mirrors
``kaolin/visualize/dash/components/autogen/behavior_manifest.json`` for the
library so the unified accessor in :mod:`kaolin.visualize.dash.builtins` can
return either origin without callers caring which.

Future versions can extend this to parse ``static schema = {...}`` and tag
``isElementBound`` / ``elementType`` by walking class declarations; the
public return shape is forward-compatible.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable

from kaolin.visualize.dash.option import BehaviorMeta

logger = logging.getLogger(__name__)

__all__ = [
    'USER_BEHAVIOR_FILE_SUFFIXES',
    'scan_user_behavior_names',
]

#: File extensions the scanner inspects.
USER_BEHAVIOR_FILE_SUFFIXES: frozenset[str] = frozenset({'.js', '.ts', '.tsx'})

# Anchored on the dotted-suffix `BehaviorRegister.register(` so namespaced
# forms (e.g. `window.kaolin.interact.behavior_register.BehaviorRegister.register(...)`)
# match. The first capture is the behavior name (single or double quoted).
_REGISTER_RE = re.compile(
    r'BehaviorRegister\.register\(\s*["\']([A-Za-z0-9_\-]+)["\']'
)

# Strip `// ...` and `/* ... */` comments before regex matching. Not a full
# JS lexer (won't respect strings that contain `//`), but good enough to
# avoid false positives from documentation/snippet examples.
_LINE_COMMENT_RE = re.compile(r'//[^\n]*')
_BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', re.DOTALL)


def _strip_comments(source: str) -> str:
    """Best-effort strip of JS/TS comments. Run before regex matching."""
    no_block = _BLOCK_COMMENT_RE.sub('', source)
    return _LINE_COMMENT_RE.sub('', no_block)


# (path -> mtime) cache so repeated calls during dev don't re-read files.
_file_cache: dict[Path, tuple[float, list[str]]] = {}


def _names_in_file(path: Path) -> list[str]:
    """Return the sorted, unique list of behavior names registered in ``path``.

    mtime-cached: re-reads the file only when its mtime advances. The cache
    is process-local; safe across many scans within a single Dash run.
    """
    try:
        mtime = path.stat().st_mtime
    except OSError as exc:
        logger.warning('user_behavior_scan: cannot stat %s: %s', path, exc)
        return []
    cached = _file_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        source = path.read_text(encoding='utf-8', errors='replace')
    except OSError as exc:
        logger.warning('user_behavior_scan: cannot read %s: %s', path, exc)
        _file_cache[path] = (mtime, [])
        return []
    stripped = _strip_comments(source)
    names = sorted({m.group(1) for m in _REGISTER_RE.finditer(stripped)})
    _file_cache[path] = (mtime, names)
    return names


def scan_user_behavior_names(directories: Iterable[str | Path]) -> dict[str, BehaviorMeta]:
    """Scan ``directories`` recursively for ``BehaviorRegister.register(...)`` calls.

    Args:
        directories: directories to scan (typically a per-app ``assets/``
            folder). Missing directories are skipped with a warning.

    Returns:
        ``{behavior_name: BehaviorMeta}`` where each :class:`BehaviorMeta`
        carries the ``source_path`` of the file that registered the behavior
        (description and options are empty in V0). If the same behavior name
        appears in multiple files, the *last* file wins (a warning is logged).
    """
    result: dict[str, BehaviorMeta] = {}
    for raw in directories:
        directory = Path(raw)
        if not directory.is_dir():
            logger.warning('user_behavior_scan: directory %s does not exist; skipping', directory)
            continue
        for path in directory.rglob('*'):
            if not path.is_file() or path.suffix not in USER_BEHAVIOR_FILE_SUFFIXES:
                continue
            for name in _names_in_file(path):
                if name in result:
                    logger.warning(
                        'user_behavior_scan: behavior %r registered in both %s and %s; '
                        'using the latter',
                        name, result[name].source_path, path)
                result[name] = BehaviorMeta(description=None, options={}, source_path=str(path))
    return result
