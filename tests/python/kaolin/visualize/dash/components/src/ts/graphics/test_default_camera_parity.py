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
"""Python<->TypeScript parity for the *default* camera interchange format.

The TypeScript viewer hard-codes ``defaultCameraParameters`` in
``kaolin/visualize/dash/components/src/ts/graphics/camera.ts``. That value must
stay byte-for-byte compatible with the Python default camera
(:func:`kaolin.render.easy_render.default_camera`) ``as_dict()`` output, since
the two sides exchange exactly this structure.

The shared golden JSON encodes the expected interchange format and is also
checked from the TS side in
``tests/ts/kaolin/visualize/dash/graphics/test_camera.ts``.

A binary golden (``default_camera_parameters.bin``) carries the same dict through
the binary wire format (:func:`kaolin.visualize.web.io.to_binary`); the TS side
decodes it with ``fromBinary`` to validate the binary interchange end-to-end.

This test deliberately separates two failure modes:
  * key/field-convention drift (the *shape* of the interchange dict), and
  * value drift (numeric parameters / view matrix),
so a mismatch points clearly at which side of the contract changed.
"""
import json
import math
import os

import pytest

from kaolin.render.camera import Camera
from kaolin.render.easy_render import default_camera
from kaolin.utils.testing import contained_torch_equal
from kaolin.visualize.web.io import from_binary

# This file: tests/python/kaolin/visualize/dash/components/src/ts/graphics/<this>
# Golden:    tests/samples/visualize/dash/components/src/ts/graphics/<golden>
# TODO: replace this hardcoded path navigation with a shared helper that maps a
#       components/src/ts/... module path to its tests/samples data directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TESTS_ROOT = os.path.abspath(os.path.join(_HERE, *([os.pardir] * 8)))  # -> tests/
_GOLDEN_DIR = os.path.join(
    _TESTS_ROOT, 'samples', 'visualize', 'dash', 'components', 'src', 'ts',
    'graphics')
GOLDEN_PATH = os.path.join(_GOLDEN_DIR, 'default_camera_parameters.json')
BINARY_GOLDEN_PATH = os.path.join(_GOLDEN_DIR, 'default_camera_parameters.bin')

# Resolution that matches the TS ``defaultCameraParameters`` (512x512).
DEFAULT_RESOLUTION = 512
APPROX_TOL = 1e-4


def _flatten_matrix(matrix):
    """Flatten a 4x4 nested list (or already-flat list) to 16 numbers."""
    flat = []
    for row in matrix:
        if isinstance(row, (list, tuple)):
            flat.extend(row)
        else:
            flat.append(row)
    return flat


def _default_camera_dict():
    return default_camera(DEFAULT_RESOLUTION).cpu().as_dict()


# The goldens are generated (and committed) on the Python side -- they are the
# gold standard for the py<->ts interchange format. First run only: uncomment to
# (re)write the goldens, then re-comment and commit them so subsequent runs (and
# the TS tests) compare against them.
#
# Two goldens are written:
#   * default_camera_parameters.json -- human-readable interchange dict.
#   * default_camera_parameters.bin  -- the same dict through the binary wire
#     format (to_binary), decoded on the TS side via fromBinary. as_dict() is
#     passed to to_binary() directly, with no normalization, so the binary golden
#     reflects the exact types/values produced by the Python camera.
#
# def test_write_golden():
#     from kaolin.visualize.web.io import to_binary
#     os.makedirs(_GOLDEN_DIR, exist_ok=True)
#
#     d = _default_camera_dict()
#     d['extrinsics']['view_matrix'] = [[float(x) for x in row]
#                                       for row in d['extrinsics']['view_matrix']]
#     d['intrinsics'] = {k: (v if isinstance(v, str) else float(v))
#                        for k, v in d['intrinsics'].items()}
#     with open(GOLDEN_PATH, 'w') as f:
#         json.dump(d, f, indent=2)
#         f.write('\n')
#
#     # Pass as_dict() straight into to_binary -- no changes to the camera dict.
#     with open(BINARY_GOLDEN_PATH, 'wb') as f:
#         f.write(to_binary(_default_camera_dict()))


@pytest.fixture(scope='module')
def golden():
    assert os.path.isfile(GOLDEN_PATH), (
        f'Golden file missing: {GOLDEN_PATH}. '
        f'Enable test_write_golden() above to create it.')
    with open(GOLDEN_PATH) as f:
        return json.load(f)


class TestDefaultCameraParity:
    def test_field_conventions(self, golden):
        """The interchange *shape* (keys) must match the golden exactly."""
        actual = _default_camera_dict()

        assert set(actual.keys()) == set(golden.keys()), (
            f'Top-level keys differ:\n  golden={sorted(golden)}\n  actual={sorted(actual)}')

        assert set(actual['intrinsics'].keys()) == set(golden['intrinsics'].keys()), (
            f"intrinsics keys differ:\n  golden={sorted(golden['intrinsics'])}"
            f"\n  actual={sorted(actual['intrinsics'])}")

        assert set(actual['extrinsics'].keys()) == set(golden['extrinsics'].keys()), (
            f"extrinsics keys differ:\n  golden={sorted(golden['extrinsics'])}"
            f"\n  actual={sorted(actual['extrinsics'])}")

    def test_values(self, golden):
        """The interchange *values* must match the golden within tolerance."""
        actual = _default_camera_dict()
        mismatches = []

        for k, gv in golden['intrinsics'].items():
            av = actual['intrinsics'].get(k)
            if isinstance(gv, str):
                if av != gv:
                    mismatches.append(f'intrinsics.{k}: golden={gv!r} actual={av!r}')
            elif av is None or not math.isclose(float(av), float(gv), abs_tol=APPROX_TOL):
                mismatches.append(f'intrinsics.{k}: golden={gv} actual={av}')

        golden_vm = _flatten_matrix(golden['extrinsics']['view_matrix'])
        actual_vm = _flatten_matrix(actual['extrinsics']['view_matrix'])
        assert len(actual_vm) == len(golden_vm) == 16, 'view_matrix should be 4x4'
        for i, (av, gv) in enumerate(zip(actual_vm, golden_vm)):
            if not math.isclose(float(av), float(gv), abs_tol=APPROX_TOL):
                mismatches.append(f'view_matrix[{i}]: golden={gv} actual={av}')

        assert not mismatches, (
            'Python default_camera().as_dict() drifted from the py<->ts golden:\n  '
            + '\n  '.join(mismatches))

    def test_binary_golden_roundtrips_to_camera(self):
        """The binary golden must decode (``from_binary``) and reconstruct
        (:meth:`Camera.from_dict`) back into the default camera.

        This exercises the full binary interchange that the TS side consumes via
        ``fromBinary``: bytes -> ``from_binary`` (decoded dict) -> ``Camera.from_dict``.
        Values are compared with :func:`kaolin.utils.testing.contained_torch_equal`
        (``approximate=True``); we compare ``as_dict()`` representations rather than
        the ``Camera`` objects directly because ``Camera.__eq__`` is an exact (bitwise)
        comparison, whereas the interchange goes through float32 and warrants a tolerance.
        """
        assert os.path.isfile(BINARY_GOLDEN_PATH), (
            f'Binary golden file missing: {BINARY_GOLDEN_PATH}. '
            f'Enable test_write_golden() above to create it.')

        with open(BINARY_GOLDEN_PATH, 'rb') as f:
            decoded = from_binary(f.read())

        actual = Camera.from_dict(decoded)
        expected = default_camera(DEFAULT_RESOLUTION).cpu()

        assert contained_torch_equal(
            actual.as_dict(), expected.as_dict(), approximate=True,
            atol=APPROX_TOL, rtol=0.0, print_error_context='default_camera'), (
            'Camera reconstructed from the binary golden differs from '
            'default_camera().\n  actual=' + repr(actual.as_dict())
            + '\n  expected=' + repr(expected.as_dict()))
