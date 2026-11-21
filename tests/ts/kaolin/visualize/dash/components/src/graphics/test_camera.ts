// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// =============================================================================
// Framework-independent tests for @kaolin/graphics/camera
//
// graphics/camera.ts holds the pure (three.js-independent) camera data model
// that is exchanged with the Python side: the CameraParameters interfaces and
// the `defaultCameraParameters` constant. These tests deliberately do NOT import
// three.js -- three.js-specific conversion lives in graphics/threejs/test_camera.ts.
//
// ── PYTHON INTEROP / GOLDEN ──────────────────────────────────────────────────
// `defaultCameraParameters` must stay byte-for-byte compatible with the Python
// default camera (kaolin.render.easy_render.default_camera(512).as_dict()).
// The golden below encodes that expected interchange format. The matching
// Python-side check lives in:
//   tests/python/.../graphics/test_default_camera_parity.py
//
// Golden files (shared with the Python parity test):
//   tests/samples/visualize/dash/components/src/ts/graphics/default_camera_parameters.json
//   tests/samples/visualize/dash/components/src/ts/graphics/default_camera_parameters.bin
// The .bin golden is written on the Python side via `kaolin.visualize.web.io.to_binary`
// and decoded here via `fromBinary`, exercising the binary wire format end-to-end.
// =============================================================================

import { assert } from 'chai';
import * as fs from 'fs';

import {
    CameraParameters,
    CameraPinholeIntrinsics,
    defaultCameraParameters,
    cameraParametersApproxEqual,
} from '@kaolin/graphics/camera';
import { flattenMatrix, convertAllMapsToObjects } from '@kaolin/util/types';
import { fromBinary } from '@kaolin/core/io';
import { dashTypescriptTestData } from '@test/helpers/paths';

const GOLDEN_PATH = dashTypescriptTestData('graphics', 'default_camera_parameters.json');
const BINARY_GOLDEN_PATH = dashTypescriptTestData('graphics', 'default_camera_parameters.bin');

const APPROX_TOL = 1e-4;

describe('visualize/dash/components/src/graphics/test_camera.ts', () => {
    describe('defaultCameraParameters (shape / field conventions)', () => {
        it('has the expected top-level and intrinsics/extrinsics keys', () => {
            assert.hasAllKeys(defaultCameraParameters, ['intrinsics', 'extrinsics']);
            assert.hasAllKeys(defaultCameraParameters.intrinsics,
                ['width', 'height', 'focal_x', 'focal_y', 'x0', 'y0', 'near', 'far', 'classname']);
            assert.hasAllKeys(defaultCameraParameters.extrinsics, ['view_matrix']);
        });

        it('is a pinhole camera with a 16-element view matrix', () => {
            assert.equal(defaultCameraParameters.intrinsics.classname, 'pinhole');
            assert.equal(flattenMatrix(defaultCameraParameters.extrinsics.view_matrix).length, 16);
        });
    });

    describe('defaultCameraParameters golden parity (py <-> ts interchange)', () => {
        // The golden is the gold standard produced on the PYTHON side, NOT here:
        //   kaolin.render.easy_render.default_camera(512).as_dict()
        // To (re)generate and commit it, use the commented golden-writer block in:
        //   tests/python/kaolin/visualize/dash/components/src/ts/graphics/test_default_camera_parity.py
        // This TS test only consumes that golden to verify the TS defaults still match.

        function loadGolden(): { intrinsics: Record<string, any>; extrinsics: { view_matrix: number[][] } } {
            assert.isTrue(fs.existsSync(GOLDEN_PATH),
                `Golden file missing: ${GOLDEN_PATH}. Enable the "writes the golden" block above to create it.`);
            return JSON.parse(fs.readFileSync(GOLDEN_PATH, 'utf-8'));
        }

        it('matches the golden field conventions (keys)', () => {
            const golden = loadGolden();
            const intr = defaultCameraParameters.intrinsics as CameraPinholeIntrinsics;

            const goldenKeys = Object.keys(golden.intrinsics).sort();
            const actualKeys = Object.keys(intr).sort();
            assert.deepEqual(actualKeys, goldenKeys,
                `Intrinsics keys differ from golden.\n  golden: ${goldenKeys}\n  actual: ${actualKeys}`);

            assert.deepEqual(Object.keys(golden.extrinsics).sort(), ['view_matrix']);
        });

        it('matches the golden values (intrinsics + view matrix)', () => {
            const golden = loadGolden();
            const intr = defaultCameraParameters.intrinsics as CameraPinholeIntrinsics;

            const mismatches: string[] = [];
            for (const [k, gv] of Object.entries(golden.intrinsics)) {
                const av = (intr as any)[k];
                if (typeof gv === 'number') {
                    if (Math.abs(av - gv) > APPROX_TOL) {
                        mismatches.push(`intrinsics.${k}: golden=${gv} actual=${av}`);
                    }
                } else if (av !== gv) {
                    mismatches.push(`intrinsics.${k}: golden=${gv} actual=${av}`);
                }
            }

            const goldenFlat = flattenMatrix(golden.extrinsics.view_matrix as any);
            const actualFlat = flattenMatrix(defaultCameraParameters.extrinsics.view_matrix);
            for (let i = 0; i < 16; ++i) {
                if (Math.abs(actualFlat[i] - goldenFlat[i]) > APPROX_TOL) {
                    mismatches.push(`view_matrix[${i}]: golden=${goldenFlat[i]} actual=${actualFlat[i]}`);
                }
            }

            assert.isEmpty(mismatches, `defaultCameraParameters drifted from golden:\n  ${mismatches.join('\n  ')}`);
        });
    });

    describe('defaultCameraParameters binary golden (py to_binary -> ts fromBinary)', () => {
        // The .bin golden is the gold standard produced on the PYTHON side, NOT here:
        //   kaolin.visualize.web.io.to_binary(default_camera(512).as_dict())
        // To (re)generate and commit it, use the commented golden-writer block in:
        //   tests/python/kaolin/visualize/dash/components/src/ts/graphics/test_default_camera_parity.py
        // This TS test only consumes that golden to verify the binary wire format round-trips
        // back into the TS `defaultCameraParameters`.

        function readArrayBuffer(p: string): ArrayBuffer {
            const buffer = fs.readFileSync(p);
            return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
        }

        // py camera -> as_dict() -> to_binary() -> fromBinary already yields the
        // CameraParameters shape (intrinsics object + a view matrix as typed-array rows).
        // convertAllMapsToObjects is the only normalization (decoded dicts are Maps); the
        // Float32Array view-matrix rows are respected (flattenMatrix handles them).
        function decodeBinaryGolden(): CameraParameters {
            return convertAllMapsToObjects(fromBinary(readArrayBuffer(BINARY_GOLDEN_PATH))) as CameraParameters;
        }

        it('decodes to the expected field conventions (keys)', () => {
            assert.isTrue(fs.existsSync(BINARY_GOLDEN_PATH),
                `Binary golden missing: ${BINARY_GOLDEN_PATH}. `
                + 'Enable the binary golden-writer block in test_default_camera_parity.py to create it.');
            const decoded = decodeBinaryGolden();
            assert.deepEqual(Object.keys(decoded).sort(), ['extrinsics', 'intrinsics']);
            assert.deepEqual(Object.keys(decoded.intrinsics).sort(),
                ['classname', 'far', 'focal_x', 'focal_y', 'height', 'near', 'width', 'x0', 'y0']);
            assert.deepEqual(Object.keys(decoded.extrinsics).sort(), ['view_matrix']);
        });

        it('decodes to values matching defaultCameraParameters', () => {
            assert.isTrue(fs.existsSync(BINARY_GOLDEN_PATH),
                `Binary golden missing: ${BINARY_GOLDEN_PATH}. `
                + 'Enable the binary golden-writer block in test_default_camera_parity.py to create it.');
            const decoded = decodeBinaryGolden();
            assert.isTrue(
                cameraParametersApproxEqual(decoded, defaultCameraParameters, APPROX_TOL),
                'Binary golden decoded params differ from defaultCameraParameters.\n'
                + `  intrinsics=${JSON.stringify(decoded.intrinsics)}\n`
                + `  view_matrix=${flattenMatrix(decoded.extrinsics.view_matrix)}`);
        });
    });

    describe('cameraParametersApproxEqual', () => {
        // Deep-ish clone with the view matrix normalized to a flat, mutable array.
        function cloneParams(p: CameraParameters): CameraParameters {
            return {
                intrinsics: { ...(p.intrinsics as any) },
                extrinsics: { view_matrix: Array.from(flattenMatrix(p.extrinsics.view_matrix)) },
            };
        }

        it('returns true for identical parameters', () => {
            assert.isTrue(cameraParametersApproxEqual(defaultCameraParameters, defaultCameraParameters));
        });

        it('is encoding-agnostic for the view matrix (Float32Array vs nested vs flat)', () => {
            const flat = Array.from(flattenMatrix(defaultCameraParameters.extrinsics.view_matrix));
            const nested = [flat.slice(0, 4), flat.slice(4, 8), flat.slice(8, 12), flat.slice(12, 16)];
            const asNested = cloneParams(defaultCameraParameters);
            asNested.extrinsics.view_matrix = nested;
            assert.isTrue(cameraParametersApproxEqual(defaultCameraParameters, asNested));
        });

        it('returns false when an intrinsics value drifts beyond eps', () => {
            const perturbed = cloneParams(defaultCameraParameters);
            (perturbed.intrinsics as CameraPinholeIntrinsics).focal_x += 0.5;
            assert.isFalse(cameraParametersApproxEqual(defaultCameraParameters, perturbed));
        });

        it('returns false when a view-matrix element drifts beyond eps', () => {
            const perturbed = cloneParams(defaultCameraParameters);
            (perturbed.extrinsics.view_matrix as number[])[0] += 0.5;
            assert.isFalse(cameraParametersApproxEqual(defaultCameraParameters, perturbed));
        });

        it('returns false when a non-numeric field (classname) differs', () => {
            const perturbed = cloneParams(defaultCameraParameters);
            (perturbed.intrinsics as CameraPinholeIntrinsics).classname = 'orthographic';
            assert.isFalse(cameraParametersApproxEqual(defaultCameraParameters, perturbed));
        });

        it('respects a custom epsilon', () => {
            const perturbed = cloneParams(defaultCameraParameters);
            (perturbed.intrinsics as CameraPinholeIntrinsics).focal_x += 0.5;
            assert.isFalse(cameraParametersApproxEqual(defaultCameraParameters, perturbed, 1e-4));
            assert.isTrue(cameraParametersApproxEqual(defaultCameraParameters, perturbed, 1.0));
        });
    });

});
