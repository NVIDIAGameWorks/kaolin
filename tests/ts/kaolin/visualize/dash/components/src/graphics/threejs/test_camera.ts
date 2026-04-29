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
// Tests for @kaolin/graphics/threejs/camera
//
// These functions convert between Kaolin's camera representation (produced on
// the Python side) and three.js cameras. Tests are split into:
//   * implemented "basic checks" (structural / type / error contracts), and
//   * pending placeholders (`it('...')` with no body) for numerical and
//     round-trip behavior that still needs to be written.
//
// ── PYTHON INTEROP CONVENTIONS (the part that changes if Python changes) ─────
//
// Camera data originates on the Python side from
//     kaolin.render.camera.Camera.as_dict()      (kaolin/render/camera/camera.py)
// then travels to the browser (JSON / decoded binary) and arrives as a
// `CameraParameters` object (see @kaolin/graphics/camera).
//
// Python-produced dict shape  --  KEEP THE FIXTURES BELOW IN SYNC WITH THIS:
//
//   {
//     "intrinsics": {            # PinholeIntrinsics._as_dict() + classname
//        "width":   int,
//        "height":  int,
//        "focal_x": float,       # focal length in pixels
//        "focal_y": float,
//        "x0": float,            # principal point offset from the canvas CENTER
//        "y0": float,            #   (0 == centered; this is NOT a top-left origin)
//        "near": float,
//        "far":  float,
//        "classname": "pinhole" | "orthographic"
//     },
//     "extrinsics": {            # CameraExtrinsics.as_dict()
//        "view_matrix": number[4][4]   # world->camera ("view") matrix, emitted
//     }                                #   by numpy .tolist() as nested rows.
//   }
//
// Conventions worth pinning down in tests:
//   * view_matrix is world-to-camera (a.k.a. world2cam). The TS layer flattens
//     it row-by-row (see flattenMatrix) and feeds three.js via
//     `Matrix4.fromArray(...).transpose()`, i.e. it is treated as ROW-major math.
//   * focal length -> three.js vertical fov: fov = 2*atan(height/(2*focal_y)).
//   * principal point (x0, y0) offsets are NOT yet applied for pinhole cameras
//     (camera.ts logs a warning); round-tripping a non-zero offset is lossy.
// =============================================================================

import { assert } from 'chai';
import * as fs from 'fs';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { createCanvas, loadImage, Canvas } from 'canvas';

import {
    CameraParameters,
    CameraExtrinsics,
    CameraPinholeIntrinsics,
    CameraOrthoIntrinsics,
    defaultCameraParameters,
    cameraParametersApproxEqual,
} from '@kaolin/graphics/camera';
import {
    intrinsicsPinholeToThreeCamera,
    intrinsicsOrthoToThreeCamera,
    intrinsicsToThreeCamera,
    kaolinCameraToThree,
    threePerspectiveCameraToKaolin,
    threeCameraToKaolin,
    defaultCamera,
} from '@kaolin/graphics/threejs/camera';
import { flattenMatrix, convertAllMapsToObjects } from '@kaolin/util/types';
import { toBinary, fromBinary } from '@kaolin/core/io';
import { sampleData, dashTypescriptTestData } from '@test/helpers/paths';

// -----------------------------------------------------------------------------
// Fixtures mirroring Python `Camera.as_dict()` output. Update these if the
// Python serialization (intrinsics keys, classname strings, matrix layout)
// changes.
// -----------------------------------------------------------------------------

const PINHOLE_INTRINSICS: CameraPinholeIntrinsics = {
    width: 640,
    height: 480,
    focal_x: 500,
    focal_y: 500,
    x0: 0,
    y0: 0,
    near: 0.1,
    far: 1000,
    classname: 'pinhole',
};

const ORTHO_INTRINSICS: CameraOrthoIntrinsics = {
    width: 640,
    height: 480,
    near: 0.1,
    far: 1000,
    classname: 'orthographic',
};

// Identity-ish view matrix (world->camera), as Python would emit via tolist().
const PINHOLE_CAMERA_PARAMS: CameraParameters = {
    intrinsics: PINHOLE_INTRINSICS,
    extrinsics: {
        view_matrix: [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -5],
            [0, 0, 0, 1],
        ],
    },
};

/**
 * Thin assertion wrapper around `cameraParametersApproxEqual` (the comparison
 * logic now lives in graphics/camera.ts) with a descriptive failure message.
 */
function assertCameraParametersEqual(
    actual: CameraParameters,
    expected: CameraParameters,
    eps: number = 1e-4,
) {
    assert.isTrue(
        cameraParametersApproxEqual(actual, expected, eps),
        `Camera parameters differ beyond eps=${eps}.\n` +
        `  actual:   intrinsics=${JSON.stringify(actual.intrinsics)} ` +
        `view_matrix=${Array.from(flattenMatrix(actual.extrinsics.view_matrix))}\n` +
        `  expected: intrinsics=${JSON.stringify(expected.intrinsics)} ` +
        `view_matrix=${Array.from(flattenMatrix(expected.extrinsics.view_matrix))}`,
    );
}

// -----------------------------------------------------------------------------

describe('visualize/dash/components/src/graphics/threejs/test_camera.ts', () => {
    describe('intrinsicsPinholeToThreeCamera', () => {
        it('returns a PerspectiveCamera with aspect/near/far/fov from intrinsics', () => {
            const cam = intrinsicsPinholeToThreeCamera(PINHOLE_INTRINSICS);

            assert.instanceOf(cam, THREE.PerspectiveCamera);
            assert.approximately(cam.aspect, PINHOLE_INTRINSICS.width / PINHOLE_INTRINSICS.height, 1e-6);
            assert.equal(cam.near, PINHOLE_INTRINSICS.near);
            assert.equal(cam.far, PINHOLE_INTRINSICS.far);

            const expectedFov =
                2 * Math.atan(PINHOLE_INTRINSICS.height / (2 * PINHOLE_INTRINSICS.focal_y)) * (180 / Math.PI);
            assert.approximately(cam.fov, expectedFov, 1e-4);
        });

        // TODO: assert principal-point offset (x0, y0) is applied once supported
        //       (currently camera.ts only logs a warning).
        it('applies principal point offset (x0, y0) once supported');
    });

    describe('intrinsicsOrthoToThreeCamera', () => {
        it('returns an OrthographicCamera with matching near/far', () => {
            const cam = intrinsicsOrthoToThreeCamera(ORTHO_INTRINSICS);

            assert.instanceOf(cam, THREE.OrthographicCamera);
            assert.equal(cam.near, ORTHO_INTRINSICS.near);
            assert.equal(cam.far, ORTHO_INTRINSICS.far);
        });

        // TODO: verify left/right/top/bottom frustum and projection against a
        //       known Kaolin orthographic camera (camera.ts marks this unverified).
        it('produces a frustum matching the Kaolin orthographic projection');
    });

    describe('intrinsicsToThreeCamera', () => {
        it('dispatches on classname to the correct camera type', () => {
            assert.instanceOf(intrinsicsToThreeCamera(PINHOLE_INTRINSICS), THREE.PerspectiveCamera);
            assert.instanceOf(intrinsicsToThreeCamera(ORTHO_INTRINSICS), THREE.OrthographicCamera);
        });

        it('throws on an unsupported classname', () => {
            const bad = { ...PINHOLE_INTRINSICS, classname: 'fisheye' } as CameraPinholeIntrinsics;
            assert.throws(() => intrinsicsToThreeCamera(bad), /Unsupported intrinsics type/);
        });
    });

    describe('kaolinCameraToThree', () => {
        it('returns a three.js camera for valid pinhole parameters', () => {
            const cam = kaolinCameraToThree(PINHOLE_CAMERA_PARAMS);
            assert.instanceOf(cam, THREE.PerspectiveCamera);
        });

        // TODO: assert the resulting camera world matrix equals inverse(view_matrix)
        //       for a known, non-trivial Python-produced view matrix.
        it('places the camera at the world transform implied by view_matrix');

        it('accepts all supported view_matrix encodings', () => {
            // One non-trivial 4x4 view matrix (rotation about Y + translation),
            // expressed in every encoding allowed by `CameraExtrinsics.view_matrix`.
            const flat = [
                Math.cos(Math.PI / 4), 0, Math.sin(Math.PI / 4), 0,
                0, 1, 0, 0,
                -Math.sin(Math.PI / 4), 0, Math.cos(Math.PI / 4), -5,
                0, 0, 0, 1,
            ];
            const nested = [flat.slice(0, 4), flat.slice(4, 8), flat.slice(8, 12), flat.slice(12, 16)];

            const encodings: Array<[string, CameraExtrinsics['view_matrix']]> = [
                ['number[] (flat 16)', flat],
                ['Float32Array (flat 16)', new Float32Array(flat)],
                ['number[][] (nested 4x4)', nested],
                ['Float32Array[] (typed-array rows)', nested.map(r => new Float32Array(r))],
                ['number[][][] (batched nested)', [nested]],
            ];

            const worldElements = (view: CameraExtrinsics['view_matrix']): number[] => {
                const cam = kaolinCameraToThree({
                    intrinsics: PINHOLE_INTRINSICS,
                    extrinsics: { view_matrix: view },
                });
                cam.updateMatrixWorld(true);
                return Array.from(cam.matrixWorld.elements);
            };

            // All encodings of the same matrix must produce the same three.js camera.
            const reference = worldElements(new Float32Array(flat));
            for (const [name, encoding] of encodings) {
                const elements = worldElements(encoding);
                for (let i = 0; i < 16; ++i) {
                    assert.approximately(elements[i], reference[i], 1e-5,
                        `matrixWorld[${i}] differs for view_matrix encoding "${name}"`);
                }
            }
        });
    });

    describe('threePerspectiveCameraToKaolin', () => {
        it('recovers focal length and pinhole intrinsics from a PerspectiveCamera', () => {
            const width = 500;
            const height = 500;
            const cam = new THREE.PerspectiveCamera(45, width / height, 0.001, 1000);
            cam.position.set(5, 0, 0);
            cam.lookAt(0, 0, 0);

            const params = threePerspectiveCameraToKaolin(cam, width, height);

            assert.equal(params.intrinsics.classname, 'pinhole');
            assert.equal(params.intrinsics.width, width);
            assert.equal(params.intrinsics.height, height);
            assert.equal(params.intrinsics.near, cam.near);
            assert.equal(params.intrinsics.far, cam.far);

            const intr = params.intrinsics as CameraPinholeIntrinsics;
            const expectedFocalY = height / (2 * Math.tan(((cam.fov * Math.PI) / 180) / 2));
            assert.approximately(intr.focal_y, expectedFocalY, 1e-3);
            assert.approximately(intr.focal_x, intr.focal_y, 1e-3);

            const viewMatrix = params.extrinsics.view_matrix as Float32Array;
            assert.instanceOf(viewMatrix, Float32Array);
            assert.equal(viewMatrix.length, 16);
        });

        // TODO: assert the emitted view_matrix matches the Python row-major
        //       convention element-for-element against a known camera pose.
        it('emits a view_matrix in the Python row-major convention');

        // TODO: map three.js view offset back to Kaolin (x0, y0) principal point.
        it('recovers principal point (x0, y0) from a three.js view offset');
    });

    describe('threeCameraToKaolin', () => {
        it('throws for a non-perspective camera', () => {
            const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
            assert.throws(() => threeCameraToKaolin(ortho, 100, 100), /Unsupported camera type/);
        });

        // TODO: support / test OrthographicCamera -> Kaolin once implemented.
        it('converts an OrthographicCamera to Kaolin orthographic parameters');
    });

    // Ported from the now-deleted tests/ts/kaolin/visualize/dash/util/test_graphics.ts
    // (its `@kaolin/util/graphics` import path no longer exists).
    describe('camera round-trip (Python <-> three.js)', () => {
        it('converts a three.js camera to verified camera parameters', () => {
            const near = 0.001;
            const far = 1000;
            const camera = new THREE.PerspectiveCamera(45, 1, near, far);
            camera.position.set(5, 0, 0);
            camera.lookAt(0, 0, 0);

            const width = 500;
            const height = 500;
            const convertedParams = threeCameraToKaolin(camera, width, height);
            const expectedViewMatrix = new Float32Array(
                [0., 0., -1., 0.,
                 0., 1., 0., 0.,
                 1., -0., -0., -5.,
                 0., 0., 0., 1.]);
            const expectedIntrinsics = {
                width, height,
                focal_x: 603.5534057617188, focal_y: 603.5534057617188,
                x0: 0, y0: 0, near, far, classname: 'pinhole',
            };
            const expected = { intrinsics: expectedIntrinsics, extrinsics: { view_matrix: expectedViewMatrix } };
            assertCameraParametersEqual(convertedParams, expected);

            // Also test round trip
            const cameraReconstructed = kaolinCameraToThree(convertedParams);
            assert.instanceOf(cameraReconstructed, THREE.PerspectiveCamera,
                'Reconstructed camera should be a THREE.PerspectiveCamera');

            const convertedParamsRoundtrip = threeCameraToKaolin(cameraReconstructed, width, height);
            assertCameraParametersEqual(convertedParamsRoundtrip, expected);
        });

        it('round-trips the default camera parameters through three.js', () => {
            const originalParams = defaultCameraParameters;

            const threeCamera = kaolinCameraToThree(originalParams);
            assert.isDefined(threeCamera, 'Three.js camera should be defined');

            const reconstructed = threeCameraToKaolin(
                threeCamera,
                originalParams.intrinsics.width,
                originalParams.intrinsics.height,
            );
            assert.isDefined(reconstructed, 'Reconstructed parameters should be defined');
            assertCameraParametersEqual(originalParams, reconstructed);
        });

        it('round-trips camera parameters with custom intrinsics', () => {
            const originalParams: CameraParameters = {
                intrinsics: {
                    width: 640,
                    height: 480,
                    focal_x: 500,
                    focal_y: 500,
                    x0: 0,
                    y0: 0,
                    near: 0.1,
                    far: 1000,
                    classname: 'pinhole',
                },
                extrinsics: {
                    view_matrix: new Float32Array([
                        Math.cos(Math.PI / 4), 0, Math.sin(Math.PI / 4), 0,
                        0, 1, 0, 0,
                        -Math.sin(Math.PI / 4), 0, Math.cos(Math.PI / 4), -5,
                        0, 0, 0, 1]),
                },
            };

            const threeCamera = kaolinCameraToThree(originalParams);
            const reconstructed = threeCameraToKaolin(
                threeCamera,
                originalParams.intrinsics.width,
                originalParams.intrinsics.height,
            );
            assert.isDefined(reconstructed, 'Reconstructed parameters should be defined');
            assertCameraParametersEqual(reconstructed, originalParams);
        });
    });

    // Round-trip that exercises the binary wire format (toBinary/fromBinary) on top
    // of the three.js conversion. Two binaries are produced from the SAME starting
    // kaolin camera:
    //   * Path A: kaolinCamera                       -> toBinary
    //   * Path B: kaolinCamera -> three -> toKaolin  -> toBinary
    // Both are decoded with fromBinary and compared with the camera-parameters
    // comparison helper, proving the three.js round-trip survives binary transport.
    describe('camera round-trip through the binary wire format (encode -> decode)', () => {
        // Decoded binary payloads come back as a Map tree with typed-array leaves;
        // normalize to the plain CameraParameters shape the comparison helper expects.
        function decodeParams(buffer: ArrayBuffer): CameraParameters {
            return convertAllMapsToObjects(fromBinary(buffer)) as CameraParameters;
        }

        async function assertBinaryRoundTrip(original: CameraParameters) {
            const { width, height } = original.intrinsics;

            // Path A: encode the original kaolin camera directly.
            const binDirect = await toBinary(original);
            assert.isNotNull(binDirect, 'toBinary(original) returned null');

            // Path B: original -> three.js -> kaolin -> encode.
            const roundTripped = threeCameraToKaolin(kaolinCameraToThree(original), width, height);
            const binRoundTrip = await toBinary(roundTripped);
            assert.isNotNull(binRoundTrip, 'toBinary(roundTripped) returned null');

            // Decode both and compare the decoded camera parameters.
            assertCameraParametersEqual(decodeParams(binRoundTrip!), decodeParams(binDirect!));
        }

        it('matches direct vs three.js-round-tripped encodings (custom pinhole)', async () => {
            await assertBinaryRoundTrip(PINHOLE_CAMERA_PARAMS);
        });

        it('matches direct vs three.js-round-tripped encodings (default camera)', async () => {
            await assertBinaryRoundTrip(defaultCameraParameters);
        });
    });

    describe('defaultCamera', () => {
        it('returns a positioned PerspectiveCamera', () => {
            const cam = defaultCamera();
            assert.instanceOf(cam, THREE.PerspectiveCamera);
            assert.isAbove(cam.position.length(), 0);
        });
    });

    // =============================================================================
    // Render parity reference (silhouette).
    //
    // Renders the fox mesh with the camera produced from `defaultCameraParameters`
    // (graphics/camera.ts) and compares the resulting silhouette to a committed
    // reference PNG. The reference is the cross-framework golden consumed by the
    // Python parity test:
    //   tests/python/.../graphics/threejs/test_camera_parity.py
    //
    // We render a lighting-independent SILHOUETTE (which pixels are covered by the
    // mesh), not shaded color: we project each triangle through the three.js camera
    // and fill it on a 2D canvas. This exercises the camera conversion math in a
    // deterministic, dependency-light way (no WebGL / GPU / display required).
    // =============================================================================

    const FOX_OBJ = sampleData('meshes', 'fox.obj');
    const REFERENCE_PNG = dashTypescriptTestData('graphics', 'threejs', 'camera_default_fox.png');

    /** Project the fox triangles through `camera` and fill a white-on-black silhouette. */
    function renderFoxSilhouette(camera: THREE.Camera, width: number, height: number): Canvas {
        const group = new OBJLoader().parse(fs.readFileSync(FOX_OBJ, 'utf-8'));
        camera.updateMatrixWorld(true);

        const canvas = createCanvas(width, height);
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, width, height);
        ctx.fillStyle = '#ffffff';

        const ndc = new THREE.Vector3();
        const cam = new THREE.Vector3();
        group.traverse((obj: any) => {
            if (!obj.isMesh) { return; }
            obj.updateWorldMatrix(true, false);
            const pos = obj.geometry.getAttribute('position');
            const world = obj.matrixWorld;
            for (let i = 0; i < pos.count; i += 3) {
                const px: number[] = [];
                const py: number[] = [];
                let behind = false;
                for (let k = 0; k < 3; ++k) {
                    const v = new THREE.Vector3().fromBufferAttribute(pos, i + k).applyMatrix4(world);
                    // Camera looks down -z; drop triangles touching/behind the camera plane.
                    cam.copy(v).applyMatrix4(camera.matrixWorldInverse);
                    if (cam.z >= 0) { behind = true; break; }
                    ndc.copy(v).project(camera);
                    px.push((ndc.x * 0.5 + 0.5) * width);
                    py.push((1 - (ndc.y * 0.5 + 0.5)) * height);
                }
                if (behind) { continue; }
                ctx.beginPath();
                ctx.moveTo(px[0], py[0]);
                ctx.lineTo(px[1], py[1]);
                ctx.lineTo(px[2], py[2]);
                ctx.closePath();
                ctx.fill();
            }
        });
        return canvas;
    }

    /** Boolean coverage mask (true == covered) from a canvas, thresholding luminance. */
    function silhouetteMask(canvas: Canvas): Uint8Array {
        const { width, height } = canvas;
        const data = canvas.getContext('2d').getImageData(0, 0, width, height).data;
        const mask = new Uint8Array(width * height);
        for (let p = 0; p < mask.length; ++p) {
            mask[p] = data[p * 4] > 127 ? 1 : 0; // red channel; white fill on black bg
        }
        return mask;
    }

    function intersectionOverUnion(a: Uint8Array, b: Uint8Array): number {
        let inter = 0;
        let union = 0;
        for (let i = 0; i < a.length; ++i) {
            if (a[i] || b[i]) { union++; if (a[i] && b[i]) { inter++; } }
        }
        return union === 0 ? 1 : inter / union;
    }

    describe('fox silhouette render parity (default camera)', () => {
        const width = defaultCameraParameters.intrinsics.width;
        const height = defaultCameraParameters.intrinsics.height;

        it('renders a plausible fox silhouette and matches the reference', async () => {
            const camera = kaolinCameraToThree(defaultCameraParameters);
            const canvas = renderFoxSilhouette(camera, width, height);
            const mask = silhouetteMask(canvas);

            // Sanity: the fox should cover a sensible, non-degenerate fraction.
            const coverage = mask.reduce((s, v) => s + v, 0) / mask.length;
            assert.isAbove(coverage, 0.02, 'Silhouette is too small; camera/mesh likely wrong');
            assert.isBelow(coverage, 0.9, 'Silhouette fills almost everything; camera/mesh likely wrong');

            // First run only: uncomment to write the reference PNG, then re-comment
            // and commit it so subsequent runs (and the Python parity test) compare.
            //
            // fs.writeFileSync(REFERENCE_PNG, canvas.toBuffer('image/png'));

            assert.isTrue(fs.existsSync(REFERENCE_PNG),
                `Reference image missing: ${REFERENCE_PNG}. Enable the write line above to create it.`);
            const refImg = await loadImage(REFERENCE_PNG);
            const refCanvas = createCanvas(width, height);
            refCanvas.getContext('2d').drawImage(refImg, 0, 0, width, height);
            const refMask = silhouetteMask(refCanvas);

            const iou = intersectionOverUnion(mask, refMask);
            assert.isAbove(iou, 0.99,
                `Rendered fox silhouette diverged from reference (IoU=${iou.toFixed(4)}). ` +
                `If camera.ts changed intentionally, regenerate ${REFERENCE_PNG}.`);
        });
    });
});
