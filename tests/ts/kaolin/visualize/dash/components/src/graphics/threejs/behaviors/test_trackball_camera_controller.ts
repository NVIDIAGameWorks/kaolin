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

import { assert } from 'chai';

import { isCameraController, isElementBoundBehavior } from '@kaolin/core/behavior';
import { cameraParametersApproxEqual, defaultCameraParameters } from '@kaolin/graphics/camera';
import {
    TrackballCameraController, TrackballCameraControllerOptionsSchema,
} from '@kaolin/graphics/threejs/behaviors/trackball_camera_controller';
import { dragPointer, makeControllerCanvas, wheel } from '@test/helpers/camera_controller';
import { registerDom, unregisterDom } from '@test/helpers/dom';

describe('visualize/dash/components/src/graphics/threejs/behaviors/test_trackball_camera_controller.ts', () => {
    before(registerDom);
    after(unregisterDom);

    describe('TrackballCameraController', () => {
        it('constructs from schema, applies overrides, and binds to a canvas', () => {
            assert.deepEqual(new TrackballCameraController().options, TrackballCameraControllerOptionsSchema.parse({}),
                'no arguments uses the schema defaults');
            const c = new TrackballCameraController({ up: [0, 1, 0], rotateSpeed: 2 });
            assert.equal(c.options.rotateSpeed, 2, 'override applied');
            assert.deepEqual(c.camera.up.toArray(), [0, 1, 0], 'camera up follows the up option');
            assert.deepEqual(new TrackballCameraController().camera.up.toArray(), [0, 0, 1], 'Kaolin scenes are +Z up');
            assert.throws(() => new TrackballCameraController({ up: [0, 1] }), /up/);
            assert.equal(c.elementType(), 'canvas', 'binds to a canvas');
            assert.isTrue(isElementBoundBehavior(c), 'is an element-bound behavior');
            assert.isTrue(isCameraController(c), 'is a camera controller');
        });

        it('creates controls on binding, applies option updates, and disposes', () => {
            const c = new TrackballCameraController();
            assert.isNull(c.controls, 'no controls before binding');
            c.setActiveElement(makeControllerCanvas());
            const controls = c.controls!;
            assert.isNotNull(controls, 'binding creates controls');
            assert.isFalse(controls.noRotate, 'rotation enabled by default');
            assert.isTrue(controls.staticMoving, 'movement is applied immediately, without damping');

            c.setOption('enableRotate', false);
            c.setOption('rotateSpeed', 3);
            assert.isTrue(controls.noRotate, 'enableRotate=false maps to the inverted noRotate flag');
            assert.equal(controls.rotateSpeed, 3, 'speed update reaches the live controls');

            c.setActive(false);
            assert.isFalse(controls.enabled, 'setActive(false) disables the controls');

            c.setCameraParams(defaultCameraParameters);
            assert.notStrictEqual(c.controls, controls, 'new camera params rebuild the controls');
            c.dispose();
            assert.isNull(c.controls, 'dispose releases the controls');
        });

        it('round-trips camera parameters and responds to pointer and wheel input', () => {
            const c = new TrackballCameraController();
            const canvas = makeControllerCanvas(512, 512);
            c.setDimensions(512, 512);
            c.setCameraParams(defaultCameraParameters);
            assert.isTrue(cameraParametersApproxEqual(c.getCameraParams(), defaultCameraParameters),
                'getCameraParams returns what setCameraParams was given');

            // Binding makes the controls re-aim the camera at its target using the
            // configured up vector, which may re-roll it. The position is unchanged.
            const bound = c.camera.position.clone();
            c.setActiveElement(canvas);
            assert.closeTo(c.camera.position.distanceTo(bound), 0, 1e-5, 'binding does not move the camera');

            const start = c.camera.position.clone();
            dragPointer(canvas, [256, 256], [320, 280]);
            c.onAnimate();
            assert.isAbove(c.camera.position.distanceTo(start), 0.1, 'a drag rotates the camera');

            // No onAnimate() here: the controller's own wheel listener must apply the zoom.
            const distance = c.camera.position.length();
            wheel(canvas, -100);
            assert.isBelow(c.camera.position.length(), distance, 'scrolling in moves the camera closer immediately');

            const before = c.camera.position.clone();
            c.setOption('enableRotate', false);
            dragPointer(canvas, [256, 256], [320, 280]);
            c.onAnimate();
            assert.closeTo(c.camera.position.distanceTo(before), 0, 1e-5, 'a drag does nothing once rotation is disabled');
        });
    });
});
