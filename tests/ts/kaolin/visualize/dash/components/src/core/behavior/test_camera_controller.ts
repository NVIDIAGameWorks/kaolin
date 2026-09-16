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
import { CameraControllerBase, isCameraController } from '@kaolin/core/behavior/camera_controller';
import { isSizeableBehavior } from '@kaolin/core/behavior/base';

// Minimal concrete controller: the base supplies dimension tracking and the
// canvas binding; subclasses only wire up native-camera conversion.
class StubController extends CameraControllerBase {
    camera = { fov: 60 };
    /** Sizes the reconciliation hook was invoked with, newest last. */
    changes: Array<{ width: number, height: number }> = [];
    override setCameraParams(_params: any): void {}
    override getCamera(): any { return this.camera; }
    override getCameraParams(): any { return { width: this.width, height: this.height }; }
    protected override onViewportSizeChanged(width: number, height: number): void {
        this.changes.push({ width, height });
    }
}

describe('visualize/dash/components/src/core/behavior/test_camera_controller.ts', () => {

    describe('CameraControllerBase', () => {
        it('tracks viewport dimensions, binds to a canvas, and satisfies the guard', () => {
            const controller = new StubController();
            assert.equal(controller.elementType(), 'canvas', 'controllers bind to a canvas');
            // No DOM is registered here, so devicePixelRatio falls back to 1 and
            // the recorded (device-pixel) size matches the css size handed in.
            controller.setDimensions(640, 480);
            assert.equal(controller.width, 640, 'setDimensions records width');
            assert.equal(controller.height, 480, 'setDimensions records height');
            controller.setDimensions(640, 480);
            assert.deepEqual(controller.changes, [{ width: 640, height: 480 }],
                'the same size again does not reach the reconciliation hook');
            assert.deepEqual(controller.getCameraParams(), { width: 640, height: 480 },
                'subclass conversion sees the tracked dimensions');
            assert.equal(controller.getCamera(), controller.camera, 'getCamera returns the owned native camera');
            assert.isTrue(isCameraController(controller), 'a CameraControllerBase satisfies the guard');
            assert.isTrue(isSizeableBehavior(controller), 'a controller is also a sizeable behavior');
        });
    });

    describe('isCameraController', () => {
        it('rejects values missing the required members', () => {
            assert.isFalse(isCameraController({ getCamera: () => null }), 'a partial implementation is rejected');
        });
    });

});
