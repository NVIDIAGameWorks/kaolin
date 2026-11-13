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

import { PlaycanvasBehavior, PlaycanvasSplatBehavior } from '@kaolin/graphics/playcanvas/behaviors/samples';
import { resizeLayer } from '@kaolin/core/layer';
import { registerDom, unregisterDom } from '@test/helpers/dom';

describe('visualize/dash/graphics/playcanvas/behaviors/test_samples.ts', () => {
    before(registerDom);
    after(unregisterDom);

    let originalRatio: number;
    beforeEach(() => {
        originalRatio = window.devicePixelRatio;
        Object.defineProperty(window, 'devicePixelRatio', { value: 2, configurable: true });
    });
    afterEach(() => {
        Object.defineProperty(window, 'devicePixelRatio', { value: originalRatio, configurable: true });
    });

    function makeApp() {
        const canvas = document.createElement('canvas');
        const device = {
            canvas,
            maxPixelRatio: 1,
            events: [] as Array<[string, number, number]>,
            fire(event: string, width: number, height: number) {
                this.events.push([event, width, height]);
            },
            get width() { return canvas.width; },
            get height() { return canvas.height; },
            resizeCanvas(width: number, height: number) {
                const bufferWidth = Math.floor(width * this.maxPixelRatio);
                const bufferHeight = Math.floor(height * this.maxPixelRatio);
                if (canvas.width !== bufferWidth || canvas.height !== bufferHeight) {
                    canvas.width = bufferWidth;
                    canvas.height = bufferHeight;
                    this.fire('resizecanvas', bufferWidth, bufferHeight);
                }
            },
        };
        return { graphicsDevice: device };
    }

    it('synchronizes one device shared by two behaviors without duplicate notifications', () => {
        const app = makeApp();
        const behaviors = [new PlaycanvasBehavior(), new PlaycanvasSplatBehavior()];
        behaviors.forEach(behavior => { behavior.app = app as any; });
        resizeLayer(app.graphicsDevice.canvas, 200, 100);
        behaviors.forEach(behavior => behavior.setDimensions(200, 100));
        assert.deepEqual(app.graphicsDevice.events, [['resizecanvas', 400, 200]]);
    });

    for (const BehaviorType of [PlaycanvasBehavior, PlaycanvasSplatBehavior]) {
        describe(BehaviorType.name, () => {
            it('applies a size requested before the graphics device existed', () => {
                const behavior = new BehaviorType();
                behavior.setDimensions(200, 100);
                const app = makeApp();
                behavior.app = app as any;
                behavior.setDimensions(200, 100);
                assert.equal(app.graphicsDevice.width, 400);
                assert.equal(app.graphicsDevice.height, 200);
                assert.equal(app.graphicsDevice.canvas.style.width, '');
            });

            it('notifies the device once after centralized sizing, including a DPR change', () => {
                const behavior = new BehaviorType();
                const app = makeApp();
                behavior.app = app as any;
                const device = app.graphicsDevice;
                assert.isTrue(resizeLayer(device.canvas, 200, 100));
                behavior.setDimensions(200, 100);
                behavior.setDimensions(200, 100);
                assert.deepEqual(device.events, [['resizecanvas', 400, 200]]);
                Object.defineProperty(window, 'devicePixelRatio', { value: 1.25, configurable: true });
                assert.isTrue(resizeLayer(device.canvas, 200, 100));
                behavior.setDimensions(200, 100);
                assert.deepEqual(device.events, [['resizecanvas', 400, 200], ['resizecanvas', 250, 125]]);
            });

            it('does not duplicate native resize notifications', () => {
                const behavior = new BehaviorType();
                const app = makeApp();
                behavior.app = app as any;
                behavior.setDimensions(200, 100);
                behavior.setDimensions(200, 100);
                assert.deepEqual(app.graphicsDevice.events, [['resizecanvas', 400, 200]]);
            });

            it('retries a failed resize notification at the same size', () => {
                const behavior = new BehaviorType();
                const app = makeApp();
                behavior.app = app as any;
                const device = app.graphicsDevice;
                resizeLayer(device.canvas, 200, 100);
                const fire = device.fire;
                device.fire = () => { throw new Error('notification failed'); };
                assert.throws(() => behavior.setDimensions(200, 100), 'notification failed');
                device.fire = fire;
                behavior.setDimensions(200, 100);
                assert.deepEqual(device.events, [['resizecanvas', 400, 200]]);
            });

            it('updates the buffer when DPR changes at the same CSS size', () => {
                const behavior = new BehaviorType();
                const app = makeApp();
                behavior.app = app as any;
                behavior.setDimensions(200, 100);
                Object.defineProperty(window, 'devicePixelRatio', { value: 1.25, configurable: true });
                behavior.setDimensions(200, 100);
                assert.equal(app.graphicsDevice.width, 250);
                assert.equal(app.graphicsDevice.height, 125);
            });

            it('allows retrying the same size after the device throws', () => {
                const behavior = new BehaviorType();
                const app = makeApp();
                behavior.app = app as any;
                const resize = app.graphicsDevice.resizeCanvas;
                app.graphicsDevice.resizeCanvas = () => { throw new Error('device resize failed'); };
                assert.throws(() => behavior.setDimensions(200, 100), 'device resize failed');
                app.graphicsDevice.resizeCanvas = resize;
                behavior.setDimensions(200, 100);
                assert.equal(app.graphicsDevice.width, 400);
                assert.equal(app.graphicsDevice.height, 200);
            });

            it('reports a device that did not apply the requested buffer size', () => {
                const behavior = new BehaviorType();
                const app = makeApp();
                behavior.app = app as any;
                app.graphicsDevice.resizeCanvas = () => {};
                assert.throws(() => behavior.setDimensions(200, 100), 'could not resize its drawing buffer');
            });
        });
    }
});
