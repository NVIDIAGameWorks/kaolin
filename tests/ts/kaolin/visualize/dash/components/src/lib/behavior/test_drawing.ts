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

import * as fs from 'fs';

import { assert } from 'chai';

import { DrawingOptionsSchema, DrawingBehavior } from '@kaolin/lib/behavior/drawing';
import { isElementBoundBehavior } from '@kaolin/core/behavior';
import { RecordedInteraction, replayInteractions } from '@kaolin/lib/behavior/record_interactions';
import { imageDataFromCanvas } from '@kaolin/util/canvas';
import { registerDom, unregisterDom, makeCanvasElement } from '@test/helpers/dom';
import { loadPngPixels, assertImagesSimilar } from '@test/helpers/image';
import { dashTypescriptTestData } from '@test/helpers/paths';

describe('visualize/dash/components/src/lib/behavior/test_drawing.ts', () => {

    describe('DrawingBehavior', () => {
        // One flowing usage test: defaults, overrides, canvas binding, and reset.
        it('constructs from schema, applies overrides, is canvas-bound, and reset stops painting', () => {
            assert.deepEqual(new DrawingBehavior().options, DrawingOptionsSchema.parse({}),
                'no-arg construction uses schema defaults');

            const b = new DrawingBehavior({ color: '#123456', thickness: 20, mode: 'erase' });
            assert.equal(b.options.color, '#123456', 'color override applied');
            assert.equal(b.options.thickness, 20, 'thickness override applied');
            assert.equal(b.options.mode, 'erase', 'mode override applied');

            assert.isTrue(isElementBoundBehavior(b), 'is an element-bound behavior');
            assert.equal(b.elementType(), 'canvas', 'binds to a canvas element');

            b.isPainting = true;
            b.lastPoint = { x: 1, y: 2 };
            b.reset();
            assert.isFalse(b.isPainting as boolean, 'reset clears isPainting');
            assert.isNull(b.lastPoint, 'reset clears lastPoint');
        });
    });

    describe('end-to-end interaction replay against golden', () => {
        // Fixtures recorded from examples/tutorial/app/interact2d_main.py: the
        // pointer/click + setOption stream and the hand-checked output PNG.
        const sample = (name: string) => dashTypescriptTestData('lib', 'behavior', name);
        const SIZE = 600;  // intrinsic canvas size; the golden PNG is 600x600.
        let rafOriginal: typeof window.requestAnimationFrame;

        before(() => {
            registerDom();
            // DrawingBehavior defers stroke segments to requestAnimationFrame; run
            // the callbacks synchronously so each replayed move paints immediately
            // and in recorded order.
            rafOriginal = window.requestAnimationFrame;
            (window as any).requestAnimationFrame = (cb: FrameRequestCallback) => { cb(0); return 0; };
        });

        after(async () => {
            (window as any).requestAnimationFrame = rafOriginal;
            await unregisterDom();
        });

        it('reproduces the golden brush drawing from the recorded interactions', async () => {
            const interactions = JSON.parse(
                fs.readFileSync(sample('drawing_interactions.json'), 'utf-8')) as RecordedInteraction[];

            const canvas = makeCanvasElement(SIZE, SIZE);
            // happy-dom has no layout, so getBoundingClientRect would be all zeros.
            // Give the canvas a fixed on-screen rect: the display size cancels out
            // (replay maps frac -> clientX with this width; getCanvasCoordinates
            // divides by it), so any non-zero size reproduces the recorded
            // canvas-pixel coordinates.
            const rect = {
                left: 0, top: 0, right: SIZE, bottom: SIZE, width: SIZE, height: SIZE,
                x: 0, y: 0, toJSON: () => ({}),
            } as DOMRect;
            canvas.getBoundingClientRect = () => rect;

            // The recorded stream opens by setting every brush option, so the
            // behavior starts from plain schema defaults and is driven entirely by
            // the replayed setOption entries (no hand-seeded initial options).
            const behavior = new DrawingBehavior();
            behavior.setActiveElement(canvas);

            replayInteractions(behavior, interactions);

            // Debug: dump the replayed canvas to inspect it by hand.
            // const blob = await blobFromCanvas(canvas, 'png');
            // fs.writeFileSync('/tmp/f.png', Buffer.from(await blob!.arrayBuffer()));

            // The replay matches the committed golden almost exactly: even at an
            // exact (0/channel) comparison only ~0.4% of pixels differ, all of them
            // anti-aliased stroke edges where the Chrome-rendered golden and the
            // node-canvas replay disagree by a sub-pixel. The tolerances absorb that
            // edge AA but nothing structural (a wrong stroke color/position dirties
            // several percent of pixels and fails).
            const actual = imageDataFromCanvas(canvas)!.data;
            const expected = await loadPngPixels(sample('drawing_golden.png'));
            assertImagesSimilar(actual, expected, { channelTolerance: 8, maxFracDiff: 0.01 },
                'replayed drawing should match the golden image');
        });
    });

});
