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
import { spy } from 'sinon';

import { isElementBoundBehavior } from '@kaolin/core/behavior';
import { KonvaSelectionOptionsSchema, KonvaSelectionBehavior } from '@kaolin/lib/behavior/konva_selection';
import { RecordedInteraction, replayInteractions } from '@kaolin/lib/behavior/record_interactions';
import { imageDataFromCanvas } from '@kaolin/util/canvas';
import { loadPngPixels, pngSize, assertImagesSimilar } from '@test/helpers/image';
import { setupKonvaHarness, KonvaHarness } from '@test/helpers/konva';
import { dashTypescriptTestData } from '@test/helpers/paths';

describe('visualize/dash/components/src/lib/behavior/test_konva_selection.ts', () => {

    describe('KonvaSelectionBehavior', () => {
        // One flowing usage test. `compositeCanvasId` is a required option (no
        // schema default), so every construction supplies it.
        it('constructs from schema, applies overrides, and binds to a div', () => {
            const id = 'composite';
            assert.deepEqual(new KonvaSelectionBehavior({ compositeCanvasId: id }).options,
                KonvaSelectionOptionsSchema.parse({ compositeCanvasId: id }),
                'construction with only the required id uses schema defaults for the rest');

            const b = new KonvaSelectionBehavior(
                { compositeCanvasId: id, action: 'subtract', mode: 'polygon' });
            assert.equal(b.options.compositeCanvasId, id, 'compositeCanvasId applied');
            assert.equal(b.options.action, 'subtract', 'action override applied');
            assert.equal(b.options.mode, 'polygon', 'mode override (inherited option) applied');

            assert.isTrue(isElementBoundBehavior(b), 'is an element-bound behavior');
            assert.equal(b.elementType(), 'div', 'binds to a div element (Konva stage host)');
        });
    });

    describe('end-to-end interaction replay against golden', () => {
        // Fixtures recorded from examples/tutorial/app/interact2d_main.py: the
        // pointer/click + setOption stream (Save Interactions As) and the
        // hand-checked composite PNG (File -> Download Selection). Replay is
        // resolution-independent (coords are stored as fractions), so we size the
        // stage / composite to the golden's pixel grid for an exact comparison.
        const sample = (name: string) => dashTypescriptTestData('lib', 'behavior', name);
        const COMPOSITE_ID = 'selection-composite';
        let harness: KonvaHarness;

        before(() => { harness = setupKonvaHarness(pngSize(sample('konva_selection_golden.png')).width); });
        after(async () => { await harness.teardown(); });

        it('resolves a registered node-canvas composite target by id (duck-typed)', () => {
            const composite = harness.compositeCanvas(COMPOSITE_ID);
            const behavior = new KonvaSelectionBehavior({ compositeCanvasId: COMPOSITE_ID });
            assert.strictEqual(behavior.resolveCompositeCanvas(), composite,
                'resolveCompositeCanvas returns the canvas-like element registered under the id');
        });

        it('reproduces the golden selection composite from the recorded interactions', async () => {
            const interactions = JSON.parse(
                fs.readFileSync(sample('konva_selection_interactions.json'), 'utf-8')) as RecordedInteraction[];

            // The composite target must share Konva's node-canvas backend so the
            // stage->composite drawImage works; the harness registers it so the
            // behavior resolves it via document.getElementById.
            const composite = harness.compositeCanvas(COMPOSITE_ID);

            // compositeCanvasId is set at construction (not uiBound, so not in the
            // recorded stream); everything else is driven by the replayed options.
            const behavior = new KonvaSelectionBehavior({ compositeCanvasId: COMPOSITE_ID });
            const container = harness.container('konva-selection-stage');
            harness.attach(behavior, container);

            replayInteractions(behavior, interactions);

            // Debug: dump the composited mask to inspect it by hand.
            fs.writeFileSync('/tmp/konva_selection.png', (composite as any).toBuffer());

            // The composite holds the flattened white mask; compare to the golden.
            // Tolerances mirror the other golden tests (sub-pixel edge AA between
            // the Chrome golden and node-canvas replay); re-tune once fixtures land.
            const actual = imageDataFromCanvas(composite)!.data;
            const expected = await loadPngPixels(sample('konva_selection_golden.png'));
            assertImagesSimilar(actual, expected, { channelTolerance: 8, maxFracDiff: 0.01 },
                'replayed selection composite should match the golden image');
        });
    });

});


describe('KonvaSelectionBehavior resized masks', () => {
    let harness: KonvaHarness;
    beforeEach(() => { harness = setupKonvaHarness(100); });
    afterEach(async () => { await harness.teardown(); });

    for (const dpr of [1.25, 2]) {
        for (const action of ['add', 'subtract', 'intersect'] as const) {
            it(`${action} aligns the resized mask at DPR ${dpr}`, () => {
                Object.defineProperty(window, 'devicePixelRatio', { value: dpr, configurable: true });
                const composite = harness.compositeCanvas('mask', 80 * dpr, 80 * dpr);
                const behavior = new KonvaSelectionBehavior({ compositeCanvasId: 'mask', mode: 'box', action });
                harness.attach(behavior, harness.container('selection'));
                behavior.setActive(true);
                behavior.setDimensions(80, 80);
                // The viewer resizes the composite layer before resizing the behavior.
                composite.width = composite.height = 100 * dpr;
                behavior.setDimensions(100, 100);
                const rasterize = spy((behavior as any).stage, 'toCanvas');
                const ctx = composite.getContext('2d')!;
                ctx.fillStyle = 'white';
                ctx.fillRect(10 * dpr, 10 * dpr, 40 * dpr, 40 * dpr);
                const event = (x: number, y: number) => ({ nativeEvent: { clientX: x, clientY: y, pointerId: 1 } } as any);
                behavior.onPointerDown(event(30, 30));
                behavior.onPointerMove(event(70, 70));
                behavior.onPointerUp(event(70, 70));

                const raster = rasterize.returnValues[0];
                assert.equal(raster.width, composite.width, 'raster is exported at device resolution');
                assert.equal(raster.height, composite.height);
                const alpha = (x: number, y: number) => ctx.getImageData(Math.floor(x * dpr), Math.floor(y * dpr), 1, 1).data[3];
                assert.equal(alpha(20, 20), action === 'intersect' ? 0 : 255, 'old mask only');
                assert.equal(alpha(40, 40), action === 'subtract' ? 0 : 255, 'overlap');
                assert.equal(alpha(60, 60), action === 'add' ? 255 : 0, 'new mask only');
                assert.equal(alpha(85, 85), 0, 'outside both masks');
                assert.equal(ctx.globalCompositeOperation, 'source-over');
                behavior.setActiveElement(null as any);
            });
        }
    }
});
