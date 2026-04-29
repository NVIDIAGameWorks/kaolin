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
