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
import { KonvaDrawingOptionsSchema, KonvaDrawingBehavior } from '@kaolin/lib/behavior/konva_drawing';
import { RecordedInteraction, replayInteractions } from '@kaolin/lib/behavior/record_interactions';
import { imageDataFromCanvas } from '@kaolin/util/canvas';
import { loadPngPixels, pngSize, assertImagesSimilar } from '@test/helpers/image';
import { setupKonvaHarness, KonvaHarness } from '@test/helpers/konva';
import { dashTypescriptTestData } from '@test/helpers/paths';

describe('visualize/dash/components/src/lib/behavior/test_konva_drawing.ts', () => {

    describe('KonvaDrawingBehavior', () => {
        // One flowing usage test: defaults, overrides, and element binding.
        it('constructs from schema, applies overrides, and binds to a div', () => {
            assert.deepEqual(new KonvaDrawingBehavior().options, KonvaDrawingOptionsSchema.parse({}),
                'no-arg construction uses schema defaults');

            const b = new KonvaDrawingBehavior({ mode: 'polygon', color: '#123456', opacity: 0.5 });
            assert.equal(b.options.mode, 'polygon', 'mode override applied');
            assert.equal(b.options.color, '#123456', 'color override applied');
            assert.equal(b.options.opacity, 0.5, 'opacity override applied');

            assert.isTrue(isElementBoundBehavior(b), 'is an element-bound behavior');
            assert.equal(b.elementType(), 'div', 'binds to a div element (Konva stage host)');
        });
    });

    describe('end-to-end interaction replay against golden', () => {
        // Fixtures recorded from examples/tutorial/app/interact2d_main.py: the
        // pointer/click + setOption stream (Save Interactions As) and the
        // hand-checked Konva stage PNG (File -> Download Konva Drawing). Replay is
        // resolution-independent (coords are stored as fractions), so we size the
        // stage to the golden's pixel grid for an exact comparison.
        const sample = (name: string) => dashTypescriptTestData('lib', 'behavior', name);
        let harness: KonvaHarness;

        before(() => { harness = setupKonvaHarness(pngSize(sample('konva_drawing_golden.png')).width); });
        after(async () => { await harness.teardown(); });

        it('reproduces the golden konva drawing from the recorded interactions', async () => {
            const interactions = JSON.parse(
                fs.readFileSync(sample('konva_drawing_interactions.json'), 'utf-8')) as RecordedInteraction[];

            // Plain defaults: the recorded stream opens by setting every option,
            // so the behavior is driven entirely by the replayed setOption entries.
            const behavior = new KonvaDrawingBehavior();
            const container = harness.container('konva-draw-stage');
            harness.attach(behavior, container);

            replayInteractions(behavior, interactions);

            // Flatten the stage to a SIZExSIZE canvas (pixelRatio 1) and compare.
            const stage = (behavior as unknown as { stage: any }).stage;
            const canvas = stage.toCanvas({ pixelRatio: 1 }) as HTMLCanvasElement;

            // Debug: dump the replayed stage to inspect it by hand.
            fs.writeFileSync('/tmp/konva_drawing.png', (canvas as any).toBuffer());

            // Tolerances mirror the brush-drawing golden test: the box / polygon /
            // freeform shapes match the committed golden except for sub-pixel AA at
            // edges where the Chrome-rendered golden and node-canvas replay disagree.
            // Re-tune once the real fixtures land if needed.
            const actual = imageDataFromCanvas(canvas)!.data;
            const expected = await loadPngPixels(sample('konva_drawing_golden.png'));
            assertImagesSimilar(actual, expected, { channelTolerance: 8, maxFracDiff: 0.01 },
                'replayed konva drawing should match the golden image');
        });
    });

});
