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
import {
    DrawRemoteImageOptionsSchema,
    DrawRemoteImageBehavior,
} from '@kaolin/lib/behavior/draw_remote_image';
import { isElementBoundBehavior, isMessageHandler } from '@kaolin/core/behavior';
import { logger } from '@kaolin/util/logging';
import { registerDom, unregisterDom, makeCanvasElement } from '@test/helpers/dom';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Attach a fresh 4×4 canvas directly to a behavior (bypasses document lookup).
function makeAttachedBehavior(tag = 'render') {
    const canvas = makeCanvasElement(4, 4);
    const b = new DrawRemoteImageBehavior({ messageTag: tag });
    b.setActiveElement(canvas);
    return { b, canvas };
}

// Spy on logger.warn for the duration of fn(), return captured messages.
function captureWarnings(fn: () => void): string[] {
    const messages: string[] = [];
    const orig = logger.warn.bind(logger);
    logger.warn = (msg: string) => messages.push(msg);
    try { fn(); } finally { logger.warn = orig; }
    return messages;
}

// Read all pixel bytes from a canvas-npm canvas.
function pixels(canvas: any): number[] {
    return Array.from((canvas.getContext('2d') as any).getImageData(0, 0, 4, 4).data as Uint8ClampedArray);
}

// Build a Map payload as the behavior expects: { img: value }.
const imgPayload = (img: any) => new Map([['img', img]]);

// ---------------------------------------------------------------------------

describe('lib/behavior/draw_remote_image', () => {

    // onMessage draws to a canvas; the happy-dom canvas adapter supplies real pixels.
    before(registerDom);
    after(unregisterDom);

    describe('DrawRemoteImageOptionsSchema', () => {
        it('defaults messageTag to "render"', () => {
            assert.equal(DrawRemoteImageOptionsSchema.parse({}).messageTag, 'render');
        });
        it('accepts an explicit messageTag', () => {
            assert.equal(DrawRemoteImageOptionsSchema.parse({ messageTag: 'frame' }).messageTag, 'frame');
        });
        it('rejects non-string messageTag', () => {
            assert.throws(() => DrawRemoteImageOptionsSchema.parse({ messageTag: 42 }));
        });
    });

    describe('DrawRemoteImageBehavior', () => {

        describe('constructor', () => {
            it('is an ElementBoundBehavior', () => {
                assert.isTrue(isElementBoundBehavior(new DrawRemoteImageBehavior()));
            });
            it('is a MessageHandler', () => {
                assert.isTrue(isMessageHandler(new DrawRemoteImageBehavior()));
            });
            it('uses schema default when called with no arguments', () => {
                assert.equal(new DrawRemoteImageBehavior().options.messageTag, 'render');
            });
            it('applies provided options', () => {
                assert.equal(new DrawRemoteImageBehavior({ messageTag: 'custom' }).options.messageTag, 'custom');
            });
        });

        describe('acceptedMessageTags', () => {
            it('returns ["render"] with defaults', () => {
                assert.deepEqual(new DrawRemoteImageBehavior().acceptedMessageTags(), ['render']);
            });
            it('reflects configured messageTag', () => {
                assert.deepEqual(new DrawRemoteImageBehavior({ messageTag: 'frame' }).acceptedMessageTags(), ['frame']);
            });
        });

        describe('onConnectionOpen', () => {
            it('does not throw', () => {
                assert.doesNotThrow(() => new DrawRemoteImageBehavior().onConnectionOpen());
            });
        });

        describe('onMessage', () => {

            it('ignores messages with a non-matching tag', () => {
                const { b, canvas } = makeAttachedBehavior('render');
                const shaped = Object.assign(new Uint8ClampedArray(4 * 4 * 4).fill(255), { shape: [4, 4, 4] });
                b.onMessage('other_tag', imgPayload(shaped));
                assert.isTrue(pixels(canvas).every(v => v === 0), 'canvas should remain blank');
            });

            it('no-ops when no canvas element is attached', () => {
                const b = new DrawRemoteImageBehavior();
                assert.doesNotThrow(() => b.onMessage('render', imgPayload(new Blob())));
            });

            it('no-ops when messageContent has no img field', () => {
                const { b, canvas } = makeAttachedBehavior();
                b.onMessage('render', new Map());
                assert.isTrue(pixels(canvas).every(v => v === 0), 'canvas should remain blank');
            });

            it('draws a typed-array payload to the canvas', () => {
                const { b, canvas } = makeAttachedBehavior();
                const shaped = Object.assign(new Uint8ClampedArray(4 * 4 * 4).fill(255), { shape: [4, 4, 4] });
                b.onMessage('render', imgPayload(shaped));
                assert.isTrue(pixels(canvas).some(v => v > 0), 'canvas should be non-blank after draw');
            });

            it('draws a Blob payload to the canvas', async () => {
                const { b, canvas } = makeAttachedBehavior();
                // Build a fully-opaque source blob via the adapter-backed toBlob.
                const src = makeCanvasElement(4, 4);
                const srcCtx = src.getContext('2d') as any;
                srcCtx.putImageData(srcCtx.createImageData(4, 4), 0, 0);
                srcCtx.fillStyle = '#ffffff';
                srcCtx.fillRect(0, 0, 4, 4);
                const blob = await new Promise<Blob>(resolve => src.toBlob(resolve, 'image/png'));

                b.onMessage('render', imgPayload(blob));
                // drawBlobToCanvas is fire-and-forget inside onMessage; wait for it.
                await new Promise(resolve => setTimeout(resolve, 200));
                assert.isTrue(pixels(canvas).some(v => v > 0), 'canvas should be non-blank after blob draw');
            });

            it('warns for unsupported payload type', () => {
                const { b } = makeAttachedBehavior();
                const msgs = captureWarnings(() => b.onMessage('render', imgPayload({ notABlob: true })));
                assert.isTrue(msgs.some(m => m.includes('unsupported image payload')));
            });
        });

    });

});
