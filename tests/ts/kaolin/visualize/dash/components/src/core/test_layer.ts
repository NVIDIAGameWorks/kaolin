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

// NOTE: happy-dom updates a canvas element's `width`/`height` properties but
// does not resize or clear its node-canvas backing surface, so whether painted
// content survived a resize is not observable here (same limitation as
// `test_canvas.ts`). Coverage is therefore the sizing contract, not the pixels.

import { assert } from 'chai';

import { resizeLayer } from '@kaolin/core/layer';
import { CanvasBehavior } from '@kaolin/core/behavior/base';
import { captureConsole } from '@test/helpers/console';
import { registerDom, unregisterDom, makeCanvasElement } from '@test/helpers/dom';

describe('visualize/dash/components/src/core/test_layer.ts', () => {

    before(registerDom);
    after(unregisterDom);

    function checkBufferSize(canvas: HTMLCanvasElement, width: number, height: number, label: string) {
        assert.equal(canvas.width, width, `${label}: buffer width`);
        assert.equal(canvas.height, height, `${label}: buffer height`);
    }

    describe('resizeLayer', () => {
        it('sizes a canvas buffer, whether or not the aspect ratio is preserved', () => {
            const canvas = makeCanvasElement(32, 16);
            assert.equal(window.devicePixelRatio || 1, 1, 'sizes below are buffer sizes, so this assumes dpr 1');
            assert.isTrue(resizeLayer(canvas, 64, 32), 'same-aspect resize succeeds');
            checkBufferSize(canvas, 64, 32, 'same-aspect resize');
            assert.isTrue(resizeLayer(canvas, 50, 40), 'aspect-changing resize succeeds');
            checkBufferSize(canvas, 50, 40, 'aspect-changing resize');
            assert.isTrue(resizeLayer(canvas, 100.9, 80.2), 'fractional resize succeeds');
            checkBufferSize(canvas, 100, 80, 'fractional size is floored');
            assert.isTrue(resizeLayer(canvas, 100, 80), 'a repeat of the current size succeeds');
            checkBufferSize(canvas, 100, 80, 'repeat request leaves the size alone');
        });

        it('resizes an existing WebGL canvas without a 2D context', () => {
            const canvas = makeCanvasElement(32, 16);
            canvas.getContext = (() => null) as any;
            assert.isTrue(resizeLayer(canvas, 64, 32));
            checkBufferSize(canvas, 64, 32, 'non-2D fallback');
        });

        it('preserves directly acquired 2D content and retries a failed snapshot', () => {
            const canvas = makeCanvasElement(32, 16);
            assert.isNotNull(canvas.getContext('2d'));
            const createElement = document.createElement;
            document.createElement = ((tag: string, ...args: any[]) => {
                const element = (createElement as any).call(document, tag, ...args);
                if (tag === 'canvas') element.getContext = () => null;
                return element;
            }) as any;
            const { restore } = captureConsole();
            try {
                assert.isFalse(resizeLayer(canvas, 64, 32));
                checkBufferSize(canvas, 32, 16, 'failure before mutation');
            } finally {
                restore();
                document.createElement = createElement;
            }
            assert.isTrue(resizeLayer(canvas, 64, 32));
            checkBufferSize(canvas, 64, 32, 'successful retry');
        });

        it('uses the 2D preservation path for behaviors sharing one canvas', () => {
            const canvas = makeCanvasElement(32, 16);
            const first = new CanvasBehavior();
            const second = new CanvasBehavior();
            first.setActiveElement(canvas);
            second.setActiveElement(canvas);
            assert.strictEqual(first.ctx, second.ctx);
            let copies = 0;
            const ctx = first.ctx!;
            const drawImage = ctx.drawImage;
            ctx.drawImage = ((...args: any[]) => {
                copies++;
                (drawImage as any).apply(ctx, args);
            }) as any;
            try {
                assert.isTrue(resizeLayer(canvas, 64, 32));
                first.setDimensions(64, 32);
                second.setDimensions(64, 32);
                assert.equal(copies, 1, 'layer content is restored once, independently of behavior count');
            } finally {
                ctx.drawImage = drawImage;
            }
        });

        it('reports a canvas buffer that did not accept its requested dimensions', () => {
            const canvas = makeCanvasElement(32, 16);
            canvas.getContext = (() => null) as any;
            Object.defineProperty(canvas, 'width', { get: () => 32, set: () => {} });
            assert.isFalse(resizeLayer(canvas, 64, 32));
        });

        it('takes css pixels, and scales only a canvas buffer to device pixels', () => {
            const originalRatio = window.devicePixelRatio;
            Object.defineProperty(window, 'devicePixelRatio', { value: 2, configurable: true });
            try {
                const canvas = makeCanvasElement(32, 16);
                assert.isTrue(resizeLayer(canvas, 64, 32), 'resize succeeds');
                checkBufferSize(canvas, 128, 64, 'buffer is the requested css size in device pixels');
                assert.isTrue(resizeLayer(canvas, 64, 32), 'repeating the css size succeeds');
                checkBufferSize(canvas, 128, 64, 'and recognizes the buffer as already that size');

                const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                assert.isTrue(resizeLayer(svg, 64, 32), 'svg resize succeeds');
                assert.equal(svg.getAttribute('width'), '64', 'an svg has no buffer, so it stays in css pixels');
            } finally {
                Object.defineProperty(window, 'devicePixelRatio', { value: originalRatio, configurable: true });
            }
        });

        it('reports a size it cannot use, leaving the canvas untouched', () => {
            const canvas = makeCanvasElement(32, 16);
            const unusable = [[0, 16], [32, -16], [NaN, 16], [32, Infinity]];
            const { calls, restore } = captureConsole();
            try {
                unusable.forEach(([width, height]) => {
                    assert.isFalse(resizeLayer(canvas, width, height), `rejects ${width}x${height}`);
                });
            } finally {
                restore();
            }
            checkBufferSize(canvas, 32, 16, 'canvas keeps its original size');
            assert.equal(calls.length, unusable.length, 'one message per rejected size');
            assert.equal(calls[0].method, 'error', 'routed to console.error');
            assert.include(String(calls[0].args[0]), 'invalid target size', 'message names the problem');
        });

        it('brings an svg element\'s size attributes in line, leaving its viewBox alone', () => {
            const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
            svg.setAttribute('viewBox', '0 0 100 100');
            assert.isTrue(resizeLayer(svg, 640, 480), 'svg resize succeeds');
            assert.equal(svg.getAttribute('width'), '640', 'width attribute follows the request');
            assert.equal(svg.getAttribute('height'), '480', 'height attribute follows the request');
            assert.equal(svg.getAttribute('viewBox'), '0 0 100 100', 'viewBox is left to the owning behavior');
        });

        it('reports that other layer types carry no size of their own', () => {
            const div = document.createElement('div');
            assert.isFalse(resizeLayer(div, 640, 480), 'a div has no size of its own to set');
            assert.equal(div.style.width, '', 'the div is left untouched');
            assert.isFalse(resizeLayer(null, 640, 480), 'a null layer reports no size');
            assert.isFalse(resizeLayer(undefined, 640, 480), 'an undefined layer reports no size');
        });
    });

});
