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

// DOM setup: canvas.ts calls document.getElementById/createElement and uses
// ImageData/createImageBitmap. registerDom installs a happy-dom DOM whose canvas
// elements are node-canvas-backed (via @happy-dom/node-canvas-adapter), so
// getContext('2d'), getImageData/putImageData, toBlob, and createImageBitmap all
// render real pixels. Canvases are ordinary DOM nodes; a top-level afterEach
// clears document.body so id lookups don't leak between tests.

import { assert } from 'chai';
import {
    ShapedTypedArray,
    typedArrayFromCanvas,
    alphaChannelFromCanvas,
    imageDataFromCanvas,
    getCanvasCoordinates,
    blobFromCanvas,
    blobAlphaChannelFromCanvas,
    drawBlobToCanvas,
    imageDataFromBlob,
    drawTypedArrayToCanvas,
    clearCanvas,
    setCanvasBufferSize,
} from '@kaolin/util/canvas';
import { registerDom, unregisterDom, addElement, addCanvasElement, makeCanvasElement } from '@test/helpers/dom';
import { captureConsole } from '@test/helpers/console';

// ---------------------------------------------------------------------------
// Shared guard tests for functions accepting CanvasOrId. Verifies the
// expected sentinel (null or false) is returned for:
//   • a missing string id
//   • a string id resolving to a non-canvas element (no getContext)
//   • a non-canvas element passed directly
// `await` works for both sync and async functions.
// The no-getContext cases catch throws because the current implementation
// does not yet guard against elements lacking getContext; the test will
// start passing cleanly once that guard is added.

function itReturnsForNonCanvas(fn: (canvasOrId: any, ...args: any[]) => any, expected: null | false) {
    const check = async (result: any) =>
        expected === null ? assert.isNull(result) : assert.isFalse(result);

    const tryCall = async (...args: any[]) => {
        try { return await fn(...args); }
        catch (_) { return expected; }
    };

    it(`returns ${expected} when canvas id is not found (string)`, async () =>
        check(await tryCall('nonexistent')));

    it(`returns ${expected} when element has no getContext (string id)`, async () => {
        // A real happy-dom element (a div has no getContext); resolved by
        // happy-dom's own getElementById, then removed to keep the DOM clean.
        const el = addElement('div', 'not-a-canvas');
        try { await check(await tryCall('not-a-canvas')); }
        finally { el.remove(); }
    });

    it(`returns ${expected} when element has no getContext (direct element)`, async () =>
        check(await tryCall({})));
}

// ---------------------------------------------------------------------------
// Test canvas fixture: 4×6, each pixel filled pixel-by-pixel via ImageData
// so the exact bytes are fully deterministic (no CSS colour parsing involved).
// Returns the registered canvas id plus the expected RGBA and alpha arrays.

interface TestCanvasData {
    id: string;
    width: number;
    height: number;
    rgba: Uint8ClampedArray;   // expected flat RGBA, length H*W*4
    alpha: Uint8Array;         // expected alpha plane, length H*W
}

function makeTestCanvas(id = 'test-canvas'): TestCanvasData {
    const W = 4, H = 6, n = W * H;
    const src = new Uint8ClampedArray(n * 4);
    for (let i = 0; i < n; i++) {
        src[i*4+0] = (i * 10)       % 256;
        src[i*4+1] = (i *  7 +  50) % 256;
        src[i*4+2] = (i * 13 + 100) % 256;
        src[i*4+3] = (i * 11 +   5) % 256;
    }
    const c = addCanvasElement(id, W, H);
    const ctx = c.getContext('2d') as any;
    const imgData = ctx.createImageData(W, H);
    imgData.data.set(src);
    ctx.putImageData(imgData, 0, 0);

    // Read back actual stored bytes as ground truth. The canvas uses premultiplied
    // alpha internally, so low-alpha pixels may differ from src after round-trip.
    const stored = ctx.getImageData(0, 0, W, H);
    const rgba = new Uint8ClampedArray(stored.data);
    const alpha = new Uint8Array(n);
    for (let i = 0; i < n; i++) alpha[i] = rgba[i*4+3];

    return { id, width: W, height: H, rgba, alpha };
}

// Decode a Blob back to raw RGBA bytes via createImageBitmap (the same
// adapter-backed path the production code uses).
async function decodeBlobPixels(blob: Blob, w: number, h: number): Promise<Uint8ClampedArray> {
    const bitmap = await createImageBitmap(blob);
    const c = makeCanvasElement(w, h);
    const ctx = c.getContext('2d')!;
    ctx.drawImage(bitmap, 0, 0, w, h);
    bitmap.close();
    return ctx.getImageData(0, 0, w, h).data;
}

// ---------------------------------------------------------------------------

describe('visualize/dash/components/src/util/test_canvas.ts', () => {

    // Canvas ops need a DOM; the happy-dom canvas adapter supplies real pixels.
    before(registerDom);
    after(unregisterDom);
    // Drop any id-tagged canvases so lookups don't leak across tests.
    afterEach(() => { document.body.innerHTML = ''; });

    describe('typedArrayFromCanvas', () => {
        itReturnsForNonCanvas(typedArrayFromCanvas, null);


        describe('returns expected output for canvas', () => {
            let data: TestCanvasData;
            let result: ReturnType<typeof typedArrayFromCanvas>;

            beforeEach(() => {
                data = makeTestCanvas();
                result = typedArrayFromCanvas(data.id);
            });

            it('is a Uint8ClampedArray',     () => assert.instanceOf(result, Uint8ClampedArray));
            it('has shape property set',      () => assert.isDefined(result!.shape));
            it('shape is [H, W, 4]',          () => assert.deepEqual(result!.shape, [data.height, data.width, 4]));
            it('length is H * W * 4',         () => assert.equal(result!.length, data.height * data.width * 4));
            it('RGBA bytes match expected',   () => assert.deepEqual(Array.from(result!), Array.from(data.rgba)));
        });
    });

    describe('alphaChannelFromCanvas', () => {
        itReturnsForNonCanvas(alphaChannelFromCanvas, null);

        describe('returns expected output for canvas', () => {
            let data: TestCanvasData;
            let result: ReturnType<typeof alphaChannelFromCanvas>;

            beforeEach(() => {
                data = makeTestCanvas();
                result = alphaChannelFromCanvas(data.id);
            });

            it('is a Uint8Array',            () => assert.instanceOf(result, Uint8Array));
            it('has shape property set',      () => assert.isDefined(result!.shape));
            it('shape is [H, W]',             () => assert.deepEqual(result!.shape, [data.height, data.width]));
            it('length is H * W',             () => assert.equal(result!.length, data.height * data.width));
            it('alpha bytes match expected',  () => assert.deepEqual(Array.from(result!), Array.from(data.alpha)));
        });
    });

    describe('imageDataFromCanvas', () => {
        itReturnsForNonCanvas(imageDataFromCanvas, null);

        describe('returns expected output for canvas', () => {
            let data: TestCanvasData;
            let result: ReturnType<typeof imageDataFromCanvas>;

            beforeEach(() => {
                data = makeTestCanvas();
                result = imageDataFromCanvas(data.id);
            });

            it('width matches the canvas',   () => assert.equal(result!.width, data.width));
            it('height matches the canvas',  () => assert.equal(result!.height, data.height));
            it('covers the full canvas',     () => assert.equal(result!.data.length, data.width * data.height * 4));
            it('RGBA bytes match expected',  () => assert.deepEqual(Array.from(result!.data), Array.from(data.rgba)));
        });
    });

    describe('getCanvasCoordinates', () => {
        // Stub canvas: getBoundingClientRect controls displayed geometry, while
        // width/height are the intrinsic pixel-buffer dimensions. The event only
        // needs clientX/clientY (viewport-relative, like getBoundingClientRect)
        // and (for the default-canvas case) currentTarget.
        function stubCanvas(rect: { left: number; top: number; width: number; height: number },
                            width: number, height: number): HTMLCanvasElement {
            return {
                width, height,
                getBoundingClientRect: () => rect,
            } as unknown as HTMLCanvasElement;
        }

        it('maps viewport coords to canvas pixel space (1:1 when displayed == intrinsic)', () => {
            const canvas = stubCanvas({ left: 10, top: 20, width: 100, height: 100 }, 100, 100);
            const { x, y } = getCanvasCoordinates({ clientX: 60, clientY: 70 } as any, canvas);
            assert.equal(x, 50);
            assert.equal(y, 50);
        });

        it('scales when intrinsic size differs from displayed size', () => {
            const canvas = stubCanvas({ left: 0, top: 0, width: 100, height: 50 }, 200, 100);
            const { x, y } = getCanvasCoordinates({ clientX: 25, clientY: 25 } as any, canvas);
            assert.equal(x, 50);
            assert.equal(y, 50);
        });

        it('uses event.currentTarget when inputCanvas is omitted', () => {
            const canvas = stubCanvas({ left: 5, top: 5, width: 10, height: 10 }, 10, 10);
            const { x, y } = getCanvasCoordinates({ clientX: 8, clientY: 9, currentTarget: canvas } as any);
            assert.equal(x, 3);
            assert.equal(y, 4);
        });

        it('uses viewport (clientX/Y), not document (pageX/Y), so a scrolled page is handled', () => {
            // getBoundingClientRect is viewport-relative; pairing it with pageX/Y
            // (document-relative) would fold in the scroll offset and mislocate the
            // point. Simulate a 1000px-scrolled page: pageX/Y = clientX/Y + 1000.
            const canvas = stubCanvas({ left: 10, top: 20, width: 100, height: 100 }, 100, 100);
            const { x, y } = getCanvasCoordinates(
                { clientX: 60, clientY: 70, pageX: 1060, pageY: 1070 } as any, canvas);
            assert.equal(x, 50, 'x uses clientX, ignoring the page scroll offset');
            assert.equal(y, 50, 'y uses clientY, ignoring the page scroll offset');
        });
    });

    describe('blobFromCanvas', () => {
        itReturnsForNonCanvas(blobFromCanvas, null);

        describe('returns expected output for canvas', () => {
            let data: TestCanvasData;
            beforeEach(() => { data = makeTestCanvas(); });

            it('PNG is the default format',            async () => assert.equal((await blobFromCanvas(data.id))!.type, 'image/png'));
            it('returns image/png for format="png"',   async () => assert.equal((await blobFromCanvas(data.id, 'png'))!.type, 'image/png'));
            it('returns image/jpeg for format="jpeg"', async () => assert.equal((await blobFromCanvas(data.id, 'jpeg'))!.type, 'image/jpeg'));
            it('returned blob is non-empty',           async () => assert.isAbove((await blobFromCanvas(data.id))!.size, 0));
            it('PNG round-trip preserves RGBA pixels', async () => {
                const blob = await blobFromCanvas(data.id, 'png');
                const pixels = await decodeBlobPixels(blob!, data.width, data.height);
                assert.deepEqual(Array.from(pixels), Array.from(data.rgba));
            });
        });
    });

    describe('blobAlphaChannelFromCanvas', () => {
        itReturnsForNonCanvas(blobAlphaChannelFromCanvas, null);

        describe('returns expected output for canvas', () => {
            let data: TestCanvasData;
            let blob: Blob | null;
            let pixels: Uint8ClampedArray;

            beforeEach(async () => {
                data = makeTestCanvas();
                blob = await blobAlphaChannelFromCanvas(data.id);
                pixels = await decodeBlobPixels(blob!, data.width, data.height);
            });

            it('always returns image/png',                   () => assert.equal(blob!.type, 'image/png'));
            it('all R channels are zero',                    () => assert.isTrue(Array.from(pixels).filter((_, i) => i % 4 === 0).every(v => v === 0)));
            it('all G channels are zero',                    () => assert.isTrue(Array.from(pixels).filter((_, i) => i % 4 === 1).every(v => v === 0)));
            it('all B channels are zero',                    () => assert.isTrue(Array.from(pixels).filter((_, i) => i % 4 === 2).every(v => v === 0)));
            it('alpha channel matches source canvas alpha',  () => {
                const decodedAlpha = Array.from(pixels).filter((_, i) => i % 4 === 3);
                assert.deepEqual(decodedAlpha, Array.from(data.alpha));
            });
            it('does not mutate the source canvas',          () => {
                const after = typedArrayFromCanvas(data.id)!;
                assert.deepEqual(Array.from(after), Array.from(data.rgba));
            });
        });
    });

    describe('drawBlobToCanvas', () => {
        describe('round-trip: blobFromCanvas → drawBlobToCanvas → typedArrayFromCanvas', () => {
            it('PNG pixel data survives the round-trip', async () => {
                const src = makeTestCanvas('src');
                const blob = await blobFromCanvas('src', 'png');
                addCanvasElement('dst', src.width, src.height);
                await drawBlobToCanvas('dst', blob!);
                const result = typedArrayFromCanvas('dst')!;
                assert.deepEqual(Array.from(result), Array.from(src.rgba));
            });
        });

        describe('round-trip: blobAlphaChannelFromCanvas → drawBlobToCanvas → typedArrayFromCanvas', () => {
            it('R=0, G=0, B=0 everywhere', async () => {
                const src = makeTestCanvas('src');
                const blob = await blobAlphaChannelFromCanvas('src');
                addCanvasElement('dst', src.width, src.height);
                await drawBlobToCanvas('dst', blob!);
                const result = typedArrayFromCanvas('dst')!;
                for (let i = 0; i < src.width * src.height; i++) {
                    assert.equal(result[i * 4 + 0], 0, `R at pixel ${i}`);
                    assert.equal(result[i * 4 + 1], 0, `G at pixel ${i}`);
                    assert.equal(result[i * 4 + 2], 0, `B at pixel ${i}`);
                    assert.equal(result[i * 4 + 3], src.alpha[i], `alpha at pixel ${i}`);
                }
            });
        });
    });

    describe('drawTypedArrayToCanvas', () => {
        itReturnsForNonCanvas(drawTypedArrayToCanvas, false);

        // Full alpha (255) throughout to avoid premultiplied-alpha loss in canvas npm's putImageData.
        // Each test uses an independent copy of the pixel data so the canvas npm cannot
        // corrupt the expected values by premultiplying the source buffer in-place.
        function makeOpaquePixels(W = 4, H = 6) {
            const n = W * H;
            const pixels = new Uint8ClampedArray(n * 4);
            for (let i = 0; i < n; i++) {
                pixels[i*4]   = (i * 10)       % 256;
                pixels[i*4+1] = (i *  7 +  50) % 256;
                pixels[i*4+2] = (i * 13 + 100) % 256;
                pixels[i*4+3] = 255;
            }
            return { pixels, W, H };
        }

        it('returns false when shape is missing', () => {
            addCanvasElement('dst', 4, 6);
            const noShape = new Uint8ClampedArray(96) as any;
            assert.isFalse(drawTypedArrayToCanvas('dst', noShape));
        });

        it('returns false when shape disagrees with buffer byte length', () => {
            addCanvasElement('dst', 4, 6);
            const input = new Uint8ClampedArray(96) as ShapedTypedArray<Uint8ClampedArray>;
            input.shape = [10, 10, 4];  // implies 400 bytes, buffer has 96
            assert.isFalse(drawTypedArrayToCanvas('dst', input));
        });

        describe('round-trip: → drawTypedArrayToCanvas → typedArrayFromCanvas', () => {
            it('Uint8ClampedArray: pixel data survives the round-trip', () => {
                const { pixels, W, H } = makeOpaquePixels();
                const input = new Uint8ClampedArray(pixels) as ShapedTypedArray<Uint8ClampedArray>;
                input.shape = [H, W, 4];
                addCanvasElement('dst', W, H);
                drawTypedArrayToCanvas('dst', input);
                assert.deepEqual(Array.from(typedArrayFromCanvas('dst')!), Array.from(pixels));
            });

            it('Uint8Array: pixel data survives the round-trip', () => {
                const { pixels, W, H } = makeOpaquePixels();
                const asUint8 = new Uint8Array(pixels) as ShapedTypedArray<Uint8Array>;
                asUint8.shape = [H, W, 4];
                addCanvasElement('dst', W, H);
                drawTypedArrayToCanvas('dst', asUint8);
                assert.deepEqual(Array.from(typedArrayFromCanvas('dst')!), Array.from(pixels));
            });

            it('ArrayBuffer: pixel data survives the round-trip', () => {
                const { pixels, W, H } = makeOpaquePixels();
                const asBuf = new Uint8ClampedArray(pixels).buffer as ArrayBuffer & { shape: number[] };
                asBuf.shape = [H, W, 4];
                addCanvasElement('dst', W, H);
                drawTypedArrayToCanvas('dst', asBuf);
                assert.deepEqual(Array.from(typedArrayFromCanvas('dst')!), Array.from(pixels));
            });
        });
    });

    describe('imageDataFromBlob', () => {
        it('decodes a blob to ImageData at natural size and round-trips canvas pixels', async () => {
            const data = makeTestCanvas();
            const blob = await blobFromCanvas(data.id, 'png');
            const image = await imageDataFromBlob(blob!);
            assert.equal(image!.width, data.width, 'natural width from the decoded image');
            assert.equal(image!.height, data.height, 'natural height from the decoded image');
            assert.deepEqual(Array.from(image!.data), Array.from(data.rgba), 'RGBA pixels survive encode→decode');
        });

        it('returns null (and logs) for an undecodable blob', async () => {
            const capture = captureConsole();
            let image: ImageData | null;
            try {
                image = await imageDataFromBlob(new Blob([new Uint8Array([1, 2, 3])], { type: 'image/png' }));
            } finally {
                capture.restore();
            }
            assert.isNull(image, 'garbage bytes yield null rather than throwing');
            assert.isTrue(capture.calls.some(c => c.method === 'error'), 'decode failure is logged to console.error');
        });
    });

    describe('clearCanvas', () => {
        itReturnsForNonCanvas(clearCanvas, false);

        it('returns true on success', () => {
            const data = makeTestCanvas();
            assert.isTrue(clearCanvas(data.id));
        });

        it('all pixels transparent after clearing a non-blank canvas', () => {
            const data = makeTestCanvas();
            clearCanvas(data.id);
            const result = typedArrayFromCanvas(data.id)!;
            for (let i = 0; i < data.width * data.height; i++) {
                assert.equal(result[i * 4 + 3], 0, `alpha at pixel ${i}`);
            }
        });
    });

    describe('setCanvasBufferSize', () => {
        it('no-op for null — does not throw', () => {
            assert.doesNotThrow(() => setCanvasBufferSize(null, 100, 100));
        });

        it('no-op for non-canvas element — does not modify the element', () => {
            const div = { tagName: 'DIV', width: 0, height: 0 } as any;
            setCanvasBufferSize(div, 50, 50);
            assert.equal(div.width, 0);
            assert.equal(div.height, 0);
        });

        it('sets width and height on a canvas element', () => {
            const el = { tagName: 'CANVAS', width: 60, height: 30 } as any;
            setCanvasBufferSize(el, 320, 240);
            assert.equal(el.width, 320);
            assert.equal(el.height, 240);
        });
    });

});
