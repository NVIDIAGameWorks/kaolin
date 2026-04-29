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

/**
 * DOM helpers for the Node.js test environment.
 *
 * Provides a real browser DOM via happy-dom ({@link registerDom} /
 * {@link unregisterDom}), opt-in per test file so suites that need no DOM stay
 * free of browser globals. Tests that only need spies/stubs should use a plain
 * sinon sandbox directly.
 *
 * The happy-dom instance is configured with `@happy-dom/node-canvas-adapter`, so
 * `document.createElement('canvas')` returns a normal happy-dom canvas element
 * that is fully node-canvas-backed: `getContext('2d')`, `getImageData` /
 * `putImageData`, `toBlob`, and `toDataURL` all render real pixels. Canvases are
 * ordinary DOM nodes, so `getElementById`/`appendChild`/`removeChild` and the
 * rest of the DOM API work without any special handling.
 *
 * happy-dom ships no image decoder, so its `createImageBitmap` rejects `Blob`
 * sources. {@link registerDom} patches in a node-canvas-backed decoder for that
 * case, letting browser code such as `createImageBitmap(blob)` work unchanged.
 *
 * @module
 */

import { GlobalRegistrator } from '@happy-dom/global-registrator';
import { CanvasAdapter } from '@happy-dom/node-canvas-adapter';
import { PropertySymbol } from 'happy-dom';
import { loadImage } from 'canvas';

/**
 * Install a real, canvas-capable browser DOM on the global scope via happy-dom.
 * Idempotent, so it is safe to call from a file-level `before` hook. Pair with
 * {@link unregisterDom}.
 */
export function registerDom(): void {
    if (!GlobalRegistrator.isRegistered) {
        GlobalRegistrator.register({
            settings: { canvasAdapter: new CanvasAdapter() },
        });
        patchCreateImageBitmapForBlobs();
    }
}

/**
 * Teach the global `createImageBitmap` to accept `Blob` sources. happy-dom has
 * no image decoder, so we decode the bytes with node-canvas, stash them on a
 * happy-dom `HTMLImageElement`, and delegate to the native `createImageBitmap`,
 * which the canvas adapter can rasterize. Non-Blob sources pass straight through.
 */
function patchCreateImageBitmapForBlobs(): void {
    const win = globalThis as any;
    const native = win.createImageBitmap.bind(win);
    const decodeBlob = async (blob: Blob, ...rest: any[]) => {
        const buffer = Buffer.from(await blob.arrayBuffer());
        const decoded = await loadImage(buffer as any);
        const image = new win.Image();
        image[PropertySymbol.buffer] = buffer;
        image.width = decoded.width;
        image.height = decoded.height;
        return native(image, ...rest);
    };
    win.createImageBitmap = (source: any, ...rest: any[]) =>
        source instanceof win.Blob ? decodeBlob(source, ...rest) : native(source, ...rest);
}

/**
 * Create an element of the given tag, tag it with an id, and attach it to
 * `document.body` so code under test can resolve it via
 * `document.getElementById`. Requires {@link registerDom} to have run.
 *
 * @param tag - HTML tag name to create (e.g. `'div'`, `'canvas'`).
 * @param id - DOM id to assign to the element.
 * @returns The attached element.
 */
export function addElement<K extends keyof HTMLElementTagNameMap>(tag: K, id: string): HTMLElementTagNameMap[K] {
    const el = document.createElement(tag);
    el.id = id;
    document.body.appendChild(el);
    return el;
}

/**
 * Create a detached, node-canvas-backed canvas element sized to the given
 * dimensions. Not attached to the document, so it cannot be resolved by id; use
 * it when passing the element directly. Requires {@link registerDom} to have run.
 *
 * @param width - Drawing-buffer width in pixels.
 * @param height - Drawing-buffer height in pixels.
 * @returns The detached canvas element.
 */
export function makeCanvasElement(width: number, height: number): HTMLCanvasElement {
    const canvas = document.createElement('canvas') as HTMLCanvasElement;
    canvas.width = width;
    canvas.height = height;
    return canvas;
}

/**
 * Create a node-canvas-backed canvas element, tag it with an id, size it, and
 * attach it to `document.body` so canvas-under-test code can resolve it via
 * `document.getElementById`. Requires {@link registerDom} to have run.
 *
 * @param id - DOM id to assign to the canvas.
 * @param width - Drawing-buffer width in pixels.
 * @param height - Drawing-buffer height in pixels.
 * @returns The attached canvas element.
 */
export function addCanvasElement(id: string, width: number, height: number): HTMLCanvasElement {
    const canvas = addElement('canvas', id);
    canvas.width = width;
    canvas.height = height;
    return canvas;
}

/**
 * Tear down the DOM installed by {@link registerDom}. Idempotent; returns the
 * happy-dom close promise so callers can `await` it from an `after` hook.
 */
export async function unregisterDom(): Promise<void> {
    if (GlobalRegistrator.isRegistered) {
        await GlobalRegistrator.unregister();
    }
}
