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
 * Helpers for testing three.js camera controllers (OrbitControls /
 * TrackballControls based) headlessly under happy-dom.
 *
 * Those controls attach native listeners to the canvas and read its layout box,
 * pointer capture and owner document. happy-dom has no layout (every rect is
 * zero) and no pointer capture, so {@link makeControllerCanvas} pins a fixed
 * layout box and stubs the capture methods. {@link dragPointer} and
 * {@link wheel} then dispatch real DOM events at the canvas.
 *
 * Requires {@link registerDom} to have run.
 *
 * @module
 */

import { makeCanvasElement } from './dom';

/**
 * Create a canvas whose layout box is pinned to `width`x`height` at the origin,
 * with the pointer-capture methods the controls call stubbed out.
 */
export function makeControllerCanvas(width = 200, height = 200): HTMLCanvasElement {
    const canvas = makeCanvasElement(width, height);
    document.body.appendChild(canvas);
    canvas.getBoundingClientRect = () => ({
        left: 0, top: 0, right: width, bottom: height, width, height,
        x: 0, y: 0, toJSON: () => ({}),
    } as DOMRect);
    Object.defineProperty(canvas, 'clientWidth', { value: width, configurable: true });
    Object.defineProperty(canvas, 'clientHeight', { value: height, configurable: true });
    canvas.setPointerCapture = () => {};
    canvas.releasePointerCapture = () => {};
    canvas.hasPointerCapture = () => false;
    return canvas;
}

/** Dispatch a pointer event of `type` at (`x`, `y`) on `target`. */
function pointer(target: EventTarget, type: string, x: number, y: number, button: number, buttons: number): void {
    const event = new PointerEvent(type, {
        clientX: x, clientY: y, button, buttons, pointerId: 1, pointerType: 'mouse',
        isPrimary: true, bubbles: true, cancelable: true,
    });
    // TrackballControls reads pageX/pageY, which happy-dom does not derive from
    // clientX/clientY. The document is never scrolled, so they are equal.
    Object.defineProperty(event, 'pageX', { value: x });
    Object.defineProperty(event, 'pageY', { value: y });
    target.dispatchEvent(event);
}

/**
 * Press at `from`, move through `steps` intermediate points to `to`, release.
 * `button` 0 rotates, 2 pans (for both OrbitControls and TrackballControls).
 */
export function dragPointer(canvas: HTMLCanvasElement, from: [number, number], to: [number, number],
                            button = 0, steps = 4): void {
    const mask = 1 << button;
    pointer(canvas, 'pointerdown', from[0], from[1], button, mask);
    for (let i = 1; i <= steps; ++i) {
        const t = i / steps;
        pointer(canvas, 'pointermove', from[0] + (to[0] - from[0]) * t, from[1] + (to[1] - from[1]) * t,
                button, mask);
    }
    pointer(canvas, 'pointerup', to[0], to[1], button, 0);
}

/** Dispatch a wheel event (positive `deltaY` zooms out) at `canvas`. */
export function wheel(canvas: HTMLCanvasElement, deltaY: number): void {
    canvas.dispatchEvent(new WheelEvent('wheel', { deltaY, bubbles: true, cancelable: true }));
}
