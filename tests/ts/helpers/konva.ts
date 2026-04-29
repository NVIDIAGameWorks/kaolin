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
 * Headless test harness for the Konva-backed behaviors (`konva_drawing`,
 * `konva_selection`).
 *
 * Konva needs a few things our plain happy-dom setup doesn't provide, and
 * getting all of them right is fiddly; this module bundles the working recipe:
 *
 *  1. **Rendering backend.** `import 'konva/canvas-backend'` (Konva v10+) routes
 *     every Konva canvas through node-canvas. happy-dom's own adapter canvases
 *     render Konva shapes blank, so this import is mandatory and is done here as
 *     a side effect.
 *  2. **Browser mode.** Konva is imported in Node, so `Konva.isBrowser` is false
 *     and it never builds the stage's `content` div nor wires pointer listeners.
 *     Our behaviors redispatch native events to `stage.content`, so we force
 *     `Konva.isBrowser = true`.
 *  3. **appendChild tolerance.** In browser mode Konva appends each layer's
 *     offscreen node-canvas to the content div; those are not happy-dom nodes, so
 *     happy-dom's `appendChild` throws. Attachment is irrelevant to rendering /
 *     capture, so we skip non-happy-dom canvas children.
 *  4. **Layout.** happy-dom has no layout, so element rects / `clientWidth` are
 *     all zero. We stub the container and the stage content to a fixed square
 *     `size`, so Konva maps pointer coordinates 1:1 onto a `size`-px stage and
 *     {@link record_interactions.eventFromSerializable | replayed events} resolve
 *     to the same pixels they were recorded at.
 *  5. **Synchronous rAF.** Freeform drawing coalesces points in
 *     `requestAnimationFrame`; we run callbacks synchronously so each replayed
 *     move paints immediately and in order.
 *  6. **No synthetic double-clicks.** Konva tracks a "within dbl-click window"
 *     flag that it clears on a `setTimeout(..., Konva.dblClickWindow)`. Replayed
 *     events fire synchronously, so that timeout never runs between clicks and
 *     the flag sticks — every recorded single click would register as a
 *     double-click (e.g. prematurely committing a polygon). We set the window to
 *     0 and run only zero-delay timeouts synchronously, so the flag clears
 *     between clicks. A genuine double-click would carry its own captured
 *     `onDoubleClick` entries, which replay drives explicitly.
 *  7. **Synchronous draws (hit testing).** Konva froze its animation frame to a
 *     `setTimeout(.., 16)` fallback at import (no rAF existed yet), so batched
 *     scene/hit draws never run during synchronous replay. Closing a polygon by
 *     clicking its start marker needs a current *hit* canvas, so we override
 *     `Konva.Util.requestAnimFrame` to run synchronously. To avoid Konva drawing
 *     a layer at construction (before it has a canvas — it would crash), we also
 *     disable `Konva.autoDrawEnabled`; the behaviors call `batchDraw` explicitly
 *     after every mutation, so rendering still happens.
 *
 * Pair {@link setupKonvaHarness} (in a `before` hook) with
 * {@link KonvaHarness.teardown} (in `after`).
 *
 * @module
 */

import Konva from 'konva';
import 'konva/canvas-backend';

import { registerDom, unregisterDom } from './dom';

/** A fixed on-screen rect of `size`x`size` anchored at the origin. */
function squareRect(size: number): DOMRect {
    return {
        left: 0, top: 0, right: size, bottom: size, width: size, height: size,
        x: 0, y: 0, toJSON: () => ({}),
    } as DOMRect;
}

/** Pin an element's layout box (`getBoundingClientRect` + `clientWidth/Height`) to `size`. */
function stubGeometry(el: HTMLElement, size: number): void {
    el.getBoundingClientRect = () => squareRect(size);
    Object.defineProperty(el, 'clientWidth', { value: size, configurable: true });
    Object.defineProperty(el, 'clientHeight', { value: size, configurable: true });
}

/** Minimal view of a Konva behavior: bind an element, expose the created stage. */
interface KonvaBehaviorLike {
    setActiveElement(element: HTMLElement): void;
}

/**
 * Active Konva test harness; see {@link setupKonvaHarness}. All stubs target a
 * single square stage of the configured `size`.
 */
export interface KonvaHarness {
    /**
     * Create a stage-container `div`, tagged with `id`, attached to the document
     * and sized (layout box pinned to the harness `size`). Pass it to
     * {@link KonvaHarness.attach}.
     */
    container(id: string): HTMLDivElement;

    /**
     * Bind `behavior` to `container` (via `setActiveElement`) and pin the
     * resulting Konva stage's inner content div to `size`, so pointer coordinates
     * map 1:1 onto the stage.
     */
    attach(behavior: KonvaBehaviorLike, container: HTMLDivElement): void;

    /**
     * Create a node-canvas-backed composite target resolvable by
     * `document.getElementById(id)` — required by `konva_selection`, whose
     * stage->composite `drawImage` only works when both share Konva's backend.
     * Sized to the harness `size` unless overridden.
     */
    compositeCanvas(id: string, width?: number, height?: number): HTMLCanvasElement;

    /** Restore every global this harness patched and tear the DOM down. */
    teardown(): Promise<void>;
}

/**
 * Install the Konva headless harness (see the module overview for the full
 * recipe) for a square stage of `size` pixels.
 *
 * @param size - Stage / container / golden edge length in pixels.
 * @returns The active {@link KonvaHarness}.
 */
export function setupKonvaHarness(size: number): KonvaHarness {
    registerDom();

    const win = window as any;
    const konvaUtil = (Konva as any).Util;
    const prevIsBrowser = Konva.isBrowser;
    const prevDblClickWindow = Konva.dblClickWindow;
    const prevAutoDraw = Konva.autoDrawEnabled;
    const prevReqAnimFrame = konvaUtil.requestAnimFrame;
    const prevRaf = win.requestAnimationFrame;
    const prevSetTimeout = win.setTimeout;
    const prevAppend = win.Node.prototype.appendChild;
    const prevGetById = window.document.getElementById.bind(window.document);
    const composites = new Map<string, HTMLCanvasElement>();

    Konva.isBrowser = true;
    Konva.dblClickWindow = 0;
    Konva.autoDrawEnabled = false;
    konvaUtil.requestAnimFrame = (cb: () => void) => { cb(); };
    win.requestAnimationFrame = (cb: FrameRequestCallback) => { cb(0); return 0; };
    win.setTimeout = ((fn: (...a: any[]) => void, delay?: number, ...args: any[]) => {
        if (delay === 0) { fn(...args); return 0; }
        return prevSetTimeout(fn, delay, ...args);
    }) as any;
    win.Node.prototype.appendChild = function (child: any) {
        if (child && typeof child.getContext === 'function' && !(child instanceof win.HTMLElement)) {
            return child;
        }
        return prevAppend.call(this, child);
    };
    win.document.getElementById = (id: string) => composites.get(id) ?? prevGetById(id);

    return {
        container(id: string): HTMLDivElement {
            const div = window.document.createElement('div');
            div.id = id;
            window.document.body.appendChild(div);
            stubGeometry(div, size);
            return div;
        },

        attach(behavior: KonvaBehaviorLike, container: HTMLDivElement): void {
            behavior.setActiveElement(container);
            const stage = (behavior as unknown as { stage?: Konva.Stage | null }).stage;
            if (stage?.content) stubGeometry(stage.content, size);
        },

        compositeCanvas(id: string, width = size, height = size): HTMLCanvasElement {
            const canvas = (Konva as any).Util.createCanvasElement() as HTMLCanvasElement;
            canvas.width = width;
            canvas.height = height;
            composites.set(id, canvas);
            return canvas;
        },

        async teardown(): Promise<void> {
            win.document.getElementById = prevGetById;
            win.Node.prototype.appendChild = prevAppend;
            win.setTimeout = prevSetTimeout;
            win.requestAnimationFrame = prevRaf;
            konvaUtil.requestAnimFrame = prevReqAnimFrame;
            Konva.autoDrawEnabled = prevAutoDraw;
            Konva.dblClickWindow = prevDblClickWindow;
            Konva.isBrowser = prevIsBrowser;
            composites.clear();
            await unregisterDom();
        },
    };
}
