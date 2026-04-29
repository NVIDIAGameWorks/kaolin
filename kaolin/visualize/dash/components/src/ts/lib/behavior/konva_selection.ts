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
 * Konva drawing variant that rasterizes each committed gesture onto an external
 * canvas, compositing with a selection action (new / add / subtract / intersect).
 *
 * @module
 */

import { z } from 'zod';

import { BehaviorRegister } from '../../core/behavior';
import {
    DraftState,
    KonvaDrawingBehavior,
    KonvaDrawingOptionsSchema,
} from './konva_drawing';


/**
 * Configuration schema for {@link KonvaSelectionBehavior}. Extends the
 * base {@link KonvaDrawingOptionsSchema} with commit-time options that control
 * how a finalized draft is composited onto the external canvas.
 *
 * ```typescript
 * KonvaDrawingOptionsSchema.extend({
 *     action: z.enum(['new', 'add', 'subtract', 'intersect']).default('new')
 *         .describe('How a committed draft is composited onto the target ' +
 *                   'canvas: replace it, union, difference, or intersect.'),
 *     compositeCanvasId: z.string()
 *         .describe('DOM id of the HTMLCanvasElement that receives the ' +
 *                   'flattened Konva composite on each committed gesture.')
 *         .meta({ uiBound: false }),
 * })
 * ```
 *
 * @group Behavior: KonvaSelectionBehavior
 */
export const KonvaSelectionOptionsSchema = KonvaDrawingOptionsSchema.extend({
    action: z.enum(['new', 'add', 'subtract', 'intersect']).default('new')
        .describe('How a committed draft is composited onto the target ' +
                  'canvas: replace it, union, difference, or intersect.'),
    compositeCanvasId: z.string()
        .describe('DOM id of the HTMLCanvasElement that receives the ' +
                  'flattened Konva composite on each committed gesture.')
        .meta({ uiBound: false }),
});

/**
 * Options for {@link KonvaSelectionBehavior}, parsed from
 * {@link KonvaSelectionOptionsSchema}.
 *
 * @group Behavior: KonvaSelectionBehavior
 */
export type KonvaSelectionOptions = z.infer<typeof KonvaSelectionOptionsSchema>;

/** How a committed draft combines with the existing composite-canvas pixels.
 *
 * @group Behavior: KonvaSelectionBehavior
 */
export type SelectionAction = KonvaSelectionOptions['action'];


/**
 * Konva drawing variant that rasterizes each successful gesture onto an
 * external `HTMLCanvasElement` (referenced by `compositeCanvasId`) and
 * composites it with the existing pixels using the configured
 * {@link SelectionAction}.
 *
 * All mode-specific behavior (box / polygon / freeform) is inherited from
 * {@link KonvaDrawingBehavior}; this class only overrides the
 * {@link KonvaDrawingBehavior.onDraftCommitted} hook to plug compositing
 * into the shared commit funnel. New modes added on the base automatically
 * participate without any change here.
 *
 * This is a {@link KonvaDrawingBehavior} subclass (hence an
 * {@link core.behavior.ElementBoundBehavior | ElementBoundBehavior}) that binds
 * to a `div` layer hosting a Konva stage.
 *
 * Registered name: `'konva_selection'`.
 *
 * Configuration schema: {@link KonvaSelectionOptionsSchema}.
 *
 * @group Behavior: KonvaSelectionBehavior
 */
export class KonvaSelectionBehavior extends KonvaDrawingBehavior {
    static override schema = KonvaSelectionOptionsSchema;

    override options: KonvaSelectionOptions;

    constructor(options: Partial<KonvaSelectionOptions> = {}) {
        super();
        this.options = KonvaSelectionOptionsSchema.parse(options);
    }

    /**
     * Resolve the composite canvas from options.compositeCanvasId.
     *
     * The element is accepted by duck-typing (presence of `getContext`) rather
     * than `instanceof HTMLCanvasElement`, so it works across realms / SSR-style
     * canvas backends (e.g. a node-canvas under happy-dom in tests) where the
     * resolved node is canvas-like but not the page's `HTMLCanvasElement`.
     *
     * @returns The canvas, or null if id is unset or the node is missing /
     *          not canvas-like.
     */
    resolveCompositeCanvas(): HTMLCanvasElement | null {
        const id = this.options.compositeCanvasId;
        if (!id) {
            return null;
        }
        const el = document.getElementById(id);
        return el && typeof (el as HTMLCanvasElement).getContext === 'function'
            ? (el as HTMLCanvasElement)
            : null;
    }

    /**
     * Flatten the committed draft into a mask image and composite it onto
     * the canvas referenced by `compositeCanvasId` using the configured
     * selection action. The base class is responsible for cleaning up the
     * polygon preview / start marker / rAF id after this returns; we only
     * have to deal with the main shape (which we destroy after capture,
     * since the composite canvas now owns the visual result).
     */
    protected override onDraftCommitted(draft: DraftState): void {
        const composite = this.resolveCompositeCanvas();
        if (!composite || !this.stage) {
            console.warn('KonvaSelectionBehavior: composite canvas ' +
                'unavailable; discarding committed draft.');
            draft.shape.destroy();
            return;
        }
        const ctx = composite.getContext('2d');
        if (!ctx) {
            console.warn('KonvaSelectionBehavior: composite canvas ' +
                'has no 2d context; discarding committed draft.');
            draft.shape.destroy();
            return;
        }

        // Re-style the shape into a white-with-alpha mask: white pixels at
        // the inside, transparent outside. Hide non-mask helpers from the
        // rasterization.
        const shape = draft.shape;
        shape.stroke(null as any);
        shape.strokeEnabled(false);
        shape.fill('#ffffff');
        shape.fillEnabled(true);
        draft.previewLine?.visible(false);
        draft.startMarker?.visible(false);
        draft.cancelMarker?.visible(false);
        this.drawLayer?.batchDraw();

        // Render the whole stage to an offscreen canvas. We use 9-arg
        // drawImage below to stretch the stage's raster to the composite
        // canvas's intrinsic pixel grid, so aspect ratios need not match.
        const stageCanvas = this.stage.toCanvas({
            pixelRatio: 1,
            imageSmoothingEnabled: false,
        });

        const prevOp = ctx.globalCompositeOperation;
        switch (this.options.action) {
            case 'new':
                ctx.clearRect(0, 0, composite.width, composite.height);
                ctx.globalCompositeOperation = 'source-over';
                break;
            case 'add':
                ctx.globalCompositeOperation = 'source-over';
                break;
            case 'intersect':
                ctx.globalCompositeOperation = 'source-in';
                break;
            case 'subtract':
                ctx.globalCompositeOperation = 'destination-out';
                break;
        }
        ctx.drawImage(
            stageCanvas,
            0, 0, stageCanvas.width, stageCanvas.height,
            0, 0, composite.width, composite.height,
        );
        ctx.globalCompositeOperation = prevOp;

        // The composite canvas is the canonical visualization now; we don't
        // need the in-stage shape any more. (The base will tear down the
        // helpers but leaves shape destruction to us by design.)
        draft.shape.destroy();
    }
}


BehaviorRegister.register('konva_selection', KonvaSelectionBehavior,
    'Konva drawing whose committed gestures are rasterized and composited ' +
    'onto an external HTMLCanvasElement (compositeCanvasId) using a ' +
    'selection action (new / add / subtract / intersect).');
