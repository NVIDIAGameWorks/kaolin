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
 * Konva-backed in-stage drawing behavior supporting box, polygon, and freeform
 * gestures, with a commit hook subclasses override to decide a draft's fate.
 *
 * @module
 */

import Konva from 'konva';
import { z } from 'zod';

import { BehaviorRegister, HandlerReturnInfo, InteractiveBehavior } from '../../core/behavior';
import { zColor } from '../../core/behavior/option';
import * as kaolin_events from '../../core/event';


/** The shape a drawing gesture produces: axis-aligned box, polygon, or freeform blob.
 *
 * @group Behavior: KonvaDrawingBehavior
 */
export type DrawingMode = 'box' | 'polygon' | 'freeform';


/**
 * Configuration schema for {@link KonvaDrawingBehavior}. Single source of truth:
 * the TS option type ({@link KonvaDrawingOptions}) is derived from it via
 * `z.infer`, defaults are read off it at construction, and the build-time
 * manifest dumper projects it to JSON for the Python auto-UI pipeline.
 *
 * Subclasses (e.g. KonvaSelectionBehavior) extend this schema with
 * their own commit-time options; mode-specific drawing options live here.
 *
 * ```typescript
 * z.object({
 *     mode: z.enum(['box', 'polygon', 'freeform']).default('box')
 *         .describe('Active draft tool.'),
 *     color: zColor().default('#00ff88')
 *         .describe('Color shared by stroke and (translucent) fill of the ' +
 *                   'in-progress draft.'),
 *     opacity: z.number().min(0).max(1).multipleOf(0.05).default(0.20)
 *         .describe('Alpha (0..1) applied to color for the draft fill.'),
 *     overlayStrokeWidth: z.number().min(1).max(20).multipleOf(1).default(2)
 *         .describe('Stroke width (px) of the in-progress draft outline.')
 *         .meta({ uiBound: false }),
 *     polygonCloseRadiusPx: z.number().min(2).max(50).multipleOf(1).default(10)
 *         .describe('Radius (stage px) of the start-vertex marker for ' +
 *                   'polygon drawing; clicking it closes the polygon.')
 *         .meta({ uiBound: false }),
 * })
 * ```
 *
 * @group Behavior: KonvaDrawingBehavior
 */
export const KonvaDrawingOptionsSchema = z.object({
    mode: z.enum(['box', 'polygon', 'freeform']).default('box')
        .describe('Active draft tool.'),
    color: zColor().default('#00ff88')
        .describe('Color shared by stroke and (translucent) fill of the ' +
                  'in-progress draft.'),
    opacity: z.number().min(0).max(1).multipleOf(0.05).default(0.20)
        .describe('Alpha (0..1) applied to color for the draft fill.'),
    overlayStrokeWidth: z.number().min(1).max(20).multipleOf(1).default(2)
        .describe('Stroke width (px) of the in-progress draft outline.')
        .meta({ uiBound: false }),
    polygonCloseRadiusPx: z.number().min(2).max(50).multipleOf(1).default(10)
        .describe('Radius (stage px) of the start-vertex marker for ' +
                  'polygon drawing; clicking it closes the polygon.')
        .meta({ uiBound: false }),
});

/**
 * Options for {@link KonvaDrawingBehavior}, parsed from {@link KonvaDrawingOptionsSchema}.
 *
 * @group Behavior: KonvaDrawingBehavior
 */
export type KonvaDrawingOptions = z.infer<typeof KonvaDrawingOptionsSchema>;


/** Parse '#rrggbb' or '#rgb' into [r, g, b] (0..255). Returns null on unknown formats. */
function parseHexColor(input: string): [number, number, number] | null {
    const m3 = /^#([0-9a-f])([0-9a-f])([0-9a-f])$/i.exec(input);
    if (m3) {
        const r = parseInt(m3[1] + m3[1], 16);
        const g = parseInt(m3[2] + m3[2], 16);
        const b = parseInt(m3[3] + m3[3], 16);
        return [r, g, b];
    }
    const m6 = /^#([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(input);
    if (m6) {
        return [parseInt(m6[1], 16), parseInt(m6[2], 16), parseInt(m6[3], 16)];
    }
    return null;
}


/** Return a CSS color string equivalent to `color` with the given alpha (0..1).
 *  Falls back to the input color unchanged if the format isn't recognized hex. */
function withAlpha(color: string, alpha: number): string {
    const a = Math.max(0, Math.min(1, alpha));
    const rgb = parseHexColor(color);
    if (rgb) {
        return `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${a})`;
    }
    return color;
}


/**
 * Live state for an in-progress drawing gesture. Exported so subclasses can
 * receive it through the {@link KonvaDrawingBehavior.onDraftCommitted} hook.
 *
 * @group Behavior: KonvaDrawingBehavior
 */
export interface DraftState {
    mode: DrawingMode;
    /** The shape that becomes the committed object (its filled interior). */
    shape: Konva.Rect | Konva.Line;
    /** For box: the pointerdown anchor; for polygon/freeform: the first vertex. */
    anchor: { x: number; y: number };
    /** Polygon: dashed live segment from last vertex to cursor. */
    previewLine?: Konva.Line;
    /** Polygon: clickable circle at the start vertex for closing the polygon. */
    startMarker?: Konva.Circle;
    /**
     * Polygon: on-canvas '✕' affordance near the start vertex that aborts
     * the current draft when tapped. Universal touch + desktop cancel
     * gesture (no keyboard or right-click required); subclasses should
     * hide it before any commit-time rasterization so it isn't baked into
     * the output.
     */
    cancelMarker?: Konva.Group;
    /** Freeform: pending rAF id for coalesced point appending. */
    rafId?: number;
}


/**
 * Konva-backed in-stage drawing behavior. Owns the entire gesture pipeline
 * (mode dispatch, draft lifecycle, restyle/discard, polygon close-marker)
 * and exposes a single virtual {@link onDraftCommitted} hook that subclasses
 * override to decide what happens to a validated draft at commit time.
 *
 * The default `onDraftCommitted` is a no-op, so committed shapes remain on
 * the Konva draw layer as persistent objects. The {@link
 * KonvaSelectionBehavior} subclass overrides the hook to rasterize
 * the draft onto an external composite canvas and then destroy the shape.
 *
 * All mode-specific logic lives here; subclasses should never need to
 * inspect `options.mode` or branch on a specific mode.
 *
 * This is an {@link core.behavior.InteractiveBehavior | InteractiveBehavior}
 * (an {@link core.behavior.ElementBoundBehavior | ElementBoundBehavior}) that
 * binds to a `div` layer hosting a Konva stage.
 *
 * Registered name: `'konva_drawing'`.
 *
 * Configuration schema: {@link KonvaDrawingOptionsSchema}.
 *
 * @group Behavior: KonvaDrawingBehavior
 */
export class KonvaDrawingBehavior extends InteractiveBehavior {
    static override schema = KonvaDrawingOptionsSchema;

    override options: KonvaDrawingOptions;

    protected stage: Konva.Stage | null = null;
    protected drawLayer: Konva.Layer | null = null;
    private resizeObserver: ResizeObserver | null = null;
    private draft: DraftState | null = null;
    private isActive: boolean = true;

    constructor(options: Partial<KonvaDrawingOptions> = {}) {
        super();
        this.options = KonvaDrawingOptionsSchema.parse(options);
    }

    override elementType(): string {
        return 'div';
    }

    override setActiveElement(element: HTMLElement): void {
        this.disposeStage();
        super.setActiveElement(element);
        if (!element) return;

        const w = Math.max(1, element.clientWidth);
        const h = Math.max(1, element.clientHeight);

        this.stage = new Konva.Stage({
            container: element as HTMLDivElement,
            width: w,
            height: h,
            listening: true,
        });

        this.drawLayer = new Konva.Layer({ listening: true });
        this.stage.add(this.drawLayer);

        this.registerKonvaHandlers();

        if (typeof ResizeObserver !== 'undefined') {
            this.resizeObserver = new ResizeObserver(() => this.handleResize());
            this.resizeObserver.observe(element);
        }

        if (!this.isActive) {
            this.drawLayer.hide();
        }
    }

    /**
     * Resize the Konva stage to match the viewer-supplied dimensions.
     * Also calls super (sets CSS `style.width`/`style.height` on the div) so
     * the container and stage stay in sync — Konva itself already writes those
     * properties on the div, so this is idempotent. Mirrors the guard used by
     * the internal `handleResize` to avoid spurious redraws.
     */
    setDimensions(width: number, height: number): void {
        if (!this.stage) return;
        const w = Math.max(1, width);
        const h = Math.max(1, height);
        if (this.stage.width() === w && this.stage.height() === h) return;
        this.stage.width(w);
        this.stage.height(h);
        this.drawLayer?.batchDraw();
    }

    /**
     * Schema-driven option reaction: routes per-option side effects.
     * Changing the drawing mode aborts any in-progress draft; visual options
     * (color / stroke / fill / start-marker radius) restyle the current draft
     * so live edits update immediately. Other options have no draft-time
     * effect and take hold on the next commit.
     */
    override updateForOptions(name?: string, _value?: any): void {
        if (name === 'mode') {
            this.discardDraft();
        } else if (
            name === 'color' ||
            name === 'overlayStrokeWidth' ||
            name === 'opacity' ||
            name === 'polygonCloseRadiusPx'
        ) {
            this.restyleDraft();
        }
    }

    /** Called by the viewer when the behavior's active flag flips. */
    setActive(active: boolean): void {
        this.isActive = active;
        if (!active) {
            this.discardDraft();
            this.drawLayer?.hide();
        } else {
            this.drawLayer?.show();
        }
        this.drawLayer?.batchDraw();
    }

    private drawingEnabled(): boolean {
        return this.isActive &&
            this.element !== null &&
            this.stage !== null;
    }

    // ---- React handlers: re-dispatch native events on the div so Konva sees them ----

    override onClick(event: React.MouseEvent): void | HandlerReturnInfo {
        this.redispatch(event, 'click');
    }

    override onDoubleClick(event: React.MouseEvent): void | HandlerReturnInfo {
        this.redispatch(event, 'dblclick');
    }

    override onPointerDown(event: React.PointerEvent): void | HandlerReturnInfo {
        this.redispatch(event, 'pointerdown');
        return { needsPointermove: true };
    }

    override onPointerMove(event: React.PointerEvent): void | HandlerReturnInfo {
        this.redispatch(event, 'pointermove');
    }

    override onPointerUp(event: React.PointerEvent): void | HandlerReturnInfo {
        this.redispatch(event, 'pointerup');
        return { needsPointermove: false };
    }

    override onPointerCancel(event: React.PointerEvent): void | HandlerReturnInfo {
        this.redispatch(event, 'pointercancel');
    }

    override onPointerEnter(event: React.PointerEvent): void | HandlerReturnInfo {
        this.redispatch(event, 'pointerenter');
    }

    override onPointerLeave(event: React.PointerEvent): void | HandlerReturnInfo {
        this.redispatch(event, 'pointerleave');
    }

    private redispatch(reactEvent: React.SyntheticEvent, nativeType: string): void {
        if (!this.drawingEnabled() || !this.stage) return;
        // Konva attaches its DOM listeners to an inner <div class="konvajs-content">
        // (Stage.content), not the user-supplied container. Synthesized events must
        // target that inner div so they reach Konva's pointer dispatch.
        const target = this.stage.content;
        if (!target) return;
        const src = reactEvent.nativeEvent as PointerEvent | MouseEvent;

        // Build a fresh native event we own (we can't re-dispatch the original).
        // For 'pointer*' types we construct a PointerEvent, otherwise a MouseEvent.
        const init: PointerEventInit = {
            bubbles: true,
            cancelable: true,
            composed: true,
            clientX: (src as MouseEvent).clientX,
            clientY: (src as MouseEvent).clientY,
            screenX: (src as MouseEvent).screenX,
            screenY: (src as MouseEvent).screenY,
            button: (src as MouseEvent).button ?? 0,
            buttons: (src as MouseEvent).buttons ?? 0,
            ctrlKey: src.ctrlKey,
            shiftKey: src.shiftKey,
            altKey: src.altKey,
            metaKey: src.metaKey,
            detail: (src as MouseEvent).detail ?? 0,
            pointerId: (src as PointerEvent).pointerId ?? 1,
            pointerType: (src as PointerEvent).pointerType ?? 'mouse',
            pressure: (src as PointerEvent).pressure ?? 0,
            isPrimary: (src as PointerEvent).isPrimary ?? true,
        };

        let synthetic: Event;
        if (nativeType.startsWith('pointer')) {
            synthetic = new PointerEvent(nativeType, init);
        } else {
            synthetic = new MouseEvent(nativeType, init as MouseEventInit);
        }
        target.dispatchEvent(synthetic);
    }

    // ---- Konva-side handlers (the actual drawing logic) ----

    private registerKonvaHandlers(): void {
        if (!this.stage) return;
        const ns = '.kaolin';
        this.stage.on('pointerdown' + ns, () => this.handleStagePointerDown());
        this.stage.on('pointermove' + ns, () => this.handleStagePointerMove());
        this.stage.on('pointerup' + ns, () => this.handleStagePointerUp());
        this.stage.on('pointercancel' + ns, () => this.handleStagePointerCancel());
        // Konva fires its synthesized "click" under different names depending on
        // the DOM event family that produced the underlying pointerdown/up:
        // mouse-family -> 'click'/'dblclick', pointer-family -> 'pointerclick'/
        // 'pointerdblclick', touch-family -> 'tap'/'dbltap'. We subscribe to all
        // three so polygon click/dblclick handling works regardless of which
        // family the (possibly synthesized) input arrived as.
        this.stage.on(`click${ns} pointerclick${ns} tap${ns}`,
            () => this.handleStageClick());
        this.stage.on(`dblclick${ns} pointerdblclick${ns} dbltap${ns}`,
            () => this.handleStageDoubleClick());
    }

    private handleStagePointerDown(): void {
        const pos = this.stage?.getPointerPosition();
        if (!pos) return;

        const mode = this.options.mode;
        if (mode === 'box') {
            this.beginBox(pos);
        } else if (mode === 'freeform') {
            this.beginFreeform(pos);
        }
    }

    private handleStagePointerMove(): void {
        const pos = this.stage?.getPointerPosition();
        if (!pos || !this.draft) return;

        if (this.draft.mode === 'box') {
            this.updateBox(pos);
        } else if (this.draft.mode === 'freeform') {
            this.queueFreeformAppend(pos);
        } else if (this.draft.mode === 'polygon') {
            this.updatePolygonPreview(pos);
        }
    }

    private handleStagePointerUp(): void {
        if (!this.draft) return;

        if (this.draft.mode === 'box') {
            this.commitBox();
        } else if (this.draft.mode === 'freeform') {
            this.commitFreeform();
        }
        // Polygon commits on dblclick or click-on-start-marker, not on pointerup.
    }

    private handleStagePointerCancel(): void {
        this.discardDraft();
    }

    private handleStageClick(): void {
        if (this.options.mode !== 'polygon') return;
        const pos = this.stage?.getPointerPosition();
        if (!pos) return;
        if (!this.draft) {
            this.beginPolygon(pos);
        } else {
            this.addPolygonVertex(pos);
        }
    }

    private handleStageDoubleClick(): void {
        if (this.draft && this.draft.mode === 'polygon') {
            this.commitPolygon();
        }
    }

    // ---- Box ----

    private beginBox(pos: { x: number; y: number }): void {
        this.discardDraft();
        const rect = new Konva.Rect({
            x: pos.x,
            y: pos.y,
            width: 0,
            height: 0,
            stroke: this.options.color,
            strokeWidth: this.options.overlayStrokeWidth,
            fill: withAlpha(this.options.color, this.options.opacity),
            listening: false,
        });
        this.drawLayer?.add(rect);
        this.draft = { mode: 'box', shape: rect, anchor: { x: pos.x, y: pos.y } };
        this.drawLayer?.batchDraw();
    }

    private updateBox(pos: { x: number; y: number }): void {
        if (!this.draft || this.draft.mode !== 'box') return;
        const rect = this.draft.shape as Konva.Rect;
        const ax = this.draft.anchor.x;
        const ay = this.draft.anchor.y;
        rect.x(Math.min(ax, pos.x));
        rect.y(Math.min(ay, pos.y));
        rect.width(Math.abs(pos.x - ax));
        rect.height(Math.abs(pos.y - ay));
        this.drawLayer?.batchDraw();
    }

    private commitBox(): void {
        if (!this.draft || this.draft.mode !== 'box') return;
        const rect = this.draft.shape as Konva.Rect;
        const minSize = 2;
        if (rect.width() < minSize || rect.height() < minSize) {
            this.discardDraft();
            return;
        }
        this.commitDraft();
    }

    // ---- Freeform ----

    private beginFreeform(pos: { x: number; y: number }): void {
        this.discardDraft();
        const line = new Konva.Line({
            points: [pos.x, pos.y, pos.x, pos.y],
            closed: true,
            tension: 0,
            stroke: this.options.color,
            strokeWidth: this.options.overlayStrokeWidth,
            fill: withAlpha(this.options.color, this.options.opacity),
            listening: false,
        });
        this.drawLayer?.add(line);
        this.draft = { mode: 'freeform', shape: line, anchor: { x: pos.x, y: pos.y } };
        this.drawLayer?.batchDraw();
    }

    private queueFreeformAppend(pos: { x: number; y: number }): void {
        if (!this.draft || this.draft.mode !== 'freeform') return;
        const line = this.draft.shape as Konva.Line;
        line.points(line.points().concat([pos.x, pos.y]));
        if (this.draft.rafId !== undefined) return;
        this.draft.rafId = window.requestAnimationFrame(() => {
            if (this.draft) this.draft.rafId = undefined;
            this.drawLayer?.batchDraw();
        });
    }

    private commitFreeform(): void {
        if (!this.draft || this.draft.mode !== 'freeform') return;
        const line = this.draft.shape as Konva.Line;
        if (line.points().length < 6) {
            this.discardDraft();
            return;
        }
        this.commitDraft();
    }

    // ---- Polygon ----

    private beginPolygon(pos: { x: number; y: number }): void {
        this.discardDraft();
        const line = new Konva.Line({
            points: [pos.x, pos.y],
            closed: false,
            stroke: this.options.color,
            strokeWidth: this.options.overlayStrokeWidth,
            fill: withAlpha(this.options.color, this.options.opacity),
            fillEnabled: false,
            listening: false,
        });
        const preview = new Konva.Line({
            points: [pos.x, pos.y, pos.x, pos.y],
            stroke: this.options.color,
            strokeWidth: this.options.overlayStrokeWidth,
            dash: [6, 4],
            listening: false,
        });
        const startMarker = new Konva.Circle({
            x: pos.x,
            y: pos.y,
            radius: this.options.polygonCloseRadiusPx,
            stroke: this.options.color,
            strokeWidth: this.options.overlayStrokeWidth,
            fill: withAlpha(this.options.color, this.options.opacity),
            listening: true,
        });
        // Close the polygon on the synthesized click (not pointerdown): otherwise
        // the marker is destroyed by commitPolygon during pointerdown, the
        // following pointerup has no shape under it, and Konva fires its
        // `pointerclick` directly on the stage -- which would then trigger
        // `handleStageClick` and start a brand-new polygon draft at the same
        // position. Listening on click/pointerclick/tap keeps the marker alive
        // through pointerup, so `cancelBubble = true` here actually suppresses
        // the stage-level click handler from also seeing the event.
        startMarker.on('click.kaolin pointerclick.kaolin tap.kaolin', (e) => {
            e.cancelBubble = true;
            this.commitPolygon();
        });
        const cancelMarker = this.createCancelMarker(pos);
        this.drawLayer?.add(line);
        this.drawLayer?.add(preview);
        this.drawLayer?.add(startMarker);
        this.drawLayer?.add(cancelMarker);
        this.draft = {
            mode: 'polygon',
            shape: line,
            anchor: { x: pos.x, y: pos.y },
            previewLine: preview,
            startMarker: startMarker,
            cancelMarker: cancelMarker,
        };
        this.drawLayer?.batchDraw();
    }

    /**
     * Build a small on-canvas '✕' affordance placed near the polygon's
     * start vertex. Tapping it aborts the current draft; it is the
     * universal touch + desktop cancel gesture for in-progress polygons
     * (touch devices have no Esc / right-click).
     *
     * The pip is rendered as a white-filled, red-stroked circle with a
     * red X drawn from two diagonal lines, so it's visually distinct
     * from the same-colored start marker even when both share the
     * polygon's accent color. It's positioned in whichever diagonal
     * quadrant has the most room around the start vertex so it stays
     * on-canvas when the polygon is started near a corner.
     */
    private createCancelMarker(pos: { x: number; y: number }): Konva.Group {
        const r = this.options.polygonCloseRadiusPx;
        const sw = this.options.overlayStrokeWidth;
        const W = this.stage?.width() ?? Infinity;
        const dx = (pos.x + 1.8 * r + r > W) ? -1.8 * r : 1.8 * r;
        const dy = (pos.y - 1.8 * r - r < 0) ? 1.8 * r : -1.8 * r;
        const group = new Konva.Group({
            x: pos.x + dx,
            y: pos.y + dy,
            listening: true,
        });
        const bg = new Konva.Circle({
            radius: r,
            fill: '#ffffff',
            stroke: '#cc2222',
            strokeWidth: sw,
            listening: true,
        });
        const arm = r * 0.55;
        const line1 = new Konva.Line({
            points: [-arm, -arm, arm, arm],
            stroke: '#cc2222',
            strokeWidth: sw,
            listening: false,
        });
        const line2 = new Konva.Line({
            points: [-arm, arm, arm, -arm],
            stroke: '#cc2222',
            strokeWidth: sw,
            listening: false,
        });
        group.add(bg);
        group.add(line1);
        group.add(line2);
        // Same click-family subscription as startMarker so the gesture
        // works for mouse ('click'), pointer ('pointerclick') and touch
        // ('tap') alike. cancelBubble keeps the stage-level handler
        // from also processing the gesture (which in polygon mode would
        // otherwise add a vertex right where the user just cancelled).
        group.on('click.kaolin pointerclick.kaolin tap.kaolin', (e) => {
            e.cancelBubble = true;
            this.discardDraft();
        });
        return group;
    }

    private addPolygonVertex(pos: { x: number; y: number }): void {
        if (!this.draft || this.draft.mode !== 'polygon') return;
        const line = this.draft.shape as Konva.Line;
        line.points(line.points().concat([pos.x, pos.y]));
        if (line.points().length >= 6) {
            line.fillEnabled(true);
        }
        this.drawLayer?.batchDraw();
    }

    private updatePolygonPreview(pos: { x: number; y: number }): void {
        if (!this.draft || this.draft.mode !== 'polygon' || !this.draft.previewLine) return;
        const pts = (this.draft.shape as Konva.Line).points();
        const lastX = pts[pts.length - 2];
        const lastY = pts[pts.length - 1];
        this.draft.previewLine.points([lastX, lastY, pos.x, pos.y]);
        this.drawLayer?.batchDraw();
    }

    private commitPolygon(): void {
        if (!this.draft || this.draft.mode !== 'polygon') return;
        const line = this.draft.shape as Konva.Line;
        if (line.points().length < 6) {
            this.discardDraft();
            return;
        }
        // Konva.Line only runs fillStrokeShape when `closed` is true; otherwise
        // its scene function only strokes the open path and silently ignores
        // fill (regardless of fillEnabled). The polygon is drawn open during
        // editing (so the trailing segment isn't visually closed back to the
        // start), but at commit time we need the closed path so:
        //   - base behavior: the persisted polygon renders as a filled shape;
        //   - selection subclass: the rasterized mask actually contains pixels
        //     when the subclass restyles to a white fill and captures.
        line.closed(true);
        this.commitDraft();
    }

    // ---- Commit / cleanup ----

    /**
     * Final shared commit funnel for every mode. Calls the subclass hook
     * {@link onDraftCommitted} with the validated draft, then tears down
     * draft bookkeeping (preview line, start marker, pending rAF) without
     * touching the main shape -- the hook is responsible for deciding the
     * shape's fate (keep on layer, rasterize, destroy, etc).
     *
     * `private` on purpose: subclasses extend behavior via the hook, not by
     * intercepting the funnel.
     */
    private commitDraft(): void {
        if (!this.draft) return;
        const draft = this.draft;
        this.onDraftCommitted(draft);
        if (draft.rafId !== undefined) {
            window.cancelAnimationFrame(draft.rafId);
        }
        draft.previewLine?.destroy();
        draft.startMarker?.off('.kaolin');
        draft.startMarker?.destroy();
        draft.cancelMarker?.off('.kaolin');
        draft.cancelMarker?.destroy();
        this.draft = null;
        this.drawLayer?.batchDraw();
    }

    /**
     * Subclass hook invoked exactly once per successful gesture, after the
     * mode-specific validation passes and before the base tears down the
     * draft helpers. The draft is still styled in draft colors and the main
     * shape is still parented to {@link drawLayer}.
     *
     * Default implementation is a no-op, so the shape persists on the draw
     * layer as a permanent Konva object. Override to rasterize, relocate, or
     * destroy. The shape is shared with the base, so destroying it is fine
     * (the base's cleanup only handles the helper nodes).
     */
    protected onDraftCommitted(_draft: DraftState): void {}

    /**
     * Abort any in-progress gesture by destroying every node owned by the
     * current draft (shape, preview line, start marker) and clearing state.
     * Used by mode-switch, pointer cancel, and the validation-failure paths
     * in each mode's `commit*`.
     *
     * `protected` so subclasses (or future helpers) can wipe state during
     * teardown; not part of the commit pipeline.
     */
    protected discardDraft(): void {
        if (!this.draft) return;
        if (this.draft.rafId !== undefined) {
            window.cancelAnimationFrame(this.draft.rafId);
        }
        this.draft.previewLine?.destroy();
        this.draft.startMarker?.off('.kaolin');
        this.draft.startMarker?.destroy();
        this.draft.cancelMarker?.off('.kaolin');
        this.draft.cancelMarker?.destroy();
        this.draft.shape.destroy();
        this.draft = null;
        this.drawLayer?.batchDraw();
    }

    private restyleDraft(): void {
        if (!this.draft) return;
        const stroke = this.options.color;
        const fill = withAlpha(this.options.color, this.options.opacity);
        const sw = this.options.overlayStrokeWidth;
        const shape = this.draft.shape;
        shape.stroke(stroke);
        shape.strokeWidth(sw);
        (shape as any).fill?.(fill);
        if (this.draft.previewLine) {
            this.draft.previewLine.stroke(stroke);
            this.draft.previewLine.strokeWidth(sw);
        }
        if (this.draft.startMarker) {
            this.draft.startMarker.stroke(stroke);
            this.draft.startMarker.strokeWidth(sw);
            this.draft.startMarker.fill(fill);
            this.draft.startMarker.radius(this.options.polygonCloseRadiusPx);
        }
        this.drawLayer?.batchDraw();
    }

    private handleResize(): void {
        if (!this.stage || !this.element) return;
        const w = Math.max(1, this.element.clientWidth);
        const h = Math.max(1, this.element.clientHeight);
        if (this.stage.width() === w && this.stage.height() === h) return;
        this.stage.width(w);
        this.stage.height(h);
        this.drawLayer?.batchDraw();
    }

    private disposeStage(): void {
        this.discardDraft();
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
            this.resizeObserver = null;
        }
        if (this.stage) {
            this.stage.off('.kaolin');
            this.stage.destroy();
            this.stage = null;
        }
        // Layer is owned by the destroyed stage; just clear our reference.
        this.drawLayer = null;
    }
}


BehaviorRegister.register('konva_drawing', KonvaDrawingBehavior,
    'Konva-backed in-stage drawing (box / polygon / freeform). Committed ' +
    'gestures remain on the Konva draw layer as persistent shapes; subclass ' +
    '(konva_selection) overrides the commit hook to rasterize onto an ' +
    'external composite canvas instead.');
