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
 * Freehand brush-drawing behavior for a canvas layer, plus the small canvas
 * geometry/drawing helpers it builds on.
 *
 * @module
 */

import React from 'react';
import { z } from 'zod';

import { BehaviorRegister, CanvasBehavior, HandlerReturnInfo } from '../../core/behavior';
import { zColor } from '../../core/behavior/option';
import * as kaolin_events from '../../core/event';
import { isFixedAspect } from '../../core/viewport';
import { getCanvasCoordinates } from '../../util/canvas';


/**
 * Configuration schema for {@link DrawingBehavior}. Single source of truth: the
 * TS option type ({@link DrawingOptions}) is derived from it via `z.infer`,
 * defaults are read off it at construction, and the build-time manifest dumper
 * projects it to JSON for the Python auto-UI pipeline.
 *
 * ```typescript
 * z.object({
 *     color: zColor().default('#ff0000').describe('Brush color.'),
 *     thickness: z.number().min(1).max(60).multipleOf(1).default(9)
 *         .describe('Brush thickness in pixels.'),
 *     mode: z.enum(['draw', 'erase']).default('draw')
 *         .describe('Whether the brush paints or erases.'),
 * })
 * ```
 *
 * @group Behavior: DrawingBehavior
 */
export const DrawingOptionsSchema = z.object({
    color: zColor().default('#ff0000').describe('Brush color.'),
    thickness: z.number().min(1).max(60).multipleOf(1).default(9)
        .describe('Brush thickness in pixels.'),
    // TODO: add brush stamp and more advanced appearance controls, such as
    //       pressure sensitivity and opacity.
    mode: z.enum(['draw', 'erase']).default('draw')
        .describe('Whether the brush paints or erases.'),
});

/**
 * Options for {@link DrawingBehavior}, parsed from {@link DrawingOptionsSchema}.
 *
 * @group Behavior: DrawingBehavior
 */
export type DrawingOptions = z.infer<typeof DrawingOptionsSchema>;


/**
 * Stroke a straight line between two points using the context's current style.
 * No-op when `ctx` is null.
 *
 * @param ctx - Target 2D context, or null.
 * @param from - Line start, in canvas pixel coordinates.
 * @param to - Line end, in canvas pixel coordinates.
 * @group Behavior: DrawingBehavior
 */
function drawLine(
    ctx: CanvasRenderingContext2D | null,
    from: { x: number; y: number },
    to: { x: number; y: number }): void {
        if (!ctx) return;

        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
};


/**
 * Fill a filled dot at `point` whose radius is half the context's line width.
 * No-op when `ctx` is null.
 *
 * @param ctx - Target 2D context, or null.
 * @param point - Dot center, in canvas pixel coordinates.
 * @group Behavior: DrawingBehavior
 */
function drawPoint(
    ctx: CanvasRenderingContext2D | null,
    point: { x: number; y: number }): void {
    if (!ctx) return;

    ctx.beginPath();
    ctx.arc(point.x, point.y, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fill();
};

/**
 * A **behavior** for the Kaolin viewer that paints freehand brush strokes onto
 * a canvas in response to pointer input. Supports a paint mode and an erase
 * mode, with configurable color and thickness ({@link DrawingOptions}).
 *
 * This is a {@link core.behavior.CanvasBehavior | CanvasBehavior} (an
 * {@link core.behavior.ElementBoundBehavior | ElementBoundBehavior}) that must
 * be configured with a canvas element.
 *
 * Registered name: `'drawing'`.
 *
 * Configuration schema: {@link DrawingOptionsSchema}.
 *
 * @group Behavior: DrawingBehavior
 */
export class DrawingBehavior extends CanvasBehavior {
    static override schema = DrawingOptionsSchema;

    override options: DrawingOptions;
    isPainting: Boolean;
    lastPoint: { x: number; y: number } | null;
    /** Last size seen by `setDimensions`; the buffer itself is already resized
     *  by the time it runs, so the previous one has to be remembered here. */
    private lastSize: { width: number, height: number } | null = null;

    constructor(options: Partial<DrawingOptions> = {}) {
      super();
      this.options = DrawingOptionsSchema.parse(options);
      this.isPainting = false;
      this.lastPoint = null;
  }

  // Whole-state recompute. The generic `setOption` on `InteractiveBehavior`
  // invokes this with the changed (name, value); we ignore them since recompute
  // is cheap.
  //
  // Intentionally does NOT mutate the canvas 2D context. The brush style
  // (line width, colors and especially `globalCompositeOperation`) is applied
  // transiently at draw time inside `save()`/`restore()` (see `withBrush`).
  // A `<canvas>` has a single 2D context object that every behavior bound to
  // that layer shares; leaving persistent state here (e.g. an `'erase'` brush
  // setting `globalCompositeOperation = 'destination-out'`) would poison
  // unrelated draws on the same layer, such as `draw_remote_image`.
  override updateForOptions() {}

  /**
   * Remaps the in-progress stroke onto the resized buffer.
   *
   * The painted bitmap is this behavior's other state, and the viewer has
   * already dealt with it: it carries the strokes across, scaled, when the
   * aspect is unchanged and clears the buffer when it is not. What is left is
   * `lastPoint`, which is in buffer pixels (see `getCanvasCoordinates`) and so
   * no longer points at the same place. A uniform scale can be followed, which
   * keeps the current segment from kinking; an aspect change cannot, and the
   * strokes the segment would join up with are gone anyway, so the stroke is
   * dropped.
   *
   * `options.thickness` is deliberately left alone: it is a user-facing brush
   * width in CSS pixels (scaled to buffer pixels at draw time, see `withBrush`),
   * so it keeps its meaning on the new buffer rather than tracking the visual
   * weight of what was drawn before.
   */
  override setDimensions(width: number, height: number): void {
    const previous = this.lastSize;
    if (previous && previous.width === width && previous.height === height) return;
    this.lastSize = { width, height };
    if (!previous || !this.lastPoint) return;

    if (isFixedAspect(previous, { width, height })) {
      const scale = width / previous.width;
      this.lastPoint = { x: this.lastPoint.x * scale, y: this.lastPoint.y * scale };
    } else {
      this.reset();
    }
  }

  /** Apply this brush's style to `ctx` for the duration of `draw`, restoring
   *  the context to its prior state afterwards so the shared canvas context is
   *  never left in a non-default state. */
  private withBrush(draw: (ctx: CanvasRenderingContext2D) => void) {
    const ctx = this.ctx;
    if (!ctx) return;

    ctx.save();
    // `thickness` is in CSS pixels, the buffer is in device pixels (the viewer
    // sizes it, see core.layer.resizeLayer), so the brush is scaled by the ratio
    // between them — otherwise it thins out on a HiDPI display. Read off the
    // element rather than from devicePixelRatio, so it is right even if the two
    // are briefly out of step.
    const canvas = ctx.canvas;
    const bufferPerCssPixel = canvas.clientWidth > 0 ? canvas.width / canvas.clientWidth : 1;
    ctx.lineWidth = this.options.thickness * bufferPerCssPixel;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    if (this.options.mode === 'draw') {
        ctx.strokeStyle = this.options.color;
        ctx.fillStyle = this.options.color;
        ctx.globalCompositeOperation = 'source-over';
    } else {
        ctx.strokeStyle = 'rgba(0,0,0,1)';
        ctx.fillStyle = 'rgba(0,0,0,1)';
        ctx.globalCompositeOperation = 'destination-out';
    }
    try {
        draw(ctx);
    } finally {
        ctx.restore();
    }
  }

  
  onClick(event: React.MouseEvent) {
    this.lastPoint = getCanvasCoordinates(event);
    this.withBrush((ctx) => drawPoint(ctx, this.lastPoint!));
  }

    onPointerLeave(event: React.PointerEvent): void | HandlerReturnInfo {
        this.onPointerUp(event);
    }

    onPointerDown(event: React.PointerEvent): void | HandlerReturnInfo {
        // TODO: add modifier keys/etc.
        this.lastPoint = getCanvasCoordinates(event);
        this.isPainting = true;
        return {needsPointermove: true};
    }

    onPointerMove(event: React.PointerEvent): void | HandlerReturnInfo {
        if (!this.isPainting || !this.lastPoint) {
            return;
        }

        let controller = this;
        window.requestAnimationFrame(function() {controller.handleDrawAnimationFrame(event);});
    }

    handleDrawAnimationFrame(event: React.PointerEvent) {
        const { x, y } = getCanvasCoordinates(event, this.element as HTMLCanvasElement);
        const from = this.lastPoint;
        if (from) {
            this.withBrush((ctx) => drawLine(ctx, from, { x, y }));
        }
        this.lastPoint = { x, y };
    }

    onPointerUp(event: React.PointerEvent): void | HandlerReturnInfo {
        this.onPointerMove(event);
        this.isPainting = false;
        return {needsPointermove: false};
    }

    /** Cancel any in-progress stroke. Invoked on mode switches / behavior resets
     *  so a half-drawn stroke does not continue into the next interaction. */
    override reset() {
        this.isPainting = false;
        this.lastPoint = null;
    }

}

BehaviorRegister.register("drawing", DrawingBehavior,
    'Brush-drawing behavior on an HTMLCanvasElement.');