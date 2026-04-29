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
 * Behavior that publishes pointer/cursor updates from a bound element over a
 * registered WebSocket connection.
 *
 * @module
 */

import React, { forwardRef, useCallback, useImperativeHandle, useRef } from 'react';
import { z } from 'zod';

import { BehaviorRegister, ClickEventHandler, ElementBoundBehavior, PointerEventHandler } from '../../core/behavior';
import * as kaolin_events from '../../core/event';
import { CursorValue, cursorFromEvent } from '../../util/cursor';
import { SendValueOptionsSchema, useSendValue } from './send_message';


/**
 * Configuration schema for {@link SendCursorBehaviorComponent}. Mirrors
 * {@link SendValueOptionsSchema} so the controls surface (connection id,
 * throttling, binary vs JSON, custom serializer / change detector) matches
 * `SendValueBehavior` / `SendCameraBehavior` exactly. The only overrides:
 *
 *   - `messageTag` defaults to `'set_cursor'` for ergonomic out-of-the-box use;
 *   - `elementType` selects which layer type this behavior attaches to,
 *     defaulting to the event canvas (`'canvas'`) so cursor coordinates are
 *     relative to the main viewer surface.
 *
 * ```typescript
 * SendValueOptionsSchema.extend({
 *     messageTag: z.string().default('set_cursor')
 *         .describe('Tag prepended to each outgoing cursor message; routes to the matching server-side handler.'),
 *     elementType: z.string().default('canvas')
 *         .describe('DOM element type the behavior binds to (e.g. canvas, div, svg). '
 *                   + 'Determines which layer the viewer attaches and which element '
 *                   + 'coordinates are measured against.')
 *         .meta({ uiBound: false }),
 * })
 * ```
 *
 * @group Behavior: SendCursorBehaviorComponent
 */
export const SendCursorOptionsSchema = SendValueOptionsSchema.extend({
    messageTag: z.string().default('set_cursor')
        .describe('Tag prepended to each outgoing cursor message; routes to the matching server-side handler.'),
    elementType: z.string().default('canvas')
        .describe('DOM element type the behavior binds to (e.g. canvas, div, svg). '
                  + 'Determines which layer the viewer attaches and which element '
                  + 'coordinates are measured against.')
        .meta({ uiBound: false }),
});

/**
 * Options for {@link SendCursorBehaviorComponent}, parsed from
 * {@link SendCursorOptionsSchema}.
 *
 * @group Behavior: SendCursorBehaviorComponent
 */
export type SendCursorOptions = z.infer<typeof SendCursorOptionsSchema>;


/**
 * Imperative-handle contract for {@link SendCursorBehaviorComponent}.
 *
 * Extends {@link ElementBoundBehavior} (so the viewer's
 * `behavior.setActiveElement` / `behavior.elementType` calls land) and both
 * {@link PointerEventHandler} and {@link ClickEventHandler} (so the duck-typed
 * `isPointerEventHandler` / `isClickEventHandler` checks pass and the viewer's
 * pointer/click dispatchers invoke each handler). `setValue` is also exposed so
 * callers can push synthetic cursor updates from outside the DOM event stream
 * if desired.
 *
 * @group Behavior: SendCursorBehaviorComponent
 */
export interface SendCursorHandle
    extends ElementBoundBehavior,
            PointerEventHandler,
            ClickEventHandler {
    setValue: (value: CursorValue, force?: boolean) => void;
}


/**
 * React-component behavior that publishes cursor updates from pointer/click
 * events over a registered WebSocket connection.
 *
 * Implemented as a `forwardRef` component (like `SendCameraBehaviorComponent`
 * and `ThreeRendererBehaviorComponent`) rather than an `InteractiveBehavior`
 * subclass: this lets us reuse {@link useSendValue} directly for throttling,
 * change detection, serialization, and connection lookup.
 *
 * The imperative handle satisfies the {@link ElementBoundBehavior},
 * {@link PointerEventHandler}, and {@link ClickEventHandler} duck contracts, so
 * once registered the viewer attaches the active element and routes every
 * pointer/click event through `setValue`. The on-wire payload defaults to
 * {@link CursorValue}; supply `toSerializableFunctionName` to reshape it.
 *
 * This is an {@link core.behavior.ElementBoundBehavior | ElementBoundBehavior},
 * a {@link core.behavior.PointerEventHandler | PointerEventHandler}, and a
 * {@link core.behavior.ClickEventHandler | ClickEventHandler}.
 *
 * Registered name: `'send_cursor'`.
 *
 * Configuration schema: {@link SendCursorOptionsSchema}.
 *
 * @group Behavior: SendCursorBehaviorComponent
 */
export const SendCursorBehaviorComponent = forwardRef<
    SendCursorHandle,
    SendCursorOptions
>(({
    connectionId,
    messageTag = 'set_cursor',
    minUpdateMs = 10,
    binary = true,
    needsUpdateFunctionName,
    toSerializableFunctionName,
    elementType = 'canvas',
}, ref) => {
    const elementRef = useRef<HTMLElement | null>(null);

    const setValue = useSendValue<CursorValue>({
        needsUpdateFunctionName,
        toSerializableFunctionName,
        messageTag,
        connectionId,
        minUpdateMs,
        binary,
    });

    const maybeSend = useCallback((event: React.PointerEvent | React.MouseEvent) => {
        if (!elementRef.current) return;
        // Touch lifecycle boundaries (pointerdown / pointerup / pointercancel)
        // are force-sent so they survive `minUpdateMs` throttling. Server-side
        // touch tracking depends on every begin / end arriving; pointermove
        // and friends stay throttled / deduped.
        const force = event.type === 'pointerdown'
            || event.type === 'pointerup'
            || event.type === 'pointercancel';
        setValue(cursorFromEvent(event, elementRef.current), force);
    }, [setValue]);

    useImperativeHandle(ref, () => ({
        // ElementBoundBehavior
        elementType: () => elementType,
        setActiveElement: (element: HTMLElement) => { elementRef.current = element; },
        // Pointer/click handlers — duck-checked by isPointerEventHandler / isClickEventHandler
        onClick: maybeSend,
        onDoubleClick: maybeSend,
        onPointerDown: maybeSend,
        onPointerMove: maybeSend,
        onPointerUp: maybeSend,
        onPointerCancel: maybeSend,
        onPointerEnter: maybeSend,
        onPointerLeave: maybeSend,
        // Escape hatch for callers that want to push synthetic updates
        setValue,
    }), [elementType, maybeSend, setValue]);

    return null;
});

(SendCursorBehaviorComponent as any).schema = SendCursorOptionsSchema;
BehaviorRegister.register('send_cursor', SendCursorBehaviorComponent,
    'Sends cursor position updates from pointer/click events over a registered WebSocket connection. '
    + 'Mirrors SendValueBehavior controls (`connectionId`, `messageTag`, `minUpdateMs`, `binary`, '
    + '`needsUpdateFunctionName`, `toSerializableFunctionName`).');
