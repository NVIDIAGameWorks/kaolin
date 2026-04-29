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
 * Pointer/cursor payload type and helpers for translating DOM pointer events
 * into element-relative coordinates.
 *
 * @module
 */

import React from 'react';

/**
 * Payload shape produced for a single pointer/click event.
 *
 * `x`/`y` are pixel offsets from the bound element's top-left; `fracX`/`fracY`
 * are the same normalized to [0,1] over the element's display size. `type`,
 * `buttons`, `pointerType`, and `pointerId` mirror the underlying DOM event so
 * the server side can distinguish moves vs. clicks, primary vs. secondary
 * button, mouse vs. touch vs. pen, etc.
 */
export interface CursorValue {
    x: number;
    y: number;
    fracX: number;
    fracY: number;
    type: string;
    buttons: number;
    pointerType?: string;
    pointerId?: number;
}

/**
 * Position of a pointer/mouse event relative to an element, as both pixel
 * offsets from the element's top-left (`x`/`y`) and fractions of the element's
 * displayed size (`fracX`/`fracY`, in [0,1]).
 */
export interface RelativeCoordinates {
    x: number;
    y: number;
    fracX: number;
    fracY: number;
}

/**
 * Locate a pointer/mouse event within an element, returning both pixel offsets
 * from the element's top-left and the same normalized to fractions of the
 * element's displayed size.
 *
 * Uses the event's viewport coordinates (`clientX`/`clientY`) against the
 * element's viewport-relative bounding box, so the result is correct even when
 * the page is scrolled (document-relative `pageX`/`pageY` would fold in the
 * scroll offset and mislocate the point). Fractions are 0 when the element has
 * zero displayed size.
 *
 * @param event - The pointer or mouse event to read.
 * @param inputElement - Element to measure against; defaults to `event.currentTarget`.
 * @returns The event position as pixel offsets and fractions of the element.
 */
export function getRelativeCoordinates(
    event: React.PointerEvent<Element> | React.MouseEvent<Element> | PointerEvent | MouseEvent,
    inputElement?: HTMLElement | null
): RelativeCoordinates {
    const element = inputElement ?? (event.currentTarget as HTMLElement);
    const rect = element.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    return {
        x,
        y,
        fracX: rect.width > 0 ? x / rect.width : 0,
        fracY: rect.height > 0 ? y / rect.height : 0,
    };
}

/**
 * Extract a {@link CursorValue} from a pointer/click event, resolved against
 * the supplied element (defaults to `event.currentTarget`). Fractional
 * coordinates are 0 when the element has zero displayed size.
 *
 * @param event - The pointer or mouse event to read.
 * @param inputElement - Element to measure against; defaults to `event.currentTarget`.
 * @returns The cursor payload for the event.
 */
export function cursorFromEvent(
    event: React.PointerEvent<Element> | React.MouseEvent<Element> | PointerEvent | MouseEvent,
    inputElement?: HTMLElement | null
): CursorValue {
    const { x, y, fracX, fracY } = getRelativeCoordinates(event, inputElement);
    const pe = event as PointerEvent;
    return {
        x,
        y,
        fracX,
        fracY,
        type: event.type,
        buttons: pe.buttons ?? 0,
        pointerType: pe.pointerType,
        pointerId: pe.pointerId,
    };
}
