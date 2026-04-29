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
 * Test-support behavior that records the pointer/click interactions routed to
 * it into an in-memory log, plus the record type and the
 * {@link requestBehaviorEdit}-friendly reporting surface used to read it back.
 *
 * @module
 */

import { z } from 'zod';

import {
    BehaviorInterface, BehaviorRegister, ClickEventHandler, HandlerReturnInfo, InteractiveBehavior, PointerEventHandler,
} from '../../core/behavior';
import { requestBehaviorEdit } from '../../core/event';
import { logger } from '../../util/logging';


/**
 * The behavior methods that {@link RecordInteractionsBehavior} can capture: the
 * {@link core.behavior.PointerEventHandler | PointerEventHandler} and
 * {@link core.behavior.ClickEventHandler | ClickEventHandler} methods plus
 * `setOption`. Each enum value is the exact method name (e.g.
 * `'onClick'`, `'setOption'`), so it doubles as the tag stored on a
 * {@link RecordedInteraction}, as a selector in the behavior's `record` option,
 * and as the dispatch target used by {@link replayInteractions}.
 *
 * @group Behavior: RecordInteractionsBehavior
 */
export enum InteractionHandlerName {
    Click = 'onClick',
    DoubleClick = 'onDoubleClick',
    PointerDown = 'onPointerDown',
    PointerMove = 'onPointerMove',
    PointerUp = 'onPointerUp',
    PointerCancel = 'onPointerCancel',
    PointerEnter = 'onPointerEnter',
    PointerLeave = 'onPointerLeave',
    SetOption = 'setOption',
}

/**
 * The pointer/click subset of {@link InteractionHandlerName} — every value
 * except {@link InteractionHandlerName.SetOption} — i.e. the names that tag a
 * {@link RecordedEventInteraction}.
 *
 * @group Behavior: RecordInteractionsBehavior
 */
export type InteractionEventHandlerName = Exclude<InteractionHandlerName, InteractionHandlerName.SetOption>;

/**
 * Self-contained, JSON-serializable snapshot of a pointer/click event, produced
 * by {@link eventToSerializable} and rebuilt by {@link eventFromSerializable}.
 *
 * Coordinates are stored **frame-independently** as fractions of the recorded
 * element's size (`fracX`/`fracY` in [0,1]); the record-time target geometry
 * (`targetWidth`/`targetHeight`) and an optional `targetRole` label are kept for
 * diagnostics and exact pixel reconstruction. The remaining fields mirror the
 * DOM event so {@link eventFromSerializable} can mint a faithful `PointerEvent`
 * (the form Konva-based behaviors read off `event.nativeEvent`).
 *
 * @group Behavior: RecordInteractionsBehavior
 */
export interface SerializableEvent {
    /** DOM `event.type`, e.g. `'pointerdown'`, `'click'`. */
    type: string;
    /** Pointer x as a fraction of the target's width ([0,1], element-relative). */
    fracX: number;
    /** Pointer y as a fraction of the target's height ([0,1], element-relative). */
    fracY: number;
    /** Record-time target rect width in px (for diagnostics / pixel reconstruction). */
    targetWidth: number;
    /** Record-time target rect height in px. */
    targetHeight: number;
    /** Optional label identifying the recorded element (e.g. `'canvas'`). */
    targetRole?: string;
    button: number;
    buttons: number;
    ctrlKey: boolean;
    shiftKey: boolean;
    altKey: boolean;
    metaKey: boolean;
    detail: number;
    pointerId?: number;
    pointerType?: string;
    pressure?: number;
    isPrimary?: boolean;
    /** Source event's `timeStamp` (ms). */
    timeStamp: number;
}

/**
 * A recorded pointer/click interaction: the {@link InteractionEventHandlerName}
 * that fired plus the {@link SerializableEvent} snapshot of the event it
 * received.
 *
 * @group Behavior: RecordInteractionsBehavior
 */
export interface RecordedEventInteraction {
    handler: InteractionEventHandlerName;
    event: SerializableEvent;
}

/**
 * A recorded {@link core.behavior.BehaviorInterface.setOption | setOption} call:
 * the option `name` written and the `value` it was set to.
 *
 * @group Behavior: RecordInteractionsBehavior
 */
export interface RecordedSetOptionInteraction {
    handler: InteractionHandlerName.SetOption;
    name: string;
    value: any;
}

/**
 * A single recorded interaction: either a pointer/click event
 * ({@link RecordedEventInteraction}) or a `setOption` call
 * ({@link RecordedSetOptionInteraction}), discriminated by `handler`.
 *
 * @group Behavior: RecordInteractionsBehavior
 */
export type RecordedInteraction = RecordedEventInteraction | RecordedSetOptionInteraction;

/**
 * Configuration schema for {@link RecordInteractionsBehavior}.
 *
 * ```typescript
 * z.object({
 *     record: z.union([z.literal('all'), z.array(z.enum(InteractionHandlerName))]).default('all')
 *         .describe('Which interaction handlers to capture: the literal "all" for every handler, '
 *             + 'or an explicit list of handler names such as "onPointerDown" / "onClick".')
 *         .meta({ uiBound: false }),
 *     elementType: z.string().default('div')
 *         .describe('DOM element type the behavior binds to (e.g. div, canvas, svg).')
 *         .meta({ uiBound: false }),
 * })
 * ```
 *
 * @group Behavior: RecordInteractionsBehavior
 */
export const RecordInteractionsOptionsSchema = z.object({
    record: z.union([z.literal('all'), z.array(z.enum(InteractionHandlerName))]).default('all')
        .describe('Which interaction handlers to capture: the literal "all" for every handler, '
            + 'or an explicit list of handler names such as "onPointerDown" / "onClick".')
        .meta({ uiBound: false }),
    elementType: z.string().default('div')
        .describe('DOM element type the behavior binds to (e.g. div, canvas, svg). '
            + 'Determines which layer the viewer attaches and which element coordinates are measured against.')
        .meta({ uiBound: false }),
});

/**
 * Options for {@link RecordInteractionsBehavior}, parsed from
 * {@link RecordInteractionsOptionsSchema}.
 *
 * @group Behavior: RecordInteractionsBehavior
 */
export type RecordInteractionsOptions = z.infer<typeof RecordInteractionsOptionsSchema>;

/**
 * Callback invoked with a snapshot of the recorded interactions.
 *
 * @group Behavior: RecordInteractionsBehavior
 */
export type InteractionsReportCallback = (records: RecordedInteraction[]) => void;

/**
 * Setter name {@link RecordInteractionsBehavior} listens on for
 * {@link requestBehaviorEdit} report requests; see
 * {@link RecordInteractionsBehavior.reportInteractions} and
 * {@link RecordInteractionsBehavior.requestInteractions}.
 *
 * @group Behavior: RecordInteractionsBehavior
 */
export const REPORT_INTERACTIONS_SETTER_NAME = 'reportInteractions';


/**
 * A **behavior** for the Kaolin viewer that records the pointer/click events
 * routed to it. It implements the full
 * {@link core.behavior.PointerEventHandler | PointerEventHandler} and
 * {@link core.behavior.ClickEventHandler | ClickEventHandler} surface; each
 * invocation of a handler selected by the `record` option appends
 * a {@link RecordedInteraction} (handler name, DOM event type, element-relative
 * {@link CursorValue}, and timestamp) to an internal log.
 *
 * Intended as test scaffolding: attach it to a viewer (or drive its handlers
 * directly) to capture the interaction stream another behavior would receive,
 * then assert against {@link getRecords}.
 *
 * Reading the log back:
 *   - With a direct reference, call {@link getRecords} (returns a copy).
 *   - With only a behavior id (e.g. an integration test that talks to the
 *     viewer over the {@link requestBehaviorEdit} event channel), send a
 *     callback through {@link reportInteractions}. The viewer's edit dispatcher
 *     invokes `behavior.reportInteractions(callback)`, which calls back with the
 *     current snapshot. {@link RecordInteractionsBehavior.requestInteractions}
 *     wraps this round-trip.
 *
 * {@link reset} clears the log. This is an
 * {@link core.behavior.InteractiveBehavior | InteractiveBehavior}
 * (an {@link core.behavior.ElementBoundBehavior | ElementBoundBehavior}).
 *
 * Registered name: `'record_interactions'`.
 *
 * Configuration schema: {@link RecordInteractionsOptionsSchema}.
 *
 * @group Behavior: RecordInteractionsBehavior
 */
export class RecordInteractionsBehavior extends InteractiveBehavior {
    static override schema = RecordInteractionsOptionsSchema;

    override options: RecordInteractionsOptions;

    /** Ordered log of captured interactions; cleared by {@link reset}. */
    private records: RecordedInteraction[];

    constructor(options: Partial<RecordInteractionsOptions> = {}) {
        super();
        this.options = RecordInteractionsOptionsSchema.parse(options ?? {});
        this.records = [];
    }

    override elementType(): string {
        return this.options.elementType;
    }

    /** Whether the `record` option selects the given handler for capture. */
    private shouldRecord(handler: InteractionHandlerName): boolean {
        return this.options.record === 'all' || this.options.record.includes(handler);
    }

    /**
     * Append one interaction to the log when its handler is selected by the
     * `record` option, resolving coordinates against the bound element (falling
     * back to `event.currentTarget` when unbound).
     */
    private record(handler: InteractionEventHandlerName, event: React.MouseEvent | React.PointerEvent): void {
        if (!this.shouldRecord(handler)) {
            return;
        }
        this.records.push({
            handler,
            event: eventToSerializable(event, this.element, this.options.elementType),
        });
    }

    /**
     * Record a `setOption` call when selected by the `record` option, then apply
     * it via the base implementation. The {@link RecordedSetOptionInteraction}
     * captures the option `name` and `value` so {@link replayInteractions} can
     * reissue the same change on another behavior.
     *
     * @param name - Option name being set.
     * @param value - New option value.
     */
    override setOption(name: string, value: any): void {
        if (this.shouldRecord(InteractionHandlerName.SetOption)) {
            this.records.push({ handler: InteractionHandlerName.SetOption, name, value });
        }
        super.setOption(name, value);
    }

    override onClick(event: React.MouseEvent): void | HandlerReturnInfo {
        this.record(InteractionHandlerName.Click, event);
    }

    override onDoubleClick(event: React.MouseEvent): void | HandlerReturnInfo {
        this.record(InteractionHandlerName.DoubleClick, event);
    }

    override onPointerDown(event: React.PointerEvent): void | HandlerReturnInfo {
        this.record(InteractionHandlerName.PointerDown, event);
    }

    override onPointerMove(event: React.PointerEvent): void | HandlerReturnInfo {
        this.record(InteractionHandlerName.PointerMove, event);
    }

    override onPointerUp(event: React.PointerEvent): void | HandlerReturnInfo {
        this.record(InteractionHandlerName.PointerUp, event);
    }

    override onPointerCancel(event: React.PointerEvent): void | HandlerReturnInfo {
        this.record(InteractionHandlerName.PointerCancel, event);
    }

    override onPointerEnter(event: React.PointerEvent): void | HandlerReturnInfo {
        this.record(InteractionHandlerName.PointerEnter, event);
    }

    override onPointerLeave(event: React.PointerEvent): void | HandlerReturnInfo {
        this.record(InteractionHandlerName.PointerLeave, event);
    }

    /**
     * Snapshot of the recorded interactions, in capture order. Returns a shallow
     * copy of the array so callers can hold or mutate it without disturbing the
     * live log.
     *
     * @returns A copy of the recorded interactions, oldest first.
     */
    getRecords(): RecordedInteraction[] {
        return [...this.records];
    }

    /**
     * Report the current {@link getRecords} snapshot to `callback`.
     *
     * This is the {@link requestBehaviorEdit}-facing entry point: the viewer's
     * edit dispatcher calls `behavior.reportInteractions(value)` with `value`
     * being the supplied callback, letting code that only holds a behavior id
     * pull the log out over the event channel. Prefer
     * {@link RecordInteractionsBehavior.requestInteractions} to issue the
     * request.
     *
     * @param callback - Receives the recorded-interactions snapshot.
     */
    reportInteractions(callback: InteractionsReportCallback): void {
        if (typeof callback !== 'function') {
            logger.error('RecordInteractionsBehavior.reportInteractions: expected a callback function, '
                + `got ${typeof callback}.`);
            return;
        }
        callback(this.getRecords());
    }

    /**
     * Clears the recorded-interactions log. Option state is preserved.
     *
     * @param _options - Unused; present to satisfy the reset contract.
     */
    override reset(_options?: any): void {
        this.records = [];
    }

    /**
     * Request the recorded interactions from a registered
     * {@link RecordInteractionsBehavior} by id, over the
     * {@link requestBehaviorEdit} event channel. `callback` is invoked
     * synchronously with the snapshot when the target viewer dispatches the
     * edit to the behavior.
     *
     * @param behaviorId - Id the behavior was registered under in the viewer.
     * @param callback - Receives the recorded-interactions snapshot.
     * @param viewerId - Target viewer id (defaults to the standard viewer id).
     */
    static requestInteractions(
        behaviorId: string,
        callback: InteractionsReportCallback,
        viewerId: string = 'kaolin-viewer',
    ): void {
        requestBehaviorEdit(behaviorId, REPORT_INTERACTIONS_SETTER_NAME, callback, viewerId);
    }
}

BehaviorRegister.register('record_interactions', RecordInteractionsBehavior,
    'Records pointer/click interactions into an in-memory log; test scaffolding for exercising other behaviors.');


/**
 * Target a {@link replayInteractions} can drive: anything implementing
 * {@link core.behavior.BehaviorInterface.setOption | setOption} and (optionally)
 * the pointer/click handlers, such as any
 * {@link core.behavior.InteractiveBehavior | InteractiveBehavior}. `element` is
 * read, when present, to resolve replayed coordinates against the bound element.
 *
 * @group Behavior: RecordInteractionsBehavior
 */
export type ReplayTarget = BehaviorInterface & PointerEventHandler & ClickEventHandler & { element?: HTMLElement | null };

/**
 * Replay a recorded interaction list against a behavior, in order.
 *
 * For each {@link RecordedEventInteraction} the matching pointer/click handler
 * (named by `handler`) is invoked with a synthetic event reconstructed from the
 * recorded {@link CursorValue}; coordinates are resolved against the target's
 * bound `element` when available. For each {@link RecordedSetOptionInteraction}
 * the target's {@link core.behavior.BehaviorInterface.setOption | setOption} is
 * called. Interactions whose handler the target does not implement are skipped
 * with a warning.
 *
 * Pairs with {@link RecordInteractionsBehavior}: capture a stream on the
 * recorder, then drive another behavior (e.g. in a unit test) with that stream.
 *
 * @param behavior - Target behavior to drive.
 * @param interactions - Interactions to replay, oldest first.
 *
 * @group Behavior: RecordInteractionsBehavior
 */
export function replayInteractions(behavior: ReplayTarget, interactions: RecordedInteraction[]): void {
    for (const interaction of interactions) {
        if (interaction.handler === InteractionHandlerName.SetOption) {
            behavior.setOption(interaction.name, interaction.value);
            continue;
        }
        const handler = (behavior as unknown as Record<string, unknown>)[interaction.handler];
        if (typeof handler !== 'function') {
            logger.warn(`replayInteractions: target behavior has no '${interaction.handler}' handler; skipping.`);
            continue;
        }
        handler.call(behavior, eventFromSerializable(interaction.event, behavior.element ?? null));
    }
}

/**
 * Capture a pointer/click event into a {@link SerializableEvent}.
 *
 * Coordinates are stored frame-independently as fractions of the target's
 * bounding rect (`fracX = (pageX - rect.left) / rect.width`, etc.), alongside
 * the record-time rect size and all the pointer/modifier metadata needed to
 * rebuild a faithful event later.
 *
 * @param event - The pointer/click event to snapshot.
 * @param element - Element coordinates are measured against; defaults to
 *     `event.currentTarget`.
 * @param targetRole - Optional label stored on the snapshot for diagnostics.
 * @returns The serializable snapshot.
 */
export function eventToSerializable(
    event: React.MouseEvent | React.PointerEvent,
    element?: HTMLElement | null,
    targetRole?: string,
): SerializableEvent {
    const target = element ?? (event.currentTarget as HTMLElement | null);
    const rect = target?.getBoundingClientRect();
    const width = rect?.width ?? 0;
    const height = rect?.height ?? 0;
    // Measure against the target's viewport-relative box. getBoundingClientRect is
    // viewport-relative, so it must be paired with clientX/clientY (also
    // viewport-relative); pageX/pageY are document-relative and would fold in the
    // page scroll offset, corrupting the stored fractions when recorded on a
    // scrolled page. This also matches the basis Konva uses at runtime (clientX).
    const offsetX = event.clientX - (rect?.left ?? 0);
    const offsetY = event.clientY - (rect?.top ?? 0);
    const pe = event as React.PointerEvent;
    return {
        type: event.type,
        fracX: width > 0 ? offsetX / width : 0,
        fracY: height > 0 ? offsetY / height : 0,
        targetWidth: width,
        targetHeight: height,
        targetRole,
        button: event.button ?? 0,
        buttons: event.buttons ?? 0,
        ctrlKey: event.ctrlKey,
        shiftKey: event.shiftKey,
        altKey: event.altKey,
        metaKey: event.metaKey,
        detail: (event as React.MouseEvent).detail ?? 0,
        pointerId: pe.pointerId,
        pointerType: pe.pointerType,
        pressure: pe.pressure,
        isPrimary: pe.isPrimary,
        timeStamp: event.timeStamp,
    };
}

/**
 * Rebuild a replayable pointer/click event from a {@link SerializableEvent}.
 *
 * Pixel coordinates are reconstructed from the stored fractions and a frame: the
 * live `element`'s bounding rect when it has a non-zero size, otherwise the
 * recorded `targetWidth`/`targetHeight` (so replay works under happy-dom, which
 * has no layout). A real `PointerEvent`/`MouseEvent` is minted and exposed as
 * `nativeEvent` — Konva-based behaviors re-dispatch using `event.nativeEvent`'s
 * `clientX`/`clientY` — while the React-level `pageX`/`pageY` (used by
 * `cursorFromEvent` / `getCanvasCoordinates`) are set to the same coordinates.
 *
 * @param data - The serialized event to reconstruct.
 * @param element - Element the coordinates resolve against, or null.
 * @returns A synthetic event with a real `nativeEvent` and no-op
 *     `preventDefault` / `stopPropagation`.
 */
export function eventFromSerializable(data: SerializableEvent, element?: HTMLElement | null): React.PointerEvent {
    const rect = element?.getBoundingClientRect();
    const width = rect?.width || data.targetWidth;
    const height = rect?.height || data.targetHeight;
    const left = rect?.left ?? 0;
    const top = rect?.top ?? 0;
    const clientX = left + data.fracX * width;
    const clientY = top + data.fracY * height;
    const init: PointerEventInit = {
        bubbles: true,
        cancelable: true,
        composed: true,
        clientX,
        clientY,
        screenX: clientX,
        screenY: clientY,
        button: data.button,
        buttons: data.buttons,
        ctrlKey: data.ctrlKey,
        shiftKey: data.shiftKey,
        altKey: data.altKey,
        metaKey: data.metaKey,
        detail: data.detail,
        pointerId: data.pointerId ?? 1,
        pointerType: data.pointerType ?? 'mouse',
        pressure: data.pressure ?? 0,
        isPrimary: data.isPrimary ?? true,
    };
    const nativeEvent = data.type.startsWith('pointer')
        ? new PointerEvent(data.type, init)
        : new MouseEvent(data.type, init as MouseEventInit);
    return {
        type: data.type,
        timeStamp: data.timeStamp,
        pageX: clientX,
        pageY: clientY,
        clientX,
        clientY,
        button: data.button,
        buttons: data.buttons,
        ctrlKey: data.ctrlKey,
        shiftKey: data.shiftKey,
        altKey: data.altKey,
        metaKey: data.metaKey,
        detail: data.detail,
        pointerId: data.pointerId,
        pointerType: data.pointerType,
        pressure: data.pressure,
        isPrimary: data.isPrimary,
        currentTarget: element,
        target: element,
        nativeEvent,
        preventDefault: () => {},
        stopPropagation: () => {},
    } as unknown as React.PointerEvent;
}
