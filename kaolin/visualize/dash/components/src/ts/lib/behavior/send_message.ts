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
 * Behaviors and a shared hook for publishing values (generic, camera) over a
 * registered WebSocket connection, with throttling and change detection.
 *
 * @module
 */

import React, { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from 'react';
import { z } from 'zod';

import { BehaviorRegister } from '../../core/behavior';
import * as io from '../../core/io';
import { WebSocketConnectionsManager } from '../../core/sockets';
import { logger } from '../../util/logging';
import { getFunctionByNameOrThrow } from '../../util/types';


/**
 * Configuration schema for {@link SendValueBehaviorComponent}, also describing
 * every argument accepted by {@link useSendValue}. This is the single source of
 * truth: the TypeScript option type {@link SendValueOptions} is derived via
 * `z.infer`, and the per-behavior schemas for `send_camera` and `send_cursor`
 * are projections of this one (`.pick(...)` / `.extend(...)`).
 *
 * Keep the defaults in sync with the destructured prop defaults in the
 * components below.
 *
 * ```typescript
 * z.object({
 *     connectionId: z.string()
 *         .describe('Id of the WebSocket connection (must be registered via add_websocket_connection).'),
 *     messageTag: z.string()
 *         .describe('Tag prepended to each outgoing message; routes to the matching server-side handler.'),
 *     minUpdateMs: z.number().min(0).default(100)
 *         .describe('Minimum time (ms) between sends. Reduces server load.'),
 *     binary: z.boolean().default(true)
 *         .describe('Serialize as kaolin binary; otherwise JSON.'),
 *     needsUpdateFunctionName: z.string().optional()
 *         .describe('Optional global-scope function name deciding if a new value warrants resending (default: shallow inequality).')
 *         .meta({ uiBound: false }),
 *     toSerializableFunctionName: z.string().optional()
 *         .describe('Optional global-scope function name converting the value to a serializable payload (default: identity).')
 *         .meta({ uiBound: false }),
 * })
 * ```
 *
 * @group Behavior: SendValueBehaviorComponent
 */
export const SendValueOptionsSchema = z.object({
    connectionId: z.string()
        .describe('Id of the WebSocket connection (must be registered via add_websocket_connection).'),
    messageTag: z.string()
        .describe('Tag prepended to each outgoing message; routes to the matching server-side handler.'),
    minUpdateMs: z.number().min(0).default(100)
        .describe('Minimum time (ms) between sends. Reduces server load.'),
    binary: z.boolean().default(true)
        .describe('Serialize as kaolin binary; otherwise JSON.'),
    needsUpdateFunctionName: z.string().optional()
        .describe('Optional global-scope function name deciding if a new value warrants resending (default: shallow inequality).')
        .meta({ uiBound: false }),
    toSerializableFunctionName: z.string().optional()
        .describe('Optional global-scope function name converting the value to a serializable payload (default: identity).')
        .meta({ uiBound: false }),
});

/**
 * Options for {@link SendValueBehaviorComponent}, parsed from
 * {@link SendValueOptionsSchema}. Also the prop type of {@link useSendValue}.
 *
 * @group Behavior: SendValueBehaviorComponent
 */
export type SendValueOptions = z.infer<typeof SendValueOptionsSchema>;

/** Imperative handle exposing `setValue` for a value-sending behavior.
 *
 * @group Behavior: SendValueBehaviorComponent
 */
export interface SendValueHandle<T> {
    setValue: (value: T, force?: boolean) => void;
}


/**
 * Hook that manages sending a value over WebSocket. When setValue is called,
 * it checks if the value has changed sufficiently. If it has, it serializes it
 * as a message and sends the message on the connection.
 *
 * The optional second argument `force` bypasses the `minUpdateMs` throttle and
 * the `needsUpdateFunction` change-detection, unconditionally encoding and
 * sending the value. Use this for one-shot "boundary" events (e.g. an
 * end-of-interaction signal) that must not be collapsed by throttling against
 * the immediately preceding event.
 *
 * @param options - Connection id, message tag, throttling, and serialization
 *                   settings (see {@link SendValueOptionsSchema}).
 * @returns A `setValue(value, force?)` callback that sends `value` on the connection.
 * @group Behavior: SendValueBehaviorComponent
 */
export function useSendValue<T>({
    needsUpdateFunctionName,
    toSerializableFunctionName,
    messageTag,
    connectionId,
    minUpdateMs,
    binary
}: SendValueOptions): (value: T, force?: boolean) => void {
    const needsUpdateFunction = useRef<((prev: T | null, curr: T) => boolean) | null>(null);
    const toSerializableFunction = useRef<((val: T) => any) | null>(null);

    const lastUpdateDate = useRef<Date | null>(null);
    const prevValue = useRef<T | null>(null);

    // Initialize update function
    useEffect(() => {
        needsUpdateFunction.current = getFunctionByNameOrThrow(
            needsUpdateFunctionName, (prev: T, current: T): boolean => { return prev != current; }
        ) as (prev: T | null, curr: T) => boolean;
    }, [needsUpdateFunctionName]);

    // Initialize toSerializable function
    useEffect(() => {
        toSerializableFunction.current = getFunctionByNameOrThrow(
            toSerializableFunctionName, (val: T) => { return val; }
        ) as (val: T) => any;
    }, [toSerializableFunctionName]);

    const getConnection = useCallback(async () => {
        const errorPrefix = `Connection with id ${connectionId}`;
        const errorSuffix = `cannot send message with tag ${messageTag}`;
        if (!WebSocketConnectionsManager.hasConnection(connectionId)) {
            logger.error(`${errorPrefix} not registered; ${errorSuffix}`);
            return null;
        }
        let ws = WebSocketConnectionsManager.getOpenConnection(connectionId);
        const isFirst = lastUpdateDate.current === null;
        if (!ws && isFirst) {
            const MAX_RETRIES = 10;
            const RETRY_DELAY_MS = 200;
            for (let i = 0; i < MAX_RETRIES && !ws; i++) {
                await new Promise(resolve => setTimeout(resolve, RETRY_DELAY_MS));
                ws = WebSocketConnectionsManager.getOpenConnection(connectionId);
            }
        }
        if (!ws) {
            logger.error(`${errorPrefix} not yet open; ${errorSuffix}`);
        }
        return ws;
    }, [connectionId]);

    const encodeMessage = useCallback(async (val: T) => {
        const message = {
            [io.MESSAGE_TAG_KEY]: messageTag,
            [io.MESSAGE_CONTENT_KEY]: toSerializableFunction.current!(val)
        };
        if (binary) {
            return await io.toBinary(message);
        } else {
            return io.toJSON(message);
        }
    }, [messageTag, binary]);

    const setValue = useCallback(async (value: T, force: boolean = false) => {
        const now = new Date();
        if (!force) {
            const throttleOk = !lastUpdateDate.current ||
                now.valueOf() - lastUpdateDate.current.valueOf() > minUpdateMs;
            const changed = needsUpdateFunction.current!(prevValue.current, value);
            if (!(throttleOk && changed)) return;
        }
        const message = await encodeMessage(value);
        
        const ws = await getConnection();
        if (ws) {
            ws.send(message);
            lastUpdateDate.current = now;
            prevValue.current = value;
        }
    }, [minUpdateMs, encodeMessage, getConnection]);

    return setValue;
}

/** Imperative handle exposing `setValue` for the generic value-sending behavior.
 *
 * @group Behavior: SendValueBehaviorComponent
 */
export interface SendValueBehaviorHandle {
    setValue: (value: any) => void;
}

/**
 * Generic React-component behavior for sending any value over a registered
 * WebSocket connection. Exposes a `setValue` imperative handle and reuses
 * {@link useSendValue} for throttling, change detection, and serialization.
 *
 * Registered name: `'send_value'`.
 *
 * Configuration schema: {@link SendValueOptionsSchema}.
 *
 * @group Behavior: SendValueBehaviorComponent
 */
export const SendValueBehaviorComponent = forwardRef<
    SendValueBehaviorHandle,
    SendValueOptions
>(({
    connectionId,
    messageTag,
    minUpdateMs = 100,
    binary = true,
    needsUpdateFunctionName,
    toSerializableFunctionName
}, ref) => {
    const setValue = useSendValue<any>({
        needsUpdateFunctionName,
        toSerializableFunctionName,
        messageTag,
        connectionId,
        minUpdateMs,
        binary
    });

    useImperativeHandle(ref, () => ({
        setValue
    }), [setValue]);

    return null;
});

(SendValueBehaviorComponent as any).schema = SendValueOptionsSchema;
BehaviorRegister.register('send_value', SendValueBehaviorComponent,
    'Generic value-publisher over a registered WebSocket connection. Exposes '
    + 'a `setValue` imperative handle.');

/**
 * Configuration schema for {@link SendCameraBehaviorComponent}. Inherits every
 * option from {@link SendValueOptionsSchema} (throttling, binary vs JSON,
 * custom serializer / change detector) and overrides only what should differ
 * for camera streaming:
 *
 *   - `messageTag` defaults to `'set_camera'` (the conventional routing tag
 *     for camera updates); callers may still override it via props.
 *   - `connectionId` is hidden from the auto-UI because the camera's
 *     websocket is bound at registration time.
 *
 * `minUpdateMs` and `binary` inherit the base defaults, which are already
 * the right values for camera streaming.
 *
 * ```typescript
 * SendValueOptionsSchema.extend({
 *     messageTag: z.string().default('set_camera')
 *         .describe('Tag prepended to each outgoing camera message; routes to the matching server-side handler.'),
 *     connectionId: z.string()
 *         .describe('Id of the WebSocket connection (must be registered via add_websocket_connection).')
 *         .meta({ uiBound: false }),
 * })
 * ```
 *
 * @group Behavior: SendCameraBehaviorComponent
 */
export const SendCameraOptionsSchema = SendValueOptionsSchema.extend({
    messageTag: z.string().default('set_camera')
        .describe('Tag prepended to each outgoing camera message; routes to the matching server-side handler.'),
    connectionId: z.string()
        .describe('Id of the WebSocket connection (must be registered via add_websocket_connection).')
        .meta({ uiBound: false }),
});

/**
 * Options for {@link SendCameraBehaviorComponent}, parsed from
 * {@link SendCameraOptionsSchema}.
 *
 * @group Behavior: SendCameraBehaviorComponent
 */
export type SendCameraOptions = z.infer<typeof SendCameraOptionsSchema>;

/** Imperative handle exposing `setCamera` for the camera-sending behavior.
 *
 * @group Behavior: SendCameraBehaviorComponent
 */
export interface SendCameraHandle {
    setCamera: (camera: any) => void;
}


/**
 * React-component behavior that forwards camera updates over a registered
 * WebSocket connection, reusing {@link useSendValue} for throttling, change
 * detection, and serialization.
 *
 * Registered name: `'send_camera'`.
 *
 * Configuration schema: {@link SendCameraOptionsSchema}.
 *
 * @group Behavior: SendCameraBehaviorComponent
 */
export const SendCameraBehaviorComponent = forwardRef<
    SendCameraHandle,
    SendCameraOptions
>(({
    connectionId,
    messageTag = 'set_camera',
    minUpdateMs = 100,
    binary = true,
    needsUpdateFunctionName,
    toSerializableFunctionName,
}, ref) => {
    const setCamera = useSendValue<any>({
        needsUpdateFunctionName,
        toSerializableFunctionName,
        messageTag,
        connectionId,
        minUpdateMs,
        binary,
    });

    useImperativeHandle(ref, () => ({
        setCamera
    }), [setCamera]);

    return null;
});
(SendCameraBehaviorComponent as any).schema = SendCameraOptionsSchema;
BehaviorRegister.register('send_camera', SendCameraBehaviorComponent,
                          'Forwards camera updates over a registered WebSocket connection.');




