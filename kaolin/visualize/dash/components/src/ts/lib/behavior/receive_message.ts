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

import { z } from 'zod';

import { Behavior, BehaviorRegister, MessageHandler } from '../../core/behavior';
import * as kaolin_events from '../../core/event';
import { getFunctionByNameOrThrow } from '../../util/types';


/**
 * Configuration schema for {@link ReceiveMessageBehavior}. Note that the schema
 * captures the *constructor input shape* — raw strings + tuple arrays as they
 * arrive from the Python side. Derived runtime state (the resolved process
 * function, the alert-behaviors `Map`) lives on the class instance, separately
 * from `this.options`, so the schema/setOption contract stays JSON-shaped.
 *
 * ```typescript
 * z.object({
 *     tag: z.string().default('')
 *         .describe('Inbound message tag this behavior subscribes to.'),
 *     msgProcessFunctionName: z.any().default(null as string | null)
 *         .describe('Global-scope function name applied to every incoming '
 *                   + 'message before fan-out (default: identity).')
 *         .meta({ kind: 'any', uiBound: false }),
 *     alertBehaviors: z.any().default([] as [string, string][])
 *         .describe('Pairs of `(behaviorId, setterName)` invoked with the '
 *                   + 'processed message on each receipt.')
 *         .meta({ kind: 'any', uiBound: false }),
 * })
 * ```
 *
 * @group Behavior: ReceiveMessageBehavior
 */
export const ReceiveMessageOptionsSchema = z.object({
    tag: z.string()
        .describe('Inbound message tag this behavior subscribes to.'),
    msgProcessFunctionName: z.any().default(null as string | null)
        .describe('Global-scope function name applied to every incoming '
                  + 'message before fan-out (default: identity).')
        .meta({ kind: 'any', uiBound: false }),
    // `[behaviorId, setterName]` pairs. Resolved into a `Map` on
    // construction / option change; auto-UI does not edit this directly.
    alertBehaviors: z.any().default([] as [string, string][])
        .describe('Pairs of `(behaviorId, setterName)` invoked with the '
                  + 'processed message on each receipt.')
        .meta({ kind: 'any', uiBound: false }),
});

/**
 * Options for {@link ReceiveMessageBehavior}, parsed from
 * {@link ReceiveMessageOptionsSchema}.
 *
 * @group Behavior: ReceiveMessageBehavior
 */
export type ReceiveMessageOptions = z.infer<typeof ReceiveMessageOptionsSchema>;


/**
 * A **behavior** for the Kaolin viewer that relays incoming WebSocket messages
 * to other behaviors. On each message whose tag matches `options.tag`, it
 * applies an optional global-scope process function and then invokes
 * `setterName(processedMessage)` on every `(behaviorId, setterName)` pair in
 * `options.alertBehaviors`.
 *
 * This is a {@link core.behavior.MessageHandler | MessageHandler}. It is not
 * element-bound (it has no DOM presence).
 *
 * Registered name: `'receive_message'`.
 *
 * Configuration schema: {@link ReceiveMessageOptionsSchema}.
 *
 * @example
 * Python-side wiring: receive each `mesh` message, convert it to a renderable
 * mesh via a global JS function, and forward the result to a renderer
 * behavior's `setMesh`. Each entry is a behavior spec
 * `(behaviorId, registeredName, elementType, options?)`:
 *
 * ```python
 * ('receive-mesh', 'receive_message', None, {
 *     'tag': 'mesh',
 *     'msgProcessFunctionName': 'kaolin.util.graphics.meshFromMessage',
 *     'alertBehaviors': [['render-mesh', 'setMesh']],
 * })
 * ('render-mesh', 'three_render_mesh', 'three-canvas')
 * ```
 *
 * @group Behavior: ReceiveMessageBehavior
 */
export class ReceiveMessageBehavior extends Behavior implements MessageHandler {
    static override schema = ReceiveMessageOptionsSchema;

    override options: ReceiveMessageOptions;

    /** Resolved global-scope process function (memoized from
     *  `options.msgProcessFunctionName`). Defaults to identity. */
    private _msgProcessFunction: (message: any) => any;
    /** Memoized form of `options.alertBehaviors` for O(1) iteration. */
    private _alertBehaviors: Map<string, string>;

    constructor(options: Partial<ReceiveMessageOptions> = {}) {
        super();
        this.options = ReceiveMessageOptionsSchema.parse(options);
        this._msgProcessFunction = getFunctionByNameOrThrow(
            this.options.msgProcessFunctionName, (message: any) => message
        ) as (message: any) => any;
        this._alertBehaviors = new Map(this.options.alertBehaviors ?? []);
    }

    acceptedMessageTags(): string[] {
        return [this.options.tag];
    }

    onConnectionOpen(): void {}

    onMessage(messageTag: string, messageContent: any | null): void {
        // Ignore messages whose tag this behavior is not subscribed to.
        if (!this.acceptedMessageTags().includes(messageTag)) {
            return;
        }
        const processed = this.processMessage(messageContent);
        for (const [name, setterName] of this._alertBehaviors.entries()) {
            // TODO: must keep relevant viewer name in the behavior and use it here to allow multi-viewport
            kaolin_events.requestBehaviorEdit(name, setterName, processed);
        }
    }

    private processMessage(messageContent: any | null): any {
        return this._msgProcessFunction(messageContent);
    }
}
BehaviorRegister.register('receive_message', ReceiveMessageBehavior,
    'WebSocket message subscriber: applies an optional process function then '
    + 'fans out to a list of other behaviors via `setterName(processedMessage)`.');
