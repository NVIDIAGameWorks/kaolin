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
 * Contract, base class, and type-guard for behaviors that receive tagged
 * messages from the viewer's connection. A handler declares which message tags
 * it accepts and is notified on connection open and on each matching message.
 *
 * @module
 */

import { makeImplementsInterfaceFunction } from '../../util/types';

/**
 * For behaviors implementing this interface, Kaolin viewer will automatically
 * rouute all incoming messages, filtered by tags. 
 *
 * @group Key Behavior Interfaces
 */
export interface MessageHandler {
    acceptedMessageTags(): string[];
    onConnectionOpen(): void;
    onMessage(messageTag: string, messageContent: any | null): void;
};

/**
 * Convenience base class for {@link MessageHandler} implementations: stores the
 * set of accepted message tags and provides no-op lifecycle defaults. Subclasses
 * typically override {@link onMessage} (and optionally {@link onConnectionOpen}).
 *
 * @group Behavior Base Classes
 */
export class MessageHandlerBase implements MessageHandler {
    protected tags: string[];

    /**
     * @param tags - Message tags this handler should be subscribed to.
     */
    constructor(tags: string[] = []) {
        this.tags = tags;
    }

    /**
     * @returns The message tags this handler accepts.
     */
    acceptedMessageTags(): string[] {
        return this.tags;
    }

    /** Called once when the viewer connection opens; no-op by default. */
    onConnectionOpen(): void {}

    /**
     * Called for each incoming message whose tag is accepted, receiving the
     * message tag and its decoded content (`null` when empty); no-op by default.
     */
    onMessage(_messageTag: string, _messageContent: any | null): void {}
};


/**
 * Runtime type-guard that reports whether a value implements
 * {@link MessageHandler}. Fails at compile time if the interface changes.
 *
 * @param object - Candidate value to test.
 * @returns `true` if `object` implements every {@link MessageHandler} member.
 *
 * @group Key Behavior Interfaces
 */
export const isMessageHandler =
    makeImplementsInterfaceFunction<MessageHandler>({
        acceptedMessageTags: true,
        onConnectionOpen: true,
        onMessage: true
    });
