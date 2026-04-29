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

// NOTE: onMessage fans out via kaolin_events.requestBehaviorEdit, which
// dispatches a CustomEvent on globalThis; happy-dom makes globalThis a real
// EventTarget, so the test simply listens for those events to observe fan-out.

import { assert } from 'chai';
import {
    ReceiveMessageBehavior,
} from '@kaolin/lib/behavior/receive_message';
import { isMessageHandler } from '@kaolin/core/behavior';
import { EditBehaviorCommand, ViewerCustomEvent, customViewerEventName } from '@kaolin/core/event';
import { registerDom, unregisterDom } from '@test/helpers/dom';

describe('visualize/dash/components/src/lib/behavior/test_receive_message.ts', () => {

    before(registerDom);
    after(unregisterDom);

    afterEach(() => { delete (window as any).__recvTest; });

    /** Record every EDIT_BEHAVIOR fan-out event; call restore() when done. */
    function captureBehaviorEdits(): { edits: EditBehaviorCommand[]; restore: () => void } {
        const edits: EditBehaviorCommand[] = [];
        const eventName = customViewerEventName(ViewerCustomEvent.EDIT_BEHAVIOR);  // default viewer id
        const listener = (e: Event) => edits.push((e as CustomEvent).detail);
        globalThis.addEventListener(eventName, listener);
        return { edits, restore: () => globalThis.removeEventListener(eventName, listener) };
    }

    describe('ReceiveMessageBehavior respects options', () => {
        it('requires a tag, accepts only that tag, and is a MessageHandler', () => {
            assert.throws(() => new ReceiveMessageBehavior({} as any),
                /tag/i, 'constructing without a tag throws (tag is required)');

            const behavior = new ReceiveMessageBehavior({ tag: 'mesh' });  // no process function
            assert.isTrue(isMessageHandler(behavior), 'satisfies the MessageHandler duck contract');
            assert.deepEqual(behavior.acceptedMessageTags(), ['mesh'], 'subscribes to exactly the configured tag');
        });
    });

    describe('ReceiveMessageBehavior basic functionality', () => {
        it('processes each payload and fans it out to every alert behavior', () => {
            (window as any).__recvTest = { double: (m: number) => m * 2 };
            const behavior = new ReceiveMessageBehavior({
                tag: 'value',
                msgProcessFunctionName: '__recvTest.double',
                alertBehaviors: [['beh-a', 'setX'], ['beh-b', 'setY']],
            });

            const { edits, restore } = captureBehaviorEdits();
            try {
                behavior.onMessage('value', 5);
            } finally { restore(); }

            assert.deepEqual(edits, [
                { behaviorId: 'beh-a', setterName: 'setX', value: 10 },
                { behaviorId: 'beh-b', setterName: 'setY', value: 10 },
            ], 'each alert behavior is invoked with its setter and the processed (doubled) value');

            // A message with a non-matching tag is ignored: no fan-out at all.
            const other = captureBehaviorEdits();
            try {
                behavior.onMessage('other-tag', 5);
            } finally { other.restore(); }
            assert.isEmpty(other.edits, 'a message with the wrong tag triggers no fan-out');
        });

        it('forwards the payload unchanged when no process function is configured', () => {
            const behavior = new ReceiveMessageBehavior({ tag: 'value', alertBehaviors: [['beh', 'set']] });

            const { edits, restore } = captureBehaviorEdits();
            try {
                behavior.onMessage('value', { a: 1 });
            } finally { restore(); }

            assert.deepEqual(edits, [{ behaviorId: 'beh', setterName: 'set', value: { a: 1 } }],
                'the identity default forwards the original payload');
        });
    });

});
