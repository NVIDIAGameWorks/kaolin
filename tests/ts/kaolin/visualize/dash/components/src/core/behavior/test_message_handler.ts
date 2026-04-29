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

import { assert } from 'chai';
import { MessageHandlerBase, isMessageHandler } from '@kaolin/core/behavior/message_handler';

describe('visualize/dash/components/src/core/behavior/test_message_handler.ts', () => {

    describe('MessageHandlerBase', () => {
        it('reports its accepted tags, runs no-op lifecycle defaults, and passes the guard', () => {
            const handler = new MessageHandlerBase(['draw', 'clear']);
            assert.sameMembers(handler.acceptedMessageTags(), ['draw', 'clear'], 'exposes the constructor tags');
            assert.isUndefined(handler.onConnectionOpen(), 'onConnectionOpen is a no-op returning void');
            assert.isUndefined(handler.onMessage('draw', { x: 1 }), 'onMessage is a no-op returning void');
            assert.isTrue(isMessageHandler(handler), 'a MessageHandlerBase satisfies the MessageHandler guard');
            assert.deepEqual(new MessageHandlerBase().acceptedMessageTags(), [], 'defaults to no accepted tags');
        });
    });

    describe('isMessageHandler', () => {
        it('rejects values missing the required members', () => {
            assert.isFalse(isMessageHandler({}), 'a bare object is not a MessageHandler');
            assert.isFalse(isMessageHandler({ acceptedMessageTags: () => [] }), 'a partial implementation is rejected');
        });
    });

});
