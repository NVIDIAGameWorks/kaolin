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
import { Registry, createGlobalRegistry } from '@kaolin/util/registry';
import { setLogLevel, LogLevel } from '@kaolin/util/logging';
import { captureConsole } from '@test/helpers/console';

describe('visualize/dash/components/src/util/test_registry.ts', () => {

    describe('Registry (instance)', () => {
        describe('basic behvior works as expected', () => {
            it('interleaves mutators with gets as expected', () => {
                const reg = new Registry<number>();
                assert.isUndefined(reg.get('a'), 'unregistered name returns undefined');

                reg.register('a', 1);
                assert.equal(reg.get('a'), 1, 'returns the just-registered value');

                reg.register('b', 2);
                assert.equal(reg.get('b'), 2, 'second name retrievable');
                assert.equal(reg.get('a'), 1, 'first name unaffected by second registration');

                reg.deregister('a');
                assert.isUndefined(reg.get('a'), 'deregistered name returns undefined');
                assert.equal(reg.get('b'), 2, 'unrelated name survives deregistration');
            });

            it('duplicate register, default behavior', () => {
                setLogLevel(LogLevel.DEBUG);
                const reg = new Registry<number>();
                reg.register('x', 1);
                const { calls, restore } = captureConsole();
                try {
                    reg.register('x', 2);
                } finally {
                    restore();
                }
                assert.equal(reg.get('x'), 1, 'original value is kept');
                assert.equal(calls.length, 1, 'exactly one console call');
                assert.equal(calls[0].method, 'warn', 'routed to console.warn');
                assert.include(String(calls[0].args[0]), 'skipping', 'warning indicates it skipped');
            });

            it('duplicate register with overwrite=true warns and replaces original', () => {
                setLogLevel(LogLevel.DEBUG);
                const reg = new Registry<number>();
                reg.register('x', 1);
                const { calls, restore } = captureConsole();
                try {
                    reg.register('x', 2, true);
                } finally {
                    restore();
                }
                assert.equal(reg.get('x'), 2, 'value is replaced');
                assert.equal(calls.length, 1, 'exactly one console call');
                assert.equal(calls[0].method, 'warn', 'routed to console.warn');
                assert.include(String(calls[0].args[0]), 'overwriting', 'warning indicates overwrite');
            });
        });
    });

    describe('createGlobalRegistry', () => {
        describe('static API mirrors Registry instance API', () => {
            it('register() stores an item retrievable via get()', () => {
                const G = createGlobalRegistry<number>();
                assert.isFalse(G.has('a'), 'false before registration');
                G.register('a', 1);
                assert.isTrue(G.has('a'), 'true after registration');
                assert.equal(G.get('a'), 1);
                G.register('b', 2);
                assert.sameMembers(G.getRegistered(), ['a', 'b']);
            });

            it('registering the same name twice keeps the first value, but logs warning', () => {
                setLogLevel(LogLevel.DEBUG);
                const G = createGlobalRegistry<number>();
                G.register('a', 1);
                const { calls, restore } = captureConsole();
                try {
                    G.register('a', 2);
                } finally {
                    restore();
                }
                assert.equal(G.get('a'), 1, 'first value is kept');
                assert.equal(calls.length, 1, 'exactly one console call');
                assert.equal(calls[0].method, 'warn', 'routed to console.warn');
                assert.include(String(calls[0].args[0]), 'skipping', 'warning indicates it skipped');
            });
        });

        describe('each call produces an independent registry', () => {
            it('items registered in one GlobalRegistry are not visible in another', () => {
                const A = createGlobalRegistry<number>();
                const B = createGlobalRegistry<number>();
                A.register('a', 1);
                A.register('b', 2);
                assert.isFalse(B.has('a'));
                assert.isFalse(B.has('b'));
                assert.isUndefined(B.get('a'));
                assert.deepEqual(B.getRegistered(), [], 'second registry stays empty');
            });
        });
    });

});
