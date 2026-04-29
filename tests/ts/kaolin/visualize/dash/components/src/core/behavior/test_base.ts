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

// DOM setup: CanvasBehavior.setActiveElement calls getContext('2d'), so the
// canvas test installs a node-canvas-backed happy-dom via registerDom.
// NOTE: setOption's schema warning routes through the module logger, so the
// soft-validation test forces LogLevel.DEBUG before capturing console output.

import { assert } from 'chai';
import { z } from 'zod';
import {
    Behavior,
    InteractiveBehavior,
    CanvasBehavior,
    isElementBoundBehavior,
    isClickEventHandler,
    isPointerEventHandler,
} from '@kaolin/core/behavior/base';
import { setLogLevel, LogLevel } from '@kaolin/util/logging';
import { registerDom, unregisterDom, makeCanvasElement } from '@test/helpers/dom';
import { captureConsole } from '@test/helpers/console';

describe('visualize/dash/components/src/core/behavior/test_base.ts', () => {

    describe('Behavior', () => {
        it('writes options through and notifies the subclass, is not element-bound, and is not a click handler', () => {
            class Recording extends Behavior {
                updates: Array<[string | undefined, any]> = [];
                override updateForOptions(name?: string, value?: any): void {
                    this.updates.push([name, value]);
                }
            }
            const behavior = new Recording();
            behavior.setOption('color', 'red');
            assert.equal(behavior.options.color, 'red', 'setOption stores the value');
            assert.deepEqual(behavior.updates.at(-1), ['color', 'red'], 'updateForOptions notified with name/value');
            behavior.setOption('size', 3);
            assert.equal(behavior.options.color, 'red', 'earlier option survives a later write');
            assert.equal(behavior.options.size, 3, 'second option is stored');
            behavior.reset();
            assert.equal(behavior.updates.length, 2, 'reset does not trigger updateForOptions');
            assert.isFalse(isElementBoundBehavior(behavior), 'Behavior is not an element-bound behavior');
            assert.isFalse(isClickEventHandler(behavior), 'Behavior is not a click handler');
            assert.isFalse(isPointerEventHandler(behavior), 'Behavior is not a pointer handler');
        });

        it('soft-validates against a Zod schema: warns on a bad value', () => {
            class Schemed extends Behavior {
                static override schema = z.object({ size: z.number() });
            }
            setLogLevel(LogLevel.DEBUG);
            const behavior = new Schemed();

            let cap = captureConsole();
            try { behavior.setOption('size', 5); } finally { cap.restore(); }
            assert.equal(behavior.options.size, 5, 'valid value is stored');
            assert.equal(cap.calls.length, 0, 'valid value produces no warning');

            cap = captureConsole();
            try { behavior.setOption('size', 'big'); } finally { cap.restore(); }
            assert.equal(cap.calls[0].method, 'warn', 'invalid value is routed to console.warn');
        });
    });

    describe('InteractiveBehavior', () => {
        it('binds to a div, tracks the active element, and exposes no-op handlers', () => {
            const behavior = new InteractiveBehavior();
            assert.equal(behavior.elementType(), 'div', 'binds to a div by default');
            assert.isNull(behavior.element, 'no active element before binding');
            const element = { tagName: 'DIV' } as unknown as HTMLElement;
            behavior.setActiveElement(element);
            assert.equal(behavior.element, element, 'setActiveElement records the element');
            assert.isTrue(isPointerEventHandler(behavior), 'InteractiveBehavior is a pointer handler');
            assert.isTrue(isClickEventHandler(behavior), 'InteractiveBehavior is not a click handler');
            assert.isTrue(isElementBoundBehavior(behavior), 'InteractiveBehavior is not an element-bound behavior');   
        });
    });

    describe('CanvasBehavior', () => {
        before(registerDom);
        after(unregisterDom);

        it('binds to a canvas and resolves its 2D context on setActiveElement', () => {
            const behavior = new CanvasBehavior();
            assert.equal(behavior.elementType(), 'canvas', 'binds to a canvas');
            assert.isNull(behavior.ctx, 'no context before binding');
            behavior.setActiveElement(makeCanvasElement(4, 4));
            assert.isNotNull(behavior.ctx, 'setActiveElement resolves the 2D context');
            assert.isTrue(isPointerEventHandler(behavior), 'CanvasBehavior is a pointer handler');
            assert.isTrue(isClickEventHandler(behavior), 'CanvasBehavior is not a click handler');
            assert.isTrue(isElementBoundBehavior(behavior), 'CanvasBehavior is an element-bound behavior');
        });
    });

    describe('type guards', () => {
        it('detect element-bound, click, and pointer capabilities', () => {
            const interactive = new InteractiveBehavior();
            assert.isTrue(isElementBoundBehavior(interactive), 'InteractiveBehavior is element-bound');
            assert.isTrue(isClickEventHandler(interactive), 'InteractiveBehavior handles click events');
            assert.isTrue(isPointerEventHandler(interactive), 'InteractiveBehavior handles pointer events');
            assert.isFalse(isElementBoundBehavior(new Behavior()), 'plain Behavior is not element-bound');
            assert.isFalse(isClickEventHandler({}), 'a bare object is not a click handler');
            assert.isFalse(isPointerEventHandler({ onClick: () => {} }), 'a click-only object is not a pointer handler');
        });
    });

});
