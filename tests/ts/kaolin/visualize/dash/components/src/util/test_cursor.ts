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
import { cursorFromEvent, CursorValue, getRelativeCoordinates, RelativeCoordinates } from '@kaolin/util/cursor';

// Stub element: both helpers only read getBoundingClientRect().
function stubElement(rect: { left: number; top: number; width: number; height: number }): HTMLElement {
    return { getBoundingClientRect: () => rect } as unknown as HTMLElement;
}

// Assert every field of a RelativeCoordinates against the expected ground truth.
function checkRelativeCoordinates(actual: RelativeCoordinates, expected: RelativeCoordinates, label: string) {
    assert.equal(actual.x, expected.x, `${label}: x`);
    assert.equal(actual.y, expected.y, `${label}: y`);
    assert.closeTo(actual.fracX, expected.fracX, 1e-9, `${label}: fracX`);
    assert.closeTo(actual.fracY, expected.fracY, 1e-9, `${label}: fracY`);
}

// Assert every field of a CursorValue against the expected ground truth.
function checkCursorValue(actual: CursorValue, expected: CursorValue, label: string) {
    assert.equal(actual.x, expected.x, `${label}: x`);
    assert.equal(actual.y, expected.y, `${label}: y`);
    assert.closeTo(actual.fracX, expected.fracX, 1e-9, `${label}: fracX`);
    assert.closeTo(actual.fracY, expected.fracY, 1e-9, `${label}: fracY`);
    assert.equal(actual.type, expected.type, `${label}: type`);
    assert.equal(actual.buttons, expected.buttons, `${label}: buttons`);
    assert.equal(actual.pointerType, expected.pointerType, `${label}: pointerType`);
    assert.equal(actual.pointerId, expected.pointerId, `${label}: pointerId`);
}

describe('visualize/dash/components/src/util/test_cursor.ts', () => {

    describe('getRelativeCoordinates', () => {
        it('maps an event + element to pixel offsets and fractions', () => {
            const element = stubElement({ left: 10, top: 20, width: 200, height: 100 });
            // clientX 110 → x 100 → fracX 0.5; clientY 70 → y 50 → fracY 0.5.
            const event = { clientX: 110, clientY: 70, currentTarget: element };
            const expected: RelativeCoordinates = { x: 100, y: 50, fracX: 0.5, fracY: 0.5 };

            // Two input forms that resolve to the same ground truth: an explicit
            // element, and the event.currentTarget fallback when it is omitted.
            checkRelativeCoordinates(getRelativeCoordinates(event as any, element), expected, 'explicit element');
            checkRelativeCoordinates(getRelativeCoordinates(event as any), expected, 'currentTarget fallback');
        });

        it('uses viewport (clientX/Y), not document (pageX/Y), so a scrolled page is handled', () => {
            // getBoundingClientRect is viewport-relative; pairing it with pageX/Y
            // (document-relative) would fold in the scroll offset and mislocate the
            // point. Simulate a 1000px-scrolled page: pageX/Y = clientX/Y + 1000.
            const element = stubElement({ left: 10, top: 20, width: 100, height: 100 });
            const result = getRelativeCoordinates(
                { clientX: 60, clientY: 70, pageX: 1060, pageY: 1070 } as any, element);
            assert.equal(result.x, 50, 'x uses clientX, ignoring the page scroll offset');
            assert.equal(result.y, 50, 'y uses clientY, ignoring the page scroll offset');
        });

        it('fracX/fracY are 0 when the element has zero displayed size', () => {
            const element = stubElement({ left: 0, top: 0, width: 0, height: 0 });
            const result = getRelativeCoordinates({ clientX: 5, clientY: 8 } as any, element);
            assert.equal(result.x, 5, 'x still computed from client coords');
            assert.equal(result.y, 8, 'y still computed from client coords');
            assert.equal(result.fracX, 0, 'fracX is 0 for zero-width element');
            assert.equal(result.fracY, 0, 'fracY is 0 for zero-height element');
        });
    });

    describe('cursorFromEvent', () => {
        it('composes a CursorValue from the relative coords plus DOM event fields', () => {
            const element = stubElement({ left: 10, top: 20, width: 200, height: 100 });
            // clientX 110 → x 100 → fracX 0.5; clientY 70 → y 50 → fracY 0.5.
            const event = {
                clientX: 110, clientY: 70, type: 'pointermove', buttons: 1,
                pointerType: 'mouse', pointerId: 7, currentTarget: element,
            };
            const expected: CursorValue = {
                x: 100, y: 50, fracX: 0.5, fracY: 0.5,
                type: 'pointermove', buttons: 1, pointerType: 'mouse', pointerId: 7,
            };

            // Two input forms that resolve to the same ground truth: an explicit
            // element, and the event.currentTarget fallback when it is omitted.
            checkCursorValue(cursorFromEvent(event as any, element), expected, 'explicit element');
            checkCursorValue(cursorFromEvent(event as any), expected, 'currentTarget fallback');
        });
    });

});
