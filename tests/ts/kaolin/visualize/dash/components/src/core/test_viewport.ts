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
import { registerDom, unregisterDom } from '@test/helpers/dom';
import {
    adjustByDevicePixelRatio,
    isFixedAspect,
    computeContainedFixedAspectSize,
    computeContainedDisplaySize,
    ViewportResizeMode,
} from '@kaolin/core/viewport';

type Size = { width: number, height: number };

function checkAspect(prev: Size, next: Size, expected: boolean, label: string, tolerance?: number) {
    const actual = tolerance === undefined
        ? isFixedAspect(prev, next)
        : isFixedAspect(prev, next, tolerance);
    assert.equal(actual, expected,
        `${label}: ${prev.width}x${prev.height} -> ${next.width}x${next.height}`);
}

describe('visualize/dash/components/src/core/test_viewport.ts', () => {

    describe('isFixedAspect', () => {
        it('accepts uniform scales and rejects aspect changes', () => {
            checkAspect({ width: 800, height: 600 }, { width: 800, height: 600 }, true, 'unchanged size');
            checkAspect({ width: 800, height: 600 }, { width: 1200, height: 900 }, true, 'uniform 1.5x scale');
            checkAspect({ width: 800, height: 600 }, { width: 400, height: 300 }, true, 'uniform shrink');
            checkAspect({ width: 1600, height: 900 }, { width: 1599, height: 900 }, true,
                'one pixel of layout rounding');
            // Aspect is unitless, which is what lets a device-pixel drawing
            // buffer be compared against a CSS-pixel viewport.
            checkAspect({ width: 1000, height: 500 }, { width: 500, height: 250 }, true,
                'device-pixel buffer vs CSS-pixel viewport');
            checkAspect({ width: 500, height: 500 }, { width: 1000, height: 500 }, false, 'widened only');
            checkAspect({ width: 1600, height: 900 }, { width: 900, height: 1600 }, false, 'transposed');
        });

        it('honors the tolerance, defaulting to one percent', () => {
            const square = { width: 100, height: 100 };
            checkAspect(square, { width: 100.5, height: 100 }, true, 'half a percent is within the default');
            checkAspect(square, { width: 102, height: 100 }, false, 'two percent exceeds the default');
            checkAspect(square, { width: 102, height: 100 }, true, 'a looser tolerance accepts it', 0.05);
            checkAspect(square, { width: 100.5, height: 100 }, false, 'a zero tolerance demands an exact match', 0);
        });

        it('rejects degenerate sizes, which have no meaningful aspect', () => {
            const valid = { width: 800, height: 600 };
            checkAspect(valid, { width: 0, height: 0 }, false, 'collapsed to zero (hidden element)');
            checkAspect({ width: 0, height: 0 }, valid, false, 'grown from zero');
            checkAspect(valid, { width: 800, height: 0 }, false, 'zero on one axis only');
            checkAspect(valid, { width: -800, height: -600 }, false, 'negative dimensions');
            checkAspect(valid, { width: NaN, height: NaN }, false, 'not a number');
            checkAspect(valid, { width: Infinity, height: Infinity }, false, 'non-finite');
        });
    });

    describe('computeContainedFixedAspectSize', () => {
        // The contract is exactness, so every expectation is also checked against
        // the source ratio rather than only against hand-computed numbers.
        function checkContained(
            initial: Size, container: Size, expected: Size, label: string) {
            const actual = computeContainedFixedAspectSize(initial, container);
            assert.deepEqual(actual, expected, `${label}: size`);
            assert.equal(actual.width * initial.height, actual.height * initial.width,
                `${label}: aspect matches the source exactly`);
            assert.isAtMost(actual.width, container.width, `${label}: fits the container width`);
            assert.isAtMost(actual.height, container.height, `${label}: fits the container height`);
            assert.isTrue(Number.isInteger(actual.width) && Number.isInteger(actual.height),
                `${label}: integer size`);
        }

        it('returns whole multiples of the reduced ratio', () => {
            // 1600x900 reduces to 16x9; 1000/16 = 62 units, 1000/9 = 111 units.
            checkContained({ width: 1600, height: 900 }, { width: 1000, height: 1000 },
                { width: 992, height: 558 }, 'width-limited 16:9');
            // A real-valued fit would give 1000x562, which is off the ratio.
            assert.notEqual(1000 * 900, 562 * 1600, 'the naive rounded fit is not exactly 16:9');

            checkContained({ width: 500, height: 500 }, { width: 333, height: 200 },
                { width: 200, height: 200 }, 'square is height-limited');
            checkContained({ width: 1920, height: 1080 }, { width: 1920, height: 1080 },
                { width: 1920, height: 1080 }, 'an exact fit is returned unchanged');
        });

        it('never returns an empty size, even below one whole unit', () => {
            // 16x9 does not fit in 10x10 at all; one unit is returned rather than zero.
            const actual = computeContainedFixedAspectSize({ width: 1600, height: 900 }, { width: 10, height: 10 });
            assert.deepEqual(actual, { width: 16, height: 9 }, 'falls back to a single unit');
        });

        it('falls back to the container when either size is degenerate', () => {
            const container = { width: 640, height: 480 };
            assert.deepEqual(computeContainedFixedAspectSize({ width: 0, height: 0 }, container), container,
                'a zero-sized camera has no ratio to hold');
            assert.deepEqual(computeContainedFixedAspectSize({ width: 16, height: 9 }, { width: 0, height: 0 }),
                { width: 0, height: 0 }, 'a collapsed container is passed straight back');
        });
    });

    describe('computeContainedDisplaySize', () => {
        it('applies the camera size, the container, or an exact fit per mode', () => {
            const camera = { width: 1600, height: 900 };
            const container = { width: 1000, height: 1000 };

            assert.deepEqual(computeContainedDisplaySize(camera, ViewportResizeMode.FIXED, container),
                camera, 'fixed ignores the container');
            assert.deepEqual(computeContainedDisplaySize(camera, ViewportResizeMode.ADAPTIVE, container),
                container, 'adaptive takes the container');
            assert.deepEqual(computeContainedDisplaySize(camera, ViewportResizeMode.FIXED_ASPECT, container),
                { width: 992, height: 558 }, 'fixed_aspect fits the camera ratio exactly');
        });

        it('floors a fractional container in adaptive mode', () => {
            assert.deepEqual(
                computeContainedDisplaySize({ width: 500, height: 500 }, ViewportResizeMode.ADAPTIVE,
                    { width: 640.7, height: 480.2 }),
                { width: 640, height: 480 }, 'sub-pixel layout sizes are floored to whole pixels');
        });
    });

});


describe('viewport device-pixel dimensions', () => {
    before(registerDom);
    after(unregisterDom);

    for (const [dpr, width, height] of [[1, 101, 51], [1.25, 126, 63], [1.5, 151, 76], [2, 202, 102]]) {
        it(`floors buffer dimensions at DPR ${dpr}`, () => {
            Object.defineProperty(window, 'devicePixelRatio', { value: dpr, configurable: true });
            assert.deepEqual(adjustByDevicePixelRatio({ width: 101, height: 51 }), { width, height });
        });
    }

    it('uses DPR 1 without a window', () => {
        const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'window')!;
        Reflect.deleteProperty(globalThis, 'window');
        try {
            assert.deepEqual(adjustByDevicePixelRatio({ width: 101.5, height: 51.5 }),
                { width: 101, height: 51 });
        } finally {
            Object.defineProperty(globalThis, 'window', descriptor);
        }
    });

    it('keeps one coprime ratio unit when it exceeds the container', () => {
        assert.deepEqual(computeContainedFixedAspectSize(
            { width: 501, height: 500 }, { width: 200, height: 200 }),
            { width: 501, height: 500 });
    });
});
