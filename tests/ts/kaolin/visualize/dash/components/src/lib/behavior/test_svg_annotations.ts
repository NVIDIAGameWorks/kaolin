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

// NOTE: the missing-asset test asserts on logged output, so it forces
// LogLevel.DEBUG first so the error is not suppressed. SvgAnnotationBehavior
// stamps real SVG nodes (document.createElementNS), so a DOM is required;
// registerDom installs happy-dom for the whole file.

import * as fs from 'fs';

import { assert } from 'chai';
import { z } from 'zod';

import { isElementBoundBehavior } from '@kaolin/core/behavior';
import { SvgAnnotationOptionsSchema, SvgAnnotationBehavior } from '@kaolin/lib/behavior/svg_annotations';
import { RecordedInteraction, replayInteractions } from '@kaolin/lib/behavior/record_interactions';
import { setLogLevel, LogLevel } from '@kaolin/util/logging';
import { registerDom, unregisterDom } from '@test/helpers/dom';
import { captureConsole } from '@test/helpers/console';
import { dashTypescriptTestData } from '@test/helpers/paths';

const SVG_NS = 'http://www.w3.org/2000/svg';

// An <svg> the behavior can bind to: viewBox fixes SvgDisplay's coordinate space
// (the viewer's svg layer is always "0 0 100 100"), and a stubbed bounding rect
// gives happy-dom (which has no layout) a non-zero size so click fractions
// resolve. The displayed size cancels out, so any non-zero rect reproduces the
// recorded fractional positions.
function makeBoundSvg(id: string): SVGElement {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', '0 0 100 100');
    svg.id = id;
    const rect = {
        left: 0, top: 0, right: 100, bottom: 100, width: 100, height: 100, x: 0, y: 0, toJSON: () => ({}),
    } as DOMRect;
    svg.getBoundingClientRect = () => rect;
    return svg;
}

// Give a stamped marker a real on-screen box so the geometric hit-test in
// markerFromEvent (which reads getBoundingClientRect) works under happy-dom.
function stubMarkerRect(marker: SVGElement, box: { left: number; top: number; width: number; height: number }) {
    const rect = {
        ...box, right: box.left + box.width, bottom: box.top + box.height, x: box.left, y: box.top, toJSON: () => ({}),
    } as DOMRect;
    marker.getBoundingClientRect = () => rect;
}

// Canonical form for comparing two full <svg> documents produced by different
// serializers (Chrome golden via svgToBlob vs happy-dom replay): re-parse and
// re-serialize both through happy-dom so self-closing / quoting / attribute style
// agree, then compare only the stamped marker children (root attributes like
// id/class/width are environment-specific) with the volatile, svg-id-namespaced
// marker ids dropped, decimals rounded (absorbs sub-ULP float differences from
// the frac->pixel->frac replay round-trip and differing record/replay display
// sizes), and inter-tag whitespace collapsed.
function canonicalizeSvg(svgDocument: string): string {
    const root = new DOMParser().parseFromString(svgDocument, 'image/svg+xml').documentElement;
    root.querySelectorAll('[id]').forEach((el) => el.removeAttribute('id'));
    const serialized = Array.from(root.childNodes)
        .map((node) => new XMLSerializer().serializeToString(node)).join('');
    return serialized
        .replace(/-?\d+\.\d+/g, (n) => String(Number(parseFloat(n).toFixed(6))))
        .replace(/>\s+</g, '><')
        .trim();
}

describe('visualize/dash/components/src/lib/behavior/test_svg_annotations.ts', () => {

    before(registerDom);
    after(unregisterDom);
    afterEach(() => { document.body.innerHTML = ''; });

    describe('SvgAnnotationOptionsSchema', () => {
        it('applies defaults, marks every field uiBound:false, and rejects an unknown onClickAction', () => {
            const parsed = SvgAnnotationOptionsSchema.parse({});
            assert.sameMembers(Object.keys(parsed.assets), ['circle', 'square'], 'default assets are circle + square');
            assert.equal(parsed.onClickAction, 'add', 'default onClickAction is "add"');
            assert.equal(parsed.activeAsset, 'circle', 'default activeAsset is "circle"');

            // Read meta the way production does (z.toJSONSchema surfaces .meta()).
            const props = (z.toJSONSchema(SvgAnnotationOptionsSchema) as any).properties;
            for (const key of ['assets', 'onClickAction', 'activeAsset']) {
                assert.isFalse(props[key].uiBound, `${key} carries uiBound:false`);
            }

            assert.isFalse(SvgAnnotationOptionsSchema.safeParse({ onClickAction: 'nope' }).success,
                'rejects onClickAction outside {add, add_or_remove}');
        });
    });

    describe('SvgAnnotationBehavior', () => {
        it('defaults from schema, binds to svg, stamps the active asset on click, and reset clears markers', () => {
            const behavior = new SvgAnnotationBehavior();
            assert.deepEqual(behavior.options, SvgAnnotationOptionsSchema.parse({}),
                'no-arg construction uses schema defaults');
            assert.isTrue(isElementBoundBehavior(behavior), 'is an element-bound behavior');
            assert.equal(behavior.elementType(), 'svg', 'binds to an svg element');

            const svg = makeBoundSvg('svg-usage');
            behavior.setActiveElement(svg);
            assert.isNotNull(behavior.svgDisplay, 'setActiveElement builds an SvgDisplay');
            assert.sameMembers(Object.keys(behavior.assets), ['circle', 'square'], 'asset groups built from options');

            const addedIndices: number[] = [];
            behavior.onMarkerAdded = (_marker, index) => addedIndices.push(index);

            behavior.onClick({ clientX: 50, clientY: 50 } as any);
            assert.equal(behavior.svgMarkers.length, 1, 'click adds one tracked marker');
            assert.equal(svg.childNodes.length, 1, 'marker is appended to the svg');
            assert.deepEqual(addedIndices, [0], 'onMarkerAdded fired with the new marker index');
            // translate = viewBox width (100) * fracX (50/100) = 50, likewise y.
            assert.equal(behavior.svgMarkers[0].getAttribute('transform'), 'translate(50, 50)',
                'marker is placed at the click position in the svg coordinate space');

            behavior.reset();
            assert.equal(behavior.svgMarkers.length, 0, 'reset clears the tracked markers');
            assert.equal(svg.childNodes.length, 0, 'reset detaches the markers from the svg');
        });

        it('logs an error and adds nothing when the active asset is missing', () => {
            setLogLevel(LogLevel.DEBUG);  // so logger.error is not suppressed
            const behavior = new SvgAnnotationBehavior({ activeAsset: 'nonexistent' });
            behavior.setActiveElement(makeBoundSvg('svg-missing'));

            const { calls, restore } = captureConsole();
            try { behavior.onClick({ clientX: 10, clientY: 10 } as any); } finally { restore(); }

            assert.equal(behavior.svgMarkers.length, 0, 'no marker added for an unknown active asset');
            assert.isTrue(calls.some((c) => c.method === 'error'), 'an error is logged');
        });

        it('add_or_remove: a click adds a marker; clicking the same marker removes it', () => {
            const behavior = new SvgAnnotationBehavior({ onClickAction: 'add_or_remove' });
            behavior.setActiveElement(makeBoundSvg('svg-toggle'));

            const removedIndices: number[] = [];
            behavior.onMarkerRemoved = (_marker, index) => removedIndices.push(index);

            behavior.onClick({ clientX: 50, clientY: 50 } as any);
            assert.equal(behavior.svgMarkers.length, 1, 'first click adds a marker');

            // Hit-testing reads each marker's bounding box; happy-dom has none, so stub one
            // that covers the click point.
            stubMarkerRect(behavior.svgMarkers[0], { left: 40, top: 40, width: 20, height: 20 });

            behavior.onClick({ clientX: 50, clientY: 50 } as any);
            assert.equal(behavior.svgMarkers.length, 0, 'second click on the marker removes it');
            assert.equal(behavior.element!.childNodes.length, 0, 'removed marker is detached from the svg');
            assert.deepEqual(removedIndices, [0], 'onMarkerRemoved fired with the removed marker index');
        });
    });

    describe('end-to-end interaction replay against SVG golden', () => {
        // Fixtures recorded from examples/tutorial/app/interact2d_main.py (svg_annotate
        // mode): the click stream (Save Interactions As) and the serialized SVG markup
        // (File -> Download SVG). The golden is vector markup, so the comparison is exact
        // (no rasterizer tolerances) once ids/whitespace are canonicalized.
        const sample = (name: string) => dashTypescriptTestData('lib', 'behavior', name);
        const interactionsPath = sample('svg_annotation_interactions.json');
        const goldenPath = sample('svg_annotation_golden.svg');
        const haveFixtures = fs.existsSync(interactionsPath) && fs.existsSync(goldenPath);

        // Skips if the fixtures are ever removed (keeps the suite green).
        (haveFixtures ? it : it.skip)('reproduces the golden SVG markup from the recorded interactions', () => {
            const interactions = JSON.parse(fs.readFileSync(interactionsPath, 'utf-8')) as RecordedInteraction[];

            // The behavior must be configured exactly as interact2d_main.py was when
            // the golden was recorded: the two filled assets (the recorded stream's
            // setOption activeAsset switches drive which one each click stamps).
            const svg = makeBoundSvg('svg-golden');
            const behavior = new SvgAnnotationBehavior({
                assets: {
                    red_circle: { elements: [{ name: 'circle', options: { fill: '#e23b3b', stroke: '#7a1010' } }] },
                    green_square: { elements: [{ name: 'rect', options: { fill: '#3bbf57', stroke: '#176b2a' } }] },
                },
                activeAsset: 'red_circle',
            });
            behavior.setActiveElement(svg);

            replayInteractions(behavior, interactions);

            const actual = canonicalizeSvg(new XMLSerializer().serializeToString(svg));
            const expected = canonicalizeSvg(fs.readFileSync(goldenPath, 'utf-8'));
            assert.equal(actual, expected, 'replayed SVG markup should match the golden');
        });
    });

});
