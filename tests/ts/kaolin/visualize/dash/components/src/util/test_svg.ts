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

// SvgDisplay builds and stamps real SVG nodes via document.createElementNS, so a
// DOM is required; registerDom installs happy-dom for the whole file.

import { assert } from 'chai';
import { SvgDisplay, svgToBlob } from '@kaolin/util/svg';
import { registerDom, unregisterDom } from '@test/helpers/dom';

const SVG_NS = 'http://www.w3.org/2000/svg';

// Build an SvgDisplay backed by an SVG whose drawing size comes from the viewBox.
function makeDisplay(viewBox = '0 0 100 200'): SvgDisplay {
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', viewBox);
    return new SvgDisplay(svg);
}

// Assert each expected attribute (with a per-attribute message); `tag` is the
// expected (lower-cased) element name.
function checkElement(elem: Element, tag: string, expected: Record<string, string>, label: string) {
    assert.equal(elem.tagName.toLowerCase(), tag, `${label}: element name`);
    for (const [k, v] of Object.entries(expected)) {
        assert.equal(elem.getAttribute(k), v, `${label}: ${k}`);
    }
}

describe('visualize/dash/components/src/util/test_svg.ts', () => {

    before(registerDom);
    after(unregisterDom);
    afterEach(() => { document.body.innerHTML = ''; });

    describe('SvgDisplay (construction)', () => {
        it('reads width/height from the viewBox, else falls back to clientWidth/clientHeight', () => {
            const fromViewBox = makeDisplay('0 0 120 80');
            assert.equal(fromViewBox.width, 120, 'width from viewBox third token');
            assert.equal(fromViewBox.height, 80, 'height from viewBox fourth token');

            const noViewBox = new SvgDisplay(document.createElementNS(SVG_NS, 'svg'));
            assert.equal(noViewBox.width, noViewBox.svg.clientWidth, 'width falls back to clientWidth');
            assert.equal(noViewBox.height, noViewBox.svg.clientHeight, 'height falls back to clientHeight');
        });
    });

    describe('SvgDisplay.createSvgElement', () => {
        it('circle: applies size/stroke defaults, then overrides with spec options', () => {
            const display = makeDisplay();              // 100 wide -> radius = 100 * 0.03
            const radius = display.width * 0.03;
            const circle = display.createSvgElement({ name: 'circle', options: { fill: '#22aa22' } });
            checkElement(circle, 'circle', {
                cx: '0', cy: '0',
                r: radius.toString(),                   // derived default radius
                stroke: '#dfdfdf',                      // default kept
                'stroke-width': (radius / 4).toString(),
                fill: '#22aa22',                        // option overrides the 'none' default
            }, 'circle');
        });

        it('rect: applies size/stroke defaults, then overrides with spec options', () => {
            const display = makeDisplay();              // 100 wide -> size = 100 * 0.06
            const size = display.width * 0.06;
            const rect = display.createSvgElement({ name: 'rect', options: { fill: 'red' } });
            checkElement(rect, 'rect', {
                x: (-size / 2).toString(), y: (-size / 2).toString(),
                width: size.toString(), height: size.toString(),
                stroke: '#dfdfdf', 'stroke-width': (size / 8).toString(),
                fill: 'red',                            // option overrides the 'none' default
            }, 'rect');
        });

        it('other element types get no defaults, only the spec options', () => {
            const display = makeDisplay();
            const line = display.createSvgElement(
                { name: 'line', options: { x1: '0', y1: '0', x2: '10', y2: '10', stroke: 'white' } });
            checkElement(line, 'line', { x1: '0', y1: '0', x2: '10', y2: '10', stroke: 'white' }, 'line');
            assert.isNull(line.getAttribute('fill'), 'no default fill on a non-circle/rect element');
        });
    });

    describe('SvgDisplay.createSvgGroup', () => {
        it('wraps every element in a <g> and applies a non-default scale transform', () => {
            const display = makeDisplay();
            const group = display.createSvgGroup({ elements: [{ name: 'circle' }, { name: 'rect' }], scale: 0.5 });
            assert.equal(group.tagName.toLowerCase(), 'g', 'creates a <g> group');
            assert.equal(group.childNodes.length, 2, 'one child per element spec');
            assert.equal((group.childNodes[0] as Element).tagName.toLowerCase(), 'circle', 'first child is the circle');
            assert.equal((group.childNodes[1] as Element).tagName.toLowerCase(), 'rect', 'second child is the rect');
            assert.equal(group.getAttribute('transform'), 'scale(0.5)', 'scale != 1 sets a scale transform');
        });

        it('omits the transform at the default scale and supports an empty group', () => {
            const group = makeDisplay().createSvgGroup({ elements: [] } as any);
            assert.equal(group.childNodes.length, 0, 'empty elements yields an empty group');
            assert.isNull(group.getAttribute('transform'), 'default scale (1.0) leaves no transform');
        });
    });

    describe('SvgDisplay.addElement', () => {
        it('clones the element, positions it via a fractional translate, and appends it', () => {
            const display = makeDisplay();              // 100 x 200
            const asset = display.createSvgElement({ name: 'circle' });
            const added = display.addElement(0.5, 0.25, asset);

            assert.notStrictEqual(added, asset, 'returns a clone, not the original element');
            assert.isFalse(display.svg.contains(asset), 'the original element is not appended');
            assert.isTrue(display.svg.contains(added), 'the clone is appended to the SVG');
            // Ids are a process-wide counter (globally unique), so assert the
            // shape and a +1 increment rather than absolute values.
            const id0 = added.getAttribute('id') ?? '';
            assert.match(id0, /^svg_group\d+$/, 'marker gets an svg_group<n> id');
            // translate = (width * relX, height * relY) = (100 * 0.5, 200 * 0.25)
            assert.equal(added.getAttribute('transform'), 'translate(50, 50)',
                'fractional position maps to an absolute translate');

            const second = display.addElement(0, 0, asset);
            const id1 = second.getAttribute('id') ?? '';
            assert.equal(Number(id1.slice('svg_group'.length)), Number(id0.slice('svg_group'.length)) + 1,
                'consecutive markers get consecutive ids');
        });

        it('prepends the translate while preserving an existing transform (e.g. asset scale)', () => {
            const display = makeDisplay();
            const asset = display.createSvgGroup({ elements: [{ name: 'circle' }], scale: 0.5 }); // transform 'scale(0.5)'
            const added = display.addElement(0, 0, asset);
            assert.equal(added.getAttribute('transform'), 'translate(0, 0) scale(0.5)',
                'translate is applied first (outer space), existing transform second');
        });
    });

    describe('svgToBlob', () => {
        it('serializes the whole <svg> (root + namespace + children) into an image/svg+xml blob', async () => {
            const svg = document.createElementNS(SVG_NS, 'svg');
            svg.setAttribute('viewBox', '0 0 100 100');
            svg.appendChild(document.createElementNS(SVG_NS, 'circle'));

            const blob = svgToBlob(svg);
            assert.equal(blob.type, 'image/svg+xml', 'blob carries the svg+xml mime type');

            const text = await blob.text();
            assert.match(text, /^<svg[\s>]/, 'includes the root <svg> tag (a standalone document, not just inner markup)');
            assert.include(text, SVG_NS, 'root carries the SVG xmlns so the file renders standalone');
            assert.include(text, '<circle', 'child markers are serialized inside the svg');
        });
    });

});
