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
 * Helpers for building and stamping configurable SVG primitives onto an SVG
 * layer: schemas describing reusable assets, the {@link SvgDisplay} that
 * instantiates and positions them, and {@link svgToBlob} for exporting an SVG
 * as a standalone file.
 *
 * @module
 */

import { z } from 'zod';

/**
 * Schema for a single SVG primitive (e.g. a circle) within an asset.
 */
export const SvgElementSchema = z.object({
    name: z.enum(['circle', 'rect', 'ellipse', 'line', 'polygon', 'polyline', 'path', 'text'])
        .describe('SVG element name'),
    options: z.record(z.string(), z.string()).optional().describe('Options to set on the element')
});

/**
 * A single SVG primitive spec, parsed from {@link SvgElementSchema}.
 */
export type SvgElementSpec = z.infer<typeof SvgElementSchema>;

/**
 * Schema for an asset: one or more SVG primitives grouped together with an
 * optional uniform scale.
 */
export const SvgAssetSchema = z.object({
    elements: z.array(SvgElementSchema).nonempty().describe('A set of elements forming an asset (one is common)'),
    scale: z.number().default(1.0).optional().describe('Scale factor for the asset')
});

/**
 * An SVG asset spec (a group of primitives), parsed from {@link SvgAssetSchema}.
 */
export type SvgAssetSpec = z.infer<typeof SvgAssetSchema>;

/**
 * Manages an SVG display by creating primitives/assets from {@link SvgElementSpec}
 * / {@link SvgAssetSpec} specs and stamping them at fractional positions over
 * the SVG's coordinate space.
 */
export class SvgDisplay {
    svg: SVGElement;
    width: number;
    height: number;
    /**
     * Per-display counter backing the ids handed to stamped markers. Marker ids
     * are namespaced by the wrapped SVG's id (see {@link addElement}), so this
     * resets per display while ids stay globally unique across displays.
     */
    private markerCount: number = 0;

    /**
     * Wrap an existing SVG element, reading its drawing size from the `viewBox`
     * (falling back to the client width/height when no `viewBox` is set).
     *
     * @param svgElement - The SVG element to manage.
     */
    constructor(svgElement: SVGElement) {
        this.svg = svgElement;

        var viewBox = this.svg.getAttribute("viewBox");
        if (!viewBox) {
            this.width = this.svg.clientWidth;
            this.height = this.svg.clientHeight;
        } else {
            var viewBoxParts = viewBox.split(" ");
            this.width = Number(viewBoxParts[2]);
            this.height = Number(viewBoxParts[3]);
        }
    }

    /**
     * Create a single SVG primitive element from a spec, applying sensible
     * size/stroke defaults for `circle` and `rect` before the spec's own options.
     *
     * @param spec - The primitive to create.
     * @returns The newly created (unattached) SVG element.
     */
    createSvgElement(spec: SvgElementSpec) {
        const elem = document.createElementNS("http://www.w3.org/2000/svg", spec.name);

        const defaults: Record<string, string> = {};
        if (spec.name === 'circle') {
            const radius = this.width * 0.03;
            defaults['cx'] = '0';
            defaults['cy'] = '0';
            defaults['r'] = radius.toString();
            defaults['stroke'] = '#dfdfdf';
            defaults['stroke-width'] = (radius / 4).toString();
            defaults['fill'] = 'none';
        } else if (spec.name === 'rect') {
            const size = this.width * 0.06;
            defaults['x'] = (-size / 2).toString();
            defaults['y'] = (-size / 2).toString();
            defaults['width'] = size.toString();
            defaults['height'] = size.toString();
            defaults['stroke'] = '#dfdfdf';
            defaults['stroke-width'] = (size / 8).toString();
            defaults['fill'] = 'none';
        }

        for (const [k, v] of Object.entries(defaults)) {
            elem.setAttribute(k, v);
        }
        if (spec.options) {
            for (const [k, v] of Object.entries(spec.options)) {
                elem.setAttribute(k, v);
            }
        }

        return elem;
    }

    /**
     * Create a `<g>` group containing every primitive in an asset spec, applying
     * the asset's optional uniform scale.
     *
     * @param assetsSpec - The asset (group of primitives) to create.
     * @returns The newly created (unattached) SVG group element.
     */
    createSvgGroup(assetsSpec: SvgAssetSpec) {
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        for (const elemSpec of assetsSpec.elements) {
            group.appendChild(this.createSvgElement(elemSpec));
        }
        const scale = assetsSpec.scale ?? 1.0;
        if (scale !== 1.0) {
            group.setAttribute('transform', `scale(${scale})`);
        }
        return group;
    }

    /**
     * Clone an element and append it to the SVG at a fractional position.
     *
     * @param relX - Horizontal position as a fraction (0..1) of the SVG width.
     * @param relY - Vertical position as a fraction (0..1) of the SVG height.
     * @param elem - The element (typically an asset group) to stamp.
     * @returns The appended clone.
     */
    addElement(relX: number, relY: number, elem: SVGElement): SVGElement {
        const copy = elem.cloneNode(true) as SVGElement;
        // Namespace marker ids by the SVG's id so they stay globally unique across
        // displays, while the per-display counter keeps them reproducible. Falls
        // back to a bare svg_group<n> when the SVG has no id.
        const prefix = this.svg.id ? `${this.svg.id}_` : '';
        copy.setAttribute('id', `${prefix}svg_group${this.markerCount++}`);

        // Translation is applied in the outer (parent) coord space; any existing
        // transform on the element (e.g. asset scale) is preserved and applied
        // first, in the asset's local coord space.
        const translate = this.makeSvgTranslationTransform(relX, relY);
        const existing = copy.getAttribute('transform');
        copy.setAttribute('transform', existing ? `${translate} ${existing}` : translate);
        if (this.svg) {
            this.svg.appendChild(copy);
        }
        return copy;
    }

    /**
     * Build an SVG `translate(...)` transform from fractional coordinates.
     *
     * @param xperc - Horizontal position as a fraction (0..1) of the SVG width.
     * @param yperc - Vertical position as a fraction (0..1) of the SVG height.
     * @returns The SVG transform string.
     */
    makeSvgTranslationTransform(xperc: number, yperc: number) {
        return "translate(" + this.width * xperc + ", " + this.height * yperc + ")";
    }
}

/**
 * Serialize an SVG element into a standalone `image/svg+xml` `Blob`.
 *
 * Unlike reading `svg.innerHTML` (which yields only the inner markup and so is
 * not a valid document), `XMLSerializer` emits the full `<svg>` root and injects
 * the SVG namespace declaration when absent, producing a valid standalone `.svg`
 * file. Pair with {@link util.file.downloadBlob | downloadBlob} to save it.
 *
 * @param svg - The SVG element to serialize.
 * @returns A Blob of type `image/svg+xml` holding the serialized document.
 */
export function svgToBlob(svg: SVGElement): Blob {
    const serialized = new XMLSerializer().serializeToString(svg);
    return new Blob([serialized], { type: 'image/svg+xml' });
}
