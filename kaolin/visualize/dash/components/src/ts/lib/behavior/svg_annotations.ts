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
 * Click-to-annotate behavior for an SVG layer, stamping configurable marker
 * assets at each click.
 *
 * @module
 */

import { z } from 'zod';

import { BehaviorRegister, InteractiveBehavior } from '../../core/behavior';
import { getRelativeCoordinates } from '../../util/cursor';
import { SvgAssetSchema, SvgDisplay } from '../../util/svg';
import { logger } from '../../util/logging';

/**
 * Configuration schema for {@link SvgAnnotationBehavior}. Defines the library
 * of stampable assets, which asset is active, and whether clicking toggles
 * markers or only adds them.
 *
 * ```typescript
 * z.object({
 *     assets: z.record(z.string(), SvgAssetSchema).default({
 *         circle: { elements: [{ name: 'circle' }] },
 *         square: { elements: [{ name: 'rect' }] },
 *     }).describe('Named library of stampable SVG assets, keyed by asset name.')
 *         .meta({ uiBound: false }),
 *     onClickAction: z.enum(['add', 'add_or_remove']).default('add')
 *         .describe('Click behavior: "add" always adds a marker; "add_or_remove" removes the ' +
 *             'clicked marker if one was hit, otherwise adds.').meta({ uiBound: false }),
 *     activeAsset: z.string().default('circle').describe('Currently active asset').meta({ uiBound: false }),
 * })
 * ```
 *
 * @group Behavior: SvgAnnotationBehavior
 */
export const SvgAnnotationOptionsSchema = z.object({
    assets: z.record(z.string(), SvgAssetSchema).default({
        circle: { elements: [{ name: 'circle' }] },
        square: { elements: [{ name: 'rect' }] },
    }).describe('Named library of stampable SVG assets, keyed by asset name.')
        .meta({ uiBound: false }),
    onClickAction: z.enum(['add', 'add_or_remove']).default('add')
        .describe('Click behavior: "add" always adds a marker; "add_or_remove" removes the ' +
            'clicked marker if one was hit, otherwise adds.').meta({ uiBound: false }),
    activeAsset: z.string().default('circle').describe('Currently active asset').meta({ uiBound: false })
});

/**
 * Options for {@link SvgAnnotationBehavior}, parsed from {@link SvgAnnotationOptionsSchema}.
 *
 * @group Behavior: SvgAnnotationBehavior
 */
export type SvgAnnotationOptions = z.infer<typeof SvgAnnotationOptionsSchema>;


/**
 * A **behavior** for the Kaolin viewer that stamps SVG marker assets onto an
 * SVG layer in response to clicks. The active asset (`options.activeAsset`) is
 * drawn at each click position; with `onClickAction === 'add_or_remove'`,
 * clicking an existing marker removes it instead. Subclasses can hook
 * {@link onMarkerAdded} / {@link onMarkerRemoved} to mirror marker state.
 *
 * This is an {@link core.behavior.InteractiveBehavior | InteractiveBehavior}
 * (an {@link core.behavior.ElementBoundBehavior | ElementBoundBehavior}) that
 * binds to an `svg` element.
 *
 * Registered name: `'svg_annotation'`.
 *
 * Configuration schema: {@link SvgAnnotationOptionsSchema}.
 *
 * @group Behavior: SvgAnnotationBehavior
 */
export class SvgAnnotationBehavior extends InteractiveBehavior {
    static override schema = SvgAnnotationOptionsSchema;

    svgDisplay: SvgDisplay | null;
    assets: Record<string, SVGElement>;
    /** Markers added to the SVG by this behavior, tracked for {@link reset}. */
    svgMarkers: SVGElement[];
    // Preserve original transforms so resize compensation never accumulates.
    private markerTransforms = new WeakMap<SVGElement, string>();
    private markerScaleY = 1;
    override options: SvgAnnotationOptions;

    constructor(options: Partial<SvgAnnotationOptions> = {}) {
        super();
        this.options = SvgAnnotationOptionsSchema.parse(options);
        this.svgDisplay = null;
        this.assets = {};
        this.svgMarkers = [];
    }

    override elementType(): string {
        return 'svg';
    }

    override setActiveElement(elem) {
        super.setActiveElement(elem);
        // Markers are stamped at `viewBox` coordinates derived from a fraction
        // of the *element* box, so the two only agree if the user space maps
        // onto the whole element. Under the default `preserveAspectRatio` a
        // square `viewBox` is letterboxed inside a non-square element and a
        // marker stops landing under the click; 'none' stretches it to fill
        // instead. setDimensions compensates glyph scaling independently of
        // those fractional positions, so circles remain circular.
        (elem as SVGElement).setAttribute('preserveAspectRatio', 'none');
        this.svgDisplay = new SvgDisplay(elem as SVGElement);

        this.assets = {};
        for (const [name, assetSpec] of Object.entries(this.options.assets)) {
            this.assets[name] = this.svgDisplay.createSvgGroup(assetSpec);
        }
        const rect = elem.getBoundingClientRect();
        this.setDimensions(rect.width, rect.height);
    }

    /** Keep glyphs uniformly scaled while their centers follow the viewport. */
    override setDimensions(width: number, height: number): void {
        if (!this.svgDisplay || !Number.isFinite(width) || !Number.isFinite(height)
            || width <= 0 || height <= 0) return;
        const scaleY = (width / this.svgDisplay.width) / (height / this.svgDisplay.height);
        if (scaleY === this.markerScaleY) return;
        this.markerScaleY = scaleY;
        this.svgMarkers.forEach(marker => this.updateMarkerScale(marker));
    }

    private updateMarkerScale(marker: SVGElement): void {
        // Append after the original translate/asset scale: only the glyph is
        // compensated, never its center. Always start from the original so
        // repeated resizes cannot accumulate scaling.
        const transform = this.markerTransforms.get(marker) ?? '';
        marker.setAttribute('transform', transform +
            (this.markerScaleY === 1 ? '' : ` scale(1, ${this.markerScaleY})`));
    }

    onClick(event: React.MouseEvent) {
        if (!this.element || !this.svgDisplay) {
            return;
        }
        if (this.options.onClickAction === 'add_or_remove') {
            const hit = this.markerFromEvent(event);
            if (hit) {
                const index = this.svgMarkers.indexOf(hit);
                this.svgMarkers.splice(index, 1);
                hit.parentNode?.removeChild(hit);
                this.onMarkerRemoved(hit, index, event);
                return;
            }
        }
        const asset = this.assets[this.options.activeAsset];
        if (!asset) {
            logger.error(
                `SvgAnnotationBehavior: activeAsset "${this.options.activeAsset}" not found in assets ` +
                `(known: ${Object.keys(this.assets).join(', ') || '<none>'}).`);
            return;
        }
        const { fracX, fracY } = getRelativeCoordinates(event, this.element);
        const marker = this.svgDisplay.addElement(fracX, fracY, asset);
        this.markerTransforms.set(marker, marker.getAttribute('transform') ?? '');
        this.updateMarkerScale(marker);
        this.svgMarkers.push(marker);
        this.onMarkerAdded(marker, this.svgMarkers.length - 1, event);
    }

    /**
     * Returns the tracked marker under the click, or `null` if the click did not
     * land on one. Hit-testing is geometric (the click's client coordinates vs.
     * each marker's bounding box) rather than `event.target`-based: pointer
     * events are captured by an overlay canvas that sits on top of the SVG
     * layer, so `event.target` is never an SVG marker. Markers are tested in
     * reverse insertion order so the topmost (most recently added) wins.
     */
    markerFromEvent(event: React.MouseEvent): SVGElement | null {
        const x = event.clientX;
        const y = event.clientY;
        for (let i = this.svgMarkers.length - 1; i >= 0; i--) {
            const marker = this.svgMarkers[i];
            const rect = marker.getBoundingClientRect();
            if (x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom) {
                return marker;
            }
        }
        return null;
    }

    /**
     * Hook invoked after a marker is added to the SVG. `index` is its position
     * in {@link svgMarkers}. Default is a no-op; subclasses override to keep
     * parallel state (e.g. a points list) in sync.
     */
    onMarkerAdded(_marker: SVGElement, _index: number, _event: React.MouseEvent): void {}

    /**
     * Hook invoked after a marker is removed from the SVG (only reachable with
     * `onClickAction === 'add_or_remove'`). `index` is the position the marker
     * occupied in {@link svgMarkers} before removal. Default is a no-op.
     */
    onMarkerRemoved(_marker: SVGElement, _index: number, _event: React.MouseEvent): void {}

    /**
     * Removes every marker this behavior has added to the SVG. Asset templates
     * and option state are preserved; only transient interaction state is
     * cleared. Subclasses with their own state should override and call
     * `super.reset(options)` first.
     */
    override reset(options?: any): void {
        for (const marker of this.svgMarkers) {
            marker.parentNode?.removeChild(marker);
        }
        this.svgMarkers = [];
    }

}

BehaviorRegister.register("svg_annotation", SvgAnnotationBehavior,
    'Click-to-annotate an SVG layer with any custom primitives.');