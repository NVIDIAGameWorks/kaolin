import React, { useEffect, useRef } from 'react';

import { resizeCanvasKeepContent, setCanvasBufferSize } from '../util/canvas';
import { logger } from '../util/logging';
import { createGlobalRegistry } from '../util/registry';
import { adjustByDevicePixelRatio, isFixedAspect } from './viewport';


// TODO: should be whatever, not fixed properties
interface LayerProps {
    id?: string;
    width?: number;
    height?: number;
    extraClasses?: string;
}

// Add proper prop types for BehaviorRunner
interface LayerSpec {
    layerIdentifier: string;
    elementType: string;
    layerProps: LayerProps | null;
    onRegister: (identifier: string, layerRef: React.RefObject<HTMLElement> | null) => void;
}

/** 
 * Type for a layer constructor function.
 * Takes a ref and optional layer props, returns a React element.
 */
type LayerConstructor = (ref: React.RefObject<HTMLElement>, props?: LayerProps) => React.ReactElement;

/**
 * Static registry class for layer types.
 * Allows mapping element type names (e.g., "canvas", "svg") to constructor functions
 * that return the corresponding TSX elements.
 * 
 * TODO: update to example that's been verified to work.
 * Example:
 window.kaolin.LayerRegister.register("custom-layer", function(ref, props) {
    return React.createElement("canvas", {
        ref: ref,
        width: props && props.width,
        height: props && props.height,
        className: "my-custom-canvas " + (props && props.extraClasses || "")
    });
});
 * ));
 */
class LayerRegister extends createGlobalRegistry<LayerConstructor>() { }

// Register default layer types
LayerRegister.register("canvas", (ref, props) => (
    <canvas
        id={props?.id}
        ref={ref as React.RefObject<HTMLCanvasElement>}
        width={props?.width}
        height={props?.height}
        className={`sub-canvas${props?.extraClasses ? ` ${props.extraClasses}` : ''}`}
    />
));

LayerRegister.register("svg", (ref, props) => (
    <svg
        id={props?.id}
        ref={ref as unknown as React.RefObject<SVGSVGElement>}
        xmlns="http://www.w3.org/2000/svg"
        width={props?.width}
        height={props?.height}
        viewBox="0 0 100 100"
        className={`sub-canvas sub-svg${props?.extraClasses ? ` ${props.extraClasses}` : ''}`}
    />
));

LayerRegister.register("div", (ref, props) => (
    <div
        id={props?.id}
        ref={ref as React.RefObject<HTMLDivElement>}
        className={`sub-canvas${props?.extraClasses ? ` ${props.extraClasses}` : ''}`}
    />
));

const LayerComponent = ({
    layerIdentifier,
    elementType,
    layerProps,
    onRegister
}: LayerSpec) => {

    const elementRef = useRef<HTMLElement>(null);

    const layerConstructor = LayerRegister.get(elementType);
    if (!layerConstructor) {
        return null;
    }

    useEffect(() => {
        // Register the API object when the component mounts
        onRegister(layerIdentifier, elementRef);

        // Return a cleanup function
        return () => {
            // Unregister (pass null or undefined for cleanup) when the component unmounts
            onRegister(layerIdentifier, null);
        };
    }, [layerIdentifier, elementType, onRegister]);

    return layerConstructor(elementRef, layerProps);
};

/**
 * Resize a layer element, keeping its content where that is both meaningful and
 * possible. What a resize costs depends on the layer type:
 *
 * - `canvas`: the size *is* the drawing buffer, and any write to it clears the
 *   buffer. 2D content is carried across only when the new size has the same
 *   aspect ratio (see
 *   {@link core.viewport.isFixedAspect | isFixedAspect}), because the only way
 *   to carry it is to stretch it and an aspect change would distort it. On an
 *   aspect change the buffer is resized and left blank for whatever owns it to
 *   repaint. An existing non-2D context falls back to a plain buffer resize.
 * - `svg`: nothing can be lost, since the content is vector and is placed by
 *   the `viewBox`, so only the `width`/`height` attributes are brought in line.
 *   Under the viewer's CSS (`.sub-canvas` stretches every layer to fill the
 *   container) those attributes do not drive layout, so this only keeps the
 *   element self-describing. Whether the `viewBox` should be rewritten to the
 *   new aspect is a question about the drawing space rather than the element,
 *   and belongs to the behavior that owns it, so it is left alone here.
 * - anything else (`div`, custom layers): nothing to do. With no drawing buffer
 *   of its own, the element is already sized by CSS.
 *
 * Sizes are in CSS pixels. A canvas' drawing buffer is the one thing that is
 * not: it is scaled by `window.devicePixelRatio` (see
 * {@link core.viewport.adjustByDevicePixelRatio | adjustByDevicePixelRatio}),
 * since a CSS-sized buffer is upscaled by the browser to fill the element and
 * renders visibly soft on a HiDPI display. This helper applies the ratio,
 * so `canvas.width` is in device pixels while a behavior's
 * `setDimensions` argument is in CSS pixels.
 *
 * After all layers are sized, the viewer calls every behavior's
 * `setDimensions` to synchronize renderer state and redraw cleared buffers.
 *
 * @param inputLayer - Layer element; null and undefined are tolerated.
 * @param newWidth - Target width in pixels; floored, must be >= 1.
 * @param newHeight - Target height in pixels; floored, must be >= 1.
 * @returns `true` if the layer now has the requested size, `false` if the layer
 *     type carries no explicit size of its own or the resize failed.
 */
function resizeLayer(inputLayer: Element | null | undefined, newWidth: number, newHeight: number): boolean {
    if (!inputLayer) {
        return false;
    }

    const width = Math.floor(newWidth);
    const height = Math.floor(newHeight);
    if (!Number.isFinite(width) || !Number.isFinite(height) || width < 1 || height < 1) {
        logger.error(`resizeLayer: invalid target size ${newWidth}x${newHeight}`);
        return false;
    }

    if (inputLayer instanceof HTMLCanvasElement) {
        const buffer = adjustByDevicePixelRatio({ width, height });
        // Already the requested size: writing it again would clear the buffer
        // for no gain.
        if (inputLayer.width === buffer.width && inputLayer.height === buffer.height) {
            return true;
        }
        // Existing WebGL contexts return null here; their behaviors redraw
        // after the buffer resize.
        if (isFixedAspect({ width: inputLayer.width, height: inputLayer.height }, buffer)
            && inputLayer.getContext('2d')) {
            return resizeCanvasKeepContent(inputLayer, buffer.width, buffer.height);
        }
        setCanvasBufferSize(inputLayer, buffer.width, buffer.height);
        return inputLayer.width === buffer.width && inputLayer.height === buffer.height;
    }

    if (inputLayer instanceof SVGSVGElement) {
        if (inputLayer.getAttribute('width') !== String(width)) {
            inputLayer.setAttribute('width', String(width));
        }
        if (inputLayer.getAttribute('height') !== String(height)) {
            inputLayer.setAttribute('height', String(height));
        }
        return true;
    }

    return false;
}

export { LayerRegister, LayerConstructor, LayerComponent, resizeLayer };