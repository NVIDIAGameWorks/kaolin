import React, { useEffect, useRef } from 'react';

import { createGlobalRegistry } from '../util/registry';


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

export { LayerRegister, LayerConstructor, LayerComponent };