import { setCanvasBufferSize } from '../util/canvas';

export enum ViewportResizeMode {
    FIXED = 'fixed',
    ADAPTIVE = 'adaptive',
    FIX_ASPECT = 'fix_aspect'
}


/**
 * Compute the largest size that preserves `initialSize`'s aspect ratio while
 * fitting inside `containerSize` (letterboxed; never overflows). Falls back to
 * the container size when the aspect ratio is degenerate (zero, negative, or
 * non-finite).
 */
export function computeContainedFixedAspectSize(
    initialSize: { width: number, height: number },
    containerSize: { width: number, height: number }
): { width: number, height: number } {
    if (initialSize.width <= 0 || initialSize.height <= 0 ||
        containerSize.width <= 0 || containerSize.height <= 0) {
        return { width: containerSize.width, height: containerSize.height };
    }
    const scale = Math.min(
        containerSize.width / initialSize.width,
        containerSize.height / initialSize.height
    );
    return {
        width: Math.floor(initialSize.width * scale),
        height: Math.floor(initialSize.height * scale),
    };
}


/**
 * Compute the canvas pixel size for a given resize mode within a container.
 *
 * - FIXED: canvas keeps its initial pixel size regardless of the container.
 * - ADAPTIVE: canvas fills the container; aspect ratio follows the container.
 * - FIX_ASPECT: largest size preserving the initial aspect ratio that still
 *   fits inside the container (letterboxed; never overflows).
 *
 * When `incorporateDevicePixelRatio` is true, the result is multiplied by
 * `window.devicePixelRatio` to yield the drawing-buffer size (sharp on hi-DPI);
 * otherwise the returned size is in CSS pixels.
 *
 * Returns integer width/height (no "px" suffix).
 */
export function computeContainedDisplaySize(
    initialSize: { width: number, height: number },
    resizeMode: ViewportResizeMode,
    containerSize: { width: number, height: number }
): { width: number, height: number } {
    let width: number;
    let height: number;

    switch (resizeMode) {
        case ViewportResizeMode.FIXED:
            width = initialSize.width;
            height = initialSize.height;
            break;

        case ViewportResizeMode.FIX_ASPECT: {
            ({ width, height } = computeContainedFixedAspectSize(initialSize, containerSize));
            break;
        }

        case ViewportResizeMode.ADAPTIVE:
        default:
            width = Math.floor(containerSize.width);
            height = Math.floor(containerSize.height);
            break;
    }
    return { width: Math.floor(width), height: Math.floor(height) };
}

export function adjustByDevicePixelRatio(size: { width: number, height: number }): { width: number, height: number } {
    const dpr = window.devicePixelRatio || 1;
    return { width: Math.floor(size.width * dpr), height: Math.floor(size.height * dpr) };
}


export function defaultSetElementDimensions(
    el: HTMLElement | null,
    width: number,
    height: number,
    incorporateDevicePixelRatio: boolean = false,
) {
    if (!el) {
        return;
    }
    
    switch (el.tagName.toLocaleLowerCase()) {
        case 'canvas': {
            const dpr = incorporateDevicePixelRatio ? (window.devicePixelRatio || 1) : 1;
            setCanvasBufferSize(el, Math.floor(width * dpr), Math.floor(height * dpr));
            break;
        }
        case 'svg': {
            const svg = el as unknown as SVGSVGElement;
            if (svg.width.baseVal.value !== width) svg.setAttribute('width', String(width));
            if (svg.height.baseVal.value !== height) svg.setAttribute('height', String(height));
            break;
        }
        case 'div': {
            const wPx = `${width}px`;
            const hPx = `${height}px`;
            if (el.style.width !== wPx) el.style.width = wPx;
            if (el.style.height !== hPx) el.style.height = hPx;
            break;
        }
        default:
            el.style.width = `${width}px`;
            el.style.height = `${height}px`;
            break;
    }
}