import { setCanvasBufferSize } from '../util/canvas';

export enum ViewportResizeMode {
    FIXED = 'fixed',
    ADAPTIVE = 'adaptive',
    FIXED_ASPECT = 'fixed_aspect'
}


/**
 * Report whether two viewport sizes share the same aspect ratio, to within a
 * small relative tolerance that absorbs the sub-pixel rounding of a layout
 * change. A `true` result means the resize was a uniform scale, so content can
 * be rescaled into it without distortion.
 *
 * Only the ratio is compared, so the two sizes need not be in the same units
 * (e.g. a canvas drawing buffer in device pixels against a viewport in CSS
 * pixels). Degenerate sizes -- zero, negative, or non-finite in either
 * dimension -- have no meaningful aspect and report `false`, so callers take
 * their aspect-changed branch instead of acting on a nonsense ratio.
 *
 * @param prevSize - Size before the resize.
 * @param newSize - Size after the resize.
 * @param tolerance - Largest relative difference between the two aspect ratios
 *     still considered equal, as a fraction (default 1%).
 * @returns `true` when both sizes are valid and their aspect ratios agree.
 */
export function isFixedAspect(
    prevSize: { width: number, height: number },
    newSize: { width: number, height: number },
    tolerance: number = 0.01
): boolean {
    const isValid = (size: { width: number, height: number }) =>
        Number.isFinite(size.width) && Number.isFinite(size.height) &&
        size.width > 0 && size.height > 0;
    if (!isValid(prevSize) || !isValid(newSize)) {
        return false;
    }
    const prevAspect = prevSize.width / prevSize.height;
    const newAspect = newSize.width / newSize.height;
    return Math.abs(prevAspect - newAspect) <= tolerance * Math.max(prevAspect, newAspect);
}


/**
 * Compute the largest integer size that has *exactly* `initialSize`'s aspect
 * ratio and still fits inside `containerSize` (letterboxed; never overflows).
 *
 * Exactness is what dictates the arithmetic. Scaling by a real factor and
 * rounding leaves the result a fraction of a pixel off the target ratio, so the
 * size is instead built as a whole multiple of the aspect ratio reduced by its
 * greatest common divisor: a 1600x900 camera reduces to 16x9, every result is a
 * multiple of that, and the ratio is preserved bit for bit.
 *
 * The cost is quantization. A camera whose dimensions are co-prime (501x500,
 * say) reduces to itself, so the only sizes available are multiples of the full
 * camera size. Callers that would rather track the container closely should
 * accept an approximate aspect instead.
 *
 * Falls back to the container size when either size is degenerate (zero,
 * negative, or non-finite).
 */
export function computeContainedFixedAspectSize(
    initialSize: { width: number, height: number },
    containerSize: { width: number, height: number }
): { width: number, height: number } {
    if (initialSize.width <= 0 || initialSize.height <= 0 ||
        containerSize.width <= 0 || containerSize.height <= 0) {
        return { width: containerSize.width, height: containerSize.height };
    }
    const unit = reduceToSmallestRatio(initialSize);
    // Whole units that fit in both directions, never fewer than one, so a
    // container narrower than a single unit still yields a usable size.
    const units = Math.max(1, Math.min(
        Math.floor(containerSize.width / unit.width),
        Math.floor(containerSize.height / unit.height)
    ));
    return { width: unit.width * units, height: unit.height * units };
}


/**
 * Reduce a size to the smallest integer pair with the same ratio, e.g.
 * 1600x900 to 16x9. Non-integer input is rounded first.
 */
function reduceToSmallestRatio(size: { width: number, height: number }): { width: number, height: number } {
    const width = Math.round(size.width);
    const height = Math.round(size.height);
    const divisor = greatestCommonDivisor(width, height);
    return divisor > 1 ? { width: width / divisor, height: height / divisor } : { width, height };
}


/** Greatest common divisor of two integers, by the Euclidean algorithm. */
function greatestCommonDivisor(a: number, b: number): number {
    let x = Math.abs(a);
    let y = Math.abs(b);
    while (y > 0) {
        const remainder = x % y;
        x = y;
        y = remainder;
    }
    return x;
}


/**
 * Compute the viewport pixel size for a given resize mode within a container.
 *
 * - FIXED: keeps its initial pixel size regardless of the container.
 * - ADAPTIVE: fills the container; aspect ratio follows the container.
 * - FIXED_ASPECT: largest size preserving the initial aspect ratio exactly that
 *   still fits inside the container (letterboxed; never overflows).
 *
 * The result is in CSS pixels; apply {@link adjustByDevicePixelRatio} to get a
 * drawing-buffer size. Width and height are integers (no "px" suffix).
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

        case ViewportResizeMode.FIXED_ASPECT: {
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
    // Falls back to 1 with no window (server-side rendering, non-DOM tests),
    // which is also what a browser reports for a non-HiDPI display.
    const dpr = (typeof window === 'undefined' ? 1 : window.devicePixelRatio) || 1;
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