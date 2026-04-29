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
 * Utilities for manipulating HTML canvas element and associated image data.
 * 
 * @module
 */

import React from 'react';

import { getRelativeCoordinates } from './cursor';

/** Typed array with shape metadata for reshaping (e.g., in NumPy) */
export type ShapedTypedArray<T extends ArrayBufferView> = T & { shape: number[] };

/** Accepts either a canvas element directly or a DOM id string. */
export type CanvasOrId = HTMLCanvasElement | string;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Extract pixel data from a canvas element as a Uint8ClampedArray.
 *
 * @param canvasOrId - Canvas element or its DOM id.
 * @returns Uint8ClampedArray with shape [height, width, 4], or null on failure.
 */
export function typedArrayFromCanvas(canvasOrId: CanvasOrId): ShapedTypedArray<Uint8ClampedArray> | null {
    const ctx = resolveContext(canvasOrId);
    if (!ctx) return null;
    const { width, height } = ctx.canvas;
    const imageData = ctx.getImageData(0, 0, width, height);
    const result = imageData.data as ShapedTypedArray<Uint8ClampedArray>;
    result.shape = [height, width, 4];
    return result;
}

/**
 * Extract just the alpha channel from a canvas element.
 *
 * @param canvasOrId - Canvas element or its DOM id.
 * @returns Uint8Array with shape [height, width], or null on failure.
 */
export function alphaChannelFromCanvas(canvasOrId: CanvasOrId): ShapedTypedArray<Uint8Array> | null {
    const ctx = resolveContext(canvasOrId);
    if (!ctx) return null;
    const { width, height } = ctx.canvas;
    const imageData = ctx.getImageData(0, 0, width, height);
    const uint32View = new Uint32Array(imageData.data.buffer);
    const numPixels = uint32View.length;
    const result = new Uint8Array(numPixels) as ShapedTypedArray<Uint8Array>;
    // RGBA in memory is stored as 0xAABBGGRR on little-endian systems.
    // Alpha is the highest byte, so shift right by 24 bits.
    for (let i = 0; i < numPixels; i++) {
        result[i] = (uint32View[i] >> 24) & 0xFF;
    }
    result.shape = [height, width];
    return result;
}

/**
 * Read the full RGBA pixel buffer of a canvas as an `ImageData`.
 *
 * @param canvasOrId - Canvas element or its DOM id.
 * @returns The canvas `ImageData`, or null if the canvas or 2D context is unavailable.
 */
export function imageDataFromCanvas(canvasOrId: CanvasOrId): ImageData | null {
    const ctx = resolveContext(canvasOrId);
    if (!ctx) return null;
    const { width, height } = ctx.canvas;
    return ctx.getImageData(0, 0, width, height);
}

/**
 * Convert a pointer/mouse event to canvas pixel coordinates.
 *
 * @param event - The pointer or mouse event to read viewport coordinates from.
 * @param inputCanvas - Canvas to measure against; defaults to `event.currentTarget`.
 * @returns The event position in the canvas's intrinsic pixel space.
 */
export function getCanvasCoordinates(
    event: React.PointerEvent<Element> | React.MouseEvent<Element> | PointerEvent | MouseEvent,
    inputCanvas?: HTMLCanvasElement): { x: number; y: number } {
    const canvas = inputCanvas ? inputCanvas : event.currentTarget as HTMLCanvasElement;
    // getRelativeCoordinates handles the viewport-vs-document (scroll) correctness;
    // scaling the fractions by the intrinsic buffer size maps into canvas pixel space.
    const { fracX, fracY } = getRelativeCoordinates(event, canvas);
    return { x: fracX * canvas.width, y: fracY * canvas.height };
}

/**
 * Draw a shaped typed-array buffer directly to a canvas element.
 *
 * Accepts Uint8ClampedArray, Uint8Array, or ArrayBuffer with a `.shape`
 * property of `[height, width, channels]`. Only 4-channel (RGBA) input is
 * supported; the byte length must equal `height * width * 4`.
 *
 * @param canvasOrId - Canvas element or its DOM id.
 * @param data - RGBA pixel data with a `.shape` property `[height, width, ...]`.
 * @returns true if successful, false otherwise.
 */
export function drawTypedArrayToCanvas(
    canvasOrId: CanvasOrId,
    data: ShapedTypedArray<Uint8ClampedArray> | ShapedTypedArray<Uint8Array> | (ArrayBuffer & { shape: number[] }),
): boolean {
    const ctx = resolveContext(canvasOrId);
    if (!ctx) return false;
    const shape: number[] | undefined = (data as any).shape;
    if (!shape || shape.length < 2) {
        console.error('drawTypedArrayToCanvas: input array is missing required shape attribute [height, width, ...]');
        return false;
    }
    const height = shape[0];
    const width  = shape[1];
    const channels = shape[2];
    if (channels != 4) {
        console.error(`drawTypedArrayToCanvas: only 4-channel RGBA input supported, but ${shape} shape given`)
    }
    // We continue in case buffer length agrees with 4-channel input
    const byteLength = data instanceof ArrayBuffer ? data.byteLength : (data as ArrayBufferView).byteLength;
    const expectedByteLength = height * width * 4;
    if (byteLength !== expectedByteLength) {
        console.error(`drawTypedArrayToCanvas: shape [${height}, ${width}, ...] requires ${expectedByteLength} bytes, ` +
            `but buffer has ${byteLength}`);
        return false;
    }
    let buffer: Uint8ClampedArray;
    if (data instanceof Uint8ClampedArray) {
        buffer = data;
    } else if (data instanceof Uint8Array) {
        buffer = new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength);
    } else {
        buffer = new Uint8ClampedArray(data as ArrayBuffer);
    }
    ctx.putImageData(new ImageData(buffer as any, width, height), 0, 0);
    return true;
}

/**
 * Encode the full RGBA contents of a canvas as a PNG or JPEG `Blob`.
 *
 * - ``'png'`` (default): lossless; alpha is preserved.
 * - ``'jpeg'``: lossy, typically smaller. JPEG cannot carry alpha — transparent
 *   pixels are composited against a background before encoding. Use PNG if you
 *   need alpha.
 *
 * @param canvasOrId - Canvas element or its DOM id.
 * @param format - ``'png'`` (default) or ``'jpeg'``.
 * @param jpegQuality - 0..1 quality for JPEG encoding (default 0.9). Ignored for PNG.
 * @returns The encoded Blob, or null if the canvas is not found or encoding fails.
 */
export async function blobFromCanvas(
    canvasOrId: CanvasOrId,
    format: 'png' | 'jpeg' = 'png',
    jpegQuality: number = 0.9,
): Promise<Blob | null> {
    const canvas = resolveCanvas(canvasOrId);
    if (!canvas) return null;
    const mimeType = format === 'png' ? 'image/png' : 'image/jpeg';
    return new Promise(resolve => {
        canvas.toBlob(blob => {
            if (!blob) console.error(`blobFromCanvas: encoder returned null (${mimeType})`);
            resolve(blob);
        }, mimeType, format === 'jpeg' ? jpegQuality : undefined);
    });
}

/**
 * Encode just the alpha channel of a canvas as a PNG `Blob`.
 *
 * The encoded image has `RGB = 0` everywhere and `A = sourceAlpha`. Read the
 * alpha channel of the decoded image to recover the mask.
 *
 * @param canvasOrId - Canvas element or its DOM id.
 * @returns PNG Blob, or null on failure.
 */
export async function blobAlphaChannelFromCanvas(canvasOrId: CanvasOrId): Promise<Blob | null> {
    const ctx = resolveContext(canvasOrId);
    if (!ctx) return null;
    const { width, height } = ctx.canvas;
    const imageData = ctx.getImageData(0, 0, width, height);
    // Zero RGB while keeping alpha. RGBA is stored as 0xAABBGGRR on
    // little-endian platforms; masking with 0xFF000000 keeps just the alpha byte.
    const u32 = new Uint32Array(imageData.data.buffer);
    for (let i = 0; i < u32.length; ++i) {
        u32[i] = u32[i] & 0xFF000000;
    }
    const scratch = document.createElement('canvas') as HTMLCanvasElement;
    scratch.width = width;
    scratch.height = height;
    const scratchCtx = scratch.getContext('2d');
    if (!scratchCtx) {
        console.error('blobAlphaChannelFromCanvas: could not get 2d context from scratch canvas');
        return null;
    }
    scratchCtx.putImageData(imageData, 0, 0);
    return new Promise(resolve => {
        scratch.toBlob(blob => {
            if (!blob) console.error('blobAlphaChannelFromCanvas: PNG encoder returned null');
            resolve(blob);
        }, 'image/png');
    });
}

/**
 * Draw a Blob onto a canvas element, replacing its current content.
 * Decode errors are logged to the console and leave the canvas unchanged.
 *
 * @param canvasOrId - Canvas element or its DOM id.
 * @param blob - Image blob to draw (PNG, JPEG, or any format supported by `createImageBitmap`).
 * @returns Promise that resolves when drawing is complete or silently on failure.
 */
export async function drawBlobToCanvas(canvasOrId: CanvasOrId, blob: Blob): Promise<void> {
    const ctx = resolveContext(canvasOrId);
    if (!ctx) return;
    try {
        const imageBitmap = await createImageBitmap(blob);
        ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
        ctx.drawImage(imageBitmap, 0, 0, ctx.canvas.width, ctx.canvas.height);
        imageBitmap.close();
    } catch (error) {
        console.error('drawBlobToCanvas: failed to decode blob into image:', error);
    }
}

/**
 * Decode an image `Blob` into its RGBA pixels as an `ImageData`, at the image's
 * own (natural) size.
 *
 * Accepts any format `createImageBitmap` supports (PNG, JPEG, ...). The blob is
 * decoded and drawn onto a scratch canvas to read back its pixels. This is the
 * client-side counterpart to a file upload: a `File` from an
 * `<input type="file">` is already a `Blob` and can be passed directly (see
 * {@link util.file.uploadImage | uploadImage}).
 *
 * @param blob - Image blob to decode.
 * @returns The decoded `ImageData` at the image's natural size, or null if
 *     decoding fails or no 2D context is available.
 */
export async function imageDataFromBlob(blob: Blob): Promise<ImageData | null> {
    let imageBitmap: ImageBitmap;
    try {
        imageBitmap = await createImageBitmap(blob);
    } catch (error) {
        console.error('imageDataFromBlob: failed to decode blob into image:', error);
        return null;
    }
    const { width, height } = imageBitmap;
    const scratch = document.createElement('canvas') as HTMLCanvasElement;
    scratch.width = width;
    scratch.height = height;
    const ctx = scratch.getContext('2d');
    if (!ctx) {
        console.error('imageDataFromBlob: could not get 2d context from scratch canvas');
        imageBitmap.close();
        return null;
    }
    ctx.drawImage(imageBitmap, 0, 0);
    imageBitmap.close();
    return ctx.getImageData(0, 0, width, height);
}

/**
 * Clear all pixels from a canvas element.
 *
 * @param canvasOrId - Canvas element or its DOM id.
 * @returns true if successful, false otherwise.
 */
export function clearCanvas(canvasOrId: CanvasOrId): boolean {
    const ctx = resolveContext(canvasOrId);
    if (!ctx) return false;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    return true;
}

/**
 * Set a canvas element's drawing-buffer size in device pixels. No-op for
 * non-canvas elements or null. Only writes when the value changes, so it is
 * safe to call from a ResizeObserver without triggering feedback loops.
 *
 * @param el - Target element; silently ignored if null or not a canvas.
 * @param width - Desired buffer width in device pixels.
 * @param height - Desired buffer height in device pixels.
 */
export function setCanvasBufferSize(el: HTMLElement | null, width: number, height: number) {
    if (el != null && el.tagName === 'CANVAS') {
        const canvas = el as HTMLCanvasElement;
        if (canvas.width !== width) canvas.width = width;
        if (canvas.height !== height) canvas.height = height;
    }
}


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Resolve a CanvasOrId to the canvas element, logging and returning null on failure.
 *
 * @param canvasOrId - Canvas element or its DOM id.
 * @returns The canvas element, or null if not found.
 */
export function resolveCanvas(canvasOrId: CanvasOrId): HTMLCanvasElement | null {
    if (typeof canvasOrId !== 'string') return canvasOrId;
    const el = document.getElementById(canvasOrId) as HTMLCanvasElement | null;
    if (!el) console.error(`Canvas with id "${canvasOrId}" not found`);
    return el;
}

/**
 * Resolve a CanvasOrId to its 2D context, logging and returning null on failure.
 *
 * @param canvasOrId - Canvas element or its DOM id.
 * @returns The 2D rendering context, or null if the canvas or context is unavailable.
 */
export function resolveContext(canvasOrId: CanvasOrId): CanvasRenderingContext2D | null {
    const canvas = resolveCanvas(canvasOrId);
    if (!canvas) return null;
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        const label = typeof canvasOrId === 'string' ? `"${canvasOrId}"` : `<element>`;
        console.error(`Could not get 2d context from canvas ${label}`);
    }
    return ctx;
}