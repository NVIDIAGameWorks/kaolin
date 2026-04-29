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
 * Image helpers for the Node.js test environment: load a PNG fixture from disk to
 * raw RGBA pixels ({@link loadPngPixels}) and compare two RGBA buffers with a
 * tolerance ({@link diffRgba} / {@link assertImagesSimilar}).
 *
 * The only node-specific piece is reading the file path; the actual decode is
 * delegated to the production helper `imageDataFromBlob` (which runs through the
 * happy-dom canvas adapter), so a DOM must be installed first (see `registerDom`
 * in {@link module:helpers/dom}). Comparison is plain arithmetic and needs no DOM.
 *
 * @module
 */

import * as fs from 'fs';
import { assert } from 'chai';

import { imageDataFromBlob } from '@kaolin/util/canvas';

/**
 * Load a PNG file from disk and decode it to raw RGBA bytes at the image's own
 * (natural) size.
 *
 * The file bytes are read here (node-only) and handed to the production
 * `imageDataFromBlob` decoder, exactly as a client would decode an uploaded
 * `File`. Requires a registered DOM.
 *
 * @param filePath - Absolute path to the PNG file.
 * @returns Flat RGBA bytes, length `imageWidth * imageHeight * 4`.
 */
export async function loadPngPixels(filePath: string): Promise<Uint8ClampedArray> {
    const buffer = fs.readFileSync(filePath);
    const blob = new Blob([new Uint8Array(buffer)], { type: 'image/png' });
    const image = await imageDataFromBlob(blob);
    if (!image) throw new Error(`loadPngPixels: failed to decode '${filePath}'`);
    return image.data;
}

/**
 * Read a PNG's pixel dimensions from its header, without decoding the image or
 * needing a DOM. A PNG is an 8-byte signature followed by the IHDR chunk, whose
 * big-endian width / height live at byte offsets 16 and 20.
 *
 * Useful for sizing a render target to a golden so the pixel grids line up.
 *
 * @param filePath - Absolute path to the PNG file.
 * @returns The image `{ width, height }` in pixels.
 */
export function pngSize(filePath: string): { width: number; height: number } {
    const buffer = fs.readFileSync(filePath);
    const signature = '\x89PNG\r\n\x1a\n';
    if (buffer.length < 24 || buffer.toString('latin1', 0, 8) !== signature) {
        throw new Error(`pngSize: '${filePath}' is not a PNG`);
    }
    return { width: buffer.readUInt32BE(16), height: buffer.readUInt32BE(20) };
}

/** Outcome of comparing two RGBA buffers, see {@link diffRgba}. */
export interface ImageDiff {
    /** Number of pixels compared (`buffer length / 4`). */
    totalPixels: number;
    /** Largest absolute per-channel difference seen across all pixels. */
    maxChannelDiff: number;
    /** Pixels with at least one channel differing by more than the tolerance. */
    numDiffPixels: number;
    /** `numDiffPixels / totalPixels`, in [0, 1]. */
    fracDiffPixels: number;
}

/**
 * Compare two equal-length flat RGBA buffers.
 *
 * A pixel "differs" when any of its four channels differs by more than
 * `channelTolerance` (absolute). This tolerates the minor per-channel deviations
 * expected between rasterizers (e.g. anti-aliasing at stroke edges) while still
 * counting the pixels that genuinely changed.
 *
 * @param actual - Flat RGBA bytes to test.
 * @param expected - Flat RGBA bytes to compare against (same length as `actual`).
 * @param channelTolerance - Max absolute per-channel difference still treated as
 *     equal. Defaults to 0 (exact).
 * @returns The {@link ImageDiff} summary.
 */
export function diffRgba(
    actual: Uint8ClampedArray, expected: Uint8ClampedArray, channelTolerance = 0): ImageDiff {
    if (actual.length !== expected.length) {
        throw new Error(`diffRgba: length mismatch (${actual.length} vs ${expected.length}).`);
    }
    const totalPixels = actual.length / 4;
    let maxChannelDiff = 0;
    let numDiffPixels = 0;
    for (let p = 0; p < totalPixels; p++) {
        let pixelDiffers = false;
        for (let c = 0; c < 4; c++) {
            const channelDiff = Math.abs(actual[p * 4 + c] - expected[p * 4 + c]);
            if (channelDiff > maxChannelDiff) maxChannelDiff = channelDiff;
            if (channelDiff > channelTolerance) pixelDiffers = true;
        }
        if (pixelDiffers) numDiffPixels++;
    }
    return { totalPixels, maxChannelDiff, numDiffPixels, fracDiffPixels: numDiffPixels / totalPixels };
}

/** Options for {@link assertImagesSimilar}. */
export interface ImagesSimilarOptions {
    /** Max absolute per-channel difference treated as equal. Defaults to 0. */
    channelTolerance?: number;
    /** Max allowed fraction of differing pixels, in [0, 1]. Defaults to 0. */
    maxFracDiff?: number;
}

/**
 * Assert two RGBA buffers match within tolerance, failing with a diagnostic that
 * reports the differing-pixel fraction and the max channel difference.
 *
 * @param actual - Flat RGBA bytes to test.
 * @param expected - Flat RGBA bytes to compare against.
 * @param options - Tolerances, see {@link ImagesSimilarOptions}.
 * @param message - Prefix for the assertion message.
 * @returns The {@link ImageDiff} summary (for further inspection on success).
 */
export function assertImagesSimilar(
    actual: Uint8ClampedArray, expected: Uint8ClampedArray,
    options: ImagesSimilarOptions = {}, message = 'images differ'): ImageDiff {
    const { channelTolerance = 0, maxFracDiff = 0 } = options;
    const diff = diffRgba(actual, expected, channelTolerance);
    assert.isAtMost(
        diff.fracDiffPixels, maxFracDiff,
        `${message}: ${(diff.fracDiffPixels * 100).toFixed(3)}% of ${diff.totalPixels} pixels differ by more than `
        + `${channelTolerance}/channel (max channel diff ${diff.maxChannelDiff}); allowed ${(maxFracDiff * 100).toFixed(3)}%`);
    return diff;
}
