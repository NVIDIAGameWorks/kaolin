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
 * Client-side file helpers: trigger browser downloads of in-memory data
 * ({@link downloadBlob} / {@link downloadTextFile}) and prompt the user to upload
 * an image ({@link uploadImage}).
 *
 * @module
 */

import { imageDataFromBlob } from './canvas';

/**
 * Trigger a browser download of an in-memory {@link Blob} as a file.
 *
 * Hands the blob to a transient `<a download>` element and clicks it so the
 * browser saves it under `filename` (the exact "Save As" prompt behavior
 * depends on the user's browser settings). Works for any content type (text,
 * JSON, images, binary, ...); the type is whatever the blob carries. The
 * temporary object URL is released after the click.
 *
 * @param filename - Suggested name for the downloaded file (e.g. `'data.json'`).
 * @param blob - The file contents to save.
 */
export function downloadBlob(filename: string, blob: Blob): void {
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    // Defer revocation so the click-initiated download isn't cancelled by
    // releasing the URL too early in some browsers.
    setTimeout(() => URL.revokeObjectURL(url), 0);
}

/**
 * Trigger a browser download of in-memory text as a file. Convenience wrapper
 * around {@link downloadBlob} that builds the blob from a string.
 *
 * @param filename - Suggested name for the downloaded file (e.g. `'data.json'`).
 * @param content - The text to save.
 * @param mimeType - MIME type for the blob; defaults to `'text/plain'`.
 */
export function downloadTextFile(filename: string, content: string, mimeType: string = 'text/plain'): void {
    downloadBlob(filename, new Blob([content], { type: mimeType }));
}

/**
 * Prompt the user to pick an image file and decode it to pixels.
 *
 * Opens a transient `<input type="file">` picker; once the user chooses a file
 * it is decoded via {@link util.canvas.imageDataFromBlob | imageDataFromBlob} at
 * its natural size. Resolves with the decoded `ImageData`, or `null` if the user
 * cancels or decoding fails. Must be called from a user gesture (e.g. a click
 * handler) for the picker to open.
 *
 * @param accept - `accept` filter for the file picker; defaults to `'image/*'`.
 * @returns The decoded `ImageData`, or null if cancelled or decoding failed.
 */
export function uploadImage(accept: string = 'image/*'): Promise<ImageData | null> {
    return new Promise(resolve => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = accept;
        input.style.display = 'none';
        document.body.appendChild(input);

        const cleanup = () => { if (input.parentNode) input.parentNode.removeChild(input); };
        input.addEventListener('change', async () => {
            const file = input.files?.[0];
            cleanup();
            resolve(file ? await imageDataFromBlob(file) : null);
        });
        // Fired when the picker is dismissed without choosing a file (modern browsers).
        input.addEventListener('cancel', () => { cleanup(); resolve(null); });
        input.click();
    });
}
