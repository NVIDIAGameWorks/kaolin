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
import { z } from 'zod';

import { BehaviorRegister, CanvasBehavior, MessageHandler } from '../../core/behavior';
import { drawBlobToCanvas, drawTypedArrayToCanvas } from '../../util/canvas';
import { logger } from '../../util/logging';

/**
 * Configuration schema for {@link DrawRemoteImageBehavior}.
 *
 * ```typescript
 * z.object({
 *     messageTag: z.string().default('render')
 *         .describe('Tag of incoming messages whose payload is rendered to '
 *                   + 'this behavior\'s active canvas.'),
 * })
 * ```
 *
 * @group Behavior: DrawRemoteImageBehavior
 */
export const DrawRemoteImageOptionsSchema = z.object({
    messageTag: z.string().default('render')
        .describe('Tag of incoming messages whose payload is rendered to '
                  + 'this behavior\'s active canvas.'),
});

/** Options for {@link DrawRemoteImageBehavior}, parsed from {@link DrawRemoteImageOptionsSchema}.
 *
 * @group Behavior: DrawRemoteImageBehavior
 */
export type DrawRemoteImageOptions = z.infer<typeof DrawRemoteImageOptionsSchema>;


/**
 * A **behavior** for the Kaolin viewer, that draws an image sent in a message
 * from the server onto a canvas. This is an {@link core.behavior.ElementBoundBehavior | ElementBoundBehavior}
 * that must be configred with a canvas element. It is also a {@link core.behavior.MessageHandler | MessageHandler},
 * that accepts messages whose tag matches `options.message_tag` and whose payload
 * contains an `img` field that is either a `Blob` (PNG/JPEG) or a typed array
 * (Uint8ClampedArray / Uint8Array / ArrayBuffer) with a `.shape` property, fully compatible
 * with the {@link core.io.toBinary} and {@link core.io.fromBinary} functions, and their python counterparts.
 * 
 * Registered name: `'draw_remote_image'`.
 * 
 * Configuration schema: {@link DrawRemoteImageOptionsSchema}.
 * 
 * @group Behavior: DrawRemoteImageBehavior
 */
export class DrawRemoteImageBehavior extends CanvasBehavior implements MessageHandler {
    static override schema = DrawRemoteImageOptionsSchema;

    override options: DrawRemoteImageOptions;

    /**
     * @param options - Partial configuration; missing fields use schema defaults.
     */
    constructor(options: Partial<DrawRemoteImageOptions> = {}) {
      super();
      this.options = DrawRemoteImageOptionsSchema.parse(options);
  }

  /**
   * @returns The list of WebSocket message tags this behavior handles.
   */
  acceptedMessageTags(): string[] {
      return [this.options.messageTag];
  }

  onConnectionOpen(): void {

  }

  /**
   * Render an incoming image payload to the active canvas.
   *
   * @param messageTag - Tag of the incoming message.
   * @param messageContent - Message payload; expected to have an `img` field.
   */
  onMessage(messageTag: string, messageContent: any | null): void {
    if (messageTag != this.options.messageTag) return;
    if (!this.ctx) {
        logger.warn(`DrawRemoteImageBehavior: canvas context is not ready for drawing`);
        return;
    }

    const render = messageContent?.get('img');
    if (!render) {
        logger.warn(`DrawRemoteImageBehavior: message with expected tag ${this.options.messageTag} has no 'img' field`);
        return;
    }

    // The server can ship the image as either a raw typed-array + .shape
    // or as a `Blob` whose mime type (`image/png` / `image/jpeg`).
    if (render instanceof Blob) {
        drawBlobToCanvas(this.ctx.canvas, render);
        return;
    }
    const shape = render.shape;
    if (shape && (render instanceof Uint8ClampedArray
                  || render instanceof Uint8Array
                  || render instanceof ArrayBuffer)) {
        drawTypedArrayToCanvas(this.ctx.canvas, render as (Uint8ClampedArray | Uint8Array | ArrayBuffer) & { shape: number[] });
        return;
    }

    logger.warn(`DrawRemoteImageBehavior: unsupported image payload of type ${typeof render}`,
                render);
  }
};
BehaviorRegister.register('draw_remote_image', DrawRemoteImageBehavior,
    'Draws incoming WebSocket image payloads (typed arrays or blobs) to its active canvas HTML element.');
