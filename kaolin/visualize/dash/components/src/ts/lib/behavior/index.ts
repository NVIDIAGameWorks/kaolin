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
 * Bundled behaviors that can be attached to a Kaolin viewer, each registered
 * with {@link core.behavior.BehaviorRegister} by a string name, shown below,
 * allowing it to be accessible from python. Each behavior further may define
 * a **schema of options** that can be set from python, to *configure* the
 * behavior or *instrument* user controls for its options.
 *
 * E.g. from python:
 * ```python
 * viewer_builder = kaolin.visualize.dash.ViewerBuilder()
 *
 * # Add a canvas layer.
 * layer_id = viewer_builder.add_layer()
 *
 * # Add a behavior, and attach to a layer (if element-bound).
 * behavior_id = viewer_builder.add_behavior(
 *     'konva_selection',  # Registered name
 *     active_layer_id=layer_id,
 *     options={"mode": "box"})
 *
 * # Use a shortcut instrument UI controls for a behavior.
 * controls = viewer_builder.add_user_behavior_options(
 *     behavior_id,
 *     options=['mode'])
 * ```
 *
 * @groupDescription Behavior: DrawingBehavior
 * Paints freehand brush strokes (draw / erase) onto a canvas in response to pointer input.
 * This is a {@link core.behavior.CanvasBehavior | CanvasBehavior}
 * (an {@link core.behavior.ElementBoundBehavior | ElementBoundBehavior}).
 *
 * Registered name: `'drawing'`.
 *
 * Key options: `color`, `thickness`, `mode`.
 *
 * @groupDescription Behavior: DrawRemoteImageBehavior
 * Draws an image sent in a message from the server onto its configured canvas element.
 * This is an {@link core.behavior.ElementBoundBehavior | ElementBoundBehavior}
 * and a {@link core.behavior.MessageHandler | MessageHandler}.
 *
 * Registered name: `'draw_remote_image'`.
 *
 * Key options: `messageTag`.
 *
 * @groupDescription Behavior: KonvaDrawingBehavior
 * Konva-backed in-stage drawing (box / polygon / freeform); committed gestures persist as shapes.
 * This is an {@link core.behavior.InteractiveBehavior | InteractiveBehavior}
 * (an {@link core.behavior.ElementBoundBehavior | ElementBoundBehavior}).
 *
 * Registered name: `'konva_drawing'`.
 *
 * Key options: `mode`, `color`, `opacity`.
 *
 * @groupDescription Behavior: KonvaSelectionBehavior
 * Konva drawing whose committed gestures are rasterized and composited onto an external canvas.
 * This is a {@link KonvaDrawingBehavior} subclass
 * (an {@link core.behavior.ElementBoundBehavior | ElementBoundBehavior}).
 *
 * Registered name: `'konva_selection'`.
 *
 * Key options: `action`, `compositeCanvasId`.
 *
 * @groupDescription Behavior: ReceiveMessageBehavior
 * Subscribes to a WebSocket message tag, optionally transforms each payload, and fans it out to other behaviors.
 * This is a {@link core.behavior.MessageHandler | MessageHandler}.
 *
 * Registered name: `'receive_message'`.
 *
 * Key options: `tag`, `alertBehaviors`.
 *
 * @groupDescription Behavior: RecordInteractionsBehavior
 * Records pointer/click interactions into an in-memory log; test scaffolding for exercising other behaviors.
 * This is an {@link core.behavior.InteractiveBehavior | InteractiveBehavior}
 * (an {@link core.behavior.ElementBoundBehavior | ElementBoundBehavior}).
 *
 * Registered name: `'record_interactions'`.
 *
 * Key options: `record`, `elementType`.
 *
 * @groupDescription Behavior: SendCursorBehaviorComponent
 * Publishes cursor/pointer updates from a bound element over a registered WebSocket connection.
 * This is an {@link core.behavior.ElementBoundBehavior | ElementBoundBehavior},
 * a {@link core.behavior.PointerEventHandler | PointerEventHandler}, and a
 * {@link core.behavior.ClickEventHandler | ClickEventHandler}.
 *
 * Registered name: `'send_cursor'`.
 *
 * Key options: `connectionId`, `messageTag`.
 *
 * @groupDescription Behavior: SendCameraBehaviorComponent
 * Forwards camera updates over a registered WebSocket connection.
 *
 * Registered name: `'send_camera'`.
 *
 * Key options: `connectionId`, `messageTag`.
 *
 * @groupDescription Behavior: SendValueBehaviorComponent
 * Generic value-publisher over a registered WebSocket connection, exposing a `setValue` handle.
 *
 * Registered name: `'send_value'`.
 *
 * Key options: `connectionId`, `messageTag`.
 *
 * @groupDescription Behavior: SvgAnnotationBehavior
 * Click-to-annotate an SVG layer with configurable marker assets.
 * This is an {@link core.behavior.InteractiveBehavior | InteractiveBehavior}
 * (an {@link core.behavior.ElementBoundBehavior | ElementBoundBehavior}).
 *
 * Registered name: `'svg_annotation'`.
 *
 * Key options: `activeAsset`, `onClickAction`.
 *
 * @module
 */


export * from './drawing';                // Behavior: DrawingBehavior
export * from './draw_remote_image';      // Behavior: DrawRemoteImageBehavior
export * from './konva_drawing';          // Behavior: KonvaDrawingBehavior
export * from './konva_selection';        // Behavior: KonvaSelectionBehavior
export * from './receive_message';        // Behavior: ReceiveMessageBehavior
export * from './record_interactions';    // Behavior: RecordInteractionsBehavior
export * from './send_cursor';            // Behavior: SendCursorBehaviorComponent
export * from './send_message';           // Behavior: SendCameraBehaviorComponent, SendValueBehaviorComponent
export * from './svg_annotations';        // Behavior: SvgAnnotationBehavior
