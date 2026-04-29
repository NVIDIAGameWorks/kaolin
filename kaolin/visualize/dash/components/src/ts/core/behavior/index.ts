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
 * Behaviors add *client-side* interactivity to the Kaolin viewer —
 * pointer/click handling,
 * tagged-message exchange, camera control, and live option edits — and are
 * attached to a viewer from Python by their registered name. See
 * {@link lib.behavior | bundled behaviors} for the ready-made set. You can also
 * write your own in plain JavaScript: implement
 * {@link BehaviorInterface | behavior interfaces} or start with a
 * {@link Behavior | base class}, then register it under a name
 * that will dynamically resolve in the browser, even if configured
 * from your python app code.
 *
 * @module
 *
 * @groupDescription Behavior Registration
 * From python, you can add any number of bundled or custom behaviors to
 * the Kaolin viewer simply by their name. To enable this,
 * {@link BehaviorRegister} is a client-side singleton that resolves behavior
 * implementations by name when your webpage loads. A behavior may be a
 * {@link Behavior} subclass, any class honoring {@link BehaviorInterface}, or a
 * React component — plain JavaScript is fine, no TypeScript or build step
 * required.
 *
 * ```javascript
 * // In an app's custom.js (loaded in the browser), kaolin is loaded automatically
 * const { Behavior, BehaviorRegister } = kaolin.core.behavior;  
 *
 * class HelloBehavior extends Behavior {
 *     // Your custom javascript, using any libraries
 * }
 *
 * // 'hello' is the name Python passes to add_behavior(...).
 * BehaviorRegister.register('hello', HelloBehavior, 'My first behavior');
 * ```
 *
 * @groupDescription Key Behavior Interfaces
 * In order to work with the viewer, custom behaviors must implement any
 * one of the following interfaces. The viewer duck-types each
 * registered behavior against these interfaces and, using the matching `is*`
 * guard, routes only the relevant events to it; implement only what you need.
 * Most have a ready-made base class (see Behavior Base Classes), so you rarely
 * implement them by hand.
 *
 * {@link BehaviorInterface} — option management, required of every behavior:
 * ```typescript
 * options: any;
 * setOption(name: string, value: any): void;
 * reset(options?: any): void;
 * ```
 * {@link ElementBoundBehavior} — bind to a DOM element:
 * ```typescript
 * elementType(): string;
 * setActiveElement(element: HTMLElement): void;
 * ```
 * {@link ClickEventHandler} / {@link PointerEventHandler} — receive click / pointer events:
 * ```typescript
 * onClick?(event); onDoubleClick?(event);
 * onPointerDown?(event); onPointerMove?(event); onPointerUp?(event);
 * onPointerCancel?(event); onPointerEnter?(event); onPointerLeave?(event);
 * ```
 * {@link MessageHandler} — receive tagged messages from the connection:
 * ```typescript
 * acceptedMessageTags(): string[];
 * onConnectionOpen(): void;
 * onMessage(messageTag: string, messageContent: any | null): void;
 * ```
 * {@link CameraControllerInterface} — own and drive a camera:
 * ```typescript
 * setDimensions(width: number, height: number): void;
 * setCameraParams(params: CameraParameters): void;
 * getCamera(): any;
 * getCameraParams(): CameraParameters;
 * ```
 *
 * @groupDescription Behavior Base Classes
 * Convenient bases that already implement the contracts above — extend one of
 * these instead of implementing interfaces by hand:
 * - {@link Behavior} — option management only; non-interactive behaviors.
 * - {@link InteractiveBehavior} — adds element binding + pointer/click handlers.
 * - {@link CanvasBehavior} — an {@link InteractiveBehavior} bound to a canvas, exposing its 2D context.
 * - {@link CameraControllerBase} — base for camera controllers.
 * - {@link MessageHandlerBase} — base for tagged-message handlers.
 *
 * Extend one in plain JavaScript (e.g. an app's custom.js), then register it
 * (see Behavior Registration):
 * ```javascript
 * const { CanvasBehavior, BehaviorRegister, OptionKind } = kaolin.core.behavior;
 *
 * class DotBehavior extends CanvasBehavior {
 *     static schema = { radius: { kind: OptionKind.INT, default: 6, min: 1, max: 40 } };
 *
 *     // Re-render whenever an option is edited from Python.
 *     updateForOptions() { this.redraw(); }
 *
 *     // CanvasBehavior provides `this.element` / `this.ctx` and the pointer handlers.
 *     onPointerDown(event) { this.x = event.offsetX; this.y = event.offsetY; this.redraw(); }
 *
 *     redraw() {
 *         if (!this.ctx || this.x === undefined) return;
 *         this.ctx.beginPath();
 *         this.ctx.arc(this.x, this.y, this.options.radius, 0, 2 * Math.PI);
 *         this.ctx.fill();
 *     }
 * }
 *
 * BehaviorRegister.register('dot', DotBehavior, 'Draws a dot where you click.');
 * ```
 *
 * @groupDescription Internal Utilities
 * Implementation details the viewer relies on internally; not intended for use
 * by behavior authors.
 */

export * from './base';
export * from './camera_controller';
export * from './message_handler';
export * from './option';
export * from './register';
