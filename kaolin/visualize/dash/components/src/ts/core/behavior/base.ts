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
 * Base classes and interfaces for KaolinViewer behaviors: the option-management
 * contract every behavior satisfies, the interactive/canvas base classes that
 * bind to a DOM element and receive pointer/click events, and the runtime
 * type-guards used to detect which capabilities a behavior implements.
 *
 * @module
 */

import { z } from 'zod';

import { logger } from '../../util/logging';
import { makeImplementsInterfaceFunction } from '../../util/types';
import { OptionSpecData } from './option';


/**
 * Minimal contract required of any behavior configured on
 * the Kaolin viewer. It captures only the members the viewer drives directly:
 * the live `options` bag, the `setOption(name, value)` edit-dispatch entry point,
 * and `reset`. Subclassing {@link Behavior} is the typical way to satisfy this;
 * user-defined classes that implement these members directly are also accepted
 * by {@link BehaviorRegister}.
 * 
 * @group Key Behavior Interfaces
 */
export interface BehaviorInterface {
    options: any;
    setOption(name: string, value: any): void;
    reset(options?: any): void;
}

/**
 * General base class for behaviors configured for KaolinViewer.
 *
 * Provides a default `setOption` / `updateForOptions` / `reset` implementation
 * suitable for most subclasses. Subclasses that need element binding or
 * pointer/click handling should extend {@link InteractiveBehavior} instead.
 *
 * @group Behavior Base Classes
 */
export class Behavior implements BehaviorInterface {
    /**
     * Subclass option dictionary (`{ optionName: value }`). Typed as `any` here
     * so the base class can read/write generically; subclasses narrow it.
     */
    options: any;
    /**
     * Optional schema describing this behavior's option surface. Declared on
     * the subclass as `static schema = {...} as const`. When present, it drives
     * the build-time manifest dumper that surfaces the option set (with defaults
     * and types) to Python for automatic UI generation. A `z.ZodObject` schema
     * is additionally consulted by `setOption` to warn about invalid writes.
     */
    static schema: Record<string, OptionSpecData> | z.ZodObject<any> | undefined;

    /** Optional human-readable description, surfaced in generated docs/UI. */
    static description: string | undefined;

    /**
     * Writes a single option value and notifies the subclass of the change.
     *
     * The value is stored on `this.options`, then `updateForOptions(name, value)`
     * is called so the subclass can react. `this.options` is lazily initialized
     * to an empty object on first write.
     *
     * When a `z.ZodObject` `static schema` is declared and the key is part of
     * it, the value is checked against that field's schema. Validation is
     * currently soft: an invalid value logs a warning but is still written
     * through (this may be tightened in a future release). Keys absent from the
     * schema, and behaviors without a Zod schema, are written through unchanged.
     *
     * @param name - Option key to write.
     * @param value - Value to store under `name`.
     */
    setOption(name: string, value: any): void {
        if (this.options === undefined || this.options === null) {
            this.options = {};
        }
        const schema = (this.constructor as typeof Behavior).schema;
        if (schema instanceof z.ZodObject) {
            const field = schema.shape[name];
            if (field) {
                const result = field.safeParse(value);
                if (!result.success) {
                    logger.warn(`setOption: invalid value for "${name}" (setting anyway): ${result.error.message}`);
                }
            }
        }
        this.options[name] = value;
        this.updateForOptions(name, value);
    }

    /**
     * Hook invoked after every `setOption` write, receiving the option `name`
     * that changed and its new `value`. Default is a no-op; override in
     * subclasses that need to react to option changes (re-render, recompute
     * derived state, etc.). The arguments are advisory — many subclasses simply
     * re-read `this.options` and ignore them.
     */
    updateForOptions(_name?: string, _value?: any): void {}

    /**
     * Resets internal behavior state. Default is a no-op for subclasses with no
     * transient state to clear.
     *
     * @param options - Optional options snapshot to reset against.
     */
    reset(options?: any): void {}
}

/**
 * Behaviors that bind to a viewer *layer* (configurable from python) and will
 * automatically receive a reference to the live element once it is mounted.
 *
 * @group Key Behavior Interfaces
 */
export interface ElementBoundBehavior {
    elementType: () => string;
    setActiveElement: (element: HTMLElement) => void;
};

/**
 * Runtime type-guard that reports whether a value implements
 * {@link ElementBoundBehavior}. Fails at compile time if the interface changes.
 *
 * @param object - Candidate value to test.
 * @returns `true` if `object` implements every {@link ElementBoundBehavior} member.
 *
 * @group Key Behavior Interfaces
 */
export const isElementBoundBehavior =
    makeImplementsInterfaceFunction<ElementBoundBehavior>({
        elementType: true,
        setActiveElement: true
    });

/**
 * Optional signal returned by an event handler to request follow-up handling,
 * e.g. that the viewer keep delivering pointer-move events.
 */
export interface HandlerReturnInfo {
    needsPointermove?: boolean;
}

/**
 * Click-style event handlers a behavior may implement. The viewer will 
 * check if the behavior implements this interface, before routing these events.
 *
 * @group Key Behavior Interfaces
 */
export interface ClickEventHandler {
    onClick?: (event: React.MouseEvent) => void | HandlerReturnInfo;
    onDoubleClick?: (event: React.MouseEvent) => void | HandlerReturnInfo;
}

/**
 * Runtime type-guard that reports whether a value implements
 * {@link ClickEventHandler}. Fails at compile time if the interface changes.
 *
 * @param object - Candidate value to test.
 * @returns `true` if `object` implements every {@link ClickEventHandler} member.
 *
 * @group Key Behavior Interfaces
 */
export const isClickEventHandler =
    makeImplementsInterfaceFunction<ClickEventHandler>({
        onClick: true,
        onDoubleClick: true,
    });

/**
 * Pointer-style event handlers a behavior may implement. The viewer routes
 * pointer events (down/move/up/cancel/enter/leave) to behaviors that implement
 * this interface.
 *
 * @group Key Behavior Interfaces
 */
export interface PointerEventHandler {
    onPointerDown?: (event: React.PointerEvent) => void | HandlerReturnInfo;
    onPointerMove?: (event: React.PointerEvent) => void | HandlerReturnInfo;
    onPointerUp?: (event: React.PointerEvent) => void | HandlerReturnInfo;
    onPointerCancel?: (event: React.PointerEvent) => void | HandlerReturnInfo;
    onPointerEnter?: (event: React.PointerEvent) => void | HandlerReturnInfo;
    onPointerLeave?: (event: React.PointerEvent) => void | HandlerReturnInfo;
}

/**
 * Runtime type-guard that reports whether a value implements
 * {@link PointerEventHandler}. Fails at compile time if the interface changes.
 *
 * @param object - Candidate value to test.
 * @returns `true` if `object` implements every {@link PointerEventHandler} member.
 *
 * @group Key Behavior Interfaces
 */
export const isPointerEventHandler =
    makeImplementsInterfaceFunction<PointerEventHandler>({
        onPointerDown: true,
        onPointerMove: true,
        onPointerUp: true,
        onPointerCancel: true,
        onPointerEnter: true,
        onPointerLeave: true,
    });


/**
 * Base class for behaviors that bind to a DOM element and handle pointer/click
 * events. Tracks the active element and provides no-op handler defaults that
 * subclasses override; defaults to binding a `div`.
 *
 * @group Behavior Base Classes
 */
export class InteractiveBehavior extends Behavior implements PointerEventHandler, ClickEventHandler, ElementBoundBehavior, BehaviorInterface {
    element: HTMLElement | null;

    constructor() {
        super();
        this.element = null;
        this.options = {};
    }

    elementType() {
        return "div";
    }

    setActiveElement(element: HTMLElement) {
        this.element = element;
    }

    onClick(event: React.MouseEvent): void | HandlerReturnInfo {}
    onDoubleClick(event: React.MouseEvent): void | HandlerReturnInfo {}
    onPointerDown(event: React.PointerEvent): void | HandlerReturnInfo {}
    onPointerMove(event: React.PointerEvent): void | HandlerReturnInfo {}
    onPointerUp(event: React.PointerEvent): void | HandlerReturnInfo {}
    onPointerCancel(event: React.PointerEvent): void | HandlerReturnInfo {}
    onPointerEnter(event: React.PointerEvent): void | HandlerReturnInfo {}
    onPointerLeave(event: React.PointerEvent): void | HandlerReturnInfo {}
}

/**
 * Interactive behavior that binds to a `canvas` element and exposes its 2D
 * rendering context (`ctx`), refreshing the context whenever the active element
 * changes.
 *
 * @group Behavior Base Classes
 */
export class CanvasBehavior extends InteractiveBehavior {
    ctx: CanvasRenderingContext2D | null;

    constructor() {
       super();
       this.ctx = null;
    }

    override elementType() {
        return "canvas";
    }

    override setActiveElement(canvas: HTMLElement) {
        super.setActiveElement(canvas);
        if (this.element) {
            this.ctx = (this.element as HTMLCanvasElement)?.getContext('2d');
        } else {
            this.ctx = null;
        }
    }
}
