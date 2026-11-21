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
 * Registry and runner for KaolinViewer behaviors: the global {@link BehaviorRegister}
 * apps use to register behavior classes or React components by name (with their
 * option schema and description), and the {@link BehaviorRunner} component the
 * viewer renders to instantiate a registered behavior and hand it back through a
 * callback.
 *
 * @module
 */

import React, { useCallback, useEffect, memo } from 'react';

import { logger } from '../../util/logging';
import { createGlobalRegistry, Registry } from '../../util/registry';
import { isClass } from '../../util/types';
import { Behavior, BehaviorInterface, isElementBoundBehavior } from './base';
import { BehaviorMeta, BehaviorMetaData, normalizeSchema } from './option';


/** Constructor for a class that produces a {@link BehaviorInterface} instance. */
type BehaviorConstructor = new (...args: any[]) => BehaviorInterface;


/**
 * Decide whether a registered item is a behavior class (vs. a React component).
 *
 * Accepts:
 *   1. The {@link Behavior} base class and any subclass (fast path — the
 *      typical in-tree library case).
 *   2. Any user-defined `class` declaration, even if it does not inherit
 *      from `Behavior`. Such classes are trusted to honor the
 *      {@link BehaviorInterface} contract; missing members surface at
 *      use-time when the viewer invokes `setOption` / etc.
 *
 * Rejects React function components and `forwardRef` components: the former
 * have no `class ` prefix in their source (`isClass` returns false), the
 * latter return objects (`typeof !== 'function'`).
 */
function isBehaviorConstructor(value: any): value is BehaviorConstructor {
  if (typeof value !== 'function' || !value.prototype) return false;
  if (value === Behavior || value.prototype instanceof Behavior) return true;
  return isClass(value);
}

/**
 * Global registry of behaviors compatible with KaolinViewer.
 *
 * Behaviors are registered by name as either a {@link Behavior} class (or any
 * class honoring {@link BehaviorInterface}) or a React component, and looked up
 * by name when the viewer instantiates them. Each registration also records
 * {@link BehaviorMeta} (description + option schema) for the schema-driven UI.
 *
 * @group Behavior Registration
 */
export class BehaviorRegister extends createGlobalRegistry<any>() {
  // Note this._registry corresponds to is_class
  private static _classRegistry = new Registry<BehaviorConstructor>();
  private static _componentRegistry = new Registry<React.Component>(); // TODO: works for function components?
  private static _meta = new Map<string, BehaviorMeta>();

  /**
   * Register a behavior under `name`, classifying it as a class or a React
   * component and recording its metadata (description + option schema).
   *
   * @param name - Unique registration name (also the snake_case behavior name).
   * @param item - Behavior class or React component to register.
   * @param description - Optional override description; falls back to `item.description`.
   */
  static override register(name: string, item: any, description?: string): void {
    const isClass = isBehaviorConstructor(item);

    if (this.has(name) && this.get(name) != isClass) {
      throw new Error(`Attempting to register both class and React component Behaviors with name "${name}"`);
    }

    super.register(name, isClass);
    const msgPrefix = `✅ GlobalRegister: adding "${name}" behavior`;
    if (isClass) {
      this._classRegistry.register(name, item as BehaviorConstructor);
      logger.info(`${msgPrefix} as class`);
    } else {
      this._componentRegistry.register(name, item as React.Component); // TODO: works for function components?
      logger.info(`${msgPrefix} as React component`);
    }

    // Register metadata
    const schema = normalizeSchema((item as any).schema);
    if (!schema) {
      logger.warn(`BehaviorRegister: behavior "${name}" has no valid schema`);
    }
    const behaviorDescription = description ?? (item as any).description ?? '';

    // Now we make BehaviorMeta and we register it in _meta
    this._meta.set(name, new BehaviorMeta(
      behaviorDescription,
      schema ?? {},
    ));
  }

  /**
   * Look up a registered behavior by name, whether it was registered as a class
   * or a React component.
   *
   * @param name - Registration name.
   * @returns The registered class or component, or `undefined` if not found.
   */
  static override get(name: string): any | undefined {
    if (this._classRegistry.has(name)) {
      return this._classRegistry.get(name);
    } else {
      return this._componentRegistry.get(name);
    }
  }

  /**
   * @param name - Registration name.
   * @returns `true` if registered as a class, `false` if as a component, `undefined` if unknown.
   */
  static isClass(name: string): boolean | undefined {
    return super.get(name);
  }

  /**
   * @param name - Registration name.
   * @returns `true` if the name is registered as a React component.
   */
  static isReactComponent(name: string) {
    return !super.get(name);  // we save bools in the base registry
  }

  /**
   * @returns Names of all behaviors registered as classes.
   */
  static getRegisteredClasses(): string[] {
    return this._classRegistry.getRegistered();
  }

  /**
   * @returns Names of all behaviors registered as React components.
   */
  static getRegisteredReactComponents(): string[] {
    return this._componentRegistry.getRegistered();
  }

  /**
   * @param name - Registration name.
   * @returns The registered behavior constructor, or `undefined` if not a class.
   */
  static getClass(name: string): BehaviorConstructor | undefined {
    return this._classRegistry.get(name);
  }

  /**
   * @param name - Registration name.
   * @returns The registered React component, or `undefined` if not a component.
   */
  static getReactComponent(name: string): React.Component | undefined { // TODO: works for function components
    return this._componentRegistry.get(name);
  }

  /**
   * @param name - Registration name.
   * @returns The behavior's recorded {@link BehaviorMeta}, or `undefined` if unknown.
   */
  static getMeta(name: string): BehaviorMeta | undefined {
    return this._meta.get(name);
  }

  /**
   * Project every registered behavior into the plain JSON manifest consumed by
   * the Python side: a `{ name: { description, options } }` record. This is the
   * single source of truth for the manifest shape — the dump script merely
   * serializes the result to disk. Behaviors whose metadata is missing are
   * skipped with a warning.
   *
   * @returns The behavior manifest as plain, serializable data.
   */
  static toJSON(): Record<string, BehaviorMetaData> {
    const manifest: Record<string, BehaviorMetaData> = {};
    for (const name of this.getRegistered()) {
      const meta = this.getMeta(name);
      if (!meta) {
        logger.warn(`BehaviorRegister: behavior "${name}" not found in manifest`);
        continue;
      }
      manifest[name] = meta.toJSON();
    }
    return manifest;
  }
};

/** Props for the {@link BehaviorRunner} component. */
interface BehaviorRunnerProps {
  behaviorName: string;
  behaviorOptions: any;
  behaviorIdentifier: string,
  onRegister: (identifier: string, behavior: BehaviorInterface | null) => void;
}

/** Instantiates a registered behavior and reports it via `onRegister`. */
const BehaviorRunnerComponent = ({
  behaviorName,
  behaviorOptions,
  behaviorIdentifier,
  onRegister
}: BehaviorRunnerProps) => {
  // Check if behavior is registered
  if (!BehaviorRegister.has(behaviorName)) {
    logger.error(`Behavior ${behaviorName} not found in BehaviorRegister. Available behaviors: ${BehaviorRegister.getRegistered()}`);
    return null;
  }

  // Figure out if it's a React component or a class
  const isClass = BehaviorRegister.isClass(behaviorName);

  // Set up registration through ref (for components only)
  const registerBehavior = useCallback((instance: any) => {
    logger.warn(`----> 🍓 Registering behavior with name ${behaviorIdentifier}`);
    console.log(instance);
    if (instance) {
      onRegister(behaviorIdentifier, instance);

      // Return a cleanup function
      return () => {
        // Unregister (pass null or undefined for cleanup) when the component unmounts
        onRegister(behaviorIdentifier, null);
      };
    } else {
      onRegister(behaviorIdentifier, null);  // TODO: needed?
    }
  }, []);

  const constructorOrFunction = BehaviorRegister.get(behaviorName);
  if (!constructorOrFunction) {
    logger.error(`Could not find registered behavior ${behaviorName}, although its type is registered --> this is a bug.`);
    return null;
  }

  let behavior = null;
  if (isClass === true) {
    behavior = new (constructorOrFunction as BehaviorConstructor)(behaviorOptions);
  }

  useEffect(() => {
    // Only need to do this for classes, React handles this though ref: registerBehavior
    if (isClass === true) {
      return registerBehavior(behavior);
    }
  }, [behaviorName, behaviorIdentifier]);

  if (isClass === true) {
    return null;
  } else {
    return React.createElement(constructorOrFunction,
      { ref: registerBehavior, ...behaviorOptions })
  }
};

/**
 * React component that instantiates the behavior registered under
 * `behaviorName` and hands the instance back to the viewer via `onRegister`.
 *
 * For class behaviors it constructs the instance directly; for React-component
 * behaviors it renders the component and captures the instance through a ref.
 * Memoized on `behaviorName` / `behaviorIdentifier`.
 *
 * @group Internal Utilities
 */
export const BehaviorRunner = memo(BehaviorRunnerComponent, (oldProps, newProps) => {
  return oldProps.behaviorIdentifier == newProps.behaviorIdentifier &&
    oldProps.behaviorName == newProps.behaviorName
}) as typeof BehaviorRunnerComponent;