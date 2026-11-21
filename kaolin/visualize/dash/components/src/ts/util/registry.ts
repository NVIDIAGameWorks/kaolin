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
 * Generic key–value registries with duplicate detection, and a factory for
 * per-class static registries.
 *
 * @module
 */

import { logger } from './logging';


/**
 * Key–value store for named items with configurable duplicate handling.
 * Useful for dynamic registries where items are registered by string name at runtime.
 */
export class Registry<T> {
    // Each class created by this function gets its OWN private static map
    private _store = new Map<string, T>();

    /** Register a new item */
    register(name: string, item: T, overwrite: boolean = false): void {
        // Check for duplicates if necessary
        if (this._store.has(name)) {
            let message = `Registry: duplicate registration for ${name} -->`;
            if (!overwrite) {
                logger.warn(`${message} skipping`);
                return;
            } else {
                logger.warn(`${message} overwriting`);
            }
        }
        this._store.set(name, item);
    }

    /** Get an item */
    get(name: string): T | undefined {
        return this._store.get(name);
    }

    /** Check if registered */
    has(name: string): boolean {
        return this._store.has(name);
    }

    /** Get all names */
    getRegistered(): string[] {
        return Array.from(this._store.keys());
    }

    /** Remove the item registered under `name`. No-op if the name is not registered. */
    deregister(name: string) {
        this._store.delete(name);
    }
};

/**
 * Create a new class whose static methods (`register`, `get`, `has`,
 * `getRegistered`) wrap a private {@link Registry} instance.
 * Each call returns an independent class with its own isolated store.
 *
 * @returns A new class with a static registry API.
 */
export function createGlobalRegistry<T>() {
    return class GlobalRegistry {
        // Each class created by this function gets its OWN private static map
        static _registry = new Registry<T>();

        /** Register a new item */
        static register(name: string, item: T): void {
            this._registry.register(name, item);
        }

        /** Get an item */
        static get(name: string): T | undefined {
            return this._registry.get(name);
        }

        /** Check if registered */
        static has(name: string): boolean {
            return this._registry.has(name);
        }

        /** Get all names */
        static getRegistered(): string[] {
            return this._registry.getRegistered();
        }
    };
}