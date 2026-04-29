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
 * Runtime type utilities and type-checking helpers for Kaolin viewer data transformations.
 *
 * @module
 */

/**
 * Recursively convert all plain objects (`{}`) in a value to `Map`s.
 * Arrays, typed arrays, Maps, and primitives are passed through unchanged;
 * only plain-object nodes (`constructor === Object`) are converted.
 *
 * @param value - The value to convert.
 * @returns The value with all plain objects replaced by Maps.
 */
export function convertAllObjectsToMaps(value: any): any {
    // Handle null and undefined
    if (value === null || value === undefined) {
        return value;
    }

    // Handle Maps - recursively convert values
    if (value instanceof Map) {
        const res = new Map();
        for (const [name, val] of (value as Map<any, any>)) {
            res.set(name, convertAllObjectsToMaps(val));
        }
        return res;
    }

    // Handle Arrays - recursively convert elements
    if (Array.isArray(value)) {
        return value.map(item => convertAllObjectsToMaps(item));
    }

    // Handle primitives - return as-is
    if (typeof value !== 'object') {
        return value;
    }

    // Handle typed arrays and other built-in objects - return as-is
    if (value.constructor !== Object) {
        return value;
    }

    // Handle plain objects - convert to Maps recursively
    const res = new Map();
    for (const [key, val] of Object.entries(value)) {
        res.set(key, convertAllObjectsToMaps(val));
    }
    return res;
}

/**
 * Recursively convert all `Map`s in a value to plain objects.
 * Arrays, typed arrays, plain objects, and primitives are passed through unchanged.
 *
 * @param value - The value to convert.
 * @returns The value with all Maps replaced by plain objects.
 */
export function convertAllMapsToObjects(value: any): any {
    // Handle null and undefined
    if (value === null || value === undefined) {
        return value;
    }

    // Handle Maps - recursively convert values
    if (value instanceof Map) {
        const res = {};
        for (const [name, val] of (value as Map<any, any>)) {
            res[name] = convertAllMapsToObjects(val);
        }
        return res;
    }

    // Handle Arrays - recursively convert elements
    if (Array.isArray(value)) {
        return value.map(item => convertAllMapsToObjects(item));
    }

    // Handle primitives - return as-is
    if (typeof value !== 'object') {
        return value;
    }

    // Handle typed arrays and other built-in objects - return as-is
    if (value.constructor !== Object) {
        return value;
    }

    // Handle plain objects - recursively convert their values
    const res = {};
    for (const [key, val] of Object.entries(value)) {
        res[key] = convertAllMapsToObjects(val);
    }
    return res;
}

/**
 * Resolve a dot-separated path string to a function on the global `window` object.
 * Logs an error and returns `undefined` if any path segment is missing or the
 * resolved value is not a function.
 *
 * @param functionPathString - Dot-separated path, e.g. `'kaolin.util.myFn'`.
 * @returns The resolved function, or `undefined` if not found.
 */
export function getFunctionByName(functionPathString: string) {
    const parts = functionPathString.split('.');
    let currentObject : any = window; // Or global in Node.js

    for (let i = 0; i < parts.length; i++) {
        if (currentObject && typeof currentObject === 'object' && parts[i] in currentObject) {
            currentObject = currentObject[parts[i]];
        } else {
            // Handle cases where the path is invalid or a part doesn't exist
            console.error(`Invalid path: ${functionPathString}. '${parts[i]}' not found.`);
            return undefined;
        }
    }
    return typeof currentObject === 'function' ? currentObject : undefined;
}

/**
 * Resolve a function by dot-separated path, returning a fallback when the path
 * is absent. Throws if a path is provided but cannot be resolved.
 *
 * @param functionPathString - Dot-separated path to look up; if falsy, `fallbackFunction` is returned.
 * @param fallbackFunction - Returned when `functionPathString` is falsy.
 * @returns The resolved function or the fallback.
 */
export function getFunctionByNameOrThrow(functionPathString: string | null | undefined, fallbackFunction?: Function) {
    let result = fallbackFunction;
    if (functionPathString) {  // If name is provided, it must be resolved
        result = getFunctionByName(functionPathString);
        if (!result) {
            throw new Error(`Could not find process function by name: ${functionPathString}`);
        }
    }
    return result;
}

// Define a constructor type for TS safety
/** Generic constructor type. */
type Constructor<T = any> = new (...args: any[]) => T;

/**
 * Return true if `variable` is a class declaration (not a plain function or arrow).
 *
 * @param variable - Value to test.
 * @returns True if `variable` is a class constructor.
 */
export function isClass(variable: any): variable is Constructor {
    // 1. It must be a function
    if (typeof variable !== 'function') {
        return false;
    }

    // 2. Convert to string and check for the "class" keyword
    // This handles "class MyClass {}" and "class {}" / "class{}" (anonymous),
    // where the keyword may be followed by whitespace or directly by "{".
    const str = variable.toString();

    // Word boundary after "class" matches whether the next char is whitespace,
    // "{", or any other non-word char; leading whitespace is tolerated.
    if (/^\s*class\b/.test(str)) {
        return true;
    }

    return false;
}


/**
 * Allows declaring functions that check if a value implements a certain interface,
 * such that any inconsistencies are detected at **compile time**.
 *
 * Basic issue is that typescript interfaces are compiled out when converting to
 * javascript. If we need a **run-time** check that a certain object conforms to
 * an interface, we need to check the presence of all the expected method names.
 *
 * Implementing such checks can be extremely brittle. This method allows declaring
 * implementsMyInterface function that will be checked at compile time against
 * the target interface.
 *
 * Example:
 * export interface MyInterface {
 *    someMethod(): string[];
 *    someOtherMethod(): void;
 *    someNotEssentialMethod(): void;
 * }
 *
 * // Still have to list all required methods, but any typos will fail at compile time.
 * export const implementsMyInterface = makeImplementsInterfaceFunction<MyInterface>({
 *      someMethod: true,
 *      someOtherMethods: true,  // <-- typo; will cause *compile-time* error
 *      someNotEssentialMethod: false
 * });
 *
 * @param requiredKeys
 * @returns
 */
export function makeImplementsInterfaceFunction<I>(requiredKeys: Record<keyof I, true> ): (object: any) => object is I {
    return function(object: any): object is I {
        if (!object || typeof object !== 'object') { return false; }

        const keysToCheck = (Object.keys(requiredKeys) as (keyof I)[]).filter((key) => requiredKeys[key] === true);

        // Iterate through them
        for (const key of keysToCheck) {
            if (!(key in object)) {
                return false;
            }
        }

        return true;
    };
}

/**
 * Convert various matrix formats to a flat 16-element array (row-major).
 *
 * Accepts nested arrays, a flat `Float32Array` / `number[]`, and an array of
 * typed-array rows (e.g. the binary wire format in `util/io` decodes a 4x4
 * view matrix into a list of per-row `Float32Array`s).
 */
export function flattenMatrix(
    matrix: number[][] | number[][][] | Float32Array | Float32Array[] | number[]): number[] {
    // Convert different matrix formats to flat array (row-major)
    if (matrix instanceof Float32Array) {
        return Array.from(matrix);
    }
    if (Array.isArray(matrix) && Array.isArray(matrix[0]) && Array.isArray(matrix[0][0])) {
        return (matrix as number[][][]).flat().reduce((accumulator, value) => accumulator.concat(value), []);
    }
    if (Array.isArray(matrix) && Array.isArray(matrix[0])) {
        // 3D array - flatten row by row
        return (matrix as number[][]).flat();
    }
    if (Array.isArray(matrix) && matrix.length > 0 && ArrayBuffer.isView(matrix[0])) {
        // Array of typed-array rows (respect the typed arrays) - flatten row-major.
        return (matrix as unknown as ArrayLike<number>[]).reduce<number[]>(
            (accumulator, row) => accumulator.concat(Array.from(row)), []);
    }
    if (Array.isArray(matrix) && matrix.length === 16) {
        return matrix as number[];
    }
    throw new Error('Unsupported matrix format. Expected a 4x4 matrix as a 2D array, '
        + 'Float32Array, array of typed-array rows, or a flat array of 16 elements.');
}

/** Utility for empty message passing. */
export function toEmptyString(val: any) {
    return '';
}