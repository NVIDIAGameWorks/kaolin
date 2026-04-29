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

import { assert } from 'chai';
import { convertAllObjectsToMaps, convertAllMapsToObjects, getFunctionByName, getFunctionByNameOrThrow, 
    isClass, flattenMatrix, makeImplementsInterfaceFunction, toEmptyString } from '@kaolin/util/types';
import { captureConsole } from '@test/helpers/console';
import { registerDom, unregisterDom } from '@test/helpers/dom';

describe('visualize/dash/components/src/util/test_types.ts', () => {
    describe('TestConvertAllObjectsToMaps', () => {
        it('should convert plain objects to Maps', () => {
            const input = { name: 'test', value: 42 };
            const result = convertAllObjectsToMaps(input);
        
            assert.instanceOf(result, Map, 'Result should be a Map');
            assert.equal(result.get('name'), 'test', 'Name should be preserved');
            assert.equal(result.get('value'), 42, 'Value should be preserved');
        });

        it('should handle nested objects recursively', () => {
            const input = {
                outer: {
                    inner: { nested: 'value' },
                    number: 123
                }
            };
            let result = convertAllObjectsToMaps(input);
        
            assert.instanceOf(result, Map, 'Root should be a Map');
            result = result as Map<string, any>;
            assert.instanceOf(result.get('outer'), Map, 'Outer should be a Map');
            assert.instanceOf(result.get('outer').get('inner'), Map, 'Inner should be a Map');
            assert.equal(result.get('outer').get('inner').get('nested'), 'value', 'Nested value should be preserved');
            assert.equal(result.get('outer').get('number'), 123, 'Number should be preserved');
        });

        it('should handle arrays with nested objects', () => {
            const input = {
                items: [1, { key: 'value' }, 'string', { another: 'object' }]
            };
            const result = convertAllObjectsToMaps(input);
        
            assert.instanceOf(result, Map, 'Root should be a Map');
            const items = result.get('items');
            assert.isArray(items, 'Items should remain an array');
            assert.equal(items[0], 1, 'First item should be unchanged');
            assert.instanceOf(items[1], Map, 'Second item should be a Map');
            assert.equal(items[1].get('key'), 'value', 'Object in array should be converted');
            assert.equal(items[2], 'string', 'String should be unchanged');
            assert.instanceOf(items[3], Map, 'Fourth item should be a Map');
            assert.equal(items[3].get('another'), 'object', 'Another object in array should be converted');
        });

        it('should preserve existing Maps and convert their values', () => {
            const innerMap = new Map([['existing', { nested: 'value' }]]);
            const input = { outerKey: innerMap };
            const result = convertAllObjectsToMaps(input);
        
            assert.instanceOf(result, Map, 'Root should be a Map');
            const outerMap = result.get('outerKey');
            assert.instanceOf(outerMap, Map, 'Inner map should remain a Map');
            assert.instanceOf(outerMap.get('existing'), Map, 'Value in existing map should be converted to Map');
            assert.equal((outerMap.get('existing') as Map<string, any>).get('nested'), 'value', 'Nested value should be preserved');
        });

        it('should preserve primitives unchanged', () => {
            const input = {
                string: 'hello',
                number: 42,
                boolean: true,
                nullValue: null,
                undefinedValue: undefined
            };
            const result = convertAllObjectsToMaps(input);
        
            assert.instanceOf(result, Map, 'Root should be a Map');
            assert.equal(result.get('string'), 'hello', 'String should be unchanged');
            assert.equal(result.get('number'), 42, 'Number should be unchanged');
            assert.equal(result.get('boolean'), true, 'Boolean should be unchanged');
            assert.isNull(result.get('nullValue'), 'Null should be unchanged');
            assert.isUndefined(result.get('undefinedValue'), 'Undefined should be unchanged');
        });

        it('should preserve typed arrays and built-in objects', () => {
            const float32Array = new Float32Array([1, 2, 3]);
            const uint8Array = new Uint8Array([4, 5, 6]);
            const date = new Date('2025-01-01');
            const input = {
                float32: float32Array,
                uint8: uint8Array,
                date: date
            };
            const result = convertAllObjectsToMaps(input);
        
            assert.instanceOf(result, Map, 'Root should be a Map');
            assert.instanceOf(result.get('float32'), Float32Array, 'Float32Array should be preserved');
            assert.instanceOf(result.get('uint8'), Uint8Array, 'Uint8Array should be preserved');
            assert.instanceOf(result.get('date'), Date, 'Date should be preserved');
            assert.deepEqual(Array.from(result.get('float32')), [1, 2, 3], 'Float32Array values should be preserved');
            assert.deepEqual(Array.from(result.get('uint8')), [4, 5, 6], 'Uint8Array values should be preserved');
            assert.equal(result.get('date').getTime(), date.getTime(), 'Date value should be preserved');
        });

        it('should handle null and undefined inputs', () => {
            assert.isNull(convertAllObjectsToMaps(null), 'Null input should return null');
            assert.isUndefined(convertAllObjectsToMaps(undefined), 'Undefined input should return undefined');
        });

        it('should handle primitive inputs', () => {
            assert.equal(convertAllObjectsToMaps('string'), 'string', 'String input should return string');
            assert.equal(convertAllObjectsToMaps(42), 42, 'Number input should return number');
            assert.equal(convertAllObjectsToMaps(true), true, 'Boolean input should return boolean');
        });

        it('should handle complex nested structure', () => {
            const input = {
                metadata: {
                    name: 'complex_test',
                    version: 1.0
                },
                data: [
                    { type: 'mesh', vertices: new Float32Array([1, 2, 3]) },
                    { type: 'material', properties: { color: 'red', metalness: 0.5 } }
                ],
                config: new Map([['setting1', { enabled: true }]])
            };
            const result = convertAllObjectsToMaps(input);
        
            // Check root structure
            assert.instanceOf(result, Map, 'Root should be a Map');
            assert.instanceOf(result.get('metadata'), Map, 'Metadata should be a Map');
            assert.isArray(result.get('data'), 'Data should remain an array');
            assert.instanceOf(result.get('config'), Map, 'Config should remain a Map');
        
            // Check metadata
            const metadata = result.get('metadata');
            assert.equal(metadata.get('name'), 'complex_test', 'Metadata name should be preserved');
            assert.equal(metadata.get('version'), 1.0, 'Metadata version should be preserved');
        
            // Check data array
            const data = result.get('data');
            assert.instanceOf(data[0], Map, 'First data item should be a Map');
            assert.instanceOf(data[0].get('vertices'), Float32Array, 'Vertices should remain Float32Array');
            assert.instanceOf(data[1], Map, 'Second data item should be a Map');
            assert.instanceOf(data[1].get('properties'), Map, 'Properties should be a Map');
        
            // Check config
            const config = result.get('config');
            assert.instanceOf(config.get('setting1'), Map, 'Config value should be converted to Map');
            assert.equal(config.get('setting1').get('enabled'), true, 'Config nested value should be preserved');
        });
    });

    describe('getFunctionByName', () => {
        // getFunctionByName resolves paths against `window`; install a real DOM and
        // hang a known nested function (and a non-function) off it for resolution.
        const myFn = () => 42;
        before(registerDom);
        after(unregisterDom);
        beforeEach(() => {
            (window as any).a = { b: { myFn } };
            (window as any).notAFn = 7;
        });
        afterEach(() => { delete (window as any).a; delete (window as any).notAFn; });

        it('resolves a dotted path to the function', () => {
            assert.equal(getFunctionByName('a.b.myFn'), myFn, 'nested path resolves to the function');
        });

        it('returns undefined for an unknown path', () => {
            const { restore } = captureConsole();
            try {
                assert.isUndefined(getFunctionByName('a.nope'), 'missing path component yields undefined');
            } finally {
                restore();
            }
        });

        it('returns undefined when the resolved value is not a function', () => {
            assert.isUndefined(getFunctionByName('notAFn'), 'non-function value yields undefined');
        });
    });

    describe('getFunctionByNameOrThrow', () => {
        // Resolves paths against `window` like getFunctionByName; install a real DOM
        // and hang a known nested function off it for resolution.
        const myFn = () => 42;
        const fallback = () => 0;
        before(registerDom);
        after(unregisterDom);
        beforeEach(() => { (window as any).a = { b: { myFn } }; });
        afterEach(() => { delete (window as any).a; });

        it('returns the fallback for a falsy path', () => {
            assert.equal(getFunctionByNameOrThrow(null, fallback), fallback, 'null path returns fallback');
            assert.equal(getFunctionByNameOrThrow(undefined, fallback), fallback, 'undefined path returns fallback');
            assert.equal(getFunctionByNameOrThrow('', fallback), fallback, 'empty-string path returns fallback');
        });

        it('returns the resolved function for a valid path', () => {
            assert.equal(getFunctionByNameOrThrow('a.b.myFn', fallback), myFn, 'valid path resolves, ignoring fallback');
        });

        it('throws for a provided path that cannot be resolved', () => {
            const { restore } = captureConsole();
            try {
                assert.throws(() => getFunctionByNameOrThrow('a.nope', fallback), Error, /a\.nope/,
                    'unresolvable path throws');
            } finally {
                restore();
            }
        });
    });

    describe('isClass', () => {
        it('returns true for class declarations and false otherwise', () => {
            class Named {}
            assert.isTrue(isClass(Named), 'named class is a class');
            assert.isTrue(isClass(class {}), 'anonymous class is a class');
            assert.isFalse(isClass(function plain() {}), 'plain function is not a class');
            assert.isFalse(isClass(() => 0), 'arrow function is not a class');
            assert.isFalse(isClass(42), 'number is not a class');
            assert.isFalse(isClass(null), 'null is not a class');
            assert.isFalse(isClass({}), 'plain object is not a class');
        });
    });

    describe('makeImplementsInterfaceFunction', () => {
        interface Iface {
            required: () => void;
            alsoRequired: () => void;
            optional: () => void;
        }
        const implementsIface = makeImplementsInterfaceFunction<Iface>({
            required: true,
            alsoRequired: true,
            optional: false,
        });

        it('checks presence of required keys only', () => {
            assert.isFalse(implementsIface(null), 'null does not implement the interface');
            assert.isFalse(implementsIface(42), 'a primitive does not implement the interface');
            assert.isTrue(implementsIface({ required: 1, alsoRequired: 2 }),
                'all required keys present (any value counts) implements the interface');
            assert.isFalse(implementsIface({ required: 1 }), 'missing a required key fails');
            assert.isTrue(implementsIface({ required: 1, alsoRequired: 2 }),
                'optional (value=false) key is not required even when absent');
        });
    });

    describe('flattenMatrix', () => {
        const expected = Array.from({ length: 16 }, (_, i) => i);

        it('flattens all supported formats to the same plain number[]', () => {
            const rows2d = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]];
            const typedRows = rows2d.map(r => new Float32Array(r));

            assert.deepEqual(flattenMatrix(rows2d), expected, '2D array flattens row-major');
            assert.deepEqual(flattenMatrix(new Float32Array(expected)), expected, 'flat Float32Array becomes number[]');
            assert.deepEqual(flattenMatrix(typedRows), expected, 'array of typed-array rows flattens row-major');
            assert.deepEqual(flattenMatrix([...expected]), expected, 'flat number[] of length 16 returned as-is');
        });

        it('returns a plain Array (not a typed array)', () => {
            assert.isArray(flattenMatrix(new Float32Array(expected)), 'result is a plain Array');
        });

        it('throws for an unsupported shape', () => {
            assert.throws(() => flattenMatrix([1, 2, 3] as number[]), Error, /Unsupported matrix format/,
                'wrong-length flat array throws');
        });
    });

    describe('TestConvertAllMapsToObjects', () => {
        it('should convert plain Maps to objects', () => {
            const input = new Map([['name', 'test'], ['value', 42]]);
            const result = convertAllMapsToObjects(input);
        
            assert.isObject(result, 'Result should be an object');
            assert.equal(result.name, 'test', 'Name should be preserved');
            assert.equal(result.value, 42, 'Value should be preserved');
        });

        it('should handle nested Maps recursively', () => {
            const input = new Map([
                ['outer', new Map([
                    ['inner', new Map([['nested', 'value']])],
                    ['number', 123]
                ])]
            ]);
            const result = convertAllMapsToObjects(input);
        
            assert.isObject(result, 'Root should be an object');
            assert.isObject(result.outer, 'Outer should be an object');
            assert.isObject(result.outer.inner, 'Inner should be an object');
            assert.equal(result.outer.inner.nested, 'value', 'Nested value should be preserved');
            assert.equal(result.outer.number, 123, 'Number should be preserved');
        });

        it('should handle arrays with nested Maps', () => {
            const input = new Map([
                ['items', [1, new Map([['key', 'value']]), 'string', new Map([['another', 'object']])]]
            ]);
            const result = convertAllMapsToObjects(input);
        
            assert.isObject(result, 'Root should be an object');
            const items = result.items;
            assert.isArray(items, 'Items should remain an array');
            assert.equal(items[0], 1, 'First item should be unchanged');
            assert.isObject(items[1], 'Second item should be an object');
            assert.equal(items[1].key, 'value', 'Map in array should be converted');
            assert.equal(items[2], 'string', 'String should be unchanged');
            assert.isObject(items[3], 'Fourth item should be an object');
            assert.equal(items[3].another, 'object', 'Another Map in array should be converted');
        });

        it('should preserve existing objects and convert their Map values', () => {
            const input = {
                outerKey: new Map([['existing', 'value']]),
                regularObject: { nested: 'unchanged' }
            };
            const result = convertAllMapsToObjects(input);
        
            assert.isObject(result, 'Root should be an object');
            assert.isObject(result.outerKey, 'Map should be converted to object');
            assert.equal(result.outerKey.existing, 'value', 'Map value should be preserved');
            assert.isObject(result.regularObject, 'Regular object should remain an object');
            assert.equal(result.regularObject.nested, 'unchanged', 'Object value should be preserved');
        });

        it('should preserve primitives unchanged', () => {
            const input = new Map([
                ['string', 'hello'],
                ['number', 42],
                ['boolean', true],
                ['nullValue', null],
                ['undefinedValue', undefined]
            ]);
            const result = convertAllMapsToObjects(input);
        
            assert.isObject(result, 'Root should be an object');
            assert.equal(result.string, 'hello', 'String should be unchanged');
            assert.equal(result.number, 42, 'Number should be unchanged');
            assert.equal(result.boolean, true, 'Boolean should be unchanged');
            assert.isNull(result.nullValue, 'Null should be unchanged');
            assert.isUndefined(result.undefinedValue, 'Undefined should be unchanged');
        });

        it('should preserve typed arrays and built-in objects', () => {
            const float32Array = new Float32Array([1, 2, 3]);
            const uint8Array = new Uint8Array([4, 5, 6]);
            const date = new Date('2025-01-01');
            const input = new Map([
                ['float32', float32Array],
                ['uint8', uint8Array],
                ['date', date]
            ]);
            const result = convertAllMapsToObjects(input);
        
            assert.isObject(result, 'Root should be an object');
            assert.instanceOf(result.float32, Float32Array, 'Float32Array should be preserved');
            assert.instanceOf(result.uint8, Uint8Array, 'Uint8Array should be preserved');
            assert.instanceOf(result.date, Date, 'Date should be preserved');
            assert.deepEqual(Array.from(result.float32), [1, 2, 3], 'Float32Array values should be preserved');
            assert.deepEqual(Array.from(result.uint8), [4, 5, 6], 'Uint8Array values should be preserved');
            assert.equal(result.date.getTime(), date.getTime(), 'Date value should be preserved');
        });

        it('should handle null and undefined inputs', () => {
            assert.isNull(convertAllMapsToObjects(null), 'Null input should return null');
            assert.isUndefined(convertAllMapsToObjects(undefined), 'Undefined input should return undefined');
        });

        it('should handle primitive inputs', () => {
            assert.equal(convertAllMapsToObjects('string'), 'string', 'String input should return string');
            assert.equal(convertAllMapsToObjects(42), 42, 'Number input should return number');
            assert.equal(convertAllMapsToObjects(true), true, 'Boolean input should return boolean');
        });

        it('should handle complex nested structure', () => {
            const input = new Map([
                ['metadata', new Map([
                    ['name', 'complex_test'],
                    ['version', 1.0]
                ])],
                ['data', [
                    new Map([['type', 'mesh'], ['vertices', new Float32Array([1, 2, 3])]]),
                    new Map([['type', 'material'], ['properties', new Map([['color', 'red'], ['metalness', 0.5]])]])
                ]],
                ['config', { setting1: new Map([['enabled', true]]) }]
            ]);
            const result = convertAllMapsToObjects(input);
        
            // Check root structure
            assert.isObject(result, 'Root should be an object');
            assert.isObject(result.metadata, 'Metadata should be an object');
            assert.isArray(result.data, 'Data should remain an array');
            assert.isObject(result.config, 'Config should remain an object');
        
            // Check metadata
            const metadata = result.metadata;
            assert.equal(metadata.name, 'complex_test', 'Metadata name should be preserved');
            assert.equal(metadata.version, 1.0, 'Metadata version should be preserved');
        
            // Check data array
            const data = result.data;
            assert.isObject(data[0], 'First data item should be an object');
            assert.instanceOf(data[0].vertices, Float32Array, 'Vertices should remain Float32Array');
            assert.isObject(data[1], 'Second data item should be an object');
            assert.isObject(data[1].properties, 'Properties should be an object');
        
            // Check config
            const config = result.config;
            assert.isObject(config.setting1, 'Config value should be converted to object');
            assert.equal(config.setting1.enabled, true, 'Config nested value should be preserved');
        });

        it('should roundtrip with convertAllObjectsToMaps', () => {
            const original = {
                metadata: {
                    name: 'roundtrip_test',
                    version: 2.0
                },
                data: [
                    { type: 'mesh', vertices: new Float32Array([1, 2, 3]) },
                    { type: 'material', properties: { color: 'blue', roughness: 0.8 } }
                ],
                primitives: {
                    string: 'test',
                    number: 42,
                    boolean: true,
                    nullValue: null
                }
            };

            // Convert to Maps and back to objects
            const asMaps = convertAllObjectsToMaps(original);
            const backToObjects = convertAllMapsToObjects(asMaps);

            // Check structure preservation
            assert.isObject(backToObjects, 'Result should be an object');
            assert.isObject(backToObjects.metadata, 'Metadata should be an object');
            assert.equal(backToObjects.metadata.name, 'roundtrip_test', 'Metadata name should be preserved');
            assert.equal(backToObjects.metadata.version, 2.0, 'Metadata version should be preserved');
        
            assert.isArray(backToObjects.data, 'Data should be an array');
            assert.isObject(backToObjects.data[0], 'First data item should be an object');
            assert.instanceOf(backToObjects.data[0].vertices, Float32Array, 'Vertices should remain Float32Array');
            assert.deepEqual(Array.from(backToObjects.data[0].vertices), [1, 2, 3], 'Vertices values should be preserved');
        
            assert.isObject(backToObjects.primitives, 'Primitives should be an object');
            assert.equal(backToObjects.primitives.string, 'test', 'String should be preserved');
            assert.equal(backToObjects.primitives.number, 42, 'Number should be preserved');
            assert.equal(backToObjects.primitives.boolean, true, 'Boolean should be preserved');
            assert.isNull(backToObjects.primitives.nullValue, 'Null should be preserved');
        });
    });

    describe('toEmptyString', () => {
        it('function exists (python API needs it)', () => {
            assert.equal(toEmptyString({name: 'any', value: 123}), '');
        });
    });


});
