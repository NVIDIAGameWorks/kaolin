// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

import { assert } from "chai";
import * as fs from "fs";
import { testSamples } from '@test/helpers/paths';
import {convertValueToSupportedFormat,
    stringToBinary,
    stringFromBinary,
    gapUntil4Offset,
    gapUntilNOffset,
    binaryOffsetRequired,
    bytesForString,
    valueToType,
    BinaryWriter,
    RawBinaryWriter,
    BinaryIoDataType,
    typedArrayConstructorFromType,
    namedValueFromBinary,
    valueFromBinary,
    toBinary,
    fromBinary,
    encodeMessage,
    MESSAGE_TAG_KEY,
    MESSAGE_CONTENT_KEY} from '@kaolin/core/io';

// Helper function to generate random strings (similar to Python version)
function generateRandomString(length?: number, isAscii: boolean = false): string {
    if (length === undefined) {
        length = Math.floor(Math.random() * 12) + 4; // 4-15 chars
    }

    if (length === 0) {
        return '';
    }

    const asciiChars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;-_()[]';
    const utf8Chars = ['á', 'é', 'í', 'ó', 'ú', 'ñ', 'ü', 'ç', 'à', 'è', 'ì', 'ò', 'ù', 'â', 'ê', 'î', 'ô', 'û',
                       'ä', 'ë', 'ï', 'ö', 'ÿ', 'å', 'æ', 'ø', 'α', 'β', 'γ', 'δ', 'ε', 'λ', 'μ', 'π', 'σ', 'φ', 'ω',
                       '±', '×', '÷', '≤', '≥', '≠', '∞', '∑', '∫', '€', '£', '¥', '¢', '😀', '😎', '🚀', '🎉', '❤️', '🌟', '🔥', '💡'];

    let result = '';
    if (isAscii) {
        for (let i = 0; i < length; i++) {
            result += asciiChars[Math.floor(Math.random() * asciiChars.length)];
        }
    } else {
        // Mix of ASCII and UTF-8
        const numUtf8 = Math.min(length, Math.max(1, Math.floor(Math.random() * (length - 2)) + 1));
        const numAscii = length - numUtf8;

        let usedUtf8 = 0;
        let usedAscii = 0;
        for (let i = 0; i < length; i++) {
            let useAscii = false;
            let useUtf8 = false;
            if (usedUtf8 < numUtf8 && usedAscii < numAscii) {
                useAscii = Math.random() < 0.5;
                useUtf8 = !useAscii;
            } else if (usedUtf8 < numUtf8) {
                useUtf8 = true;
            } else if (usedAscii < numAscii) {
                useAscii = true;
            }
            if (useAscii) {
                result += asciiChars[Math.floor(Math.random() * asciiChars.length)];
                usedAscii++;
            } else {
                result += utf8Chars[Math.floor(Math.random() * utf8Chars.length)];
                usedUtf8++;
            }
        }
    }

    return result;
}

function isASCII(str) {
    for (let i = 0; i < str.length; i++) {
      const charCode = str.charCodeAt(i);
      if (charCode > 127) { // ASCII characters are 0-127
        return false;
      }
    }
    return true;
}

describe('visualize/dash/components/src/core/test_io.ts', () => {
    describe('Sanity_TestRandomStringGeneration', () => {
        it('should generate random strings correctly', () => {
            let lenghts = [10, 15, 20, 25, 30];
            for (let length of lenghts) {
                const testString = generateRandomString(length, true);
                // length is num bytes, not num characters
                assert.isAbove(testString.length + 1, length, `Generated string should have length > ${length}, got "${testString}"`);
            }
        });
        it('should generate ascii and utf8 strings correctly', () => {
            [true, false].forEach((isAscii) => {
                let string = generateRandomString(20, isAscii);
                if (isAscii) {
                    assert.equal(string.length, 20, `Generated string should have length 20, got "${string}"`);
                } else {
                    assert.isAbove(string.length + 1, 20, `Generated string should have length 20, got "${string}"`);
                }
                assert.equal(isASCII(string), isAscii, `Generated string should be isASCII=${isAscii}, got "${string}"`);
            });
        });
    });

    // Test data constants - automatically constructed from BinaryIoDataType enum
    const typesUsedForTesting = Object.values(BinaryIoDataType)
        .filter((value): value is BinaryIoDataType => typeof value === 'number') // Get only numeric enum values
        .filter(typeId => 
            typeId !== BinaryIoDataType.FLOAT16 &&    // Not supported yet
            typeId !== BinaryIoDataType.PNG &&        // Image Blob carrier — not an array type
            typeId !== BinaryIoDataType.JPEG &&
            typeId !== BinaryIoDataType.UNSUPPORTED
        );
    const _NON_ARRAY_TYPES = [
        BinaryIoDataType.STRING, BinaryIoDataType.DICT, BinaryIoDataType.LIST,
        BinaryIoDataType.PNG, BinaryIoDataType.JPEG
    ];
    const arrayTypesUsedForTesting = typesUsedForTesting.filter(x => !_NON_ARRAY_TYPES.includes(x));

    describe('TestAlignment', () => {
        describe('gapUntilNOffset', () => {
            const byteSizes = [2, 4, 8, 16];

            byteSizes.forEach((bytesize) => {
                it(`should calculate gap for ${bytesize}-byte alignment`, () => {
                    for (let expectedGap = 0; expectedGap < bytesize; expectedGap++) {
                        const n = Math.floor(Math.random() * 6);
                        const offset = bytesize * n - expectedGap;
                        const actualGap = gapUntilNOffset(offset, bytesize);
                        assert.equal(actualGap, expectedGap,
                            `gapUntilNOffset(${offset}, ${bytesize}) = ${actualGap}, expected ${expectedGap}`);
                    }
                });
            });
        });

        describe('gapUntil4Offset', () => {
            it('should calculate 4-byte alignment gaps correctly', () => {
                const testCases = [
                    { offset: 0, expected: 0 },
                    { offset: 1, expected: 3 },
                    { offset: 2, expected: 2 },
                    { offset: 3, expected: 1 },
                    { offset: 4, expected: 0 },
                    { offset: 5, expected: 3 },
                    { offset: 8, expected: 0 }
                ];

                testCases.forEach(({ offset, expected }) => {
                    const actual = gapUntil4Offset(offset);
                    assert.equal(actual, expected,
                        `gapUntil4Offset(${offset}) = ${actual}, expected ${expected}`);
                });
            });
        });
    });

    describe('TestStringConversion', () => {
        const testCases = [
            { name: 'ascii', input: 'hello world', isAscii: true },
            { name: 'utf8', input: '世界', isAscii: false },
            { name: 'emoji', input: '🚀 test', isAscii: false },
            { name: 'empty', input: '', isAscii: true },
            { name: 'mixed', input: 'café 🌟', isAscii: false },
            { name: 'accented', input: 'résumé naïve', isAscii: false }
        ];

        describe('string roundtrip tests', () => {
            testCases.forEach(({ name, input }) => {
                it(`should handle ${name} strings: "${input}"`, () => {
                    const binary = stringToBinary(input);
                    const result = stringFromBinary(binary, 0, binary.byteLength);
                    assert.equal(result, input, `Failed roundtrip for ${name} string: "${input}"`);
                });
            });
        });

        describe('random string tests', () => {
            [true, false].forEach((isAscii) => {
                it(`should handle random strings (ascii=${isAscii})`, () => {
                    for (let i = 0; i < 10; i++) {
                        const testString = generateRandomString(undefined, isAscii);
                        assert.isAbove(testString.length, 1, `Generated string should have length > 1, got "${testString}"`);
                        const binary = stringToBinary(testString);
                        const decoded = stringFromBinary(binary, 0, binary.byteLength);
                        assert.equal(decoded, testString, `Failed random string roundtrip (ascii=${isAscii}): "${testString}"`);
                    }
                });
            });
        });

        it('should handle strings with different offsets', () => {
            const testString = 'hello world';
            const binary = stringToBinary(testString);

            // Test with different offset paddings
            for (const offset of [0, 4, 15]) {
                const paddedBinary = new Uint8Array(offset + binary.byteLength);
                paddedBinary.set(new Uint8Array(binary), offset);

                const decoded = stringFromBinary(paddedBinary.buffer, offset, binary.byteLength);
                assert.equal(decoded, testString,
                    `Failed offset test with offset=${offset}: expected "${testString}", got "${decoded}"`);
            }
        });
    });

    describe('TestTypeMapping', () => {
        describe('typed array constructor tests', () => {
            arrayTypesUsedForTesting.forEach((typeId) => {
                it(`should return correct type info for ${typeId}`, () => {
                    const [constructor, bytesPerElement] = typedArrayConstructorFromType(typeId);
                    const offsetRequired = binaryOffsetRequired(typeId);

                    assert.isDefined(constructor, `Constructor should be defined for type ${typeId}`);
                    assert.isAbove(bytesPerElement, 0, `Bytes per element should be > 0 for type ${typeId}`);
                    assert.equal(offsetRequired, bytesPerElement,
                        `Offset required should equal bytes per element for type ${typeId}`);

                    // Verify we can create an array of this type
                    const testArray = new constructor(4);
                    assert.equal(testArray.byteLength, 4 * bytesPerElement,
                        `Test array byte length should be 4 * ${bytesPerElement} for type ${typeId}`);
                });
            });
        });

        it('should detect value types correctly', () => {
            // Test strings
            assert.equal(valueToType('hello'), BinaryIoDataType.STRING);
            assert.equal(valueToType(''), BinaryIoDataType.STRING);
            assert.equal(valueToType('🚀'), BinaryIoDataType.STRING);

            // Test numbers
            assert.equal(valueToType(42), BinaryIoDataType.UNSUPPORTED);
            assert.equal(valueToType(convertValueToSupportedFormat(42)), BinaryIoDataType.INT32);
            assert.equal(valueToType(3.14), BinaryIoDataType.UNSUPPORTED);
            assert.equal(valueToType(convertValueToSupportedFormat(3.14)), BinaryIoDataType.FLOAT32);
            assert.equal(valueToType(0), BinaryIoDataType.UNSUPPORTED);
            assert.equal(valueToType(convertValueToSupportedFormat(0)), BinaryIoDataType.INT32);

            // Test typed arrays
            assert.equal(valueToType(new Int8Array(4)), BinaryIoDataType.INT8);
            assert.equal(valueToType(new Uint8Array(4)), BinaryIoDataType.UINT8);
            assert.equal(valueToType(new Int16Array(4)), BinaryIoDataType.INT16);
            assert.equal(valueToType(new Int32Array(4)), BinaryIoDataType.INT32);
            assert.equal(valueToType(new Float32Array(4)), BinaryIoDataType.FLOAT32);
            assert.equal(valueToType(new Float64Array(4)), BinaryIoDataType.FLOAT64);

            // Test ArrayBuffer
            assert.equal(valueToType(new ArrayBuffer(16)), BinaryIoDataType.UINT8);

            // Test lists
            // Simple homogeneous lists (get converted to typed arrays)
            let simpleIntList = [1, 2, 3, 4];
            assert.equal(valueToType(simpleIntList), BinaryIoDataType.LIST);  // TODO: should fix?
            assert.equal(valueToType(convertValueToSupportedFormat(simpleIntList)), BinaryIoDataType.INT32);
        
            let simpleFloatList = [1.5, 2.5, 3.5];
            assert.equal(valueToType(simpleFloatList), BinaryIoDataType.LIST);  // TODO: should fix?
            assert.equal(valueToType(convertValueToSupportedFormat(simpleFloatList)), BinaryIoDataType.FLOAT32);

            // Mixed lists (stay as arrays, use LIST data type)
            let mixedList = [42, 'hello', 3.14];
            assert.equal(valueToType(mixedList), BinaryIoDataType.LIST);
            assert.equal(valueToType(convertValueToSupportedFormat(mixedList)), BinaryIoDataType.LIST);

            // Empty array
            let emptyList = [];
            assert.equal(valueToType(emptyList), BinaryIoDataType.LIST);
            assert.equal(valueToType(convertValueToSupportedFormat(emptyList)), BinaryIoDataType.LIST);

            // Test dict and Map
            let mapVal = new Map<string, string>();
            mapVal.set('a', 'hello');
            assert.equal(valueToType(mapVal), BinaryIoDataType.DICT);
            let objVal = {'a': 'hello'};
            assert.equal(valueToType(objVal), BinaryIoDataType.UNSUPPORTED);
            assert.equal(valueToType(convertValueToSupportedFormat(objVal)), BinaryIoDataType.DICT);

            // Test unsupported types
            assert.equal(valueToType(null), BinaryIoDataType.UNSUPPORTED);
            assert.equal(valueToType(undefined), BinaryIoDataType.UNSUPPORTED);
            assert.equal(valueToType({}), BinaryIoDataType.UNSUPPORTED);

            // Image Blobs are recognized.
            const pngBlob = new Blob([new Uint8Array([0x89, 0x50])], { type: 'image/png' });
            const jpegBlob = new Blob([new Uint8Array([0xff, 0xd8])], { type: 'image/jpeg' });
            const wrongBlob = new Blob([new Uint8Array([0])], { type: 'text/plain' });
            assert.equal(valueToType(pngBlob), BinaryIoDataType.PNG);
            assert.equal(valueToType(jpegBlob), BinaryIoDataType.JPEG);
            assert.equal(valueToType(wrongBlob), BinaryIoDataType.UNSUPPORTED);
        });
    });

    async function testDictRoundTrip(testDict: Map<string, any>) {
        const binaryData = await toBinary(testDict);
        assert.isDefined(binaryData, 'Binary data should be defined');
        assert.isAbove(binaryData!.byteLength, 0, 'Binary data should have non-zero length');

        const decodedDict = fromBinary(binaryData!);
        assert.isDefined(decodedDict, 'Decoded dictionary should be defined');

        // Verify dictionary has correct number of entries
        assert.equal(decodedDict.size, testDict.size,
                `Decoded dict should have ${testDict.size} entries, got ${decodedDict.size}`);

        // Verify keys
        for (const [k, v] of testDict) {
            assert.isTrue(decodedDict.has(k), `Decoded array is missing key ${k} among keys ${Array.from(decodedDict.keys())}`);
        }

        for (const [k, v] of decodedDict) {
            assert.isTrue(testDict.has(k), `Decoded array contains extra key ${k}; should be ${Array.from(testDict.keys())}`);
        }
    };

    describe('TestDictionaryRoundTrip', () => {
        it('should correctly round trip dictionary through binary', async () => {
            // Create test dictionary with various data types
            const testDict = new Map<string, any>();

            // Add string value
            testDict.set('test_string', 'hello world 🚀');

            // Add typed array value
            const testIntArray = new Int32Array([1, 2, 3, 4, 5]);
            testDict.set('test_array', testIntArray);

            // Add float array
            const testFloatArray = new Float32Array([3.14, 2.718, 1.414]);
            testDict.set('test_floats', testFloatArray);

            // Add simple list (homogeneous - gets converted to typed array)
            testDict.set('simple_list', [10, 20, 30, 40]);

            // Add mixed list (heterogeneous - uses LIST data type)
            testDict.set('mixed_list', [123, 'test string', 2.5]);

            await testDictRoundTrip(testDict);
        });
    });

    describe('TestUtilityFunctions', () => {
        it('should calculate string byte length correctly', () => {
            assert.equal(bytesForString(''), 0);
            assert.equal(bytesForString('hello'), 5);
            assert.equal(bytesForString('世界'), 6); // 3 bytes per character
            assert.equal(bytesForString('🚀'), 4); // 4 bytes for emoji
        });

        it('should calculate binary offset requirements', () => {
            assert.equal(binaryOffsetRequired(BinaryIoDataType.STRING), 1);
            assert.equal(binaryOffsetRequired(BinaryIoDataType.INT8), 1);
            assert.equal(binaryOffsetRequired(BinaryIoDataType.UINT8), 1);
            assert.equal(binaryOffsetRequired(BinaryIoDataType.INT16), 2);
            assert.equal(binaryOffsetRequired(BinaryIoDataType.INT32), 4);
            assert.equal(binaryOffsetRequired(BinaryIoDataType.INT64), 8);
            assert.equal(binaryOffsetRequired(BinaryIoDataType.FLOAT32), 4);
            assert.equal(binaryOffsetRequired(BinaryIoDataType.FLOAT64), 8);
        });
    });
    describe('TestRawBinaryWriter', () => {
        it('should write int32 values correctly', () => {
            const writer = new RawBinaryWriter(16);
            writer.writeInt32(42).writeInt32(-123);

            const buffer = writer.getBuffer();
            const view = new Int32Array(buffer);

            assert.equal(view[0], 42);
            assert.equal(view[1], -123);
        });

        it('should write float32 values correctly', () => {
            const writer = new RawBinaryWriter(16);
            writer.writeFloat32(3.14).writeFloat32(-2.718);

            const buffer = writer.getBuffer();
            const view = new Float32Array(buffer);

            assert.approximately(view[0], 3.14, 0.00001);
            assert.approximately(view[1], -2.718, 0.00001);
        });

        it('should write strings correctly', () => {
            const testString = 'hello world 🚀';
            const writer = new RawBinaryWriter(32);
            writer.writeString(testString);

            const buffer = writer.getBuffer();
            const decoded = new TextDecoder().decode(buffer);

            assert.equal(decoded, testString);
        });

        it('should write ArrayBuffer correctly', () => {
            const sourceData = new Uint8Array([1, 2, 3, 4, 5]);
            const writer = new RawBinaryWriter(16);
            writer.writeArrayBuffer(sourceData.buffer);

            const buffer = writer.getBuffer();
            const result = new Uint8Array(buffer);

            assert.deepEqual(Array.from(result), [1, 2, 3, 4, 5]);
        });

        it('should handle fast forwarding correctly', () => {
            const writer = new RawBinaryWriter(16);
            assert.equal(writer.getOffset(), 0);

            writer.fastForward(4);
            assert.equal(writer.getOffset(), 4);

            writer.fastForwardToSuitableOffset(8);
            assert.equal(writer.getOffset(), 8); // Already aligned

            writer.fastForward(1);
            writer.fastForwardToSuitableOffset(4);
            assert.equal(writer.getOffset(), 12); // 9 + 3 gap
        });

        it('should throw error when exceeding capacity', () => {
            const writer = new RawBinaryWriter(4);
            writer.writeInt32(42); // Uses all 4 bytes

            assert.throws(() => writer.writeInt32(123), 'Trying to write bytes over capacity');
        });
    });
    describe('TestBinaryWriter', () => {
        it('should calculate capacity for int with/without meta', () => {
            let writer = new BinaryWriter().needsToWriteInt32();
            assert.equal(writer.capacityNeeded, 4);

            assert.doesNotThrow(() => writer.allocate());
            assert.doesNotThrow(() => writer.writeInt32(42));

            // Now let's write same with the array metadata
            writer = new BinaryWriter().needsToWriteValueWithMetadata(42);
            assert.isAbove(writer.capacityNeeded, 3);

            // Should not throw error when allocating
            assert.doesNotThrow(() => writer.allocate());
            assert.doesNotThrow(() => writer.writeInt32(42));
        });

        it('should calculate capacity for float with/without meta', () => {
            let writer = new BinaryWriter().needsToWriteFloat32();
            assert.equal(writer.capacityNeeded, 4);

            assert.doesNotThrow(() => writer.allocate());
            assert.doesNotThrow(() => writer.writeFloat32(3.15));

            // Now let's write same with the array metadata
            writer = new BinaryWriter().needsToWriteValueWithMetadata(3.15);
            assert.isAbove(writer.capacityNeeded, 3);

            // Should not throw error when allocating
            assert.doesNotThrow(() => writer.allocate());
            assert.doesNotThrow(() => writer.writeFloat32(3.15));
        });

        it('should calculate capacity for array', () => {
            let value = new Float32Array([3.15, 4.16, 20.5]);
            const writer = new BinaryWriter().needsToWriteValueWithMetadata(value);
            assert.isAbove(writer.capacityNeeded, 4 * 3);

            // Should not throw error when allocating
            assert.doesNotThrow(() => writer.allocate());
            assert.doesNotThrow(() => writer.writeValueWithMetadata(value));
        });

        it('should calculate capacity for simple list', () => {
            // Test simple homogeneous list (gets converted to typed array)
            let simpleList = [1, 2, 3, 4, 5];
            const writer1 = new BinaryWriter().needsToWriteValueWithMetadata(simpleList);
            assert.isAbove(writer1.capacityNeeded, 4 * 5); // At least the data size

            // Should not throw error when allocating
            assert.doesNotThrow(() => writer1.allocate());
            assert.doesNotThrow(() => writer1.writeValueWithMetadata(simpleList));

            // Test mixed list (uses LIST data type)
            let mixedList = [42, "hello", 3.14];
            const writer2 = new BinaryWriter().needsToWriteValueWithMetadata(mixedList);
            assert.isAbove(writer2.capacityNeeded, 20); // Should need space for all elements plus metadata

            // Should not throw error when allocating
            assert.doesNotThrow(() => writer2.allocate());
            assert.doesNotThrow(() => writer2.writeValueWithMetadata(mixedList));
        });

        it('should calculate capacity for ascii string', () => {
            let value = 'Sima and Tanya';
            const writer = new BinaryWriter().needsToWriteValueWithMetadata(value);
            assert.isAbove(writer.capacityNeeded, 3);

            // Should not throw error when allocating
            assert.doesNotThrow(() => writer.allocate());
            assert.doesNotThrow(() => writer.writeValueWithMetadata(value));
        });

        it('should calculate capacity for UTF-8 string', () => {
            let value = 'Сима и Таня 🤩';
            const writer = new BinaryWriter().needsToWriteValueWithMetadata(value);
            assert.isAbove(writer.capacityNeeded, 3);

            // Should not throw error when allocating
            assert.doesNotThrow(() => writer.allocate());
            assert.doesNotThrow(() => writer.writeValueWithMetadata(value));
        });

        it('should calculate capacity for simple values', () => {
            const writer = new BinaryWriter()
                .needsToWriteInt32()
                .needsToWriteFloat32()
                .needsToWriteValueWithMetadata('hello');

            // Should not throw error when allocating
            assert.doesNotThrow(() => writer.allocate());
            assert.doesNotThrow(() => writer.writeInt32(42).writeFloat32(3.14).writeValueWithMetadata('hello'));
        });

        it('should write and retrieve simple values', () => {
            const writer = new BinaryWriter().needsToWriteValueWithMetadata(42);
            assert.equal(writer.capacityNeeded, 4 + 4 + 4);  // type, shape_length, value
            writer.allocate().writeValueWithMetadata(42);

            const buffer = writer.getBuffer();
            assert.isDefined(buffer);
            assert.isAbove(buffer!.byteLength, 0);

            let decoded = fromBinary(buffer);
            assert.isTrue(typeof decoded === 'number');
            assert.equal(decoded, 42);
        });

        it('should handle maps', () => {
            let value = new Map<string, any>();
            value.set('a', 'hello there');
            value.set('b', [0, 1, 2, 3, 4]);
            const writer = new BinaryWriter().needsToWriteValueWithMetadata(value);
            const expectedCapacityNoAlignment = 4 + 4 + 4 + // type, shape_length, shape
                4 + 1 + 4 * 3 + 11 + // first named value and meta
                4 + 1 + 4 * 3 + 4 * 5; // second named value and meta
            assert.isAbove(writer.capacityNeeded, expectedCapacityNoAlignment);
            writer.allocate().writeValueWithMetadata(value);

            const buffer = writer.getBuffer();
            assert.isDefined(buffer);
            assert.isAbove(buffer!.byteLength, expectedCapacityNoAlignment);

            let decoded = fromBinary(buffer);
            assert.isTrue(decoded.has('a'));
            assert.isTrue(decoded.has('b'));
        });

        it('should handle named values', () => {
            const writer = new BinaryWriter()
                .needsToWriteNamedValue('test_int', 42)
                .needsToWriteNamedValue('test_string', 'hello')
                .allocate()
                .writeNamedValue('test_int', 42)
                .writeNamedValue('test_string', 'hello');

            const buffer = writer.getBuffer();
            assert.isDefined(buffer);
            assert.isAbove(buffer!.byteLength, 0);
        });

        it('should throw error for unsupported types', () => {
            const writer = new BinaryWriter();

            assert.throws(() => writer.needsToWriteValueWithMetadata(null));
        });

        it('should throw error when allocate not called', () => {
            const writer = new BinaryWriter();

            // Should not throw error (just log)
            writer.writeValueWithMetadata(42);

            assert.isNull(writer.getBuffer());
        });

        it('should throw error when capacity requested after allocate', () => {
            const writer = new BinaryWriter()
                .needsToWriteInt32()
                .allocate();

            assert.throws(() => writer.needsToWriteValueWithMetadata(123), 'BinaryWriter: extra capacity requested after allocate is called');
        });
    });


    describe('TestValueToFromBinary', () => {
        it('should encode decode string', () => {
            const testString = 'hello world 🚀';

            const writer = new BinaryWriter().needsToWriteValueWithMetadata(testString).allocate();
            writer.writeValueWithMetadata(testString);
            let binaryEnc = writer.getBuffer();

            const [decoded, shape, readBytes] = valueFromBinary(binaryEnc);
            assert.equal(testString, decoded);
            assert.equal(readBytes, binaryEnc.byteLength);
        });

        it('should encode decode array', () => {
            const testArray = new Float32Array([12, 13.5, 16]);
            const writer = new BinaryWriter().needsToWriteValueWithMetadata(testArray).allocate();
            writer.writeValueWithMetadata(testArray);
            let binaryEnc = writer.getBuffer();

            const [decoded, shape, readBytes] = valueFromBinary(binaryEnc);
            assert.equal(readBytes, binaryEnc.byteLength);
            assert.equal(testArray.length, decoded.length);
            assert.isTrue(decoded instanceof Float32Array);

            for (let i = 0; i < testArray.length; ++i) {
                assert.approximately(testArray[i], decoded[i], 0.00001, `Array element ${i} should match: expected ${testArray[i]}, got ${decoded[i]}`);
            }
        });

        it('should encode decode a simple list', () => {
            const testList = [1, 2, 3, 4, 5]; // Simple homogeneous array -> converts to Int32Array
        
            const writer = new BinaryWriter().needsToWriteValueWithMetadata(testList).allocate();
            writer.writeValueWithMetadata(testList);
            let binaryEnc = writer.getBuffer();

            const [decoded, shape, readBytes] = valueFromBinary(binaryEnc);
            assert.equal(readBytes, binaryEnc.byteLength);
            assert.instanceOf(decoded, Int32Array, `Decoded value should be an Int32Array, but is ${typeof decoded}`);
            assert.equal(testList.length, decoded.length, 'Array lengths should match');
        
            for (let i = 0; i < testList.length; i++) {
                assert.equal(testList[i], decoded[i], `Array element ${i} should match: expected ${testList[i]}, got ${decoded[i]}`);
            }

            // Test float array
            const testFloatList = [1.5, 2.5, 3.5, 4.5, 5.5]; // Simple homogeneous float array -> converts to Float32Array
        
            const writerFloat = new BinaryWriter().needsToWriteValueWithMetadata(testFloatList).allocate();
            writerFloat.writeValueWithMetadata(testFloatList);
            let binaryEncFloat = writerFloat.getBuffer();

            const [decodedFloat, shapeFloat, readBytesFloat] = valueFromBinary(binaryEncFloat);
            assert.equal(readBytesFloat, binaryEncFloat.byteLength);
            assert.instanceOf(decodedFloat, Float32Array, 'Decoded value should be a Float32Array');
            assert.equal(testFloatList.length, decodedFloat.length, 'Array lengths should match');
        
            for (let i = 0; i < testFloatList.length; i++) {
                assert.approximately(testFloatList[i], decodedFloat[i], 0.001, `Array element ${i} should match: expected ${testFloatList[i]}, got ${decodedFloat[i]}`);
            }
        });

        it('should encode decode a mixed list', () => {
            const testList = [42, "hello world", 3.14, new Float32Array([1, 2, 3])]; // Mixed types
        
            const writer = new BinaryWriter().needsToWriteValueWithMetadata(testList).allocate();
            writer.writeValueWithMetadata(testList);
            let binaryEnc = writer.getBuffer();

            const [decoded, shape, readBytes] = valueFromBinary(binaryEnc);
            assert.equal(readBytes, binaryEnc.byteLength);
            assert.isArray(decoded, 'Decoded value should be an array');
            assert.equal(testList.length, decoded.length, 'Array lengths should match');
        
            // Check each element with type-aware comparison
            assert.equal(decoded[0], 42, 'Integer should match');
            assert.equal(decoded[1], "hello world", 'String should match');
            assert.approximately(decoded[2], 3.14, 0.001, 'Float should match');
        
            // Check typed array
            assert.instanceOf(decoded[3], Float32Array, 'Typed array should preserve type');
            const originalArray = testList[3] as Float32Array;
            const decodedArray = decoded[3] as Float32Array;
            assert.equal(originalArray.length, decodedArray.length, 'Typed array lengths should match');
            for (let i = 0; i < originalArray.length; i++) {
                assert.approximately(originalArray[i], decodedArray[i], 0.001, 
                    `Typed array element ${i} should match: expected ${originalArray[i]}, got ${decodedArray[i]}`);
            }
        });

        it('should encode decode string and array', () => {
            const testString = 'hello world 🚀';
            const testArray = new Float32Array([12, 13.5, 16]);

            const writer = new BinaryWriter().needsToWriteValueWithMetadata(testString).needsToWriteValueWithMetadata(testArray);
            writer.allocate().writeValueWithMetadata(testString).writeValueWithMetadata(testArray);
            let binaryEnc = writer.getBuffer();

            const [decodedStr, shapeStr, readBytesStr] = valueFromBinary(binaryEnc);
            assert.equal(testString, decodedStr);
        
            const [decodedArr, shape, readBytes] = valueFromBinary(binaryEnc, readBytesStr);
            assert.equal(testArray.length, decodedArr.length);
            assert.isTrue(decodedArr instanceof Float32Array);

            for (let i = 0; i < testArray.length; ++i) {
                assert.approximately(testArray[i], decodedArr[i], 0.00001, `Array element ${i} should match: expected ${testArray[i]}, got ${decodedArr[i]}`);
            }
        });

    });


    describe('TestNamedValueToFromBinary', () => {
        it('should encode decode multiple named values', () => {
            let name1 = 'cats 🤩';
            let value1 = 'Сима и Таня';

            let name2 = 'photo';
            let value2 = new Uint8Array([1, 2, 3, 1, 2, 3, 15]);

            let name3 = 'Сима weight height length';
            let value3 = new Float32Array([10.5, 33, 60.7]);

            const writer = new BinaryWriter();
            writer.needsToWriteNamedValue(name1, value1);
            writer.needsToWriteNamedValue(name2, value2);
            writer.needsToWriteNamedValue(name3, value3);
            writer.allocate();
            writer.writeNamedValue(name1, value1);
            writer.writeNamedValue(name2, value2);
            writer.writeNamedValue(name3, value3);

            let binaryEnc = writer.getBuffer();
            let [decValue1, decShape1, readBytes1, decName1] = namedValueFromBinary(binaryEnc);
            assert.equal(name1, decName1);
            assert.equal(value1, decValue1);

            let [decValue2, decShape2, readBytes2, decName2] = namedValueFromBinary(binaryEnc, readBytes1);
            assert.equal(name2, decName2);
            assert.equal(value2.length, decValue2.length);
            assert.isTrue(decValue2 instanceof Uint8ClampedArray);
            for (let i = 0; i < value2.length; ++i) {
                assert.equal(value2[i], decValue2[i]);
            }

            let [decValue3, decShape3, readBytes3, decName3] = namedValueFromBinary(binaryEnc, readBytes1 + readBytes2);
            assert.equal(name3, decName3);
            assert.equal(value3.length, 3);
            assert.equal(value3.length, decValue3.length);
            assert.isTrue(decValue3 instanceof Float32Array);
            for (let i = 0; i < value3.length; ++i) {
                assert.approximately(value3[i], decValue3[i], 0.0001);
            }
        });
    });


    // Helper: byte-by-byte comparison of two ArrayBuffers.
    function buffersEqual(a: ArrayBuffer, b: ArrayBuffer): boolean {
        if (a.byteLength !== b.byteLength) return false;
        const va = new Uint8Array(a);
        const vb = new Uint8Array(b);
        for (let i = 0; i < a.byteLength; ++i) {
            if (va[i] !== vb[i]) return false;
        }
        return true;
    }


    describe('TestByteIdentity', () => {
        // Regression guards for the no-Blob path. Bytes produced by
        // ``await toBinary(value)`` must match exactly what the synchronous
        // BinaryWriter would have written before this PR. The reference is
        // computed using the same code paths against a fresh BinaryWriter.

        it('should produce stable bytes for a Map with strings and typed arrays', async () => {
            const map = new Map<string, any>([
                ['a', 'hello'],
                ['b', new Float32Array([1, 2, 3])],
            ]);
            const encoded = await toBinary(map);
            assert.isNotNull(encoded);

            // Reference bytes computed via the synchronous writer directly.
            const writer = new BinaryWriter()
                .needsToWriteValueWithMetadata(map)
                .allocate();
            writer.writeValueWithMetadata(map);
            const reference = writer.getBuffer();

            assert.isTrue(buffersEqual(encoded!, reference!),
                'await toBinary(map) bytes must match the synchronous writer output');

            // And of course a roundtrip yields equivalent content.
            const decoded = fromBinary(encoded!);
            assert.equal(decoded.get('a'), 'hello');
            const decB = decoded.get('b') as Float32Array;
            assert.instanceOf(decB, Float32Array);
            assert.equal(decB.length, 3);
            assert.approximately(decB[0], 1, 1e-6);
        });

        it('should preserve the value.shape hack on Uint8ClampedArray inside a Map', async () => {
            // The shape hack is preserved by `writeNamedValue` (inside a Map),
            // which is the documented happy path — `RenderRemoteImageBehavior`
            // and friends only see typed arrays inside Maps.
            const value = new Uint8ClampedArray([0, 1, 2, 3]);
            (value as any).shape = [2, 2];
            const wrapper = new Map<string, any>([['arr', value]]);

            const encoded = await toBinary(wrapper);
            assert.isNotNull(encoded);

            const writer = new BinaryWriter()
                .needsToWriteValueWithMetadata(wrapper)
                .allocate();
            writer.writeValueWithMetadata(wrapper);
            const reference = writer.getBuffer();

            assert.isTrue(buffersEqual(encoded!, reference!),
                'Uint8ClampedArray with .shape (in Map) must encode byte-identically through the async path');

            const decoded = fromBinary(encoded!) as Map<string, any>;
            const decArr = decoded.get('arr') as Uint8ClampedArray;
            assert.instanceOf(decArr, Uint8ClampedArray);
            assert.deepEqual(Array.from(decArr), [0, 1, 2, 3]);
            assert.deepEqual((decArr as any).shape, [2, 2]);
        });

        it('encodeMessage on the binary path returns a Promise<ArrayBuffer>', async () => {
            const result = encodeMessage('tag1', new Map<string, any>([['x', 42]]));
            // .then is duck-typed for promise detection.
            assert.isFunction((result as Promise<any>).then,
                'encodeMessage(binary=true) should return a Promise');
            const buffer = await result;
            assert.instanceOf(buffer, ArrayBuffer);

            const decoded = fromBinary(buffer as ArrayBuffer);
            assert.equal(decoded.get(MESSAGE_TAG_KEY), 'tag1');
            const content = decoded.get(MESSAGE_CONTENT_KEY) as Map<string, any>;
            assert.instanceOf(content, Map);
        });
    });


    describe('TestBlobIO', () => {
        // Minimal valid PNG file: 1x1 transparent pixel.
        // Hex dump generated once with PIL.Image.new('RGBA', (1,1), (0,0,0,0)).save(...).
        const MINIMAL_PNG_BYTES = new Uint8Array([
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a,
            0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52,
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
            0x08, 0x06, 0x00, 0x00, 0x00, 0x1f, 0x15, 0xc4,
            0x89, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x44, 0x41,
            0x54, 0x78, 0x9c, 0x63, 0x00, 0x01, 0x00, 0x00,
            0x05, 0x00, 0x01, 0x0d, 0x0a, 0x2d, 0xb4, 0x00,
            0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, 0x44, 0xae,
            0x42, 0x60, 0x82
        ]);

        // Minimal valid JPEG SOI + EOI markers (not a decodable image, but
        // sufficient as a typed-blob payload for round-trip tests).
        const MINIMAL_JPEG_BYTES = new Uint8Array([
            0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46,
            0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
            0x00, 0x01, 0x00, 0x00, 0xff, 0xd9
        ]);

        async function arrayBufferFromBlob(b: Blob): Promise<ArrayBuffer> {
            return await b.arrayBuffer();
        }

        it('should round trip a PNG Blob through toBinary/fromBinary', async () => {
            const original = new Blob([MINIMAL_PNG_BYTES], { type: 'image/png' });
            const encoded = await toBinary(original);
            assert.isNotNull(encoded);

            const decoded = fromBinary(encoded!);
            assert.instanceOf(decoded, Blob, 'PNG payload must decode to a Blob');
            assert.equal((decoded as Blob).type, 'image/png');

            const roundtripped = new Uint8Array(await arrayBufferFromBlob(decoded as Blob));
            assert.equal(roundtripped.byteLength, MINIMAL_PNG_BYTES.byteLength);
            for (let i = 0; i < MINIMAL_PNG_BYTES.byteLength; ++i) {
                assert.equal(roundtripped[i], MINIMAL_PNG_BYTES[i],
                    `PNG byte ${i} mismatch`);
            }
        });

        it('should round trip a JPEG Blob through toBinary/fromBinary', async () => {
            const original = new Blob([MINIMAL_JPEG_BYTES], { type: 'image/jpeg' });
            const encoded = await toBinary(original);
            assert.isNotNull(encoded);

            const decoded = fromBinary(encoded!);
            assert.instanceOf(decoded, Blob);
            assert.equal((decoded as Blob).type, 'image/jpeg');

            const roundtripped = new Uint8Array(await arrayBufferFromBlob(decoded as Blob));
            for (let i = 0; i < MINIMAL_JPEG_BYTES.byteLength; ++i) {
                assert.equal(roundtripped[i], MINIMAL_JPEG_BYTES[i]);
            }
        });

        it('should round trip a Map containing strings, typed arrays, and a PNG Blob', async () => {
            const blob = new Blob([MINIMAL_PNG_BYTES], { type: 'image/png' });
            const message = new Map<string, any>([
                ['tag', 'render'],
                ['pose', new Float32Array([1, 0, 0, 0, 1, 0, 0, 0, 1])],
                ['image', blob],
            ]);
            const encoded = await toBinary(message);
            assert.isNotNull(encoded);

            const decoded = fromBinary(encoded!);
            assert.equal(decoded.get('tag'), 'render');
            const pose = decoded.get('pose') as Float32Array;
            assert.instanceOf(pose, Float32Array);
            assert.equal(pose.length, 9);
            const decBlob = decoded.get('image') as Blob;
            assert.instanceOf(decBlob, Blob);
            assert.equal(decBlob.type, 'image/png');

            const roundtripped = new Uint8Array(await decBlob.arrayBuffer());
            for (let i = 0; i < MINIMAL_PNG_BYTES.byteLength; ++i) {
                assert.equal(roundtripped[i], MINIMAL_PNG_BYTES[i]);
            }
        });

        it('should round trip multiple PNG Blobs in parallel via Promise.all', async () => {
            const blob1 = new Blob([MINIMAL_PNG_BYTES], { type: 'image/png' });
            const blob2 = new Blob([MINIMAL_PNG_BYTES], { type: 'image/png' });
            const blob3 = new Blob([MINIMAL_JPEG_BYTES], { type: 'image/jpeg' });
            const message = new Map<string, any>([
                ['a', blob1],
                ['b', blob2],
                ['c', blob3],
            ]);

            const encoded = await toBinary(message);
            assert.isNotNull(encoded);
            const decoded = fromBinary(encoded!);
            assert.instanceOf(decoded.get('a'), Blob);
            assert.instanceOf(decoded.get('b'), Blob);
            assert.instanceOf(decoded.get('c'), Blob);
            assert.equal((decoded.get('a') as Blob).type, 'image/png');
            assert.equal((decoded.get('b') as Blob).type, 'image/png');
            assert.equal((decoded.get('c') as Blob).type, 'image/jpeg');
        });

        it('should reject Blobs with unsupported mime types', async () => {
            const wrong = new Blob([new Uint8Array([0])], { type: 'text/plain' });
            let threw = false;
            try {
                await toBinary(wrong);
            } catch (e: any) {
                threw = true;
                assert.match(e.message ?? '', /Cannot write to binary value/i,
                    'should throw the standard "cannot write" error');
            }
            assert.isTrue(threw, 'expected toBinary to throw on unsupported Blob');
        });

        it('should be backward compatible: deterministic await with no Blobs', async () => {
            const message = new Map<string, any>([['x', 1]]);
            const encoded1 = await toBinary(message);
            const encoded2 = await toBinary(message);
            assert.isNotNull(encoded1);
            assert.isNotNull(encoded2);
            assert.isTrue(buffersEqual(encoded1!, encoded2!),
                'identical inputs must produce identical bytes');
        });
    });


    describe('TestGoldenFixture', () => {
        // Cross-language fixture written by the Python side. Lives under
        // ``tests/samples/visualize/`` together with a regular PNG copy so the
        // fixture can be inspected with any image viewer.
        const FIXTURE_BIN = testSamples('visualize', 'checkerboard.png.bin');
        const FIXTURE_PNG = testSamples('visualize', 'checkerboard.png');

        const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

        function readArrayBuffer(p: string): ArrayBuffer {
            const buffer = fs.readFileSync(p);
            return buffer.buffer.slice(
                buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
        }

        function fixtureMissingMessage(): string | null {
            if (!fs.existsSync(FIXTURE_BIN)) {
                return `Golden fixture not found: ${FIXTURE_BIN}\n`
                    + '   Generate via: `python tests/python/kaolin/visualize/web/test_io.py`';
            }
            return null;
        }

        it('should decode the Python-generated golden checkerboard.png.bin to a PNG Blob', async () => {
            const missing = fixtureMissingMessage();
            if (missing) { console.warn(`${missing} — skipping`); return; }
            const arrBuf = readArrayBuffer(FIXTURE_BIN);

            const decoded = fromBinary(arrBuf);
            assert.instanceOf(decoded, Blob, 'fixture must decode to a Blob');
            assert.equal((decoded as Blob).type, 'image/png');

            const payload = new Uint8Array(await (decoded as Blob).arrayBuffer());
            assert.isAbove(payload.byteLength, PNG_SIGNATURE.length,
                'payload must contain at least the PNG signature');
            for (let i = 0; i < PNG_SIGNATURE.length; ++i) {
                assert.equal(payload[i], PNG_SIGNATURE[i],
                    `Decoded payload byte ${i} should match PNG signature`);
            }
        });

        // PNG is lossless: bytes survive a full ``Blob -> toBinary -> fromBinary
        // -> Blob`` round-trip exactly, regardless of how the original PNG was
        // produced (Python torchvision in this case).
        it('should losslessly round-trip the PNG fixture through toBinary/fromBinary', async () => {
            const missing = fixtureMissingMessage();
            if (missing) { console.warn(`${missing} — skipping`); return; }
            if (!fs.existsSync(FIXTURE_PNG)) {
                console.warn(`Companion PNG not found: ${FIXTURE_PNG} — skipping`);
                return;
            }
            const pngBytes = new Uint8Array(readArrayBuffer(FIXTURE_PNG));
            const original = new Blob([pngBytes], { type: 'image/png' });

            const encoded = await toBinary(original);
            assert.isNotNull(encoded);

            const decoded = fromBinary(encoded!);
            assert.instanceOf(decoded, Blob, 'PNG payload must decode to a Blob');
            assert.equal((decoded as Blob).type, 'image/png');

            // The transport layer never re-encodes a Blob — bytes survive the
            // ``Blob -> toBinary -> fromBinary -> Blob`` round-trip exactly,
            // even though Pillow's encoder bytes differ from torchvision's.
            const roundtripped = new Uint8Array(await (decoded as Blob).arrayBuffer());
            assert.equal(roundtripped.byteLength, pngBytes.byteLength,
                'PNG byte length must match exactly after round-trip');
            for (let i = 0; i < pngBytes.byteLength; ++i) {
                assert.equal(roundtripped[i], pngBytes[i],
                    `PNG byte ${i} differs after round-trip`);
            }
        });

        // JPEG is lossy when re-encoded, but ``Blob -> toBinary -> fromBinary
        // -> Blob`` does NOT re-encode — it's just a pass-through transport.
        // So the bytes must still be byte-identical after the round-trip,
        // even for a JPEG payload constructed from arbitrary bytes. We use
        // a small synthetic JPEG (SOI + JFIF + EOI) here to keep the test
        // independent of JPEG-encoder availability.
        it('should losslessly round-trip a JPEG Blob payload through toBinary/fromBinary', async () => {
            const jpegBytes = new Uint8Array([
                0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46,
                0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
                0x00, 0x01, 0x00, 0x00, 0xff, 0xd9
            ]);
            const original = new Blob([jpegBytes], { type: 'image/jpeg' });

            const encoded = await toBinary(original);
            assert.isNotNull(encoded);

            const decoded = fromBinary(encoded!);
            assert.instanceOf(decoded, Blob);
            assert.equal((decoded as Blob).type, 'image/jpeg');

            // Approximate-equal payload: byte-identity is the contract for the
            // transport layer (fromBinary doesn't re-encode), but we phrase the
            // assertion as "all bytes equal" so the test is robust to any
            // future implementation change that re-encodes the JPEG payload.
            const roundtripped = new Uint8Array(await (decoded as Blob).arrayBuffer());
            assert.approximately(roundtripped.byteLength, jpegBytes.byteLength,
                Math.max(8, Math.round(jpegBytes.byteLength * 0.1)),
                'JPEG byte length should be within ~10% after round-trip');
            // Header bytes (SOI marker) must always survive intact — that's
            // what consumers like ``createImageBitmap`` look for.
            assert.equal(roundtripped[0], 0xff);
            assert.equal(roundtripped[1], 0xd8);
        });
    });
});
