/**
 * @groupDescription Essential
 * Elements most useful for client-side development. 
 * @showGroups
 */


import { logger } from '../util/logging';
import { convertAllMapsToObjects, convertAllObjectsToMaps } from '../util/types';

/**
 * Downloads a file from URL.
 * @param filename - Name for downloaded file
 * @param url - URL to download from
 */
export function downloadURL(filename: string, url: string): void {
    const a = document.createElement("a");
    document.body.appendChild(a);
    a.style.display = "none";
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
}

/**
 * Type codes for binary serialization format.
 * 
 * Must match the Python `BinaryIoDataType` enum in `kaolin.visualize.web.io`.
 * @group Essential
 */
export enum BinaryIoDataType {
    INT8 = 0,
    UINT8 = 1,
    INT16 = 2,
    INT32 = 3,
    UINT32 = 4,
    INT64 = 5,
    FLOAT16 = 6,
    FLOAT32 = 7,
    FLOAT64 = 8,
    STRING = 9,
    DICT = 10,
    LIST = 11,
    PNG = 12,
    JPEG = 13,
    UNSUPPORTED = 100
}

/**
 * Internal shim used to carry a Blob's bytes through the synchronous
 * {@link BinaryWriter}. {@link toBinary} pre-resolves every Blob in the input
 * tree into one of these before invoking the writer.
 *
 * @internal
 */
type _PreloadedBlob = {
    __preloadedBlob: true;
    type: 'image/png' | 'image/jpeg';
    bytes: Uint8Array;
};

function _isPreloadedBlob(value: any): value is _PreloadedBlob {
    return value && typeof value === 'object' && (value as any).__preloadedBlob === true;
}

function _isNumberOrBool(value: any): Boolean {
    return (typeof value === 'number') || (typeof value === 'boolean');
}

function _isIntOrBool(value: any): Boolean {
    return (typeof value === 'number' && Number.isInteger(value)) || (typeof value === 'boolean');
}

const _PNG_MIME = 'image/png';
const _JPEG_MIME = 'image/jpeg';

/**
 * Walks `value` (Map/Array/object) and replaces every Blob whose mime type
 * is `image/png` or `image/jpeg` with a {@link _PreloadedBlob} shim
 * carrying the resolved bytes. Promise.all is used so multiple Blobs resolve
 * in parallel. Non-Blob nodes are returned by reference.
 *
 * Blobs with other mime types are left untouched here; they will be rejected
 * by {@link valueToType}/{@link BinaryWriter}.
 *
 * @internal
 */
async function _resolveBlobs(value: any): Promise<any> {
    if (value instanceof Blob) {
        if (value.type === _PNG_MIME || value.type === _JPEG_MIME) {
            const bytes = new Uint8Array(await value.arrayBuffer());
            return { __preloadedBlob: true, type: value.type, bytes } as _PreloadedBlob;
        }
        return value;
    }
    if (value instanceof Map) {
        const entries = await Promise.all(
            Array.from(value.entries()).map(async ([k, v]) => [k, await _resolveBlobs(v)] as [any, any])
        );
        return new Map(entries);
    }
    if (Array.isArray(value)) {
        return await Promise.all(value.map(_resolveBlobs));
    }
    // Plain objects (the dict-on-the-wire route once converted by
    // convertValueToSupportedFormat). Skip typed arrays / DataView (they're
    // ArrayBuffer views) so we don't accidentally reflect into binary buffers.
    if (value && typeof value === 'object'
        && !ArrayBuffer.isView(value) && !(value instanceof ArrayBuffer)) {
        // Plain Object literal — recurse into its enumerable string keys.
        // Note: convertValueToSupportedFormat normally turns plain objects
        // into Maps before serialization, so this branch handles the case
        // where toBinary is called directly on a literal containing Blobs.
        const proto = Object.getPrototypeOf(value);
        if (proto === Object.prototype || proto === null) {
            const out: Record<string, any> = {};
            const keys = Object.keys(value);
            const resolved = await Promise.all(keys.map(k => _resolveBlobs((value as any)[k])));
            for (let i = 0; i < keys.length; ++i) {
                out[keys[i]] = resolved[i];
            }
            return out;
        }
    }
    return value;
}

/**
 * For Kaolin conventions on client and server side, the key to include in the message
 * dictionary to attach a message tag. For example 'tag': 'render'.
 * @group Essential
 */
export const MESSAGE_TAG_KEY = 'tag';  // key for message tag in dictionary

/**
 * For Kaolin conventions on client and server side, the key to include in the message
 * dictionary to attach a the message content.
 * @group Essential
 */
export const MESSAGE_CONTENT_KEY = 'msg';


/**
 * Encode a tagged message for transport over the websocket / Dash bridge.
 *
 * Returns a `Promise` when `binary` is true (image bytes inside Blobs are
 * resolved asynchronously via {@link toBinary}). The JSON path stays sync.
 *
 * @group Essential
 */
export async function encodeMessage(tag: string, content: any, binary: boolean = true)
    : Promise<ArrayBuffer | string | null> {
    let msg = new Map<string, any>([
        [MESSAGE_TAG_KEY, tag],
        [MESSAGE_CONTENT_KEY, content]
    ]);
    if (binary) {
        return await toBinary(msg);
    } else {
        return toJSON(msg);
    }
}


export function binaryOffsetRequired(typeId: BinaryIoDataType | number): number {
    if (typeId == BinaryIoDataType.STRING
        || typeId == BinaryIoDataType.PNG
        || typeId == BinaryIoDataType.JPEG) {
        return 1;
    } else {
        return typedArrayConstructorFromType(typeId)[1];
    }
}

/**
 * Returns typed array constructor and bytes per element for given type ID.
 * @param typeId - Binary I/O data type ID
 * @returns [constructor, bytesPerElement]
 */
export function typedArrayConstructorFromType(typeId: number) : [any, number] {
    switch (typeId) {
        case BinaryIoDataType.INT8:
            return [Int8Array, 1];
        case BinaryIoDataType.UINT8:
            return [Uint8ClampedArray, 1];
        case BinaryIoDataType.INT16:
            return [Int16Array, 2];
        case BinaryIoDataType.INT32:
            return [Int32Array, 4];
        case BinaryIoDataType.UINT32:
            return [Uint32Array, 4];
        case BinaryIoDataType.INT64:
            return [BigInt64Array, 8];
//         case BinaryIoDataType.FLOAT16:  // TODO: typescript can't compile this
//             return [Float16Array, 2];
        case BinaryIoDataType.FLOAT32:
            return [Float32Array, 4];
        case BinaryIoDataType.FLOAT64:
            return [Float64Array, 8];
        default:
            logger.error(`Unknown array type ${typeId}`);
            return [null, 0];
        }
}

export function convertValueToSupportedFormat(value: any): any {
    // Pass image Blobs and pre-resolved blob shims through unchanged.
    if (value instanceof Blob || _isPreloadedBlob(value)) {
        return value;
    }

    // Handle JavaScript arrays
    if (Array.isArray(value)) {
        if (value.length === 0) {
            return [];  // Keep empty arrays as arrays
        }
        
        // Check if all elements are numbers
        const allNumbers = value.every(item => _isNumberOrBool(item));
        
        if (allNumbers) {
            // Check if all elements are integers
            const allIntegers = value.every(item => _isIntOrBool(item));
            
            if (allIntegers) {
                return new Int32Array(value);
            } else {
                return new Float32Array(value);
            }
        } else {
            // Mixed types - keep as regular array for LIST support
            return value;
        }
    }

    if (valueToType(value) != BinaryIoDataType.UNSUPPORTED) {
        return value;
    }

    // Handle numbers
    if (_isNumberOrBool(value)) {
        if (_isIntOrBool(value)) {
            return new Int32Array([value]);
        } else {
            return new Float32Array([value]);
        }
    }
    
    // Handle dictionaries
    if (value instanceof Map) {
        return value;
    } else if (typeof value === 'object' && value !== null) {
        return new Map(Object.entries(value));
    }

    // Unsupported types
    throw new Error(`Cannot encode value of type ${typeof value} to binary`);
}

export function valueToType(value: any): BinaryIoDataType {
    // Handle strings
    if (typeof value === 'string') {
        return BinaryIoDataType.STRING;
    }
    if (value instanceof Map) {
        return BinaryIoDataType.DICT;
    }
    if (Array.isArray(value)) {
        return BinaryIoDataType.LIST;
    }

    // Handle typed arrays
    if (value instanceof Int8Array) {
        return BinaryIoDataType.INT8;
    }
    if (value instanceof Uint8Array || value instanceof Uint8ClampedArray || value instanceof ArrayBuffer) {
        return BinaryIoDataType.UINT8;
    }
    if (value instanceof Int16Array) {
        return BinaryIoDataType.INT16;
    }
    if (value instanceof Int32Array) {
        return BinaryIoDataType.INT32;
    }
    if (value instanceof Uint32Array) {
        return BinaryIoDataType.UINT32;
    }
    if (value instanceof BigInt64Array) {
        return BinaryIoDataType.INT64;
    }
    if (value instanceof Float32Array) {
        return BinaryIoDataType.FLOAT32;
    }
    if (value instanceof Float64Array) {
        return BinaryIoDataType.FLOAT64;
    }

    // Image Blobs (PNG / JPEG). Other Blob mime types are not supported.
    if (value instanceof Blob) {
        if (value.type === _PNG_MIME)  return BinaryIoDataType.PNG;
        if (value.type === _JPEG_MIME) return BinaryIoDataType.JPEG;
        logger.error(`Unsupported Blob mime type '${value.type}'; only image/png and image/jpeg are supported`);
        return BinaryIoDataType.UNSUPPORTED;
    }
    if (_isPreloadedBlob(value)) {
        return value.type === _PNG_MIME ? BinaryIoDataType.PNG : BinaryIoDataType.JPEG;
    }

    return BinaryIoDataType.UNSUPPORTED;
}

/**
 * Creates typed array or string from binary data.
 *
 * @param binaryData - Input binary data
 * @param offset - Byte offset in data
 * @param length - Number of elements
 * @param typeId - Data type ID, respecting BinaryIoDataType
 * @returns [typedValue, byteLength]
 */
export function typedValueFromBinary(binaryData: ArrayBuffer, offset: number, length: number, typeId: number) :
[ArrayBuffer | string | number | Map<string, any> | any[] | Blob | null, number] {
    if (typeId == BinaryIoDataType.STRING) {
        return [stringFromBinary(binaryData, offset, length), length];
    } else if (typeId == BinaryIoDataType.DICT) {
        return _dictFromBinary(binaryData, offset, length);
    } else if (typeId == BinaryIoDataType.LIST) {
        return _listFromBinary(binaryData, offset, length);
    } else if (typeId == BinaryIoDataType.PNG || typeId == BinaryIoDataType.JPEG) {
        // Copy out so the resulting Blob does not retain a reference to the
        // larger transport ArrayBuffer.
        const bytes = new Uint8Array(binaryData, offset, length).slice();
        const mime = typeId == BinaryIoDataType.PNG ? _PNG_MIME : _JPEG_MIME;
        return [new Blob([bytes], { type: mime }), length];
    } else {
        const [typedArrayConstructor, bytesPerElement] = typedArrayConstructorFromType(typeId);
        if (typedArrayConstructor === null) {
            return [null, 0];
        }
        let gap = gapUntilNOffset(offset, bytesPerElement);
        const byteLength = length * bytesPerElement;
        return [new typedArrayConstructor(binaryData, offset + gap, length), byteLength + gap];
    }
}

/**
 * Decodes ascii string of specified length from binary data.
 *
 * @param binaryData - Input binary data
 * @param offset - Byte offset in data
 * @param length - String length in bytes
 * @returns Decoded string
 */
export function stringFromBinary(binaryData: ArrayBuffer, offset: number, length: number) : string {
    // TODO: check this; do we need to slice?
    const value = new TextDecoder().decode(binaryData.slice(offset, offset + length));
    return value;
}

/**
 * Encodes string to binary data using UTF-8 encoding.
 *
 * @param str - String to encode
 * @returns ArrayBuffer containing UTF-8 encoded bytes
 */
export function stringToBinary(str: string): ArrayBuffer {
    return new TextEncoder().encode(str).buffer;
}

/**
 * Parses dictionary from binary data, assuming specific layout:
 * - 4 bytes for int32 number of values
 * - the rest: subsequent encoding of all named values, as assumed by
 *             namedValueFromBinary
 *
 * @param binaryData - Input binary data
 * @param offset - Byte offset in data
 * @returns Map of name-value pairs, where value has .value and .shape
 */
export function _dictFromBinary(binaryData: ArrayBuffer, offset: number, numValues: number) : [Map<string, any>, number] {
    let readBytes = 0;

    const result = new Map<string, any>();
    for (let i = 0; i < numValues; ++i) {
        let [value, shape, newReadBytes, name] = namedValueFromBinary(binaryData, offset + readBytes);
        if (value === null) {
            logger.error(`Failed to read dictionary value with name ${name}; returning incomplete dict`);
            break;
        }
        if (result.has(name)) {
            logger.error(`Value clash for dictionary value with name ${name}; only keeping one value`);
        }
        result.set(name, value);
        readBytes += newReadBytes;
    }
    return [result, readBytes];
}

/**
 * Parses list from binary data, assuming specific layout:
 * - 4 bytes for int32 number of values
 * - the rest: subsequent encoding of all values, as assumed by valueFromBinary
 *
 * @param binaryData - Input binary data
 * @param offset - Byte offset in data
 * @param numValues - Number of values to read
 * @returns Array of values
 */
export function _listFromBinary(binaryData: ArrayBuffer, offset: number, numValues: number) : [any[], number] {
    let readBytes = 0;

    const result: any[] = [];
    for (let i = 0; i < numValues; ++i) {
        let [value, shape, newReadBytes] = valueFromBinary(binaryData, offset + readBytes);
        if (value === null) {
            logger.error(`Failed to read list value at index ${i}; returning incomplete list`);
            break;
        }
        result.push(value);
        readBytes += newReadBytes;
    }
    return [result, readBytes];
}

/**
 * Serialize a value to binary format. Supports most basic types, like
 * `Map` or standard javascript Object `{}`, `Array`, `ArrayBuffer`,
 * `TypedArray`, `number`, `string`. Nesting of `Map` and `Array` instances
 * is also supported. PNG/JPEG `Blob` instances (mime types `image/png` /
 * `image/jpeg`) are encoded as compressed-image payloads.
 *
 * Always returns a `Promise` because `Blob` bytes are only accessible
 * asynchronously. Internally, every `Blob` in the value tree is resolved
 * (in parallel via `Promise.all`) into a private byte shim before the
 * synchronous {@link BinaryWriter} runs. For values that contain no Blobs
 * this only adds a single microtask hop and the bytes are byte-identical
 * to a direct synchronous encoding.
 *
 * The resulting buffer can be decoded using {@link fromBinary} or by its
 * sister python implementation in `kaolin.visualize.web`.
 *
 * @param value - Value to serialize.
 * @returns Promise resolving to binary data as `ArrayBuffer`, or `null` on failure.
 * @group Essential
 */
export async function toBinary(value: any) : Promise<ArrayBuffer | null> {
    const resolved = await _resolveBlobs(value);
    let writer = new BinaryWriter();
    writer.needsToWriteValueWithMetadata(resolved);
    writer.allocate();
    writer.writeValueWithMetadata(resolved);
    return writer.getBuffer();
}

/**
 * Encodes value to JSON. Works even for `Map` objects.
 * 
 * @param value 
 * @returns encoded string
 * @group Essential
 */
export function toJSON(value: any) : string {
    return JSON.stringify(convertAllMapsToObjects(value));
}

/**
 * Deserialize a value from binary format, such as that written by {@link toBinary}, or
 * its sister python implementation in `kaolin.visualize.web`. All arrays of numbers
 * will be decoded to `TypedArray`, and only mixed type arrays will be decoded to
 * javascript `Array`. All `TypedArray` instances will have an extra `shape` member.
 * All dictionaries will be decoded to `Map`. PNG / JPEG image payloads come back
 * as plain `Blob` instances with the corresponding `image/png` / `image/jpeg`
 * mime type set, so callers can pipe them through `createImageBitmap`,
 * `URL.createObjectURL`, etc. This call stays synchronous even when the
 * payload is a compressed image.
 *
 * @param binaryData - Binary data to deserialize
 * @returns Deserialized value
 * @group Essential
 */
export function fromBinary(binaryData: ArrayBuffer) : any {
    let [value, shape, readBytes] = valueFromBinary(binaryData, 0);
    if (readBytes != binaryData.byteLength) {
        logger.warn(`Read ${readBytes} bytes, not full message length ${binaryData.byteLength}`);
    }
    return value;
}

/**
 * Parses JSON message, and ensures all dictionaries/objects are represented as `Maps`, to
 * be consistent with {@link fromBinary}.
 * 
 * @param data 
 * @returns Parsed value
 * @group Essential
 */
export function fromJSON(data: string) : any {
    const message = JSON.parse(data);
    return convertAllObjectsToMaps(message);
}

/**
 * Parses named value from binary data, assuming specific layout:
 * - 4 bytes for int32 name length
 * - name_length bytes for the ascii name of the value
 * - the rest is same layout as assumed by valueFromBinary
 *
 * @param binaryData - Input binary data
 * @param offset - Byte offset in data
 * @returns [value, shape, readBytes, name]
 */
export function namedValueFromBinary(binaryData: ArrayBuffer, offset: number = 0):
[ArrayBuffer | string | number | Map<string, any> | any[] | Blob | null, number[], number, string] {
    // Fast forward to suitable offset
    let readBytes = gapUntil4Offset(offset);
    const nameLength = new Int32Array(binaryData, offset + readBytes, 1);
    readBytes += 4;
    const name = stringFromBinary(binaryData, offset + readBytes, nameLength[0]);
    readBytes += nameLength[0];
    const [typedArray, shape, arrayReadBytes] = valueFromBinary(binaryData, offset + readBytes);
    readBytes += arrayReadBytes;
    return [typedArray, shape, readBytes, name];
}



/**
 * Parses value from binary data, assuming specific layout:
 * - 2 * 4 bytes for int32 meta (shape length and type code, see BinaryIoDataType)
 * - N * 4 bytes for N-dimensional int32 shape
 * - the rest for raw data
 *
 * @param binaryData - Input binary data
 * @param offset - Byte offset in data
 * @returns [value, shape, readBytes]
 */
export function valueFromBinary(binaryData: ArrayBuffer, offset: number = 0):
[ArrayBuffer|string|number|Map<string, any>|any[]|Blob|null, number[], number] {
    const metaLength = 2;
    let readBytes = gapUntil4Offset(offset);
    const meta = new Int32Array(binaryData, offset + readBytes, metaLength);
    readBytes += 4 * metaLength;
    const shapeLength = meta[0];
    const typeCode = meta[1];

    let totalLength: number = 0;
    let isPrimitive = false;
    let shape = [];
    if (shapeLength == 0) {
        totalLength = 1;  // primitive type
        isPrimitive = true;
    } else {
        shape = Array.from(new Int32Array(binaryData, offset + readBytes, shapeLength));
        readBytes += 4 * shapeLength;
        totalLength = shape.reduce((acc, curr) => acc * curr, 1);
    }

    let [typedValue, valueReadBytes] = typedValueFromBinary(binaryData, offset + readBytes, totalLength, typeCode);
    readBytes += valueReadBytes;

    if (shape.length > 1 && ArrayBuffer.isView(typedValue)) {  // TODO: hack to store shape metadata; fix later
        typedValue['shape'] = shape;
    }

    if (isPrimitive) {
        typedValue = typedValue[0];
    }
    return [typedValue, shape, readBytes];
}


export const isClientLittleEndian = (() => {
    const buffer = new ArrayBuffer(2);
    new DataView(buffer).setInt16(0, 256, true /* littleEndian */);
    // Int16Array uses the platform's endianness.
    return new Int16Array(buffer)[0] === 256;
})();


/**
 * Calculate bytes needed for raw string (no length prefix).
 * @param str - String to measure
 * @returns Number of bytes needed
 */
export function bytesForString(str: string): number {
    return new TextEncoder().encode(str).length;
}

/**
 * Arrays such as Int32Array cannot start at offsets that are not
 * a mulitple of 4. This helps us find the right offset.
 */
export function gapUntil4Offset(currentOffset: number): number {
    return gapUntilNOffset(currentOffset, 4);
}

export function gapUntilNOffset(currentOffset: number, N: number): number {
    return (N - (currentOffset % N)) % N;
}

export class BinaryWriter {
    private rawWriter: RawBinaryWriter | null = null;
    private capacityNeeded: number = 0;

    constructor() {
    }

    needsToWriteNamedValue(name: string, value: any): BinaryWriter {
        if (this.rawWriter !== null) {
            throw new Error('BinaryWriter: extra capacity requested after allocate is called');
        }

        // Calculate bytes needed for name (4 bytes for length + name length)
        this.needsToWriteInt32();
        this.capacityNeeded += bytesForString(name);
        // Add bytes for the value itself
        this.needsToWriteValueWithMetadata(value);
        return this;
    }

    needsToWriteInt32(): BinaryWriter {
        if (this.rawWriter !== null) {
            throw new Error('BinaryWriter: extra capacity requested after allocate is called');
        }

        this.capacityNeeded += gapUntil4Offset(this.capacityNeeded) + 4;
        return this;
    }

    needsToWriteFloat32(): BinaryWriter {
        if (this.rawWriter !== null) {
            throw new Error('BinaryWriter: extra capacity requested after allocate is called');
        }

        this.capacityNeeded += gapUntil4Offset(this.capacityNeeded) + 4;
        return this;
    }

    writeInt32(value: number): BinaryWriter {
        if (!this.rawWriter) {
            logger.error('Must call allocate first');
            return this;
        }

        this.rawWriter.fastForwardToSuitableOffset(4);
        this.rawWriter.writeInt32(value);
        return this;
    }

    writeFloat32(value: number): BinaryWriter {
        if (!this.rawWriter) {
            logger.error('Must call allocate first');
            return this;
        }

        this.rawWriter.fastForwardToSuitableOffset(4);
        this.rawWriter.writeFloat32(value);
        return this;
    }

    writeNamedValue(name: string, value: any): BinaryWriter {
        this.writeInt32(bytesForString(name));
        this.rawWriter.writeString(name);
        this.writeValueWithMetadata(value, value?.shape);
        return this;
    }

    needsToWriteValueWithMetadata(inputValue: any): BinaryWriter {
        let shape = null;
        if (ArrayBuffer.isView(inputValue) && inputValue['shape']) {  // TODO: hack to store shape metadata; fix later
            shape = inputValue['shape'];
        }
        return this._needsToWriteValueWithMetadata(inputValue, shape);
    }

    _needsToWriteValueWithMetadata(inputValue: any, shape: Int32Array | number[] | null = null): BinaryWriter {
        if (this.rawWriter !== null) {
            throw new Error('BinaryWriter: extra capacity requested after allocate is called');
        }

        const isPrimitive = _isNumberOrBool(inputValue);
        const value = convertValueToSupportedFormat(inputValue);
        const typeId = valueToType(value);
        if (typeId == BinaryIoDataType.UNSUPPORTED) {
            throw new Error('Cannot write to binary value of type ' + typeof value);
        }

        this.capacityNeeded += gapUntil4Offset(this.capacityNeeded);
        this.capacityNeeded += 8;  // type and shape length

        switch(typeId) {
            case BinaryIoDataType.STRING:
                this.capacityNeeded += 4;  // length
                this.capacityNeeded += bytesForString(value);
                break;
            case BinaryIoDataType.DICT:
                this.capacityNeeded += 4;  // length
                for (const [name, val] of (value as Map<string, any>)) {
                    // Calculate bytes for name (4 bytes for length + name length)
                    this.needsToWriteNamedValue(name, val);
                }
                break;
            case BinaryIoDataType.LIST:
                this.capacityNeeded += 4;  // length
                for (const val of (value as any[])) {
                    this.needsToWriteValueWithMetadata(val);
                }
                break;
            case BinaryIoDataType.PNG:
            case BinaryIoDataType.JPEG: {
                // shape header: 1 int32 carrying the payload byte length;
                // payload: pre-resolved Uint8Array bytes (1-byte aligned).
                this.capacityNeeded += 4;  // payload byte length stored as the 1-D shape
                const blob = value as _PreloadedBlob;
                this.capacityNeeded += blob.bytes.byteLength;
                break;
            }
            default:
                // All others are arrays
                let defaultShapeLength = isPrimitive ? 0 : 1;
                let intsForShape = (shape !== null && !isPrimitive) ? shape.length : defaultShapeLength;
                this.capacityNeeded += intsForShape * 4;
                this.capacityNeeded += gapUntilNOffset(this.capacityNeeded, binaryOffsetRequired(typeId));
                this.capacityNeeded += value.byteLength;
        }
        return this;
    }

    allocate(): BinaryWriter {
        // Check doesn't already have a rawWriter
        if (this.rawWriter !== null) {
            logger.warn("BinaryWriter already allocated. Cannot allocate twice.");
        } else {
            this.rawWriter = new RawBinaryWriter(this.capacityNeeded);
        }
        return this;
    }

    writeValueWithMetadata(inputValue: any, shape: Int32Array | number[] | null = null): BinaryWriter {
        if (!this.rawWriter) {
            logger.error('Must call allocate first');
            return this;
        }

        const isPrimitive = _isNumberOrBool(inputValue);
        let value = convertValueToSupportedFormat(inputValue);
        let typeId = valueToType(value);
        if (typeId == BinaryIoDataType.UNSUPPORTED) {
            throw new Error('Cannot write to binary value of type ' + typeof value);
        }
        let defaultShapeLength = isPrimitive ? 0 : 1;
        let intsForShape = (shape !== null && !isPrimitive) ? shape.length : defaultShapeLength;

        this.rawWriter.fastForwardToSuitableOffset(4);
        this.rawWriter.writeInt32(intsForShape);
        this.rawWriter.writeInt32(typeId);

        if (typeId == BinaryIoDataType.STRING) {
            let stringBytes = bytesForString(value);
            this.rawWriter.writeInt32(stringBytes);  // shape contains length
            this.rawWriter.writeString(value);
        } else if (typeId == BinaryIoDataType.PNG || typeId == BinaryIoDataType.JPEG) {
            const blob = value as _PreloadedBlob;
            this.rawWriter.writeInt32(blob.bytes.byteLength);  // 1-D shape = byte length
            this.rawWriter.writeTypedArray(blob.bytes);
        } else if (isPrimitive) {
            // Check if it's an integer
            if (_isIntOrBool(inputValue)) {
                this.rawWriter.writeInt32(Number(value));
            } else {
                this.rawWriter.writeFloat32(value);
            }
        } else if (typeId == BinaryIoDataType.DICT) {
            this.rawWriter.writeInt32(value.size);
            for (const [name, val] of (value as Map<string, any>)) {
                this.writeNamedValue(name, val);
            }
        } else if (typeId == BinaryIoDataType.LIST) {
            this.rawWriter.writeInt32(value.length);
            for (const val of (value as any[])) {
                this.writeValueWithMetadata(val);
            }
        } else if (value instanceof ArrayBuffer || (ArrayBuffer.isView(value) && !(value instanceof DataView))) {
            let bytesPerElement = typedArrayConstructorFromType(typeId)[1];
            let bytes = value.byteLength;
            let flattenedShape = Math.floor(value.byteLength / bytesPerElement);
            if (flattenedShape * bytesPerElement != value.byteLength) {
                logger.error(`Computed byte length is wrong for ${value}: ${flattenedShape} * ${bytesPerElement} != ${value.byteLength}`)
            }

            if (shape === null) {
                this.rawWriter.writeInt32(flattenedShape);
            } else {
                shape = Array.from(shape);
                if (shape.length < 1) {
                    this.rawWriter.writeInt32(flattenedShape);
                } else {
                    const inputFlattenedShape = shape.reduce((acc, curr) => acc * curr, 1);
                    if (inputFlattenedShape * bytesPerElement != value.byteLength) {
                        logger.error(`Incorrect shape ${shape}: ${inputFlattenedShape} * ${bytesPerElement} != ${value.byteLength}; ignoring provided shape`);
                    }
                    for (let i = 0; i < shape.length; ++i) {
                        this.rawWriter.writeInt32(shape[i]);
                    }
                }
            }
            this.rawWriter.fastForwardToSuitableOffset(bytesPerElement);
            if (value instanceof ArrayBuffer) {
                this.rawWriter.writeArrayBuffer(value);
            } else {
                this.rawWriter.writeTypedArray(value as any);
            }
        } else {
            logger.error(`Bug: Cannot write value (detected type ${typeId}): ${value}`);
        }
        return this;
    }



    getBuffer(): ArrayBuffer | null {
        if (this.rawWriter !== null) {
            return this.rawWriter.getBuffer();
        }
        return null;
    }
}


/**
 * BinaryWriter class for efficiently writing multiple data types to a pre-allocated buffer.
 * Encodes everything using default system endinanness.
 */
export class RawBinaryWriter {
    private buffer: ArrayBuffer;
    private offset: number;

    /**
     * Creates a new BinaryWriter with pre-allocated capacity.
     * @param capacity - Total bytes to allocate
     */
    constructor(capacity: number) {
        this.buffer = new ArrayBuffer(capacity);
        this.offset = 0;
    }

    /**
     * Get the current write offset.
     */
    getOffset(): number {
        return this.offset;
    }

    /**
     * Get the buffer trimmed to actual used size.
     */
    getBuffer(): ArrayBuffer {
        return this.buffer.slice(0, this.offset);
    }

    isWithinCapacity(bytesNeeded: number) {
        return this.offset + bytesNeeded <= this.buffer.byteLength;
    }

    assertWithinCapacity(bytesNeeded: number) {
        if (!this.isWithinCapacity(bytesNeeded)) {
            throw new Error('Trying to write bytes over capacity');
        }
    }

    fastForward(numBytes: number): RawBinaryWriter {
        this.offset += numBytes;
        return this;
    }

    fastForwardToSuitableOffset(multiple: number): RawBinaryWriter {
        let gap = gapUntilNOffset(this.offset, multiple);
        this.fastForward(gap);
        return this;
    }

    /**
     * Write a 32-bit signed integer (little-endian).
     * @param value - Integer value to write
     * @returns This RawBinaryWriter for chaining
     */
    writeInt32(value: number): RawBinaryWriter {
        this.assertWithinCapacity(4);
        // Note: we do not use DataView, because then encoding of the
        // arrays might end up being inconsistent compared to encoding
        // of the numbers. It will be easier to fix overall endinanness
        // than deal with inconsistent encodings.
        const meta = new Int32Array(this.buffer, this.offset, 1);
        meta[0] = value;
        this.offset += 4;
        return this;
    }

    /**
     * Write a 32-bit float (little-endian).
     * @param value - Float value to write
     * @returns This RawBinaryWriter for chaining
     */
    writeFloat32(value: number): RawBinaryWriter {
        this.assertWithinCapacity(4);
        // Note: we do not use DataView, because then encoding of the
        // arrays might end up being inconsistent compared to encoding
        // of the numbers. It will be easier to fix overall endinanness
        // than deal with inconsistent encodings.
        const meta = new Float32Array(this.buffer, this.offset, 1);
        meta[0] = value;
        this.offset += 4;
        return this;
    }

    /**
     * Write raw bytes from an ArrayBuffer.
     * @param data - ArrayBuffer to write
     * @returns This RawBinaryWriter for chaining
     */
    writeArrayBuffer(data: ArrayBuffer): RawBinaryWriter {
        this.assertWithinCapacity(data.byteLength);
        new Uint8Array(this.buffer, this.offset, data.byteLength).set(new Uint8Array(data));
        this.offset += data.byteLength;
        return this;
    }

    /**
     * Write raw string bytes without length prefix.
     * @param str - String to write
     * @returns This RawBinaryWriter for chaining
     */
    writeString(str: string): RawBinaryWriter {
        const stringBytes = new TextEncoder().encode(str);
        this.assertWithinCapacity(stringBytes.length);
        new Uint8Array(this.buffer, this.offset, stringBytes.length).set(stringBytes);
        this.offset += stringBytes.length;
        return this;
    }

    /**
     * Write a typed array (Int32Array, Float32Array, etc.).
     * @param array - Typed array to write
     * @returns This RawBinaryWriter for chaining
     */
    writeTypedArray(array: Int8Array | Uint8Array | Int16Array | Uint16Array | Int32Array | Uint32Array | Float32Array | Float64Array): RawBinaryWriter {
        return this.writeArrayBuffer((array.buffer as ArrayBuffer).slice(array.byteOffset, array.byteOffset + array.byteLength));
    }
}