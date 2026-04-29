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
 * Schema-driven options for behaviors configured for the Kaolin viewer.
 *
 * A behavior declares its option surface via a single `static schema = {...}`
 * declaration. The schema is the source of truth: default values are read off
 * it and a build-time dumper projects it into the JSON manifest Python uses for
 * automatic UI generation.
 *
 * Two shapes work together, mirroring the Python `option.py` dict-vs-class
 * split:
 * - {@link OptionSpecData} — the plain object shape used for *authoring*
 *   (`static schema = { name: { kind, ... } }`) and as the JSON wire format.
 * - {@link OptionSpec} — a validated, normalized runtime instance. Its
 *   {@link OptionSpec.toJSON} emits `OptionSpecData` (so `JSON.stringify` round-
 *   trips through {@link OptionSpec.fromJson}), parallel to the Python
 *   `OptionSpec.as_dict` / `OptionSpec.from_dict` pair.
 *
 * Options are flat: each is a single scalar value ({@link OptionKind}). Nested
 * or structured option groups are not supported — {@link normalizeSchema} drops
 * such fields with a warning, and constructing an `OptionSpec` of kind
 * {@link OptionKind.GROUP} throws.
 *
 * @module
 */

import { z } from 'zod';

import { logger } from '../../util/logging';


/**
 * Supported, finite set of option value kinds. `ANY` is the escape hatch for
 * values the auto-UI pipeline does not know how to render — the value is
 * passed through verbatim.
 *
 * Values must match the Python `OptionKind` enum in
 * `kaolin.visualize.dash.option`. `GROUP` exists for parity with Python (nested
 * groups), but is not representable on the TypeScript side: constructing an
 * {@link OptionSpec} with this kind throws.
 */
export enum OptionKind {
    INT = 'int',
    FLOAT = 'float',
    STRING = 'string',
    COLOR = 'color',
    BOOL = 'bool',
    ENUM = 'enum',
    GROUP = 'group',
    ANY = 'any',
}


/**
 * Plain object shape of a single behavior option, used for authoring a
 * `static schema` and as the JSON wire format (manifest entries). This is the
 * inert, unvalidated counterpart to the {@link OptionSpec} class — parallel to
 * a Python `OptionSpec` dict (`as_dict` / `from_dict`).
 *
 * This interface exists purely for TypeScript type-checking of authored
 * schemas and JSON payloads; it carries no behavior. Use the {@link OptionSpec}
 * class for validated, normalized runtime instances.
 *
 * Field names are kept in sync with the Python `OptionSpec` in
 * `kaolin.visualize.dash.option`.
 */
export interface OptionSpecData {
    /** Option name. Optional when authored as a value in a
     *  `{ name: OptionSpecData }` record (the record key supplies it); always
     *  present on the wire. */
    name?: string;
    /** Value kind, drives which auto-UI control is generated. */
    kind: OptionKind;
    /** Default value (stored in its JSON form). */
    default?: any;
    /** Human-readable label / tooltip; surfaced in the auto-UI. */
    description?: string;
    /** Numeric (`int`/`float`): optional bounds + step. */
    min?: number;
    max?: number;
    step?: number;
    /** Enum: the allowed value set. */
    values?: readonly any[];
    /** Whether the option should be accessible to users through the UI. */
    uiBound?: boolean;
}


/**
 * Validated, normalized runtime description of a single behavior option.
 *
 * Behavior options are flat: an `OptionSpec` describes one scalar value
 * (int / float / string / color / bool / enum / any). The {@link OptionKind.GROUP}
 * kind (nested groups) is not supported on the TypeScript side and throws on
 * construction.
 *
 * Mirrors the Python `OptionSpec` class: {@link OptionSpec.toJSON} /
 * {@link OptionSpec.fromJson} are the TypeScript analogues of `as_dict` /
 * `from_dict`.
 */
export class OptionSpec {
    /** Option name; matches its key within an authored `{ name: OptionSpecData }` record. */
    name: string;
    /** Value kind, drives which auto-UI control is generated. */
    kind: OptionKind;
    /** Default value (stored in its JSON form). */
    default?: any;
    /** Human-readable label / tooltip; surfaced in the auto-UI. */
    description?: string;
    /** Numeric (`int`/`float`): optional bounds + step. */
    min?: number;
    max?: number;
    step?: number;
    /** Enum: the allowed value set. */
    values?: readonly any[];
    /** Whether the option should be accessible to users through the UI. */
    uiBound: boolean;

    /**
     * Build a normalized option spec from a single init object.
     *
     * @param init - Option fields; `name` and `kind` are required. `uiBound`
     *   defaults to `true` so the emitted manifest is self-describing.
     * @throws If `kind` is {@link OptionKind.GROUP} (unsupported in TypeScript).
     */
    constructor(init: OptionSpecData & { name: string }) {
        if (init.kind === OptionKind.GROUP) {
            throw new Error(
                `OptionSpec "${init.name}": kind '${OptionKind.GROUP}' (nested group) `
                + `is not supported on the TypeScript side`);
        }
        this.name = init.name;
        this.kind = init.kind;
        this.default = init.default;
        this.description = init.description;
        this.min = init.min;
        this.max = init.max;
        this.step = init.step;
        this.values = init.values;
        this.uiBound = init.uiBound ?? true;
        this.validate();
    }

    /**
     * Warn (without throwing) about internally inconsistent fields, e.g. numeric
     * bounds on a non-numeric kind or an enum missing its `values`. Parallel to
     * the Python `OptionSpec.validate`, but soft: registration is never broken.
     */
    validate(): void {
        if (!this.name) {
            logger.warn(`OptionSpec: missing name (kind=${this.kind})`);
        }
        if (!(Object.values(OptionKind) as string[]).includes(this.kind)) {
            logger.warn(`OptionSpec "${this.name}": unknown kind '${this.kind}'`);
        }
        const numeric = this.kind === OptionKind.INT || this.kind === OptionKind.FLOAT;
        for (const attr of ['min', 'max', 'step'] as const) {
            if (this[attr] !== undefined && !numeric) {
                logger.warn(
                    `OptionSpec "${this.name}": "${attr}" is only meaningful for numeric `
                    + `kinds, not '${this.kind}'`);
            }
        }
        if (this.min !== undefined && this.max !== undefined && this.min > this.max) {
            logger.warn(`OptionSpec "${this.name}": min (${this.min}) > max (${this.max})`);
        }
        if (this.kind === OptionKind.ENUM) {
            if (!this.values || this.values.length === 0) {
                logger.warn(`OptionSpec "${this.name}": enum kind requires a non-empty "values" list`);
            }
        } else if (this.values !== undefined) {
            logger.warn(
                `OptionSpec "${this.name}": "values" is only meaningful for enum kind, `
                + `not '${this.kind}'`);
        }
    }

    /**
     * Emit the plain {@link OptionSpecData} wire form, omitting unset optional
     * fields. Automatically invoked by `JSON.stringify`, so serializing an
     * `OptionSpec` (e.g. when dumping the manifest) yields clean JSON that
     * {@link OptionSpec.fromJson} can round-trip. Parallel to Python `as_dict`.
     *
     * @returns The JSON-friendly option spec.
     */
    toJSON(): OptionSpecData {
        const out: OptionSpecData = { name: this.name, kind: this.kind };
        if (this.default !== undefined) out.default = this.default;
        if (this.description !== undefined) out.description = this.description;
        if (this.min !== undefined) out.min = this.min;
        if (this.max !== undefined) out.max = this.max;
        if (this.step !== undefined) out.step = this.step;
        if (this.values !== undefined) out.values = this.values;
        out.uiBound = this.uiBound;
        return out;
    }

    /**
     * Build an `OptionSpec` from its plain {@link OptionSpecData} form (the
     * inverse of {@link OptionSpec.toJSON}). Parallel to Python `from_dict`.
     *
     * @param data - A plain option spec; must carry a `name` and `kind`.
     * @returns The normalized, validated spec.
     */
    static fromJson(data: OptionSpecData): OptionSpec {
        return new OptionSpec({ ...data, name: data.name ?? '', kind: data.kind });
    }
}


/**
 * Build a plain `{ key: defaultValue }` object from an authored option-spec
 * record (keyed by option name). Used by plain-JS clients (e.g. an app's
 * `custom.js`) to seed a behavior's initial options from its declared defaults.
 *
 * @param schema - Authored `{ name: OptionSpecData }` record to read defaults from.
 * @returns An object mapping each option key to its declared default value.
 */
export function defaultsFromSchema(schema: Record<string, OptionSpecData>): Record<string, any> {
    const out: Record<string, any> = {};
    for (const key of Object.keys(schema)) {
        out[key] = schema[key].default;
    }
    return out;
}


// --- Sugar helpers for the few cases zod doesn't cover natively ---

/**
 * Zod string subtype that the auto-UI renders as a color picker.
 *
 * @returns A Zod string schema tagged with `kind: 'color'` metadata.
 */
export const zColor = () => z.string().meta({ kind: 'color' });


/** Auto-UI escape hatch: arbitrary value, not surfaced in UI by default. */
// export const zAny = <T>() =>
//       z.any().meta({ kind: 'any', uiBound: false });


/** Map a single JSON-Schema property (output of `z.toJSONSchema`) to one
 *  nameless {@link OptionSpecData} (the caller fills `name` from the schema
 *  key), or `null` if its kind is outside our limited flat vocabulary. */
function jsonPropToOptionSpec(p: Record<string, any>): Omit<OptionSpecData, 'name'> | null {
    const base: Partial<OptionSpecData> = {};
    if (p.default !== undefined) base.default = p.default;
    if (p.description !== undefined) base.description = p.description;
    if (p.uiBound !== undefined) base.uiBound = p.uiBound;

    let kind: OptionKind | null = null;
    const extra: Partial<OptionSpecData> = {};
    if (p.kind === 'any') kind = OptionKind.ANY;
    else if (p.kind === 'color') kind = OptionKind.COLOR;
    else if (Array.isArray(p.enum)) { kind = OptionKind.ENUM; extra.values = p.enum; }
    else if (p.type === 'integer' || p.type === 'number') {
        kind = p.type === 'integer' ? OptionKind.INT : OptionKind.FLOAT;
        if (p.minimum !== undefined) extra.min = p.minimum;
        if (p.maximum !== undefined) extra.max = p.maximum;
        if (p.multipleOf !== undefined) extra.step = p.multipleOf;
    }
    else if (p.type === 'string') kind = OptionKind.STRING;
    else if (p.type === 'boolean') kind = OptionKind.BOOL;
    return kind ? { kind, ...base, ...extra } : null;
}

/** Returns `true` if a JSON-Schema property describes a nested or otherwise
 *  structured value (object, array, or a composite `anyOf`/`oneOf`/`allOf`)
 *  rather than a flat scalar. Behavior options are flat-only. */
function isStructuredProp(p: Record<string, any>): boolean {
    return !!p && (p.type === 'object' || p.type === 'array'
        || Array.isArray(p.anyOf) || Array.isArray(p.oneOf) || Array.isArray(p.allOf));
}

/** Convert a `z.object({...})` schema into validated {@link OptionSpec}
 *  instances keyed by name. Behavior options are flat: properties whose zod
 *  type does not map to a supported {@link OptionKind} are dropped with a
 *  warning, and nested or structured properties get a specific flat-only
 *  warning. Uses `z.toJSONSchema` as the building block so metadata attached
 *  via `.meta({...})` (e.g. `kind: 'color' | 'any'`, `uiBound`) and constraints
 *  (`min`/`max`/`step`, defaults, descriptions) flow through automatically. */
const zodToOptionsSchema = (schema: z.ZodObject<any>): Record<string, OptionSpec> => {
    const json = z.toJSONSchema(schema) as { properties?: Record<string, any> };
    const out: Record<string, OptionSpec> = {};
    for (const [k, p] of Object.entries(json.properties ?? {})) {
        const data = jsonPropToOptionSpec(p);
        if (data) {
            out[k] = new OptionSpec({ ...data, name: k });
        } else if (isStructuredProp(p)) {
            logger.warn(`normalizeSchema: option "${k}" uses a nested/structured schema, `
                + `which is not supported (behavior options are flat); dropping it`);
        } else {
            logger.warn(`normalizeSchema: dropped option "${k}" with unsupported type`);
        }
    }
    return out;
};

/**
 * Normalize a behavior's `static schema` declaration into validated
 * {@link OptionSpec} instances keyed by name.
 *
 * Accepts either a Zod object schema or a plain `{ name: OptionSpecData }`
 * record, and returns one normalized `OptionSpec` per option (with `uiBound`
 * filled in).
 *
 * @param input - The behavior's `static schema`: a Zod object, a plain
 *   `{ name: OptionSpecData }` record, or a nullish/invalid value.
 * @returns The normalized specs keyed by name, or `null` if no usable schema
 *   was provided.
 */
export const normalizeSchema = (input: any): Record<string, OptionSpec> | null => {
    if (input == null) return null;
    if (input instanceof z.ZodObject) return zodToOptionsSchema(input);
    if (typeof input !== 'object') return null;
    const out: Record<string, OptionSpec> = {};
    for (const key of Object.keys(input)) {
        const data = input[key] as OptionSpecData;
        out[key] = new OptionSpec({ ...data, name: data.name ?? key });
    }
    return out;
};


/** Plain JSON form of a single behavior's manifest entry — the inert,
 *  serializable counterpart to {@link BehaviorMeta}. */
export interface BehaviorMetaData {
    /** Human-readable behavior description, or `null` if none. */
    description: string | null;
    /** Option specs keyed by name (each a plain {@link OptionSpecData}). */
    options: Record<string, OptionSpecData>;
}


/**
 * Metadata recorded for each registered behavior: its human-readable
 * description and normalized option specs (keyed by name). Drives the
 * schema-based UI and the Python-side manifest; {@link BehaviorMeta.toJSON} /
 * {@link BehaviorMeta.fromJson} round-trip through the {@link BehaviorMetaData}
 * wire form, parallel to the Python `BehaviorMeta.as_dict` / `from_dict`.
 *
 * TODO: extend both this class and the Python `BehaviorMeta` with the
 * behavior's interface flags once the dumper can emit them, e.g.
 * `isElementBound` / `isMessageHandler` / `isCameraController`.
 *
 * @group Behavior Registration
 */
export class BehaviorMeta {
    description: string | null;
    options: Record<string, OptionSpec> | Record<string, never>;

    /**
     * @param description - Human-readable behavior description, or `null` if none.
     * @param options - Normalized {@link OptionSpec} instances keyed by name, or `{}` if the behavior has none.
     */
    constructor(
        description: string | null = null,
        options: Record<string, OptionSpec> | Record<string, never> = {},
    ) {
        this.description = description;
        this.options = options;
    }

    /**
     * Project this metadata into its plain JSON form: each {@link OptionSpec} is
     * serialized to {@link OptionSpecData} via {@link OptionSpec.toJSON}. Invoked
     * automatically by `JSON.stringify`.
     *
     * @returns The serializable behavior metadata.
     */
    toJSON(): BehaviorMetaData {
        const options: Record<string, OptionSpecData> = {};
        for (const [name, spec] of Object.entries(this.options)) {
            options[name] = (spec as OptionSpec).toJSON();
        }
        return { description: this.description, options };
    }

    /**
     * Build a `BehaviorMeta` from its plain {@link BehaviorMetaData} form (the
     * inverse of {@link BehaviorMeta.toJSON}). Parallel to Python `from_dict`.
     *
     * @param data - A plain behavior-meta entry.
     * @returns The metadata with options rebuilt as {@link OptionSpec} instances.
     */
    static fromJson(data: BehaviorMetaData): BehaviorMeta {
        const options: Record<string, OptionSpec> = {};
        for (const [name, spec] of Object.entries(data.options ?? {})) {
            options[name] = OptionSpec.fromJson({ ...spec, name: spec.name ?? name });
        }
        return new BehaviorMeta(data.description ?? null, options);
    }
}