#!/usr/bin/env tsx
/**
 * Dump the library behavior manifest used by Python introspection.
 *
 * Run via `npm run build:dash:manifest` (or directly with `tsx`). Output:
 *   kaolin/visualize/dash/components/autogen/behavior_manifest.json
 *
 * The manifest is built entirely by {@link BehaviorRegister.toJSON} (the single
 * source of truth for its shape); this script only side-effect-imports the
 * registration-bearing barrels so every `BehaviorRegister.register(...)` call
 * runs, then writes the result to disk with deterministic key ordering. Adding
 * a new behavior anywhere under `kaolin/visualize/dash/components/src/ts/`
 * widens the manifest with no edits here.
 */

import { writeFileSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

// Side-effect imports: trigger every `BehaviorRegister.register(...)` call.
// We avoid the root barrel `src/ts/index.ts` because it pulls in
// `KaolinViewerInternal.tsx`, which imports a CSS asset that `tsx` cannot
// load. Importing the registration-bearing subbarrels directly is enough.
import '../src/ts/lib/behavior';
import '../src/ts/graphics';

import { BehaviorRegister } from '../src/ts/core/behavior';


/** Deterministic JSON serialization with sorted keys. */
function stableStringify(value: unknown, indent: number = 2): string {
    return JSON.stringify(value, (_key, val) => {
        if (val && typeof val === 'object' && !Array.isArray(val)) {
            const sorted: Record<string, unknown> = {};
            for (const k of Object.keys(val as Record<string, unknown>).sort()) {
                sorted[k] = (val as Record<string, unknown>)[k];
            }
            return sorted;
        }
        return val;
    }, indent) + '\n';
}


function main() {
    const here = dirname(fileURLToPath(import.meta.url));
    const outPath = resolve(here, '..', 'autogen', 'behavior_manifest.json');
    mkdirSync(dirname(outPath), { recursive: true });

    const manifest = BehaviorRegister.toJSON();
    writeFileSync(outPath, stableStringify(manifest));
    console.log(`Wrote ${Object.keys(manifest).length} behavior entries to ${outPath}`);
}

main();
