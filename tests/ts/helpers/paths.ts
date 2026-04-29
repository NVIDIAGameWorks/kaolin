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
 * Filesystem-path helpers for locating committed test fixtures, so tests don't
 * hand-count `'..'` segments back to the repo root.
 *
 * Build paths from the repo root with {@link repoPath}, from the shared
 * cross-language fixtures with {@link testSamples}, from the dash TypeScript
 * sample tree with {@link dashTypescriptTestData}, or from bundled assets with
 * {@link sampleData}.
 *
 * @module
 */

import * as path from 'path';

/**
 * Absolute path to the repository root (the directory that contains `tests/`),
 * resolved from this file's location (`tests/ts/helpers/`).
 */
export const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');

/**
 * Join path segments onto {@link REPO_ROOT}.
 *
 * @param segments - Path segments relative to the repo root.
 * @returns Absolute path under the repo root.
 */
export function repoPath(...segments: string[]): string {
    return path.join(REPO_ROOT, ...segments);
}

/**
 * Join path segments onto `tests/samples/`, the home of committed test fixtures
 * (including cross-language goldens shared with the Python suite).
 *
 * @param segments - Path segments relative to `tests/samples/`.
 * @returns Absolute path under `tests/samples/`.
 */
export function testSamples(...segments: string[]): string {
    return repoPath('tests', 'samples', ...segments);
}

/**
 * Join path segments onto the dash TypeScript sample tree,
 * `tests/samples/visualize/dash/components/src/ts/`, which mirrors the
 * `src/ts/` source layout for per-module fixtures.
 *
 * @param segments - Path segments relative to the dash TypeScript sample tree
 *     (e.g. `'lib', 'behavior', 'drawing_golden.png'`).
 * @returns Absolute path under the dash TypeScript sample tree.
 */
export function dashTypescriptTestData(...segments: string[]): string {
    return testSamples('visualize', 'dash', 'components', 'src', 'ts', ...segments);
}

/**
 * Join path segments onto the repo's bundled `sample_data/` assets (meshes,
 * point clouds, etc.).
 *
 * @param segments - Path segments relative to `sample_data/` (e.g.
 *     `'meshes', 'fox.obj'`).
 * @returns Absolute path under `sample_data/`.
 */
export function sampleData(...segments: string[]): string {
    return repoPath('sample_data', ...segments);
}
