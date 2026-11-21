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
 * Shared client-side utilities shipped with Kaolin: canvas image manipulation, leveled
 * logging, and other helpers.
 *
 * Most useful utilities for custom apps:
 * - {@link canvas} — HTML canvas manipulation, including reading/writing pixels.
 * - {@link cursor} — pointer event to element-relative coordinate helpers.
 * - {@link file} — file helpers.
 * - {@link logging} — leveled console logger.
 * - {@link svg} — build and stamp configurable SVG primitives onto an SVG layer.
 * 
 * 
 * @groupDescription Internal Utilities
 * Framework plumbing not intended for typical viewer client code.
 *
 * @module
 */

export * as canvas from './canvas';
export * as cursor from './cursor';
export * as file from './file';
export * as fps from './fps';
export * as logging from './logging';
export * as registry from './registry';
export * as svg from './svg';
export * from './tab_session';
export * as types from './types';