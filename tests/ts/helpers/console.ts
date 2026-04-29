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

/** console method used by {@link ConsoleLogger} routing. */
export type ConsoleMethod = 'log' | 'info' | 'warn' | 'error';

/** One captured call to a stubbed console method. */
export interface CapturedConsoleCall {
    method: ConsoleMethod;
    args: unknown[];
}

/** Result of {@link captureConsole}; call `restore()` when done. */
export interface ConsoleCapture {
    calls: CapturedConsoleCall[];
    restore: () => void;
}

/**
 * Replace `console.log/info/warn/error` with recorders that append to `calls`.
 * Always call `restore()` (e.g. in a `finally` block) to put console back.
 */
export function captureConsole(): ConsoleCapture {
    const calls: CapturedConsoleCall[] = [];
    const orig = {
        log: console.log,
        info: console.info,
        warn: console.warn,
        error: console.error,
    };
    console.log = (...args: unknown[]) => calls.push({ method: 'log', args });
    console.info = (...args: unknown[]) => calls.push({ method: 'info', args });
    console.warn = (...args: unknown[]) => calls.push({ method: 'warn', args });
    console.error = (...args: unknown[]) => calls.push({ method: 'error', args });
    return {
        calls,
        restore: () => {
            console.log = orig.log;
            console.info = orig.info;
            console.warn = orig.warn;
            console.error = orig.error;
        },
    };
}
