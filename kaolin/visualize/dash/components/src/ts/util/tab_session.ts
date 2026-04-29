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


import { logger } from './logging';

/** Session storage key used to persist the per-tab UUID. @group Internal Utilities*/
export const TAB_STORAGE_KEY = 'kaolin_tab_uuid';

let _cached: string | null = null;

function _generateUuid(): string {
    // Prefer the standard secure-context API.
    const c = (globalThis as any).crypto;
    if (c && typeof c.randomUUID === 'function') {
        return c.randomUUID();
    }
    // Fallback for non-secure contexts: 32 hex chars laid out as a UUID.
    // Not cryptographically strong; only used as an opaque routing token.
    const hex = '0123456789abcdef';
    let s = '';
    for (let i = 0; i < 32; i++) {
        s += hex[Math.floor(Math.random() * 16)];
    }
    return (
        s.slice(0, 8) + '-' +
        s.slice(8, 12) + '-' +
        '4' + s.slice(13, 16) + '-' +    // version 4
        '8' + s.slice(17, 20) + '-' +    // variant
        s.slice(20, 32)
    );
}

/**
 * Return per-browser-tab UUID, minted on first access and persisted via sessionStorage.
 *
 * Why this exists:
 * - The Dash transport (HTTP `@callback`s) and the WebSocket transport never
 *   naturally meet on the server side. A single uuid that lives in the browser
 *   tab can be shared across both transports as a routing key, so server-side
 *   code can correlate `dcc.Store` updates with the per-tab
 *   `WebSocketHandlerManager` instance(s).
 *
 * Reachable from app code as `window.kaolin.util.getTabUuid()`.
 *
 * @returns A UUID v4 string unique to this browser tab session.
 * @group Internal Utilities
 */
export function getTabUuid(): string {
    if (_cached !== null) {
        return _cached;
    }
    let uuid: string | null = null;
    try {
        uuid = window.sessionStorage.getItem(TAB_STORAGE_KEY);
    } catch (e) {
        logger.warn(`tabSession: sessionStorage unavailable, generating ephemeral uuid`, e);
    }
    if (!uuid) {
        uuid = _generateUuid();
        try {
           // `sessionStorage` is scoped per tab in every modern browser: a new tab gets
           // a fresh uuid; a reload preserves the existing one.
           // One known wrinkle: Chrome's "Duplicate Tab" copies `sessionStorage`, which
           // means two tabs can briefly share a uuid. 
            window.sessionStorage.setItem(TAB_STORAGE_KEY, uuid);
        } catch (e) {
            logger.warn(`tabSession: failed to persist uuid to sessionStorage`, e);
        }
        logger.info(`tabSession: minted new tab uuid ${uuid}`);
    } else {
        logger.debug(`tabSession: reusing tab uuid ${uuid}`);
    }
    _cached = uuid;
    return uuid;
}

/**
 * Clear the in-memory tab-uuid cache so the next {@link getTabUuid} call re-reads
 * `sessionStorage` (or re-mints). Exists only to let tests simulate a fresh tab /
 * page reload; not part of the public API.
 *
 * @hidden
 */
export function __resetTabUuidCache(): void {
    _cached = null;
}
