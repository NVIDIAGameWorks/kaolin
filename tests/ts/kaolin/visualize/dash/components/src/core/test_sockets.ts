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

// NOTE: these tests drive the page URL through happy-dom's `setURL`, since the
// functions under test read `window.location.protocol` and `.host`.
//
// The upgrade is now unconditional: on a secure page every address is rewritten,
// sentinel or not, where previously only the WINDOW_LOCATION sentinel influenced
// the scheme. The two surfaces therefore split cleanly, and are tested separately:
//   - `upgradeUrlScheme` decides the scheme, from the page's protocol alone.
//   - `resolveWindowLocation` substitutes the sentinel host, then always defers
//     to the above.
//
// Both `upgradeUrlScheme` and `appendTabUuid` report a bad address through the
// logger, so those two tests run under {@link capturingAtDebug}, which drops to
// DEBUG so the warning is not suppressed and then puts the previous level back.
// The log level is global, so every change to it here is saved and restored;
// leaving one behind would silently change what later tests log.
// `appendTabUuid` is made deterministic by seeding sessionStorage with a known
// uuid and clearing the module cache, rather than by stubbing `getTabUuid`.
//
// WebSocketConnectionsManager is driven against {@link FakeWebSocket} installed
// over the global, so no network is involved and tests can fire the handshake
// and inbound events by hand. Its own describe pins the log level to ERROR,
// because `addConnection` logs at INFO on every call and would otherwise bury
// the test output; ERROR still lets the two error assertions through.

import { assert } from 'chai';
import { registerDom, unregisterDom } from '@test/helpers/dom';
import { captureConsole, ConsoleMethod } from '@test/helpers/console';
import { setLogLevel, getLogLevel, LogLevel } from '@kaolin/util/logging';
import { TAB_STORAGE_KEY, getTabUuid, __resetTabUuidCache } from '@kaolin/util/tab_session';
import {
    resolveWindowLocation, upgradeUrlScheme, appendTabUuid, WebSocketConnectionsManager,
} from '@kaolin/core/sockets';

const HTTPS_PAGE = 'https://myhost:8443/app';
const HTTP_PAGE = 'http://plainhost:8080/app';
const TAB_UUID = 'test-uuid-1234';

/** Serve the page from `pageUrl`; both the sentinel and the upgrade read window.location. */
function onPage(pageUrl: string): void {
    (globalThis as any).happyDOM.setURL(pageUrl);
}

/** Make `getTabUuid` deterministic, so a socket's `?tab=` can be asserted exactly. */
function seedTabUuid(): void {
    onPage(HTTPS_PAGE);
    window.sessionStorage.setItem(TAB_STORAGE_KEY, TAB_UUID);
    __resetTabUuidCache();
    // Warm the cache here so the one-off "reusing tab uuid" line does not land in
    // the middle of the test output.
    capturing(() => getTabUuid());
}

/** Serve the page from `pageUrl`, then assert what `fn(input)` returns. */
function checkOnPage(pageUrl: string, fn: (s: string) => string,
                     input: string, expected: string, label: string): void {
    onPage(pageUrl);
    assert.equal(fn(input), expected, label);
}

const checkUpgraded = (pageUrl: string, input: string, expected: string, label: string) =>
    checkOnPage(pageUrl, upgradeUrlScheme, input, expected, label);

const checkResolved = (pageUrl: string, input: string, expected: string, label: string) =>
    checkOnPage(pageUrl, resolveWindowLocation, input, expected, label);

/** Run `body` with console captured, returning the calls it made. */
function capturing(body: () => void): ReturnType<typeof captureConsole>['calls'] {
    const { calls, restore } = captureConsole();
    try {
        body();
    } finally {
        restore();
    }
    return calls;
}

/**
 * Run `body` at DEBUG with console captured, so a logged warning is neither suppressed nor
 * printed. The level is global, so it is saved and restored: leaving DEBUG behind would
 * change what later tests log and make them order-dependent.
 */
function capturingAtDebug(body: () => void): ReturnType<typeof captureConsole>['calls'] {
    const previousLevel = getLogLevel();
    setLogLevel(LogLevel.DEBUG);
    try {
        return capturing(body);
    } finally {
        setLogLevel(previousLevel);
    }
}

/**
 * Assert that `calls` holds exactly one diagnostic, at `method`, optionally naming
 * `mentions`.
 *
 * Deliberately does NOT match the wording. What a diagnostic owes its reader is a severity
 * and the offending value; the prose around that is free to change, and asserting on it only
 * makes the suite fail when someone improves a message. `mentions` is therefore passed the
 * value that was interpolated in — an input or an identifier — never explanatory text.
 */
function checkLogged(calls: ReturnType<typeof captureConsole>['calls'], method: ConsoleMethod,
                     label: string, mentions?: string): void {
    assert.equal(calls.length, 1, `${label}: exactly one message logged`);
    assert.equal(calls[0].method, method, `${label}: routed to console.${method}`);
    if (mentions !== undefined) {
        assert.include(String(calls[0].args[0]), mentions,
            `${label}: message names "${mentions}", so the reader can see what failed`);
    }
}

/**
 * Stand-in for the browser WebSocket, installed over the global so the manager
 * constructs these instead of dialing out. Records every instance, and exposes the
 * inbound events as plain calls so a test can drive the handshake itself.
 */
class FakeWebSocket {
    static OPEN = 1;
    static CLOSED = 3;
    static instances: FakeWebSocket[] = [];

    readyState = 0;
    closeCalls = 0;
    onopen: (() => void) | null = null;
    onmessage: ((event: unknown) => void) | null = null;
    onerror: ((event: unknown) => void) | null = null;
    onclose: (() => void) | null = null;

    constructor(public url: string) {
        FakeWebSocket.instances.push(this);
    }

    close(): void {
        this.closeCalls += 1;
        this.readyState = FakeWebSocket.CLOSED;
    }

    /** The socket most recently constructed by the code under test. */
    static last(): FakeWebSocket {
        return FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
    }

    /** Complete the handshake the way a server would, so the manager marks it open. */
    fireOpen(): void {
        this.readyState = FakeWebSocket.OPEN;
        this.onopen!();
    }
}

/** Wipe the manager's static maps, which otherwise leak between tests. */
function resetManager(): void {
    const statics = WebSocketConnectionsManager as any;
    for (const map of [statics._registry, statics._isOpen, statics._idToAddress,
        statics._subscribers]) {
        map.clear();
    }
}

/**
 * Subscribe under `id`, appending a `<name>:<event>` line to `log` for every callback,
 * so fan-out, ordering and delivery counts are all visible in one assertion.
 */
function subscribe(log: string[], name: string, address: string, id: string = name): () => void {
    return WebSocketConnectionsManager.subscribeToConnection(
        address, 'main-ws', id,
        () => log.push(`${name}:open`),
        (event) => log.push(`${name}:msg(${(event as MessageEvent).data})`),
        () => log.push(`${name}:error`),
        () => log.push(`${name}:close`));
}

describe('visualize/dash/components/src/core/test_sockets.ts', () => {
    before(() => registerDom());
    after(async () => await unregisterDom());

    describe('upgradeUrlScheme', () => {
        it('strengthens insecure schemes on a secure page, and changes nothing on a plain one', () => {
            // The page protocol is the only input: an https page upgrades both transports,
            // and an address already secure is left as-is.
            checkUpgraded(HTTPS_PAGE, 'ws://10.0.0.4:9000/api', 'wss://10.0.0.4:9000/api',
                'ws upgraded to wss on an https page');
            checkUpgraded(HTTPS_PAGE, 'http://10.0.0.4:9000/api', 'https://10.0.0.4:9000/api',
                'http upgraded to https on an https page');
            checkUpgraded(HTTPS_PAGE, 'wss://10.0.0.4:9000/api', 'wss://10.0.0.4:9000/api',
                'wss already secure, so unchanged');
            checkUpgraded(HTTPS_PAGE, 'https://10.0.0.4:9000/api', 'https://10.0.0.4:9000/api',
                'https already secure, so unchanged');

            // Rewriting round-trips through the URL parser, so the result comes back
            // normalized: a host-only address gains the empty path.
            checkUpgraded(HTTPS_PAGE, 'ws://10.0.0.4:9000', 'wss://10.0.0.4:9000/',
                'upgraded address is returned URL-normalized, with a trailing slash');

            // A plain page returns the string untouched, before any parsing, so an explicit
            // wss survives and the normalization above does not apply.
            checkUpgraded(HTTP_PAGE, 'ws://10.0.0.4:9000/api', 'ws://10.0.0.4:9000/api',
                'ws left alone on an http page');
            checkUpgraded(HTTP_PAGE, 'wss://10.0.0.4:9000/api', 'wss://10.0.0.4:9000/api',
                'wss never downgraded to match an http page');
            checkUpgraded(HTTP_PAGE, 'ws://10.0.0.4:9000', 'ws://10.0.0.4:9000',
                'no parsing on an http page, so no normalization either');
        });

        it('warns about an unparseable address and passes it through', () => {
            onPage(HTTPS_PAGE);
            let upgraded: string;
            const calls = capturingAtDebug(() => {
                upgraded = upgradeUrlScheme('not a url');
            });
            assert.equal(upgraded!, 'not a url', 'unparseable address is returned unchanged');
            checkLogged(calls, 'warn', 'unparseable address', 'not a url');
        });
    });

    describe('resolveWindowLocation', () => {
        it('fills the sentinel with the page host, then upgrades the scheme either way', () => {
            // 'ws://WINDOW_LOCATION/websocket/' is the exact address every app passes to
            // add_websocket_connection, so this pair is the one that matters: the sentinel
            // must reach wss on an https page, or the browser blocks it as mixed content.
            checkResolved(HTTPS_PAGE, 'ws://WINDOW_LOCATION/websocket/',
                'wss://myhost:8443/websocket/',
                'sentinel takes the page host AND is upgraded to wss on an https page');
            checkResolved(HTTP_PAGE, 'ws://WINDOW_LOCATION/websocket/',
                'ws://plainhost:8080/websocket/',
                'same address stays plain ws when the page itself is http');

            // Reaching wss above also proves the substitution ran before the upgrade: parsing
            // first would have lowercased the host to 'window_location' and never matched.
            checkResolved(HTTPS_PAGE, 'wss://WINDOW_LOCATION/websocket/',
                'wss://myhost:8443/websocket/',
                'an already-secure sentinel resolves to the same place');
            checkResolved(HTTPS_PAGE, 'WINDOW_LOCATION/websocket/', 'myhost:8443/websocket/',
                'sentinel still supplies no scheme when the caller omits one, so the upgrade '
                + 'has no transport to strengthen');

            // Anything without the sentinel goes straight to upgradeUrlScheme, so the page
            // protocol decides, and the host is left exactly as the caller gave it.
            checkResolved(HTTPS_PAGE, 'ws://10.0.0.4:9000/api', 'wss://10.0.0.4:9000/api',
                'explicit remote address is upgraded on an https page');
            checkResolved(HTTP_PAGE, 'ws://10.0.0.4:9000/api', 'ws://10.0.0.4:9000/api',
                'explicit remote address untouched on an http page');
        });
    });

    describe('appendTabUuid', () => {
        before(() => seedTabUuid());

        it('adds the tab uuid without disturbing an existing query, and lets a caller override it', () => {
            assert.equal(appendTabUuid('wss://myhost:8443/websocket/'),
                `wss://myhost:8443/websocket/?tab=${TAB_UUID}`,
                'tab uuid appended to a bare address');
            assert.equal(appendTabUuid('wss://myhost:8443/ws?foo=1'),
                `wss://myhost:8443/ws?foo=1&tab=${TAB_UUID}`,
                'existing query parameters are preserved');
            assert.equal(appendTabUuid('wss://myhost:8443/ws?tab=custom'),
                'wss://myhost:8443/ws?tab=custom',
                'a tab already set by the caller wins');
            assert.equal(appendTabUuid(appendTabUuid('wss://myhost:8443/ws')),
                `wss://myhost:8443/ws?tab=${TAB_UUID}`,
                'idempotent, so re-appending does not duplicate the parameter');
        });

        it('warns and passes the address through when it cannot be parsed', () => {
            let appended: string;
            const calls = capturingAtDebug(() => {
                appended = appendTabUuid('not a url');
            });
            assert.equal(appended!, 'not a url', 'unparseable address is returned unchanged');
            checkLogged(calls, 'warn', 'unparseable WS address', 'not a url');
        });
    });

    describe('WebSocketConnectionsManager', () => {
        // The address every app passes to add_websocket_connection, and what it becomes on
        // an https page. Using the real one keeps the sentinel and the upgrade in the loop.
        const APP_ADDRESS = 'ws://WINDOW_LOCATION/websocket/';
        const RESOLVED = 'wss://myhost:8443/websocket/';

        let realWebSocket: unknown;
        let realLogLevel: LogLevel;

        beforeEach(() => {
            realWebSocket = (globalThis as any).WebSocket;
            realLogLevel = getLogLevel();
            setLogLevel(LogLevel.ERROR);
            (globalThis as any).WebSocket = FakeWebSocket;
            FakeWebSocket.instances = [];
            resetManager();
            seedTabUuid();
        });

        afterEach(() => {
            resetManager();
            (globalThis as any).WebSocket = realWebSocket;
            setLogLevel(realLogLevel);
        });

        it('opens one socket per address, fans events out, and forgets it all on close', () => {
            const log: string[] = [];
            WebSocketConnectionsManager.addConnection(APP_ADDRESS, 'main-ws');

            assert.equal(FakeWebSocket.instances.length, 1, 'exactly one socket opened');
            assert.equal(FakeWebSocket.last().url, `${RESOLVED}?tab=${TAB_UUID}`,
                'socket dials the resolved address carrying the tab uuid');
            assert.isTrue(WebSocketConnectionsManager.hasConnection('main-ws'),
                'found by its short identifier');
            assert.isTrue(WebSocketConnectionsManager.hasConnection(APP_ADDRESS),
                'found by the unresolved address the caller used, uuid not required');
            assert.isFalse(WebSocketConnectionsManager.isConnectionOpen('main-ws'),
                'not open until the handshake completes');
            assert.isUndefined(WebSocketConnectionsManager.getOpenConnection('main-ws'),
                'no open socket to hand out yet');

            WebSocketConnectionsManager.addConnection(APP_ADDRESS, 'main-ws');
            assert.equal(FakeWebSocket.instances.length, 1,
                'adding the same address again reuses the socket rather than dialing twice');

            subscribe(log, 'A', APP_ADDRESS);
            subscribe(log, 'B', APP_ADDRESS);
            assert.deepEqual(log, [], 'nothing delivered while the socket is still connecting');

            FakeWebSocket.last().fireOpen();
            assert.deepEqual(log, ['A:open', 'B:open'], 'open reaches every subscriber');
            assert.isTrue(WebSocketConnectionsManager.isConnectionOpen('main-ws'),
                'reported open once the handshake lands');
            assert.equal(WebSocketConnectionsManager.getOpenConnection('main-ws'),
                FakeWebSocket.last() as unknown as WebSocket,
                'the open socket is now the one handed out');

            log.length = 0;
            FakeWebSocket.last().onmessage!({ data: 'hi' });
            assert.deepEqual(log, ['A:msg(hi)', 'B:msg(hi)'], 'message reaches every subscriber');

            log.length = 0;
            FakeWebSocket.last().onerror!({});
            assert.deepEqual(log, ['A:error', 'B:error'], 'error reaches every subscriber');

            const socket = FakeWebSocket.last();
            WebSocketConnectionsManager.closeConnection('main-ws');
            assert.equal(socket.closeCalls, 1, 'a socket still open is actually closed');

            // Closing drops the alias, so the identifier is no longer recognised as one and
            // falls through to being retried as an address. That warns, which this describe's
            // ERROR level suppresses; the warning itself is asserted under upgradeUrlScheme.
            assert.isFalse(WebSocketConnectionsManager.hasConnection('main-ws'),
                'identifier no longer resolves');
            assert.isFalse(WebSocketConnectionsManager.hasConnection(APP_ADDRESS),
                'address no longer resolves');
            assert.isUndefined(WebSocketConnectionsManager.getConnection(APP_ADDRESS),
                'socket is no longer retrievable');
        });

        it('replays open to a late subscriber, ignores a duplicate id, and stops on unsubscribe', () => {
            const log: string[] = [];
            const unsubscribeA = subscribe(log, 'A', APP_ADDRESS);
            FakeWebSocket.last().fireOpen();

            log.length = 0;
            subscribe(log, 'C', APP_ADDRESS);
            assert.deepEqual(log, ['C:open'],
                'a subscriber joining after the handshake is told immediately, not left waiting');

            log.length = 0;
            subscribe(log, 'A-replacement', APP_ADDRESS, 'A');
            assert.deepEqual(log, [], 'a repeat of an existing id gets no replay');

            log.length = 0;
            FakeWebSocket.last().onmessage!({ data: 'm' });
            assert.deepEqual(log, ['A:msg(m)', 'C:msg(m)'],
                'the id keeps its original callbacks, and nobody is served twice');

            log.length = 0;
            unsubscribeA();
            FakeWebSocket.last().onmessage!({ data: 'n' });
            assert.deepEqual(log, ['C:msg(n)'], 'unsubscribing stops delivery for that one only');

            log.length = 0;
            FakeWebSocket.last().onclose!();
            assert.deepEqual(log, ['C:close'], 'a server-side close notifies whoever remains');
            assert.isFalse(WebSocketConnectionsManager.isConnectionOpen('main-ws'),
                'manager stops reporting it open when the server hangs up');
        });

        it('refuses to point an existing identifier at a second address', () => {
            WebSocketConnectionsManager.addConnection('ws://one.example:1/ws', 'shared-id');
            const calls = capturing(() => {
                WebSocketConnectionsManager.addConnection('ws://two.example:2/ws', 'shared-id');
            });

            checkLogged(calls, 'error', 'identifier clash', 'shared-id');
            assert.equal(WebSocketConnectionsManager.getConnection('shared-id'),
                FakeWebSocket.instances[0] as unknown as WebSocket,
                'identifier still resolves to the address that claimed it first');
            assert.equal(FakeWebSocket.instances.length, 2,
                'the second socket is still opened; only the alias is refused');
        });

        it('resolves a lone connection with no argument, but refuses to guess between two', () => {
            WebSocketConnectionsManager.addConnection('ws://one.example:1/ws');
            assert.equal(WebSocketConnectionsManager.getConnection(),
                FakeWebSocket.instances[0] as unknown as WebSocket,
                'the only connection needs no argument to find');

            WebSocketConnectionsManager.addConnection('ws://two.example:2/ws');
            const calls = capturing(() => {
                assert.isUndefined(WebSocketConnectionsManager.getConnection(),
                    'ambiguous lookup returns undefined rather than picking one');
            });
            // No value to name here: the complaint is about the missing argument itself, so
            // severity and being reported once is the whole contract.
            checkLogged(calls, 'error', 'ambiguous lookup');

            WebSocketConnectionsManager.closeAll();
            assert.isFalse(WebSocketConnectionsManager.hasConnection('ws://one.example:1/ws'),
                'first connection forgotten');
            assert.isFalse(WebSocketConnectionsManager.hasConnection('ws://two.example:2/ws'),
                'second connection forgotten');
        });
    });
});
