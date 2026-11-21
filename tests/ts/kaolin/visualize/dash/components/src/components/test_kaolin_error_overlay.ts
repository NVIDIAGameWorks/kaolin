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

// Tests for KaolinErrorOverlay.
//
// The WebSocket layer is stubbed via sinon so subscribeToConnection captures
// the onMessage handler, letting us inject fake kaolin_error messages without
// a real network connection. The component is mounted with React's act() in a
// happy-dom DOM so state updates flush synchronously.

import { assert } from 'chai';
import sinon from 'sinon';
import React, { act } from 'react';
import { createRoot, Root } from 'react-dom/client';

import { WebSocketConnectionsManager } from '@kaolin/core/sockets';
import * as io from '@kaolin/core/io';
import { registerDom, unregisterDom } from '@test/helpers/dom';
import KaolinErrorOverlay from '@kaolin/components/KaolinErrorOverlay';

describe('tests/ts/kaolin/visualize/dash/components/src/components/test_kaolin_error_overlay.ts', () => {
    let sandbox: sinon.SinonSandbox;
    let roots: Root[] = [];
    // Captured onMessage callback injected by subscribeToConnection stub.
    let capturedOnMessage: ((event: MessageEvent) => void) | null;

    before(() => {
        registerDom();
        (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
    });

    after(async () => {
        await unregisterDom();
    });

    beforeEach(() => {
        sandbox = sinon.createSandbox();
        capturedOnMessage = null;

        // Stub the connection manager so no real WS is opened.
        sandbox.stub(WebSocketConnectionsManager, 'subscribeToConnection').callsFake(
            (_addr: string, _id: string | undefined, _subId: string,
             _onOpen: () => void, onMessage: (e: MessageEvent) => void) => {
                capturedOnMessage = onMessage;
                return () => {};
            },
        );
    });

    afterEach(() => {
        act(() => { roots.forEach(r => r.unmount()); });
        roots = [];
        sandbox.restore();
        delete (globalThis as any).dash_clientside;
    });

    /** Mount the overlay with the given props and return its container element. */
    function mount(props: { debug?: boolean } = {}): HTMLElement {
        const container = document.createElement('div');
        document.body.appendChild(container);
        const root = createRoot(container);
        roots.push(root);
        act(() => {
            root.render(React.createElement(KaolinErrorOverlay, {
                id: 'test-overlay',
                websocket_addresses: ['ws://localhost/websocket/'],
                ...props,
            } as any));
        });
        return container;
    }

    /** Deliver a kaolin_error WS message to the captured onMessage handler. */
    async function sendKaolinError(): Promise<void> {
        assert.isNotNull(capturedOnMessage, 'subscribeToConnection was not called');
        const payload = JSON.stringify({
            [io.MESSAGE_TAG_KEY]: 'kaolin_error',
            [io.MESSAGE_CONTENT_KEY]: {},
        });
        const event = new MessageEvent('message', { data: payload });
        await act(async () => { await capturedOnMessage!(event); });
    }

    describe('KaolinErrorOverlay', () => {
        it('is not visible on mount', () => {
            const container = mount();
            assert.isNull(container.querySelector('.btn-close'));
        });

        it('shows modal on kaolin_error message', async () => {
            const container = mount({ debug: false });
            await sendKaolinError();
            assert.isNotNull(container.querySelector('.btn-close'), 'close button should appear');
        });

        it('does not show modal in debug mode', async () => {
            mount({ debug: true });
            await sendKaolinError();
            // In debug mode the Dash debug pane handles errors; no overlay.
            assert.isNull(document.querySelector('.btn-close'));
        });

        it('dismisses on close (×) button click', async () => {
            const container = mount({ debug: false });
            await sendKaolinError();
            const closeBtn = container.querySelector('.btn-close') as HTMLButtonElement;
            assert.isNotNull(closeBtn);
            act(() => { closeBtn.click(); });
            assert.isNull(container.querySelector('.btn-close'), 'modal should be gone after close');
        });

        it('dismisses on Dismiss button click', async () => {
            const container = mount({ debug: false });
            await sendKaolinError();
            const dismissBtn = container.querySelector('.btn.btn-primary.btn-sm') as HTMLButtonElement;
            assert.isNotNull(dismissBtn, 'dismiss button should exist');
            act(() => { dismissBtn.click(); });
            assert.isNull(container.querySelector('.btn-close'), 'modal should be gone after dismiss');
        });

        it('ignores messages with a different tag', async () => {
            const container = mount({ debug: false });
            assert.isNotNull(capturedOnMessage);
            const payload = JSON.stringify({
                [io.MESSAGE_TAG_KEY]: 'render',
                [io.MESSAGE_CONTENT_KEY]: {},
            });
            await act(async () => {
                await capturedOnMessage!(new MessageEvent('message', { data: payload }));
            });
            assert.isNull(container.querySelector('.btn-close'), 'unrelated tags must not trigger overlay');
        });

        it('can be triggered multiple times after dismissal', async () => {
            const container = mount({ debug: false });
            await sendKaolinError();
            const closeBtn = container.querySelector('.btn-close') as HTMLButtonElement;
            act(() => { closeBtn.click(); });
            assert.isNull(container.querySelector('.btn-close'), 'dismissed');

            // Second error should re-open the modal.
            await sendKaolinError();
            assert.isNotNull(container.querySelector('.btn-close'), 'should re-appear on second error');
        });

        it('shows modal on binary ArrayBuffer kaolin_error message', async () => {
            const container = mount({ debug: false });
            assert.isNotNull(capturedOnMessage);

            // Encode a kaolin_error as a binary kaolin message and deliver it as
            // an ArrayBuffer — exercises the io.fromBinary decode path.
            const encoded = await io.encodeMessage('kaolin_error', {}, true);
            const buffer: ArrayBuffer = encoded instanceof Uint8Array
                ? encoded.buffer
                : encoded as ArrayBuffer;
            const event = new MessageEvent('message', { data: buffer });
            await act(async () => { await capturedOnMessage!(event); });

            assert.isNotNull(container.querySelector('.btn-close'), 'modal should appear for binary message');
        });

        it('routes to Dash relay store in debug mode instead of showing modal', async () => {
            // In debug mode KaolinErrorOverlay must call set_props on the relay store
            // and NOT show the overlay modal.
            const setProps = sandbox.stub();
            (globalThis as any).dash_clientside = { set_props: setProps };

            mount({ debug: true });
            await sendKaolinError();

            // Modal must NOT appear.
            assert.isNull(document.querySelector('.btn-close'), 'overlay must not show in debug mode');
            // Relay store must be updated.
            assert.isTrue(setProps.calledOnce, 'set_props must be called once');
            const [storeId, update] = setProps.firstCall.args;
            assert.equal(storeId, '_kaolin-ws-error-relay');
            assert.isObject(update.data);
        });

        it('subscribes to all addresses in the array', () => {
            const subscribeSpy = WebSocketConnectionsManager.subscribeToConnection as sinon.SinonStub;
            mount({ debug: false });
            // The component has one address → subscribeToConnection called once.
            assert.equal(subscribeSpy.callCount, 1);
        });

        it('subscribes using tuple [address, id] spec', () => {
            const subscribeSpy = WebSocketConnectionsManager.subscribeToConnection as sinon.SinonStub;
            // Use tuple form: [address, identifier].
            const container = document.createElement('div');
            document.body.appendChild(container);
            const root = createRoot(container);
            roots.push(root);
            act(() => {
                root.render(React.createElement(KaolinErrorOverlay, {
                    id: 'overlay-tuple',
                    websocket_addresses: [['ws://localhost/ws/', 'main-ws']] as any,
                    debug: false,
                } as any));
            });
            assert.equal(subscribeSpy.callCount, 1);
            const [addr, wsId] = subscribeSpy.firstCall.args;
            assert.equal(addr, 'ws://localhost/ws/');
            assert.equal(wsId, 'main-ws');
        });

        it('unsubscribes from all connections on unmount', () => {
            let unsubCalled = false;
            (WebSocketConnectionsManager.subscribeToConnection as sinon.SinonStub).callsFake(
                () => () => { unsubCalled = true; }
            );
            const container = document.createElement('div');
            document.body.appendChild(container);
            const root = createRoot(container);
            act(() => {
                root.render(React.createElement(KaolinErrorOverlay, {
                    id: 'overlay-unsub',
                    websocket_addresses: ['ws://localhost/ws/'],
                } as any));
            });
            act(() => { root.unmount(); });
            assert.isTrue(unsubCalled, 'unsubscribe must be called on unmount');
        });
    });
});
