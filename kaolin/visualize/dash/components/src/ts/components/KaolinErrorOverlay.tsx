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

import React, { useState, useEffect, useCallback } from 'react';
import * as io from '../core/io';
import { WebSocketConnectionsManager } from '../core/sockets';

// Note on the relay mechanism: must match WebappBuilder.WS_ERROR_RELAY_STORE_ID in builder.py.
const WS_ERROR_RELAY_STORE_ID = '_kaolin-ws-error-relay';

type WebSocketAddressSpec = string | [string, string];

interface Props {
    /** Dash component id. */
    id: string;
    /**
     * WebSocket server addresses to listen on for ``kaolin_error`` messages.
     * Each element is either a plain address string or an ``[address, id]`` tuple —
     * the same format as ``KaolinViewerInternal.websocket_addresses``. The
     * underlying connections are shared with any viewer on the same page, so no
     * extra socket is opened.
     */
    websocket_addresses?: WebSocketAddressSpec[];
    /**
     * When ``True``, forward ``kaolin_error`` messages to Dash's debug pane via
     * the relay Store instead of showing the modal.
     */
    debug?: boolean;
    setProps?: (props: Partial<Props>) => void;
}

/**
 * App-level server-error overlay.
 *
 * Subscribes to the same WebSocket connections as the viewer and shows a
 * dismissible modal whenever a ``kaolin_error`` message arrives. Add it once
 * to the app layout via ``WebappBuilder._setup_error_overlay`` — it is
 * intentionally separate from ``KaolinViewerInternal`` so that apps with
 * multiple viewers still display a single, shared error modal.
 */
const KaolinErrorOverlay = ({ id, websocket_addresses, debug }: Props) => {
    const [visible, setVisible] = useState(false);

    const handleMessage = useCallback(async (event: MessageEvent) => {
        let decoded: Map<string, any> | null = null;
        if (event.data instanceof Blob) {
            decoded = io.fromBinary(await event.data.arrayBuffer());
        } else if (event.data instanceof ArrayBuffer) {
            decoded = io.fromBinary(event.data);
        } else if (typeof event.data === 'string') {
            decoded = io.fromJSON(event.data);
        }
        if (decoded?.get(io.MESSAGE_TAG_KEY) === 'kaolin_error') {
            if (debug) {
                // In debug mode forward to the Dash debug pane via the relay Store.
                // MESSAGE_CONTENT_KEY maps to a Map<string,any> after fromJSON decode.
                const content: Map<string, any> | null = decoded.get(io.MESSAGE_CONTENT_KEY) ?? null;
                (window as any).dash_clientside?.set_props?.(WS_ERROR_RELAY_STORE_ID, {
                    data: {
                        error_type: content?.get('error_type') ?? 'Error',
                        message: content?.get('message') ?? '',
                        traceback: content?.get('traceback') ?? '',
                    },
                });
            } else {
                setVisible(true);
            }
        }
    }, [debug]);

    useEffect(() => {
        if (!websocket_addresses?.length) return;
        const unsubs: Array<() => void> = [];
        for (const spec of websocket_addresses) {
            const addr = typeof spec === 'string' ? spec : spec[0];
            const wsId = typeof spec === 'string' ? undefined : spec[1];
            unsubs.push(
                WebSocketConnectionsManager.subscribeToConnection(
                    addr, wsId, id,
                    () => {}, handleMessage, () => {}, () => {}
                )
            );
        }
        return () => unsubs.forEach(u => u());
    }, [websocket_addresses, id, handleMessage]);

    if (!visible) return null;

    return (
        <div style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.55)',
            // zIndex above Bootstrap navbar dropdowns (~1030) and Dash's own overlays.
            zIndex: 9999,
            display: 'flex',
            // flex-start + padding keeps the modal fully visible when the mobile
            // browser chrome (address bar, nav bar) reduces the visible viewport.
            alignItems: 'flex-start',
            justifyContent: 'center',
            padding: '5vh 5%',
            overflowY: 'auto',
        }}>
            <div style={{ maxWidth: 520, width: '100%', display: 'flex', flexDirection: 'column' }}>
                <div style={{
                    background: '#fff',
                    borderRadius: 8,
                    boxShadow: '0 8px 32px rgba(0,0,0,0.35)',
                    overflow: 'hidden',
                    display: 'flex',
                    flexDirection: 'column',
                    // Limit height so body text stays scrollable on short screens.
                    maxHeight: '90vh',
                }}>
                    <div style={{
                        background: '#dc3545',
                        color: '#fff',
                        padding: '12px 16px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        flexShrink: 0,
                    }}>
                        <strong>⚠ Server Error</strong>
                        <button
                            type='button'
                            className='btn-close btn-close-white'
                            aria-label='Dismiss'
                            onClick={() => setVisible(false)}
                        />
                    </div>
                    <div style={{ padding: '16px', color: '#212529' }}>
                        <p style={{ margin: 0 }}>
                            A server error has occurred. Use the{' '}
                            <code style={{
                                background: '#e8e8e8',
                                color: '#333',
                                padding: '1px 5px',
                                borderRadius: 3,
                                fontSize: '0.9em',
                            }}>--debug</code>
                            {' '}flag or check the server logs for more info.
                        </p>
                    </div>
                    <div style={{
                        padding: '12px 16px',
                        borderTop: '1px solid #dee2e6',
                        display: 'flex',
                        justifyContent: 'flex-end',
                    }}>
                        <button
                            type='button'
                            className='btn btn-primary btn-sm'
                            onClick={() => setVisible(false)}
                        >
                            Dismiss
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

KaolinErrorOverlay.defaultProps = {
    websocket_addresses: undefined,
    debug: false,
};

export default KaolinErrorOverlay;
