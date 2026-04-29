import { logger } from '../util/logging';
import { getTabUuid } from '../util';



/**
 * Replaces special string "WINDOW_LOCATION" with window's current location,
 * if present, or returns original url.
 * 
 * @param url the address stirng such as wss://0.19.18.70:8080/websocket or any other
 */
function resolveWindowLocation(url: string): string {
    if (url.includes('WINDOW_LOCATION')) {
        return url.replace('WINDOW_LOCATION', window.location.host);
    }
    return url;
}

/**
 * Append the per-tab uuid as a `tab=<uuid>` query parameter to a WS URL.
 * Idempotent: if `tab` is already set, the existing value wins so callers can
 * override for tests. Used internally; the registry is keyed on the unparameterized
 * address so callers like `getOpenConnection('main-ws')` need not know about the param.
 */
function appendTabUuid(address: string): string {
    try {
        const u = new URL(address);
        if (!u.searchParams.has('tab')) {
            u.searchParams.set('tab', getTabUuid());
        }
        return u.toString();
    } catch (e) {
        logger.warn(`appendTabUuid: failed to parse WS address ${address}, sending without tab uuid`, e);
        return address;
    }
}

/**
 * One viewer's set of callbacks on a shared connection. A single WebSocket can
 * have many subscribers (e.g. multiple {@link KaolinViewerInternal} instances
 * pointing at the same address); the manager fans every socket event out to all
 * of them so each viewer dispatches incoming messages to its own behaviors.
 */
interface ConnectionSubscriber {
    onOpen: () => void;
    onMessage: (event: MessageEvent) => void;
    onError: (event: Event) => void;
    onClose: () => void;
}

export class WebSocketConnectionsManager {
    static _registry = new Map<string, WebSocket>();
    static _isOpen = new Map<string, boolean>();
    static _idToAddress = new Map<string, string>();
    // address -> (subscriberId -> subscriber). Keyed by id so re-subscribing the
    // same subscriber is idempotent (no duplicate event delivery). The ws.on*
    // handlers fan every event out to all current subscribers.
    static _subscribers = new Map<string, Map<string, ConnectionSubscriber>>();

    /**
     * Open a WebSocket to `unresolvedAddress` if one does not already exist
     * (no-op otherwise). Registers `identifier` (e.g. 'main-ws') as an alias for
     * the resolved address so the send side can look the socket up by id.
     *
     * This only manages the socket itself; use {@link subscribeToConnection} to
     * receive its events.
     */
    static addConnection(unresolvedAddress: string, identifier?: string): void {
        const address = resolveWindowLocation(unresolvedAddress);
        const fullAddress = appendTabUuid(address);
        logger.info(`WebSocket starting connection to: ${unresolvedAddress} --> ${address} (with tab uuid)`);

        if (this._registry.has(address)) {
            return;
        }

        const ws = new WebSocket(fullAddress);
        this._registry.set(address, ws);
        this._isOpen.set(address, false);

        if (identifier) {
            const previousValue = this._idToAddress.get(identifier);
            if (previousValue) {
                logger.error(`Short identifier clash ${identifier}: attempting to assign to both ${previousValue} and ${address}`);
            } else {
                this._idToAddress.set(identifier, address);
            }
        }

        ws.onopen = () => {
            logger.info(`WebSocket connection established: ${address}`);
            this._isOpen.set(address, true);
            this._forEachSubscriber(address, (s) => s.onOpen());
        };
        ws.onmessage = (event) => this._forEachSubscriber(address, (s) => s.onMessage(event));
        ws.onerror = (event) => this._forEachSubscriber(address, (s) => s.onError(event));
        ws.onclose = () => {
            logger.info(`WebSocket connection closed: ${address}`);
            this._isOpen.set(address, false);
            this._forEachSubscriber(address, (s) => s.onClose());
        };
    }

    private static _forEachSubscriber(address: string, fn: (s: ConnectionSubscriber) => void): void {
        this._subscribers.get(address)?.forEach(fn);
    }

    /**
     * Subscribe `subscriberId`'s callbacks to the connection at `unresolvedAddress`,
     * opening the socket (via {@link addConnection}) if needed.
     *
     * Idempotent: re-subscribing an already-registered `subscriberId` is ignored,
     * so events are never delivered twice to the same subscriber. If the socket is
     * already open, `onOpen` is invoked immediately so a late subscriber does not
     * miss the open event (otherwise it is delivered by `ws.onopen`).
     *
     * Returns an unsubscribe function that removes this subscriber (call it on
     * unmount). The underlying socket is left open for any remaining subscribers.
     */
    static subscribeToConnection(
        unresolvedAddress: string,
        identifier: string | undefined,
        subscriberId: string,
        onOpen: () => void,
        onMessage: (event: MessageEvent) => void,
        onError: (event: Event) => void,
        onClose: () => void,
    ): () => void {
        this.addConnection(unresolvedAddress, identifier);
        const address = resolveWindowLocation(unresolvedAddress);

        let subscribers = this._subscribers.get(address);
        if (!subscribers) {
            subscribers = new Map<string, ConnectionSubscriber>();
            this._subscribers.set(address, subscribers);
        }

        const unsubscribe = () => { this._subscribers.get(address)?.delete(subscriberId); };

        if (subscribers.has(subscriberId)) {
            // Already subscribed; ignore so events aren't delivered twice.
            return unsubscribe;
        }
        subscribers.set(subscriberId, { onOpen, onMessage, onError, onClose });

        // Already open: replay onOpen so this subscriber doesn't miss the event
        // that fired before it joined.
        if (this._isOpen.get(address)) {
            onOpen();
        }
        return unsubscribe;
    }

    static defaultAddress() {
        if (this._registry.size === 1) {
            return this._registry.keys().next().value;
        }
        return undefined;
    }

    static resolveAddress(unresolvedAddressOrId?: string): string | undefined {
        // Can skip providing argument, if only one connection present
        if (!unresolvedAddressOrId) {
            unresolvedAddressOrId = this.defaultAddress();
            if (!unresolvedAddressOrId) {
                logger.error(`WebSocketManager cannot guess which connection is requsted by getConnection(); provide argument`);
                return undefined;
            }
        }
        // Retrieve by ID
        if (this._idToAddress.has(unresolvedAddressOrId)) {
            return this._idToAddress.get(unresolvedAddressOrId);
        }
        // Or process address
        return resolveWindowLocation(unresolvedAddressOrId);
    }

    static getConnection(unresolvedAddressOrId?: string): WebSocket | undefined {
        const address = this.resolveAddress(unresolvedAddressOrId);
        return this._registry.get(address);
    }

    static hasConnection(unresolvedAddressOrId?: string): boolean {
        const address = this.resolveAddress(unresolvedAddressOrId);
        return this._registry.has(address);
    }

    static isConnectionOpen(unresolvedAddressOrId?: string) {
        const address = this.resolveAddress(unresolvedAddressOrId);
        return this._isOpen.get(address);
    }

    static getOpenConnection(unresolvedAddressOrId?: string): WebSocket | undefined {
        if (this.isConnectionOpen(unresolvedAddressOrId)) {
            return this.getConnection(unresolvedAddressOrId);
        }
        return undefined;
    }

    static closeConnection(unresolvedAddressOrId?: string): void {
        const address = this.resolveAddress(unresolvedAddressOrId);

        if (this._registry.has(address)) {
            let ws = this._registry.get(address);
            if (ws && this._isOpen.get(address) && ws.readyState === WebSocket.OPEN) {
                ws.close();
            }
            this._registry.delete(address);
            this._isOpen.delete(address);
            this._subscribers.delete(address);

            // Delete any id that points to this address
            for (const [id, addr] of this._idToAddress.entries()) {
                if (addr === address) {
                    this._idToAddress.delete(id);
                }
            }
        }
    }

    static closeAll(): void {
        for (const address of this._registry.keys()) {
            this.closeConnection(address);
        }
    }
}
