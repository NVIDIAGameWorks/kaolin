import '../../css/viewer.css';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { getCanvasCoordinates } from '../util/canvas';
import { BehaviorRunner, CameraControllerInterface, isCameraController, InteractiveBehavior, isElementBoundBehavior, isClickEventHandler, isPointerEventHandler, ClickEventHandler, PointerEventHandler, MessageHandler, isMessageHandler } from '../core/behavior';
import * as kaolin_events from '../core/event';
import * as io from '../core/io';
import { LayerComponent } from '../core/layer';
import { WebSocketConnectionsManager } from '../core/sockets';
import { computeContainedDisplaySize, computeContainedFixedAspectSize, defaultSetElementDimensions, ViewportResizeMode } from '../core/viewport';
import { FpsReadout, InteractiveFps, InteractiveFpsProvider } from '../util/fps';
import { logger } from '../util/logging';
import * as graphics from '../graphics';
import { DashComponentProps } from '../props';

/** Reserved layer names that map to special elements in the viewer */
export const ViewerReservedNames = {
    eventCanvas: 'EVENT_CANVAS',
    setBehaviorActive: kaolin_events.ACTIVE_BEHAVIOR_SETTER_NAME
} as const;

type Props = {
    /**
     * WebSocket server addresses to connect to.
     * Each element is either a string (address) or [address, id] tuple.
     */
    websocket_addresses?: (string | [string, string])[];

    /** Starting camera paremters. */
    camera_parameters?: graphics.camera.CameraParameters;

    /** Layer specs as list of [layerId, layerName, layerProps] tuples. */
    layers?: [string, string, any][];

    /** Optional behaviors as list of [behaviorId, behaviorName, activeLayerId, behaviorProps, isActive] tuples. */
    behaviors?: [string, string, string | null, any, boolean][];

    /** Behaviors listening for the camera updates. */
    camera_listeners?: [string, string, boolean][];

    /**
     * Behavior ids that act as camera controllers (own a camera and implement
     * the camera-controller API). Multiple may be registered, but only one is
     * active at a time (active = its entry in the behavior active-status map);
     * the active controller is the source dispatched to camera listeners.
     */
    camera_controllers?: string[];

    /**
     * Rules for mirroring an option change on one behavior onto others. Each rule
     * is `[optionName, sourceBehaviorId, targetBehaviorIds, targetOptionNames]`:
     *   - `optionName`: the option whose change triggers the broadcast, or `'*'`
     *     to match every option change (in which case the changed option's name is
     *     reused on each target and `targetOptionNames` must be `null`).
     *   - `sourceBehaviorId`: only broadcast when this behavior is the one that
     *     changed; `null` means a change on any behavior triggers it.
     *   - `targetBehaviorIds`: behaviors that receive the value; `null` means all
     *     other registered behaviors.
     *   - `targetOptionNames`: option name to set on each target (aligned with
     *     `targetBehaviorIds`); `null` reuses `optionName` on every target.
     * Multiple rules may share an `optionName`; every matching rule fires.
     */
    broadcast_behavior_options?: [string, string | null, string[] | null, string[] | null][];

    /** How the viewport canvas resizes within its container. */
    resize_mode?: ViewportResizeMode;

    /**
     * Named modes as a list of [modeName, [behaviorIds...], uiIcon, description]
     * tuples. `uiIcon` is an optional Bootstrap Icons class (e.g. 'bi-binoculars')
     * and `description` is an optional tooltip / aria label.
     *
     * Modes with a `uiIcon` are rendered as buttons in the in-viewer overlay.
     * The first nine modes (in registration order) get keyboard shortcuts 1..9
     * auto-assigned by the React side — there is no Python-facing shortcut arg.
     *
     * The viewer does NOT track which mode is currently active — that knowledge
     * would be unreliable since apps can toggle individual behaviors at runtime.
     * Initial active state is expected to be baked into the per-behavior
     * `isActive` flags in `behaviors`; see `ViewerBuilder.set_active_mode`.
     */
    modes?: [string, string[], string | null, string | null][];

    /**
     * If set, enables an in-viewer FPS overlay driven by incoming WebSocket
     * messages whose tag equals this string (typically `'render'`, set via
     * `ViewerBuilder.show_remote_fps`). Every matching message increments the
     * sliding-window FPS tracker and updates the overlay. When unset, no
     * tracker is created and the overlay is not rendered.
     */
    remote_fps_source?: string;

    /** Optional suffix for the FPS overlay (e.g. target rate from remote rendering). */
    remote_fps_label?: string;

    /**
     * Read-only output prop: name of the currently-active mode, or `null`
     * when no registered mode matches the current per-behavior active state.
     *
     * The viewer publishes this via Dash `setProps` whenever the derived
     * active mode changes (initial mount, mode-button click, keyboard
     * shortcut, `requestMode(...)`, or any direct behavior toggle that lands
     * the system in or out of a registered mode). Subscribe with a standard
     * Dash callback:
     *
     *     app.clientside_callback(
     *         "function(mode) { ... }",
     *         Output(...), Input('viewer-id', 'active_mode'))
     *
     * One-way (viewer → app). Writing this prop from a callback does NOT
     * switch modes — call `requestMode(name, viewerId)` for that.
     */
    active_mode?: string | null;

} & DashComponentProps;



/**
 * Component description
 */
// const KaolinViewer : React.FC<Props> = ({ websocket_address }: Props) => {
const KaolinViewerInternal = (props: Props) => {
    const { id: id, websocket_addresses: websocketAddresses, remote_fps_source: remoteFpsSource } = props;
    const resizeMode = useMemo(() => props.resize_mode ?? ViewportResizeMode.FIXED, [props.resize_mode]);

    // Sliding-window FPS tracker for render-on-demand. Lives in a ref so it
    // survives re-renders without losing samples. Only created when the
    // Python side opted in via `show_remote_fps`; absent otherwise so the
    // happy path pays no overhead.
    const fpsTrackerRef = useRef<InteractiveFps | null>(null);
    if (remoteFpsSource && fpsTrackerRef.current === null) {
        fpsTrackerRef.current = new InteractiveFps();
    }

    const eventCanvasRef = useRef<HTMLCanvasElement>(null);
    const viewerFrameRef = useRef<HTMLDivElement>(null);

    // The viewer no longer owns a camera; it pipes the (optional) input camera
    // parameters to camera controllers and dispatches the active controller's
    // camera to listeners. Controller ids are listed in `camera_controllers`.
    const cameraControllerIds = useMemo(() => props.camera_controllers ?? [], [props.camera_controllers]);
    // Controller ids already seeded with the initial camera (one-shot, so the
    // element-binding effect re-runs do not clobber user-driven camera changes).
    const initializedControllers = useRef<Set<string>>(new Set());
    // Latest computed canvas drawing-buffer size (device pixels). Source of
    // truth for sizing canvas layers and for seeding camera-controller
    // dimensions; populated by the resize observer (fires on mount too).
    const lastCanvasSizeRef = useRef<{ width: number, height: number } | null>(null);

    const isAnimating = useRef<boolean>(false);  // TODO: I don't think we need this one, why was it added
    // Pending trailing camera-update timer for wheel/zoom. OrbitControls applies
    // the dolly in its own native `wheel` listener, which can run after our React
    // onWheel handler — so a synchronous read here dispatches a pre-zoom camera,
    // leaving listeners one tick stale (visible as a jump on the next click/drag).
    // We coalesce rapid wheel ticks into a single trailing send once the dolly
    // has settled. Lives in a ref so it survives re-renders.
    const wheelSettleTimeout = useRef<ReturnType<typeof setTimeout> | null>(null);
    //const cameraUpdateDate = useRef<Date | null>(null);
    //const minCamUpdateMs = useRef<number>(10);

    // TODO: convert to Map<string, boolean> to match canvasBehaviors Map structure
    const behaviorCursorMoveActive = useRef<boolean[]>([]);

    const [layers, setLayers] = useState(new Map<string, React.RefObject<HTMLElement>>());
    const [behaviors, setBehaviors] = useState(new Map<string, InteractiveBehavior>());
    const [behaviorActiveStatus, setBehaviorActiveStatus] = useState<Map<string, boolean>>(
        new Map(props.behaviors?.map(
            ([behaviorId, _name, _layerId, _options, isActive]) => [behaviorId, isActive ?? true] as [string, boolean])));
    const [cameraListeners, setCameraListeners] = useState<Map<string, [string, boolean]>>(
        new Map(props.camera_listeners?.map(
            ([behaviorId, setterName, useRawObject]) => [behaviorId, [setterName, useRawObject]] as [string, [string, boolean]])));

    // Named modes → set of behavior ids that should be active in that mode.
    // We do NOT track "current mode" — apps can flip individual behaviors at
    // runtime so any cached current-mode value would quickly go stale.
    const modes = useMemo(
        () => new Map<string, Set<string>>(
            (props.modes ?? []).map(([name, ids]) => [name, new Set(ids)])),
        [props.modes]);

    // Ordered list used to drive the overlay UI and keyboard shortcuts. Shortcuts
    // are assigned at this layer rather than coming from Python — the first nine
    // modes registered get '1'..'9' bound to them; anything beyond gets none.
    const modeDisplayList = useMemo(
        () => (props.modes ?? []).map(([name, ids, icon, description], i) => ({
            name,
            ids,
            icon,
            description,
            shortcut: i < 9 ? String(i + 1) : null,
        })),
        [props.modes]);

    const shortcutToMode = useMemo(() => {
        const m = new Map<string, string>();
        modeDisplayList.forEach(({ name, shortcut }) => {
            if (shortcut) m.set(shortcut, name);
        });
        return m;
    }, [modeDisplayList]);

    // Ref to access latest behaviors in closures (e.g., WebSocket handlers) and
    // external (non-React) event listeners. Kept in lock-step with the
    // `behaviors` state synchronously inside `registerBehavior` — the sole
    // writer — so it is never stale; do NOT sync it via an effect (that lags a
    // render behind and reintroduces the registration race).
    const behaviorsRef = useRef<Map<string, InteractiveBehavior>>(behaviors);

    // Ref to access latest layers in stable closures (e.g., resize observer).
    // Kept in lock-step with the `layers` state synchronously inside
    // `registerLayer` — the sole writer — so it is never stale; do NOT sync it
    // via an effect (that lags a render behind).
    const layersRef = useRef<Map<string, React.RefObject<HTMLElement>>>(layers);

    // Ref to access latest behavior active status in closures (e.g. the
    // WebSocket message handler). Kept in lock-step with the state synchronously
    // at every `setBehaviorActiveStatus` call site; do NOT sync it via an effect
    // (that lags a render behind).
    const behaviorActiveStatusRef = useRef<Map<string, boolean>>(behaviorActiveStatus);

    // Tracks whether the mouse is currently over this viewer's outer container.
    // Used to gate keyboard mode shortcuts so multiple viewers on the same page
    // can coexist with independent shortcuts.
    const isHoveredRef = useRef<boolean>(false);


    const registerLayer = useCallback((identifier: string, layerRef: React.RefObject<HTMLElement> | null) => {
        // Build from the ref (kept in lock-step below) and update it
        // synchronously, before setLayers, so external/stable closures never
        // observe a stale map between render and the post-commit effect.
        const newLayers = new Map(layersRef.current);
        if (layerRef) {
            newLayers.set(identifier, layerRef);
        } else {
            newLayers.delete(identifier);
        }
        layersRef.current = newLayers;
        setLayers(newLayers);
    }, []);


    // Create a stable function to register (or unregister) an API
    const registerBehavior = useCallback((identifier: string, behavior: InteractiveBehavior | null) => {
        // Build the next map from the ref (kept in lock-step below) rather than
        // through a functional setState updater. The ref is then updated
        // synchronously, BEFORE setBehaviors, so external (non-React) event
        // handlers — notably the EDIT_BEHAVIOR window listener — never observe a
        // stale map in the window between render and the post-commit effect that
        // used to be the only place this ref was synced.
        const newBehaviors = new Map(behaviorsRef.current);
        if (behavior) {
            // Register: Add the behavior to the map
            console.log(`Registering behavior ${identifier} for a total of ${newBehaviors.size + 1}`);
            newBehaviors.set(identifier, behavior);
        } else {
            // Unregister: remove the behavior from the map
            console.log(`Unregistering behavior ${identifier}`);
            newBehaviors.delete(identifier);
        }
        behaviorsRef.current = newBehaviors;
        setBehaviors(newBehaviors);

        // Also update active status map on unregister; allow re-seeding if the
        // controller is later re-registered.
        if (!behavior) {
            initializedControllers.current.delete(identifier);
            const newStatus = new Map(behaviorActiveStatusRef.current);
            newStatus.delete(identifier);
            behaviorActiveStatusRef.current = newStatus;
            setBehaviorActiveStatus(newStatus);
        }
    }, []);


    // WebSocket event handlers
    const handleWebSocketOpen = () => {
        maybeSendCameraUpdate();
    };

    const handleMessage = useCallback((messageTag: string | null, messageContent: any | null) => {
        // Render-on-demand FPS: every inbound message whose tag matches the
        // configured remote frame source counts as one "rendered frame" for
        // the sliding-window tracker. Tracker ref is only populated when
        // remote_fps_source is set, so the no-tracking path is a single
        // pointer comparison.
        if (fpsTrackerRef.current && messageTag === remoteFpsSource) {
            fpsTrackerRef.current.frameReceived();
        }

        behaviorsRef.current.forEach((behavior, identifier) => {
            // Skip inactive behaviors
            if (behaviorActiveStatusRef.current.get(identifier) === false) return;

            if (isMessageHandler(behavior)) {
                let handler = ((behavior as any) as MessageHandler);

                if (handler.acceptedMessageTags().includes(messageTag)) {
                    // TODO: should make async.
                    handler.onMessage(messageTag, messageContent);
                }
            }
        });
    }, []); // No dependency on behaviors - we use the ref

    // Derive the active camera controller, mirroring `activeModeName`: the first
    // listed controller whose active-status is not false (only one is active at
    // a time). Memoized so dependent effects re-run when activation changes.
    const activeCameraController = useMemo<CameraControllerInterface | null>(() => {
        for (const controllerId of cameraControllerIds) {
            if (behaviorActiveStatus.get(controllerId) === false) continue;
            const behavior = behaviors.get(controllerId);
            if (behavior && isCameraController(behavior)) {
                return behavior as unknown as CameraControllerInterface;
            }
        }
        return null;
    }, [cameraControllerIds, behaviorActiveStatus, behaviors]);

    const maybeSendCameraUpdate = useCallback(() => {
        if (!activeCameraController) return;

        cameraListeners.forEach((listenerOptions, behaviorId) => {
            // Skip inactive behaviors
            if (behaviorActiveStatus.get(behaviorId) === false) return;

            const behavior = behaviors.get(behaviorId);
            const [setterName, isRawObject] = listenerOptions;
            if (behavior) {
                const value = isRawObject ? activeCameraController.getCamera() : activeCameraController.getCameraParams();
                (behavior as any)[setterName](value);
            }
        });
    }, [activeCameraController, cameraListeners, behaviorActiveStatus, behaviors]);

    const handleWebSocketMessage = async (event: MessageEvent) => {
        let decodedMessage = null;

        // Step 1: Decode message
        if (event.data instanceof Blob) {
            // TODO: does this happen or did Cursor invent this?
            // Convert Blob to ArrayBuffer
            const arrayBuffer = await event.data.arrayBuffer();
            decodedMessage = io.fromBinary(arrayBuffer);
        } else if (event.data instanceof ArrayBuffer) {
            decodedMessage = io.fromBinary(event.data);
        } else if (typeof event.data === 'string') {
            try {
                decodedMessage = io.fromJSON(event.data);
            } catch (error) {
                logger.error('Received text message (not JSON):', event.data);
                return;
                // Handle plain text messages here if needed
            }
        } else {
            logger.error('Received message of unknown type:', typeof event.data, event.data);
            return
        }

        // Step 2: Handle message
        const messageTag: string = decodedMessage?.get(io.MESSAGE_TAG_KEY);
        const messageContent: any = decodedMessage?.get(io.MESSAGE_CONTENT_KEY);
        handleMessage(messageTag, messageContent);
    };

    const handleWebSocketError = (error: Event) => {
        console.error('WebSocket error:', error);
    };

    const handleWebSocketClose = () => {
        // TODO: show some sort of error somewhere?
        console.log('WebSocket connection closed');
    };

    // Route click / double-click events to active behaviors that implement the
    // click-handler contract (checked via isClickEventHandler).
    const applyClickHandlers = (event: React.MouseEvent<HTMLCanvasElement>, handlerName: keyof ClickEventHandler) => {
        behaviors.forEach((behavior, identifier) => {
            // Skip inactive behaviors
            if (behaviorActiveStatus.get(identifier) === false) return;

            if (isClickEventHandler(behavior)) {
                behavior[handlerName]?.(event as any);
            }
        });
    };

    // Route pointer events to active behaviors that implement the pointer-handler
    // contract (checked via isPointerEventHandler). This is the desired check for
    // most interactive behaviors (e.g. drawing).
    const applyPointerHandlers = (event: React.PointerEvent<HTMLCanvasElement>, handlerName: keyof PointerEventHandler) => {
        behaviors.forEach((behavior, identifier) => {
            // Skip inactive behaviors
            if (behaviorActiveStatus.get(identifier) === false) return;

            if (isPointerEventHandler(behavior)) {
                behavior[handlerName]?.(event as any);
            }
        });
    };

    const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
        const { x, y } = getCanvasCoordinates(event);

        console.log('Canvas clicked at:', { x, y });

        applyClickHandlers(event, 'onClick');
    };

    const handleCanvasDoubleClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
        const { x, y } = getCanvasCoordinates(event);

        console.log('Double click at:', { x, y });

        applyClickHandlers(event, 'onDoubleClick');
    };

    const handleCanvasPointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
        const { x, y } = getCanvasCoordinates(event);

        console.log('Pointer down at:', { x, y });
        // Capture the pointer so pointermove keeps firing even outside the element
        event.currentTarget.setPointerCapture(event.pointerId);

        applyPointerHandlers(event, 'onPointerDown');

        if (!isAnimating.current) {
            isAnimating.current = true;
            animate();
        }
    };

    const handleCanvasPointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
        applyPointerHandlers(event, 'onPointerMove');
    };

    const handleCanvasPointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
        const { x, y } = getCanvasCoordinates(event);

        console.log('Pointer up at:', { x, y });

        applyPointerHandlers(event, 'onPointerUp');

        isAnimating.current = false;
        maybeSendCameraUpdate();  // TODO: should force?
    };

    const handleCanvasPointerCancel = (event: React.PointerEvent<HTMLCanvasElement>) => {
        console.log('Pointer cancel');
        applyPointerHandlers(event, 'onPointerCancel');

        isAnimating.current = false;
    };

    const handleCanvasPointerEnter = (event: React.PointerEvent<HTMLCanvasElement>) => {
        console.log('Pointer entered canvas');
        applyPointerHandlers(event, 'onPointerEnter');

        maybeSendCameraUpdate();
    };

    const handleCanvasPointerLeave = (event: React.PointerEvent<HTMLCanvasElement>) => {
        console.log('Pointer left canvas');
        applyPointerHandlers(event, 'onPointerLeave');
    };

    const handleCanvasWheel = (event: React.WheelEvent<HTMLCanvasElement>) => {
        // Immediate send keeps listeners roughly in sync during a continuous
        // scroll; it may lag the dolly by one tick (see wheelSettleTimeout note).
        maybeSendCameraUpdate();

        // Trailing send after the wheel goes idle: by now OrbitControls has
        // committed the final dolly, so this dispatches the settled camera and
        // eliminates the post-scroll jump. Rapid ticks reset the timer so only
        // the last (settled) state is sent.
        if (wheelSettleTimeout.current !== null) clearTimeout(wheelSettleTimeout.current);
        wheelSettleTimeout.current = setTimeout(() => {
            wheelSettleTimeout.current = null;
            maybeSendCameraUpdate();
        }, 120);
    };

    const animate = () => {
        if (isAnimating.current) {
            requestAnimationFrame(animate);
        }

        // Handle all behaviors that require animation updates (including camera controller)
        behaviors.forEach((behavior, identifier) => {
            // Skip inactive behaviors
            if (behaviorActiveStatus.get(identifier) === false) return;

            if (behavior['onAnimate']) {
                behavior['onAnimate']();
            }
        });

        // Send camera updates to listeners
        maybeSendCameraUpdate();
    };

    // Shared activation path: bumps the behaviorActiveStatus map and calls the
    // behavior's own setActive() if it exposes one. Used by both the EDIT_BEHAVIOR
    // single-behavior flow and the SET_MODE multi-behavior sweep below.
    const setBehaviorActive = useCallback((behaviorId: string, active: boolean) => {
        // Read via ref so this callback stays stable (empty deps); the window
        // event listeners that depend on it then need not re-subscribe on every
        // behavior registration.
        const behavior = behaviorsRef.current.get(behaviorId);
        if (!behavior) {
            logger.error('Behavior not found: ' + behaviorId);
            return;
        }
        const newStatus = new Map(behaviorActiveStatusRef.current);
        newStatus.set(behaviorId, active);
        behaviorActiveStatusRef.current = newStatus;
        setBehaviorActiveStatus(newStatus);
        if (behavior['setActive']) {
            behavior['setActive'](active);
        }
    }, []);

    // Apply a named mode: iterates every registered behavior and activates it iff
    // it appears in the target mode's id set. Behaviors that exist but are not
    // listed in the mode are deactivated. No subset bookkeeping is required.
    const applyMode = useCallback((modeName: string) => {
        const target = modes.get(modeName);
        if (!target) {
            logger.error(`Unknown mode: ${modeName}`);
            return;
        }
        behaviorsRef.current.forEach((_behavior, behaviorId) => {
            setBehaviorActive(behaviorId, target.has(behaviorId));
        });
    }, [modes, setBehaviorActive]);

    // Mirror an option change from one behavior onto others per the
    // `broadcast_behavior_options` rules. For each rule whose option name matches
    // and whose source is unset or equals `originId`, the value is written to the
    // rule's target behaviors. A null target list means "all other registered
    // behaviors"; in that case each write is wrapped in try/catch so a behavior
    // that does not accept the option is skipped rather than aborting the
    // broadcast, whereas explicitly listed targets are written without that guard
    // so misconfigured rules surface as errors. The originating behavior is not
    // re-written here (the caller already updated it), and writes go straight to
    // `setOption`, so a broadcast never triggers further broadcasts.
    const broadcastOptionToBehaviors = useCallback(
        (originId: string, optionName: string, value: any) => {
            const rules = props.broadcast_behavior_options;
            if (!rules) return;
            for (const [ruleOption, sourceId, targetIds, targetOptionNames] of rules) {
                if (ruleOption !== '*' && ruleOption !== optionName) continue;
                if (sourceId != null && sourceId !== originId) continue;

                const explicitTargets = targetIds != null;
                const ids = explicitTargets
                    ? targetIds
                    : Array.from(behaviorsRef.current.keys()).filter(bid => bid !== originId);

                ids.forEach((targetId, index) => {
                    const target = behaviorsRef.current.get(targetId);
                    if (!target) {
                        logger.error(`broadcastOptionToBehaviors: target behavior '${targetId}' not found.`);
                        return;
                    }
                    const targetOption = targetOptionNames != null ? targetOptionNames[index] : optionName;
                    if (explicitTargets) {
                        target.setOption(targetOption, value);
                    } else {
                        try {
                            target.setOption(targetOption, value);
                        } catch (error) {
                            logger.warn(
                                `broadcastOptionToBehaviors: skipping '${targetId}'.'${targetOption}': ${error}`);
                        }
                    }
                });
            }
        }, [props.broadcast_behavior_options]);

    // Derive the currently-active mode name from behavior active status. A mode
    // is "active" iff every behavior it lists is on AND no behavior outside the
    // mode is on. This keeps the overlay correct even when apps toggle individual
    // behaviors directly (no stored mode state to go stale).
    const activeModeName = useMemo<string | null>(() => {
        for (const { name, ids } of modeDisplayList) {
            if (ids.length === 0) continue;
            const idSet = new Set(ids);
            const everyOn = ids.every(id => behaviorActiveStatus.get(id) === true);
            if (!everyOn) continue;
            let allOthersOff = true;
            behaviorActiveStatus.forEach((on, id) => {
                if (on && !idSet.has(id)) allOthersOff = false;
            });
            if (allOthersOff) return name;
        }
        return null;
    }, [modeDisplayList, behaviorActiveStatus]);

    // Publish the derived active-mode name back to Dash so apps can subscribe
    // with a standard clientside_callback. One-way: writing the prop from a
    // callback does NOT switch modes — `requestMode(...)` is the inbound path.
    useEffect(() => {
        props.setProps({ active_mode: activeModeName });
    }, [activeModeName]);

    // Behaviors are read through `behaviorsRef` (not the `behaviors` state) so this
    // effect can keep a stable dependency set and register the window listeners once,
    // rather than tearing them down and re-adding them on every behavior registration.
    useEffect(() => {
        const handleEditBehaviorEvent = (event: CustomEvent) => {
            logger.info('Edit behavior event received: ' + JSON.stringify(event));

            const command: kaolin_events.EditBehaviorCommand = kaolin_events.parseEditBehaviorEvent(event);
            if (!command) {
                logger.error('Invalid edit behavior command: ' + JSON.stringify(event));
                return;
            }

            if (command.setterName == ViewerReservedNames.setBehaviorActive) {
                setBehaviorActive(command.behaviorId, Boolean(command.value));
            } else {
                const behavior = behaviorsRef.current.get(command.behaviorId);
                if (!behavior) {
                    logger.error('Behavior not found: ' + command.behaviorId);
                    return;
                }
                if (!behavior[command.setterName]) {
                    logger.error('Setter not found: ' + command.setterName + ' for behavior: ' + JSON.stringify(behavior));
                    return;
                }
                try {
                    // Schema-driven option path: payload is {name, value}; unpack
                    // to the two-argument call `behavior.setOption(name, value)`.
                    if (command.setterName === kaolin_events.SET_OPTION_SETTER_NAME &&
                        command.value && typeof command.value === 'object' &&
                        'name' in command.value) {
                        behavior[command.setterName](command.value.name, command.value.value);
                        broadcastOptionToBehaviors(command.behaviorId, command.value.name, command.value.value);
                    } else {
                        behavior[command.setterName](command.value);
                    }
                } catch (error) {
                    logger.error('Error executing setter: ' + command.setterName + ' for behavior: ' + JSON.stringify(behavior) + ' error: ' + JSON.stringify(error));
                }
            }
        };

        const handleSetModeEvent = (event: CustomEvent) => {
            logger.info('Set mode event received: ' + JSON.stringify(event));
            const cmd = kaolin_events.parseSetModeEvent(event);
            if (!cmd) {
                logger.error('Invalid set-mode command: ' + JSON.stringify(event));
                return;
            }
            applyMode(cmd.modeName);
        };

        const editEventName = kaolin_events.customViewerEventName(kaolin_events.ViewerCustomEvent.EDIT_BEHAVIOR, id);
        const modeEventName = kaolin_events.customViewerEventName(kaolin_events.ViewerCustomEvent.SET_MODE, id);
        window.addEventListener(editEventName, handleEditBehaviorEvent as EventListener);
        window.addEventListener(modeEventName, handleSetModeEvent as EventListener);

        return () => {
            window.removeEventListener(editEventName, handleEditBehaviorEvent as EventListener);
            window.removeEventListener(modeEventName, handleSetModeEvent as EventListener);
        };
    }, [id, setBehaviorActive, applyMode, broadcastOptionToBehaviors]);

    // Keyboard mode shortcuts: 1..9 (top-row digits only) auto-bound to the
    // first nine modes in registration order. Fires only while the mouse is
    // over this viewer's outer container — multiple viewers on the same page
    // can therefore coexist with independent shortcuts. Also rejects modifier
    // combos, editable focus targets, and numpad keys so we don't fight
    // browser/OS shortcuts or interrupt typing.
    useEffect(() => {
        if (shortcutToMode.size === 0) return;

        const isEditableTarget = (el: EventTarget | null): boolean => {
            if (!(el instanceof HTMLElement)) return false;
            const tag = el.tagName;
            if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
            if (el.isContentEditable) return true;
            return false;
        };

        const handleKey = (event: KeyboardEvent) => {
            if (!isHoveredRef.current) return;
            if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
            if (isEditableTarget(event.target)) return;
            if (event.code && event.code.startsWith('Numpad')) return;
            const modeName = shortcutToMode.get(event.key);
            if (!modeName) return;
            event.preventDefault();
            applyMode(modeName);
        };
        document.addEventListener('keydown', handleKey);
        return () => document.removeEventListener('keydown', handleKey);
    }, [shortcutToMode, applyMode]);

    // Connect behaviors to their active layers when both are registered
    useEffect(() => {
        props.behaviors?.forEach(([behaviorId, behaviorName, activeLayerIdentifier, _options]) => {
            if (!activeLayerIdentifier) return;

            const behavior = behaviors.get(behaviorId);
            if (!behavior) return;

            // Check for reserved layer names
            let targetElement: HTMLElement | null = null;
            if (activeLayerIdentifier === ViewerReservedNames.eventCanvas) {
                targetElement = eventCanvasRef.current;
            } else {
                const layerRef = layers.get(activeLayerIdentifier);
                targetElement = layerRef?.current || null;
            }

            if (!targetElement) return;

            const layerType = targetElement.tagName.toLowerCase();
            if (layerType !== behavior.elementType()) {
                logger.error(`Behavior ${behaviorId} (${behaviorName}) is attached to layer ${activeLayerIdentifier} ` +
                    `of wrong type ${layerType}: ${behavior.elementType()} expected`);
            } else {
                logger.info(`Behavior ${behaviorId} (${behaviorName}) is attached to layer ${activeLayerIdentifier} (${layerType})`);
            }
            behavior.setActiveElement(targetElement);

            // Camera controllers are element-bound (e.g. OrbitControls need the
            // canvas) AND own the camera. Now that the element is set, seed the
            // designated controllers with viewport dimensions and the initial
            // camera. Gated by `cameraControllerIds` (not duck-typing, so plain
            // camera listeners are not initialized as controllers) and run once
            // so effect re-runs don't clobber user-driven camera changes.
            if (cameraControllerIds.includes(behaviorId) &&
                !initializedControllers.current.has(behaviorId) &&
                isCameraController(behavior)) {
                const controller = behavior as unknown as CameraControllerInterface;
                const canvas = targetElement as HTMLCanvasElement;
                // TODO: not clear if this is correct here
                // Prefer the resize observer's computed size; fall back to the
                // canvas' own buffer size (or 500) before the first resize.
                const seedSize = lastCanvasSizeRef.current
                    ?? { width: canvas.width || 500, height: canvas.height || 500 };
                controller.setDimensions(seedSize.width, seedSize.height);
                if (props.camera_parameters) {
                    controller.setCameraParams(props.camera_parameters);
                }
                initializedControllers.current.add(behaviorId);
            }
        });
    }, [layers, behaviors, cameraControllerIds, props.camera_parameters]);

    // Once camera controllers are initialized (and as listeners register), the
    // active controller pushes its camera to all listeners. The binding effect
    // above seeds each controller with the initial camera; this propagates it on.
    useEffect(() => {
        maybeSendCameraUpdate();
        setTimeout(maybeSendCameraUpdate, 500);  // Guard against any race between camera/render behaviors
    }, [activeCameraController, behaviors]);

    // WebSocket initialization - runs when websocket_address changes.
    // Each viewer subscribes its own handlers to the (possibly shared) connection
    // so that incoming messages are dispatched to *this* viewer's behaviors too;
    // multiple viewers can share one address (see WebSocketConnectionsManager).
    useEffect(() => {
        const unsubscribers: Array<() => void> = [];
        for (const spec of websocketAddresses || []) {
            const wsAddress = typeof spec === 'string' ? spec : spec[0];
            const wsId = typeof spec === 'string' ? undefined : spec[1];
            try {
                // Subscribe under this viewer's unique id, so re-subscribing is
                // idempotent. subscribeToConnection always returns an unsubscribe fn.
                unsubscribers.push(WebSocketConnectionsManager.subscribeToConnection(
                    wsAddress,
                    wsId,
                    id,
                    handleWebSocketOpen,
                    handleWebSocketMessage,
                    handleWebSocketError,
                    handleWebSocketClose));
            } catch (error) {
                console.error(`Failed to subscribe to WebSocket connection: ${wsAddress}`, error);
            }
        }
        return () => unsubscribers.forEach((u) => u());
    }, [websocketAddresses]); // Only runs when websocketAddresses changes

    // General initialization - runs only once on mount
    useEffect(() => {
        animate();
        return () => {
            // Cancel any pending trailing wheel-settle send so it can't fire
            // (and touch behaviors) after the viewer unmounts.
            if (wheelSettleTimeout.current !== null) {
                clearTimeout(wheelSettleTimeout.current);
                wheelSettleTimeout.current = null;
            }
        };
    }, []); // Empty dependency array = runs only once


    const computeViewportSize = useCallback(() => {
        const container = document.getElementById(id);
        if (!container) return;

        const containerSize = { width: container.clientWidth, height: container.clientHeight };
        if (containerSize.width === 0 || containerSize.height === 0) return null;  // TODO   

        // Let us consider initial size as the effective pixel resolution, i.e. after device pixel ratio adjustment.
        const intrinsics = props.camera_parameters?.intrinsics as { width?: number, height?: number } | { width: 500, height: 500 };
        const dprInverse = 1.0 / (window.devicePixelRatio || 1);
        const initialSize = resizeMode == ViewportResizeMode.FIXED ? { width: intrinsics.width, height: intrinsics.height } :
            { width: Math.ceil(intrinsics.width * dprInverse), height: Math.ceil(intrinsics.height * dprInverse) };

        const frameSize = (resizeMode == ViewportResizeMode.ADAPTIVE ? containerSize : computeContainedFixedAspectSize(initialSize, containerSize));
        const displaySize = computeContainedDisplaySize(initialSize, resizeMode, containerSize);

        logger.debug(`[KaolinViewer ${id}] computed target viewport size: ${displaySize.width}x${displaySize.height} ` +
            `frame size: ${frameSize.width}x${frameSize.height} ` +
            `(container ${containerSize.width}x${containerSize.height}, initial ${initialSize.width}x${initialSize.height}, mode ${resizeMode})`);
        return { frame: frameSize, display: displaySize, useDPR: resizeMode != ViewportResizeMode.FIXED };
    }, [resizeMode, props.camera_parameters]);

    // Compute the contained canvas size for the current container, resize every
    // canvas layer's drawing buffer to it, push the size to all camera
    // controllers, and request a fresh frame. Reads latest layers/behaviors via
    // refs so it can stay stable (the resize observer captures it once).
    const applyViewportSize = useCallback(() => {
        const sizes = computeViewportSize();
        if (!sizes) return;  // container not ready

        // Let's modify default layers
        // Inner CSS container sizing:
        //  - Adaptive --> 100%, 100%
        //  - Fixed/Fixed aspect --> px sizes, not accounting for devicePixelRatio
        defaultSetElementDimensions(viewerFrameRef.current, sizes.frame.width, sizes.frame.height);

        if (resizeMode === ViewportResizeMode.FIXED) return;

        defaultSetElementDimensions(eventCanvasRef.current, sizes.display.width, sizes.display.height, sizes.useDPR);

        // Let's modify all layers
        layersRef.current.forEach((layerRef, _layerId) => {
            if (layerRef.current) {
                defaultSetElementDimensions(layerRef.current, sizes.display.width, sizes.display.height, sizes.useDPR);
            }
        });

        // Also inform all behaviors that have a corresponding method
        behaviors.forEach((behavior, identifier) => {
            if ((behavior as any)['setDimensions']) {
                (behavior as any)['setDimensions'](sizes.display.width, sizes.display.height);
            }
        });

        lastCanvasSizeRef.current = sizes.display;

        // TODO: this is too slow
        // Immediate camera update so listeners stay in sync during a resize drag.
        maybeSendCameraUpdate();
    }, [computeViewportSize, cameraControllerIds, maybeSendCameraUpdate, resizeMode]);

    // Fire on viewer container size change (also fires once on observe, i.e. on
    // mount) to resize canvas layers and update camera controllers.
    // The rAF wrapper coalesces multiple ResizeObserver firings within one frame
    // (common during smooth resize drags) into a single applyViewportSize call,
    // capping work to the display refresh rate without adding perceptible lag.
    // The deferred maybeSendCameraUpdate fires only after resizing has been idle
    // for 300 ms (i.e. drag stopped), resetting on every new ResizeObserver entry.
    useEffect(() => {
        // FIXED mode keeps a constant canvas size regardless of the container, so
        // there's nothing to react to — skip installing the ResizeObserver entirely.
        const container = document.getElementById(id);
        if (!container) return;
        let pendingRaf: number | null = null;
        let pendingTimeout: ReturnType<typeof setTimeout> | null = null;
        const resizeObserver = new ResizeObserver(() => {
            if (pendingRaf !== null) cancelAnimationFrame(pendingRaf);
            if (pendingTimeout !== null) clearTimeout(pendingTimeout);
            pendingRaf = requestAnimationFrame(() => {
                pendingRaf = null;
                applyViewportSize();
                pendingTimeout = setTimeout(maybeSendCameraUpdate, 300);
            });
        });
        resizeObserver.observe(container);
        return () => {
            if (pendingRaf !== null) cancelAnimationFrame(pendingRaf);
            if (pendingTimeout !== null) clearTimeout(pendingTimeout);
            resizeObserver.disconnect();
        };
    }, [id, applyViewportSize, maybeSendCameraUpdate, resizeMode]);

    // Re-apply when layers or behaviors change so newly mounted canvas layers
    // get the current size even if the container itself did not resize.
    useEffect(() => {
        applyViewportSize();
    }, [layers, behaviors, applyViewportSize]);

    // Requirements on the size...
    // v0: div fills the available space, including changing aspect ratio, can change server camera
    // v1: div fills available space, but keeps aspect ratio (determined by input camera)
    // 
    // function resizeCanvas() {
    //var width = $("#palette-panel").width();
    //$("#palette-canvas").css("width", width);
    //$("#palette-canvas").css("height", width);
    //}
    return (
        <div id={id} className='kaolin-viewport-sizing-container'>
            <div className='kaolin-viewer-container' ref={viewerFrameRef}
                onMouseEnter={() => { isHoveredRef.current = true; }}
                onMouseLeave={() => { isHoveredRef.current = false; }}
            >
                <div className='kaolin-version-overlay'>Kaolin 0.18.0</div>
                {fpsTrackerRef.current && (
                    <InteractiveFpsProvider tracker={fpsTrackerRef.current}>
                        <FpsReadout className='kaolin-fps-overlay' customString={props.remote_fps_label} />
                    </InteractiveFpsProvider>
                )}
                {modeDisplayList.some(m => m.icon) && (
                    <div className='kaolin-mode-overlay'>
                        {modeDisplayList
                            .filter(m => m.icon)
                            .map(m => {
                                const isActive = activeModeName === m.name;
                                const label = m.description ?? m.name;
                                const tooltip = m.shortcut ? `${label} (${m.shortcut})` : label;
                                return (
                                    <button
                                        key={m.name}
                                        type='button'
                                        className={`kaolin-mode-btn${isActive ? ' active' : ''}`}
                                        title={tooltip}
                                        onClick={() => applyMode(m.name)}
                                        aria-pressed={isActive}
                                        aria-label={label}
                                    >
                                        <i className={`bi ${m.icon}`} aria-hidden='true'></i>
                                    </button>
                                );
                            })}
                    </div>
                )}
                {props.layers?.map(([identifier, name, options]) => (
                    <LayerComponent
                        key={identifier}
                        layerIdentifier={identifier}
                        elementType={name}
                        layerProps={{ id: identifier, ...options }}
                        onRegister={registerLayer}
                    />
                ))}
                {props.behaviors?.map(([identifier, name, _layerIdentifier, options]) => (
                    <BehaviorRunner
                        behaviorName={name}
                        behaviorOptions={options}
                        behaviorIdentifier={identifier}
                        onRegister={registerBehavior}
                    />
                ))}
                <canvas
                    width={(props.camera_parameters?.intrinsics as { width?: number })?.width ?? 500}
                    height={(props.camera_parameters?.intrinsics as { height?: number })?.height ?? 500}
                    ref={eventCanvasRef}
                    className='sub-canvas'
                    onClick={handleCanvasClick}
                    onDoubleClick={handleCanvasDoubleClick}
                    onPointerDown={handleCanvasPointerDown}
                    onPointerMove={handleCanvasPointerMove}
                    onPointerUp={handleCanvasPointerUp}
                    onPointerCancel={handleCanvasPointerCancel}
                    onPointerEnter={handleCanvasPointerEnter}
                    onPointerLeave={handleCanvasPointerLeave}
                    onWheel={handleCanvasWheel}
                ></canvas>
            </div>
        </div>
    )
}

KaolinViewerInternal.defaultProps = {
    websocket_address: undefined
};

export default KaolinViewerInternal;

/**
 * 1. react-resizable-panels (Highly Recommended)
This is one of the best and most modern libraries for creating flexible, accessible, and high-performance split layouts.

Approach: Uses PanelGroup, Panel, and PanelResizeHandle components to structure your layout declaratively.

Features: Supports both horizontal and vertical splitting, collapse/expand states, and optional automatic layout persistence (saving the user's size preferences to local storage).

Control: You place the <PanelResizeHandle /> component directly between your <Panel> components, and that handle is the draggable divider.
 */