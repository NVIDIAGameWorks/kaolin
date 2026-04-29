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
 * FPS tracking for interactive Kaolin viewers.
 *
 * **Key classes / hooks:**
 * - {@link InteractiveFps} — framework-agnostic burst-aware sliding-window tracker.
 *   Measures FPS only over the most recent activity burst; returns `null` while idle
 *   so overlays never show a stale or misleading rate.
 * - {@link InteractiveFpsProvider} — React context provider that owns (or wraps an
 *   external) `InteractiveFps` instance and re-renders consumers at a configurable
 *   `refreshMs` interval.
 * - {@link useFps} — hook that subscribes to the current {@link FpsValue} snapshot.
 * - {@link useFrameReceiver} — hook that returns a stable callback to call each time
 *   a frame arrives; does not itself cause a re-render.
 * - {@link FpsReadout} — zero-config overlay component; shows an em dash while idle.
 *
 * **Typical usage inside a viewer component:**
 * ```tsx
 * // Wrap the subtree that both produces and displays frames.
 * <InteractiveFpsProvider windowMs={1000} idleThresholdMs={500} refreshMs={250}>
 *   <ViewportCanvas />   // calls useFrameReceiver() and notifies on each render
 *   <FpsReadout />       // re-renders at most once per refreshMs
 * </InteractiveFpsProvider>
 *
 * // Inside ViewportCanvas:
 * const frameReceived = useFrameReceiver();
 * // ... after painting each frame:
 * frameReceived();
 * ```
 *
 * @module
 */

import React, { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react';

// TODO: this module should be a behavior

/** Constructor options for {@link InteractiveFps}. */
export interface InteractiveFpsOptions {
    /** Max age (ms) of frame timestamps kept in the sliding window. Defaults to 1000. */
    windowMs?: number;
    /**
     * Gap (ms) between consecutive frames that marks the end of a render burst.
     * Larger than the longest frame you'd expect inside a burst (≈3× frame time
     * is a reasonable floor) but small enough to clearly separate bursts.
     * Defaults to 500.
     */
    idleThresholdMs?: number;
    /** Time source; injectable for tests. Defaults to `performance.now`. */
    now?: () => number;
}

/**
 * Sliding-window FPS tracker tuned for render-on-demand viewers.
 *
 * In a continuous-render pipeline FPS is just `frames / elapsedTime`, but
 * interactive viewers only render when something changes (drag, hover, server
 * push) so frame arrivals come in bursts separated by arbitrarily long idle
 * gaps. A naive moving average would drag the reported rate toward zero during
 * those gaps even though the burst itself may have been smooth.
 *
 * This class measures FPS only over the most recent `windowMs` of frame
 * arrivals, and treats any gap longer than `idleThresholdMs` as the end of the
 * current burst: samples accumulated before the gap are discarded so the next
 * burst's rate isn't polluted by the pre-idle samples. While idle (no frame
 * within `idleThresholdMs`) {@link getFps} returns `null` so callers can render
 * a neutral indicator (e.g. "—") rather than a stale or zero number.
 *
 * Framework-agnostic; React consumers should use {@link InteractiveFpsProvider}
 * together with {@link useFps} / {@link useFrameReceiver}.
 */
export class InteractiveFps {
    private readonly windowMs: number;
    private readonly idleThresholdMs: number;
    private readonly now: () => number;
    private frames: number[] = [];

    constructor(options: InteractiveFpsOptions = {}) {
        this.windowMs = options.windowMs ?? 1000;
        this.idleThresholdMs = options.idleThresholdMs ?? 500;
        this.now = options.now ?? (() => performance.now());
    }

    /**
     * Record that a frame has just been rendered. If the gap since the previous
     * frame exceeds `idleThresholdMs`, prior samples are discarded so the new
     * burst is measured in isolation.
     */
    frameReceived(): void {
        const t = this.now();
        const prev = this.frames.length > 0 ? this.frames[this.frames.length - 1] : null;
        if (prev !== null && t - prev > this.idleThresholdMs) {
            this.frames.length = 0;
        }
        this.frames.push(t);
        this.trim(t);
    }

    /**
     * Current FPS averaged over the sliding window, or `null` when no rate
     * can be meaningfully reported (fewer than two samples, or currently idle).
     */
    getFps(): number | null {
        const t = this.now();
        this.trim(t);
        if (this.frames.length < 2) return null;
        const last = this.frames[this.frames.length - 1];
        if (t - last > this.idleThresholdMs) return null;
        const first = this.frames[0];
        const elapsedMs = t - first;
        if (elapsedMs <= 0) return null;
        return (this.frames.length * 1000) / elapsedMs;
    }

    /** True iff no frame has arrived within `idleThresholdMs`. */
    isIdle(): boolean {
        if (this.frames.length === 0) return true;
        return this.now() - this.frames[this.frames.length - 1] > this.idleThresholdMs;
    }

    /** Number of frame samples currently retained in the window. */
    sampleCount(): number {
        return this.frames.length;
    }

    /** Discard all accumulated samples. */
    reset(): void {
        this.frames.length = 0;
    }

    private trim(t: number): void {
        const cutoff = t - this.windowMs;
        let drop = 0;
        while (drop < this.frames.length && this.frames[drop] < cutoff) drop++;
        if (drop > 0) this.frames.splice(0, drop);
    }
}


// ---------------------------------------------------------------------------
// React layer: Provider + hooks
// ---------------------------------------------------------------------------

/**
 * Snapshot of the tracker exposed to React consumers. `fps` is `null` while
 * idle or before two samples have arrived (see {@link InteractiveFps.getFps}).
 */
export interface FpsValue {
    fps: number | null;
    isIdle: boolean;
}

const IDLE_VALUE: FpsValue = { fps: null, isIdle: true };

// Two contexts on purpose: components that only *publish* frames (the viewer)
// shouldn't re-render every time the displayed FPS ticks. Splitting the
// receiver from the value keeps publisher renders stable.
const FpsValueContext = createContext<FpsValue>(IDLE_VALUE);
const FrameReceiverContext = createContext<(timestamp?: number) => void>(() => { });

export interface InteractiveFpsProviderProps extends InteractiveFpsOptions {
    /**
     * Optional externally-owned tracker. Pass when the *producer* of frames
     * sits above the provider in the React tree (e.g. the viewer itself
     * receives the WebSocket messages but wraps only its overlay in the
     * provider). When omitted, the provider lazily creates and owns its own
     * tracker — useful when producer and consumer live inside the same
     * subtree.
     */
    tracker?: InteractiveFps;
    /**
     * How often (ms) the provider polls the underlying tracker and re-renders
     * consumers if the displayed value changed. Decoupled from the actual
     * frame rate so consumers don't re-render per frame. Defaults to 250 ms.
     */
    refreshMs?: number;
    children?: React.ReactNode;
}

/**
 * Provider that hosts an {@link InteractiveFps} instance and exposes its
 * current value to descendant components. Wrap the part of the tree that both
 * produces frames (calls {@link useFrameReceiver}) and consumes FPS (calls
 * {@link useFps}).
 *
 * Option props are read once at mount; remount the provider to change them.
 * This avoids subtle bugs where an unmemoized inline option object would
 * silently rebuild the tracker and lose its history.
 */
export const InteractiveFpsProvider: React.FC<InteractiveFpsProviderProps> = ({
    tracker: externalTracker,
    windowMs,
    idleThresholdMs,
    now,
    refreshMs = 250,
    children,
}) => {
    const internalTrackerRef = useRef<InteractiveFps | null>(null);
    if (!externalTracker && internalTrackerRef.current === null) {
        internalTrackerRef.current = new InteractiveFps({ windowMs, idleThresholdMs, now });
    }
    const tracker = externalTracker ?? (internalTrackerRef.current as InteractiveFps);

    const [value, setValue] = useState<FpsValue>(IDLE_VALUE);

    useEffect(() => {
        const tick = () => {
            const fps = tracker.getFps();
            const isIdle = tracker.isIdle();
            setValue(prev => (prev.fps === fps && prev.isIdle === isIdle ? prev : { fps, isIdle }));
        };
        const id = window.setInterval(tick, refreshMs);
        return () => window.clearInterval(id);
    }, [tracker, refreshMs]);

    const frameReceived = useCallback(() => {
        tracker.frameReceived();
    }, [tracker]);

    return (
        <FrameReceiverContext.Provider value={frameReceived}>
            <FpsValueContext.Provider value={value}>
                {children}
            </FpsValueContext.Provider>
        </FrameReceiverContext.Provider>
    );
};

/**
 * Subscribe to the current FPS value. The calling component re-renders only
 * when `fps` or `isIdle` actually change (at most once per `refreshMs`).
 * Outside an {@link InteractiveFpsProvider} returns the idle sentinel so
 * overlays can render harmlessly when no provider is mounted.
 *
 * @returns The current {@link FpsValue} snapshot.
 */
export function useFps(): FpsValue {
    return useContext(FpsValueContext);
}

/**
 * Stable callback to invoke when a frame has been rendered. Calling it does
 * not re-render the component; the value is fanned out to {@link useFps}
 * consumers on the provider's refresh tick. Outside an
 * {@link InteractiveFpsProvider} this is a no-op.
 *
 * @returns Stable frame-receiver callback; a no-op outside a provider.
 */
export function useFrameReceiver(): () => void {
    return useContext(FrameReceiverContext);
}


/**
 * Tiny default overlay component. Render somewhere inside an
 * {@link InteractiveFpsProvider} for a no-fuss FPS readout; shows an em dash
 * while idle so the indicator never reports a misleading rate.
 */
export const FpsReadout: React.FC<{ className?: string; digits?: number; customString?: string }> = ({
    className,
    digits = 0,
    customString,
}) => {
    const { fps } = useFps();
    return (
        <span className={className}>
            {fps == null ? '—' : fps.toFixed(digits)} fps{customString ? ` ${customString}` : ''}
        </span>
    );
};
