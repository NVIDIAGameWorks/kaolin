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

// NOTE: InteractiveFps accepts an injectable `now` clock, so all timing tests
// can run without real wall-clock delays. The React layer (InteractiveFpsProvider,
// useFps, useFrameReceiver, FpsReadout) requires a React + jsdom setup and is
// intentionally left for a dedicated React component test file.

import { assert } from 'chai';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { InteractiveFps, InteractiveFpsProvider, FpsReadout } from '@kaolin/util/fps';

class CustomNow {
    t: number;

    constructor() {
        this.t = 0;
    }

    now(): number {
        return this.t;
    }
}

describe('visualize/dash/components/src/util/test_fps.ts', () => {
    describe('InteractiveFps', () => {
        it('properly tracks windowed fps and idle status', () => {
            let myNow = new CustomNow();
            const fps = new InteractiveFps({ windowMs: 200, idleThresholdMs: 100, now: () => myNow.now() });

            assert.equal(fps.sampleCount(), 0, 'fresh tracker has no frame samples');
            assert.isTrue(fps.isIdle(), 'fresh tracker is idle before any frames');
            assert.isNull(fps.getFps(), 'getFps is null with zero samples');

            fps.frameReceived();
            assert.equal(fps.sampleCount(), 1, 'one frameReceived increments sampleCount to 1');
            assert.isNull(fps.getFps(), 'getFps is null with only one sample');
            myNow.t = 50; fps.frameReceived();
            assert.equal(fps.getFps(), 2 / 0.05, 'getFps reflects two frames over 50 ms');
            myNow.t = 100; fps.frameReceived();
            myNow.t = 150; fps.frameReceived();
            assert.approximately(fps.getFps(), 4 / 0.150, 0.1, 'getFps reflects four frames over 150 ms');
            assert.isFalse(fps.isIdle(), 'tracker is active immediately after a frame burst');
            myNow.t = 200;
            assert.isFalse(fps.isIdle(), 'tracker stays active within idleThresholdMs of last frame');
            assert.approximately(fps.getFps(), 4 / 0.20, 0.1, 'getFps still reflects in-window frames at t=200 ms');
            myNow.t = 300;
            assert.isTrue(fps.isIdle(), 'tracker is idle once idleThresholdMs has passed since last frame');
            assert.isNull(fps.getFps(), 'getFps is null while idle');
        });
    });
    describe('React End to End', () => {
        // Dependency-free smoke test using server-side rendering: it verifies the
        // provider mounts and the readout reads the initial (idle) context value.
        // SSR does NOT run effects, so the provider's setInterval tick never fires;
        // the reactive frame -> tick -> readout-update path needs a DOM + act()
        // (jsdom) and is out of scope here.
        it('renders the idle readout through the provider', () => {
            const html = renderToStaticMarkup(
                React.createElement(
                    InteractiveFpsProvider,
                    { now: () => 0 },
                    React.createElement(FpsReadout, null),
                ),
            );
            assert.include(html, '\u2014', 'idle readout shows an em dash');
            assert.include(html, 'fps', 'readout shows the fps label');
        });
    });
});
   