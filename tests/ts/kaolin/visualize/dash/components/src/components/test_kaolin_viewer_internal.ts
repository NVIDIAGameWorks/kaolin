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

import { assert } from 'chai';
import sinon from 'sinon';
import React, { act, forwardRef, useImperativeHandle } from 'react';
import { createRoot, Root } from 'react-dom/client';

import { BehaviorRegister, InteractiveBehavior } from '@kaolin/core/behavior';
import { registerDom, unregisterDom } from '@test/helpers/dom';

describe('visualize/dash/components/src/components/test_kaolin_viewer_internal.ts', () => {
    let Viewer: React.ComponentType<any>;
    let root: Root;
    let container: HTMLDivElement;
    let sandbox: sinon.SinonSandbox;
    let previousActEnvironment: any;

    before(() => {
        registerDom();
        previousActEnvironment = (globalThis as any).IS_REACT_ACT_ENVIRONMENT;
        (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
        // Node needs no stylesheet; real layout is checked separately in Chrome.
        const cssLoader = require.extensions['.css'];
        require.extensions['.css'] = () => {};
        try {
            Viewer = require('@kaolin/components/KaolinViewerInternal').default;
        } finally {
            if (cssLoader) require.extensions['.css'] = cssLoader;
            else delete require.extensions['.css'];
        }
    });
    after(async () => {
        (globalThis as any).IS_REACT_ACT_ENVIRONMENT = previousActEnvironment;
        await unregisterDom();
    });
    beforeEach(() => {
        sandbox = sinon.createSandbox();
        container = document.createElement('div');
        document.body.appendChild(container);
        root = createRoot(container);
    });
    afterEach(() => {
        act(() => root.unmount());
        container.remove();
        sandbox.restore();
    });

    for (const asClass of [true, false]) {
        for (const dpr of [1, 2]) {
            it(`binds ${asClass ? 'class' : 'React'} renderers before same-aspect resizing at DPR ${dpr}`, () => {
                sandbox.stub(window, 'devicePixelRatio').value(dpr);
                const requests = new Map<string, string[]>();
                const getContext = HTMLCanvasElement.prototype.getContext;
                // Record renderer context requests without needing a GPU in happy-dom.
                sandbox.stub(HTMLCanvasElement.prototype, 'getContext').callsFake(function (type: any, ...args: any[]) {
                    if (!this.id.startsWith('render-')) return getContext.apply(this, [type, ...args] as any);
                    const calls = requests.get(this.id) ?? [];
                    calls.push(type);
                    requests.set(this.id, calls);
                    return type === 'webgl2' ? {} as any : null;
                });
                class Renderer extends InteractiveBehavior {
                    override elementType() { return 'canvas'; }
                    override setActiveElement(element: HTMLElement) {
                        super.setActiveElement(element);
                        (element as HTMLCanvasElement).getContext('webgl2');
                    }
                    override setDimensions(width: number, height: number) {
                        assert.isNotNull(this.element, 'renderer must bind before sizing');
                        const canvas = this.element as HTMLCanvasElement;
                        assert.equal(canvas.width, width * dpr, 'layer buffer is resized before the behavior');
                        assert.equal(canvas.height, height * dpr);
                    }
                }
                const RendererComponent = forwardRef((_props, ref) => {
                    useImperativeHandle(ref, () => new Renderer(), []);
                    return null;
                });
                sandbox.stub(BehaviorRegister, 'has').returns(true);
                sandbox.stub(BehaviorRegister, 'isClass').returns(asClass);
                sandbox.stub(BehaviorRegister, 'get').returns(asClass ? Renderer : RendererComponent);

                for (const ids of [['render-first'], ['render-first', 'render-late']]) {
                    act(() => root.render(React.createElement(Viewer, {
                        id: 'viewer',
                        setProps: () => {},
                        camera_parameters: { intrinsics: { width: 64, height: 32 } },
                        layers: ids.map(id => [id, 'canvas', { width: 32, height: 16 }]),
                        behaviors: ids.map(id => [id + '-behavior', 'renderer', id, {}, true]),
                    })));
                    for (const id of ids) {
                        assert.equal(requests.get(id)?.[0], 'webgl2', 'renderer acquires the first context');
                        const canvas = document.getElementById(id) as HTMLCanvasElement;
                        assert.equal(canvas.width, 64 * dpr);
                        assert.equal(canvas.height, 32 * dpr);
                    }
                }
            });
        }
    }
});
