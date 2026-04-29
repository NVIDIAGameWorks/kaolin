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

// NOTE: useSendValue is a hook, so it is driven through the real exported
// components (which expose its setValue/setCamera via an imperative handle) so
// the initialization effects actually run. The components are mounted into a
// happy-dom DOM with react-dom and flushed with React's act(); the WebSocket
// layer is faked with a sinon stub on WebSocketConnectionsManager. Log level is
// raised to ERROR so the only console output we ever assert on is the
// not-registered error.

import { assert } from 'chai';
import sinon from 'sinon';
import React, { act } from 'react';
import { createRoot, Root } from 'react-dom/client';

import {
    SendValueBehaviorComponent,
    SendValueBehaviorHandle,
    SendCameraBehaviorComponent,
    SendCameraHandle,
    SendValueOptions,
} from '@kaolin/lib/behavior/send_message';
import * as io from '@kaolin/core/io';
import { CameraParameters, CameraPinholeIntrinsics, defaultCameraParameters } from '@kaolin/graphics/camera';
import { WebSocketConnectionsManager } from '@kaolin/core/sockets';
import { setLogLevel, LogLevel } from '@kaolin/util/logging';
import { registerDom, unregisterDom } from '@test/helpers/dom';
import { captureConsole } from '@test/helpers/console';

const CONN = 'test-conn';
const TAG = 'test_value';

describe('visualize/dash/components/src/lib/behavior/test_send_message.ts', () => {

    // The components need a DOM (react-dom) and React's act() environment.
    before(() => {
        registerDom();
        (globalThis as any).IS_REACT_ACT_ENVIRONMENT = true;
    });
    after(async () => { await unregisterDom(); });

    let sandbox: sinon.SinonSandbox;
    let roots: Root[] = [];

    beforeEach(() => {
        sandbox = sinon.createSandbox();
        setLogLevel(LogLevel.ERROR);
    });

    afterEach(() => {
        act(() => { roots.forEach(root => root.unmount()); });
        roots = [];
        sandbox.restore();
        delete (window as any).__sendTest;  // where mock up functions live
        setLogLevel(LogLevel.DEBUG);
    });

    // --- helpers ----------------------------------------------------------

    /** Register an always-open fake connection; returns its recording `send` spy. */
    function stubOpenConnection(): sinon.SinonSpy {
        const send = sandbox.spy();
        sandbox.stub(WebSocketConnectionsManager, 'hasConnection').returns(true);
        sandbox.stub(WebSocketConnectionsManager, 'getOpenConnection').returns({ send } as any);
        return send;
    }

    /** Mount a forwardRef behavior component, flush its effects, and return its handle. */
    function mountBehavior<H>(component: any, props: object): H {
        const ref = React.createRef<H>();
        const root = createRoot(document.createElement('div'));
        roots.push(root);
        act(() => { root.render(React.createElement(component, { ...props, ref } as any)); });
        return (ref as any).current as H;
    }

    /** Mount a SendValueBehaviorComponent with test-friendly defaults (no throttle, JSON). */
    function mountSender(opts: Partial<SendValueOptions> = {}): SendValueBehaviorHandle {
        return mountBehavior<SendValueBehaviorHandle>(SendValueBehaviorComponent,
            { connectionId: CONN, messageTag: TAG, minUpdateMs: 0, binary: false, ...opts });
    }

    /** Run an imperative-handle call and await the async send it triggers. */
    async function callAndFlush(fn: () => void): Promise<void> {
        await act(async () => { await (fn() as unknown as Promise<void>); });
    }

    /** Decode one sent payload (binary or JSON) into its {tag, msg} pair. */
    function decodePayload(payload: any): { tag: any; msg: any } {
        const message = typeof payload === 'string' ? io.fromJSON(payload) : io.fromBinary(payload);
        return { tag: message.get(io.MESSAGE_TAG_KEY), msg: message.get(io.MESSAGE_CONTENT_KEY) };
    }

    // --- options ----------------------------------------------------------

    describe('SendValueBehaviorComponent respects options', () => {

        it('messageTag and toSerializableFunctionName shape the payload', async () => {
            const send = stubOpenConnection();
            (window as any).__sendTest = { serialize: (v: string) => `wrapped:${v}` };
            const handle = mountSender({ messageTag: 'custom_tag', toSerializableFunctionName: '__sendTest.serialize' });

            await callAndFlush(() => handle.setValue('hello'));

            const { tag, msg } = decodePayload(send.firstCall.args[0]);
            assert.equal(tag, 'custom_tag', 'payload carries the configured messageTag');
            assert.equal(msg, 'wrapped:hello', 'payload content is produced by toSerializableFunction');
        });

        it('binary selects the ArrayBuffer vs JSON-string wire format', async () => {
            const send = stubOpenConnection();
            const jsonHandle = mountSender({ binary: false });
            const binaryHandle = mountSender({ binary: true });
            await callAndFlush(() => jsonHandle.setValue('v-json'));
            await callAndFlush(() => binaryHandle.setValue('v-bin')); 

            assert.isString(send.firstCall.args[0], 'binary:false sends a JSON string');
            assert.isNotString(send.secondCall.args[0], 'binary:true sends bytes, not a string');
            assert.equal(decodePayload(send.firstCall.args[0]).msg, 'v-json', 'JSON payload round-trips the value');
            assert.equal(decodePayload(send.secondCall.args[0]).msg, 'v-bin', 'binary payload round-trips the value');
        });

        it('minUpdateMs throttles repeat sends unless force=true', async () => {
            const send = stubOpenConnection();
            const handle = mountSender({ minUpdateMs: 100000 });  // far longer than the test runs

            await callAndFlush(() => handle.setValue('a'));               // first send always goes
            await callAndFlush(() => handle.setValue('b'));               // within the window -> dropped
            await callAndFlush(() => (handle.setValue as any)('c', true)); // force bypasses the throttle

            assert.equal(send.callCount, 2, 'throttled middle send is dropped; forced send still goes through');
            assert.equal(decodePayload(send.firstCall.args[0]).msg, 'a', 'first value sent immediately');
            assert.equal(decodePayload(send.secondCall.args[0]).msg, 'c', 'forced value is the second one delivered');
        });

        it('needsUpdateFunctionName decides whether a value warrants sending', async () => {
            const send = stubOpenConnection();
            (window as any).__sendTest = { unchanged: () => false };
            const handle = mountSender({ needsUpdateFunctionName: '__sendTest.unchanged' });

            await callAndFlush(() => handle.setValue('x'));               // change-detector says "no change"
            assert.isTrue(send.notCalled, 'nothing sent when needsUpdateFunction reports no change');

            await callAndFlush(() => (handle.setValue as any)('x', true)); // force ignores change detection
            assert.isTrue(send.calledOnce, 'forced send ignores the change detector');
        });
    });

    // --- end-to-end -------------------------------------------------------

    describe('SendValueBehaviorComponent basic functionality', () => {

        it('looks up the configured connection and delivers the tagged value', async () => {
            const send = stubOpenConnection();
            const handle = mountSender({ connectionId: 'my-conn', messageTag: 'set_thing' });

            await callAndFlush(() => handle.setValue({ hello: 'world' }));

            assert.isTrue((WebSocketConnectionsManager.hasConnection as sinon.SinonStub).calledWith('my-conn'),
                'connection is looked up by the configured connectionId');
            const { tag, msg } = decodePayload(send.firstCall.args[0]);
            assert.equal(tag, 'set_thing', 'message carries the configured tag');
            assert.deepEqual(msg, new Map([['hello', 'world']]), 'message content round-trips the sent value');
        });

        it('round-trips structured nested values through both wire formats', async () => {
            const send = stubOpenConnection();
            const meta = { a: 5, b: 'hello' };                  // shared, decodes to a Map either way
            const expectedMeta = new Map<string, any>([['a', 5], ['b', 'hello']]);
            const binaryHandle = mountSender({ binary: true });
            const jsonHandle = mountSender({ binary: false });

            await callAndFlush(() => binaryHandle.setValue({ image: new Uint8Array([10, 20, 30, 40]), meta }));
            await callAndFlush(() => jsonHandle.setValue({ matrix: [1, 3, 4, 5, 9], meta }));

            // Binary: typed arrays survive as typed arrays (UINT8 decodes to a
            // Uint8ClampedArray); nested dicts become Maps.
            const binMsg = decodePayload(send.firstCall.args[0]).msg as Map<string, any>;
            assert.instanceOf(binMsg.get('image'), Uint8ClampedArray, 'binary: image decodes back to a byte array');
            assert.deepEqual(Array.from(binMsg.get('image') as Uint8ClampedArray), [10, 20, 30, 40],
                'binary: image bytes survive the round-trip');
            assert.deepEqual(binMsg.get('meta'), expectedMeta, 'binary: nested meta decodes to a Map of its entries');

            // JSON: arrays stay arrays; nested objects become Maps (parity with fromBinary).
            const jsonMsg = decodePayload(send.secondCall.args[0]).msg as Map<string, any>;
            assert.deepEqual(jsonMsg.get('matrix'), [1, 3, 4, 5, 9], 'json: matrix array survives the round-trip');
            assert.deepEqual(jsonMsg.get('meta'), expectedMeta, 'json: nested meta decodes to a Map of its entries');
        });

        it('logs an error and sends nothing when the connection is not registered', async () => {
            sandbox.stub(WebSocketConnectionsManager, 'hasConnection').returns(false);
            const getOpen = sandbox.stub(WebSocketConnectionsManager, 'getOpenConnection');
            const handle = mountSender({ connectionId: 'missing' });

            const { calls, restore } = captureConsole();
            try {
                await callAndFlush(() => handle.setValue('v'));
            } finally { restore(); }

            assert.isTrue(getOpen.notCalled, 'no socket fetched for an unregistered connection');
            assert.equal(calls.length, 1, 'exactly one log line');
            assert.equal(calls[0].method, 'error', 'logged at error level');
            assert.include(String(calls[0].args[0]), 'not registered', 'error explains the connection is not registered');
        });
    });

    // --- delayed connection ----------------------------------------------

    describe('SendValueBehaviorComponent test delayed connection', () => {

        it('polls a registered-but-not-yet-open connection until it opens', async () => {
            const send = sandbox.spy();
            sandbox.stub(WebSocketConnectionsManager, 'hasConnection').returns(true);
            const getOpen = sandbox.stub(WebSocketConnectionsManager, 'getOpenConnection');
            getOpen.onFirstCall().returns(undefined as any);  // not open on the first look
            getOpen.returns({ send } as any);                 // opens on a subsequent retry

            const handle = mountSender();
            await callAndFlush(() => handle.setValue('late'));

            assert.isAbove(getOpen.callCount, 1, 'polled again after the initial miss');
            assert.isTrue(send.calledOnce, 'message delivered once the connection opened');
            assert.equal(decodePayload(send.firstCall.args[0]).msg, 'late', 'the awaited value is the one delivered');
        });
    });

    // --- sibling component ------------------------------------------------

    describe('SendCameraBehaviorComponent', () => {

        const focalOf = (cam: CameraParameters) => (cam.intrinsics as CameraPinholeIntrinsics).focal_x;
        /** A real default camera with its focal length nudged by `df` pixels. */
        const tweakedCamera = (df: number): CameraParameters => ({
            ...defaultCameraParameters,
            intrinsics: { ...defaultCameraParameters.intrinsics, focal_x: focalOf(defaultCameraParameters) + df },
        });
        /** A tiny real delay so the (minUpdateMs=0) throttle never masks the change-detector. */
        const passThrottle = () => new Promise(resolve => setTimeout(resolve, 5));

        // Realistic camera streaming: serialize only the intrinsics, and re-send
        // only when the focal length (fov) moves by more than a threshold.
        it('streams only the intrinsics and re-sends only when the fov changes enough', async () => {
            const send = stubOpenConnection();
            (window as any).__sendTest = {
                intrinsicsOnly: (cam: CameraParameters) => cam.intrinsics,
                fovChanged: (prev: CameraParameters | null, curr: CameraParameters) =>
                    prev === null || Math.abs(focalOf(prev) - focalOf(curr)) > 1.0,
            };
            const handle = mountBehavior<SendCameraHandle>(SendCameraBehaviorComponent, {
                connectionId: CONN, minUpdateMs: 0, binary: false,
                toSerializableFunctionName: '__sendTest.intrinsicsOnly',
                needsUpdateFunctionName: '__sendTest.fovChanged',
            });

            // First camera always goes; payload is just the intrinsics under the default tag.
            await callAndFlush(() => handle.setCamera(defaultCameraParameters));
            const { tag, msg } = decodePayload(send.firstCall.args[0]);
            assert.equal(tag, 'set_camera', 'camera behavior defaults to the set_camera tag');
            assert.deepEqual(msg, new Map(Object.entries(defaultCameraParameters.intrinsics)),
                'toSerializable trims the payload down to the camera intrinsics');

            // A sub-threshold focal change is suppressed by needsUpdate.
            await passThrottle();
            await callAndFlush(() => handle.setCamera(tweakedCamera(0.1)));
            assert.equal(send.callCount, 1, 'a tiny fov change is not re-sent');

            // A large focal change clears the threshold and is sent.
            await passThrottle();
            await callAndFlush(() => handle.setCamera(tweakedCamera(50)));
            assert.equal(send.callCount, 2, 'a large fov change is re-sent');
        });
    });

});
