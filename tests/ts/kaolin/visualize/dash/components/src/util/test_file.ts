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
import { downloadBlob, downloadTextFile, uploadImage } from '@kaolin/util/file';
import { registerDom, unregisterDom, makeCanvasElement } from '@test/helpers/dom';

describe('visualize/dash/components/src/util/test_file.ts', () => {

    // These helpers drive a real <a download> against the happy-dom DOM; sinon
    // spies the object-URL lifecycle and anchor click, and fakes the deferred
    // revoke timer so the assertions stay synchronous.
    before(registerDom);
    after(unregisterDom);

    let sandbox: sinon.SinonSandbox;
    let clock: sinon.SinonFakeTimers;
    let createUrl: sinon.SinonStub;
    let revokeUrl: sinon.SinonStub;
    let click: sinon.SinonStub;

    beforeEach(() => {
        sandbox = sinon.createSandbox();
        clock = sandbox.useFakeTimers();
        let counter = 0;
        createUrl = sandbox.stub(URL, 'createObjectURL').callsFake(() => `blob:mock/${counter++}`);
        revokeUrl = sandbox.stub(URL, 'revokeObjectURL');
        click = sandbox.stub(HTMLAnchorElement.prototype, 'click');
    });

    afterEach(() => sandbox.restore());

    describe('downloadBlob', () => {
        it('builds a clickable <a download> from the blob, then revokes the URL after the click', () => {
            const blob = new Blob(['hi'], { type: 'text/plain' });
            downloadBlob('out.bin', blob);

            assert.isTrue(createUrl.calledOnceWithExactly(blob), 'object URL created from the given blob');
            const anchor = click.thisValues[0] as HTMLAnchorElement;
            assert.equal(anchor.download, 'out.bin', 'anchor download set to filename');
            assert.equal(anchor.href, 'blob:mock/0', 'anchor href set to the object URL');
            assert.isTrue(click.calledOnce, 'anchor clicked exactly once');
            assert.isNull(anchor.parentNode, 'anchor detached from the DOM after click');
            assert.isTrue(revokeUrl.notCalled, 'URL not revoked synchronously (deferred)');

            clock.tick(0);
            assert.isTrue(revokeUrl.calledOnceWithExactly('blob:mock/0'), 'URL revoked after the deferred timer');
        });
    });

    describe('downloadTextFile', () => {
        it('wraps text in a typed blob (default text/plain) and delegates to downloadBlob', async () => {
            downloadTextFile('data.json', '{"a":1}', 'application/json');
            const jsonBlob = createUrl.firstCall.args[0] as Blob;
            assert.equal(jsonBlob.type, 'application/json', 'blob carries the given mimeType');
            assert.equal(await jsonBlob.text(), '{"a":1}', 'blob holds the given content');
            assert.equal((click.thisValues[0] as HTMLAnchorElement).download, 'data.json', 'filename forwarded to the anchor');

            downloadTextFile('note.txt', 'plain');
            assert.equal((createUrl.secondCall.args[0] as Blob).type, 'text/plain', 'mimeType defaults to text/plain when omitted');

            clock.tick(0);  // flush deferred revokes before the sandbox is restored
        });
    });

    describe('uploadImage', () => {
        // Build a real PNG File via the canvas adapter, then drive the hidden
        // <input type=file> the picker creates by setting `files` and dispatching
        // the matching event (no actual OS file dialog in the test environment).
        async function makePngFile(width: number, height: number, fill = '#ffffff'): Promise<File> {
            const src = makeCanvasElement(width, height);
            const ctx = src.getContext('2d')!;
            ctx.fillStyle = fill;
            ctx.fillRect(0, 0, width, height);
            const blob = await new Promise<Blob>(resolve => src.toBlob(b => resolve(b!), 'image/png'));
            return new File([await blob.arrayBuffer()], 'pic.png', { type: 'image/png' });
        }

        function pickerInput(): HTMLInputElement {
            return document.body.querySelector('input[type=file]') as HTMLInputElement;
        }

        it('decodes the chosen file to ImageData', async () => {
            const file = await makePngFile(2, 3);
            const promise = uploadImage();
            const input = pickerInput();
            Object.defineProperty(input, 'files', { value: [file], configurable: true });
            input.dispatchEvent(new Event('change'));

            const image = await promise;
            assert.equal(image!.width, 2, 'decoded ImageData width matches the file');
            assert.equal(image!.height, 3, 'decoded ImageData height matches the file');
            assert.isNull(pickerInput(), 'transient picker input removed from the DOM');
        });

        it('resolves null when the picker is cancelled', async () => {
            const promise = uploadImage();
            pickerInput().dispatchEvent(new Event('cancel'));
            assert.isNull(await promise, 'cancelled picker yields null');
            assert.isNull(pickerInput(), 'transient picker input removed from the DOM');
        });
    });

});
