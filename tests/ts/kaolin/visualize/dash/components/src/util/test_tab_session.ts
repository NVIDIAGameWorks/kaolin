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

// NOTE: __resetTabUuidCache() clears the module-level cache so a reset can stand in
// for a page reload: the cache is dropped while sessionStorage (if any) persists.
// The no-storage path intentionally hits the internal try/catch, so we raise the log
// level to ERROR to keep those expected warnings out of the test output.

import { assert } from 'chai';
import sinon from 'sinon';
import { getTabUuid, __resetTabUuidCache } from '@kaolin/util/tab_session';
import { setLogLevel, LogLevel } from '@kaolin/util/logging';
import { registerDom, unregisterDom } from '@test/helpers/dom';

const UUID_REGEX = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

describe('visualize/dash/components/src/util/test_tab_session.ts', () => {

    // getTabUuid reads/writes window.sessionStorage, provided by the happy-dom DOM.
    before(registerDom);
    after(unregisterDom);

    let sandbox: sinon.SinonSandbox;

    beforeEach(() => {
        sandbox = sinon.createSandbox();
        setLogLevel(LogLevel.ERROR);
    });

    afterEach(() => {
        sandbox.restore();
        window.sessionStorage.clear();
        __resetTabUuidCache();
    });

    it('works without session storage', () => {
        // Swap in a storage whose getItem/setItem throw, exercising the internal
        // try/catch fallbacks. Replacing the whole accessor (rather than stubbing
        // methods on happy-dom's Proxy-backed Storage) leaves the real storage
        // intact for the next test after restore.
        const blocked = () => { throw new Error('sessionStorage blocked'); };
        sandbox.stub(window, 'sessionStorage').get(() => ({ getItem: blocked, setItem: blocked }) as any);
        __resetTabUuidCache();

        const uuid = getTabUuid();
        assert.match(uuid, UUID_REGEX, 'mints a v4 uuid even with no storage');
        assert.equal(getTabUuid(), uuid, 'in-memory cache keeps it stable within the session');
        __resetTabUuidCache();              // simulate a reload: nothing was persisted
        assert.notEqual(getTabUuid(), uuid, 'a fresh session mints a new uuid when nothing persisted');
    });

    it('works with session storage', () => {
        __resetTabUuidCache();
        const uuid = getTabUuid();
        assert.match(uuid, UUID_REGEX, 'mints a v4 uuid on first access');
        __resetTabUuidCache();              // simulate a reload: storage persists, cache does not
        assert.equal(getTabUuid(), uuid, 'recovers the same uuid from sessionStorage, not the cache');
    });

});
