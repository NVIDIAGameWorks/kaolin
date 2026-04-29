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

// NOTE: ConsoleLogger delegates to console.error/warn/info/log. Use
// captureConsole from @test/helpers/console to record calls without noisy output.

import { assert } from 'chai';
import { ConsoleLogger, LogLevel,logger, setLogLevel, getLogLevel } from '@kaolin/util/logging';
import { captureConsole } from '@test/helpers/console';

describe('visualize/dash/components/src/util/test_logging.ts', () => {

    describe('default logger basic conventions', () => {
        it('it logs at the right level', () => {
            setLogLevel(LogLevel.DEBUG);
            const { calls, restore } = captureConsole();
            try {
                logger.debug('ping');
                logger.info('pong');
                logger.warn('buzz');
                logger.error('bazz');
                assert.equal(calls.length, 4, 'all calls logged at debug level');
                // debug log
                assert.equal(calls[0].method, 'log', 'debug routes to console.log');
                assert.include(String(calls[0].args[0]), 'ping', 'formatted message contains text');
                // info log
                assert.equal(calls[1].method, 'info', 'info routes to console.info');
                assert.include(String(calls[1].args[0]), 'pong', 'formatted message contains text');
                // warn log
                assert.equal(calls[2].method, 'warn', 'warn routes to console.warn');
                assert.include(String(calls[2].args[0]), 'buzz', 'formatted message contains text');
                // error log
                assert.equal(calls[3].method, 'error', 'error routes to console.error');
                assert.include(String(calls[3].args[0]), 'bazz', 'formatted message contains text');
            } finally {
                restore();
            }
        });
        it('it logs at the right level', () => {
            setLogLevel(LogLevel.DEBUG);
            let levels = [LogLevel.ERROR, LogLevel.WARN, LogLevel.INFO, LogLevel.DEBUG];
            for (let i = 0; i < levels.length; ++i) {
                let expectedNumLogs = i + 1; 
                setLogLevel(levels[i]);
                const { calls, restore } = captureConsole();
                try {
                    logger.debug('anyone');
                    logger.info('lived in a pretty how town');
                    logger.warn('with up so floating many bells down');
                    logger.error('he sang his didnt he danced his did.');
                    assert.equal(calls.length, expectedNumLogs);
                } finally {
                    restore();
                }
            }
        });
    });
});
