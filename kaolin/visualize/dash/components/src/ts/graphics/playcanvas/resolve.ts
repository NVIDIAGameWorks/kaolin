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

import * as _pc from 'playcanvas';

/**
 * At run-time, resolves the playcanvas dependency, allowing playcanvas to be imported
 * as either standard .min.js file or a deferred importmap module.
 * 
 * @returns: playcanvas module
 * 
 * Note: playcanvas is not bundled with the kaolin javascript bundle, which plotly
 * dash builds as standard UMD bundle. Instead, playcanvas must be imported into the dash
 * app as an external script. For example, like this:
 * <script src="https://cdn.jsdelivr.net/npm/playcanvas@2.14.0/build/playcanvas.min.js"></script>
 * 
 * However, for playcanvas in particular, some scripts can only work if playcanvas is
 * imported as a module via importmap:
 * <script type="importmap">
 *       {
 *          "imports": {
 *               "playcanvas": "https://cdn.jsdelivr.net/npm/playcanvas@2.14.0/+esm"
 *           }
 *       }
 * </script>
 * Which can be made global like this:
 * <script type="module">
 *   import * as pc from 'playcanvas';
 *   console.log('Setting window.pc');
 *   window.pc = pc;
 * </script>
 * 
 * In order to make this file work properly for that, we resolve dependency at code run-time,
 * not at load time (webpack behavior) when the window.pc is not yet set. 
 * 
 */
export function resolvePlaycanvas(localPc?: any) {
    console.log('🍓->>>>>>> Resolving playcanvas');
    if (localPc && localPc.Application) {
        return localPc;
    }
    return _pc && _pc.Application ? _pc : (globalThis as any).pc;
};

console.log('🍓->>>>>>> Loaded playcanvas/resolve.ts');
//import { Application } from 'playcanvas';