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