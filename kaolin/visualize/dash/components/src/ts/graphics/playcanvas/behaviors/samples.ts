import * as _pc from 'playcanvas';
import { z } from 'zod';

import { BehaviorRegister, CameraControllerInterface, InteractiveBehavior } from '../../../core/behavior';
import { CameraOrthoIntrinsics, CameraParameters, CameraPinholeIntrinsics, defaultCameraParameters } from '../../camera';
import { updatePlaycanvasCameraWithKaolinParams } from '../camera';
import { resolvePlaycanvas } from '../resolve';

export class PlaycanvasBehavior extends InteractiveBehavior {
    static override schema = z.object({});

    app: any;
    camera: any;

    constructor() {
        super();
        this.app = null;
        this.camera = null;
    }

    override elementType() {
        return "canvas";
    }

    override setActiveElement(canvas: HTMLElement) {
        super.setActiveElement(canvas);
        this.setup();
    }

    /**
     * Tells the graphics device the new resolution. In CSS pixels, since
     * `resizeCanvas` applies its own pixel ratio (matched to the viewer's in
     * `setup`), and never through `app.resizeCanvas`, which would write the
     * layer's inline CSS size. The viewer has already given the canvas this
     * buffer size, so the device's write is a no-op and nothing is cleared; the
     * call is what keeps PlayCanvas' own resolution and viewport in step.
     *
     * `resizeCanvas` is itself guarded against an unchanged size, and the app
     * may not exist on the first call (the element is bound by `setActiveElement`),
     * in which case `setup` picks the size up off the canvas.
     */
    setDimensions(width: number, height: number) {
        this.app?.graphicsDevice.resizeCanvas(width, height);
    }

    setup() {
        const pc = resolvePlaycanvas(_pc);

        // create an application
        const canvas = this.element as HTMLCanvasElement;
        if (this.app !== null) {
            return;
        }

        this.app = new pc.Application(canvas);
        // The viewer sizes layer buffers in device pixels; PlayCanvas caps its
        // own ratio at 1 by default, so `resizeCanvas` would scale a CSS size to
        // a *smaller* buffer than the viewer wrote and undo it. Matching the
        // ratio makes the two agree exactly.
        this.app.graphicsDevice.maxPixelRatio = window.devicePixelRatio || 1;
        // FIXED, not AUTO: the viewer owns this canvas' buffer size, and AUTO
        // has PlayCanvas re-derive it from the element's client box every frame
        // (AppBase.updateCanvasSize), overwriting what the viewer wrote. CSS
        // pixels, like `resizeCanvas` (which this calls through to) — the
        // element fills the layer, so its client box is the viewer's size.
        this.app.setCanvasResolution(pc.RESOLUTION_FIXED, canvas.clientWidth, canvas.clientHeight);
        this.app.setCanvasFillMode(pc.FILLMODE_NONE);
        this.app.start();

        // create a camera
        this.camera = new pc.Entity();
        this.camera.addComponent('camera', {
            clearColor: new pc.Color(0.3, 0.3, 0.7)
        });
        this.camera.setPosition(0, 0, 3);
        this.app.root.addChild(this.camera);

        // create a light
        const light = new pc.Entity();
        light.addComponent('light');
        light.setEulerAngles(45, 45, 0);
        this.app.root.addChild(light);

        // create a box
        const box = new pc.Entity();
        box.addComponent('model', {
            type: 'box'
        });
        this.app.root.addChild(box);

        // rotate the box
        this.app.on('update', (dt) => box.rotate(10 * dt, 20 * dt, 30 * dt));
    }
}
BehaviorRegister.register("playcanvas", PlaycanvasBehavior,
    'Placeholder PlayCanvas demo behavior: spawns a rotating box. '
    + 'Intended as a wiring example.');


export class PlaycanvasSplatBehavior extends InteractiveBehavior implements CameraControllerInterface {
    static override schema = z.object({});

        app: pc.Application | null;
        camera: pc.Entity | null;
        // Viewport size in CSS pixels, as the graphics device wants it. Unlike
        // CameraControllerBase these are not the intrinsics' pixel dimensions:
        // getCameraParams echoes back what it was given rather than converting
        // from the native camera.
        width = 0;
        height = 0;
        // Last applied params; echoed back by getCameraParams until a native
        // pc.Entity -> CameraParameters converter exists (this behavior is
        // currently driven externally rather than generating camera motion).
        lastParams: CameraParameters | null = null;

    constructor() {
        super();
        this.app = null;
        this.camera = null;
    }

    override elementType() {
        return "canvas";
    }

    override setActiveElement(canvas: HTMLElement) {
        super.setActiveElement(canvas);
        this.setup();
    }

    /**
     * Records the size for the camera converter and tells the graphics device
     * the new resolution. CSS pixels (`resizeCanvas` applies its own pixel
     * ratio, matched to the viewer's in `setup`), and never `app.resizeCanvas`,
     * which would write the layer's inline CSS size. The viewer has already
     * given the canvas this buffer size, so the device's write is a no-op that
     * clears nothing; it only brings PlayCanvas' own resolution and viewport in
     * step.
     *
     * Returns on an unchanged size, and `setup` is async, so the app may not
     * exist yet on the first call; it re-applies the recorded size once it does.
     */
    setDimensions(width: number, height: number) {
        if (this.width === width && this.height === height) {
            return;
        }
        this.width = width;
        this.height = height;
        this.app?.graphicsDevice.resizeCanvas(width, height);
    }

    setCameraParams(cameraParams: CameraParameters) {
        this.lastParams = cameraParams;
        if (this.camera) {
            updatePlaycanvasCameraWithKaolinParams(this.camera, cameraParams);
        }
    }

    getCamera(): pc.Entity | null {
        return this.camera;
    }

    getCameraParams(): CameraParameters {
        return this.lastParams ?? defaultCameraParameters;
    }

    async setup() {
        // create an application
        const canvas = this.element as HTMLCanvasElement;
        if (this.app !== null) {
            return;
        }

        console.log('🍓->>>>>>> Creating playcanvas component.');
        const pc = resolvePlaycanvas(_pc);
        const app = new pc.Application(canvas, {
            graphicsDeviceOptions: {
                antialias: false,
                alpha: true // <--- ENABLE ALPHA CHANNEL HERE
            }
        });
        app.setCanvasFillMode(pc.FILLMODE_NONE);
        // The viewer sizes layer buffers in device pixels; PlayCanvas caps its
        // own ratio at 1 by default, so `resizeCanvas` would scale a CSS size to
        // a *smaller* buffer than the viewer wrote and undo it. Matching the
        // ratio makes the two agree exactly.
        app.graphicsDevice.maxPixelRatio = window.devicePixelRatio || 1;
        // FIXED, not AUTO: the viewer owns this canvas' buffer size, and AUTO
        // has PlayCanvas re-derive it from the element's client box every frame
        // (AppBase.updateCanvasSize), overwriting what the viewer wrote. In CSS
        // pixels, like `resizeCanvas` this calls through to: the viewer's size,
        // recorded by setDimensions before this async setup finished, or the
        // element's client box (it fills the layer) until that arrives.
        // No window resize listener either, for the same reason: resizing is
        // driven by the viewer, which observes its own container.
        app.setCanvasResolution(pc.RESOLUTION_FIXED,
                                this.width || canvas.clientWidth, this.height || canvas.clientHeight);
        app.start();

        // Load assets
        const assets = [
            new pc.Asset('camera-controls', 'script', {
                url: 'https://cdn.jsdelivr.net/npm/playcanvas@2.14.0/scripts/esm/camera-controls.mjs'
            }),
            new pc.Asset('toy', 'gsplat', {
                url: '/_kaolin_dynamic/splat.ply'
                //url: 'https://developer.playcanvas.com/assets/toy-cat.sog'
            })
        ];

        const loader = new pc.AssetListLoader(assets, app.assets);
         await new Promise<void>((resolve, reject) => {
            // We pass our custom anonymous function into loader.load
            loader.load((err, failed) => {
                if (!err) {
                    // Success: Unblock the await
                    console.error('Resource loaded OK');
                    resolve();
                } else {
                    // Failure: Log the info as requested
                    console.error('failed to load!');
                    console.error(err);
                    console.error(failed);

                    // CRITICAL: Tell the promise it failed so the code stops waiting
                    //reject(err);
                    resolve();
                }
            });
        });
        //await new Promise(resolve => loader.load(resolve));

        // Store app reference
        this.app = app;

        // Create camera entity
        this.camera = new pc.Entity('Camera');
        this.camera.setPosition(0, 0, 2.5);
        this.camera.addComponent('camera', {
            clearColor: new pc.Color(0, 0, 0, 0) // <--- R, G, B, Alpha (0 = transparent)
        });
        //this.camera.addComponent('script');
        //this.camera.script.create('cameraControls');
        app.root.addChild(this.camera);

        // Apply any params received before the (async) camera was ready.
        if (this.lastParams) {
            updatePlaycanvasCameraWithKaolinParams(this.camera, this.lastParams);
        }

        // Create splat entity
        const splat = new pc.Entity('Toy Cat');
        //splat.setPosition(0, -0.7, 0);
        //splat.setEulerAngles(0, 0, 180);
        splat.addComponent('gsplat', { asset: assets[1] });
        app.root.addChild(splat);
    }
}
BehaviorRegister.register("playcanvas_splat", PlaycanvasSplatBehavior,
    'PlayCanvas gsplat viewer. Loads `/_kaolin_dynamic/splat.ply` into a '
    + 'transparent-clear scene; imperative `setCameraParams` syncs an external Kaolin camera.');
