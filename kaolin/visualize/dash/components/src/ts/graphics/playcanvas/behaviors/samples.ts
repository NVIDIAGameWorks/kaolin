import * as _pc from 'playcanvas';
import { z } from 'zod';

import { BehaviorRegister, CameraControllerInterface, InteractiveBehavior } from '../../../core/behavior';
import { adjustByDevicePixelRatio } from '../../../core/viewport';
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

    setDimensions(width: number, height: number) {
        const displaySize = adjustByDevicePixelRatio({ width, height });
        this.app?.resizeCanvas(displaySize.width, displaySize.height);
    }

    setup() {
        const pc = resolvePlaycanvas(_pc);

        // create an application
        const canvas = this.element as HTMLCanvasElement;
        if (this.app !== null) {
            return;
        }

        this.app = new pc.Application(canvas);
        this.app.setCanvasResolution(pc.RESOLUTION_AUTO);
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

    setDimensions(width: number, height: number) {
        this.width = width;
        this.height = height;
        const displaySize = adjustByDevicePixelRatio({ width, height });
        this.app?.resizeCanvas(displaySize.width, displaySize.height);
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
        app.setCanvasResolution(pc.RESOLUTION_AUTO);
        app.start();

        window.addEventListener('resize', () => app.resizeCanvas());

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
