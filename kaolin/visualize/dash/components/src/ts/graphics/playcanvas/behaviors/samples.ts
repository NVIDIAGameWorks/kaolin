import * as _pc from 'playcanvas';
import { z } from 'zod';

import { BehaviorRegister, CameraControllerInterface, InteractiveBehavior } from '../../../core/behavior';
import { CameraOrthoIntrinsics, CameraParameters, CameraPinholeIntrinsics, defaultCameraParameters } from '../../camera';
import { updatePlaycanvasCameraWithKaolinParams } from '../camera';
import { resolvePlaycanvas } from '../resolve';
import { adjustByDevicePixelRatio } from '../../../core/viewport';

// Cache per device so behaviors sharing an app do not repeat resize events.
const deviceSizes = new WeakMap<pc.GraphicsDevice, { width: number, height: number }>();

/** Synchronize the graphics device after the viewer has sized its layer. */
function resizePlaycanvas(app: pc.Application | null, width: number, height: number): void {
    if (!app) return;
    const device = app.graphicsDevice;
    device.maxPixelRatio = window.devicePixelRatio || 1;
    const buffer = adjustByDevicePixelRatio({ width, height });
    const alreadySized = device.canvas.width === buffer.width && device.canvas.height === buffer.height;
    device.resizeCanvas(width, height);
    if (device.width !== buffer.width || device.height !== buffer.height) {
        throw new Error(`PlayCanvas could not resize its drawing buffer to ${buffer.width}x${buffer.height}`);
    }
    const last = deviceSizes.get(device);
    // resizeCanvas skips its notification when the layer pass already sized the
    // buffer. Notify dependent render targets once, without clearing it again.
    if (alreadySized && (!last || last.width !== buffer.width || last.height !== buffer.height)) {
        device.fire('resizecanvas', buffer.width, buffer.height);
    }
    deviceSizes.set(device, buffer);
}

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

    /** Synchronize PlayCanvas with the layer; unchanged sizes are a no-op. */
    setDimensions(width: number, height: number) {
        resizePlaycanvas(this.app, width, height);
    }

    setup() {
        const pc = resolvePlaycanvas(_pc);

        // create an application
        const canvas = this.element as HTMLCanvasElement;
        if (this.app !== null) {
            return;
        }

        this.app = new pc.Application(canvas);
        this.app.graphicsDevice.maxPixelRatio = window.devicePixelRatio || 1;
        // FIXED keeps PlayCanvas from resizing every frame; the viewer supplies
        // committed CSS sizes to this behavior instead.
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

    /** Record pending CSS dimensions and resize through the graphics device. */
    setDimensions(width: number, height: number) {
        this.width = width;
        this.height = height;
        // The device checks the actual buffer, so failed or pre-setup requests
        // can be retried even when the requested CSS size is unchanged.
        resizePlaycanvas(this.app, width, height);
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
        // TODO: Derive intrinsics from the native camera and DPR-adjusted buffer;
        // echoed parameters can retain stale pixel dimensions after resizing.
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
        app.graphicsDevice.maxPixelRatio = window.devicePixelRatio || 1;
        // Resize only on viewer commits, using any size received before setup.
        // AUTO would override this from the CSS box on every frame.
        app.setCanvasResolution(pc.RESOLUTION_FIXED,
                                this.width || canvas.clientWidth, this.height || canvas.clientHeight);
        app.start();
        // Accept resizes while assets load, rather than leaving the buffer at
        // the size captured before the await below.
        this.app = app;

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
