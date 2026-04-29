import * as THREE from 'three';
// TODO: when upgrading threejs, use 'three/addons/controls/OrbitControls' instead of 'three/examples/jsm/controls/OrbitControls'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { z } from 'zod';

import { BehaviorRegister, CameraControllerBase } from '../../../core/behavior';
import { CameraParameters } from '../../camera';
import { defaultCamera, kaolinCameraToThree, threeCameraToKaolin } from '../camera';


/**
 * Schema for {@link OrbitCameraController}. Exposes the subset of three.js
 * `OrbitControls` parameters we want configurable from Python, plus the world
 * `up` direction applied to the owned camera. Single source of truth: the TS
 * option type ({@link OrbitCameraControllerOptions}) is derived via `z.infer`,
 * defaults are read off it at construction, and the build-time manifest dumper
 * projects it to JSON for the Python auto-UI pipeline.
 *
 * Numeric/boolean defaults mirror OrbitControls' own defaults so the
 * interaction is unchanged unless explicitly overridden. `up` defaults to +Z
 * because Kaolin scenes are +Z up (three.js itself defaults to +Y up).
 */
export const OrbitCameraControllerOptionsSchema = z.object({
    up: z.array(z.number()).length(3).default([0, 0, 1])
        .describe('World up direction for the camera as [x, y, z]. Kaolin '
                  + 'scenes are +Z up by default (three.js uses +Y up).')
        .meta({ kind: 'any', uiBound: false }),
    enableDamping: z.boolean().default(false)
        .describe('Smoothly interpolate orbit/pan/zoom motion (adds inertia). '
                  + 'Relies on the per-frame onAnimate update this controller '
                  + 'already runs.'),
    dampingFactor: z.number().min(0).max(1).multipleOf(0.01).default(0.05)
        .describe('Inertia factor used when enableDamping is true; lower '
                  + 'values give more inertia.'),
    enableRotate: z.boolean().default(true)
        .describe('Allow orbiting (rotating) around the target.'),
    enablePan: z.boolean().default(true)
        .describe('Allow panning the target across the view plane.'),
    enableZoom: z.boolean().default(true)
        .describe('Allow dolly/zoom.'),
    screenSpacePanning: z.boolean().default(true)
        .describe('Pan in screen space (true) versus along the world ground '
                  + 'plane (false).'),
    rotateSpeed: z.number().min(0).max(10).multipleOf(0.1).default(1)
        .describe('Orbit rotation speed multiplier.'),
    zoomSpeed: z.number().min(0).max(10).multipleOf(0.1).default(1)
        .describe('Zoom/dolly speed multiplier.'),
    panSpeed: z.number().min(0).max(10).multipleOf(0.1).default(1)
        .describe('Pan speed multiplier.'),
});

export type OrbitCameraControllerOptions = z.infer<typeof OrbitCameraControllerOptionsSchema>;


export class OrbitCameraController extends CameraControllerBase {
    static override schema = OrbitCameraControllerOptionsSchema;

    override options: OrbitCameraControllerOptions;
    camera: THREE.Camera;
    controls: OrbitControls | null;

    constructor(options: Partial<OrbitCameraControllerOptions> = {}) {
        super();
        this.options = OrbitCameraControllerOptionsSchema.parse(options);
        this.camera = defaultCamera();
        this.applyUp();
        this.controls = null;
    }

    setCameraParams(params: CameraParameters) {
        this.camera = kaolinCameraToThree(params);
        this.applyUp();
        if (this.element) {
            this.initControls();
        }
    }

    getCamera(): THREE.Camera {
        return this.camera;
    }

    getCameraParams(): CameraParameters {
        return threeCameraToKaolin(this.camera, this.width, this.height);
    }

    setDimensions(width: number, height: number) {
        super.setDimensions(width, height);
        if (this.camera instanceof THREE.PerspectiveCamera && height > 0) {
            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();
        }
    }

    setActiveElement(element: HTMLElement) {
        super.setActiveElement(element);
        // If camera is already set, create controls
        if (this.element && this.camera) {
            this.initControls();
        }
    }

    /**
     * Apply the configured world up direction to the owned camera. Kaolin
     * scenes are +Z up, which differs from three.js' default +Y up, so the up
     * vector must be set explicitly before OrbitControls captures it.
     */
    private applyUp() {
        const [x, y, z] = this.options.up;
        this.camera.up.set(x, y, z);
    }

    /** Push the configured OrbitControls option values onto the live controls. */
    private applyControlOptions() {
        const c = this.controls;
        if (!c) return;
        c.enableDamping = this.options.enableDamping;
        c.dampingFactor = this.options.dampingFactor;
        c.enableRotate = this.options.enableRotate;
        c.enablePan = this.options.enablePan;
        c.enableZoom = this.options.enableZoom;
        c.screenSpacePanning = this.options.screenSpacePanning;
        c.rotateSpeed = this.options.rotateSpeed;
        c.zoomSpeed = this.options.zoomSpeed;
        c.panSpeed = this.options.panSpeed;
        c.update();
    }

    private initControls() {
        // Dispose existing controls if any
        if (this.controls) {
            this.controls.dispose();
            this.controls = null;
        }

        if (this.element && this.camera) {
            this.controls = new OrbitControls(this.camera, this.element);
            this.applyControlOptions();
        }
    }

    /**
     * React to live option edits dispatched via setOption. Changing `up`
     * requires re-seeding OrbitControls (it captures the camera's up vector
     * when constructed), so the controls are rebuilt; every other option is
     * applied in place on the existing controls.
     */
    override updateForOptions(name?: string, _value?: any): void {
        if (name === 'up') {
            this.applyUp();
            if (this.element) {
                this.initControls();
            }
        } else {
            this.applyControlOptions();
        }
    }

    onAnimate() {
        if (this.controls) {
            this.controls.update();
        }
    }

    setActive(enabled: boolean) {
        if (this.controls) {
            this.controls.enabled = enabled;
        }
    }

    dispose() {
        if (this.controls) {
            this.controls.dispose();
            this.controls = null;
        }
    }
}
BehaviorRegister.register('threejs_camera_orbit', OrbitCameraController,
    'Three.js orbit-camera controller. Bound to the event canvas; updates a '
    + 'shared THREE.Camera in response to pointer drags / wheel.');


// TODO, add more options and TrackBall Controls:
//controls.current = new TrackballControls(camera, eventCanvasRef.current);
// controls.current.enableDamping = true;
// controls.current.dampingFactor = 0.25;
// controls.current.enableZoom = true;
// controls.current.enablePan = true;
// controls.current.enableRotate = true;
// controls.current.enableZoom = true;
// controls.current.enablePan = true;
// controls.current.enableRotate = true;
