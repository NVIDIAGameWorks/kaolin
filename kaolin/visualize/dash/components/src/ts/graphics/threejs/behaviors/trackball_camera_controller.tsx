import * as THREE from 'three';
// TODO: when upgrading threejs, use 'three/addons/controls/TrackballControls' instead
import { TrackballControls } from 'three/examples/jsm/controls/TrackballControls';
import { z } from 'zod';

import { BehaviorRegister, CameraControllerBase } from '../../../core/behavior';
import { CameraParameters } from '../../camera';
import { defaultCamera, kaolinCameraToThree, threeCameraToKaolin } from '../camera';


/**
 * Schema for {@link TrackballCameraController}. Uses three.js TrackballControls
 * which allows unrestricted free rotation (no polar-angle clamping, no gimbal
 * lock). Suitable when you want to spin an object in any direction continuously.
 */
export const TrackballCameraControllerOptionsSchema = z.object({
    up: z.array(z.number()).length(3).default([0, 0, 1])
        .describe('Initial world up direction for the camera as [x, y, z].')
        .meta({ kind: 'any', uiBound: false }),
    enableRotate: z.boolean().default(true)
        .describe('Allow free rotation around the target.'),
    enablePan: z.boolean().default(true)
        .describe('Allow panning the target.'),
    enableZoom: z.boolean().default(true)
        .describe('Allow zoom/dolly.'),
    rotateSpeed: z.number().min(0).max(10).multipleOf(0.1).default(1)
        .describe('Rotation speed multiplier.'),
    zoomSpeed: z.number().min(0).max(10).multipleOf(0.1).default(1.2)
        .describe('Zoom speed multiplier.'),
    panSpeed: z.number().min(0).max(10).multipleOf(0.1).default(0.3)
        .describe('Pan speed multiplier.'),
});

export type TrackballCameraControllerOptions = z.infer<typeof TrackballCameraControllerOptionsSchema>;


export class TrackballCameraController extends CameraControllerBase {
    static override schema = TrackballCameraControllerOptionsSchema;

    override options: TrackballCameraControllerOptions;
    camera: THREE.Camera;
    controls: TrackballControls | null;
    private _onWheel: () => void;

    constructor(options: Partial<TrackballCameraControllerOptions> = {}) {
        super();
        this.options = TrackballCameraControllerOptionsSchema.parse(options);
        this.camera = defaultCamera();
        this.applyUp();
        this.controls = null;
        // TrackballControls defers zoom to update(), unlike OrbitControls which
        // calls update() inside its wheel handler. This native listener runs after
        // TrackballControls queues the delta but before React's synthetic onWheel
        // fires maybeSendCameraUpdate(), so the camera position is current.
        this._onWheel = () => { if (this.controls) this.controls.update(); };
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
        if (this.controls) {
            this.controls.handleResize();
        }
    }

    setActiveElement(element: HTMLElement) {
        super.setActiveElement(element);
        if (this.element && this.camera) {
            this.initControls();
            this.element.addEventListener('wheel', this._onWheel, { passive: true });
        }
    }

    private applyUp() {
        const [x, y, z] = this.options.up;
        this.camera.up.set(x, y, z);
    }

    private applyControlOptions() {
        const c = this.controls;
        if (!c) return;
        c.noRotate = !this.options.enableRotate;
        c.noPan = !this.options.enablePan;
        c.noZoom = !this.options.enableZoom;
        c.rotateSpeed = this.options.rotateSpeed;
        c.zoomSpeed = this.options.zoomSpeed;
        c.panSpeed = this.options.panSpeed;
        // Apply movements immediately with no damping so wheel zoom has no
        // residual state that would re-fire on the next pointer interaction.
        c.staticMoving = true;
    }

    private initControls() {
        if (this.controls) {
            this.controls.dispose();
            this.controls = null;
        }
        if (this.element && this.camera) {
            this.controls = new TrackballControls(this.camera, this.element);
            this.applyControlOptions();
        }
    }

    override updateForOptions(_name?: string, _value?: any): void {
        this.applyControlOptions();
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
        if (this.element) {
            this.element.removeEventListener('wheel', this._onWheel);
        }
        if (this.controls) {
            this.controls.dispose();
            this.controls = null;
        }
    }
}
BehaviorRegister.register('threejs_trackball_controller', TrackballCameraController,
    'Three.js trackball-camera controller. Allows unrestricted free rotation '
    + 'with no polar-angle clamping (no gimbal lock at poles).');
