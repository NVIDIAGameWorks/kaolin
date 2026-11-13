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

/**
 * Contract, type-guard, and base class for camera-controller behaviors. A
 * controller owns a camera in its own native format and exchanges state with
 * the viewer exclusively through universal {@link CameraParameters}.
 *
 * @module
 */

import { z } from 'zod';

import { CameraParameters } from '../../graphics/camera';
import { makeImplementsInterfaceFunction } from '../../util/types';
import { adjustByDevicePixelRatio } from '../viewport';
import { InteractiveBehavior, SizeableBehavior } from './base';

/**
 * Contract for behaviors that own and drive a camera in their own native
 * format (THREE.Camera, pc.Entity, ...). The viewer lists these in its
 * `camera_controllers` prop and talks to them only through universal
 * {@link CameraParameters}; each controller converts to/from its native camera
 * with its own converter. React-component controllers expose the same members
 * through an imperative handle.
 *
 * A controller is a {@link SizeableBehavior}: the inherited `setDimensions`
 * supplies the viewport size that its parameter converter projects against.
 *
 * @group Key Behavior Interfaces
 */
export interface CameraControllerInterface extends SizeableBehavior {
    /** Update the owned camera from universal parameters. */
    setCameraParams(params: CameraParameters): void;
    /** Native camera object (THREE.Camera, pc.Entity, ...). */
    getCamera(): any;
    /** Convert the owned camera to universal parameters. */
    getCameraParams(): CameraParameters;
}

/**
 * Runtime type-guard that reports whether a value implements
 * {@link CameraControllerInterface}. Fails at compile time if the interface changes.
 *
 * @param object - Candidate value to test.
 * @returns `true` if `object` implements every controller member.
 *
 * @group Key Behavior Interfaces
 */
export const isCameraController =
    makeImplementsInterfaceFunction<CameraControllerInterface>({
        setDimensions: true,
        setCameraParams: true,
        getCamera: true,
        getCameraParams: true,
    });

/**
 * Base for class-based camera controllers: tracks viewport dimensions and binds
 * to the event canvas. Subclasses own the native camera and its conversion.
 *
 * @group Behavior Base Classes
 */
export abstract class CameraControllerBase extends InteractiveBehavior implements CameraControllerInterface {
    static override schema = z.object({});

    /** Viewport size used by the parameter converter, in device pixels. */
    width = 0;
    height = 0;

    override elementType() { return 'canvas'; }

    /**
     * Records the viewport size used by the controller's parameter converter.
     *
     * The recorded size is in device pixels, not the CSS pixels handed in:
     * {@link CameraParameters} are image intrinsics, describing the pixels a
     * renderer produces, and that is the layer's drawing buffer (sized by
     * {@link core.layer.resizeLayer | resizeLayer}) rather than its CSS box. A
     * server asked to render at the CSS size would come back half-resolution on
     * a HiDPI display.
     *
     * Called on every resize commit and on every layer or behavior
     * registration, so the same size arrives repeatedly. Discarding the repeats
     * is this method's job, which is why a subclass with work to do on a real
     * change implements {@link onViewportSizeChanged} instead of overriding
     * here — there is then no way to forget the check.
     *
     * @param width - Viewport width in CSS pixels.
     * @param height - Viewport height in CSS pixels.
     */
    setDimensions(width: number, height: number): void {
        const size = adjustByDevicePixelRatio({ width, height });
        if (this.width === size.width && this.height === size.height) {
            return;
        }
        this.width = size.width;
        this.height = size.height;
        this.onViewportSizeChanged(width, height);
    }

    /**
     * Reconcile the native camera with a viewport size that really changed:
     * reproject, reset native controls. No-op by default, for a controller whose
     * converter just reads the recorded size.
     *
     * @param width - Viewport width in CSS pixels, as the aspect ratio is the
     *     same either way; `this.width` / `this.height` hold it in device pixels.
     * @param height - Viewport height in CSS pixels.
     */
    protected onViewportSizeChanged(_width: number, _height: number): void {}

    /**
     * Updates the owned native camera from universal parameters.
     *
     * @param params - Universal camera parameters to apply.
     */
    abstract setCameraParams(params: CameraParameters): void;

    /**
     * @returns The owned native camera object (THREE.Camera, pc.Entity, ...).
     */
    abstract getCamera(): any;

    /**
     * @returns The owned camera converted to universal parameters.
     */
    abstract getCameraParams(): CameraParameters;
}
