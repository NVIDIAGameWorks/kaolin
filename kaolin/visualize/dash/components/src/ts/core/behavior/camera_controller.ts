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
import { InteractiveBehavior } from './base';

/**
 * Contract for behaviors that own and drive a camera in their own native
 * format (THREE.Camera, pc.Entity, ...). The viewer lists these in its
 * `camera_controllers` prop and talks to them only through universal
 * {@link CameraParameters}; each controller converts to/from its native camera
 * with its own converter. React-component controllers expose the same members
 * through an imperative handle.
 *
 * @group Key Behavior Interfaces
 */
export interface CameraControllerInterface {
    /** Viewport size used by the controller's CameraParameters converter. */
    setDimensions(width: number, height: number): void;
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

    width = 0;
    height = 0;

    override elementType() { return 'canvas'; }

    /**
     * Records the viewport size used by the controller's parameter converter.
     *
     * @param width - Viewport width in pixels.
     * @param height - Viewport height in pixels.
     */
    setDimensions(width: number, height: number) {
        this.width = width;
        this.height = height;
    }

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
