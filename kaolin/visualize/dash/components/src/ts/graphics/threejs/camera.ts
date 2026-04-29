
import * as THREE from 'three';

import { logger } from '../../util/logging';
import { flattenMatrix } from '../../util/types';
import { CameraOrthoIntrinsics, CameraParameters, CameraPinholeIntrinsics } from '../camera';

/**
 * Create a Three.js camera from intrinsics only (no extrinsics)
 */
export function intrinsicsPinholeToThreeCamera(intrinsics: CameraPinholeIntrinsics): THREE.PerspectiveCamera {
    const aspect = intrinsics.width / intrinsics.height;
    const fov = 2 * Math.atan(intrinsics.height / (2 * intrinsics.focal_y)) * (180 / Math.PI);

    const camera = new THREE.PerspectiveCamera(fov, aspect, intrinsics.near, intrinsics.far);

    // Handle principal point offset
    if (intrinsics.x0 !== 0 || intrinsics.y0 !== 0) {
        logger.warn(`Principal point offset is not yet supported for pinhole cameras. x0: ${intrinsics.x0}, y0: ${intrinsics.y0}`);
        // const offsetX = intrinsics.x0 - intrinsics.width / 2;
        // const offsetY = intrinsics.y0 - intrinsics.height / 2;

        // camera.setViewOffset(
        //     intrinsics.width,
        //     intrinsics.height,
        //     offsetX,
        //     offsetY,
        //     intrinsics.width,
        //     intrinsics.height
        // );
    }

    return camera;
}

export function intrinsicsOrthoToThreeCamera(intrinsics: CameraOrthoIntrinsics): THREE.OrthographicCamera {
    // TODO: add tests and verify
    const camera = new THREE.OrthographicCamera(
        -intrinsics.width / 2,
        intrinsics.width / 2,
        intrinsics.height / 2,
        -intrinsics.height / 2,
        intrinsics.near,
        intrinsics.far
    );
    return camera;
}

export function intrinsicsToThreeCamera(intrinsics: CameraPinholeIntrinsics | CameraOrthoIntrinsics): THREE.Camera {
    if (intrinsics.classname == 'pinhole') {
        return intrinsicsPinholeToThreeCamera(intrinsics as CameraPinholeIntrinsics);
    } else if (intrinsics.classname == 'orthographic') {
        return intrinsicsOrthoToThreeCamera(intrinsics as CameraOrthoIntrinsics);
    } else {
        throw new Error(`Unsupported intrinsics type ${intrinsics.classname}. Expected a pinhole or orthographic.`);
    }
}

/**
 * Convert camera parameters to Three.js PerspectiveCamera
 */
export function kaolinCameraToThree(params: CameraParameters): THREE.Camera {
    const { intrinsics, extrinsics } = params;

    // Convert intrinsics to Three.js camera parameters
    let camera = intrinsicsToThreeCamera(intrinsics);

    // Convert view matrix (world-to-camera) to camera transform (camera-to-world)
    const viewMatrix = flattenMatrix(extrinsics.view_matrix);

    // Create Three.js Matrix4 from view matrix (assuming row-major order)
    const threeViewMatrix = new THREE.Matrix4().fromArray(viewMatrix).transpose().invert();

    camera.applyMatrix4(threeViewMatrix);
    camera.updateMatrixWorld(true);

    return camera;
}

/**
 * Convert Three.js PerspectiveCamera back to camera parameters
 */
export function threePerspectiveCameraToKaolin(
    camera: THREE.PerspectiveCamera,
    width: number,
    height: number
): CameraParameters {
    // Calculate focal lengths from FOV
    const fovRad = (camera.fov * Math.PI) / 180;
    const focal_y = height / (2 * Math.tan(fovRad / 2));
    const focal_x = focal_y; // Assuming square pixels; adjust if needed

    // Handle principal point; 
    // by default, kaolin assumes the NDC origin is at the canvas center of (0, 0).
    let x0 = 0;
    let y0 = 0;

    if (camera.view !== null && camera.view.enabled) {
        x0 = - (camera.view.offsetX / camera.view.fullWidth) * width;
        y0 = + (camera.view.offsetY / camera.view.fullHeight) * height; // Y is flipped
    }

    // Get world matrix (camera-to-world transform)
    camera.updateMatrixWorld();

    const viewMatrix = new Float32Array(new THREE.Matrix4().copy(camera.matrixWorldInverse).transpose().elements);
    viewMatrix['shape'] = [4, 4];

    return {
        intrinsics: {
            width,
            height,
            focal_x,
            focal_y,
            x0,
            y0,
            near: camera.near,
            far: camera.far,
            classname: 'pinhole'
        },
        extrinsics: {
            view_matrix: viewMatrix
        }
    };
}

/**
 * Convert Three.js PerspectiveCamera back to camera parameters
 */
export function threeCameraToKaolin(
    camera: THREE.Camera,
    width: number,
    height: number
): CameraParameters {
    if (camera instanceof THREE.PerspectiveCamera) {
        return threePerspectiveCameraToKaolin(camera as THREE.PerspectiveCamera, width, height);
    } else {
        throw new Error(`Unsupported camera type ${typeof camera}. Expected a THREE.PerspectiveCamera.`);
    }
}

export function defaultCamera(): THREE.Camera {
    const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 1000);
    camera.position.set(2, 0, 1.5);
    camera.lookAt(0, 0, 0);
    return camera;
    // TODO: use this instead when working return kaolinCameraToThree(defaultCameraParameters);
}

