import * as _pc from 'playcanvas';

import { logger } from '../../util/logging';
import { flattenMatrix } from '../../util/types';
import { CameraOrthoIntrinsics, CameraParameters, CameraPinholeIntrinsics } from '../camera';
import { resolvePlaycanvas } from './resolve';


function intrinsicsOrthoToCameraParams(ortho: CameraOrthoIntrinsics) : any {
    let pc = resolvePlaycanvas(_pc);
    return {
            projection: pc.PROJECTION_ORTHOGRAPHIC,
            orthoHeight: ortho.height / 2,
            nearClip: ortho.near,
            farClip: ortho.far
        };
}

function intrinsicsPinholeToCameraParams(pinhole: CameraPinholeIntrinsics): any {
    let pc = resolvePlaycanvas(_pc);

    const fov = 2 * Math.atan(pinhole.height / (2 * pinhole.focal_y)) * (180 / Math.PI);
    const aspect = pinhole.width / pinhole.height;

    if (pinhole.x0 !== 0 || pinhole.y0 !== 0) {
            logger.warn(`Principal point offset not yet supported for PlayCanvas. x0: ${pinhole.x0}, y0: ${pinhole.y0}`);
    }

    return {
            projection: pc.PROJECTION_PERSPECTIVE,
            fov: fov,
            aspectRatio: aspect,
            nearClip: pinhole.near,
            farClip: pinhole.far
    };
}

export function intrinsicsToCameraParams(intrinsics: CameraPinholeIntrinsics | CameraOrthoIntrinsics): any {
    if (intrinsics.classname == 'pinhole') {
        return intrinsicsPinholeToCameraParams(intrinsics as CameraPinholeIntrinsics);
    } else if (intrinsics.classname == 'orthographic') {
        return intrinsicsOrthoToCameraParams(intrinsics as CameraOrthoIntrinsics);
    } else {
        throw new Error(`Unsupported intrinsics type ${intrinsics.classname}. Expected a pinhole or orthographic.`);
    }
}


/**
 * Convert Kaolin camera parameters to a PlayCanvas Entity with camera component.
 * 
 * @param params Kaolin camera parameters (intrinsics + extrinsics)
 * @returns PlayCanvas Entity with configured camera component
 */
export function kaolinCameraToPlaycanvas(params: CameraParameters): pc.Entity {
    let pc = resolvePlaycanvas(_pc);

    const cameraEntity = new pc.Entity('Camera');
    updatePlaycanvasCameraWithKaolinParams(cameraEntity, params);

    return cameraEntity;
}


/**
 * Update Playcanvas camera.
 * 
 * @param params Kaolin camera parameters (intrinsics + extrinsics)
 * @returns PlayCanvas Entity with configured camera component
 */
export function updatePlaycanvasCameraWithKaolinParams(cameraEntity: pc.Entity, params: CameraParameters) {
    const { intrinsics, extrinsics } = params;
    let pc = resolvePlaycanvas(_pc);

    
    // Determine projection type and set intrinsics
    let componentParams : any = intrinsicsToCameraParams(intrinsics);
    let cameraComponent = cameraEntity.camera;
    if (!cameraComponent) {
        cameraEntity.addComponent('camera', componentParams);
    } else {
        for (const [key, value] of Object.entries(componentParams)) {
            cameraComponent[key] = value;
        }
    }

    // this._pose.copy(this._controller.update(frame, dt));
    //     this._camera.entity.setPosition(this._pose.position);
    //     this._camera.entity.setEulerAngles(this._pose.angles);

    // Apply extrinsics (view matrix is world-to-camera, we need camera-to-world)
    const viewMatrix = flattenMatrix(extrinsics.view_matrix);

    // Create PlayCanvas Mat4 from row-major array, transpose, then invert to get camera-to-world
    const pcViewMatrix = new pc.Mat4();
    pcViewMatrix.set(viewMatrix);
    
    // Kaolin stores row-major, PlayCanvas uses column-major internally
    // Transpose to convert, then invert to get camera-to-world transform
    const cameraToWorld = new pc.Mat4();
    pcViewMatrix.transpose();
    cameraToWorld.invert(pcViewMatrix);  // set cameraToWorld to inverse of pcViewMatrix, transposed

    // Extract position and rotation from camera-to-world matrix
    const translation: pc.Vec3 = cameraToWorld.getTranslation();
    // const rotation = new pc.Quat();
    // rotation.setFromMat4(cameraToWorld);

    const eulers = cameraToWorld.getEulerAngles();
    const scale : pc.Vec3 = cameraToWorld.getScale();
    

    

    cameraEntity.setPosition(translation);
    cameraEntity.setEulerAngles(eulers);
}