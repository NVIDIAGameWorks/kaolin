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

import { flattenMatrix } from '../util/types';

// Camera parameter interfaces
export interface CameraPinholeIntrinsics {
    width: number;
    height: number;
    focal_x: number;
    focal_y: number;
    x0: number;
    y0: number;
    near: number;
    far: number;
    classname?: string;
}

export interface CameraOrthoIntrinsics {
    width: number;
    height: number;
    near: number;
    far: number;
    classname?: string;
}

export interface CameraExtrinsics {
    // 4x4 matrix; `Float32Array[]` is the per-row typed-array form produced when a
    // Python `as_dict()` camera is decoded from the binary wire format (util/io).
    view_matrix: number[][] | number[][][] | Float32Array | Float32Array[] | number[];
}

export interface CameraParameters {
    extrinsics: CameraExtrinsics;
    intrinsics: CameraPinholeIntrinsics | CameraOrthoIntrinsics;
}

export const defaultCameraParameters: CameraParameters = {
    intrinsics: {
        width: 512,
        height: 512,
        focal_x: 618.0386962890625,
        focal_y: 618.0386962890625,
        x0: 0.0,
        y0: 0.0,
        near: 0.01,
        far: 100.0,
        classname: 'pinhole'
    },
    extrinsics: {
        view_matrix: new Float32Array([
            0.7071067690849304, 0.0, -0.7071067690849304, 0.0,
            -0.40824827551841736, 0.8164965510368347, -0.40824827551841736, 0.0,
            0.5773502588272095, 0.5773502588272095, 0.5773502588272095, -1.7320507764816284,
            0.0, 0.0, 0.0, 1.0])
    }
};

/**
 * Returns true if two sets of camera parameters are approximately equal: all 16
 * view-matrix elements and every intrinsics field agree within `eps`. Numeric
 * intrinsics are compared with tolerance; non-numeric fields (e.g. `classname`)
 * must match exactly, and both sides must expose the same set of fields. Any
 * supported `view_matrix` encoding is accepted (it is flattened before compare).
 *
 * @param a First camera parameters.
 * @param b Second camera parameters.
 * @param eps Absolute tolerance for numeric comparisons (default 1e-4).
 */
export function cameraParametersApproxEqual(
    a: CameraParameters,
    b: CameraParameters,
    eps: number = 1e-4,
): boolean {
    const av = flattenMatrix(a.extrinsics.view_matrix);
    const bv = flattenMatrix(b.extrinsics.view_matrix);
    if (av.length !== bv.length) {
        return false;
    }
    for (let i = 0; i < av.length; ++i) {
        if (Math.abs(av[i] - bv[i]) > eps) {
            return false;
        }
    }

    const ai = a.intrinsics as Record<string, any>;
    const bi = b.intrinsics as Record<string, any>;
    const keys = new Set([...Object.keys(ai), ...Object.keys(bi)]);
    for (const k of keys) {
        const x = ai[k];
        const y = bi[k];
        if (typeof x === 'number' && typeof y === 'number') {
            if (Math.abs(x - y) > eps) {
                return false;
            }
        } else if (x !== y) {
            return false;
        }
    }
    return true;
}