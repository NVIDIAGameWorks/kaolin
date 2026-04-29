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

// =============================================================================
// Tests for @kaolin/graphics/threejs/geometry
//
// These functions build three.js meshes from messages produced on the Python
// side and dispose of three.js resources. Tests are split into:
//   * implemented "basic checks" (structural / type / error contracts), and
//   * pending placeholders (`it('...')` with no body) for the richer material,
//     uv and disposal behavior that still needs to be written.
//
// ── PYTHON INTEROP CONVENTIONS (the part that changes if Python changes) ─────
//
// Mesh data originates on the Python side from
//     kaolin.visualize.web.sockets.RemoteRenderHandler.encode_mesh()
//     (kaolin/visualize/web/sockets.py)
// and arrives here decoded as a `Map<string, any>` (see @kaolin/util/io).
//
// Kaolin sends UN-INDEXED ("face_*") geometry by default: three.js uses a
// single shared index for positions/normals/uvs, whereas typical mesh formats
// use separate per-attribute indices, so the server unrolls the geometry.
//
// Keys produced by Python  --  KEEP THE FIXTURES BELOW IN SYNC WITH THIS:
//
//   face_vertices         Float32Array, length n_tri*3*3   (xyz per face-vertex)
//   face_normals          Float32Array, length n_tri*3*3   (optional)
//   face_uvs              Float32Array, length n_tri*3*2   (optional)
//   material_assignments  Int8Array,    length n_tri       (material idx / face)
//   materials             Array<dict>,  each with "classname" + tensor attrs
//
// Alternative INDEXED path also accepted by meshFromMessage():
//   vertices  Float32Array (n_vert*3)   +   faces  Uint32Array (n_tri*3)
//
// NOT supported (meshFromMessage logs warn/error, does not throw):
//   separate attribute indices: normals / face_normals_idx, uvs / face_uvs_idx,
//   and mixing unindexed attributes (face_normals/face_uvs) with indexed faces.
// =============================================================================

import { assert } from 'chai';
import * as THREE from 'three';

import {
    defaultMesh,
    meshFromMessage,
    disposeObject,
} from '@kaolin/graphics/threejs/geometry';

// -----------------------------------------------------------------------------
// Fixtures mirroring Python `encode_mesh()` output (decoded to a Map). Update
// these helpers if the Python message keys / dtypes / layout change.
// -----------------------------------------------------------------------------

/** Single triangle as UN-INDEXED face_vertices (the Kaolin default path). */
function unindexedTriangleMessage(): Map<string, any> {
    return new Map<string, any>([
        ['face_vertices', new Float32Array([
            0, 0, 0,
            1, 0, 0,
            0, 1, 0,
        ])],
    ]);
}

/** Single triangle as INDEXED vertices + faces (the alternative path). */
function indexedTriangleMessage(): Map<string, any> {
    return new Map<string, any>([
        ['vertices', new Float32Array([
            0, 0, 0,
            1, 0, 0,
            0, 1, 0,
        ])],
        ['faces', new Uint32Array([0, 1, 2])],
    ]);
}

// -----------------------------------------------------------------------------

describe('visualize/dash/graphics/threejs/test_geometry.ts', () => {
    describe('defaultMesh', () => {
        it('returns a three.js Mesh with geometry and material', () => {
            const mesh = defaultMesh();
            assert.instanceOf(mesh, THREE.Mesh);
            assert.instanceOf((mesh as THREE.Mesh).geometry, THREE.BufferGeometry);
            assert.isDefined((mesh as THREE.Mesh).material);
        });
    });

    describe('meshFromMessage (unindexed face_vertices)', () => {
        it('builds non-indexed geometry with a position attribute', () => {
            const mesh = meshFromMessage(unindexedTriangleMessage());

            assert.instanceOf(mesh, THREE.Mesh);
            const position = mesh.geometry.getAttribute('position');
            assert.isDefined(position);
            assert.equal(position.itemSize, 3);
            assert.equal(position.count, 3); // 1 triangle * 3 vertices
            assert.isNull(mesh.geometry.getIndex(), 'unindexed geometry should have no index');
        });

        it('computes vertex normals when none are provided', () => {
            const mesh = meshFromMessage(unindexedTriangleMessage());
            assert.isDefined(mesh.geometry.getAttribute('normal'));
        });

        // TODO: when face_normals are provided, they should be used verbatim
        //       (not recomputed).
        it('uses provided face_normals instead of recomputing');

        // TODO: face_uvs (n_tri*3*2) should be attached as a 2-component uv attr.
        it('attaches face_uvs as a 2-component uv attribute');
    });

    describe('meshFromMessage (indexed vertices + faces)', () => {
        it('builds indexed geometry with position and index', () => {
            const mesh = meshFromMessage(indexedTriangleMessage());

            assert.instanceOf(mesh, THREE.Mesh);
            assert.equal(mesh.geometry.getAttribute('position').count, 3);
            const index = mesh.geometry.getIndex();
            assert.isNotNull(index);
            assert.equal(index!.count, 3);
        });

        // TODO: assert a warning is logged when unindexed face_normals/face_uvs are
        //       combined with indexed faces (unsupported combination).
        it('warns when unindexed attributes are mixed with indexed faces');
    });

    describe('meshFromMessage (materials)', () => {
        // TODO: with `materials` + `material_assignments`, the mesh should carry a
        //       material array and geometry groups grouping consecutive faces by
        //       material index.
        it('creates per-material geometry groups from material_assignments');

        // TODO: a material dict that fails to convert should fall back to the red
        //       error material rather than throwing.
        it('falls back to an error material on conversion failure');
    });

    describe('meshFromMessage (malformed input)', () => {
        // TODO: an empty / unrecognized message should warn and still return a Mesh
        //       (current behavior logs and continues).
        it('handles a message with no recognized geometry keys');
    });

    describe('disposeObject', () => {
        it('handles null / undefined without throwing', () => {
            assert.doesNotThrow(() => disposeObject(null));
            assert.doesNotThrow(() => disposeObject(undefined));
        });

        // NOTE: currently fails with "Maximum call stack size exceeded".
        // geometry.ts does `object.traverse(child => disposeObject(child))`, but
        // THREE.Object3D.traverse() invokes the callback on the object itself
        // (then its descendants), so disposeObject recurses on itself forever.
        // Left pending until the production bug is fixed (e.g. iterate children
        // directly instead of traverse, or skip self in the callback).
        it('disposes a simple mesh without throwing');

        // TODO: spy on geometry/material .dispose() to assert they are actually
        //       called, recursively over children, and that disposeMaterials=false
        //       skips material disposal.
        it('recursively disposes children geometry and materials');
        it('skips material disposal when disposeMaterials is false');
    });
});
