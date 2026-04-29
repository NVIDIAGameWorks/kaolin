
import * as THREE from 'three';

import { logger } from '../../util/logging';
import { dictToThreeMaterial, disposeMaterial } from './materials';


export function defaultMesh(): THREE.Object3D {
    const geometry = new THREE.SphereGeometry(1, 32, 32);
    const material = new THREE.MeshLambertMaterial({
        color: 0x00ff88,
        wireframe: false
    });
    const mesh = new THREE.Mesh(geometry, material);
    return mesh;
};

export function meshFromMessage(message: Map<string, any>): THREE.Mesh {
    const geometry = new THREE.BufferGeometry();
    let usesIndices = false;

    // Handle positions
    if (message.has('face_vertices')) {
        // Assume Float32Array of length n_triangles * 3 * 3);
        const face_vertices = message.get('face_vertices');
        geometry.setAttribute('position', new THREE.BufferAttribute(face_vertices, 3));
    } else if (message.has('vertices') && message.has('faces')) {
        usesIndices = true;
        // Assume vertices: Float32Array n_vertices * 3
        const vertices = message.get('vertices');
        // Assume Uint32Array n_triangles * 3
        const faces = message.get('faces');
        geometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
        geometry.setIndex(new THREE.BufferAttribute(faces, 1));
    } else {
        logger.warn(`Cannot parse mesh from message with keys ${[...message.keys()]}`)
    }
    geometry.computeBoundingSphere();

    // Handle normals
    if (message.has('face_normals')) {
        // Unindexed normals (one normal per vertex per face)
        if (usesIndices) {
            logger.warn("Cannot use unindexed normals with indexed faces; pass in unindexed geometry instead (mesh.face_vertices).");
        } else {
            const face_normals = message.get('face_normals');
            geometry.setAttribute('normal', new THREE.BufferAttribute(face_normals, 3));
        }
    } else {
        // Indexed normals (per-vertex normals that use same face indices)
        if (message.has('normals') || message.has('face_normals_idx')) {
            logger.warn("Separate normal indices (face_normals_idx) not yet supported. Using same indices as faces.");
        }
        geometry.computeVertexNormals();
    }

    // Handle UVs
    if (message.has('face_uvs')) {
        // Unindexed UVs (one UV coordinate per vertex per face)
        if (usesIndices) {
            logger.warn("Cannot use unindexed uvs with indexed faces; pass in unindexed geometry instead (mesh.face_vertices).");
        } else {
            const face_uvs = message.get('face_uvs');
            geometry.setAttribute('uv', new THREE.BufferAttribute(face_uvs, 2));
        }
    } else if (message.has('uvs') || message.has('face_uvs_idx')) {
        // Separate UV indices not supported in Three.js
        logger.error("Separate UV indices (face_uvs_idx) are not supported. Three.js requires UVs to use the same indices as vertices. Please unroll the geometry on the server side.");
    }

    const materialAssignments = message.get('material_assignments'); // Typed array with material index per face
    const materialDicts = message.get('materials'); // List of material dicts

    if (!materialAssignments || !materialDicts) {
        const defaultMaterial = new THREE.MeshLambertMaterial({
            color: 0x00ff88,
            wireframe: false,
            side: THREE.DoubleSide
        });
        return new THREE.Mesh(geometry, defaultMaterial);
    }

    // Handle materials
    const errorMaterial = new THREE.MeshLambertMaterial({ color: 0xff0000, side: THREE.DoubleSide });
    let materials: THREE.Material[] = [];
    for (const matDict of materialDicts) {
        try {
            const threeMat = dictToThreeMaterial(matDict);
            materials.push(threeMat);
        } catch (e) {
            logger.warn(`Failed to convert material: ${e}. Using default error material.`);
            materials.push(errorMaterial);
        }
    }

    // Create material groups based on assignments
    // Group consecutive faces with same material index
    geometry.clearGroups();
    let currentMaterialIdx = materialAssignments[0];
    let groupStart = 0;

    for (let faceIdx = 1; faceIdx <= materialAssignments.length; faceIdx++) {
        const nextMaterialIdx = faceIdx < materialAssignments.length ? materialAssignments[faceIdx] : -1;

        if (nextMaterialIdx !== currentMaterialIdx) {
            // End current group
            const groupCount = (faceIdx - groupStart) * (usesIndices ? 1 : 3); // 3 vertices per triangle
            geometry.addGroup(groupStart * 3, groupCount, currentMaterialIdx);

            // Start new group
            groupStart = faceIdx;
            currentMaterialIdx = nextMaterialIdx;
        }
    }

    const mesh = new THREE.Mesh(geometry, materials);
    return mesh;
};

/**
 * Disposes all of the contained objects and materials of the given object.
 * 
 * @param object Object3D to dispose (recursively visits children and their resources).
 * @param disposeMaterials When true, also disposes materials and their textures.
 * 
 */
export function disposeObject(object: THREE.Object3D | null | undefined, disposeMaterials: boolean = true): void {
    if (!object) {
        return;
    }
    
    object.traverse((child: any) => {
        disposeObject(child);
    });

    // Dispose own material
    if (disposeMaterials) {
        const material = (object as any).material as THREE.Material | THREE.Material[] | undefined;
        if (material) {
            if (Array.isArray(material)) {
                material.forEach(disposeMaterial);
            } else {
                disposeMaterial(material);
            }
        }
    }

    const geometry = (object as any).geometry as THREE.BufferGeometry | undefined;
    if (geometry && typeof geometry.dispose === 'function') {
        geometry.dispose();
    }

    // Dispose the object itself
    if (typeof (object as any).dispose === 'function') {
        (object as any).dispose();
    }
}

