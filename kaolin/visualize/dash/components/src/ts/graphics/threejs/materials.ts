import * as THREE from 'three';

import { logger } from '../../util/logging';

/**
 * Convert typed array with shape to Three.js DataTexture
 * Expects data in HWC (Height, Width, Channels) format
 */
function typedArrayToTexture(data: Float32Array | Uint8Array | Uint8ClampedArray, shape?: number[]): THREE.DataTexture | null {
    if (!data || !shape || shape.length !== 3) {
        logger.warn(`Cannot create texture from data with shape ${shape}`);
        return null;
    }

    // Expect shape as [height, width, channels] (HWC format)
    const height = shape[0];
    const width = shape[1];
    const channels = shape[2];

    // Verify expected data length
    const expectedLength = height * width * channels;
    if (data.length !== expectedLength) {
        logger.warn(`Texture data length mismatch. Expected HWC format with shape [height=${height}, width=${width}, channels=${channels}] (${expectedLength} elements), got ${data.length} elements. Please ensure textures are in HWC format before sending.`);
        return null;
    }

    let textureData = data;

    // TODO: didn't test on Float32Array

    // Determine format based on channels
    let format: THREE.PixelFormat;
    if (channels === 1) {
        format = THREE.RedFormat;
    } else if (channels === 4) {
        format = THREE.RGBAFormat;
    } else {
        // https://discourse.threejs.org/t/using-three-rgbaformat-instead-of-three-rgbformat/37048/7
        // https://github.com/mrdoob/three.js/pull/23228
        logger.warn(`The texture format with ${channels} channels is not supported, send eighter 1 or 4 channels`);
        return null;
    }

    const texture = new THREE.DataTexture(textureData, width, height, format);
    texture.needsUpdate = true;
    return texture;
}

function isIndexable(value: any): boolean {
    return Array.isArray(value) || ArrayBuffer.isView(value);
}

/**
 * Convert PBRMaterial dict to Three.js Material
 */
export function dictToPBRMaterial(dict: Map<string, any>): THREE.Material {
    // Convert to params expected by THREE js
    const materialParams: any = { side: THREE.DoubleSide };

    function maybe_set(in_name: string, out_name: string, fn: (value: any) => any): void {
        if (dict.has(in_name)) {
            const in_value = dict.get(in_name);
            const out_value = fn(in_value)
            if (out_value) {
                materialParams[out_name] = out_value
            }
        }
    }

    maybe_set('diffuse_color', 'color', (x) => new THREE.Color(x[0], x[1], x[2]));
    maybe_set('roughness_value', 'roughness', (x) => isIndexable(x) ? x[0] : x);
    maybe_set('metallic_value', 'metalness', (x) => isIndexable(x) ? x[0] : x);
    maybe_set('opacity_value', 'opacity', (x) => isIndexable(x) ? x[0] : x);
    maybe_set('diffuse_texture', 'map', (x) => typedArrayToTexture(x, x.shape));
    maybe_set('diffuse_colorspace', 'colorSpace', (x) => x === 'sRGB' ? THREE.SRGBColorSpace : null);
    if (!materialParams.color) {
        materialParams.color = new THREE.Color().setRGB(1.0, 1.0, 1.0)
    }
    maybe_set('roughness_texture', 'roughnessMap', (x) => typedArrayToTexture(x, x.shape));
    maybe_set('metallic_texture', 'metalnessMap', (x) => typedArrayToTexture(x, x.shape));
    maybe_set('normals_texture', 'normalMap', (x) => typedArrayToTexture(x, x.shape));
    maybe_set('opacity_texture', 'alphaMap', (x) => typedArrayToTexture(x, x.shape));
    if (materialParams.alphaMap || (materialParams.opacity && materialParams.opacity < 1.0)) {
        materialParams.transparent = true;
    }
    maybe_set('displacement_texture', 'displacementMap', (x) => typedArrayToTexture(x, x.shape));
    maybe_set('displacement_value', 'displacementScale', (x) => isIndexable(x) ? x[0] : x);

    // Decide between MeshStandardMaterial and MeshPhysicalMaterial
    // Use MeshPhysicalMaterial if clearcoat or IOR are present
    const hasClearcoat = dict.has('clearcoat_value') || dict.has('clearcoat_texture');
    const hasIor = dict.has('ior_value');

    if (hasClearcoat || hasIor) {
        // Use MeshPhysicalMaterial for advanced features
        maybe_set('clearcoat_value', 'clearcoat', (x) => isIndexable(x) ? x[0] : x);
        maybe_set('clearcoat_roughness_value', 'clearcoatRoughness', (x) => isIndexable(x) ? x[0] : x);
        maybe_set('clearcoat_texture', 'clearcoatMap', (x) => typedArrayToTexture(x, x.shape));
        maybe_set('clearcoat_roughness_texture', 'clearcoatRoughnessMap', (x) => typedArrayToTexture(x, x.shape));
        maybe_set('ior_value', 'ior', (x) => isIndexable(x) ? x[0] : x);
        maybe_set('transmittance_value', 'transmission', (x) => isIndexable(x) ? x[0] : x);
        maybe_set('transmittance_texture', 'transmissionMap', (x) => typedArrayToTexture(x, x.shape));

        return new THREE.MeshPhysicalMaterial(materialParams);
    } else {
        // Use MeshStandardMaterial for basic PBR
        return new THREE.MeshStandardMaterial(materialParams);
    }
}

export function dictToThreeMaterial(dict: Map<string, any>): THREE.Material {
    if (dict.has('classname') && dict.get('classname') == 'pbr') {
        return dictToPBRMaterial(dict);
    } else {
        throw new Error(`Unsupported material type ${dict.get('classname')}. Expected a PBR material.`);
    }
}

export function disposeMaterial(mat: THREE.Material | undefined | null): void {

        if (!mat) { return; }

        // TODO: go through documentation and actually use the correct maps
        const maps = [
            'color',
            'normalMap',
            'metalnessMap',
            'roughnessMap',
            'envMap',
            'emissiveMap',
            'alphaMap',
            'displacementMap',
            'aoMap',
            'bumpMap',
            'lightMap',
        ];
        const anyMaterial = mat as any;

        // TODO: way too easy to make mistakes with this implementation; fix to be more robust
        for (const key of maps) {
            const tex = anyMaterial[key] as THREE.Texture | undefined;
            if (tex && typeof tex.dispose === 'function') {
                tex.dispose();
                anyMaterial[key] = null;
            }
        }

        if (typeof mat.dispose === 'function') {
            mat.dispose();
        }

}