import * as THREE from 'three';
import React, { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react';
import { z } from 'zod';

import { BehaviorRegister, ElementBoundBehavior } from '../../../core/behavior';
import { adjustByDevicePixelRatio } from '../../../core/viewport';
import { logger } from '../../../util/logging';
import { defaultCamera } from '../camera';
import { defaultMesh, disposeObject } from '../geometry';


/**
 * Schema for {@link ThreeRendererBehaviorComponent}. React-component schemas
 * describe the JSX props passed by the viewer; the actual prop defaults live
 * in the component's destructuring (keep them in sync).
 */
export const ThreeRendererOptionsSchema = z.object({
    ownsGeometry: z.boolean().default(true)
        .describe('When true, the renderer disposes the active mesh\'s '
                  + 'geometry/materials on swap. Set to false if upstream '
                  + 'code retains references.'),
    clearColor: z.number().int().default(0xffffff)
        .describe('WebGL clear (background) color as a 24-bit hex integer, e.g. 0xffffff for white.'),
    clearAlpha: z.number().min(0).max(1).default(1.0)
        .describe('Clear color alpha. 0 = fully transparent, 1 = fully opaque.'),
    defaultMeshColor: z.number().int().default(0xcccccc)
        .describe('Color applied to untextured (no map) MeshLambertMaterial faces, as a 24-bit hex integer.'),
});

export type ThreeRendererProps = z.infer<typeof ThreeRendererOptionsSchema>;

export interface ThreeRendererHandle extends ElementBoundBehavior {
    setMesh: (mesh: THREE.Object3D | null) => void;
    onAnimate: () => void;
    setCamera: (camera: THREE.Camera) => void;  // Note: API needs to change to enable supporting other camera controllers
};

export const ThreeRendererBehaviorComponent: React.FC<ThreeRendererProps> = forwardRef<
    ThreeRendererHandle, ThreeRendererProps>((
        {
            ownsGeometry,
            clearColor,
            clearAlpha,
            defaultMeshColor,
        },
        ref) => {
        const [mesh, setMesh] = useState<THREE.Object3D | null>(null);
        const acitveMeshRef = useRef<THREE.Object3D | null>(null);
        const cameraRef = useRef<THREE.Camera>(defaultCamera());
        const canvasRef = useRef<HTMLCanvasElement>(null);
        const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
        const sceneRef = useRef<THREE.Scene | null>(new THREE.Scene());
        const groupRef = useRef<THREE.Group | null>(new THREE.Group());

        // Initialize the renderer once the active canvas element is set
        const setActiveElement = useCallback((element: HTMLElement | null) => {
            if (canvasRef.current) {
                logger.warn('Three active element already set; ignoring');
                return;
            }
            canvasRef.current = (element as HTMLCanvasElement);

            if (!canvasRef.current) {
                return;
            }

            rendererRef.current = new THREE.WebGLRenderer({ canvas: canvasRef.current, antialias: true });
            //rendererRef.current.setSize(500, 500);
            rendererRef.current.setClearColor(clearColor, clearAlpha);
        }, []);

        // Called in the viewer animate loop
        const onAnimate = useCallback(() => {
            rendererRef.current?.render(sceneRef.current, cameraRef.current);
        }, []);

        const setCamera = useCallback((camera: THREE.Camera) => {
            cameraRef.current = camera;
            // Schedule a render so wheel-zoom and any non-drag camera updates
            // are visible immediately (onAnimate only runs during drag).
            requestAnimationFrame(() => {
                if (rendererRef.current && sceneRef.current) {
                    rendererRef.current.render(sceneRef.current, cameraRef.current);
                }
            });
        }, []);

        const setDimensions = useCallback((width: number, height: number) => {
            const displaySize = adjustByDevicePixelRatio({ width, height });
            if (rendererRef.current) {
                // false = don't touch CSS size; drawing-buffer sizing is managed externally.
                // Scale by DPR: input is CSS pixels, WebGL drawing buffer needs device pixels.
                rendererRef.current.setSize(displaySize.width, displaySize.height, false);
            }
            const cam = cameraRef.current;
            if (cam instanceof THREE.PerspectiveCamera && height > 0) {
                cam.aspect = width / height;
                cam.updateProjectionMatrix();
            }
        }, []);

        const cleanUpMesh = useCallback(() => {
            if (acitveMeshRef.current) {
                groupRef.current.remove(acitveMeshRef.current);
                groupRef.current.clear();
                if (ownsGeometry) {
                    disposeObject(acitveMeshRef.current);
                }
                acitveMeshRef.current = null;
            }
        }, [ownsGeometry]);

        // Initialize Three.js scene, only runs once
        useEffect(() => {
            sceneRef.current.add(groupRef.current);

            const light1 = new THREE.DirectionalLight(0xffffff, 0.8);
            light1.position.set(1, 1, 1);
            sceneRef.current.add(light1);

            const light2 = new THREE.DirectionalLight(0xffffff, 0.4);
            light2.position.set(-1, -1, -0.5);
            sceneRef.current.add(light2);

            sceneRef.current.add(new THREE.AmbientLight(0x404040, 0.4));

            return () => {
                // Clean up on component unmount; note that renderer is created in setActiveElement
                rendererRef.current?.dispose();
                cleanUpMesh();
                disposeObject(sceneRef.current);
                sceneRef.current = null;
                groupRef.current = null;
            };
        }, []);  // Only runs once


        // Update mesh when meshData changes
        useEffect(() => {
            if (mesh === acitveMeshRef.current) return;

            cleanUpMesh();
            groupRef.current.add(mesh);
            const mat = mesh && (mesh as THREE.Mesh).material;
            if (mat && !Array.isArray(mat) && !(mat as THREE.MeshLambertMaterial).map) {
                (mat as THREE.MeshLambertMaterial).color?.setHex(defaultMeshColor);
            }
            acitveMeshRef.current = mesh;
            console.log('>> Setting new mesh');
            console.log(mesh);
        }, [mesh]);

        useImperativeHandle(ref, () => ({
            // ElementBoundBehavior
            setActiveElement: setActiveElement,
            elementType: (): string => { return "canvas" },
            setDimensions: setDimensions,
            // TODO: other, should be also declared in an interface
            onAnimate: onAnimate,
            setCamera: setCamera,
            // Custom getters
            setMesh
        }), []);

        return null;
    });

(ThreeRendererBehaviorComponent as any).schema = ThreeRendererOptionsSchema;
BehaviorRegister.register('threejs_render', ThreeRendererBehaviorComponent,
    'Three.js mesh renderer bound to an HTMLCanvasElement. Imperative handle '
    + 'exposes `setMesh` and `setCamera`.');
