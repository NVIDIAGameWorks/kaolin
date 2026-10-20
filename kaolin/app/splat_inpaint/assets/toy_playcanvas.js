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

// TODO: Masha to port this somewhere else; DO NOT REMOVE THIS VALUABLE EXAMPLE

/**
 * PlaycanvasSplatBehavior implemented in plain JavaScript.
 * Requires kaolin (compiled TypeScript bundle) and window.pc (PlayCanvas).
 */
class PlaycanvasSplatBehaviorJS extends kaolin.core.behavior.InteractiveBehavior {
    constructor() {
        super();
        this.app = null;
        this.camera = null;
    }

    elementType() {
        return "canvas";
    }

    setActiveElement(canvas) {
        super.setActiveElement(canvas);
        this.setup();
    }

    async setup() {
        // create an application
        const canvas = this.element;
        if (this.app !== null) {
            return;
        }

        console.log('🍓->>>>>>> Creating playcanvas splat component (JS) --> CUSTOM!!.');
        const pc = kaolin.graphics.playcanvas.resolvePlaycanvas();
        const app = new pc.Application(canvas, {
            graphicsDeviceOptions: {
                alpha: true,
                antialias: false
            }
        });
        app.setCanvasFillMode(pc.FILLMODE_NONE);
        app.setCanvasResolution(pc.RESOLUTION_AUTO);
        app.start();

        const camera = new pc.Entity();
        camera.addComponent('camera', {
            clearColor: new pc.Color(0, 0, 0, 0)
        });
        camera.setPosition(0, 0, 3);
        app.root.addChild(camera);

        // create a light
        const light = new pc.Entity();
        light.addComponent('light');
        light.setEulerAngles(45, 45, 0);
        app.root.addChild(light);

        // create a box
        const box = new pc.Entity();
        box.addComponent('model', {
            type: 'box'
        });
        app.root.addChild(box);

        // rotate the box
        app.on('update', (dt) => box.rotate(10 * dt, 20 * dt, 30 * dt));
    }
}

// Register the behavior with kaolin's BehaviorRegister
kaolin.core.behavior.BehaviorRegister.register("toy_playcanvas", PlaycanvasSplatBehaviorJS);