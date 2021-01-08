// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

var nvidia = nvidia || {};
nvidia.test = nvidia.test || {};

/** A few client side convenience functions to use during cypress tests. */
nvidia.test.canvas = null;

nvidia.test.getCanvas = function(width) {
    if (!nvidia.test.canvas) {
        nvidia.test.canvas = document.createElement("canvas");
    }
    nvidia.test.canvas.width = width;
    nvidia.test.canvas.height = width;
    return nvidia.test.canvas;
};

nvidia.test.stripBase64Marker = function(dataStr) {
    let marker = ';base64,';
    let idx = dataStr.indexOf(marker) + marker.length;
    return dataStr.substring(idx);
};

nvidia.test.convertDataUrl = function(dataUrl, width) {
    return new Promise((resolve, reject) => {
        let img = document.createElement("img");
        img.onload = function() {
            let canvas = nvidia.test.getCanvas(width);
            let ctx = canvas.getContext("2d");
            ctx.drawImage(img, 0, 0, width, width);
            let imgData = ctx.getImageData(0, 0, width, width);
            let imgDataCopy = ctx.createImageData(imgData);
            imgDataCopy.data.set(imgData.data);
            //let imgDataCopy = new Uint8ClampedArray(imgData.data);
            resolve([imgDataCopy, canvas.toDataURL()]);
        };
        img.onerror = reject;
        img.src = dataUrl;
    });
};

nvidia.test.imageDataToDataUrl = function(imageData) {
    let canvas = nvidia.test.getCanvas(imageData.width);
    let ctx = canvas.getContext("2d");
    ctx.putImageData(imageData, 0, 0);
    return canvas.toDataURL();
};

/**
* Computes per-pixel difference between two ImageData objects.
* Returns:
*/
nvidia.test.getImageDiff = function(expected, actual, thresh) {
    if (thresh === undefined) {
        thresh = 0.2;
    }

    let canvas = nvidia.test.getCanvas(expected.width);
    let ctx = canvas.getContext("2d");
    let diff = ctx.createImageData(actual);

    let num_disagree = 0;
    for (let x = 0; x < expected.width; ++x) {
        for (let y = 0; y < expected.height; ++y) {
            let idx = (y * expected.height + x) * 4;
            let dsum = 0;
            for (let c = 0; c < 3; ++c) {
                let d = Math.abs(expected.data[idx] - actual.data[idx]);
                dsum += Math.pow(d/255, 2);
            }
            diff.data[idx + 3] = 255;  // Alpha

            dsum = Math.sqrt(dsum);
            if (dsum > thresh) {
                diff.data[idx] = 255;
                num_disagree += 1;
            }
        }
    }
    return [diff, num_disagree];
};
