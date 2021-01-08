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

const IMG_WIDTH = 300;
const MAX_DIFFERING_PIXELS = Math.floor(IMG_WIDTH * IMG_WIDTH * 0.02);
const DEBUG_FOLDER = 'tests/integration/experimental/dash3d/cypress/test_output/';

Cypress.Commands.add('checkCanvasRendering', (view_selector, data_name, testsubfolder) => {
    cy.window().then((win) => {
        expect(cy.get(view_selector));
        cy.get(view_selector)
            .then(($el) => {
                return win.nvidia.test.convertDataUrl($el.get(0).toDataURL(), IMG_WIDTH);
            })
            .then((actual) => {
                cy.get(data_name)
                    .then((img_data) => {
                        return win.nvidia.test.convertDataUrl('data:image/png;base64,' + img_data, IMG_WIDTH);
                    })
                    .then((expected) => {
                        console.log('Actual: ');
                        console.log(actual);
                        console.log('Expected: ');
                        console.log(expected);
                        let cmpare = win.nvidia.test.getImageDiff(expected[0], actual[0]);
                        console.log(cmpare);
                        let fprefix = DEBUG_FOLDER + testsubfolder;
                        cy.writeFile(fprefix + '/expected/' + data_name.slice(1) + '_expected.png',
                                     win.nvidia.test.stripBase64Marker(expected[1]), 'base64')
                            .then(() => {
                                cy.writeFile(fprefix + '/actual/' + data_name.slice(1) + '.png',
                                             win.nvidia.test.stripBase64Marker(actual[1]), 'base64');
                                })
                            .then(() => {
                                cy.writeFile(fprefix + '/expected/' + data_name.slice(1) + '_diff.png',
                                             win.nvidia.test.stripBase64Marker(
                                                 win.nvidia.test.imageDataToDataUrl(cmpare[0])), 'base64');
                                })
                            .then(() => {
                                expect(cmpare[1]).to.be.lessThan(MAX_DIFFERING_PIXELS);
                                });
                    });
            });
        });
});