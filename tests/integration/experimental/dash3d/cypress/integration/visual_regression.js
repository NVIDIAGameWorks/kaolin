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

const assert = require('assert');


const TYPES_TO_TEST = ['mesh', 'pointcloud'];
const NVIEWS = 2;

// This tests the renderings in the viewports against ground truth images
describe('Visual Regression', () => {
    beforeEach(function() {
        // To update these can use one of 2 ways:
        // 1. Look at test output saved in DEBUG_FOLDER
        // 2. Load dash3d localhost:8008 and run commands like this in console:
        // nvidia.util.downloadURL('mesh0.png', $("#mesh-view0 canvas")[0].toDataURL())

        // Initial renderings
        cy.fixture('images/mesh_gt_id0.png').as('mesh0_data'); // fixture data names
        cy.fixture('images/mesh_output_id0_final.png').as('mesh1_data');
        cy.fixture('images/pointcloud_input_id0.png').as('pointcloud0_data');
        cy.fixture('images/pointcloud_output_id0_final.png').as('pointcloud1_data');


        // Specific renderings (caused by user input)
        cy.fixture('images/mesh_gt_id0.png').as('mesh_ground_truth_id0');
        cy.fixture('images/mesh_gt_id1.png').as('mesh_ground_truth_id1');
        cy.fixture('images/mesh_output_id0_final.png').as('mesh_output_id0');  // last it
        cy.fixture('images/mesh_output_id0_it50.png').as('mesh_output_id0_it50');
        cy.fixture('images/mesh_output_id1_final.png').as('mesh_output_id1');  // last it
        cy.fixture('images/mesh_output_id1_it50.png').as('mesh_output_id1_it50');
        cy.fixture('images/pointcloud_input_id0.png').as('pointcloud_input_id0');
        cy.fixture('images/pointcloud_input_id1.png').as('pointcloud_input_id1');
        cy.fixture('images/pointcloud_output_id0_final.png').as('pointcloud_output_id0');  // last it
        cy.fixture('images/pointcloud_output_id0_it50.png').as('pointcloud_output_id0_it50');
        cy.fixture('images/pointcloud_output_id1_final.png').as('pointcloud_output_id1');  // last it
        cy.fixture('images/pointcloud_output_id1_it50.png').as('pointcloud_output_id1_it50');
    })
    it('Initial Page Rendering', () => {
        cy.visit('http://localhost:8008/');

        // Note: this part depends on the initial rendering, which may change
        cy.wait(2000).then(() => {
            cy.wrap(TYPES_TO_TEST).each((tname) => {
                cy.wrap([0, 1]).each((v) => {
                    // e.g. '#mesh-view0 canvas'
                    var view_selector = '#' + tname + '-view' + v + ' canvas';
                    var data_name = '@' + tname + v + '_data';  // fixture data name
                    cy.checkCanvasRendering(view_selector, data_name, 'test_initial_render');
                });
            });
        });
    });
    it('Setting Category and ID', () => {
        cy.visit('http://localhost:8008/');

        // Select the right id and category and test that we can load
        // requested geometry in every viewport
        var cats_per_type = { 'mesh': ['ground_truth', 'output'],
                              'pointcloud': ['input', 'output'] };
        cy.wait(2000).then(() => {
            cy.wrap(TYPES_TO_TEST).each((tname) => {
                cy.wrap([0, 1]).each((view_id) => {
                    cy.wrap(cats_per_type[tname]).each((cat_name) => {
                        cy.wrap([0, 1]).each((mesh_id) => {
                            // e.g. '#mesh-view0 canvas'
                            var view_selector = '#' + tname + '-view' + view_id + ' canvas';
                            var category_selector = '#' + tname + '-header' + view_id + ' select.cat';
                            var id_selector = '#' + tname + '-header' + view_id + ' select.id';
                            var data_name = '@' + tname + '_' + cat_name + '_id' + mesh_id;
                            // Set category and id in the viewport
                            cy.get(id_selector).select('id ' + mesh_id).then(() => {
                                cy.get(category_selector).select(cat_name).wait(1000).then(() => {
                                    console.log('Set category ' + cat_name + ' and id ' + mesh_id);
                                    // Check rendering
                                    cy.checkCanvasRendering(view_selector, data_name, 'test_set_category_and_id');

                                });
                            });
                        });
                    });
                });
            });
        });
    });
    it('Setting Global Iteration Number', () => {
        cy.visit('http://localhost:8008/');

        cy.get('#mesh-header0 select.cat').select('output').then(() => {
            cy.get('#mesh-header0 select.id').select('id 0').then(() => {
                cy.get('#mesh-header1 select.cat').select('ground_truth').then(() => {
                    cy.get('#mesh-header1 select.id').select('id 0').then(() => {
                        cy.get('#pointcloud-header0 select.cat').select('output').then(() => {
                            cy.get('#pointcloud-header0 select.id').select('id 0').then(() => {
                                cy.get('#pointcloud-header1 select.cat').select('input').then(() => {
                                    cy.get('#pointcloud-header1 select.id').select('id 0').then(() => {
                                        cy.get('#timeslider').invoke('val', 50).trigger('change').wait(1000).then(() => {
                                            let test_subfolder = 'test_set_its';
                                            cy.checkCanvasRendering(
                                                '#mesh-view0 canvas', '@mesh_output_id0_it50', test_subfolder);
                                            cy.checkCanvasRendering(
                                                '#mesh-view1 canvas', '@mesh_ground_truth_id0', test_subfolder);
                                            cy.checkCanvasRendering(
                                                '#pointcloud-view0 canvas', '@pointcloud_output_id0_it50', test_subfolder);
                                            cy.checkCanvasRendering(
                                                '#pointcloud-view1 canvas', '@pointcloud_input_id0', test_subfolder);
                                        });
                                    });
                                });
                            });
                        });
                    });
                });
            });
        });
    });

})
