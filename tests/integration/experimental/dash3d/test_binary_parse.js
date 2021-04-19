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

var assert = require('assert');
var THREE = require('three');
var fs = require('fs');
var path = require('path');

var geometry = require('../../../../kaolin/experimental/dash3d/src/geometry.js');
var util = require('../../../../kaolin/experimental/dash3d/src/util.js');

var binaries = [];
before(function(done){
    util.set_global_log_level('INFO');  // Comment out to DEBUG if needed

    var paths = ['meshes0_1.bin', 'meshes2.bin', 'clouds0_1.bin', 'clouds2.bin'];
    for (var i = 0; i < paths.length; ++i) {
        var p = path.join(__dirname, '_out', paths[i]);
        util.timed_log('Parsing binary file at path ' + p);
        var res = fs.readFileSync(p);
        var res_buffer = new Uint8Array(res).buffer;
        binaries.push(res_buffer);
    }
    done();
});

describe("Binary Mesh Parsing", function() {
  describe("Reading and checking two meshes from _out/meshes0_1.bin", function() {
    let geos = null;
    it('two meshes should be parsed', function() {
        geos = geometry.BufferedGeometriesFromBinary(binaries[0], 0);
        assert.equal(geos.length, 2);
    });
    it('two meshes should have correct number of vertices and faces', function() {
        assert.equal(geos[0].getAttribute('position').count, 4);
        assert.equal(geos[0].getIndex().count, 2 * 3);

        assert.equal(geos[1].getAttribute('position').count, 100);
        assert.equal(geos[1].getIndex().count, 100 * 3);
    });
    it('first mesh should have correct geometry values', function() {
        let expected_face_idx = [0, 1, 2, 2, 1, 3];
        for (let i = 0; i < expected_face_idx.length; ++i) {
            assert.equal(geos[0].getIndex().array[i], expected_face_idx[i],
                         'unexpected face index at ' + i);
        }
        let expected_positions = [1.0, 2.0, 3.0,
                                  10.0, 20.0, 30.0,
                                  2.0, 4.0, 6.0,
                                  15.0, 25.0, 35.0];
        for (let i = 0; i < expected_positions.length; ++i) {
            assert.equal(geos[0].getAttribute('position').array[i],
                         expected_positions[i],
                         'unexpected position at ' + i);
        }
    });
    it('correct bounding box should be computed for both meshes', function() {
        let bbox = geometry.GetBoundingBox(geos);
        assert.equal(bbox.min.x, 0);
        assert.equal(bbox.min.y, 1);
        assert.equal(bbox.min.z, 2);
        assert.equal(bbox.max.x, 297);
        assert.equal(bbox.max.y, 298);
        assert.equal(bbox.max.z, 299);
    });
  });
  describe("Reading and checking one mesh from _out/meshes2.bin", function() {
    let geos = null;
    it('one mesh should be parsed', function() {
        geos = geometry.BufferedGeometriesFromBinary(binaries[1], 0);
        assert.equal(geos.length, 1);
    });
    it('one mesh should have correct number of vertices and faces', function() {
        assert.equal(geos[0].getAttribute('position').count, 3000);
        assert.equal(geos[0].getIndex().count, 6000 * 3);
    });
  });
});

describe("Binary Pointcloud Parsing", function() {
  describe("Reading and checking two point clouds from _out/clouds0_1.bin", function() {
    let geos = null;
    it('two point clouds should be parsed', function() {
        geos = geometry.PtCloudsFromBinary(binaries[2], 0);
        assert.equal(geos.length, 2);
    });
    it('two point clouds should have correct number of points', function() {
        assert.equal(geos[0].instanceCount, 4);
        assert.equal(geos[1].instanceCount, 100);
    });
    it('first point cloud should have correct geometry values', function() {
        let expected_positions = [1.0, 2.0, 3.0,
                                  10.0, 20.0, 30.0,
                                  2.0, 4.0, 6.0,
                                  15.0, 25.0, 35.0];
        for (let i = 0; i < expected_positions.length; ++i) {
            assert.equal(geos[0].getAttribute('instanceTranslation').array[i],
                         expected_positions[i],
                         'unexpected position at ' + i);
        }
    });
    it('second point cloud should have correct geometry values', function() {
        for (let i = 0; i < 300; ++i) {
            assert.equal(geos[1].getAttribute('instanceTranslation').array[i],
                         i + 0.0,
                         'unexpected position at ' + i);
        }
    });
    it('correct bounding box should be computed for both point clouds', function() {
        let bbox = geometry.GetBoundingBox(geos);
        assert.equal(Math.round(bbox.min.x * 1000), 0);
        assert.equal(Math.round(bbox.min.y * 1000), 1 * 1000);
        assert.equal(Math.round(bbox.min.z * 1000), 2 * 1000);
        assert.equal(Math.round(bbox.max.x * 1000), 297 * 1000);
        assert.equal(Math.round(bbox.max.y * 1000), 298 * 1000);
        assert.equal(Math.round(bbox.max.z * 1000), 299 * 1000);
    });
  });
  describe("Reading and checking one point cloud from _out/clouds2.bin", function() {
    let geos = null;
    it('one point cloud should be parsed', function() {
        geos = geometry.PtCloudsFromBinary(binaries[3], 0);
        assert.equal(geos.length, 1);
    });
    it('one point cloud should have correct number of points', function() {
        assert.equal(geos[0].instanceCount, 3000);
    });
  });
})