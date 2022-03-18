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
nvidia.geometry = nvidia.geometry || {};

if (typeof require !== 'undefined') {
    var THREE = require('three');
    nvidia.util = require('./util.js');
}

/**
  * Compute bounding box of the geometry types that are parsed by methods in this
  * module.
*/
nvidia.geometry.GetBoundingBox = function(geometries) {
  let bb = new THREE.Box3();
  bb.min.set(0, 0, 0);
  bb.max.set(0, 0, 0);

  for (var g = 0; g < geometries.length; ++g) {
    let bb2 = null;
    if (geometries[g].hasOwnProperty("positionBoundingBox")) {
        bb2 = geometries[g]["positionBoundingBox"];
    } else {
        // Note that this does not work for instanced geometry
        geometries[g].computeBoundingBox();
        bb2 = geometries[g].boundingBox;
    }

    if (g === 0) {
        bb.min.set(bb2.min.x, bb2.min.y, bb2.min.z);
        bb.max.set(bb2.max.x, bb2.max.y, bb2.max.z);
    } else {
        bb.min.x = Math.min(bb.min.x, bb2.min.x);
        bb.min.y = Math.min(bb.min.y, bb2.min.y);
        bb.min.z = Math.min(bb.min.z, bb2.min.z);
        bb.max.x = Math.max(bb.max.x, bb2.max.x);
        bb.max.y = Math.max(bb.max.y, bb2.max.y);
        bb.max.z = Math.max(bb.max.z, bb2.max.z);
    }
  }
  return bb;
};

/**
  * Parse point clouds from binary data written by the web server.
  */
nvidia.geometry.PtCloudsFromBinary = function(binary_data, initial_offset, sphere_radius) {
    let global_info = new Int32Array(binary_data, initial_offset, 4);
    const n_clouds = global_info[0];
    const texture_mode = global_info[1];
    // TBD information stored in the next two ints
    nvidia.util.timed_log("Decoding " + n_clouds + " point clouds");

    if (texture_mode !== 0) {
        console.error('Texture mode ' + texture_mode + ' not supported. ' +
            'Version mismatch between python and JS code.');
    }

    let geometries = [];
    let read_start = initial_offset + 4 * 4;  // 4 * 4 bytes used for n_clouds read above

    for (let m = 0; m < n_clouds; ++m) {
        let meta = new Int32Array(binary_data, read_start, 2);
        read_start += 2 * 4;
        let n_vertices = meta[0];
        let tbd_info = meta[1];
        nvidia.util.timed_log(n_vertices + " points in pointcloud");

        let bounds = new Float32Array(binary_data, read_start, 6);
        read_start += 6 * 4;
        let bbox = new THREE.Box3();
        bbox.min.set(bounds[0], bounds[1], bounds[2]);
        bbox.max.set(bounds[3], bounds[4], bounds[5]);

        let positions = new Float32Array(binary_data, read_start, n_vertices * 3);
        read_start += n_vertices * 3 * 4;

        const sphere = new THREE.SphereBufferGeometry(sphere_radius, 10, 10);
        let geo = new THREE.InstancedBufferGeometry();
        geo.index = sphere.index;
        geo.attributes = sphere.attributes;
        geo.instanceCount = n_vertices;

        geo.setAttribute("instanceTranslation", new THREE.InstancedBufferAttribute(positions, 3)); // built-in
        geo["instanceRadius"] = sphere_radius;  // custom
        geo["positionBoundingBox"] = bbox;  // custom
        geometries.push(geo);
    }
    return geometries;
};

nvidia.geometry.BufferedGeometriesFromBinary = function(binary_data, initial_offset) {
  nvidia.util.timed_log("Parsing binary data");
  var global_info = new Int32Array(binary_data, initial_offset, 4);
  var n_meshes = global_info[0];
  var texture_mode = global_info[1];
  // TBD information stored in the next two ints
  nvidia.util.timed_log("Decoding " + n_meshes + " meshes");

  if (texture_mode !== 0) {
    console.error('Texture mode ' + texture_mode + ' not supported. ' +
        'Version mismatch between python and JS code.');
  }

  var geometries = [];
  var read_start = initial_offset + 4 * 4;  // 4 * 4 bytes used for n_meshes read above
  for (var m = 0; m < n_meshes; ++m) {
    var meta = new Int32Array(binary_data, read_start, 2);
    read_start += 2 * 4;
    var n_vertices = meta[0];
    var n_triangles = meta[1];
    nvidia.util.timed_log(n_vertices + " verts, " + n_triangles + "triangles");

    var geometry = new THREE.BufferGeometry();
    if (n_vertices == 0) {
      nvidia.util.timed_log("Using unindexed geometry");
      var positions = new Float32Array(binary_data, read_start, n_triangles * 3 * 3);
      read_start += n_triangles * 3 * 3 * 4;  // 4 bytes * 3/ vert * 3 vert/ tri
      geometry.setAttribute(
    'position', new THREE.BufferAttribute(positions, 3));
    } else {
      nvidia.util.timed_log("Using indexed geometry");
      var positions = new Float32Array(binary_data, read_start, n_vertices * 3);
      read_start += n_vertices * 3 * 4;
      var triangles = new Uint32Array(binary_data, read_start, n_triangles * 3);
      read_start += n_triangles * 3 * 4;
      geometry.setAttribute( 'position', new THREE.BufferAttribute( positions, 3 ) );
      geometry.setIndex(new THREE.BufferAttribute( triangles, 1 ) );
    }
    geometry.computeBoundingSphere();

    nvidia.util.timed_log("Done parsing mesh " + m);
    geometries.push(geometry);
  }
  return geometries;
};

// Note: there used to be an issue with max chunk size that could be sent through
// a socket, requiring the following for mesh geometries:
// var chunkSize = 65536;
// var offsets = 1; //n_triangles / chunkSize;
// for ( var i = 0; i < offsets; i ++ ) {
//   var offset = {
//     start: i * chunkSize * 3,
//     index: i * chunkSize * 3,
//     count: n_triangles * 3 //Math.min( n_triangles - ( i * chunkSize ), chunkSize ) * 3
//   };
//   geometry.offsets.push( offset );
// }

if (typeof module !== 'undefined') {
    module.exports = nvidia.geometry;
}
