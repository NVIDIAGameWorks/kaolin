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

nvidia.ThreeJsRenderer = function(elemid, optional_camera) {
  this.meshes = [];
  this.container = document.getElementById(elemid);

  this.target = new THREE.Vector3(0.5, 0.5, 0.5);
  this.xzAngle = 0;
  this.radius = 5;

  this.camera = null;
  this.scene = null;
  this.renderer = null;

  this.mesh = null;

  // this.defaultMaterial = new THREE.MeshBasicMaterial( {
  //   color: 0x76B900,
  //   side: THREE.DoubleSide
  // });

  this.fragment_shaders = {};
  this.defaultMaterial = new THREE.ShaderMaterial({
    vertexShader: $("#shader-vs").text(),
    fragmentShader: $("#shader-fs").text()
  });

  $.get('static/green_plastic.frag', function(u){
    return function(data) {
      u.fragment_shaders["green_plastic"] = data;
      u.defaultMaterial = new THREE.ShaderMaterial({
        vertexShader: $("#shader-vs").text(),
        fragmentShader: u.fragment_shaders["green_plastic"]
      });
      console.log('Initialized green plastic shader');
    };}(this));

  this.init(optional_camera);
};

nvidia.ThreeJsRenderer.prototype.init = function(optional_camera) {
  var aspect = 1.0;  // width/height
  if (optional_camera) {
    this.camera = optional_camera;
  } else {
    this.camera = new THREE.PerspectiveCamera(20, 1.0, 0.1, 10000);
    this.camera.position.y = 0.5;
  }

  this.scene = new THREE.Scene();
  this.scene.background = new THREE.Color( 0xffffff );

  var light = new THREE.DirectionalLight( 0xffffff );
  light.position.set( 0.1, 3, 0.1 );
  this.scene.add( light );
  light.castShadow = true;

  var ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
  this.scene.add(ambientLight);

  // background ----------------------------------------------------------------
  // var canvas = document.createElement( 'canvas' );
  // canvas.width = 128;
  // canvas.height = 128;
  // var context = canvas.getContext( '2d' );
  // var gradient = context.createRadialGradient( canvas.width / 2, canvas.height / 2, 0, canvas.width / 2, canvas.height / 2, canvas.width / 2 );
  // gradient.addColorStop( 0.1, 'rgba(210,210,210,1)' );
  // gradient.addColorStop( 1, 'rgba(255,255,255,1)' );
  // context.fillStyle = gradient;
  // context.fillRect( 0, 0, canvas.width, canvas.height );

  // shadows -------------------------------------------------------------------
  //var shadowTexture = new THREE.CanvasTexture( canvas );
  var shadowMaterial = new THREE.MeshPhongMaterial( {
    color: 0xfafafa,
    side: THREE.DoubleSide
  });
  var shadowGeo = new THREE.PlaneBufferGeometry( 100, 100, 1, 1 );
  var shadowMesh = new THREE.Mesh( shadowGeo, shadowMaterial );
  shadowMesh.position.x = -1.0;
  shadowMesh.position.y = 0;
  shadowMesh.position.z = -1.0;
  shadowMesh.rotation.x = -0.5 * Math.PI;
  shadowMesh.receiveShadow = true;
  //this.scene.add( shadowMesh );

  var worldAxis = new THREE.AxesHelper(20);
  this.scene.add(worldAxis);

  this.renderer = new THREE.WebGLRenderer( { antialias: true, preserveDrawingBuffer: true } );
  this.scene.background = new THREE.Color("rgb(250, 250, 250)");
  this.renderer.setPixelRatio( window.devicePixelRatio );
  this.renderer.setSize($(this.container).width(), $(this.container).width());
  this.renderer.shadowMap.enabled = true;
  this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  this.container.appendChild( this.renderer.domElement );

  var self = this;
  $(document).keyup(function(e) {
    console.log('Key up ' + e.which);
    if (e.which === 85) { // "u" up
      self.target.y += 0.2;

    } else if (e.which === 68) {  // "d" down
      self.target.y -= 0.2;

    } else if (e.which === 82) {  // "r" right
      self.xzAngle += Math.PI * 0.1;
    } else if (e.which === 76) {  // "l" left
      self.xzAngle -= Math.PI * 0.1;

    } else {
      return;
    }
    e.preventDefault();
    self.setManualCamera();
    self.render();
  });

  if (!optional_camera) {
    this.setManualCamera();
  }

  this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
  this.controls.update();

  var renderer = this;
  var animate = function() {
    requestAnimationFrame( animate );
    renderer.controls.update();
    renderer.render();
  };
  animate();
};

// nvidia.ThreeJsRenderer.prototype.initScalePosition = function(boundingBox) {
//   this.model_scale = RenderUtils.computeGeometryScale(boundingBox);
//   this.model_position = new THREE.Vector3(
//       -this.model_scale * (boundingBox.max.x + boundingBox.min.x) / 2.0,
//       -this.model_scale * boundingBox.min.y + 0.2,
//       -this.model_scale * (boundingBox.max.z + boundingBox.min.z) / 2.0);
// };

nvidia.ThreeJsRenderer.prototype.setMeshes = function(geometries, material) {
  // if (!this.meshes) {
  //   var boundingBox = this.getBoundingBox(geometries);
  //   console.log("Found bounding box");
  //   console.log(boundingBox);
  //   this.initScalePosition(boundingBox);
  //   this.meshes = [];
  // } else {
  for (var m = 0; m < this.meshes.length; ++m) {
    this.scene.remove(this.meshes[m]);
  }
  this.meshes = [];
  //}

  if (!material) {
    material = this.defaultMaterial;
  }

  for (var g = 0; g < geometries.length; ++g) {
    var geometry = geometries[g];
    var mesh = new THREE.Mesh(geometry, material);
    // mesh.position.set(this.model_position.x,
    //                   this.model_position.y,
    //                   this.model_position.z);
    // mesh.scale.set(this.model_scale, this.model_scale, this.model_scale);
    // if (geometry.__platoMatrix) {
    //   mesh.customMatrix = geometry.__platoMatrix;
    // }
    this.meshes.push(mesh);
    this.scene.add(mesh);
  }

  this.render();
  console.log("Rendered");
};

nvidia.ThreeJsRenderer.prototype.setManualCamera = function() {
  this.camera.position.z = 0.5 + this.radius * Math.cos(this.xzAngle);
  this.camera.position.x = 0.5 + this.radius * Math.sin(this.xzAngle);
  this.camera.lookAt(this.target);
};

nvidia.ThreeJsRenderer.prototype.render = function() {
  //this.camera.position.x += ( mouseX - this.camera.position.x ) * 0.05;
  //this.camera.position.y += ( - mouseY - this.camera.position.y ) * 0.05;

  this.renderer.render( this.scene, this.camera );
};

if (typeof module !== 'undefined') {
    module.exports = nvidia.ThreeJsRenderer;
}
