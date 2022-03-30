// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
var nvidia = nvidia || {};

// Main Controller ------------------------------------------------------------
nvidia.Controller = function() {
  this.supported_types = ["mesh", "pointcloud"];  // TODO: extend to others (Requires DOM additions)
  this.ws = null;              // web socket
  this.dir_info = null;        // information about checkpoint log directory
  this.renderers = {};         // type_string : [list]
  this.active_views = {};      // type_str : [ [category, mesh_id, timecode] ]

  this.init();
};

nvidia.Controller.prototype.init = function() {
  this.initWebSocket();
  this.initRenderers();
  this.initSidebarEvents();
};

// Warning: DOM-dependent
nvidia.Controller.prototype.getHeaderId = function(type, i) {
  return type + "-header" + i;
};

nvidia.Controller.prototype.getInfoId = function(type, i) {
  return type + "-info" + i;
};

nvidia.Controller.prototype.getViewContainerId = function(type, i) {
  return type + "-view" + i;
};

nvidia.Controller.prototype.initRenderers = function() {
  let camera = null;
  for (let t = 0; t < this.supported_types.length; ++t) {
    const type = this.supported_types[t];
    this.renderers[type] = [];

    // DOM is built to support a set number of renderers per type
    const containers = $("#" + type + "-all .view-container");
    nvidia.util.timed_log("Parsed " + containers.length + " " + type + " viewport containers.");

    // Then assume existence of ids: {type}-header{i}, {type}-view{i}, {type}-info{i}
    for (let i = 0; i < containers.length; ++i) {
      const id = this.getViewContainerId(type, i);
      const new_renderer = new nvidia.ThreeJsRenderer(id, camera);
      if (this.shouldLinkCameras() && camera === null) {
        camera = new_renderer.camera;
      }
      this.renderers[type].push(new_renderer);
    }
  }
};

nvidia.Controller.prototype.makeGeometryRequest = function(type, view_id, time) {
  if (this.active_views[type].length <= view_id) {
    nvidia.util.timed_log('Cannot make ' + type + ' geometry request for non-existent viewport ' + view_id);
    return null;
  }
  const view_info = this.active_views[type][view_id];
  let req = {
      type: type,
      category: view_info["category"],
      id: view_info["id"],
      time: view_info["time"],
      view_id: view_id
    };

  if (time !== undefined) {
    // Request update
    req.current_time = view_info["time"];
    req.time = time;
  }
  return req;
};

nvidia.Controller.prototype.initViews = function() {
  // Find largest timestamp and update the timestamp controller

  // Select which meshes to display
  let end_time = 0;
  for (let t = 0; t < this.supported_types.length; ++t) {
    const type = this.supported_types[t];
    const categories = this.dir_info[type];
    const renderers = this.renderers[type];
    this.active_views[type] = [];  // reset

    $("#" + type + "-all .id").empty();
    const cat_selectors = $("#" + type + "-all .cat");
    cat_selectors.empty();
    cat_selectors.off("change");

    if (!categories || categories.length == 0) {
      nvidia.util.timed_log("No entries for type " + type);
      continue;
    }

    // Update DOM to reflect categories
    cat_selectors.append($("<option></option>")
        .attr("value", "-1").text("No Category").attr("selected", true));
    for (let i = 0; i < categories.length; ++i) {
      cat_selectors.append($("<option></option>")
          .attr("value", i).text(categories[i]["category"]));
      end_time = Math.max(categories[i]["end_time"], end_time);
    }

    // Select an initial set of categories and ids to display.
    // Currently selects 0th id from all categories, if there are enough viewports.
    // If more viewports remain, selects subsequent ids from the last category if it has them.
    // Better logic is definitely possible, but none would fit all scenarios.
    let id_idx = 0;
    const nviews = renderers.length;
    for (let i = 0; i < nviews; ++i) {
      let cat_idx = Math.min(i, categories.length - 1);
      let cat = categories[cat_idx];

      if (i >= categories.length) {
        id_idx = Math.min(cat["ids"].length - 1, id_idx + 1);
      }
      this.active_views[type].push({
        category: cat["category"],
        id: cat["ids"][id_idx],
        time: cat["end_time"]
      });

      const hsel = "#" + this.getHeaderId(type, i);
      const cat_dropdown = $(hsel + " .cat");
      cat_dropdown.val(cat_idx);

      const id_dropdown = $(hsel + " .id");
      id_dropdown.off("change");
      id_dropdown.empty();
      $.each(cat["ids"], function(v) {
        id_dropdown.append($("<option></option>")
            .attr("value", v).text("id " + v)); });
      id_dropdown.val(cat["ids"][id_idx]);

      const callback_maker = function(controller, typename, view_idx) {
        return function() { controller.onViewDropdownChange(typename, view_idx); }
        };
      id_dropdown.on("change",  callback_maker(this, type, i));
      cat_dropdown.on("change", callback_maker(this, type, i));

      if (cat["ids"].length < 2) {
        id_dropdown.hide();
      }
    }
  }
  $("#timeslider").attr("max", end_time).val(end_time);
  this.requestGeometryForAllViews();
};

nvidia.Controller.prototype.initSidebarEvents = function() {
  // Note: this is very DOM-dependent
  $("#timeslider").on("change", function(c) {
    return function(e) { c.onTimeSliderValueChange($("#timeslider").val()); }; }(this));

  $("#radius").on("change", function(e) {nvidia.util.updateCurrentUrlParam("radius", $(this).val());});
  $("#linkcam").on("change", function(e) {nvidia.util.updateCurrentUrlParam("linkcam", this.checked);});
  $("#maxviews").on("change", function(e) {nvidia.util.updateCurrentUrlParam("maxviews", $(this).val());});

  $("#refresh").click(function(e) { location.reload(); });
};

nvidia.Controller.prototype.getSphereRadius = function() {
  return $("#radius").val();
};

nvidia.Controller.prototype.shouldLinkCameras = function() {
  return $("#linkcam")[0].checked;
};

nvidia.Controller.prototype.onTimeSliderValueChange = function(time) {
  this.requestGeometryForAllViews(time);
};

nvidia.Controller.prototype.onViewDropdownChange = function(type, view_id) {
  const hsel = "#" + this.getHeaderId(type, view_id);
  const cat_val = $(hsel + " .cat").val();
  const id_val = $(hsel + " .id").val();

  this.active_views[type][view_id].id = id_val;
  this.active_views[type][view_id].category = this.dir_info[type][cat_val].category;
  this.active_views[type][view_id].time = $("#timeslider").val();

  this.requestGeometry([this.makeGeometryRequest(type, view_id)]);
};

nvidia.Controller.prototype.requestGeometryForAllViews = function(time) {
  let geo_requests = [];
  for (let t = 0; t < this.supported_types.length; ++t) {
    const type = this.supported_types[t];

    for (let view_id = 0; view_id < this.active_views[type].length; ++view_id) {
      geo_requests.push(this.makeGeometryRequest(type, view_id, time));
    }
  }
  this.requestGeometry(geo_requests);
};

nvidia.Controller.prototype.initWebSocket = function() {
  this.ws = new WebSocket("ws://" + window.location.host + "/websocket/"); //window.location.pathname);
  this.ws.binaryType = "arraybuffer";

  this.ws.onopen = function(c){return function() { c.onopen();} }(this);
  this.ws.onmessage = function(c){return function(evt) { c.onmessage(evt);} }(this);
  this.ws.onclose = function(c){return function() { c.onclose();} }(this);
};

nvidia.Controller.prototype.requestGeometry = function(geo_requests) {
  var request = {
    type: "geometry",
    data: geo_requests
  }
  this.sendMessage(JSON.stringify(request));
};

nvidia.Controller.prototype.sendMessage = function(str) {
  nvidia.util.timed_log("Sending message: " + str);
  this.ws.send(str);
};

nvidia.Controller.prototype.onopen = function() {
  nvidia.util.timed_log("Connection open");
};

nvidia.Controller.prototype.onmessage = function (evt) {
  // TODO: ensure this part happens in the background, if not already so
  nvidia.util.timed_log("Message received");
  //ws_evt = evt;

  if (typeof evt.data === "string") {
    nvidia.util.timed_log("Got text message: " + evt.data);

    this.processTextMessage(evt.data);

  } else if (evt.data instanceof ArrayBuffer) {
    nvidia.util.timed_log("Got arraybuffer message");

    const geo_data = this.parseBinaryGeometry(evt.data);
    if (geo_data) {
      this.renderers[geo_data.type][geo_data.view_id].setMeshes(geo_data.geos);
      this.active_views[geo_data.type][geo_data.view_id].time = geo_data.time;

      var stats_str = "";
      if (geo_data.type === "mesh") {
        nvidia.util.timed_log("Setting meshes for " + geo_data.view_id);
        stats_str = "nverts: " + geo_data.geos[0].getAttribute("position").count;
      } else if (geo_data.type === "pointcloud") {
        nvidia.util.timed_log("Setting meshes for " + geo_data.view_id);
        stats_str = "npts: " + geo_data.geos[0].getAttribute("instanceTranslation").count;
      }
      const bbox = nvidia.geometry.GetBoundingBox(geo_data.geos);
      bbox_str = "bbox: " + bbox.min.x.toFixed(3) + "..." + bbox.max.x.toFixed(3) + ", " +
        bbox.min.y.toFixed(3) + "..." + bbox.max.y.toFixed(3) + ", " +
        bbox.min.z.toFixed(3) + "..." + bbox.max.z.toFixed(3);
      const infoel = $("#" + this.getInfoId(geo_data.type, geo_data.view_id));
      infoel.empty();
      infoel.append($("<div></div>").text(bbox_str));
      infoel.append($("<div></div>").text(stats_str));
      infoel.append($("<div></div>").text("timecode: " + geo_data.time));
    }
  } else {
    nvidia.util.timed_log("Got unknown message");
    console.log(evt);
  }
};

nvidia.Controller.prototype.parseBinaryGeometry = function(binary_data) {
  nvidia.util.timed_log("Parsing binary data");
  const global_info = new Int32Array(binary_data, 0, 4);
  const data_type = global_info[0];
  const viewport_id = global_info[1];
  const snap_time = global_info[2];
  const tbd_value = global_info[3];
  let data_type_name = "unknown";

  if (data_type === 0) {
    const geos = nvidia.geometry.BufferedGeometriesFromBinary(binary_data, 4 * 4);
    return {
      "type": "mesh",
      "geos": geos,
      "view_id": viewport_id,
      "time": snap_time
    };
  } else if (data_type === 1) {
    const geos = nvidia.geometry.PtCloudsFromBinary(binary_data, 4 * 4, this.getSphereRadius());
    return {
     "type": "pointcloud",
      "geos": geos,
      "view_id": viewport_id,
      "time": snap_time
    };
  } else {
    nvidia.util.timed_log("Received unsupported data type " + data_type + " for viewport " + viewport_id);
    return null;
  }
};

nvidia.Controller.prototype.processTextMessage = function(message_data) {
  const message = JSON.parse(message_data);
  if (message["type"] === "dirinfo" && message["data"]) {
    this.dir_info = message["data"];
    this.initViews();
  } else {
    nvidia.util.timed_log("Unexpected message:");
    console.log(message);
  }
};

nvidia.Controller.prototype.onclose = function() {
  nvidia.util.timed_log("Connection closed");
};

if (typeof module !== 'undefined') {
    module.exports = nvidia.Controller;
}
