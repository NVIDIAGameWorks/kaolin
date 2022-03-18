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
nvidia.util = nvidia.util || {};

nvidia.util.LOG_LEVELS = { DEBUG: 10, INFO: 20, WARN: 30, ERROR: 40 };
nvidia.util.GLOBAL_LOG_LEVEL = nvidia.util.LOG_LEVELS.DEBUG;

nvidia.util.set_global_log_level = function(level_name) {
    nvidia.util.GLOBAL_LOG_LEVEL = nvidia.util.LOG_LEVELS[level_name];
};

nvidia.util.timed_log = function(message, level_name) {
  let level = nvidia.util.LOG_LEVELS['DEBUG'];
  if (level_name && nvidia.util.LOG_LEVELS[level_name]) {
    level = nvidia.util.LOG_LEVELS[level_name];
   }
   if (level >= nvidia.util.GLOBAL_LOG_LEVEL) {
     let d = new Date();
     let msg = d.getMinutes() + ":" + d.getSeconds() + ":" + d.getMilliseconds() + "  " + message;
     if (level >= nvidia.util.LOG_LEVELS['ERROR']) {
        console.error(msg);
     } else if (level >= nvidia.util.LOG_LEVELS['WARN']) {
        console.warn(msg);
     } else if (level >= nvidia.util.LOG_LEVELS['INFO']) {
        console.info(msg);
     } else {
        console.log(msg);
     }
   }
};

nvidia.util.detect_native_byteorder = function() {
    let array_uint32 = new Uint32Array([0x11223344]);
    let array_uint8 = new Uint8Array(array_uint32.buffer);

    if (array_uint8[0] === 0x44) {
        return 'little';
    } else if (array_uint8[0] === 0x11) {
        return 'big';
    } else {
        return 'unknown';
    }
};

nvidia.util.downloadURL = function(filename, url) {
  var a = document.createElement("a");
  document.body.appendChild(a);
  a.style = "display: none";
  a.href = url;
  a.download = filename;
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
};

nvidia.util.updateCurrentUrlParam = function(key, val) {
  let url = new URL(window.location.href);
  let params = url.searchParams;

  if (val == params.get(key)) {
    return false;
  }

  if (val === undefined || val === false) {
    params.delete(key);
  } else {
    params.set(key, val);
  }

  let new_url = url.toString();
  window.history.replaceState({}, document.title, new_url);
  return true;
};

if (typeof module !== 'undefined') {
    module.exports = nvidia.util;
}
