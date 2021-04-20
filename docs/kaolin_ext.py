# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

KAOLIN_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)

def run_apidoc(_):
    from sphinx.ext import apidoc
    EXCLUDE_PATHS = [
        str(os.path.join(KAOLIN_ROOT, path)) for path in [
            "**.so",
            "kaolin/ops/conversions/pointcloud.py",
            "kaolin/ops/conversions/sdf.py",
            "kaolin/ops/conversions/trianglemesh.py",
            "kaolin/ops/conversions/voxelgrid.py",
            "kaolin/ops/mesh/check_sign.py",
            "kaolin/ops/mesh/mesh.py",
            "kaolin/ops/mesh/trianglemesh.py",
            "kaolin/render/mesh/rasterization.py",
            "kaolin/render/mesh/utils.py",
            "kaolin/visualize/timelapse.py"
        ]
    ]

    EXCLUDE_GEN_RST = [str(os.path.join(KAOLIN_ROOT, "docs", "modules", path))
                       for path in ["setup.rst", "kaolin.rst", "kaolin.version.rst"]]
    DOCS_MODULE_PATH = os.path.join(KAOLIN_ROOT, "docs", "modules")

    argv = [
      "-eT",
      "-d", "2",
      "--templatedir",
      DOCS_MODULE_PATH,
      "-o",
      DOCS_MODULE_PATH,
      KAOLIN_ROOT,
      *EXCLUDE_PATHS
    ]
    apidoc.main(argv)

    for f in EXCLUDE_GEN_RST:
        os.remove(f)

def setup(app):
    app.connect("builder-inited", run_apidoc)
