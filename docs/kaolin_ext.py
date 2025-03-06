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
    # This is running sphinx-apidoc which is automatically generating
    # .rst files for each python file in kaolin
    # This won't override existing .rst files
    # Like kaolin.ops.rst where we added an introduction
    from sphinx.ext import apidoc
    # Those files are excluded from parsing
    # Such as files where the functions are forwarded to the parent namespace
    EXCLUDE_PATHS = [
        str(os.path.join(KAOLIN_ROOT, path)) for path in [
            "setup.py",
            "**.so",
            "kaolin/version.py",
            "kaolin/experimental/",
            "kaolin/version.txt",
            "kaolin/io/usd/utils.py",
            "kaolin/io/usd/mesh.py",
            "kaolin/io/usd/materials.py",
            "kaolin/io/usd/voxelgrid.py",
            "kaolin/io/usd/pointcloud.py",
            "kaolin/math/quat/angle_axis.py",
            "kaolin/math/quat/euclidean.py",
            "kaolin/math/quat/matrix44.py",
            "kaolin/math/quat/quaternion.py",
            "kaolin/math/quat/rotation33.py",
            "kaolin/math/quat/transform.py",
            "kaolin/math/quat/util.py",
            "kaolin/ops/conversions/pointcloud.py",
            "kaolin/ops/conversions/sdf.py",
            "kaolin/ops/conversions/trianglemesh.py",
            "kaolin/ops/conversions/voxelgrid.py",
            "kaolin/ops/conversions/tetmesh.py",
            "kaolin/ops/conversions/flexicubes/*.py",
            "kaolin/ops/conversions/gaussians.py",
            "kaolin/ops/mesh/check_sign.py",
            "kaolin/ops/mesh/mesh.py",
            "kaolin/ops/mesh/tetmesh.py",
            "kaolin/ops/mesh/trianglemesh.py",
            "kaolin/ops/gaussian/densifier.py",
            "kaolin/ops/spc/spc.py",
            "kaolin/ops/spc/convolution.py",
            "kaolin/ops/spc/points.py",
            "kaolin/ops/spc/uint8.py",
            "kaolin/ops/spc/raytraced_spc_dataset.py",
            "kaolin/ops/spc/bf_recon.py",
            "kaolin/physics/materials/linear_elastic_material.py",
            "kaolin/physics/materials/material_forces.py",
            "kaolin/physics/materials/material_utils.py",
            "kaolin/physics/materials/muscle_material.py",
            "kaolin/physics/materials/neohookean_elastic_material.py",
            "kaolin/physics/materials/utils.py",
            "kaolin/physics/simplicits/easy_api.py",
            "kaolin/physics/simplicits/losses.py",
            "kaolin/physics/simplicits/network.py",
            "kaolin/physics/simplicits/precomputed.py",
            "kaolin/physics/simplicits/simplicits_scene_forces.py",
            "kaolin/physics/simplicits/skinning.py",
            "kaolin/physics/simplicits/losses_warp.py",
            "kaolin/physics/simplicits/utils.py",
            "kaolin/physics/utils/finite_diff.py",
            "kaolin/physics/utils/force_wrapper.py",
            "kaolin/physics/utils/misc.py",
            "kaolin/physics/utils/optimization.py",
            "kaolin/physics/utils/scene_forces.py",
            "kaolin/physics/utils/torch_utilities.py",
            "kaolin/physics/utils/warp_utilities.py",
            "kaolin/render/lighting/sg.py",
            "kaolin/render/lighting/sh.py",
            "kaolin/render/easy_render/common.py",
            "kaolin/render/easy_render/mesh.py",
            "kaolin/render/mesh/deftet.py",
            "kaolin/render/mesh/dibr.py",
            "kaolin/render/mesh/rasterization.py",
            "kaolin/render/mesh/utils.py",
            "kaolin/render/spc/raytrace.py",
            "kaolin/rep/spc.py",
            "kaolin/visualize/timelapse.py",
            "kaolin/visualize/ipython.py",
            "kaolin/framework/*",
            "kaolin/render/camera/camera.py",
            "kaolin/render/camera/coordinates.py",
            "kaolin/render/camera/extrinsics_backends.py",
            "kaolin/render/camera/extrinsics.py",
            "kaolin/render/camera/intrinsics_ortho.py",
            "kaolin/render/camera/intrinsics_pinhole.py",
            "kaolin/render/camera/intrinsics.py",
            "kaolin/render/camera/legacy.py",
            "kaolin/render/camera/raygen.py",
            "kaolin/render/camera/gsplats.py",
            "kaolin/render/camera/polyscope.py",
            "kaolin/render/camera/trajectory.py",
            "kaolin/non_commercial/flexicubes/",
            "kaolin/non_commercial/flexicubes/flexicubes.py",
            "kaolin/non_commercial/flexicubes/tables.py"

        ]
    ]

    DOCS_MODULE_PATH = os.path.join(KAOLIN_ROOT, "docs", "modules")

    argv = [
        "-eT",
        "-d", "2",
        "--templatedir",
        DOCS_MODULE_PATH,
        "-o", DOCS_MODULE_PATH,
        os.path.join(KAOLIN_ROOT, "kaolin"),
        *EXCLUDE_PATHS
    ]
    apidoc.main(argv)
    os.remove(os.path.join(DOCS_MODULE_PATH, 'kaolin.rst'))


def setup(app):
    app.connect("builder-inited", run_apidoc)
