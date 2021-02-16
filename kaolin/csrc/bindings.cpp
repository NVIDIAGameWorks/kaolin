// Copyright (c) 2020,21 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <torch/extension.h>

#include "./ops/packed_simple_sum.h"
#include "./ops/tile_to_packed.h"
#include "./ops/mesh/mesh_intersection.h"
#include "./ops/conversions/unbatched_mcube/unbatched_mcube.h"
#include "./metrics/sided_distance.h"
#include "./metrics/unbatched_triangle_distance.h"
#include "./render/dibr.h"
#include "./ops/conversions/mesh_to_spc/mesh_to_spc.h"
#include "./ops/spc/spc.h"
#include "./ops/spc/feature_grids.h"
#include "./render/spc/raytrace.h"
#include "./ops/spc/spc_query.h"
#include "./ops/spc/spc_utils.h"


namespace kaolin {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef WITH_CUDA
  py::module ops = m.def_submodule("ops");
  ops.def("packed_simple_sum_cuda", &packed_simple_sum_cuda);
  ops.def("packed_simple_sum_out_cuda", &packed_simple_sum_out_cuda);
  ops.def("tile_to_packed_cuda", &tile_to_packed_cuda);
  ops.def("tile_to_packed_out_cuda", &tile_to_packed_out_cuda);
    py::module ops_mesh = ops.def_submodule("mesh");
    ops_mesh.def("unbatched_mesh_intersection_cuda", &unbatched_mesh_intersection_cuda);
    py::module ops_conversions = ops.def_submodule("conversions");
    ops_conversions.def("unbatched_mcube_forward_cuda", &unbatched_mcube_forward_cuda);
    ops_conversions.def("mesh_to_spc", &mesh_to_spc);
    py::module ops_spc = ops.def_submodule("spc");
    ops_spc.def("spc_query", &spc_query);
    ops_spc.def("spc_point2morton", &spc_point2morton);
    ops_spc.def("spc_morton2point", &spc_morton2point);
    ops_spc.def("spc_point2coeff", &spc_point2coeff);
    ops_spc.def("spc_point2jacobian", &spc_point2jacobian);
    ops_spc.def("spc_point2corners", &spc_point2corners);
    ops_spc.def("points_to_octree", &points_to_octree);
    ops_spc.def("ScanOctrees", &ScanOctrees);
    ops_spc.def("GeneratePoints", &GeneratePoints);
    ops_spc.def("Conv3d_forward", &Conv3d_forward);
    ops_spc.def("Conv3d_backward", &Conv3d_backward);
    ops_spc.def("ConvTranspose3d_forward", &ConvTranspose3d_forward);
    ops_spc.def("ConvTranspose3d_backward", &ConvTranspose3d_backward);
    ops_spc.def("to_dense_forward", &to_dense_forward);
    ops_spc.def("to_dense_backward", &to_dense_backward);
  py::module metrics = m.def_submodule("metrics");
  metrics.def("sided_distance_forward_cuda", &sided_distance_forward_cuda);
  metrics.def("sided_distance_backward_cuda", &sided_distance_backward_cuda);
  metrics.def("unbatched_triangle_distance_forward_cuda", &unbatched_triangle_distance_forward_cuda);
  metrics.def("unbatched_triangle_distance_backward_cuda", &unbatched_triangle_distance_backward_cuda);
  py::module render = m.def_submodule("render");
  py::module render_mesh = render.def_submodule("mesh");
  render_mesh.def("packed_rasterize_forward_cuda", &packed_rasterize_forward_cuda);
  render_mesh.def("generate_soft_mask_cuda", &generate_soft_mask_cuda);
  render_mesh.def("rasterize_backward_cuda", &rasterize_backward_cuda);
  py::module render_spc = render.def_submodule("spc");
  render_spc.def("ray_aabb", &spc_ray_aabb);
  render_spc.def("raytrace", &spc_raytrace);
  render_spc.def("generate_primary_rays", &generate_primary_rays);
  render_spc.def("remove_duplicate_rays", &remove_duplicate_rays);
  render_spc.def("mark_first_hit", &mark_first_hit);
  render_spc.def("generate_shadow_rays", &generate_shadow_rays);
#endif
}

}  // namespace kaolin
