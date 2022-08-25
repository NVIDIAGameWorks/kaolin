// Copyright (c) 2020,21-22 NVIDIA CORPORATION & AFFILIATES.
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
#include "./render/mesh/deftet.h"
#include "./render/mesh/dibr_soft_mask.h"
#include "./render/mesh/rasterization.h"
#include "./render/sg/unbatched_reduced_sg_inner_product.h"
#include "./ops/conversions/mesh_to_spc/mesh_to_spc.h"
#include "./ops/spc/spc.h"
#include "./ops/spc/feature_grids.h"
#include "./render/spc/raytrace.h"
#include "./ops/spc/query.h"
#include "./ops/spc/point_utils.h"

namespace kaolin {

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::module ops = m.def_submodule("ops");
  ops.def("packed_simple_sum_cuda", &packed_simple_sum_cuda);
  ops.def("packed_simple_sum_out_cuda", &packed_simple_sum_out_cuda);
  ops.def("tile_to_packed_cuda", &tile_to_packed_cuda);
  ops.def("tile_to_packed_out_cuda", &tile_to_packed_out_cuda);
    py::module ops_mesh = ops.def_submodule("mesh");
    ops_mesh.def("unbatched_mesh_intersection_cuda", &unbatched_mesh_intersection_cuda);
    py::module ops_conversions = ops.def_submodule("conversions");
    ops_conversions.def("unbatched_mcube_forward_cuda", &unbatched_mcube_forward_cuda);
    ops_conversions.def("mesh_to_spc_cuda", &mesh_to_spc_cuda);
    py::module ops_spc = ops.def_submodule("spc");
#if WITH_CUDA
    ops_spc.def("query_cuda", &query_cuda);
    ops_spc.def("query_multiscale_cuda", &query_multiscale_cuda);
    ops_spc.def("points_to_morton_cuda", &points_to_morton_cuda);
    ops_spc.def("morton_to_points_cuda", &morton_to_points_cuda);
    ops_spc.def("interpolate_trilinear_cuda", &interpolate_trilinear_cuda);
    ops_spc.def("coords_to_trilinear_cuda", &coords_to_trilinear_cuda);
    //ops_spc.def("coord_to_trilinear_jacobian_cuda", &coord_to_trilinear_jacobian_cuda);
    ops_spc.def("points_to_corners_cuda", &points_to_corners_cuda);
#endif  // WITH_CUDA
    ops_spc.def("points_to_octree", &points_to_octree);
    ops_spc.def("morton_to_octree", &morton_to_octree);
    ops_spc.def("scan_octrees_cuda", &scan_octrees_cuda);
    ops_spc.def("generate_points_cuda", &generate_points_cuda);
    ops_spc.def("Conv3d_forward", &Conv3d_forward);
    ops_spc.def("Conv3d_backward", &Conv3d_backward);
    ops_spc.def("ConvTranspose3d_forward", &ConvTranspose3d_forward);
    ops_spc.def("ConvTranspose3d_backward", &ConvTranspose3d_backward);
    ops_spc.def("to_dense_forward", &to_dense_forward);
    ops_spc.def("to_dense_backward", &to_dense_backward);
  py::module metrics = m.def_submodule("metrics");
  metrics.def("sided_distance_forward_cuda", &sided_distance_forward_cuda);
  metrics.def("sided_distance_backward_cuda", &sided_distance_backward_cuda);
  metrics.def("unbatched_triangle_distance_forward_cuda",
              &unbatched_triangle_distance_forward_cuda);
  metrics.def("unbatched_triangle_distance_backward_cuda",
              &unbatched_triangle_distance_backward_cuda);
  py::module render = m.def_submodule("render");
  py::module render_mesh = render.def_submodule("mesh");
  render_mesh.def("packed_rasterize_forward_cuda", &packed_rasterize_forward_cuda);
  render_mesh.def("rasterize_backward_cuda", &rasterize_backward_cuda);
  render_mesh.def("dibr_soft_mask_forward_cuda", &dibr_soft_mask_forward_cuda);
  render_mesh.def("dibr_soft_mask_backward_cuda", &dibr_soft_mask_backward_cuda);
  render_mesh.def("deftet_sparse_render_forward_cuda", &deftet_sparse_render_forward_cuda);
  render_mesh.def("deftet_sparse_render_backward_cuda", &deftet_sparse_render_backward_cuda);
  py::module render_spc = render.def_submodule("spc");
  render_spc.def("raytrace_cuda", &raytrace_cuda);
  render_spc.def("generate_primary_rays_cuda", &generate_primary_rays_cuda); // Deprecate soon
  render_spc.def("mark_pack_boundaries_cuda", &mark_pack_boundaries_cuda);
  render_spc.def("generate_shadow_rays_cuda", &generate_shadow_rays_cuda); // Deprecate soon
  render_spc.def("inclusive_sum_cuda", &inclusive_sum_cuda);
  render_spc.def("diff_cuda", &diff_cuda);
  render_spc.def("sum_reduce_cuda", &sum_reduce_cuda);
  render_spc.def("cumsum_cuda", &cumsum_cuda);
  render_spc.def("cumprod_cuda", &cumprod_cuda);
  py::module render_sg = render.def_submodule("sg");
  render_sg.def("unbatched_reduced_sg_inner_product_forward_cuda",
		&unbatched_reduced_sg_inner_product_forward_cuda);
  render_sg.def("unbatched_reduced_sg_inner_product_backward_cuda",
		&unbatched_reduced_sg_inner_product_backward_cuda);
}

}  // namespace kaolin
