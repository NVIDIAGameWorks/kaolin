// Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.

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
  py::module metrics = m.def_submodule("metrics");
  metrics.def("sided_distance_forward_cuda", &sided_distance_forward_cuda);
  metrics.def("sided_distance_backward_cuda", &sided_distance_backward_cuda);
  metrics.def("unbatched_triangle_distance_forward_cuda", &unbatched_triangle_distance_forward_cuda);
  metrics.def("unbatched_triangle_distance_backward_cuda", &unbatched_triangle_distance_backward_cuda);
  py::module render = m.def_submodule("render");
  render.def("packed_rasterize_forward_cuda", &packed_rasterize_forward_cuda);
  render.def("generate_soft_mask_cuda", &generate_soft_mask_cuda);
  render.def("rasterize_backward_cuda", &rasterize_backward_cuda);
#endif
}

}  // namespace kaolin
