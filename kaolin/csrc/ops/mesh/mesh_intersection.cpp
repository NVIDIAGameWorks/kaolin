// Copyright (c) 2019,20-22 NVIDIA CORPORATION & AFFILIATES.
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

#include <ATen/ATen.h>

#include "../../check.h"

namespace kaolin {

#ifdef WITH_CUDA
void unbatched_mesh_intersection_cuda_impl(
    const at::Tensor points,
    const at::Tensor verts_1,
    const at::Tensor verts_2,
    const at::Tensor verts_3,
    at::Tensor result);
#endif


at::Tensor unbatched_mesh_intersection_cuda(
    const at::Tensor points, 
    const at::Tensor verts_1,
    const at::Tensor verts_2,
    const at::Tensor verts_3) {

  at::TensorArg points_arg{points, "points", 1};
  at::TensorArg verts_1_arg{verts_1, "verts_1", 2};
  at::TensorArg verts_2_arg{verts_2, "verts_2", 3};
  at::TensorArg verts_3_arg{verts_3, "verts_3", 4};

  const int num_points = points.size(0);
  const int num_vertices = verts_1.size(0);

  at::checkAllSameGPU(__func__, {
      points_arg, verts_1_arg, verts_2_arg, verts_3_arg});
  at::checkAllContiguous(__func__, {
      points_arg, verts_1_arg, verts_2_arg, verts_3_arg});
  at::checkAllSameType(__func__, {
      points_arg, verts_1_arg, verts_2_arg, verts_3_arg});

  at::checkSize(__func__, points_arg, {num_points, 3});
  at::checkSize(__func__, verts_1_arg, {num_vertices, 3});
  at::checkSize(__func__, verts_2_arg, {num_vertices, 3});
  at::checkSize(__func__, verts_3_arg, {num_vertices, 3});

  at::Tensor out = at::zeros({num_points}, points.options());

#ifdef WITH_CUDA    
  unbatched_mesh_intersection_cuda_impl(points, verts_1, verts_2, verts_3, out);
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif
  return out;
}

}  // namespace kaolin
