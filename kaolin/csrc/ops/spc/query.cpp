// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
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
#include "../../spc_math.h"

namespace kaolin {

#ifdef WITH_CUDA

void query_cuda_impl(
    at::Tensor octree,
    at::Tensor prefix_sum,
    at::Tensor query_points,
    at::Tensor pidx,
    uint target_level);

#endif // WITH_CUDA

at::Tensor query_cuda(   
    at::Tensor octree,
    at::Tensor prefix_sum,
    at::Tensor query_points,
    uint target_level) {
#ifdef WITH_CUDA
  at::TensorArg octree_arg{octree, "octree", 1};
  at::TensorArg prefix_sum_arg{prefix_sum, "prefix_sum", 2};
  at::TensorArg query_points_arg{query_points, "query_points", 3};
  at::checkAllSameGPU(__func__, {octree_arg, prefix_sum_arg, query_points_arg});
  at::checkAllContiguous(__func__,  {octree_arg, prefix_sum_arg, query_points_arg});
  at::checkScalarType(__func__, octree_arg, at::kByte);
  at::checkScalarType(__func__, prefix_sum_arg, at::kInt);
  at::checkScalarType(__func__, query_points_arg, at::kShort);

  int num_query = query_points.size(0);
  at::Tensor pidx = at::zeros({ num_query }, octree.options().dtype(at::kInt));
  query_cuda_impl(octree, prefix_sum, query_points, pidx, target_level);
  return pidx;
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA

}

} // namespace kaolin

