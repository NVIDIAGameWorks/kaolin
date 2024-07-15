// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

void query_cuda_impl_empty(
    at::Tensor octree,
    at::Tensor empty,
    at::Tensor prefix_sum,
    at::Tensor query_coords,
    at::Tensor pidx,
    uint32_t target_level);

#endif // WITH_CUDA

// empty version
at::Tensor query_cuda_empty(
    at::Tensor octree,
    at::Tensor empty,
    at::Tensor prefix_sum,
    at::Tensor query_coords,
    uint32_t target_level) {
#ifdef WITH_CUDA
  at::TensorArg octree_arg{octree, "octree", 1};
  at::TensorArg empty_arg{empty, "empty", 2};
  at::TensorArg prefix_sum_arg{prefix_sum, "prefix_sum", 3};
  at::TensorArg query_coords_arg{query_coords, "query_coords", 4};
  at::checkAllSameGPU(__func__, {octree_arg, empty_arg, prefix_sum_arg, query_coords_arg});
  at::checkAllContiguous(__func__,  {octree_arg, empty_arg, prefix_sum_arg, query_coords_arg});
  at::checkScalarType(__func__, octree_arg, at::kByte);
  at::checkScalarType(__func__, empty_arg, at::kByte);
  at::checkScalarType(__func__, prefix_sum_arg, at::kInt);
  at::checkScalarTypes(__func__, query_coords_arg, {at::kHalf, at::kFloat, at::kDouble});

  int num_query = query_coords.size(0);
  at::Tensor pidx = at::zeros({ num_query }, octree.options().dtype(at::kInt));
  query_cuda_impl_empty(octree, empty, prefix_sum, query_coords, pidx, target_level);
  return pidx;
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA

}

} // namespace kaolin

