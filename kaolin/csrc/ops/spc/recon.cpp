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

#include <stdlib.h>
#include <stdio.h>
#include <ATen/ATen.h>

#include "../../check.h"
#ifdef WITH_CUDA
#include "../../spc_math.h"
#endif

#include <iostream>

namespace kaolin {
using namespace std;
using namespace at::indexing;

#ifdef WITH_CUDA

// uint64_t GetTempSize(void* d_temp_storage, uint32_t* d_M0, uint32_t* d_M1, uint32_t max_total_points);

at::Tensor inclusive_sum_cuda_x(at::Tensor Inputs);
at::Tensor  build_mip2d_cuda(at::Tensor image, at::Tensor In, int mip_levels, float maxdepth, bool true_depth);

std::vector<at::Tensor>  subdivide_cuda(at::Tensor Points, at::Tensor Insum);
std::vector<at::Tensor>  compactify_cuda(at::Tensor Points, at::Tensor Insum);

#endif // WITH_CUDA

at::Tensor inclusive_sum(at::Tensor inputs)
{
#ifdef WITH_CUDA
  return inclusive_sum_cuda_x(inputs);
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}

at::Tensor build_mip2d(at::Tensor image, at::Tensor In, int mip_levels, float max_depth, bool true_depth)
{
#ifdef WITH_CUDA
  return build_mip2d_cuda(image, In, mip_levels, max_depth, true_depth);
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}

std::vector<at::Tensor> subdivide(at::Tensor Points, at::Tensor Insum)
{
#ifdef WITH_CUDA
  return subdivide_cuda(Points, Insum);
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}


std::vector<at::Tensor> compactify(at::Tensor Points, at::Tensor Insum)
{
#ifdef WITH_CUDA
  return compactify_cuda(Points, Insum);
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
}

}  // namespace kaolin
