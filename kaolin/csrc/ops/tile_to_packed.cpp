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

#include <ATen/ATen.h>

#include "../check.h"

namespace kaolin {

#ifdef WITH_CUDA
void tile_to_packed_cuda_kernel_launcher(
    at::Tensor values_tensor,
    at::Tensor shape_per_tensor,
    at::Tensor output);
#endif

at::Tensor tile_to_packed_cuda(
    at::Tensor values_tensor,
    at::Tensor shape_per_tensor,
    int total_numel) {
  CHECK_CONTIGUOUS(values_tensor);
  CHECK_CONTIGUOUS(shape_per_tensor);
  CHECK_CUDA(values_tensor);
  CHECK_CPU(shape_per_tensor);
  auto output = at::empty({total_numel, 1}, values_tensor.options());
#ifdef WITH_CUDA
  tile_to_packed_cuda_kernel_launcher(
    values_tensor,
    shape_per_tensor,
    output);
#else
  AT_ERROR("tile_to_packed is not built with CUDA");
#endif
  return output;
}

at::Tensor tile_to_packed_out_cuda(
    at::Tensor values_tensor,
    at::Tensor shape_per_tensor,
    at::Tensor output) {
  CHECK_CONTIGUOUS(values_tensor);
  CHECK_CONTIGUOUS(shape_per_tensor);
  CHECK_CUDA(values_tensor);
  CHECK_CPU(shape_per_tensor);
#ifdef WITH_CUDA
  tile_to_packed_cuda_kernel_launcher(
    values_tensor,
    shape_per_tensor,
    output);
#else
  AT_ERROR("tile_to_packed is not built with CUDA");
#endif
  return output;
}

}  // namespace kaolin
