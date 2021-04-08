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

#include <ATen/ATen.h>
#include "../check.h"

namespace kaolin {

#ifdef WITH_CUDA
at::Tensor packed_simple_sum_cuda_kernel_launcher(
    at::Tensor packed_tensor,
    at::Tensor shape_per_tensor,
    at::Tensor output);
#endif

at::ScalarType accumulate_type(const at::ScalarType input_type) {
  switch (input_type) {
    case at::ScalarType::Half:
      return at::ScalarType::Float;
    case at::ScalarType::Bool:
    case at::ScalarType::Byte:
    case at::ScalarType::Char:
    case at::ScalarType::Short:
    case at::ScalarType::Int:
      return at::ScalarType::Long;
    default:
      return input_type;
  }
};

at::Tensor packed_simple_sum_cuda(
    at::Tensor packed_tensor,
    at::Tensor shape_per_tensor) {
  CHECK_CONTIGUOUS(packed_tensor);
  CHECK_CONTIGUOUS(shape_per_tensor);
  CHECK_CUDA(packed_tensor);
  CHECK_CPU(shape_per_tensor);
  // By default certains types shouldn't be accumulated with the same type
  // (example fp16 range is too small)
  auto output_dtype = accumulate_type(packed_tensor.scalar_type());
  auto output = at::empty({shape_per_tensor.size(0)},
			  packed_tensor.options().dtype(output_dtype));
#ifdef WITH_CUDA
  packed_simple_sum_cuda_kernel_launcher(
    packed_tensor,
    shape_per_tensor,
    output);
#else
  AT_ERROR("packed_simple_sum not built with CUDA");
#endif  // WITH_CUDA
  return output;
}

at::Tensor packed_simple_sum_out_cuda(
    at::Tensor packed_tensor,
    at::Tensor shape_per_tensor,
    at::Tensor output) {
  CHECK_CONTIGUOUS(packed_tensor);
  CHECK_CONTIGUOUS(shape_per_tensor);
  CHECK_CUDA(packed_tensor);
  CHECK_CPU(shape_per_tensor);
#ifdef WITH_CUDA
  packed_simple_sum_cuda_kernel_launcher(
    packed_tensor,
    shape_per_tensor,
    output);
#else
  AT_ERROR("packed_simple_sum not built with CUDA");
#endif
  return output;
}

}  // namespace kaolin
