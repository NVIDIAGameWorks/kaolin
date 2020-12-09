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

#include <torch/extension.h>

#ifdef WITH_CUDA
#include "../../check.h"

#define CHECK_PACKED(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_SHAPE_PER_TENSOR(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

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

at::Tensor packed_simple_sum_cuda_forward(
    at::Tensor packed_tensor,
    at::Tensor shape_per_tensor,
    at::Tensor output);

torch::Tensor packed_simple_sum_forward(
    torch::Tensor packed_tensor,
    torch::Tensor shape_per_tensor) {
  CHECK_PACKED(packed_tensor);
  CHECK_SHAPE_PER_TENSOR(shape_per_tensor);
  // By default certains types shouldn't be accumulated with the same type
  // (example fp16 range is too small)
  auto output_dtype = accumulate_type(packed_tensor.scalar_type());
  auto output = at::empty({shape_per_tensor.size(0)},
                          packed_tensor.options().dtype(output_dtype));
  packed_simple_sum_cuda_forward(
    packed_tensor,
    shape_per_tensor,
    output);
  return output;
}

torch::Tensor packed_simple_sum_forward_out(
    torch::Tensor packed_tensor,
    torch::Tensor shape_per_tensor,
    torch::Tensor output) {
  CHECK_PACKED(packed_tensor);
  CHECK_SHAPE_PER_TENSOR(shape_per_tensor);
  packed_simple_sum_cuda_forward(
    packed_tensor,
    shape_per_tensor,
    output);
  return output;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef WITH_CUDA
  m.def("simple_forward", &packed_simple_sum_forward, "Packed simple sum forward (CUDA)");
  m.def("simple_forward_out", &packed_simple_sum_forward_out,
        "Packed simple sum forward with provided output tensor (CUDA)");
#endif
}
