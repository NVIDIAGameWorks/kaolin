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

#define CHECK_VALUES(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_SHAPE_PER_TENSOR(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

void tile_to_packed_cuda_forward(
    at::Tensor values_tensor,
    at::Tensor shape_per_tensor,
    at::Tensor output);

torch::Tensor tile_to_packed_forward(
    torch::Tensor values_tensor,
    torch::Tensor shape_per_tensor,
    int total_numel) {
  CHECK_VALUES(values_tensor);
  CHECK_SHAPE_PER_TENSOR(shape_per_tensor);
  auto output = at::empty({total_numel, 1}, values_tensor.options());
  tile_to_packed_cuda_forward(
    values_tensor,
    shape_per_tensor,
    output);
  return output;
}

torch::Tensor tile_to_packed_forward_out(
    torch::Tensor values_tensor,
    torch::Tensor shape_per_tensor,
    torch::Tensor output) {
  CHECK_VALUES(values_tensor);
  CHECK_SHAPE_PER_TENSOR(shape_per_tensor);
  tile_to_packed_cuda_forward(
    values_tensor,
    shape_per_tensor,
    output);
  return output;
}
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
#ifdef WITH_CUDA
  m.def("forward", &tile_to_packed_forward, "tile_to_packed forward (CUDA)");
  m.def("forward_out", &tile_to_packed_forward_out,
        "tile_to_packed forward with provided output (CUDA)");
#endif
}
