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

#ifdef WITH_CUDA
#include "../../spc_math.h"
#endif

namespace kaolin {

#ifdef WITH_CUDA

void to_dense_forward_cuda_kernel_launch(
    at::Tensor points,
    int level,
    at::Tensor pyramid,
    at::Tensor features,
    at::Tensor outputs);

void to_dense_backward_cuda_kernel_launch(
    at::Tensor points,
    int level,
    at::Tensor pyramid,
    at::Tensor features,
    at::Tensor grad_outputs,
    at::Tensor grad_features);

#endif  // WITH_CUDA

using namespace at::indexing;

at::Tensor to_dense_forward(
    at::Tensor points,
    int level,
    at::Tensor pyramid,
    at::Tensor features) {
#ifdef WITH_CUDA
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(pyramid);
  CHECK_CONTIGUOUS(features);
  CHECK_CUDA(points);
  CHECK_CPU(pyramid);
  CHECK_CUDA(features);

  int feature_size = features.size(1);
  int batch_size = pyramid.size(0);
  int grid_size = (0x1 << level);

  int num_features = features.size(0);
  uint psize = pyramid.index({ Slice(None), 0, level }).sum().item<int>();

  TORCH_CHECK(num_features == psize);

  at::Tensor outputs = at::zeros({batch_size, feature_size, grid_size, grid_size, grid_size},
                                 points.options().dtype(at::kFloat));

  to_dense_forward_cuda_kernel_launch(points, level, pyramid, features, outputs);
  return outputs;
#else
  AT_ERROR("to_dense_forward not built with CUDA");
#endif  // WITH_CUDA

}


at::Tensor to_dense_backward(
    at::Tensor points,
    int level,
    at::Tensor pyramid,
    at::Tensor features,
    at::Tensor grad_outputs) {
#ifdef WITH_CUDA
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(pyramid);
  CHECK_CONTIGUOUS(features);
  CHECK_CONTIGUOUS(grad_outputs)
  CHECK_CUDA(points);
  CHECK_CPU(pyramid);
  CHECK_CUDA(features);
  CHECK_CUDA(grad_outputs);

  at::Tensor grad_features = at::zeros_like(features);

  to_dense_backward_cuda_kernel_launch(points, level, pyramid, features,
                                       grad_outputs, grad_features);
  return grad_features;
#else
  AT_ERROR("to_dense_backward not built with CUDA");
#endif  // WITH_CUDA

}

}  // namespace kaolin
