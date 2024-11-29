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

#include "../../../check.h"

namespace kaolin {

#ifdef WITH_CUDA

std::vector<at::Tensor> gs_to_spc_cuda_impl(
  const at::Tensor& means3D,
  const at::Tensor& scales,
  const at::Tensor& rotations,
  const at::Tensor& opacities,
  const float iso,
  const float tol,
  const uint32_t target_level);

#endif

std::vector<at::Tensor>
gs_to_spc_cuda(
  const at::Tensor& means3D,
  const at::Tensor& scales,
  const at::Tensor& rotations,
  const at::Tensor& opacities,
  const float iso,
  const float tol,
  const uint32_t level) {
#ifdef WITH_CUDA

  at::TensorArg means3D_arg{means3D, "means3D", 1};
  at::TensorArg scales_arg{scales, "scales", 2};
  at::TensorArg rotations_arg{rotations, "rotations", 3};
  at::TensorArg opacities_arg{opacities, "opacities", 4};

  at::checkAllSameGPU(__func__, {
      means3D_arg,
      scales_arg,
      rotations_arg,
      opacities_arg});
  at::checkAllContiguous(__func__, {
      means3D_arg,
      scales_arg,
      rotations_arg,
      opacities_arg});

  const int num_gaussians = means3D.size(0);
  at::checkSize(__func__, means3D_arg, {num_gaussians, 3});
  at::checkSize(__func__, scales_arg, {num_gaussians, 3});
  at::checkSize(__func__, rotations_arg, {num_gaussians, 4});
  at::checkSize(__func__, opacities_arg, {num_gaussians});
  at::checkScalarType(__func__, means3D_arg, at::kFloat);
  at::checkScalarType(__func__, scales_arg, at::kFloat);
  at::checkScalarType(__func__, rotations_arg, at::kFloat);
  at::checkScalarType(__func__, opacities_arg, at::kFloat);

  return gs_to_spc_cuda_impl(means3D, scales, rotations, opacities, iso, tol, level);

#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif

}

}  // namespace kaolin
