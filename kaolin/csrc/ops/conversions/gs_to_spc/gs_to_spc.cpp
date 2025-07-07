// Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
  const float iso,
  const float tol,
  const uint32_t target_level);

at::Tensor integrate_gs_cuda_impl(
  const at::Tensor& points,
  const at::Tensor& gaus_id,
  const at::Tensor& means3D,
  const at::Tensor& cov3Dinv,
  const at::Tensor& opacities,
  const uint32_t    level,
  const uint32_t    step);

#endif

std::vector<at::Tensor>
gs_to_spc_cuda(
  const at::Tensor& means3D,
  const at::Tensor& scales,
  const at::Tensor& rotations,
  const float iso,
  const float tol,
  const uint32_t level) {
#ifdef WITH_CUDA
  at::TensorArg means3D_arg{means3D, "means3D", 1};
  at::TensorArg scales_arg{scales, "scales", 2};
  at::TensorArg rotations_arg{rotations, "rotations", 3};

  at::checkAllSameGPU(__func__, {
      means3D_arg,
      scales_arg,
      rotations_arg});
  at::checkAllContiguous(__func__, {
      means3D_arg,
      scales_arg,
      rotations_arg});
  at::checkScalarType(__func__, means3D_arg, at::kFloat);
  at::checkAllSameType(__func__, {
      means3D_arg,
      scales_arg,
      rotations_arg});

  const int num_gaussians = means3D.size(0);
  at::checkSize(__func__, means3D_arg, {num_gaussians, 3});
  at::checkSize(__func__, scales_arg, {num_gaussians, 3});
  at::checkSize(__func__, rotations_arg, {num_gaussians, 4});

  return gs_to_spc_cuda_impl(means3D, scales, rotations, iso, tol, level);

#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif
}


at::Tensor integrate_gs_cuda(
  const at::Tensor& points,
  const at::Tensor& gaus_id,
  const at::Tensor& means3D,
  const at::Tensor& cov3Dinv,
  const at::Tensor& opacities,
  const uint32_t    level,
  const uint32_t    step) {

#ifdef WITH_CUDA
  at::TensorArg points_arg{points, "points", 0};
  at::TensorArg gaus_id_arg{gaus_id, "gaus_id", 1};
  at::TensorArg means3D_arg{means3D, "means3D", 2};
  at::TensorArg cov3Dinv_arg{cov3Dinv, "cov3Dinv", 3};
  at::TensorArg opacities_arg{opacities, "opacities", 4};

  at::checkAllSameGPU(__func__, {
      points_arg,
      gaus_id_arg,
      means3D_arg,
      cov3Dinv_arg,
      opacities_arg});
  at::checkAllContiguous(__func__, {
      points_arg,
      gaus_id_arg,
      means3D_arg,
      cov3Dinv_arg,
      opacities_arg});

  at::checkScalarType(__func__, points_arg, at::kShort);
  at::checkScalarType(__func__, gaus_id_arg, at::kLong);
  at::checkScalarType(__func__, means3D_arg, at::kFloat);
  at::checkAllSameType(__func__, {
      means3D_arg,
      cov3Dinv_arg,
      opacities_arg});

  const int num_intersection = points.size(0);
  const int num_gaussians = means3D.size(0);
  at::checkSize(__func__, points_arg, {num_intersection, 3});
  at::checkSize(__func__, gaus_id_arg, {num_intersection});
  at::checkSize(__func__, means3D_arg, {num_gaussians, 3});
  at::checkSize(__func__, cov3Dinv_arg, {num_gaussians, 6});
  at::checkNumel(__func__, opacities_arg, num_gaussians);

  return integrate_gs_cuda_impl(points, gaus_id, means3D, cov3Dinv, opacities, level, step);
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif
}



}  // namespace kaolin
