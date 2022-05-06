// Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
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

namespace kaolin {

#ifdef WITH_CUDA

void unbatched_reduced_sg_inner_product_forward_cuda_impl(
    const at::Tensor intensity,
    const at::Tensor direction,
    const at::Tensor sharpness,
    const at::Tensor other_intensity,
    const at::Tensor other_direction,
    const at::Tensor other_sharpness,
    at::Tensor output);

void unbatched_reduced_sg_inner_product_backward_cuda_impl(
    at::Tensor grad_out,
    at::Tensor intensity,
    at::Tensor direction,
    at::Tensor sharpness,
    at::Tensor other_intensity,
    at::Tensor other_direction,
    at::Tensor other_sharpness,
    at::Tensor grad_intensity,
    at::Tensor grad_direction,
    at::Tensor grad_sharpness,
    at::Tensor grad_other_intensity,
    at::Tensor grad_other_direction,
    at::Tensor grad_other_sharpness);

#endif

at::Tensor unbatched_reduced_sg_inner_product_forward_cuda(
    at::Tensor intensity,
    at::Tensor direction,
    at::Tensor sharpness,
    at::Tensor other_intensity,
    at::Tensor other_direction,
    at::Tensor other_sharpness) {

  at::TensorArg intensity_arg{
      intensity, "intensity", 1};
  at::TensorArg direction_arg{
      direction, "direction", 2};
  at::TensorArg sharpness_arg{
      sharpness, "sharpness", 3};
  at::TensorArg other_intensity_arg{
      other_intensity, "other_intensity", 4};
  at::TensorArg other_direction_arg{
      other_direction, "other_direction", 5};
  at::TensorArg other_sharpness_arg{
      other_sharpness, "other_sharpness", 6};

  at::checkAllSameGPU(__func__, {
      intensity_arg,
      direction_arg,
      sharpness_arg,
      other_intensity_arg,
      other_direction_arg,
      other_sharpness_arg
  });
  at::checkAllContiguous(__func__, {
      intensity_arg,
      direction_arg,
      sharpness_arg,
      other_intensity_arg,
      other_direction_arg,
      other_sharpness_arg
  });
  at::checkAllSameType(__func__, {
      intensity_arg,
      direction_arg,
      sharpness_arg,
      other_intensity_arg,
      other_direction_arg,
      other_sharpness_arg
  });

  const int num_sg = intensity.size(0);
  const int num_other = other_intensity.size(0);

  at::checkSize(__func__, intensity_arg, {num_sg, 3});
  at::checkSize(__func__, direction_arg, {num_sg, 3});
  at::checkSize(__func__, sharpness_arg, {num_sg});
  at::checkSize(__func__, other_intensity_arg, {num_other, 3});
  at::checkSize(__func__, other_direction_arg, {num_other, 3});
  at::checkSize(__func__, other_sharpness_arg, {num_other});

  at::Tensor output = at::zeros_like(intensity);

#ifdef WITH_CUDA
  if (num_sg > 0) {
    unbatched_reduced_sg_inner_product_forward_cuda_impl(
        intensity,
        direction,
        sharpness,
        other_intensity,
        other_direction,
        other_sharpness,
        output
    );
  }
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif
  return output;
}

std::vector<at::Tensor> unbatched_reduced_sg_inner_product_backward_cuda(
    at::Tensor grad_out,
    at::Tensor intensity,
    at::Tensor direction,
    at::Tensor sharpness,
    at::Tensor other_intensity,
    at::Tensor other_direction,
    at::Tensor other_sharpness) {

  at::TensorArg grad_out_arg{
      grad_out, "grad_out", 1};
  at::TensorArg intensity_arg{
      intensity, "intensity", 2};
  at::TensorArg direction_arg{
      direction, "direction", 3};
  at::TensorArg sharpness_arg{
      sharpness, "sharpness", 4};
  at::TensorArg other_intensity_arg{
      other_intensity, "other_intensity", 5};
  at::TensorArg other_direction_arg{
      other_direction, "other_direction", 6};
  at::TensorArg other_sharpness_arg{
      other_sharpness, "other_sharpness", 7};

  at::checkAllSameGPU(__func__, {
      grad_out_arg,
      intensity_arg,
      direction_arg,
      sharpness_arg,
      other_intensity_arg,
      other_direction_arg,
      other_sharpness_arg
  });
  at::checkAllContiguous(__func__, {
      grad_out_arg,
      intensity_arg,
      direction_arg,
      sharpness_arg,
      other_intensity_arg,
      other_direction_arg,
      other_sharpness_arg
  });
  at::checkAllSameType(__func__, {
      grad_out_arg,
      intensity_arg,
      direction_arg,
      sharpness_arg,
      other_intensity_arg,
      other_direction_arg,
      other_sharpness_arg
  });

  const int num_sg = intensity.size(0);
  const int num_other = other_intensity.size(0);

  at::checkSize(__func__, grad_out_arg, {num_sg, 3});
  at::checkSize(__func__, intensity_arg, {num_sg, 3});
  at::checkSize(__func__, direction_arg, {num_sg, 3});
  at::checkSize(__func__, sharpness_arg, {num_sg});
  at::checkSize(__func__, other_intensity_arg, {num_other, 3});
  at::checkSize(__func__, other_direction_arg, {num_other, 3});
  at::checkSize(__func__, other_sharpness_arg, {num_other});

  auto grad_intensity = at::zeros_like(intensity);
  auto grad_direction = at::zeros_like(direction);
  auto grad_sharpness = at::zeros_like(sharpness);
  auto grad_other_intensity = at::zeros_like(other_intensity);
  auto grad_other_direction = at::zeros_like(other_direction);
  auto grad_other_sharpness = at::zeros_like(other_sharpness);

#ifdef WITH_CUDA
  if (num_sg > 0) {
    unbatched_reduced_sg_inner_product_backward_cuda_impl(
        grad_out,
        intensity,
        direction,
        sharpness,
        other_intensity,
        other_direction,
        other_sharpness,
        grad_intensity,
        grad_direction,
        grad_sharpness,
        grad_other_intensity,
        grad_other_direction,
        grad_other_sharpness
    );
  } 
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
  return {grad_intensity, grad_direction, grad_sharpness,
          grad_other_intensity, grad_other_direction, grad_other_sharpness};
}

}  // namespace kaolin
