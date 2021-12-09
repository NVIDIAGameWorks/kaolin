// Copyright (c) 2019,20-21-22 NVIDIA CORPORATION & AFFILIATES.
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

void dibr_soft_mask_forward_cuda_impl(
    const at::Tensor face_vertices_image,
    const at::Tensor face_large_bboxes,
    const at::Tensor selected_face_idx,
    at::Tensor close_face_prob,
    at::Tensor close_face_idx,
    at::Tensor close_face_dist_type,
    at::Tensor soft_mask,
    const float sigmainv,
    const float multiplier);

void dibr_soft_mask_backward_cuda_impl(
    const at::Tensor grad_soft_mask,
    const at::Tensor soft_mask,
    const at::Tensor selected_face_idx,
    const at::Tensor close_face_prob,
    const at::Tensor close_face_idx,
    const at::Tensor close_face_dist_type,
    const at::Tensor face_vertices_image,
    at::Tensor grad_face_vertices_image,
    const float sigmainv,
    const float multiplier);

#endif  // WITH_CUDA

std::vector<at::Tensor> dibr_soft_mask_forward_cuda(
    const at::Tensor face_vertices_image, // x,y coordinates of face_vertices in camera plan.
    const at::Tensor face_large_bboxes,   // enlarged bounding boxes of faces in camera plan.
    const at::Tensor selected_face_idx,   // indices within each mesh of selected faces.
    const float sigmainv,                 // smoothness term for soft mask, the higher the shaper. default: 7000
    const int knum,                       // Max number of faces influencing a pixel.
    const float multiplier) {             // coordinates multiplier used to improve numerical precision.

  at::TensorArg face_vertices_image_arg{
      face_vertices_image, "face_vertices_image", 1};
  at::TensorArg face_large_bboxes_arg{
      face_large_bboxes, "face_bboxes", 2};
  at::TensorArg selected_face_idx_arg{
      selected_face_idx, "selected_face_idx", 3};

  at::checkAllSameGPU(__func__, {
      face_vertices_image_arg,
      face_large_bboxes_arg,
      selected_face_idx_arg});
  at::checkAllContiguous(__func__, {
      face_vertices_image_arg,
      face_large_bboxes_arg,
      selected_face_idx_arg});

  const int batch_size = face_vertices_image.size(0);
  const int num_faces = face_vertices_image.size(1);
  const int height = selected_face_idx.size(1);
  const int width = selected_face_idx.size(2);

  at::checkSize(__func__, face_vertices_image_arg,
                {batch_size, num_faces, 3, 2});
  at::checkSize(__func__, face_large_bboxes_arg,
                {batch_size, num_faces, 4});
  at::checkSize(__func__, selected_face_idx_arg,
                {batch_size, height, width});

  auto options = face_vertices_image.options();
  // faces nearby per pixel
  at::Tensor close_face_idx = at::full({batch_size, height, width, knum},
                                       -1, options.dtype(at::kLong));
  // defined by: exp(-dist * sigmainv)
  // where "dist" is the smallest distance between the face and the pixel
  at::Tensor close_face_prob = at::zeros({batch_size, height, width, knum},
                                         options);
  // type of distance from the face to the pixel
  // 1, 2, 3 means minimal distance from pixel to edges
  // 4, 5, 6 means minimal distance from pixel to vertices
  at::Tensor close_face_dist_type = at::zeros({batch_size, height, width, knum},
                                              options.dtype(at::kByte));
  at::Tensor soft_mask = at::zeros({batch_size, height, width}, options);

#if WITH_CUDA
  dibr_soft_mask_forward_cuda_impl(
      face_vertices_image, face_large_bboxes, selected_face_idx,
      close_face_prob, close_face_idx, close_face_dist_type,
      soft_mask, sigmainv, multiplier);
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif
  return {soft_mask, close_face_prob, close_face_idx, close_face_dist_type};
}

at::Tensor dibr_soft_mask_backward_cuda(
    const at::Tensor grad_soft_mask,
    const at::Tensor soft_mask,
    const at::Tensor selected_face_idx,
    const at::Tensor close_face_prob,
    const at::Tensor close_face_idx,
    const at::Tensor close_face_dist_type,
    const at::Tensor face_vertices_image,
    const float sigmainv,
    const float multiplier) {
  at::TensorArg grad_soft_mask_arg{
      grad_soft_mask, "grad_soft_mask", 1};
  at::TensorArg soft_mask_arg{
      soft_mask, "soft_mask", 2};
  at::TensorArg selected_face_idx_arg{
      selected_face_idx, "selected_face_idx", 3};
  at::TensorArg close_face_prob_arg{
      close_face_prob, "close_face_prob", 4};
  at::TensorArg close_face_idx_arg{
      close_face_idx, "close_face_idx", 5};
  at::TensorArg close_face_dist_type_arg{
      close_face_dist_type, "close_face_dist_type", 6};
  at::TensorArg face_vertices_image_arg{
      face_vertices_image, "face_vertices_image", 7};

  at::checkAllSameGPU(__func__, {
      grad_soft_mask_arg,
      soft_mask_arg,
      close_face_idx_arg,
      close_face_dist_type_arg,
      close_face_prob_arg,
      face_vertices_image_arg});

  at::checkAllContiguous(__func__, {
      grad_soft_mask_arg,
      soft_mask_arg,
      close_face_prob_arg,
      close_face_idx_arg,
      close_face_dist_type_arg,
      face_vertices_image_arg});

  const int batch_size = face_vertices_image.size(0);
  const int num_faces = face_vertices_image.size(1);
  const int height = selected_face_idx.size(1);
  const int width = selected_face_idx.size(2);
  const int knum = close_face_idx.size(-1);
  
  at::checkSize(__func__, grad_soft_mask_arg,
                {batch_size, height, width});
  at::checkSize(__func__, soft_mask_arg,
                {batch_size, height, width});
  at::checkSize(__func__, selected_face_idx_arg,
                {batch_size, height, width});
  at::checkSize(__func__, close_face_prob_arg,
                {batch_size, height, width, knum});
  at::checkSize(__func__, close_face_idx_arg,
                {batch_size, height, width, knum});
  at::checkSize(__func__, close_face_dist_type_arg,
                {batch_size, height, width, knum});
  at::checkSize(__func__, face_vertices_image_arg,
                {batch_size, num_faces, 3, 2});

  at::Tensor grad_face_vertices_image = at::zeros_like(face_vertices_image);

#if WITH_CUDA
  dibr_soft_mask_backward_cuda_impl(
    grad_soft_mask, soft_mask, selected_face_idx, close_face_prob,
    close_face_idx, close_face_dist_type, face_vertices_image,
    grad_face_vertices_image, sigmainv, multiplier);
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif
  return grad_face_vertices_image;
}

}  // namespace kaolin
