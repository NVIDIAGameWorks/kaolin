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

void packed_rasterize_forward_cuda_impl(
    const at::Tensor face_vertices_z,
    const at::Tensor face_vertices_image,
    const at::Tensor face_bboxes,
    const at::Tensor face_features,
    const at::Tensor first_idx_face_per_mesh,
    at::Tensor selected_face_idx,
    at::Tensor output_weights,
    at::Tensor interpolated_features,
    const float multiplier,
    const float eps);

void rasterize_backward_cuda_impl(
    const at::Tensor grad_interpolated_features,
    const at::Tensor interpolated_features,
    const at::Tensor selected_face_idx,
    const at::Tensor output_weights,
    const at::Tensor face_vertices_image,
    const at::Tensor face_features,
    at::Tensor grad_face_vertices_image,
    at::Tensor grad_face_features,
    const float eps);

#endif  // WITH_CUDA

std::vector<at::Tensor> packed_rasterize_forward_cuda(
    const int height,
    const int width,
    const at::Tensor face_vertices_z,         // depth of face_vertices in camera ref.
    const at::Tensor face_vertices_image,     // x,y coordinates of face_vertices in camera plan.
    const at::Tensor face_bboxes,             // bounding boxes of faces in camera plan.
    const at::Tensor face_features,           // features to interpolate.
    const at::Tensor first_idx_face_per_mesh, // indices of the first face of each mesh (+ last_idx + 1).
    const float multiplier,                   // coordinates multiplier used to improve numerical precision.
    const float eps) {                        // epsilon value to use for barycentric weights normalization

  at::TensorArg face_vertices_z_arg{
      face_vertices_z, "face_vertices_z", 3};
  at::TensorArg face_vertices_image_arg{
      face_vertices_image, "face_vertices_image", 4};
  at::TensorArg face_bboxes_arg{face_bboxes, "face_bboxes", 5};
  at::TensorArg face_features_arg{face_features, "face_features", 6};
  at::TensorArg first_idx_face_per_mesh_arg{
      first_idx_face_per_mesh, "first_idx_face_per_mesh", 7};
  // Usually for packed representation first_idx is on CPU
  // but here it's on GPU because or the way we computed it on DIBRasterization
  at::checkAllSameGPU(__func__, {
      face_vertices_z_arg, face_vertices_image_arg, face_bboxes_arg,
      face_features_arg, first_idx_face_per_mesh_arg});
  at::checkAllContiguous(__func__, {
      face_vertices_z_arg, face_vertices_image_arg, face_bboxes_arg,
      face_features_arg, first_idx_face_per_mesh_arg});

  const int num_faces = face_vertices_z.size(0);
  const int batch_size = first_idx_face_per_mesh.size(0) - 1;
  const int feat_dim = face_features.size(2);

  at::checkSize(__func__, face_vertices_z_arg, {num_faces, 3});
  at::checkSize(__func__, face_vertices_image_arg, {num_faces, 3, 2});
  at::checkSize(__func__, face_bboxes_arg, {num_faces, 4});
  at::checkSize(__func__, face_features_arg, {num_faces, 3, feat_dim});
  at::checkSize(__func__, first_idx_face_per_mesh_arg, {batch_size + 1});

  auto options = face_vertices_z.options();
  at::Tensor selected_face_idx = at::full({batch_size, height, width},
                                          -1, options.dtype(at::kLong));
  at::Tensor output_weights = at::zeros({batch_size, height, width, 3},
                                        options);
  at::Tensor interpolated_features = at::zeros({batch_size, height, width, feat_dim},
                                               options);

#if WITH_CUDA
  packed_rasterize_forward_cuda_impl(
      face_vertices_z, face_vertices_image, face_bboxes,
      face_features, first_idx_face_per_mesh, selected_face_idx,
      output_weights, interpolated_features, multiplier, eps);
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
  return {interpolated_features, selected_face_idx, output_weights};
}

std::vector<at::Tensor> rasterize_backward_cuda(
    const at::Tensor grad_interpolated_features, // gradients from interpolated_features.
    const at::Tensor interpolated_features,      // the interpolated face features.
    const at::Tensor selected_face_idx,          // indices within each mesh of selected faces.
    const at::Tensor output_weights,             // the weights used for interpolation.
    const at::Tensor face_vertices_image,        // x,y coordinates of face_vertices in camera plan.
    const at::Tensor face_features,              // features to interpolate.
    const float eps) {                           // epsilon value to use for barycentric weights normalization
  at::TensorArg grad_interpolated_features_arg{
      grad_interpolated_features, "grad_interpolated_features", 1};
  at::TensorArg interpolated_features_arg{
      interpolated_features, "interpolated_features", 2};
  at::TensorArg selected_face_idx_arg{
      selected_face_idx, "selected_face_idx", 3};
  at::TensorArg output_weights_arg{
      output_weights, "output_weights", 4};
  at::TensorArg face_vertices_image_arg{
      face_vertices_image, "face_vertices_image", 5};
  at::TensorArg face_features_arg{
      face_features, "face_features", 6};

  at::checkAllSameGPU(__func__, {
      grad_interpolated_features_arg, interpolated_features_arg,
      selected_face_idx_arg, output_weights_arg, face_vertices_image_arg,
      face_features_arg});

  at::checkAllContiguous(__func__, {
      grad_interpolated_features_arg, interpolated_features_arg,
      selected_face_idx_arg, output_weights_arg, face_vertices_image_arg,
      face_features_arg});

  const int batch_size = grad_interpolated_features.size(0);
  const int height = grad_interpolated_features.size(1);
  const int width = grad_interpolated_features.size(2);
  const int feat_dim = grad_interpolated_features.size(3);
  const int num_faces = face_vertices_image.size(1);

  at::checkSize(__func__, grad_interpolated_features_arg,
                {batch_size, height, width, feat_dim});
  at::checkSize(__func__, interpolated_features_arg,
                {batch_size, height, width, feat_dim});
  at::checkSize(__func__, selected_face_idx_arg,
                {batch_size, height, width});
  at::checkSize(__func__, output_weights_arg,
                {batch_size, height, width, 3});
  at::checkSize(__func__, face_vertices_image_arg,
                {batch_size, num_faces, 3, 2});
  at::checkSize(__func__, face_features_arg,
                {batch_size, num_faces, 3, feat_dim});

  at::Tensor grad_face_vertices_image = at::zeros_like(face_vertices_image);
  at::Tensor grad_face_features = at::zeros_like(face_features);

#if WITH_CUDA
  rasterize_backward_cuda_impl(
      grad_interpolated_features, interpolated_features, selected_face_idx,
      output_weights, face_vertices_image, face_features,
      grad_face_vertices_image, grad_face_features, eps);
#else
  KAOLIN_NO_CUDA_ERROR(__func__);
#endif  // WITH_CUDA
  return {grad_face_vertices_image, grad_face_features};
}

}  // namespace kaolin
