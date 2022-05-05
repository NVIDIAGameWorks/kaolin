// Copyright (c) 2021,22 NVIDIA CORPORATION & AFFILIATES.
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

#include <limits>

#include <ATen/ATen.h>

namespace kaolin {

#ifdef WITH_CUDA

void deftet_sparse_render_forward_cuda_impl(
    const at::Tensor face_vertices_z,
    const at::Tensor face_vertices_image,
    const at::Tensor face_bboxes,
    const at::Tensor pixel_coords,
    const at::Tensor pixel_depth_ranges,
    at::Tensor selected_face_idx,
    at::Tensor pixel_depths,
    at::Tensor w0_arr,
    at::Tensor w1_arr,
    const float eps);

void deftet_sparse_render_backward_cuda_impl(
    const at::Tensor grad_interpolated_features,
    const at::Tensor face_idx,
    const at::Tensor weights,
    const at::Tensor face_vertices_image,
    const at::Tensor face_features,
    at::Tensor grad_face_vertices_image,
    at::Tensor grad_face_features,
    const float eps);

#endif // WITH_CUDA

std::vector<at::Tensor> deftet_sparse_render_forward_cuda(
    const at::Tensor face_vertices_z,
    const at::Tensor face_vertices_image,
    const at::Tensor face_bboxes,
    const at::Tensor pixel_coords,
    const at::Tensor pixel_depth_ranges,
    const int knum,
    const float eps) {

  at::TensorArg face_vertices_z_arg{face_vertices_z, "face_vertices_z", 1};
  at::TensorArg face_vertices_image_arg{
      face_vertices_image, "face_vertices_image", 2};
  at::TensorArg face_bboxes_arg{face_bboxes, "face_bboxes", 3};
  at::TensorArg pixel_coords_arg{pixel_coords, "pixel_coords", 4};
  at::TensorArg pixel_depth_ranges_arg{
      pixel_depth_ranges, "pixel_depth_ranges", 5};
  at::checkAllSameGPU(__func__, {
      face_vertices_z_arg, face_vertices_image_arg, face_bboxes_arg,
      pixel_coords_arg, pixel_depth_ranges_arg});

  at::checkAllContiguous(__func__, {
      face_vertices_z_arg, face_vertices_image_arg, face_bboxes_arg,
      pixel_coords_arg, pixel_depth_ranges_arg});

  int batch_size = face_vertices_z.size(0);
  int num_faces = face_vertices_z.size(1);
  int num_points = pixel_coords.size(1);

  at::checkSize(__func__, face_vertices_z_arg,
                {batch_size, num_faces, 3});
  at::checkSize(__func__, face_vertices_image_arg,
                {batch_size, num_faces, 3, 2});
  at::checkSize(__func__, face_bboxes_arg,
                {batch_size, num_faces, 4});
  at::checkSize(__func__, pixel_coords_arg,
                {batch_size, num_points, 2});
  at::checkSize(__func__, pixel_depth_ranges_arg,
                {batch_size, num_points, 2});

  auto options = face_vertices_z.options();
  auto selected_face_idx = at::full({batch_size, num_points, knum}, -1,
                                    options.dtype(at::kLong));
  auto pixel_depths = at::full({batch_size, num_points, knum},
                               -std::numeric_limits<float>::infinity(),
			       options);
  auto w0_arr = at::zeros({batch_size, num_points, knum}, options);
  auto w1_arr = at::zeros({batch_size, num_points, knum}, options);

#if WITH_CUDA
  deftet_sparse_render_forward_cuda_impl(
      face_vertices_z, face_vertices_image, face_bboxes,
      pixel_coords, pixel_depth_ranges, selected_face_idx,
      pixel_depths, w0_arr, w1_arr, eps);
#else
  AT_ERROR("In deftet_sparse_render_forward_cuda: Kaolin built without CUDA,"
	   " cannot run with GPU tensors");
#endif // WITH_CUDA

  return {selected_face_idx, pixel_depths, w0_arr, w1_arr};
}

std::vector<at::Tensor> deftet_sparse_render_backward_cuda(
    const at::Tensor grad_interpolated_features,
    const at::Tensor face_idx,
    const at::Tensor weights,
    const at::Tensor face_vertices_image,
    const at::Tensor face_features,
    const float eps) {

  at::TensorArg grad_interpolated_features_arg{
    grad_interpolated_features, "grad_interpolated_features", 1};
  at::TensorArg face_idx_arg{face_idx, "face_idx", 2};
  at::TensorArg weights_arg{weights, "weights", 3};
  at::TensorArg face_vertices_image_arg{
    face_vertices_image, "face_vertices_image", 4};
  at::TensorArg face_features_arg{face_features, "face_features", 5};
  at::checkAllSameGPU(__func__, {
      grad_interpolated_features_arg, face_idx_arg, weights_arg,
      face_vertices_image_arg, face_features_arg});
  at::checkAllContiguous(__func__, {
      grad_interpolated_features_arg, face_idx_arg, weights_arg,
      face_vertices_image_arg, face_features_arg});

  int batch_size = grad_interpolated_features.size(0);
  int num_pixels = grad_interpolated_features.size(1);
  int knum = grad_interpolated_features.size(2);
  int feat_dim = grad_interpolated_features.size(3);
  int num_faces = face_vertices_image.size(1);

  at::checkSize(__func__, grad_interpolated_features_arg,
                {batch_size, num_pixels, knum, feat_dim});
  at::checkSize(__func__, face_idx_arg,
                {batch_size, num_pixels, knum});
  at::checkSize(__func__, weights_arg,
                {batch_size, num_pixels, knum, 3});
  at::checkSize(__func__, face_vertices_image_arg,
                {batch_size, num_faces, 3, 2});
  at::checkSize(__func__, face_features_arg,
                {batch_size, num_faces, 3, feat_dim});
  auto grad_face_vertices_image = at::zeros_like(face_vertices_image);
  auto grad_face_features = at::zeros_like(face_features);

#if WITH_CUDA
  deftet_sparse_render_backward_cuda_impl(
      grad_interpolated_features, face_idx, weights, face_vertices_image,
      face_features, grad_face_vertices_image, grad_face_features, eps);
#else
  AT_ERROR("In deftet_sparse_render_backward_cuda: Kaolin built without CUDA,"
	   " cannot run with GPU tensors");
#endif // WITH_CUDA

  return {grad_face_vertices_image, grad_face_features};
}

}

