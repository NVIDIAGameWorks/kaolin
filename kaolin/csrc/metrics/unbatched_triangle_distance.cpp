// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/ATen.h>

#include "../check.h"

namespace kaolin {

#ifdef WITH_CUDA

void unbatched_triangle_distance_forward_cuda_impl(
    at::Tensor points,
    at::Tensor face_vertices,
    at::Tensor dist,
    at::Tensor face_idx,
    at::Tensor dist_type);

void unbatched_triangle_distance_backward_cuda_impl(
    at::Tensor grad_dist,
    at::Tensor points,
    at::Tensor face_vertices,
    at::Tensor face_idx,
    at::Tensor dist_type,
    at::Tensor grad_points,
    at::Tensor grad_face_vertices);

#endif  // WITH_CUDA


void unbatched_triangle_distance_forward_cuda(
    at::Tensor points,
    at::Tensor face_vertices,
    at::Tensor dist,
    at::Tensor face_idx,
    at::Tensor dist_type) {
  CHECK_CUDA(points);
  CHECK_CUDA(face_vertices);
  CHECK_CUDA(dist);
  CHECK_CUDA(face_idx);
  CHECK_CUDA(dist_type);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(face_vertices);
  CHECK_CONTIGUOUS(dist);
  CHECK_CONTIGUOUS(face_idx);
  CHECK_CONTIGUOUS(dist_type);
  const int num_points = points.size(0);
  const int num_faces = face_vertices.size(0);
  CHECK_SIZES(points, num_points, 3);
  CHECK_SIZES(face_vertices, num_faces, 3, 3);
  CHECK_SIZES(dist, num_points);
  CHECK_SIZES(face_idx, num_points);
  CHECK_SIZES(dist_type, num_points);
#if WITH_CUDA
  unbatched_triangle_distance_forward_cuda_impl(
      points, face_vertices, dist, face_idx, dist_type);
#else
  AT_ERROR("unbatched_triangle_distance not built with CUDA");
#endif
}

void unbatched_triangle_distance_backward_cuda(
    at::Tensor grad_dist,
    at::Tensor points,
    at::Tensor face_vertices,
    at::Tensor face_idx,
    at::Tensor dist_type,
    at::Tensor grad_points,
    at::Tensor grad_face_vertices) {
  CHECK_CUDA(grad_dist);
  CHECK_CUDA(points);
  CHECK_CUDA(face_vertices);
  CHECK_CUDA(face_idx);
  CHECK_CUDA(dist_type);
  CHECK_CUDA(grad_points);
  CHECK_CUDA(grad_face_vertices);
  CHECK_CONTIGUOUS(grad_dist);
  CHECK_CONTIGUOUS(points);
  CHECK_CONTIGUOUS(face_vertices);
  CHECK_CONTIGUOUS(face_idx);
  CHECK_CONTIGUOUS(dist_type);
  CHECK_CONTIGUOUS(grad_points);
  CHECK_CONTIGUOUS(grad_face_vertices);

  const int num_points = points.size(0);
  const int num_faces = face_vertices.size(0);
  CHECK_SIZES(grad_dist, num_points);
  CHECK_SIZES(points, num_points, 3);
  CHECK_SIZES(face_vertices, num_faces, 3, 3);
  CHECK_SIZES(face_idx, num_points);
  CHECK_SIZES(dist_type, num_points);
  CHECK_SIZES(grad_points, num_points, 3);
  CHECK_SIZES(grad_face_vertices, num_faces, 3, 3);

#if WITH_CUDA
  unbatched_triangle_distance_backward_cuda_impl(
      grad_dist, points, face_vertices, face_idx, dist_type,
      grad_points, grad_face_vertices);
#else
  AT_ERROR("unbatched_triangle_distance_backward not built with CUDA");
#endif
}

}  // namespace kaolin
