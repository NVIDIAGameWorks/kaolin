// Copyright (c) 2019,21-22 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef KAOLIN_RENDER_MESH_RASTERIZATION_H_
#define KAOLIN_RENDER_MESH_RASTERIZATION_H_

#include <ATen/ATen.h>

namespace kaolin {

std::vector<at::Tensor> packed_rasterize_forward_cuda(
    const int height,
    const int width,
    const at::Tensor face_vertices_z,         // depth of face_vertices in camera ref.
    const at::Tensor face_vertices_image,     // x,y coordinates of face_vertices in camera plan.
    const at::Tensor face_bboxes,             // bounding boxes of faces in camera plan.
    const at::Tensor face_features,           // features to interpolate.
    const at::Tensor first_idx_face_per_mesh, // indices of the first face of each mesh (+ last_idx + 1).
    const float multiplier,                   // coordinates multiplier used to improve numerical precision.
    const float eps);                         // epsilon value to use for barycentric weights normalization

std::vector<at::Tensor> rasterize_backward_cuda(
    const at::Tensor grad_interpolated_features, // gradients from interpolated_features.
    const at::Tensor interpolated_features,      // the interpolated face features.
    const at::Tensor selected_face_idx,          // indices within each mesh of selected faces.
    const at::Tensor output_weights,             // the weights used for interpolation.
    const at::Tensor face_vertices_image,        // x,y coordinates of face_vertices in camera plan.
    const at::Tensor face_features,              // features to interpolate.
    const float eps);                            // epsilon value to use for barycentric weights normalization.

}  // namespace kaolin

#endif // KAOLIN_RENDER_MESH_RASTERIZATION_H_
