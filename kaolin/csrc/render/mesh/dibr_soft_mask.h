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

#ifndef KAOLIN_RENDER_MESH_DIBR_SOFT_MASK_H_
#define KAOLIN_RENDER_MESH_DIBR_SOFT_MASK_H_

#include <ATen/ATen.h>

namespace kaolin {

std::vector<at::Tensor> dibr_soft_mask_forward_cuda(
    const at::Tensor face_vertices_image, // x,y coordinates of face_vertices in camera plan.
    const at::Tensor face_large_bboxes,   // enlarged bounding boxes of faces in camera plan.
    const at::Tensor selected_face_idx,   // indices within each mesh of selected faces.
    const float sigmainv,                 // smoothness term for soft mask, the higher the shaper. Default: 7000
    const int knum,                       // Max number of faces influencing a pixel.
    const float multiplier);              // coordinates multiplier used to improve numerical precision.

at::Tensor dibr_soft_mask_backward_cuda(
    const at::Tensor grad_soft_mask,
    const at::Tensor soft_mask,
    const at::Tensor selected_face_idx,
    const at::Tensor close_face_prob,
    const at::Tensor close_face_idx,
    const at::Tensor close_face_dist_type,
    const at::Tensor face_vertices_image,
    const float sigmainv,
    const float multiplier);

}  // namespace kaolin

#endif // KAOLIN_RENDER_MESH_DIBR_SOFT_MASK_H_
