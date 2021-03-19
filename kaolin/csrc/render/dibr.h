// Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//    http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef KAOLIN_RENDER_DIBR_H_
#define KAOLIN_RENDER_DIBR_H_

#include <ATen/ATen.h>

namespace kaolin {

void packed_rasterize_forward_cuda(
    at::Tensor face_vertices_z,         // depth of face_vertices in camera ref.
    at::Tensor face_vertices_image,     // x,y coordinates of face_vertices in camera plan.
    at::Tensor face_bboxes,             // bounding boxes of faces in camera plan.
    at::Tensor face_features,           // features to interpolate.
    at::Tensor first_idx_face_per_mesh, // indices of the first face of each mesh (+ last_idx + 1).
    at::Tensor selected_face_idx,       // output: indices within each mesh of selected faces.
    at::Tensor output_weights,          // output: the weights used for interpolation.
    at::Tensor interpolated_features,   // output: the interpolated face features.
    float multiplier);                  // coordinates multiplier used to improve numerical precision.

void generate_soft_mask_cuda(
    at::Tensor face_vertices_image, // x,y coordinates of face_vertices in camera plan.
    at::Tensor face_bboxes,         // enlarged bounding boxes of faces in camera plan.
    at::Tensor selected_face_idx,   // indices within each mesh of selected faces.
    at::Tensor probface_bxhxwxk,    // k nearby faces that influences it, 0 for void, per pixel
    at::Tensor probcase_bxhxwxk,    // type of distance from the face to the pixel
                                    // (0, 1, 2 means minimal distance from pixel to edges
                                    // 3, 4, 5 means minimal distance from pixel to vertices), per pixel
    at::Tensor probdis_bxhxwxk,     // distance probability, per pixel
    at::Tensor improb_bxhxwx1,      // soft mask, per pixel
    float multiplier,               // coordinates multiplier used to improve numerical precision.
    float sigmainv);                // smoothness term for soft mask, the higher the shaper, range is (1/3e-4, 1/3e-5). default: 7000

void rasterize_backward_cuda(
    at::Tensor grad_interpolated_features, // gradients from interpolated_features.
    at::Tensor grad_improb_bxhxwx1,        // gradients from improb_bxhxwx1.
    at::Tensor interpolated_features,      // the interpolated face features.
    at::Tensor improb_bxhxwx1,             // soft mask, per pixel.
    at::Tensor selected_face_idx,          // indices within each mesh of selected faces.
    at::Tensor output_weights,             // the weights used for interpolation.
    at::Tensor probface_bxhxwxk,           // k nearby faces that influences it, 0 for void, per pixel.
    at::Tensor probcase_bxhxwxk,           // type of distance from the face to the pixel
                                           // (0, 1, 2 means minimal distance from pixel to edges
                                           // 3, 4, 5 means minimal distance from pixel to vertices), per pixel.
    at::Tensor probdis_bxhxwxk,            // distance probability, per pixel.
    at::Tensor face_vertices_image,        // x,y coordinates of face_vertices in camera plan.
    at::Tensor face_features,              // features to interpolate.
    at::Tensor grad_face_vertices_image,   // output: gradients to face_vertices_image from the interpolated features
    at::Tensor grad_face_features,         // outputs: gradients to face_features
    at::Tensor grad_points2dprob_bxfx6,    // outputs: gradients to face_vertices_image from the soft mask
    int multiplier,                        // coordinates multiplier used to improve numerical precision.
    int sigmainv);                         // smoothness term for soft mask, the higher the shaper, range is (1/3e-4, 1/3e-5). default: 7000

}  // namespace kaolin

#endif // KAOLIN_RENDER_DIBR_H_
