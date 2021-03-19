// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
  
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

#include "../check.h"

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_DIM4(x, b, h, w, d) TORCH_CHECK((x.dim() == 4) && (x.size(0) == b) && (x.size(1) == h) && (x.size(2) == w) && (x.size(3) == d), #x " must be same im size")
#define CHECK_DIM3(x, b, f, d) TORCH_CHECK((x.dim() == 3) && (x.size(0) == b) && (x.size(1) == f) && (x.size(2) == d), #x " must be same point size")
#define CHECK_DIM2(x, b, d) TORCH_CHECK((x.dim() == 2) && (x.size(0) == b) && (x.size(1) == d), #x " wrong size")
#define CHECK_DIM1(x, b) TORCH_CHECK((x.dim() == 1) && (x.size(0) == b), #x " wrong size")

namespace kaolin {

#ifdef WITH_CUDA

void packed_rasterize_forward_cuda_kernel_launcher(
    at::Tensor face_vertices_z,
    at::Tensor face_vertices_image,
    at::Tensor face_bboxes,
    at::Tensor face_features,
    at::Tensor first_idx_face_per_mesh,
    at::Tensor selected_face_idx,
    at::Tensor output_weights,
    at::Tensor interpolated_features,
    float multiplier);

void rasterize_backward_cuda_kernel_launcher(
    at::Tensor grad_interpolated_features,
    at::Tensor grad_improb_bxhxwx1,
    at::Tensor interpolated_features,
    at::Tensor improb_bxhxwx1,
    at::Tensor selected_face_idx,
    at::Tensor output_weights,
    at::Tensor probface_bxhxwxk,
    at::Tensor probcase_bxhxwxk,
    at::Tensor probdis_bxhxwxk,
    at::Tensor face_vertices_image,
    at::Tensor face_features,
    at::Tensor grad_face_vertices_image,
    at::Tensor grad_face_features,
    at::Tensor grad_points2dprob_bxfx6,
    int multiplier,
    int sigmainv);

void generate_soft_mask_cuda_kernel_launcher(
    at::Tensor face_vertices_image,
    at::Tensor face_bboxes,
    at::Tensor selected_face_idx,
    at::Tensor probface_bxhxwxk,
    at::Tensor probcase_bxhxwxk,
    at::Tensor probdis_bxhxwxk,
    at::Tensor improb_bxhxwx1,
    float multiplier,
    float sigmainv);

#endif  // WITH_CUDA

void packed_rasterize_forward_cuda(
    at::Tensor face_vertices_z,         // depth of face_vertices in camera ref.
    at::Tensor face_vertices_image,     // x,y coordinates of face_vertices in camera plan.
    at::Tensor face_bboxes,             // bounding boxes of faces in camera plan.
    at::Tensor face_features,           // features to interpolate.
    at::Tensor first_idx_face_per_mesh, // indices of the first face of each mesh (+ last_idx + 1).
    at::Tensor selected_face_idx,       // output: indices within each mesh of selected faces.
    at::Tensor output_weights,          // output: the weights used for interpolation.
    at::Tensor interpolated_features,   // output: the interpolated face features.
    float multiplier) {                 // coordinates multiplier used to improve numerical precision.

  CHECK_INPUT(face_vertices_z);
  CHECK_INPUT(face_vertices_image);
  CHECK_INPUT(face_bboxes);
  CHECK_INPUT(face_features);
  // Usually for packed representation first_idx is on CPU
  // but here it's on GPU because or the way we computed it on DIBRasterization
  CHECK_INPUT(first_idx_face_per_mesh);

  CHECK_INPUT(selected_face_idx);
  CHECK_INPUT(output_weights);

  CHECK_INPUT(interpolated_features);

  int num_faces = face_vertices_z.size(0);
  int batch_size = interpolated_features.size(0);
  int height = interpolated_features.size(1);
  int width = interpolated_features.size(2);
  int num_features = interpolated_features.size(3);

  CHECK_DIM2(face_vertices_z, num_faces, 3);
  CHECK_DIM3(face_vertices_image, num_faces, 3, 2);
  CHECK_DIM2(face_bboxes, num_faces, 4);
  CHECK_DIM3(face_features, num_faces, 3, num_features);
  CHECK_DIM1(first_idx_face_per_mesh, batch_size + 1)

  CHECK_DIM3(selected_face_idx, batch_size, height, width);
  CHECK_DIM4(output_weights, batch_size, height, width, 3);
  CHECK_DIM4(interpolated_features, batch_size, height, width, num_features);

#if WITH_CUDA
  packed_rasterize_forward_cuda_kernel_launcher(
      face_vertices_z, face_vertices_image, face_bboxes,
      face_features, first_idx_face_per_mesh, selected_face_idx,
      output_weights, interpolated_features, multiplier);
  return;
#else
  AT_ERROR("packed_rasterize_forward not built with CUDA");
#endif  // WITH_CUDA
}

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
    float sigmainv) {               // smoothness term for soft mask, the higher the shaper, range is (1/3e-4, 1/3e-5). default: 7000

  CHECK_INPUT(face_vertices_image);
  CHECK_INPUT(face_bboxes);
  CHECK_INPUT(selected_face_idx);

  CHECK_INPUT(probface_bxhxwxk);
  CHECK_INPUT(probcase_bxhxwxk);
  CHECK_INPUT(probdis_bxhxwxk);

  CHECK_INPUT(improb_bxhxwx1);

  int batch_size = face_vertices_image.size(0);
  int num_faces = face_vertices_image.size(1);
  int height = improb_bxhxwx1.size(1);
  int width = improb_bxhxwx1.size(2);

  int knum = probface_bxhxwxk.size(3);

  CHECK_DIM4(face_vertices_image, batch_size, num_faces, 3, 2);
  CHECK_DIM3(face_bboxes, batch_size, num_faces, 4);

  CHECK_DIM3(selected_face_idx, batch_size, height, width);

  CHECK_DIM4(probface_bxhxwxk, batch_size, height, width, knum);
  CHECK_DIM4(probcase_bxhxwxk, batch_size, height, width, knum);
  CHECK_DIM4(probdis_bxhxwxk, batch_size, height, width, knum);

  CHECK_DIM3(improb_bxhxwx1, batch_size, height, width);

#if WITH_CUDA
  generate_soft_mask_cuda_kernel_launcher(
      face_vertices_image, face_bboxes, selected_face_idx,
      probface_bxhxwxk, probcase_bxhxwxk, probdis_bxhxwxk,
      improb_bxhxwx1, multiplier, sigmainv);
  return;
#else
  AT_ERROR("generate_soft_mask not built with CUDA");
#endif
}

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
    int sigmainv) {                        // smoothness term for soft mask, the higher the shaper, range is (1/3e-4, 1/3e-5). default: 7000

  CHECK_INPUT(grad_interpolated_features);
  CHECK_INPUT(grad_improb_bxhxwx1);
  CHECK_INPUT(interpolated_features);
  CHECK_INPUT(improb_bxhxwx1);
  CHECK_INPUT(selected_face_idx);
  CHECK_INPUT(output_weights);

  CHECK_INPUT(probface_bxhxwxk);
  CHECK_INPUT(probcase_bxhxwxk);
  CHECK_INPUT(probdis_bxhxwxk);

  CHECK_INPUT(face_vertices_image);
  CHECK_INPUT(face_features);
  CHECK_INPUT(grad_face_vertices_image);
  CHECK_INPUT(grad_face_features);
  CHECK_INPUT(grad_points2dprob_bxfx6);

  int bnum = grad_interpolated_features.size(0);
  int height = grad_interpolated_features.size(1);
  int width = grad_interpolated_features.size(2);
  int dnum = grad_interpolated_features.size(3);
  int fnum = grad_face_vertices_image.size(1);
  int knum = probface_bxhxwxk.size(3);

  CHECK_DIM4(grad_interpolated_features, bnum, height, width, dnum);
  CHECK_DIM3(grad_improb_bxhxwx1, bnum, height, width);

  CHECK_DIM4(interpolated_features, bnum, height, width, dnum);
  CHECK_DIM3(improb_bxhxwx1, bnum, height, width);

  CHECK_DIM3(selected_face_idx, bnum, height, width);
  CHECK_DIM4(output_weights, bnum, height, width, 3);

  CHECK_DIM4(probface_bxhxwxk, bnum, height, width, knum);
  CHECK_DIM4(probface_bxhxwxk, bnum, height, width, knum);
  CHECK_DIM4(probdis_bxhxwxk, bnum, height, width, knum);

  CHECK_DIM4(face_vertices_image, bnum, fnum, 3, 2);
  CHECK_DIM4(face_features, bnum, fnum, 3, dnum);
  CHECK_DIM4(grad_face_vertices_image, bnum, fnum, 3, 2);
  CHECK_DIM4(grad_face_features, bnum, fnum, 3, dnum);
  CHECK_DIM4(grad_points2dprob_bxfx6, bnum, fnum, 3, 2);

#if WITH_CUDA
  rasterize_backward_cuda_kernel_launcher(grad_interpolated_features, grad_improb_bxhxwx1,
      interpolated_features, improb_bxhxwx1, selected_face_idx, output_weights,
      probface_bxhxwxk, probcase_bxhxwxk, probdis_bxhxwxk, face_vertices_image,
      face_features, grad_face_vertices_image, grad_face_features,
      grad_points2dprob_bxfx6, multiplier, sigmainv);
  return;
#else
  AT_ERROR("rasterize_backward not built with CUDA");
#endif  // WITH_CUDA
}

}  // namespace kaolin

#undef CHECK_DIM1
#undef CHECK_DIM2
#undef CHECK_DIM3
#undef CHECK_DIM4
