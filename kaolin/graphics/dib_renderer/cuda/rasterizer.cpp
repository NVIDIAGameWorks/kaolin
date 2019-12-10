/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <torch/extension.h>

// C++ interface
#define CHECK_CUDA(x)                                                          \
  AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

#define CHECK_DIM3(x, b, h, w, d)                                              \
  AT_ASSERTM((x.size(0) == b) && (x.size(1) == h) && (x.size(2) == w) &&       \
                 (x.size(3) == d),                                             \
             #x " must be same im size")
#define CHECK_DIM2(x, b, f, d)                                                 \
  AT_ASSERTM((x.size(0) == b) && (x.size(1) == f) && (x.size(2) == d),         \
             #x " must be same point size")

////////////////////////////////////////////////////////////
// CUDA forward declarations
void dr_cuda_forward_batch(
    at::Tensor points3d_bxfx9, at::Tensor points2d_bxfx6,
    at::Tensor pointsdirect_bxfx1, at::Tensor pointsbbox_bxfx4,
    at::Tensor pointsbbox2_bxfx4, at::Tensor pointsdep_bxfx1,
    at::Tensor colors_bxfx3d, at::Tensor imidx_bxhxwx1,
    at::Tensor imdep_bxhxwx1, at::Tensor imwei_bxhxwx3,
    at::Tensor probface_bxhxwxk, at::Tensor probcase_bxhxwxk,
    at::Tensor probdis_bxhxwxk, at::Tensor probdep_bxhxwxk,
    at::Tensor probacc_bxhxwxk, at::Tensor im_bxhxwxd,
    at::Tensor improb_bxhxwx1, int multiplier, int sigmainv);

void dr_forward_batch(at::Tensor points3d_bxfx9, at::Tensor points2d_bxfx6,
                      at::Tensor pointsdirect_bxfx1,
                      at::Tensor pointsbbox_bxfx4, at::Tensor pointsbbox2_bxfx4,
                      at::Tensor pointsdep_bxfx1, at::Tensor colors_bxfx3d,
                      at::Tensor imidx_bxhxwx1, at::Tensor imdep_bxhxwx1,
                      at::Tensor imwei_bxhxwx3, at::Tensor probface_bxhxwxk,
                      at::Tensor probcase_bxhxwxk, at::Tensor probdis_bxhxwxk,
                      at::Tensor probdep_bxhxwxk, at::Tensor probacc_bxhxwxk,
                      at::Tensor im_bxhxwxd, at::Tensor improb_bxhxwx1,
                      int multiplier, int sigmainv) {

  CHECK_INPUT(points3d_bxfx9);
  CHECK_INPUT(points2d_bxfx6);
  CHECK_INPUT(pointsdirect_bxfx1);
  CHECK_INPUT(pointsbbox_bxfx4);
  CHECK_INPUT(pointsbbox2_bxfx4);
  CHECK_INPUT(pointsdep_bxfx1);
  CHECK_INPUT(colors_bxfx3d);

  CHECK_INPUT(imidx_bxhxwx1);
  CHECK_INPUT(imdep_bxhxwx1);
  CHECK_INPUT(imwei_bxhxwx3);

  CHECK_INPUT(probface_bxhxwxk);
  CHECK_INPUT(probcase_bxhxwxk);
  CHECK_INPUT(probdis_bxhxwxk);
  CHECK_INPUT(probdep_bxhxwxk);
  CHECK_INPUT(probacc_bxhxwxk);

  CHECK_INPUT(im_bxhxwxd);
  CHECK_INPUT(improb_bxhxwx1);

  int bnum = points3d_bxfx9.size(0);
  int fnum = points3d_bxfx9.size(1);
  int height = im_bxhxwxd.size(1);
  int width = im_bxhxwxd.size(2);
  int dnum = im_bxhxwxd.size(3);

  int knum = probface_bxhxwxk.size(3);

  CHECK_DIM2(points3d_bxfx9, bnum, fnum, 9);
  CHECK_DIM2(points2d_bxfx6, bnum, fnum, 6);
  CHECK_DIM2(pointsdirect_bxfx1, bnum, fnum, 1);
  CHECK_DIM2(pointsbbox_bxfx4, bnum, fnum, 4);
  CHECK_DIM2(pointsbbox2_bxfx4, bnum, fnum, 4);
  CHECK_DIM2(pointsdep_bxfx1, bnum, fnum, 1);
  CHECK_DIM2(colors_bxfx3d, bnum, fnum, 3 * dnum);

  CHECK_DIM3(imidx_bxhxwx1, bnum, height, width, 1);
  CHECK_DIM3(imdep_bxhxwx1, bnum, height, width, 1);
  CHECK_DIM3(imwei_bxhxwx3, bnum, height, width, 3);

  CHECK_DIM3(probface_bxhxwxk, bnum, height, width, knum);
  CHECK_DIM3(probcase_bxhxwxk, bnum, height, width, knum);
  CHECK_DIM3(probdis_bxhxwxk, bnum, height, width, knum);
  CHECK_DIM3(probdep_bxhxwxk, bnum, height, width, knum);
  CHECK_DIM3(probacc_bxhxwxk, bnum, height, width, knum);

  CHECK_DIM3(im_bxhxwxd, bnum, height, width, dnum);
  CHECK_DIM3(improb_bxhxwx1, bnum, height, width, 1);

  dr_cuda_forward_batch(points3d_bxfx9, points2d_bxfx6, pointsdirect_bxfx1,
                        pointsbbox_bxfx4, pointsbbox2_bxfx4, pointsdep_bxfx1,
                        colors_bxfx3d, imidx_bxhxwx1, imdep_bxhxwx1,
                        imwei_bxhxwx3, probface_bxhxwxk, probcase_bxhxwxk,
                        probdis_bxhxwxk, probdep_bxhxwxk, probacc_bxhxwxk,
                        im_bxhxwxd, improb_bxhxwx1, multiplier, sigmainv);

  return;
}

////////////////////////////////////////////////////////////
void dr_cuda_backward_batch(
    at::Tensor grad_image_bxhxwxd, at::Tensor grad_improb_bxhxwx1,
    at::Tensor image_bxhxwxd, at::Tensor improb_bxhxwx1,
    at::Tensor imidx_bxhxwx1, at::Tensor imwei_bxhxwx3,
    at::Tensor probface_bxhxwxk, at::Tensor probcase_bxhxwxk,
    at::Tensor probdis_bxhxwxk, at::Tensor probdep_bxhxwxk,
    at::Tensor probacc_bxhxwxk, at::Tensor points2d_bxfx6,
    at::Tensor colors_bxfx3d, at::Tensor grad_points2d_bxfx6,
    at::Tensor grad_colors_bxfx3d, at::Tensor grad_points2dprob_bxfx6,
    at::Tensor debug_im_bxhxwx3, int multiplier, int sigmainv);

void dr_backward_batch(at::Tensor grad_image_bxhxwxd,
                       at::Tensor grad_improb_bxhxwx1, at::Tensor image_bxhxwxd,
                       at::Tensor improb_bxhxwx1, at::Tensor imidx_bxhxwx1,
                       at::Tensor imwei_bxhxwx3, at::Tensor probface_bxhxwxk,
                       at::Tensor probcase_bxhxwxk, at::Tensor probdis_bxhxwxk,
                       at::Tensor probdep_bxhxwxk, at::Tensor probacc_bxhxwxk,
                       at::Tensor points2d_bxfx6, at::Tensor colors_bxfx3d,
                       at::Tensor grad_points2d_bxfx6,
                       at::Tensor grad_colors_bxfx3d,
                       at::Tensor grad_points2dprob_bxfx6,
                       at::Tensor debug_im_bxhxwx3, int multiplier,
                       int sigmainv) {

  CHECK_INPUT(grad_image_bxhxwxd);
  CHECK_INPUT(grad_improb_bxhxwx1);
  CHECK_INPUT(image_bxhxwxd);
  CHECK_INPUT(improb_bxhxwx1);
  CHECK_INPUT(imidx_bxhxwx1);
  CHECK_INPUT(imwei_bxhxwx3);

  CHECK_INPUT(probface_bxhxwxk);
  CHECK_INPUT(probcase_bxhxwxk);
  CHECK_INPUT(probdis_bxhxwxk);
  CHECK_INPUT(probdep_bxhxwxk);
  CHECK_INPUT(probacc_bxhxwxk);

  CHECK_INPUT(points2d_bxfx6);
  CHECK_INPUT(colors_bxfx3d);
  CHECK_INPUT(grad_points2d_bxfx6);
  CHECK_INPUT(grad_colors_bxfx3d);
  CHECK_INPUT(grad_points2dprob_bxfx6);
  CHECK_INPUT(debug_im_bxhxwx3);

  int bnum = grad_image_bxhxwxd.size(0);
  int height = grad_image_bxhxwxd.size(1);
  int width = grad_image_bxhxwxd.size(2);
  int dnum = grad_image_bxhxwxd.size(3);
  int fnum = grad_points2d_bxfx6.size(1);
  int knum = probface_bxhxwxk.size(3);

  CHECK_DIM3(grad_image_bxhxwxd, bnum, height, width, dnum);
  CHECK_DIM3(grad_improb_bxhxwx1, bnum, height, width, 1);

  CHECK_DIM3(image_bxhxwxd, bnum, height, width, dnum);
  CHECK_DIM3(improb_bxhxwx1, bnum, height, width, 1);

  CHECK_DIM3(imidx_bxhxwx1, bnum, height, width, 1);
  CHECK_DIM3(imwei_bxhxwx3, bnum, height, width, 3);

  CHECK_DIM3(probface_bxhxwxk, bnum, height, width, knum);
  CHECK_DIM3(probface_bxhxwxk, bnum, height, width, knum);
  CHECK_DIM3(probdis_bxhxwxk, bnum, height, width, knum);
  CHECK_DIM3(probdep_bxhxwxk, bnum, height, width, knum);
  CHECK_DIM3(probacc_bxhxwxk, bnum, height, width, knum);

  CHECK_DIM2(points2d_bxfx6, bnum, fnum, 6);
  CHECK_DIM2(colors_bxfx3d, bnum, fnum, 3 * dnum);
  CHECK_DIM2(grad_points2d_bxfx6, bnum, fnum, 6);
  CHECK_DIM2(grad_colors_bxfx3d, bnum, fnum, 3 * dnum);
  CHECK_DIM2(grad_points2dprob_bxfx6, bnum, fnum, 6);

  CHECK_DIM3(debug_im_bxhxwx3, bnum, height, width, 3);

  dr_cuda_backward_batch(
      grad_image_bxhxwxd, grad_improb_bxhxwx1, image_bxhxwxd, improb_bxhxwx1,
      imidx_bxhxwx1, imwei_bxhxwx3, probface_bxhxwxk, probcase_bxhxwxk,
      probdis_bxhxwxk, probdep_bxhxwxk, probacc_bxhxwxk, points2d_bxfx6,
      colors_bxfx3d, grad_points2d_bxfx6, grad_colors_bxfx3d,
      grad_points2dprob_bxfx6, debug_im_bxhxwx3, multiplier, sigmainv);

  return;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dr_forward_batch, "dr forward batch (CUDA)");
  m.def("backward", &dr_backward_batch, "dr backward batch (CUDA)");
}
