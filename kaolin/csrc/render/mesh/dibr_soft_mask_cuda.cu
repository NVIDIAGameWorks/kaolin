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
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <THC/THCAtomics.cuh>

#include "../../utils.h"

#define EPS 1e-7

namespace kaolin {

template<typename scalar_t>
__global__ void dibr_soft_mask_forward_cuda_kernel(
    const scalar_t* __restrict__ face_vertices_image,
    const scalar_t* __restrict__ face_bboxes,
    const int64_t* __restrict__ selected_face_idx,
    scalar_t* __restrict__ close_face_prob,
    int64_t* __restrict__ close_face_idx,
    uint8_t* __restrict__ close_face_dist_type,
    scalar_t* __restrict__ soft_mask,
    int batch_size,
    int height,
    int width,
    int num_faces,
    int knum,
    float sigmainv,
    float multiplier) {

  // bidx * height * width + heiidx * width + wididx
  int presentthread = blockIdx.x * blockDim.x + threadIdx.x;

  int wididx = presentthread % width;
  presentthread = (presentthread - wididx) / width;

  int heiidx = presentthread % height;
  int bidx = (presentthread - heiidx) / height;

  if (bidx >= batch_size || heiidx >= height || wididx >= width) {
    return;
  }

  // which pixel it belongs to
  const int totalidx1 = bidx * height * width + heiidx * width + wididx;
  const int totalidxk = totalidx1 * knum;

  // which face it belongs to?
  // face begins from 1
  // convert it into int, use round!
  int fidxint = selected_face_idx[totalidx1];

  // not covered by any faces
  // maybe we can search its neighbour
  if (fidxint >= 0) {
    soft_mask[totalidx1] = 1.0;
  }
  //  pixels not covered by any faces
  else {

    // pixel coordinate
    scalar_t x0 = multiplier / width * (2 * wididx + 1 - width);
    scalar_t y0 = multiplier / height * (height - 2 * heiidx - 1);

    int kid = 0;

    for (int fidxint = 0; fidxint < num_faces; fidxint++) {

      // which face it belongs to
      const int shift1 = bidx * num_faces + fidxint;
      const int shift4 = shift1 * 4;
      const int shift6 = shift1 * 6;

      ///////////////////////////////////////////////////////////////
      // will this pixel is influenced by this face?
      scalar_t xmin = face_bboxes[shift4 + 0];
      scalar_t ymin = face_bboxes[shift4 + 1];
      scalar_t xmax = face_bboxes[shift4 + 2];
      scalar_t ymax = face_bboxes[shift4 + 3];

      // not covered by this face!
      if (x0 < xmin || x0 >= xmax || y0 < ymin || y0 >= ymax) {
        continue;
      }

      //////////////////////////////////////////////////////////
      scalar_t pdis[6];

      // perdis
      for (int i = 0; i < 3; i++) {

        int pshift = shift6 + i * 2;
        scalar_t x1 = face_vertices_image[pshift + 0];
        scalar_t y1 = face_vertices_image[pshift + 1];

        int pshift2 = shift6 + ((i + 1) % 3) * 2;
        scalar_t x2 = face_vertices_image[pshift2 + 0];
        scalar_t y2 = face_vertices_image[pshift2 + 1];

        // ax + by + c = 0
        scalar_t A = y2 - y1;
        scalar_t B = x1 - x2;
        scalar_t C = x2 * y1 - x1 * y2;

        // dissquare = d^2 = (ax+by+c)^2 / (a^2+b^2)
        // up = ax + by + c
        // down = a^2 + b^2
        // dissquare = up^2 / down
        scalar_t up = A * x0 + B * y0 + C;
        scalar_t down = A * A + B * B;

        // is it a bad triangle?
        scalar_t x3 = B * B * x0 - A * B * y0 - A * C;
        scalar_t y3 = A * A * y0 - A * B * x0 - B * C;
        x3 = x3 / (down + EPS);
        y3 = y3 / (down + EPS);

        scalar_t direct = (x3 - x1) * (x3 - x2) + (y3 - y1) * (y3 - y2);

        if (direct > 0) {
          // bad triangle
          pdis[i] = 4 * multiplier * multiplier;
        } else {
          // perpendicular  distance
          pdis[i] = up * up / (down + EPS);
        }
      }

      ////////////////////////////////////////////////////////////
      // point distance
      for (int i = 0; i < 3; i++) {
        int pshift = shift6 + i * 2;
        scalar_t x1 = face_vertices_image[pshift + 0];
        scalar_t y1 = face_vertices_image[pshift + 1];
        pdis[i + 3] = (x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1);
      }

      int edgeid = 0;
      scalar_t dissquare = pdis[0];

      for (int i = 1; i < 6; i++) {
        if (dissquare > pdis[i]) {
          dissquare = pdis[i];
          edgeid = i;
        }
      }

      scalar_t z = sigmainv * dissquare / multiplier / multiplier;

      scalar_t prob = exp(-z);

      close_face_prob[totalidxk + kid] = prob;
      close_face_idx[totalidxk + kid] = fidxint;
      close_face_dist_type[totalidxk + kid] = edgeid + 1;
      kid++;

      if (kid >= knum)
        break;
    }

    scalar_t allprob = 1.0;
    for (int i = 0; i < kid; i++) {
      scalar_t prob = close_face_prob[totalidxk + i];
      allprob *= (1.0 - prob);
    }

    // final result
    allprob = 1.0 - allprob;
    soft_mask[totalidx1] = allprob;
  }
}

void dibr_soft_mask_forward_cuda_impl(
    const at::Tensor face_vertices_image,
    const at::Tensor face_large_bboxes,
    const at::Tensor selected_face_idx,
    at::Tensor close_face_prob,
    at::Tensor close_face_idx,
    at::Tensor close_face_dist_type,
    at::Tensor soft_mask,
    const float sigmainv,
    const float multiplier) {

  const int batch_size = face_vertices_image.size(0);
  const int num_faces = face_vertices_image.size(1);
  const int height = selected_face_idx.size(1);
  const int width = selected_face_idx.size(2);
  const int knum = close_face_idx.size(3);

  const int num_pixels = batch_size * height * width;

  AT_DISPATCH_FLOATING_TYPES(face_vertices_image.scalar_type(),
      "dibr_soft_mask_forward_cuda", [&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(face_vertices_image));
        auto stream = at::cuda::getCurrentCUDAStream();

	const int block_size = 512;
        const int grid_size = (num_pixels + block_size - 1) / block_size;
        const dim3 threads(block_size, 1, 1);
        const dim3 blocks(grid_size, 1, 1);

        dibr_soft_mask_forward_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            face_vertices_image.data_ptr<scalar_t>(),
            face_large_bboxes.data_ptr<scalar_t>(),
            selected_face_idx.data_ptr<int64_t>(),
            close_face_prob.data_ptr<scalar_t>(),
            close_face_idx.data_ptr<int64_t>(),
            close_face_dist_type.data_ptr<uint8_t>(),
            soft_mask.data_ptr<scalar_t>(),
            batch_size, height, width, num_faces, knum, sigmainv, multiplier
	);
        AT_CUDA_CHECK(cudaGetLastError());
  });
  return;
}

template<typename scalar_t>
__global__ void dibr_soft_mask_backward_cuda_kernel(
    const scalar_t* __restrict__ grad_soft_mask,
    const scalar_t* __restrict__ soft_mask,
    const int64_t* __restrict__ selected_face_idx,
    const scalar_t* __restrict__ close_face_prob,
    const int64_t* __restrict__ close_face_idx,
    const uint8_t* __restrict__ close_face_dist_type,
    const scalar_t* __restrict__ face_vertices_image,
    scalar_t* __restrict__ grad_face_vertices_image,
    int batch_size, int height, int width, int num_faces,
    int knum, float sigmainv, float multiplier) {

  int presentthread = blockIdx.x * blockDim.x + threadIdx.x;

  int wididx = presentthread % width;
  presentthread = (presentthread - wididx) / width;

  int heiidx = presentthread % height;
  int bidx = (presentthread - heiidx) / height;

  if (bidx >= batch_size || heiidx >= height || wididx >= width)
    return;

  // which pixel it belongs to
  const int totalidx1 = bidx * height * width + heiidx * width + wididx;
  const int totalidxk = totalidx1 * knum;

  // coordinates
  scalar_t x0 = multiplier / width * (2 * wididx + 1 - width);
  scalar_t y0 = multiplier / height * (height - 2 * heiidx - 1);

  // which face it belongs to?
  int fidxint = selected_face_idx[totalidx1];

  // not covered by any faces
  if (fidxint < 0) {

    scalar_t dLdp = grad_soft_mask[totalidx1];
    scalar_t allprob = soft_mask[totalidx1];

    for (int kid = 0; kid < knum; kid++) {

      int fidxint = close_face_idx[totalidxk + kid];

      if (fidxint < 0)
        break;

      const int shift1 = bidx * num_faces + fidxint;
      const int shift6 = shift1 * 6;

      scalar_t prob = close_face_prob[totalidxk + kid];

      scalar_t dLdz = -1.0 * sigmainv * dLdp * (1.0 - allprob)
          / (1.0 - prob + EPS) * prob;

      int edgecase = close_face_dist_type[totalidxk + kid];
      int edgeid = edgecase - 1;

      if (edgeid >= 3) {

        // point distance
        int pshift = shift6 + (edgeid - 3) * 2;
        scalar_t x1 = face_vertices_image[pshift + 0];
        scalar_t y1 = face_vertices_image[pshift + 1];

        scalar_t dLdx1 = dLdz * 2 * (x1 - x0);
        scalar_t dLdy1 = dLdz * 2 * (y1 - y0);

        atomicAdd(grad_face_vertices_image + pshift + 0,
            dLdx1 / multiplier);
        atomicAdd(grad_face_vertices_image + pshift + 1,
            dLdy1 / multiplier);

      } else {

        // perpendicular distance

        int pshift = shift6 + edgeid * 2;
        scalar_t x1 = face_vertices_image[pshift + 0];
        scalar_t y1 = face_vertices_image[pshift + 1];

        int pshift2 = shift6 + ((edgeid + 1) % 3) * 2;
        scalar_t x2 = face_vertices_image[pshift2 + 0];
        scalar_t y2 = face_vertices_image[pshift2 + 1];

        // ax + by + c = 0
        scalar_t A = y2 - y1;
        scalar_t B = x1 - x2;
        scalar_t C = x2 * y1 - x1 * y2;

        // dissquare = d^2 = (ax+by+c)^2 / (a^2+b^2)
        // up = ax + by + c
        // down = a^2 + b^2
        // dissquare = up^2 / down
        scalar_t up = A * x0 + B * y0 + C;
        scalar_t down = A * A + B * B;
        scalar_t dissquare = up * up / (down + EPS);

        scalar_t dzdA = 2 * (x0 * up - dissquare * A) / (down + EPS);
        scalar_t dzdB = 2 * (y0 * up - dissquare * B) / (down + EPS);
        scalar_t dzdC = 2 * up / (down + EPS);

        scalar_t dLdx1 = dLdz * (dzdB - y2 * dzdC);
        scalar_t dLdy1 = dLdz * (x2 * dzdC - dzdA);

        scalar_t dLdx2 = dLdz * (y1 * dzdC - dzdB);
        scalar_t dLdy2 = dLdz * (dzdA - x1 * dzdC);

        atomicAdd(grad_face_vertices_image + pshift + 0,
            dLdx1 / multiplier);
        atomicAdd(grad_face_vertices_image + pshift + 1,
            dLdy1 / multiplier);

        atomicAdd(grad_face_vertices_image + pshift2 + 0,
            dLdx2 / multiplier);
        atomicAdd(grad_face_vertices_image + pshift2 + 1,
            dLdy2 / multiplier);
      }
    }
  }

  return;
}

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
    const float multiplier) {


  int batch_size = face_vertices_image.size(0);
  int num_faces = face_vertices_image.size(1);
  int height = selected_face_idx.size(1);
  int width = selected_face_idx.size(2);
  int knum = close_face_idx.size(3);

  const int num_pixels = batch_size * height * width;

  AT_DISPATCH_FLOATING_TYPES(face_vertices_image.scalar_type(),
      "dibr_soft_mask_backward_cuda", [&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(face_vertices_image));
        auto stream = at::cuda::getCurrentCUDAStream();

        const int block_size = 1024;
        const int grid_size = (num_pixels + block_size - 1) / block_size;
        const dim3 threads(block_size, 1, 1);
        const dim3 blocks(grid_size, 1, 1);

        dibr_soft_mask_backward_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
             grad_soft_mask.data_ptr<scalar_t>(),
             soft_mask.data_ptr<scalar_t>(),
             selected_face_idx.data_ptr<int64_t>(),
             close_face_prob.data_ptr<scalar_t>(),
             close_face_idx.data_ptr<int64_t>(),
             close_face_dist_type.data_ptr<uint8_t>(),
             face_vertices_image.data_ptr<scalar_t>(),
             grad_face_vertices_image.data_ptr<scalar_t>(),
             batch_size, height, width, num_faces,
             knum, sigmainv, multiplier
	);
	AT_CUDA_CHECK(cudaGetLastError());

  });
  return;
}

}
