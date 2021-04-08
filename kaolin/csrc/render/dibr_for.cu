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

#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../utils.h"

#define eps 1e-7

#define PRIVATE_CASE_TYPE_AND_VAL(ENUM_TYPE, TYPE, TYPE_NAME, VAL, ...) \
  case ENUM_TYPE: { \
    using TYPE_NAME = TYPE; \
    const int block_size = VAL; \
    return __VA_ARGS__(); \
  }

#define DISPATCH_INPUT_TYPES(TYPE, TYPE_NAME, SCOPE_NAME, ...) \
  [&] { \
    switch(TYPE) \
    { \
      PRIVATE_CASE_TYPE_AND_VAL(at::ScalarType::Float, float, TYPE_NAME, 1024, __VA_ARGS__) \
      PRIVATE_CASE_TYPE_AND_VAL(at::ScalarType::Double, double, TYPE_NAME, 512, __VA_ARGS__) \
      default: \
        AT_ERROR(#SCOPE_NAME, " not implemented for '", toString(TYPE), "'"); \
    } \
  }()

namespace kaolin {

template<typename scalar_t, int BLOCK_SIZE>
__global__ void packed_rasterize_forward_cuda_kernel(
    const scalar_t* __restrict__ face_vertices_z,
    const scalar_t* __restrict__ face_vertices_image,
    const scalar_t* __restrict__ face_bboxes,
    const scalar_t* __restrict__ face_features,
    const int64_t* __restrict__ first_idx_face_per_mesh,
    int64_t* __restrict__ selected_face_idx,
    scalar_t* __restrict__ output_weights,
    scalar_t* __restrict__ interpolated_features,
    int batch_size,
    int height,
    int width,
    int num_faces,
    int num_features,
    float multiplier) {

  __shared__ scalar_t shm_pointsbbox[BLOCK_SIZE][4];
  for (int bidx = blockIdx.y; bidx < batch_size; bidx += gridDim.y) {
    for (int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
         pixel_idx < width * height;
         pixel_idx += gridDim.x * blockDim.x) {

      const int wididx = pixel_idx % width;
      const int heiidx = (pixel_idx - wididx) / width;

      const int first_id_faces = first_idx_face_per_mesh[bidx];
      const int last_id_faces = first_idx_face_per_mesh[bidx + 1];
      scalar_t max_z0 = -INFINITY;
      int max_face_idx = -1;
      scalar_t max_w0 = 0.;
      scalar_t max_w1 = 0.;
      scalar_t max_w2 = 0.;
      bool is_active_pixel = heiidx < height;
      // which pixel it belongs to
      const int totalidx1 = bidx * height * width + pixel_idx;
      const int totalidx3 = totalidx1 * 3;
      const int totalidxd = totalidx1 * num_features;

      // pixel coordinate
      scalar_t x0 = multiplier / width * (2 * wididx + 1 - width);
      scalar_t y0 = multiplier / height * (height - 2 * heiidx - 1);

      for (int start_face_idx = first_id_faces;
           start_face_idx < last_id_faces;
           start_face_idx += BLOCK_SIZE) {
        const int remaining_faces = last_id_faces - start_face_idx;
        const int num_faces_this_iter = remaining_faces > BLOCK_SIZE ? BLOCK_SIZE : remaining_faces;
        __syncthreads();
#pragma unroll
        for (int ii = 0; ii < 4; ii++) {
          const int _start_idx = start_face_idx * 4 + threadIdx.x + ii * blockDim.x;
          if (_start_idx < (last_id_faces * 4)) {
            shm_pointsbbox[((threadIdx.x - (threadIdx.x % 4) + ii * blockDim.x) / 4)][threadIdx.x % 4] = \
                face_bboxes[_start_idx];
          }
        }
        __syncthreads();
        if (!(is_active_pixel)) {
          continue;
        }
        for (int ii = 0; ii < num_faces_this_iter; ii++) {
          int face_idx = ii + start_face_idx;
          // will this pixel be influenced by this face?
          scalar_t xmin = shm_pointsbbox[ii][0];
          scalar_t ymin = shm_pointsbbox[ii][1];
          scalar_t xmax = shm_pointsbbox[ii][2];
          scalar_t ymax = shm_pointsbbox[ii][3];

          // not covered by this face!
          if (x0 < xmin || x0 >= xmax || y0 < ymin || y0 >= ymax) {
            continue;
          }

          const int shift1 = face_idx;
          const int shift3 = shift1 * 3;
          const int shift6 = shift1 * 6;
          //const int shift9 = shift1 * 9;

          // if this pixel is covered by this face, then we check its depth and weights
          scalar_t ax = face_vertices_image[shift6 + 0];
          scalar_t ay = face_vertices_image[shift6 + 1];
          scalar_t bx = face_vertices_image[shift6 + 2];
          scalar_t by = face_vertices_image[shift6 + 3];
          scalar_t cx = face_vertices_image[shift6 + 4];
          scalar_t cy = face_vertices_image[shift6 + 5];

          // replace with other variables
          scalar_t m = bx - ax;
          scalar_t p = by - ay;

          scalar_t n = cx - ax;
          scalar_t q = cy - ay;

          scalar_t s = x0 - ax;
          scalar_t t = y0 - ay;

          // m* w1 + n * w2 = s
          // p * w1 + q * w2 = t
          scalar_t k1 = s * q - n * t;
          scalar_t k2 = m * t - s * p;
          scalar_t k3 = m * q - n * p;

          scalar_t w1 = k1 / (k3 + eps);
          scalar_t w2 = k2 / (k3 + eps);
          scalar_t w0 = 1 - w1 - w2; // TODO(cfujitsang): 1. instead of 1 (but would change values)

          // not lie in the triangle
          // some tmies, there would be small shift in boundaries
          if (w0 < -eps || w1 < -eps || w2 < -eps) {
            continue;
          }

          // if it is perspective, then this way has a little error
          // because face plane may not be parallel to the image plane
          // but let's ignore it first
          scalar_t az = face_vertices_z[shift3 + 0];
          scalar_t bz = face_vertices_z[shift3 + 1];
          scalar_t cz = face_vertices_z[shift3 + 2];

          scalar_t z0 = w0 * az + w1 * bz + w2 * cz;

          // it will be filled by a nearer face
          if (z0 <= max_z0) {
            continue;
          }
          max_z0 = z0;
          max_face_idx = face_idx;
          max_w0 = w0;
          max_w1 = w1;
          max_w2 = w2;
        }
      }
      if (max_face_idx > -1) {
        // index
        selected_face_idx[totalidx1] = max_face_idx - first_id_faces;
        const int shift3d = max_face_idx * 3 * num_features;

        // wei
        output_weights[totalidx3 + 0] = max_w0;
        output_weights[totalidx3 + 1] = max_w1;
        output_weights[totalidx3 + 2] = max_w2;

        // color
        for (int d = 0; d < num_features; d++) {
          scalar_t r0 = face_features[shift3d + d];
          scalar_t r1 = face_features[shift3d + num_features + d];
          scalar_t r2 = face_features[shift3d + num_features + num_features + d];
          interpolated_features[totalidxd + d] = max_w0 * r0 + max_w1 * r1 + max_w2 * r2;
        }
      }
    }
  }
}


void packed_rasterize_forward_cuda_kernel_launcher(
    at::Tensor face_vertices_z,
    at::Tensor face_vertices_image,
    at::Tensor face_bboxes,
    at::Tensor face_features,
    at::Tensor num_face_per_mesh,
    at::Tensor selected_face_idx,
    at::Tensor output_weights,
    at::Tensor interpolated_features,
    float multiplier) {

  const int num_faces = face_vertices_z.size(1);
  const int batch_size = interpolated_features.size(0);
  const int height = interpolated_features.size(1);
  const int width = interpolated_features.size(2);
  const int num_features = interpolated_features.size(3);
  const int num_pixels = height * width;

  DISPATCH_INPUT_TYPES(face_vertices_z.scalar_type(), scalar_t,
    "packed_rasterize_forward_cuda_kernel", [&] {

      const int num_blocks_per_sample = num_pixels / block_size + 1;
      const dim3 threads(block_size, 1, 1);
      const dim3 blocks(num_blocks_per_sample, 1, 1);

      packed_rasterize_forward_cuda_kernel<scalar_t, block_size><<<blocks, threads>>>(
          face_vertices_z.data_ptr<scalar_t>(),
          face_vertices_image.data_ptr<scalar_t>(),
          face_bboxes.data_ptr<scalar_t>(),
          face_features.data_ptr<scalar_t>(),
          num_face_per_mesh.data_ptr<int64_t>(),
          selected_face_idx.data_ptr<int64_t>(),
          output_weights.data_ptr<scalar_t>(),
          interpolated_features.data_ptr<scalar_t>(),
          batch_size, height, width, num_faces, num_features, multiplier);
    });
}

template<typename scalar_t>
__global__ void generate_soft_mask_cuda_kernel(
    const scalar_t* __restrict__ face_vertices_image,
    const scalar_t* __restrict__ pointsbbox2_bxfx4,
    const int64_t* __restrict__ selected_face_idx,
    scalar_t* __restrict__ probface_bxhxwxk,
    scalar_t* __restrict__ probcase_bxhxwxk,
    scalar_t* __restrict__ probdis_bxhxwxk,
    scalar_t* __restrict__ improb_bxhxwx1,
    int bnum, int height, int width, int fnum,
    int knum, float multiplier, float sigmainv) {

  // bidx * height * width + heiidx * width + wididx
  int presentthread = blockIdx.x * blockDim.x + threadIdx.x;

  int wididx = presentthread % width;
  presentthread = (presentthread - wididx) / width;

  int heiidx = presentthread % height;
  int bidx = (presentthread - heiidx) / height;

  if (bidx >= bnum || heiidx >= height || wididx >= width) {
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
    improb_bxhxwx1[totalidx1] = 1.0;
  }
  //  pixels not covered by any faces
  else {

    // pixel coordinate
    scalar_t x0 = 1.0 * multiplier / width * (2 * wididx + 1 - width);
    scalar_t y0 = 1.0 * multiplier / height * (height - 2 * heiidx - 1);

    int kid = 0;

    for (int fidxint = 0; fidxint < fnum; fidxint++) {

      // which face it belongs to
      const int shift1 = bidx * fnum + fidxint;
      const int shift4 = shift1 * 4;
      const int shift6 = shift1 * 6;

      ///////////////////////////////////////////////////////////////
      // will this pixel is influenced by this face?
      scalar_t xmin = pointsbbox2_bxfx4[shift4 + 0];
      scalar_t ymin = pointsbbox2_bxfx4[shift4 + 1];
      scalar_t xmax = pointsbbox2_bxfx4[shift4 + 2];
      scalar_t ymax = pointsbbox2_bxfx4[shift4 + 3];

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
        x3 = x3 / (down + eps);
        y3 = y3 / (down + eps);

        scalar_t direct = (x3 - x1) * (x3 - x2) + (y3 - y1) * (y3 - y2);

        if (direct > 0) {
          // bad triangle
          pdis[i] = 4 * multiplier * multiplier;
        } else {
          // perpendicular  distance
          pdis[i] = up * up / (down + eps);
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

      probface_bxhxwxk[totalidxk + kid] = fidxint + 1.0;
      probcase_bxhxwxk[totalidxk + kid] = edgeid + 1.0;
      probdis_bxhxwxk[totalidxk + kid] = prob;
      kid++;

      if (kid >= knum)
        break;
    }

    scalar_t allprob = 1.0;
    for (int i = 0; i < kid; i++) {
      scalar_t prob = probdis_bxhxwxk[totalidxk + i];
      allprob *= (1.0 - prob);
    }

    // final result
    allprob = 1.0 - allprob;
    improb_bxhxwx1[totalidx1] = allprob;
  }
}

void generate_soft_mask_cuda_kernel_launcher(
    at::Tensor face_vertices_image,
    at::Tensor face_bboxes,
    at::Tensor selected_face_idx,
    at::Tensor probface_bxhxwxk,
    at::Tensor probcase_bxhxwxk,
    at::Tensor probdis_bxhxwxk,
    at::Tensor improb_bxhxwx1,
    float multiplier,
    float sigmainv) {

  int batch_size = face_vertices_image.size(0);
  int num_faces = face_vertices_image.size(1);
  int height = selected_face_idx.size(1);
  int width = selected_face_idx.size(2);
  int knum = probface_bxhxwxk.size(3);

  const int num_pixels = batch_size * height * width;

  DISPATCH_INPUT_TYPES(face_vertices_image.scalar_type(), scalar_t,
      "generate_soft_mask_cuda_kernel", [&] {

        const int grid_size = num_pixels / block_size + 1;
        const dim3 threads(block_size, 1, 1);
        const dim3 blocks(grid_size, 1, 1);

        generate_soft_mask_cuda_kernel<scalar_t><<<blocks, threads>>>(
            face_vertices_image.data_ptr<scalar_t>(),
            face_bboxes.data_ptr<scalar_t>(),
            selected_face_idx.data_ptr<int64_t>(),
            probface_bxhxwxk.data_ptr<scalar_t>(),
            probcase_bxhxwxk.data_ptr<scalar_t>(),
            probdis_bxhxwxk.data_ptr<scalar_t>(),
            improb_bxhxwx1.data_ptr<scalar_t>(),
            batch_size, height, width, num_faces, knum, multiplier, sigmainv);
      });
  return;
}

}  // namespace kaolin

#undef PRIVATE_CASE_TYPE_AND_VAL
#undef DISPATCH_INPUT_TYPES
