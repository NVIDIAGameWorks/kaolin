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
#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAGuard.h>

#include "../../utils.h"

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
    const int batch_size,
    const int height,
    const int width,
    const int num_faces,
    const int num_features,
    const float multiplier,
    const float eps) {

  __shared__ scalar_t shm_pointsbbox[BLOCK_SIZE][4];
  for (int bidx = blockIdx.y; bidx < batch_size; bidx += gridDim.y) {
    for (int start_pixel_idx = blockIdx.x * blockDim.x;
         start_pixel_idx < width * height;
         start_pixel_idx += gridDim.x * blockDim.x) {
      const int pixel_idx = start_pixel_idx + threadIdx.x;

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
          // This is a bounding box of the face
          const scalar_t xmin = shm_pointsbbox[ii][0];
          const scalar_t ymin = shm_pointsbbox[ii][1];
          const scalar_t xmax = shm_pointsbbox[ii][2];
          const scalar_t ymax = shm_pointsbbox[ii][3];

          // The pixel doesn't lie in the bounding box
          if (x0 < xmin || x0 >= xmax || y0 < ymin || y0 >= ymax) {
            continue;
          }

          const int shift1 = face_idx;
          const int shift3 = shift1 * 3;
          const int shift6 = shift1 * 6;

          // if this pixel is covered by this face, then we check its depth and weights
          const scalar_t ax = face_vertices_image[shift6 + 0];
          const scalar_t ay = face_vertices_image[shift6 + 1];
          const scalar_t bx = face_vertices_image[shift6 + 2];
          const scalar_t by = face_vertices_image[shift6 + 3];
          const scalar_t cx = face_vertices_image[shift6 + 4];
          const scalar_t cy = face_vertices_image[shift6 + 5];

          const scalar_t a_edge_x = ax - x0;
          const scalar_t a_edge_y = ay - y0;
          const scalar_t b_edge_x = bx - x0;
          const scalar_t b_edge_y = by - y0;
          const scalar_t c_edge_x = cx - x0;
          const scalar_t c_edge_y = cy - y0;
          scalar_t w0 = b_edge_x * c_edge_y - b_edge_y * c_edge_x;
          scalar_t w1 = c_edge_x * a_edge_y - c_edge_y * a_edge_x;
          scalar_t w2 = a_edge_x * b_edge_y - a_edge_y * b_edge_x;
          scalar_t norm = w0 + w1 + w2;
          norm += copysign(static_cast<double>(eps),
                           static_cast<double>(norm));
          w0 /= norm;
          w1 /= norm;
          w2 /= norm;

          // The pixel doesn't lie in the triangle
          if (w0 < 0. || w1 < 0. || w2 < 0.) {
            continue;
          }

          // if it is perspective, then this way has a little error
          // because face plane may not be parallel to the image plane
          // but let's ignore it first
          const scalar_t az = face_vertices_z[shift3 + 0];
          const scalar_t bz = face_vertices_z[shift3 + 1];
          const scalar_t cz = face_vertices_z[shift3 + 2];

          const scalar_t z0 = w0 * az + w1 * bz + w2 * cz;

          // The intersection is not the closest from the camera
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
          const scalar_t r0 = face_features[shift3d + d];
          const scalar_t r1 = face_features[shift3d + num_features + d];
          const scalar_t r2 = face_features[shift3d + num_features + num_features + d];
          interpolated_features[totalidxd + d] = max_w0 * r0 + max_w1 * r1 + max_w2 * r2;
        }
      }
    }
  }
}


void packed_rasterize_forward_cuda_impl(
    const at::Tensor face_vertices_z,
    const at::Tensor face_vertices_image,
    const at::Tensor face_bboxes,
    const at::Tensor face_features,
    const at::Tensor num_face_per_mesh,
    at::Tensor selected_face_idx,
    at::Tensor output_weights,
    at::Tensor interpolated_features,
    const float multiplier,
    const float eps) {

  const int num_faces = face_vertices_z.size(1);
  const int batch_size = interpolated_features.size(0);
  const int height = interpolated_features.size(1);
  const int width = interpolated_features.size(2);
  const int num_features = interpolated_features.size(3);
  const int num_pixels = height * width;

  DISPATCH_INPUT_TYPES(face_vertices_z.scalar_type(), scalar_t,
    "packed_rasterize_forward_cuda", [&] {
      const at::cuda::OptionalCUDAGuard device_guard(at::device_of(face_vertices_z));
      auto stream = at::cuda::getCurrentCUDAStream();

      const int num_blocks_per_sample = num_pixels / block_size + 1;
      const dim3 threads(block_size, 1, 1);
      const dim3 blocks(num_blocks_per_sample, 1, 1);

      packed_rasterize_forward_cuda_kernel<scalar_t, block_size><<<blocks, threads, 0, stream>>>(
          face_vertices_z.data_ptr<scalar_t>(),
          face_vertices_image.data_ptr<scalar_t>(),
          face_bboxes.data_ptr<scalar_t>(),
          face_features.data_ptr<scalar_t>(),
          num_face_per_mesh.data_ptr<int64_t>(),
          selected_face_idx.data_ptr<int64_t>(),
          output_weights.data_ptr<scalar_t>(),
          interpolated_features.data_ptr<scalar_t>(),
          batch_size, height, width, num_faces, num_features,
	  multiplier, eps);
      AT_CUDA_CHECK(cudaGetLastError());
    });
}

template<typename scalar_t>
__global__ void rasterize_backward_cuda_kernel(
    const scalar_t* __restrict__ grad_interpolated_features,
    const int64_t* __restrict__ selected_face_idx,
    const scalar_t* __restrict__ output_weights,
    const scalar_t* __restrict__ face_vertices_image,
    const scalar_t* __restrict__ face_features,
    scalar_t* __restrict__ grad_face_vertices_image,
    scalar_t* __restrict__ grad_face_features,
    const int batch_size,
    const int height,
    const int width,
    const int num_faces,
    const int feat_dim,
    const float eps) {
  const int num_pixels = height * width;
  // Each iteration is treating a single feature of a single pixel
  for (int true_pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
       true_pixel_idx < batch_size * num_pixels;
       true_pixel_idx += blockDim.x * gridDim.x) {
    const int pixel_idx = true_pixel_idx % num_pixels;
    const int batch_idx = (true_pixel_idx - pixel_idx) / num_pixels;

    const int start_weight_idx = true_pixel_idx * 3;
    const int start_feat_idx = true_pixel_idx * feat_dim;

    const int face_idx = selected_face_idx[true_pixel_idx];

    if (face_idx >= 0) {
      const int true_face_idx = batch_idx * num_faces + face_idx;
      const int start_image_idx = true_face_idx * 6;
      const int start_features_idx = true_face_idx * 3 * feat_dim;

      // gradient of face_features
#pragma unroll
      for (int ii = 0; ii < 3; ii++) {
        scalar_t w = output_weights[start_weight_idx + ii];
        int pointshift = start_features_idx + ii * feat_dim;

        for (int feat_idx = 0; feat_idx < feat_dim; feat_idx++) {
          int colorshift = pointshift + feat_idx;

          // this should be atomic operation
          scalar_t *addr = grad_face_features + colorshift;
          scalar_t val = grad_interpolated_features[start_feat_idx + feat_idx] * w;
          atomicAdd(addr, val);
        }
      }

      // gradient of points
      // here, we calculate dl/dp
      // dl/dp = dldI * dI/dp
      // dI/dp = c0 * dw0 / dp + c1 * dw1 / dp + c2 * dw2 / dp
      const scalar_t ax = face_vertices_image[start_image_idx + 0];
      const scalar_t ay = face_vertices_image[start_image_idx + 1];
      const scalar_t bx = face_vertices_image[start_image_idx + 2];
      const scalar_t by = face_vertices_image[start_image_idx + 3];
      const scalar_t cx = face_vertices_image[start_image_idx + 4];
      const scalar_t cy = face_vertices_image[start_image_idx + 5];

      const scalar_t aw = output_weights[start_weight_idx + 0];
      const scalar_t bw = output_weights[start_weight_idx + 1];
      const scalar_t cw = output_weights[start_weight_idx + 2];

      const scalar_t x0 = aw * ax + bw * bx + cw * cx;
      const scalar_t y0 = aw * ay + bw * by + cw * cy;

      const scalar_t m = bx - ax;
      const scalar_t p = by - ay;

      const scalar_t n = cx - ax;
      const scalar_t q = cy - ay;

      const scalar_t s = x0 - ax;
      const scalar_t t = y0 - ay;

      // m * w1 + n * w2 = s
      // p * w1 + q * w2 = t
      // w1 = (sq - nt) / (mq - np)
      // w2 = (mt - sp) / (mq - np)

      const scalar_t k1 = s * q - n * t;
      const scalar_t k2 = m * t - s * p;
      scalar_t k3 = m * q - n * p;
      k3 += copysign(static_cast<double>(eps), static_cast<double>(k3));

      const scalar_t dk1dm = 0;
      const scalar_t dk1dn = -t;
      const scalar_t dk1dp = 0;
      const scalar_t dk1dq = s;
      const scalar_t dk1ds = q;
      const scalar_t dk1dt = -n;

      const scalar_t dk2dm = t;
      const scalar_t dk2dn = 0;
      const scalar_t dk2dp = -s;
      const scalar_t dk2dq = 0;
      const scalar_t dk2ds = -p;
      const scalar_t dk2dt = m;

      const scalar_t dk3dm = q;
      const scalar_t dk3dn = -p;
      const scalar_t dk3dp = -n;
      const scalar_t dk3dq = m;
      const scalar_t dk3ds = 0;
      const scalar_t dk3dt = 0;

      // w1 = k1 / k3
      // w2 = k2 / k3
      // we need divide k3 ^ 2
      const scalar_t dw1dm = dk1dm * k3 - dk3dm * k1;
      const scalar_t dw1dn = dk1dn * k3 - dk3dn * k1;
      const scalar_t dw1dp = dk1dp * k3 - dk3dp * k1;
      const scalar_t dw1dq = dk1dq * k3 - dk3dq * k1;
      const scalar_t dw1ds = dk1ds * k3 - dk3ds * k1;
      const scalar_t dw1dt = dk1dt * k3 - dk3dt * k1;

      const scalar_t dw2dm = dk2dm * k3 - dk3dm * k2;
      const scalar_t dw2dn = dk2dn * k3 - dk3dn * k2;
      const scalar_t dw2dp = dk2dp * k3 - dk3dp * k2;
      const scalar_t dw2dq = dk2dq * k3 - dk3dq * k2;
      const scalar_t dw2ds = dk2ds * k3 - dk3ds * k2;
      const scalar_t dw2dt = dk2dt * k3 - dk3dt * k2;

      const scalar_t dw1dax = -(dw1dm + dw1dn + dw1ds);
      const scalar_t dw1day = -(dw1dp + dw1dq + dw1dt);
      const scalar_t dw1dbx = dw1dm;
      const scalar_t dw1dby = dw1dp;
      const scalar_t dw1dcx = dw1dn;
      const scalar_t dw1dcy = dw1dq;

      const scalar_t dw2dax = -(dw2dm + dw2dn + dw2ds);
      const scalar_t dw2day = -(dw2dp + dw2dq + dw2dt);
      const scalar_t dw2dbx = dw2dm;
      const scalar_t dw2dby = dw2dp;
      const scalar_t dw2dcx = dw2dn;
      const scalar_t dw2dcy = dw2dq;

      for (int feat_idx = 0; feat_idx < feat_dim; feat_idx++) {

        const scalar_t c0 = face_features[start_features_idx + feat_idx];
        const scalar_t c1 = face_features[start_features_idx + feat_dim + feat_idx];
        const scalar_t c2 = face_features[start_features_idx + feat_dim + feat_dim + feat_idx];

        const scalar_t dIdax = (c1 - c0) * dw1dax + (c2 - c0) * dw2dax;
        const scalar_t dIday = (c1 - c0) * dw1day + (c2 - c0) * dw2day;
        const scalar_t dIdbx = (c1 - c0) * dw1dbx + (c2 - c0) * dw2dbx;
        const scalar_t dIdby = (c1 - c0) * dw1dby + (c2 - c0) * dw2dby;
        const scalar_t dIdcx = (c1 - c0) * dw1dcx + (c2 - c0) * dw2dcx;
        const scalar_t dIdcy = (c1 - c0) * dw1dcy + (c2 - c0) * dw2dcy;

        const scalar_t dldI = grad_interpolated_features[start_feat_idx + feat_idx] / (k3 * k3);

        atomicAdd(grad_face_vertices_image + start_image_idx + 0, dldI * dIdax);
        atomicAdd(grad_face_vertices_image + start_image_idx + 1, dldI * dIday);

        atomicAdd(grad_face_vertices_image + start_image_idx + 2, dldI * dIdbx);
        atomicAdd(grad_face_vertices_image + start_image_idx + 3, dldI * dIdby);

        atomicAdd(grad_face_vertices_image + start_image_idx + 4, dldI * dIdcx);
        atomicAdd(grad_face_vertices_image + start_image_idx + 5, dldI * dIdcy);
      }
    }
  }
}

void rasterize_backward_cuda_impl(
    const at::Tensor grad_interpolated_features,
    const at::Tensor interpolated_features,
    const at::Tensor selected_face_idx,
    const at::Tensor output_weights,
    const at::Tensor face_vertices_image,
    const at::Tensor face_features,
    at::Tensor grad_face_vertices_image,
    at::Tensor grad_face_features,
    const float eps) {

  const int batch_size = grad_interpolated_features.size(0);
  const int height = grad_interpolated_features.size(1);
  const int width = grad_interpolated_features.size(2);
  const int feat_dim = grad_interpolated_features.size(3);
  const int num_faces = grad_face_vertices_image.size(1);

  // for bxhxw image size
  const int threads = 512;
  const int total_num_pixels = batch_size * height * width;
  const int blocks = total_num_pixels / threads;

  // we exchange block and thread!
  AT_DISPATCH_FLOATING_TYPES(grad_interpolated_features.scalar_type(),
    "rasterize_backward_cuda", ([&] {
      const at::cuda::OptionalCUDAGuard device_guard(at::device_of(grad_interpolated_features));
      auto stream = at::cuda::getCurrentCUDAStream();
      rasterize_backward_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          grad_interpolated_features.data_ptr<scalar_t>(),
          selected_face_idx.data_ptr<int64_t>(),
          output_weights.data_ptr<scalar_t>(),
          face_vertices_image.data_ptr<scalar_t>(),
          face_features.data_ptr<scalar_t>(),
          grad_face_vertices_image.data_ptr<scalar_t>(),
          grad_face_features.data_ptr<scalar_t>(),
          batch_size, height, width, num_faces, feat_dim, eps);
      AT_CUDA_CHECK(cudaGetLastError());
    }));
}

}  // namespace kaolin

#undef PRIVATE_CASE_TYPE_AND_VAL
#undef DISPATCH_INPUT_TYPES
