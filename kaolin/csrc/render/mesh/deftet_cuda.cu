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

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAGuard.h>

#include "../../utils.h"

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 32
#define BLOCK_SIZE 1024
#define NUM_SHARED_FACES 256
#define FULL_MASK 0xffffffff

namespace kaolin {

template <typename scalar_t>
__global__ void deftet_sparse_render_forward_cuda_kernel(
    const scalar_t *__restrict__ face_vertices_z,
    const scalar_t *__restrict__ face_vertices_image,
    const scalar_t *__restrict__ face_bboxes,

    const scalar_t *__restrict__ pixel_coords,
    const scalar_t *__restrict__ depth_limits,

    int64_t *__restrict__ face_ids,
    scalar_t *__restrict__ pixel_depths,
    scalar_t *__restrict__ w0_arr,
    scalar_t *__restrict__ w1_arr,

    const int batch_size,
    const int num_faces,
    const int num_pixels,
    const int knum,
    const float eps) {
  scalar_t x0, y0, min_depth, max_depth;
  // prefix mask is a mask representing the threadIdx.x
  // example: for threadIdx.x == 2 then the binary mask is
  // 0000 0000 0000 0000 0000 0000 0000 0011
  unsigned prefix_mask = 0;
#pragma unroll
  for (int i = 0; i < BLOCK_SIZE_X; i++) {
    if (i < threadIdx.x) {
      prefix_mask = ((prefix_mask << 1) + 1);
    }
  }
  const int threadGlobIdx = threadIdx.x + threadIdx.y * blockDim.x;
  __shared__ scalar_t shm_face_bboxes[32][5][8];
  for (int start_batch_idx = 0; start_batch_idx < batch_size;
       start_batch_idx += gridDim.x) {
    const int batch_idx = start_batch_idx + blockIdx.x;
    // threads with the same threadIdx.x (within the same warp)
    // shared the same pixel
    for (int start_pixel_idx = 0; start_pixel_idx < num_pixels;
         start_pixel_idx += blockDim.y * gridDim.y) {
      const int pixel_idx = start_pixel_idx + blockIdx.y * blockDim.y + threadIdx.y;
      const bool is_active_pixel = pixel_idx < num_pixels;
      const int main_idx = batch_idx * num_pixels + pixel_idx;
      const int pixel_coords_idx = main_idx * 2;
      if (is_active_pixel) {
        // TODO(cfujitsang): could also vectorize this load (Also ILP)
        // pixel coordinates
        x0 = pixel_coords[pixel_coords_idx + 0];
        y0 = pixel_coords[pixel_coords_idx + 1];
        min_depth = depth_limits[pixel_coords_idx + 0];
        max_depth = depth_limits[pixel_coords_idx + 1];
      }
      int num_depths = 0;
      // Here we are batching load from face_bboxes to maximize bandwidth
      // we are loading 1 bboxe per thread (1024)
      // start_face_idx is the first index of the current load batch within the mesh
      // (i.e: 0 means first face of a mesh)
      for (int start_face_idx = 0; start_face_idx < num_faces; start_face_idx += NUM_SHARED_FACES) {
        // _start_idx is the index of the start_face_idx within the whole batch of mesh
        // (i.e: 0 means first face of the first mesh)
        const int _start_idx = batch_idx * num_faces + start_face_idx;
        __syncthreads();
        const int pointsbbox_idx = threadGlobIdx + _start_idx * 4;
        // All the load are coalescent we use a shared memory of dims [32][5][8]
        // to avoid bank conflicts in the second step
        if (threadGlobIdx + start_face_idx * 4 < num_faces * 4) {
          shm_face_bboxes[threadIdx.y][threadIdx.x % 4][(threadIdx.x - (threadIdx.x % 4)) / 4] =
              face_bboxes[pointsbbox_idx];
        }
        __syncthreads();
        // We needed all the threads to be active for loading face_bboxes
        // in shared memory but now we can skip the computation
        // if the thread is not processing any pixel.
        if (!(is_active_pixel)) {
          continue;
        }
#pragma unroll
        for (int sub_start_face_idx = 0; sub_start_face_idx < NUM_SHARED_FACES;
             sub_start_face_idx += blockDim.x) {
          const int i = sub_start_face_idx + threadIdx.x;
          const int last_idx = i % 8;
          const int first_idx = (i - last_idx) / 8;
          scalar_t w0, w1, w2, pixel_depth;
          bool is_intersecting = false;
          const int face_idx = start_face_idx + i;
          if (face_idx < num_faces) {
            const int shift1 = batch_idx * num_faces + face_idx;
            const int shift6 = shift1 * 6;
            const scalar_t xmin = shm_face_bboxes[first_idx][0][last_idx];
            const scalar_t ymin = shm_face_bboxes[first_idx][1][last_idx];
            const scalar_t xmax = shm_face_bboxes[first_idx][2][last_idx];
            const scalar_t ymax = shm_face_bboxes[first_idx][3][last_idx];
            // Is the pixel covered by the bounding box?
            // [min, max)
            if (x0 >= xmin && x0 < xmax && y0 >= ymin && y0 < ymax) {
              const scalar_t ax = face_vertices_image[shift6 + 0];
              const scalar_t ay = face_vertices_image[shift6 + 1];
              const scalar_t bx = face_vertices_image[shift6 + 2];
              const scalar_t by = face_vertices_image[shift6 + 3];
              const scalar_t cx = face_vertices_image[shift6 + 4];
              const scalar_t cy = face_vertices_image[shift6 + 5];

              // Compute barycenter weights for the intersection
	      scalar_t a_edge_x = ax - x0;
	      scalar_t a_edge_y = ay - y0;
	      scalar_t b_edge_x = bx - x0;
	      scalar_t b_edge_y = by - y0;
	      scalar_t c_edge_x = cx - x0;
	      scalar_t c_edge_y = cy - y0;
	      scalar_t _w0 = b_edge_x * c_edge_y - b_edge_y * c_edge_x;
	      scalar_t _w1 = c_edge_x * a_edge_y - c_edge_y * a_edge_x;
	      scalar_t _w2 = a_edge_x * b_edge_y - a_edge_y * b_edge_x;
	      scalar_t norm = _w0 + _w1 + _w2;
	      scalar_t norm_eps = copysignf(static_cast<double>(eps),
                                      static_cast<double>(norm));
	      w0 = _w0 / (norm + norm_eps);
	      w1 = _w1 / (norm + norm_eps);
	      w2 = _w2 / (norm + norm_eps);
              // Is the pixel covered by the face?
	      // Using a boundary epsilon is necessary in case the pixel is on an edge
              if (w0 >= 0. && w1 >= 0. && w2 >= 0.) {
                // Here we are computing intersection depth
                // we can use either distance from camera or
                // distance from image plan as it won't affect ordering
                const int shift3 = shift1 * 3;
                scalar_t az = face_vertices_z[shift3 + 0];
                scalar_t bz = face_vertices_z[shift3 + 1];
                scalar_t cz = face_vertices_z[shift3 + 2];
                // TODO(cfujitsang): can we use tensorcore ?
                pixel_depth = w0 * az + w1 * bz + w2 * cz;

                if (pixel_depth < max_depth && pixel_depth >= min_depth && num_depths < knum) {
                  is_intersecting = true;
                }
              }
            }
          }
          // Since warp are sharing the faces for the same pixel we are sharing
          // the information of what thread have an intersection to render
          // this information is stored as a mask in intersection_mask
          unsigned intersection_mask = __ballot_sync(FULL_MASK, is_intersecting);
          int num_inserted = __popc(intersection_mask);
          if (is_intersecting) {
            // With the intersection mask we can compute insertion position for each threads
            // so we ensure that insertion is coalescent without holes
            // example:
            // if threadIdx.x 0 and 2 have an intersection to render but not threadIdx.x 1,
            // then threadIdx.x 0 and 2 should intersect in two consecutive addresses.
            unsigned prefix_intersection_mask = intersection_mask & prefix_mask;
            int insertion_idx = num_depths + __popc(prefix_intersection_mask);
            if (insertion_idx < knum) {
              int true_insertion_idx = insertion_idx + (batch_idx * num_pixels + pixel_idx) * knum;
              face_ids[true_insertion_idx] = face_idx;
              w0_arr[true_insertion_idx] = w0;
              w1_arr[true_insertion_idx] = w1;
              pixel_depths[true_insertion_idx] = pixel_depth;
            }
          }
          num_depths += num_inserted;
        }
      }
    }
  }
}

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
    const float eps) {
  int batch_size = face_vertices_z.size(0);
  int num_faces = face_vertices_z.size(1);

  int num_pixels = selected_face_idx.size(1);
  int knum = selected_face_idx.size(2);
  const int num_thread_per_pixel = BLOCK_SIZE_X;
  const int num_pixel_per_block = BLOCK_SIZE_Y;
  const int num_block_per_sample = (num_pixels + num_pixel_per_block - 1) / num_pixel_per_block;
  const dim3 threads(num_thread_per_pixel, num_pixel_per_block, 1);
  const dim3 blocks(batch_size, num_block_per_sample, 1);

  AT_DISPATCH_FLOATING_TYPES(face_vertices_z.scalar_type(),
    "deftet_sparse_render_forward_cuda", ([&] {
      const at::cuda::OptionalCUDAGuard device_guard(at::device_of(face_vertices_z));
      auto stream = at::cuda::getCurrentCUDAStream();

      deftet_sparse_render_forward_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        face_vertices_z.data_ptr<scalar_t>(),
        face_vertices_image.data_ptr<scalar_t>(),
        face_bboxes.data_ptr<scalar_t>(),
        pixel_coords.data_ptr<scalar_t>(),
        pixel_depth_ranges.data_ptr<scalar_t>(),
        selected_face_idx.data_ptr<int64_t>(),
        pixel_depths.data_ptr<scalar_t>(),
        w0_arr.data_ptr<scalar_t>(),
        w1_arr.data_ptr<scalar_t>(),
        batch_size, num_faces, num_pixels, knum,
	eps);
      AT_CUDA_CHECK(cudaGetLastError());
  }));
  return;
}

template <typename scalar_t>
__global__ void deftet_sparse_render_backward_cuda_kernel(
    const scalar_t *__restrict__ grad_interpolated_features,
    const int64_t *__restrict__ face_ids,
    const scalar_t *__restrict__ weights,

    const scalar_t *__restrict__ face_vertices_image,
    const scalar_t *__restrict__ face_features,

    scalar_t *__restrict__ grad_face_vertices_image,
    scalar_t *__restrict__ grad_face_features,

    const int batch_size,
    const int num_faces,
    const int num_pixels,
    const int knum,
    const int feat_dim,
    const float eps) {

  // Each iteration is treating a single feature of a single pixel
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < batch_size * num_pixels * knum;
       idx += blockDim.x * gridDim.x) {
    const int k_idx = idx % knum;
    const int true_pixel_idx = (idx - k_idx) / knum;
    const int pixel_idx = true_pixel_idx % num_pixels;
    const int batch_idx = (true_pixel_idx - pixel_idx) / num_pixels;

    const int start_weight_idx = idx * 3;
    const int start_feat_idx = idx * feat_dim;

    const int face_idx = face_ids[idx];

    if (face_idx >= 0) {
      const int true_face_idx = batch_idx * num_faces + face_idx;
      const int start_image_idx = true_face_idx * 6;
      const int start_features_idx = true_face_idx * 3 * feat_dim;

      // gradient of face_features
#pragma unroll
      for (int ii = 0; ii < 3; ii++) {
        scalar_t w = weights[start_weight_idx + ii];
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

      const scalar_t aw = weights[start_weight_idx + 0];
      const scalar_t bw = weights[start_weight_idx + 1];
      const scalar_t cw = weights[start_weight_idx + 2];

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
      // Need to explicitly cast because there is a bug on windows.
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

void deftet_sparse_render_backward_cuda_impl(
    const at::Tensor grad_interpolated_features,
    const at::Tensor face_idx,
    const at::Tensor weights,
    const at::Tensor face_vertices_image,
    const at::Tensor face_features,
    at::Tensor grad_face_vertices_image,
    at::Tensor grad_face_features,
    const float eps) {

  int batch_size = grad_interpolated_features.size(0);
  int num_pixels = grad_interpolated_features.size(1);
  int knum = grad_interpolated_features.size(2);
  int feat_dim = grad_interpolated_features.size(3);

  int num_faces = grad_face_vertices_image.size(1);

  // for bxhxw image size
  const int threads = 512;
  const int totalthread = batch_size * num_pixels * knum;
  const int blocks = (totalthread + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(grad_interpolated_features.scalar_type(),
    "deftet_sparse_render_backward_cuda", ([&] {
      const at::cuda::OptionalCUDAGuard device_guard(at::device_of(face_vertices_image));
      auto stream = at::cuda::getCurrentCUDAStream();
      deftet_sparse_render_backward_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
          grad_interpolated_features.data_ptr<scalar_t>(),
          face_idx.data_ptr<int64_t>(),
          weights.data_ptr<scalar_t>(),
          face_vertices_image.data_ptr<scalar_t>(),
          face_features.data_ptr<scalar_t>(),
          grad_face_vertices_image.data_ptr<scalar_t>(),
          grad_face_features.data_ptr<scalar_t>(),
          batch_size, num_faces, num_pixels, knum, feat_dim,
          eps);
      AT_CUDA_CHECK(cudaGetLastError());
    })
  );
}


}  // namespace kaolin
