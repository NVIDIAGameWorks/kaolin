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

// Spherical Gaussians fused Inner Product + sum on "others"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <c10/cuda/CUDAGuard.h>

#include "../../utils.h"

namespace kaolin {

// TODO(cfujitsang): There is a faster implementation if num_sg is very big or num_other is very small
//                   We should have both with kernel selection
__global__
void unbatched_reduced_sg_inner_product_forward_cuda_kernel(
    const float* __restrict__ intensity,
    const float* __restrict__ direction,
    const float* __restrict__ sharpness,
    const float* __restrict__ other_intensity,
    const float* __restrict__ other_direction,
    const float* __restrict__ other_sharpness,
    const int num_sg,
    const int num_other,
    float* output) {
  __shared__ float shm[32][3];
  for (int start_sg_idx = blockDim.x * blockIdx.x;
       start_sg_idx < num_sg;
       start_sg_idx += gridDim.x * blockDim.x) {
    int sg_idx = start_sg_idx + threadIdx.y;
    // Load the directions
    __syncthreads();
    if (threadIdx.y == 0) {
#pragma unroll
      for (int ii = 0; ii < 3; ii++) {
        int inner_idx = blockDim.x * ii + threadIdx.x;
        if (start_sg_idx * 3 + inner_idx < num_sg * 3) {
          shm[inner_idx / 3][inner_idx % 3] = direction[start_sg_idx * 3 + inner_idx];
        }
      }
    }
    // Load the sharpness
    float sharp = sharpness[sg_idx];
    // Load the intensity
    __syncthreads();
    float val_x = shm[threadIdx.y][0] * sharp;
    float val_y = shm[threadIdx.y][1] * sharp;
    float val_z = shm[threadIdx.y][2] * sharp;
    __syncthreads();
    if (threadIdx.y == 0) {
#pragma unroll
      for (int ii = 0; ii < 3; ii++) {
        int inner_idx = blockDim.x * ii + threadIdx.x;
        if (start_sg_idx * 3 + inner_idx < num_sg * 3) {
          shm[inner_idx / 3][inner_idx % 3] = intensity[start_sg_idx * 3 + inner_idx];
        }
      }
    }
    __syncthreads();
    float intensity_x = shm[threadIdx.y][0];
    float intensity_y = shm[threadIdx.y][1];
    float intensity_z = shm[threadIdx.y][2];
    float sum_x = 0.;
    float sum_y = 0.;
    float sum_z = 0.;
    if (start_sg_idx < num_sg) {
      for (int start_other_idx = 0; start_other_idx < num_other; start_other_idx += blockDim.x) {
	// load and share "others" sg parameters to do the broadcast operations
	// While the block is 32x32 we are only loading on the first 32 threads
	// The broadcast operation is then done with all the threads
        __syncthreads();
        int other_idx = start_other_idx + threadIdx.x;
        float other_sharp = other_sharpness[other_idx];
        if (threadIdx.y == 0) {
#pragma unroll
          for (int jj = 0; jj < 3; jj++) {
            int inner_idx = blockDim.x * jj + threadIdx.x;
            if (start_other_idx * 3 + inner_idx < num_other * 3) {
              shm[inner_idx / 3][inner_idx % 3] = other_direction[start_other_idx * 3 + inner_idx];
            }
          }
        }
        __syncthreads();
        float other_val_x = shm[threadIdx.x][0] * other_sharp;
        float other_val_y = shm[threadIdx.x][1] * other_sharp;
        float other_val_z = shm[threadIdx.x][2] * other_sharp;
        __syncthreads();
        if (threadIdx.y == 0) {
#pragma unroll
          for (int jj = 0; jj < 3; jj++) {
            int inner_idx = blockDim.x * jj + threadIdx.x;
            if (start_other_idx * 3 + inner_idx < num_other * 3) {
              shm[inner_idx / 3][inner_idx % 3] = other_intensity[start_other_idx * 3 + inner_idx];
            }
          }
        }
        float tmp_x = val_x + other_val_x;
        float tmp_y = val_y + other_val_y;
        float tmp_z = val_z + other_val_z;
        float um = sqrtf(tmp_x * tmp_x + tmp_y * tmp_y + tmp_z * tmp_z);
        float lm = sharp + other_sharp;
        __syncthreads();
        float intensity_prod_x = shm[threadIdx.x][0] * intensity_x;
        float intensity_prod_y = shm[threadIdx.x][1] * intensity_y;
        float intensity_prod_z = shm[threadIdx.x][2] * intensity_z;
        float mul_coeff = expf(um - lm) * 2. * M_PI * (1. - expf(-2 * um)) / um;
        if (other_idx < num_other) {
          sum_x += intensity_prod_x * mul_coeff;
          sum_y += intensity_prod_y * mul_coeff;
          sum_z += intensity_prod_z * mul_coeff;
        }
      }
      // Sum reduction across threads of same threadIdx.y
      for (int offset = 16; offset > 0; offset /= 2) {
        sum_x += __shfl_down_sync(0xfffffff, sum_x, offset);
        sum_y += __shfl_down_sync(0xfffffff, sum_y, offset);
        sum_z += __shfl_down_sync(0xfffffff, sum_z, offset);
      }
      // Use shared memory for coalescent store
      __syncthreads();
      if (threadIdx.x == 0) {
        shm[threadIdx.y][0] = sum_x;
        shm[threadIdx.y][1] = sum_y;
        shm[threadIdx.y][2] = sum_z;
      }
      __syncthreads();
      if (threadIdx.y == 0) {
#pragma unroll
        for (int ii = 0; ii < 3; ii++) {
          int inner_idx = blockDim.x * ii + threadIdx.x;
          if (start_sg_idx * 3 + inner_idx < num_sg * 3) {
            output[start_sg_idx * 3 + inner_idx] = shm[inner_idx / 3][inner_idx % 3];
          }
        }
      }
    }
  }
}

__global__
void unbatched_reduced_sg_inner_product_backward_cuda_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ intensity,
    const float* __restrict__ direction,
    const float* __restrict__ sharpness,
    const float* __restrict__ other_intensity,
    const float* __restrict__ other_direction,
    const float* __restrict__ other_sharpness,
    const int num_sg,
    const int num_other,
    float* __restrict__ grad_intensity,
    float* __restrict__ grad_direction,
    float* __restrict__ grad_sharpness,
    float* __restrict__ grad_other_intensity_cache,
    float* __restrict__ grad_other_direction_cache,
    float* __restrict__ grad_other_sharpness_cache) {
  // TODO(cfujitsang): need to add a lot of comments
  volatile __shared__ float shm[16][17];
  volatile float twopi = 2. * M_PI;
  volatile float zero = 0.;
 
  for (int start_sg_idx = blockDim.x * blockIdx.x;
       start_sg_idx < num_sg;
       start_sg_idx += gridDim.x * blockDim.x) {
    int sg_idx = start_sg_idx + threadIdx.y;
    bool is_active_sg = sg_idx < num_sg;
    __syncthreads();
    if (threadIdx.y == 0) {
#pragma unroll
      for (int ii = 0; ii < 3; ii++) {
        const int inner_idx = blockDim.x * ii + threadIdx.x;
        if (start_sg_idx * 3 + inner_idx < num_sg * 3) {
          shm[inner_idx / 3][inner_idx % 3] = direction[start_sg_idx * 3 + inner_idx];
        }
      }
    }
    float sharp = sharpness[sg_idx];
    __syncthreads();
    float direction_x = is_active_sg ? shm[threadIdx.y][0] : zero;
    float direction_y = is_active_sg ? shm[threadIdx.y][1] : zero;
    float direction_z = is_active_sg ? shm[threadIdx.y][2] : zero;
    __syncthreads();
    if (threadIdx.y == 0) {
#pragma unroll
      for (int ii = 0; ii < 3; ii++) {
        int inner_idx = blockDim.x * ii + threadIdx.x;
        if (start_sg_idx * 3 + inner_idx < num_sg * 3) {
          shm[inner_idx / 3][inner_idx % 3] = intensity[start_sg_idx * 3 + inner_idx];
        }
      }
    }
    __syncthreads();
    // need to put zero because of some NaN bug on ampere
    float intensity_x = is_active_sg ? shm[threadIdx.y][0] : zero;
    float intensity_y = is_active_sg ? shm[threadIdx.y][1] : zero;
    float intensity_z = is_active_sg ? shm[threadIdx.y][2] : zero;
    __syncthreads();
    if (threadIdx.y == 0) {
#pragma unroll
      for (int ii = 0; ii < 3; ii++) {
        int inner_idx = blockDim.x * ii + threadIdx.x;
        if (start_sg_idx * 3 + inner_idx < num_sg * 3) {
          shm[inner_idx / 3][inner_idx % 3] = grad_out[start_sg_idx * 3 + inner_idx];
        }
      }
    }
    __syncthreads();
    float grad_re_x = is_active_sg ? shm[threadIdx.y][0] : zero;
    float grad_re_y = is_active_sg ? shm[threadIdx.y][1] : zero;
    float grad_re_z = is_active_sg ? shm[threadIdx.y][2] : zero;
    float sum_grad_intensity_x = zero;
    float sum_grad_intensity_y = zero;
    float sum_grad_intensity_z = zero;
    float sum_grad_prod1_x = zero;
    float sum_grad_prod1_y = zero;
    float sum_grad_prod1_z = zero;
    float sum_grad_lm = zero;
    if (start_sg_idx < num_sg) {
      for (int start_other_idx = 0; start_other_idx < num_other; start_other_idx += blockDim.x) {
        __syncthreads();
        int other_idx = start_other_idx + threadIdx.x;
        float other_sharp = other_sharpness[other_idx];
        if (threadIdx.y == 0) {
#pragma unroll
          for (int jj = 0; jj < 3; jj++) {
            int inner_idx = blockDim.x * jj + threadIdx.x;
            if (start_other_idx * 3 + inner_idx < num_other * 3) {
              shm[inner_idx / 3][inner_idx % 3] = other_direction[start_other_idx * 3 + inner_idx];
            }
          }
        }
        __syncthreads();
        float other_direction_x = shm[threadIdx.x][0];
        float other_direction_y = shm[threadIdx.x][1];
        float other_direction_z = shm[threadIdx.x][2];
        __syncthreads();
        if (threadIdx.y == 0) {
#pragma unroll
          for (int jj = 0; jj < 3; jj++) {
            int inner_idx = blockDim.x * jj + threadIdx.x;
            if (start_other_idx * 3 + inner_idx < num_other * 3) {
              shm[inner_idx / 3][inner_idx % 3] = other_intensity[start_other_idx * 3 + inner_idx];
            }
          }
        }

        float um_x = direction_x * sharp + other_direction_x * other_sharp;
        float um_y = direction_y * sharp + other_direction_y * other_sharp;
        float um_z = direction_z * sharp + other_direction_z * other_sharp;
        float um_length = sqrtf(um_x * um_x + um_y * um_y + um_z * um_z);
        float lm = sharp + other_sharp;
        float exp_val = expf(um_length - lm);
        float other_exp = expf(-2 * um_length);
        float other = 1. - other_exp;
        float exp_ratio = twopi * exp_val * other / um_length;
        __syncthreads();
        float intensity_prod_x = is_active_sg ? shm[threadIdx.x][0] * intensity_x : zero;
        float intensity_prod_y = is_active_sg ? shm[threadIdx.x][1] * intensity_y : zero;
        float intensity_prod_z = is_active_sg ? shm[threadIdx.x][2] * intensity_z : zero;
        volatile float grad_intensity_prod_x = is_active_sg ? grad_re_x * exp_ratio : zero;
        volatile float grad_intensity_prod_y = is_active_sg ? grad_re_y * exp_ratio : zero;
        volatile float grad_intensity_prod_z = is_active_sg ? grad_re_z * exp_ratio : zero;
        float grad_exp_ratio = \
            grad_re_x * intensity_prod_x + \
            grad_re_y * intensity_prod_y + \
            grad_re_z * intensity_prod_z;
        float grad_pre_mul_exp_mul = twopi * grad_exp_ratio / um_length;
        float grad_exp_val = grad_pre_mul_exp_mul * other;
        float grad_other = grad_pre_mul_exp_mul * exp_val;
        float grad_subtract = is_active_sg ? grad_exp_val * exp_val : zero;
        float grad_um_length = grad_subtract + grad_other * 2. * other_exp - \
                               grad_exp_ratio * exp_ratio / um_length;
        float scaled_grad_um_length = grad_um_length / um_length;
        float grad_um_x = is_active_sg ? scaled_grad_um_length * um_x : zero;
        float grad_um_y = is_active_sg ? scaled_grad_um_length * um_y : zero;
        float grad_um_z = is_active_sg ? scaled_grad_um_length * um_z : zero;

        if (other_idx < num_other) {
          sum_grad_intensity_x += grad_intensity_prod_x * shm[threadIdx.x][0];
          sum_grad_intensity_y += grad_intensity_prod_y * shm[threadIdx.x][1];
          sum_grad_intensity_z += grad_intensity_prod_z * shm[threadIdx.x][2];
          sum_grad_prod1_x += grad_um_x;
          sum_grad_prod1_y += grad_um_y;
          sum_grad_prod1_z += grad_um_z;
          sum_grad_lm += grad_subtract;
        }
        __syncthreads();
        shm[threadIdx.y][threadIdx.x] = grad_intensity_prod_x * intensity_x;
        __syncthreads();
        grad_intensity_prod_x = shm[threadIdx.x][threadIdx.y];
        __syncthreads();
        shm[threadIdx.y][threadIdx.x] = grad_intensity_prod_y * intensity_y;
        __syncthreads();
        grad_intensity_prod_y = shm[threadIdx.x][threadIdx.y];
        __syncthreads();
        shm[threadIdx.y][threadIdx.x] = grad_intensity_prod_z * intensity_z;
        __syncthreads();
        grad_intensity_prod_z = shm[threadIdx.x][threadIdx.y];
        __syncthreads();
        for (int offset = 8; offset > 0; offset /= 2) {
          grad_intensity_prod_x += __shfl_down_sync(0xffffffff, grad_intensity_prod_x, offset);
          grad_intensity_prod_y += __shfl_down_sync(0xffffffff, grad_intensity_prod_y, offset);
          grad_intensity_prod_z += __shfl_down_sync(0xffffffff, grad_intensity_prod_z, offset);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          shm[threadIdx.y][0] = grad_intensity_prod_x;
          shm[threadIdx.y][1] = grad_intensity_prod_y;
          shm[threadIdx.y][2] = grad_intensity_prod_z;
        }
        __syncthreads();
        if (threadIdx.y == 0) {
#pragma unroll
          for (int ii = 0; ii < 3; ii++) {
            int inner_idx = blockDim.x * ii + threadIdx.x;
            if (start_other_idx * 3 + inner_idx < num_other * 3) {
              atomicAdd(&grad_other_intensity_cache[blockIdx.x * 3 * num_other +
                                                    start_other_idx * 3 + inner_idx],
                        shm[inner_idx / 3][inner_idx % 3]);
            }
          }
        }
        __syncthreads();
        shm[threadIdx.y][threadIdx.x] = grad_um_x;
        __syncthreads();
        grad_um_x = shm[threadIdx.x][threadIdx.y];
        __syncthreads();
        shm[threadIdx.y][threadIdx.x] = grad_um_y;
        __syncthreads();
        grad_um_y = shm[threadIdx.x][threadIdx.y];
        __syncthreads();
        shm[threadIdx.y][threadIdx.x] = grad_um_z;
        __syncthreads();
        grad_um_z = shm[threadIdx.x][threadIdx.y];
        __syncthreads();
        for (int offset = 8; offset > 0; offset /= 2) {
          grad_um_x += __shfl_down_sync(0xffffffff, grad_um_x, offset);
          grad_um_y += __shfl_down_sync(0xffffffff, grad_um_y, offset);
          grad_um_z += __shfl_down_sync(0xffffffff, grad_um_z, offset);
        }
        __syncthreads();
        if (threadIdx.y == 0) {
          shm[threadIdx.x][3] = other_sharp;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          shm[threadIdx.y][0] = grad_um_x * shm[threadIdx.y][3];
          shm[threadIdx.y][1] = grad_um_y * shm[threadIdx.y][3];
          shm[threadIdx.y][2] = grad_um_z * shm[threadIdx.y][3];
        }
        __syncthreads();
        if (threadIdx.y == 0) {
#pragma unroll
          for (int ii = 0; ii < 3; ii++) {
            int inner_idx = blockDim.x * ii + threadIdx.x;
            if (start_other_idx * 3 + inner_idx < num_other * 3) {
              atomicAdd(&grad_other_direction_cache[blockIdx.x * 3 * num_other +
                                                    start_other_idx * 3 + inner_idx],
                        shm[inner_idx / 3][inner_idx % 3]);
            }
          }
        }
        __syncthreads();
        shm[threadIdx.y][threadIdx.x] = grad_subtract;
        __syncthreads();
        grad_subtract = shm[threadIdx.x][threadIdx.y];
        __syncthreads();
        for (int offset = 8; offset > 0; offset /= 2) {
          grad_subtract += __shfl_down_sync(0xffffffff, grad_subtract, offset);
        }
        if (threadIdx.y == 0) {
          shm[threadIdx.x][1] = other_direction_x;
          shm[threadIdx.x][2] = other_direction_y;
          shm[threadIdx.x][3] = other_direction_z;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
          shm[threadIdx.y][0] = \
              grad_um_x * shm[threadIdx.y][1] + \
              grad_um_y * shm[threadIdx.y][2] + \
              grad_um_z * shm[threadIdx.y][3] - \
              grad_subtract;
        }
        __syncthreads();
        if (threadIdx.y == 0) {
          if (other_idx < num_other) {
            atomicAdd(&grad_other_sharpness_cache[blockIdx.x * num_other + start_other_idx + threadIdx.x],
                      shm[threadIdx.x][0]);
          }
        }
      }
      __syncthreads();
      for (int offset = 8; offset > 0; offset /= 2) {
        sum_grad_intensity_x += __shfl_down_sync(0xffffffff, sum_grad_intensity_x, offset);
        sum_grad_intensity_y += __shfl_down_sync(0xffffffff, sum_grad_intensity_y, offset);
        sum_grad_intensity_z += __shfl_down_sync(0xffffffff, sum_grad_intensity_z, offset);
        sum_grad_prod1_x += __shfl_down_sync(0xffffffff, sum_grad_prod1_x, offset);
        sum_grad_prod1_y += __shfl_down_sync(0xffffffff, sum_grad_prod1_y, offset);
        sum_grad_prod1_z += __shfl_down_sync(0xffffffff, sum_grad_prod1_z, offset);
        sum_grad_lm += __shfl_down_sync(0xffffffff, sum_grad_lm, offset);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        shm[threadIdx.y][0] = sum_grad_intensity_x;
        shm[threadIdx.y][1] = sum_grad_intensity_y;
        shm[threadIdx.y][2] = sum_grad_intensity_z;
      }
      __syncthreads();
      if (threadIdx.y == 0) {
#pragma unroll
        for (int ii = 0; ii < 3; ii++) {
          int inner_idx = blockDim.x * ii + threadIdx.x;
          if (start_sg_idx * 3 + inner_idx < num_sg * 3) {
            grad_intensity[start_sg_idx * 3 + inner_idx] = shm[inner_idx / 3][inner_idx % 3];
          }
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        shm[threadIdx.y][0] = sum_grad_prod1_x * sharp;
        shm[threadIdx.y][1] = sum_grad_prod1_y * sharp;
        shm[threadIdx.y][2] = sum_grad_prod1_z * sharp;
      }
      __syncthreads();
      if (threadIdx.y == 0) {
#pragma unroll
        for (int ii = 0; ii < 3; ii++) {
          int inner_idx = blockDim.x * ii + threadIdx.x;
          if (start_sg_idx * 3 + inner_idx < num_sg * 3) {
            grad_direction[start_sg_idx * 3 + inner_idx] = shm[inner_idx / 3][inner_idx % 3];
          }
        }
      }
      __syncthreads();
      if (threadIdx.x == 0) {
        shm[threadIdx.y][0] = \
          sum_grad_prod1_x * direction_x + \
          sum_grad_prod1_y * direction_y + \
          sum_grad_prod1_z * direction_z - sum_grad_lm;
      }
      __syncthreads();
      if (threadIdx.y == 0) {
        int inner_idx = start_sg_idx + threadIdx.x;
        if (inner_idx < num_sg) {
          grad_sharpness[inner_idx] = shm[threadIdx.x][0];
        }
      }
    }
  }
}

void unbatched_reduced_sg_inner_product_forward_cuda_impl(
    const at::Tensor intensity,
    const at::Tensor direction,
    const at::Tensor sharpness,
    const at::Tensor other_intensity,
    const at::Tensor other_direction,
    const at::Tensor other_sharpness,
    at::Tensor output) {
  const int num_sg = intensity.size(0);
  const int num_other = other_intensity.size(0);

  const dim3 threads(32, 32, 1);
  const int blocks = (num_sg + 32 - 1) / 32;

  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(intensity));
  auto stream = at::cuda::getCurrentCUDAStream();

  // TODO(cfujitsang): extend support to double / fp16
  unbatched_reduced_sg_inner_product_forward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      intensity.data_ptr<float>(),
      direction.data_ptr<float>(),
      sharpness.data_ptr<float>(),
      other_intensity.data_ptr<float>(),
      other_direction.data_ptr<float>(),
      other_sharpness.data_ptr<float>(),
      num_sg,
      num_other,
      output.data_ptr<float>());
  AT_CUDA_CHECK(cudaGetLastError());
}

void unbatched_reduced_sg_inner_product_backward_cuda_impl(
    at::Tensor grad_out,
    at::Tensor intensity,
    at::Tensor direction,
    at::Tensor sharpness,
    at::Tensor other_intensity,
    at::Tensor other_direction,
    at::Tensor other_sharpness,
    at::Tensor grad_intensity,
    at::Tensor grad_direction,
    at::Tensor grad_sharpness,
    at::Tensor grad_other_intensity,
    at::Tensor grad_other_direction,
    at::Tensor grad_other_sharpness) {
  const int num_sg = intensity.size(0);
  const int num_other = other_intensity.size(0);
  int blocks = (num_sg + 16 - 1) / 16;
  if (blocks > 128)
    blocks = 128;

  at::Tensor grad_other_intensity_cache = at::zeros({blocks, num_other, 3},
                                                    grad_other_intensity.options());
  at::Tensor grad_other_direction_cache = at::zeros({blocks, num_other, 3},
                                                    grad_other_direction.options());
  at::Tensor grad_other_sharpness_cache = at::zeros({blocks, num_other},
                                                    grad_other_sharpness.options());
  const dim3 threads(16, 16, 1);
  const at::cuda::OptionalCUDAGuard device_guard(at::device_of(grad_out));
  auto stream = at::cuda::getCurrentCUDAStream();

  unbatched_reduced_sg_inner_product_backward_cuda_kernel<<<blocks, threads, 0, stream>>>(
      grad_out.data_ptr<float>(),
      intensity.data_ptr<float>(),
      direction.data_ptr<float>(),
      sharpness.data_ptr<float>(),
      other_intensity.data_ptr<float>(),
      other_direction.data_ptr<float>(),
      other_sharpness.data_ptr<float>(),
      num_sg,
      num_other,
      grad_intensity.data_ptr<float>(),
      grad_direction.data_ptr<float>(),
      grad_sharpness.data_ptr<float>(),
      grad_other_intensity_cache.data_ptr<float>(),
      grad_other_direction_cache.data_ptr<float>(),
      grad_other_sharpness_cache.data_ptr<float>());

  AT_CUDA_CHECK(cudaGetLastError());
  at::sum_out(grad_other_intensity, grad_other_intensity_cache, 0);
  at::sum_out(grad_other_direction, grad_other_direction_cache, 0);
  at::sum_out(grad_other_sharpness, grad_other_sharpness_cache, 0);
}

}  // namespace kaolin

