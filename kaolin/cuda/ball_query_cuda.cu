// Copyright (c) 2017, Geometric Computation Group of Stanford University

// The MIT License (MIT)

// Copyright (c) 2017 Charles R. Qi

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_util.h"
#include <curand.h>
#include <curand_kernel.h>

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
__global__ void query_ball_point_kernel(int b, int n, int m, float radius,
                                        int nsample,
                                        const float *__restrict__ new_xyz,
                                        const float *__restrict__ xyz,
                                        int *__restrict__ idx) {
  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    for (int k = 0, cnt = 0; k < n && cnt < nsample; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
          }
        }
        idx[j * nsample + cnt] = k;
        ++cnt;
      }
    }
  }
}

void query_ball_point_kernel_launcher(int b, int n, int m, float radius,
                                      int nsample, const float *new_xyz,
                                      const float *xyz, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  query_ball_point_kernel<<<b, opt_n_threads(m), 0, stream>>>(
      b, n, m, radius, nsample, new_xyz, xyz, idx);

  CUDA_CHECK_ERRORS();
}

// input: new_xyz(b, m, 3) xyz(b, n, 3)
// output: idx(b, m, nsample)
__global__ void query_ball_random_point_kernel(
    int seed, curandState *rand_states, int b, int n, int m, float radius,
    int nsample, const float *__restrict__ new_xyz,
    const float *__restrict__ xyz, int *__restrict__ idx) {

  int id = threadIdx.x + blockIdx.x * blockDim.x;
  curandState *local_state = rand_states + id;

  // TODO: optimize: curand_init is slow.
  curand_init(seed, id, 0, local_state);
  // // A potentially faster but less accurate version:
  // curand_init(seed + id * 1337, 0, 0, &rand_states[id]);

  int batch_index = blockIdx.x;
  xyz += batch_index * n * 3;
  new_xyz += batch_index * m * 3;
  idx += m * nsample * batch_index;

  int index = threadIdx.x;
  int stride = blockDim.x;

  float radius2 = radius * radius;
  for (int j = index; j < m; j += stride) {
    float new_x = new_xyz[j * 3 + 0];
    float new_y = new_xyz[j * 3 + 1];
    float new_z = new_xyz[j * 3 + 2];
    for (int k = 0, cnt = 0; k < n; ++k) {
      float x = xyz[k * 3 + 0];
      float y = xyz[k * 3 + 1];
      float z = xyz[k * 3 + 2];
      float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                 (new_z - z) * (new_z - z);
      if (d2 < radius2) {
        if (cnt == 0) {
          for (int l = 0; l < nsample; ++l) {
            idx[j * nsample + l] = k;
          }
        } else if (cnt < nsample) {
          idx[j * nsample + cnt] = k;
        } else {
          unsigned int r = curand_uniform(local_state) * (cnt + 1);
          if (r < nsample) {
            idx[j * nsample + r] = k;
          }
        }
        ++cnt;
      }
    }
  }
}

void query_ball_random_point_kernel_launcher(int seed, int b, int n, int m,
                                             float radius, int nsample,
                                             const float *new_xyz,
                                             const float *xyz, int *idx) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  int grid_dim = b;
  int block_dim = opt_n_threads(m);
  int num_threads = grid_dim * block_dim;

  curandState *rand_states;
  cudaMalloc((void **)&rand_states, num_threads * sizeof(curandState));

  query_ball_random_point_kernel<<<grid_dim, block_dim, 0, stream>>>(
      seed, rand_states, b, n, m, radius, nsample, new_xyz, xyz, idx);

  cudaFree(rand_states);

  CUDA_CHECK_ERRORS();
}

// input: points(b, c, n) idx(b, npoints, nsample)
// output: out(b, c, npoints, nsample)
__global__ void gather_by_index_kernel(int b, int c, int n, int npoints,
                                       int nsample,
                                       const float *__restrict__ points,
                                       const int *__restrict__ idx,
                                       float *__restrict__ out) {
  int batch_index = blockIdx.x;
  points += batch_index * n * c;
  idx += batch_index * npoints * nsample;
  out += batch_index * npoints * nsample * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;
    const int j = i % npoints;
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k];
      out[(l * npoints + j) * nsample + k] = points[l * n + ii];
    }
  }
}

void gather_by_index_kernel_launcher(int b, int c, int n, int npoints,
                                     int nsample, const float *points,
                                     const int *idx, float *out) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  gather_by_index_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
      b, c, n, npoints, nsample, points, idx, out);

  CUDA_CHECK_ERRORS();
}

// input: grad_out(b, c, npoints, nsample), idx(b, npoints, nsample)
// output: grad_points(b, c, n)
__global__ void gather_by_index_grad_kernel(int b, int c, int n, int npoints,
                                            int nsample,
                                            const float *__restrict__ grad_out,
                                            const int *__restrict__ idx,
                                            float *__restrict__ grad_points) {
  int batch_index = blockIdx.x;
  grad_out += batch_index * npoints * nsample * c;
  idx += batch_index * npoints * nsample;
  grad_points += batch_index * n * c;

  const int index = threadIdx.y * blockDim.x + threadIdx.x;
  const int stride = blockDim.y * blockDim.x;
  for (int i = index; i < c * npoints; i += stride) {
    const int l = i / npoints;
    const int j = i % npoints;
    for (int k = 0; k < nsample; ++k) {
      int ii = idx[j * nsample + k];
      atomicAdd(grad_points + l * n + ii,
                grad_out[(l * npoints + j) * nsample + k]);
    }
  }
}

void gather_by_index_grad_kernel_launcher(int b, int c, int n, int npoints,
                                          int nsample, const float *grad_out,
                                          const int *idx, float *grad_points) {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  gather_by_index_grad_kernel<<<b, opt_block_config(npoints, c), 0, stream>>>(
      b, c, n, npoints, nsample, grad_out, idx, grad_points);

  CUDA_CHECK_ERRORS();
}
