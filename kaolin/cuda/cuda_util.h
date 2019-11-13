// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

// TODO: refactor: change other extensions to use this file

// #include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define TOTAL_THREADS 512

inline int opt_n_threads(int work_size) {
  const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);

  return max(min(1 << pow_2, TOTAL_THREADS), 1);
}

inline dim3 opt_block_config(int x, int y) {
  const int x_threads = opt_n_threads(x);
  const int y_threads =
      max(min(opt_n_threads(y), TOTAL_THREADS / x_threads), 1);
  dim3 block_config(x_threads, y_threads, 1);

  return block_config;
}

#define CUDA_CHECK_ERRORS()                                                    \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    if (cudaSuccess != err) {                                                  \
      fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",           \
              cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__,          \
              __FILE__);                                                       \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)
