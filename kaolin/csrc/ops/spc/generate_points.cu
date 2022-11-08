// Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
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

#include "../../spc_math.h"
#include "../../utils.h"
#include "../../spc_utils.cuh"

#define THREADS_PER_BLOCK 64

namespace kaolin {

void generate_points_cuda_impl(
    at::Tensor octrees,
    at::Tensor points,
    at::Tensor morton,
    at::Tensor pyramid,
    at::Tensor prefix_sum) {
  int batch_size = pyramid.size(0);
  int max_level = pyramid.size(2) - 2;

  point_data* points_ptr = reinterpret_cast<point_data*>(points.data_ptr<int16_t>());
  morton_code* morton_ptr = reinterpret_cast<morton_code*>(morton.data_ptr<int64_t>());
  uint* prefix_sum_ptr = reinterpret_cast<uint*>(prefix_sum.data_ptr<int>());
  uchar* octree_ptr = octrees.data_ptr<uint8_t>();
  uint* pyramid_ptr = reinterpret_cast<uint*>(pyramid.data_ptr<int>());
  int l;

  for (int batch = 0; batch < batch_size; batch++) {
    uint* curr_pyramid_ptr = pyramid_ptr;
    uint* curr_pyramid_sum_ptr = pyramid_ptr + max_level + 2;

    morton_code* curr_morton_ptr = morton_ptr;
    uchar*       curr_octree_ptr = octree_ptr;
    uint*        curr_prefix_sum_ptr = prefix_sum_ptr + 1;
    uint         osize = curr_pyramid_sum_ptr[max_level];

    morton_code m0 = 0;
    cudaMemcpy(curr_morton_ptr, &m0, sizeof(morton_code), cudaMemcpyHostToDevice);
    AT_CUDA_CHECK(cudaGetLastError());

    l = 0;
    while (l < max_level) {
      int points_in_level = curr_pyramid_ptr[l++];
      nodes_to_morton_cuda_kernel<<<(points_in_level + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK,
                                    THREADS_PER_BLOCK>>>(curr_octree_ptr, curr_prefix_sum_ptr, 
                                            curr_morton_ptr, morton_ptr, points_in_level);
      curr_octree_ptr += points_in_level;
      curr_prefix_sum_ptr += points_in_level;
      curr_morton_ptr += points_in_level;
    }

    uint total_points = curr_pyramid_sum_ptr[l + 1];

    morton_to_points_cuda_kernel<<<(total_points + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK,
                                   THREADS_PER_BLOCK>>>(morton_ptr, points_ptr, total_points);
    AT_CUDA_CHECK(cudaGetLastError());

    points_ptr += total_points;
    octree_ptr += osize;
    prefix_sum_ptr += (osize + 1);
    pyramid_ptr += 2 * (max_level + 2);
  }

  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace kaolin
