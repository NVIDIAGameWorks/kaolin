// Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
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


#define CUB_NS_PREFIX namespace kaolin {
#define CUB_NS_POSTFIX }
#define CUB_NS_QUALIFIER ::kaolin::cub

#include <cub/device/device_scan.cuh>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "../../spc_math.h"
#include "../../utils.h"
#include "../../spc_utils.cuh"

#define THREADS_PER_BLOCK 64

namespace kaolin {

__global__ void scan_nodes_cuda_kernel(
    const uint num_bytes,
    const uint8_t *octree,
    uint *octrees_ptr) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_bytes)
    octrees_ptr[tidx] = __popc(octree[tidx]);
}

int scan_octrees_cuda_impl(
    at::Tensor octrees,
    at::Tensor lengths,
    at::Tensor num_childrens_per_node,
    at::Tensor prefix_sum,
    at::Tensor pyramid) {
  int batch_size = lengths.size(0);
  // get tensor data pointers
  uint8_t* octrees_ptr = octrees.data_ptr<uint8_t>();
  uint* num_childrens_per_node_ptr = reinterpret_cast<uint*>(num_childrens_per_node.data_ptr<int>());
  uint* prefix_sum_ptr = reinterpret_cast<uint*>(prefix_sum.data_ptr<int>());
  auto pyramid_acc = pyramid.accessor<int, 3>();
  
  void* temp_storage_ptr = NULL;
  uint64_t temp_storage_bytes = get_cub_storage_bytes(
        temp_storage_ptr, num_childrens_per_node_ptr, prefix_sum_ptr, num_childrens_per_node.size(0));
  at::Tensor temp_storage = at::zeros({(int64_t) temp_storage_bytes },
                                      octrees.options().dtype(at::kByte));
  temp_storage_ptr = (void*) temp_storage.data_ptr<uint8_t>();

  // octree, prefix_sum head pointers for each batch
  uint* EX0 = prefix_sum_ptr;
  uint8_t* O0 = octrees_ptr;

  int level;

  for (int batch = 0; batch < batch_size; batch++) {
    uint  osize = lengths[batch].item<int>();

    // compute bit counts of each octree byte
    scan_nodes_cuda_kernel<<< (osize + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(
        osize, O0, num_childrens_per_node_ptr);
    // compute inclusive sum of bit counts
    CubDebugExit(cub::DeviceScan::InclusiveSum(
        temp_storage_ptr, temp_storage_bytes, num_childrens_per_node_ptr,
        EX0, osize));

    uint currSum, prevSum = 0;

    pyramid_acc[batch][0][0] = 1;
    pyramid_acc[batch][1][0] = 0;
    pyramid_acc[batch][1][1] = 1;

    level = 1;
    while (pyramid_acc[batch][1][level] <= osize) {
      cudaMemcpy(&currSum, EX0 + prevSum, sizeof(uint), cudaMemcpyDeviceToHost);
      AT_CUDA_CHECK(cudaGetLastError());

      pyramid_acc[batch][0][level] = currSum - prevSum;
      pyramid_acc[batch][1][level+1] = pyramid_acc[batch][0][level] + pyramid_acc[batch][1][level];

      prevSum = currSum;
      level++;
    }

    O0 += osize;
    EX0 += osize;
  }

  AT_CUDA_CHECK(cudaGetLastError());

  return level-1;
}

}  // namespace kaolin
