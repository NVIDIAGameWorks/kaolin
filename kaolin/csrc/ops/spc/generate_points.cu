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


#define CUB_NS_PREFIX namespace kaolin {
#define CUB_NS_POSTFIX }

#include <cub/device/device_scan.cuh>

#include <ATen/ATen.h>

#include "../../spc_math.h"
#include "../../utils.h"

#define THREADS_PER_BLOCK 64

namespace kaolin {

using namespace cub;

__global__ void NodesToMortonX(
    const uint Psize,
    const uchar* Odata,
    const uint* PrefixSum,
    const morton_code* MdataIn,
    morton_code* MdataOut) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < Psize) {
     uchar bits = Odata[tidx];
     morton_code code = MdataIn[tidx];
     int addr = PrefixSum[tidx];

     for (int i = 7; i >= 0; i--) {
       if (bits&(0x1 << i))
         MdataOut[addr--] = 8 * code + i;
     }
  }
}

__global__ void MortonToPointX(
    const uint Psize,
    morton_code* Mdata,
    point_data* Pdata) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < Psize)
    Pdata[tidx] = ToPoint(Mdata[tidx]);
}

void generate_points_cuda_kernel_launch(
    at::Tensor octrees,
    at::Tensor points,
    at::Tensor morton,
    at::Tensor pyramids,
    at::Tensor prefix_sum) {
  int batch_size = pyramids.size(0);
  int level = pyramids.size(2) - 2;

  point_data* P0 = reinterpret_cast<point_data*>(points.data_ptr<int16_t>());
  morton_code* M0 = reinterpret_cast<morton_code*>(morton.data_ptr<int64_t>());
  uint* EX0 = reinterpret_cast<uint*>(prefix_sum.data_ptr<int>());
  uchar* O0 = octrees.data_ptr<uint8_t>();
  uint* h0 = reinterpret_cast<uint*>(pyramids.data_ptr<int>());
  int l;

  for (int batch = 0; batch < batch_size; batch++) {
    uint* Pmid = h0;
    uint* PmidSum = h0 + level + 2;

    morton_code*  M = M0;
    uchar*      O = O0;
    uint*       S = EX0 + 1;
    uint      osize = PmidSum[level];

    morton_code m0 = 0;
    cudaMemcpy(M, &m0, sizeof(morton_code), cudaMemcpyHostToDevice);
    CUDA_CHECK(cudaGetLastError());

    l = 0;
    while (l < level) {
      int Lsize = Pmid[l++];
      NodesToMortonX<<<(Lsize + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK,
                       THREADS_PER_BLOCK>>>(Lsize, O, S, M, M0);
      O += Lsize;
      S += Lsize;
      M += Lsize;
    }

    uint totalPoints = PmidSum[l + 1];

    MortonToPointX<<<(totalPoints + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK,
                     THREADS_PER_BLOCK>>>(totalPoints, M0, P0);
    CUDA_CHECK(cudaGetLastError());

    P0 += totalPoints;
    O0 += osize;
    EX0 += (osize + 1);
    h0 += 2 * (level + 2);
  }

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace kaolin
