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

#include "../../spc_math.h"
#include "../../spc_utils.cuh"

namespace kaolin {

__global__ void query_cuda_kernel(
    const point_data* query_points, // sample locations [n, 4]
    const uint* prefix_sum,
    const uchar* octree,
    const uint level,
    uint* pidx,
    const int n
) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    if (idx > n) return;

    for (int i=idx; i<n; i+=stride) {
        pidx[i] = identify(query_points[i], level, prefix_sum, octree);
    }
}

void query_cuda_impl(
    at::Tensor octree,
    at::Tensor prefix_sum,
    at::Tensor query_points,
    at::Tensor pidx,
    uint target_level) {

    int num_query = query_points.size(0);
    
    const int threads = 256;
    const int blocks = (num_query + threads - 1) / threads;
    query_cuda_kernel<<<blocks, threads>>>(
        reinterpret_cast<point_data*>(query_points.data_ptr<short>()),
        reinterpret_cast<uint*>(prefix_sum.data_ptr<int>()),
        octree.data_ptr<uchar>(),
        target_level,
        reinterpret_cast<uint*>(pidx.data_ptr<int>()),
        num_query
    );
}

}  // namespace kaolin
