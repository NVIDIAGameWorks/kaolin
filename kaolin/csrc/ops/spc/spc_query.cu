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


#include <stdlib.h>
#include <stdio.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

#include "spc_query.h"

#include "../../spc_math.h"

__device__ int Identify(
    const point_data    k,
    const uint          Level,
    const uint* PrefixSum,
    const uchar* Oroot)
{
    int maxval = (0x1 << Level) - 1; // seems you could do this better using Morton codes
    // Check if in bounds
    if (k.x < 0 || k.y < 0 || k.z < 0 || k.x > maxval || k.y > maxval || k.z > maxval) {
        return -1;
    }
    int ord = 0;
    for (uint l = 0; l < Level; l++)
    {
        uint depth = Level - l - 1;
        uint mask = (0x1 << depth);
        uint child_idx = ((mask & k.x) << 2 | (mask & k.y) << 1 | (mask & k.z)) >> depth;
        uchar bits = Oroot[ord];
        // if bit set, keep going
        if (bits & (0x1 << child_idx))
        {
            // count set bits up to child - inclusive sum
            uint cnt = __popc(bits & ((0x2 << child_idx) - 1));
            ord = PrefixSum[ord] + cnt;
            if (depth == 0) {
                return ord;
            }
        }
        else {
            return -1;
        }
    }
    return ord; // only if called with Level=0
}

__global__ void spc_query_kernel(
    const point_data* query_points, // sample locations [n, 4]
    const uint* PrefixSum,
    const uchar* Oroot,
    const uint Level,
    uint* spc_idxes,
    const int n
) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    if (idx > n) return;

    for (int i=idx; i<n; i+=stride) {
        spc_idxes[i] = Identify(query_points[i], Level, PrefixSum, Oroot);
    }
}

at::Tensor spc_query(
    at::Tensor octree,
    at::Tensor points,
    at::Tensor pyramid, 
    at::Tensor prefixsum,
    at::Tensor query_points,
    uint targetLevel) {

    int num_query = query_points.size(0);
    
    at::Tensor hit_idx = at::zeros({ num_query }, octree.options().dtype(at::kInt));

    const int threads = 256;
    const int blocks = (num_query + threads - 1) / threads;
    spc_query_kernel<<<blocks, threads>>>(
        reinterpret_cast<point_data*>(query_points.data_ptr<short>()),
        reinterpret_cast<uint*>(prefixsum.data_ptr<int>()),
        octree.data_ptr<uchar>(),
        targetLevel,
        reinterpret_cast<uint*>(hit_idx.data_ptr<int>()),
        num_query
    );
    return hit_idx;
}

