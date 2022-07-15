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
#include <c10/cuda/CUDAGuard.h>

#include "../../spc_math.h"
#include "../../spc_utils.cuh"

namespace kaolin {

template<typename scalar_t>
__global__ void query_cuda_kernel(
    const scalar_t* query_coords, // sample locations [n, 3]
    const int32_t* prefix_sum,
    const uchar* octree,
    const uint32_t level,
    int32_t* pidx,
    const int64_t n
) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    if (idx > n) return;
        
    int32_t resolution = 1 << level;

    for (int i=idx; i<n; i+=stride) {
        point_data point = make_point_data(
            floor(resolution * (query_coords[i*3 + 0] * 0.5 + 0.5)),
            floor(resolution * (query_coords[i*3 + 1] * 0.5 + 0.5)),
            floor(resolution * (query_coords[i*3 + 2] * 0.5 + 0.5))
        );
        pidx[i] = identify(point, level, prefix_sum, octree);
    }
}

template<typename scalar_t>
__global__ void query_multiscale_cuda_kernel(
    const scalar_t* query_coords, // sample locations [n, 3]
    const int32_t* prefix_sum,
    const uchar* octree,
    const uint32_t level,
    int32_t* pidx,
    const int64_t n
) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    if (idx > n) return;
    
    int32_t resolution = 1 << level;

    for (int i=idx; i<n; i+=stride) {
        point_data point = make_point_data(
            resolution * (query_coords[i*3 + 0] * 0.5 + 0.5),
            resolution * (query_coords[i*3 + 1] * 0.5 + 0.5),
            resolution * (query_coords[i*3 + 2] * 0.5 + 0.5)
        );
        identify_multiscale(point, level, prefix_sum, octree, pidx + (i*(level+1)));
    }
}

void query_cuda_impl(
    at::Tensor octree,
    at::Tensor prefix_sum,
    at::Tensor query_coords,
    at::Tensor pidx,
    uint32_t target_level) {

    int32_t num_query = query_coords.size(0);
    
    const int threads = 256;
    const int blocks = (num_query + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query_coords.type(), "query_cuda", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(pidx));
        auto stream = at::cuda::getCurrentCUDAStream();
        query_cuda_kernel<<<blocks, threads, 0, stream>>>(
            query_coords.data_ptr<scalar_t>(),
            prefix_sum.data_ptr<int>(),
            octree.data_ptr<uchar>(),
            target_level,
            pidx.data_ptr<int>(),
            num_query
        );
    }));
}

void query_multiscale_cuda_impl(
    at::Tensor octree,
    at::Tensor prefix_sum,
    at::Tensor query_coords,
    at::Tensor pidx,
    uint32_t target_level) {

    int32_t num_query = query_coords.size(0);
    
    const int threads = 256;
    const int blocks = (num_query + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query_coords.type(), "query_multiscale_cuda", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(pidx));
        auto stream = at::cuda::getCurrentCUDAStream();
        query_multiscale_cuda_kernel<<<blocks, threads, 0, stream>>>(
            query_coords.data_ptr<scalar_t>(),
            prefix_sum.data_ptr<int>(),
            octree.data_ptr<uchar>(),
            target_level,
            pidx.data_ptr<int>(),
            num_query
        );
    }));
}

}  // namespace kaolin
