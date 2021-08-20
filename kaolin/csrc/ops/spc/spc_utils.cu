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
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "../../spc_math.h"

__global__ void spc_point2morton_kernel(
    const point_data* points,   
    morton_code* morton_codes,
    const int64_t num_points
){
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;
    if (idx > num_points) return;

    for (int64_t i=idx; i<num_points; i+=stride) {
        morton_codes[i] = ToMorton(points[i]);
    }
}

__global__ void spc_morton2point_kernel(
    const morton_code* morton_codes,
    point_data* points,   
    const int64_t num_points
){
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;
    if (idx > num_points) return;

    for (int64_t i=idx; i<num_points; i+=stride) {
        points[i] = ToPoint(morton_codes[i]);
    }
}

at::Tensor spc_morton2point(at::Tensor morton_codes) {
    int64_t num_points = morton_codes.size(0);
    at::Tensor points = at::zeros({num_points, 3}, at::device(at::kCUDA).dtype(at::kShort));
    spc_morton2point_kernel << <(num_points + 1023) / 1024, 1024 >> > (
        reinterpret_cast<morton_code*>(morton_codes.data_ptr<int64_t>()),
        reinterpret_cast<point_data*>(points.data_ptr<short>()),
        num_points);
    return points;
}

at::Tensor spc_point2morton(at::Tensor points) {
    int64_t num_points = points.size(0);
    at::Tensor morton_codes = at::zeros({num_points}, at::device(at::kCUDA).dtype(at::kLong));
    spc_point2morton_kernel << <(num_points + 1023) / 1024, 1024 >> > (
        reinterpret_cast<point_data*>(points.data_ptr<short>()),
        reinterpret_cast<morton_code*>(morton_codes.data_ptr<int64_t>()),
        num_points);
    return morton_codes;
}

__global__ void spc_point2coeff_kernel(
    const float3* x,
    const point_data* pts,
    float* coeffs,
    const int64_t num
){
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;
    if (idx > num) return;

    for (int64_t i=idx; i<num; i+=stride) { 
        float3 x_ = make_float3(x[i].x - pts[i].x, x[i].y - pts[i].y, x[i].z - pts[i].z);
        float3 _x = make_float3(1.0 - x_.x, 1.0 - x_.y, 1.0 - x_.z);
        coeffs[i * 8 + 0] = _x.x * _x.y * _x.z;
        coeffs[i * 8 + 1] = _x.x * _x.y * x_.z;
        coeffs[i * 8 + 2] = _x.x * x_.y * _x.z;
        coeffs[i * 8 + 3] = _x.x * x_.y * x_.z;
        coeffs[i * 8 + 4] = x_.x * _x.y * _x.z;
        coeffs[i * 8 + 5] = x_.x * _x.y * x_.z;
        coeffs[i * 8 + 6] = x_.x * x_.y * _x.z;
        coeffs[i * 8 + 7] = x_.x * x_.y * x_.z;
    }
}

at::Tensor spc_point2coeff(
    at::Tensor x,
    at::Tensor pts 
) {
    int64_t num = x.size(0);
    at::Tensor coeffs = at::zeros({num, 8}, at::device(at::kCUDA).dtype(at::kFloat));
    spc_point2coeff_kernel<<<(num + 1023) / 1024, 1024>>>(
        reinterpret_cast<float3*>(x.data_ptr<float>()),
        reinterpret_cast<point_data*>(pts.data_ptr<short>()),
        coeffs.data_ptr<float>(),
        num
    );
    return coeffs;
}

__global__ void spc_point2jacobian_kernel(
    const float3* x,
    float* jacobians,
    const int64_t num
){
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;
    if (idx > num) return;

    for (int64_t i=idx; i<num; i+=stride) { 
        float3 x_ = make_float3(x[i].x, x[i].y, x[i].z);
        float3 _x = make_float3(x_.x-1.0, x_.y-1.0, x_.z-1.0);

        jacobians[i * 24 + 0] = -1.0 * _x.y * _x.z;
        jacobians[i * 24 + 1] = -1.0 * _x.x * _x.z;
        jacobians[i * 24 + 2] = -1.0 * _x.x * _x.y;

        jacobians[i * 24 + 3] = _x.y * x_.z;
        jacobians[i * 24 + 4] = _x.x * x_.z;
        jacobians[i * 24 + 5] = _x.x * _x.y;
        
        jacobians[i * 24 + 6] = x_.y * _x.z;
        jacobians[i * 24 + 7] = _x.x * _x.z;
        jacobians[i * 24 + 8] = _x.x * x_.y;

        jacobians[i * 24 + 9]  = -1.0 * x_.y * x_.z;
        jacobians[i * 24 + 10] = -1.0 * _x.x * x_.z;
        jacobians[i * 24 + 11] = -1.0 * _x.x * x_.y;

        jacobians[i * 24 + 12] = _x.y * _x.z;
        jacobians[i * 24 + 13] = x_.x * _x.z;
        jacobians[i * 24 + 14] = x_.x * _x.y;

        jacobians[i * 24 + 15] = -1.0 * _x.y * x_.z;
        jacobians[i * 24 + 16] = -1.0 * x_.x * x_.z; 
        jacobians[i * 24 + 17] = -1.0 * x_.x * _x.y;
        
        jacobians[i * 24 + 18] = -1.0 * x_.y * _x.z;
        jacobians[i * 24 + 19] = -1.0 * x_.x * _x.z;
        jacobians[i * 24 + 20] = -1.0 * x_.x * x_.y;

        jacobians[i * 24 + 21] = x_.y * x_.z;
        jacobians[i * 24 + 22] = x_.x * x_.z;
        jacobians[i * 24 + 23] = x_.x * x_.y;
    }
}

at::Tensor spc_point2jacobian(
    at::Tensor x // N x 3 tensor of local space coordinates
) {
    int64_t num = x.size(0);
    at::Tensor jacobians = at::zeros({num, 8, 3}, at::device(at::kCUDA).dtype(at::kFloat));
    spc_point2jacobian_kernel<<<(num+1023)/1024, 1024>>>(
        reinterpret_cast<float3*>(x.data_ptr<float>()),
        jacobians.data_ptr<float>(),
        num
    );
    return jacobians;
}

__global__ void spc_point2corners_kernel(
    const point_data* point,
    point_data* corners,
    const int64_t num
){
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;
    if (idx > num) return;

    for (int64_t i=idx; i<num; i+=stride) { 
        for (int j=0; j<8; ++j) {
            corners[i*8 + j].x = point[i].x + ((j & 4) >> 2);
            corners[i*8 + j].y = point[i].y + ((j & 2) >> 1);
            corners[i*8 + j].z = point[i].z + ((j & 1) >> 0);
        }
    }
}

at::Tensor spc_point2corners(
    at::Tensor points
) {
    int64_t num = points.size(0);
    at::Tensor corners = at::zeros({num, 8, 3}, at::device(at::kCUDA).dtype(at::kShort));
    spc_point2corners_kernel<<<(num+1023)/1024, 1024>>>(
        reinterpret_cast<point_data*>(points.data_ptr<short>()),
        reinterpret_cast<point_data*>(corners.data_ptr<short>()),
        num
    );
    return corners;
}


