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

__global__ void points_to_corners_cuda_kernel(
    const point_data* point,
    point_data* corners,
    const int64_t num_points
){
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;
    if (idx > num_points) return;

    for (int64_t i=idx; i<num_points; i+=stride) { 
#       pragma unroll
        for (int j=0; j<8; ++j) {
            corners[i*8 + j].x = point[i].x + ((j & 4) >> 2);
            corners[i*8 + j].y = point[i].y + ((j & 2) >> 1);
            corners[i*8 + j].z = point[i].z + ((j & 1) >> 0);
        }
    }
}

__global__ void coords_to_trilinear_cuda_kernel(
    const float3* coords,
    const point_data* points,
    float* coeffs,
    const int64_t num_coords
){
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;
    if (idx > num_coords) return;

    for (int64_t i=idx; i<num_coords; i+=stride) { 
        float3 x_ = make_float3(coords[i].x - points[i].x, coords[i].y - points[i].y, coords[i].z - points[i].z);
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

__global__ void coords_to_trilinear_jacobian_cuda_kernel(
    const float3* coords,
    float* jacobians,
    const int64_t num_coords
){
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;
    if (idx > num_coords) return;

    for (int64_t i=idx; i<num_coords; i+=stride) { 
        // TODO(ttakikawa): This should really be wrt points to make it consistent with other API
        float3 x_ = make_float3(coords[i].x, coords[i].y, coords[i].z);
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

void morton_to_points_cuda_impl(at::Tensor morton_codes, at::Tensor points) {
    int64_t num_points = morton_codes.size(0);
    morton_to_points_cuda_kernel<<<(num_points + 1023) / 1024, 1024>>>(
        reinterpret_cast<morton_code*>(morton_codes.data_ptr<int64_t>()),
        reinterpret_cast<point_data*>(points.data_ptr<short>()),
        num_points);
}

void points_to_morton_cuda_impl(at::Tensor points, at::Tensor morton_codes) {
    int64_t num_points = points.size(0);
    points_to_morton_cuda_kernel<<<(num_points + 1023) / 1024, 1024>>>(
        reinterpret_cast<point_data*>(points.data_ptr<short>()),
        reinterpret_cast<morton_code*>(morton_codes.data_ptr<int64_t>()),
        num_points);
}

void coords_to_trilinear_cuda_impl(
    at::Tensor coords,
    at::Tensor points,
    at::Tensor coeffs
) {
    int64_t num_coords = coords.size(0);
    coords_to_trilinear_cuda_kernel<<<(num_coords + 1023) / 1024, 1024>>>(
        reinterpret_cast<float3*>(coords.data_ptr<float>()),
        reinterpret_cast<point_data*>(points.data_ptr<short>()),
        coeffs.data_ptr<float>(),
        num_coords
    );
}

void coords_to_trilinear_jacobian_cuda_impl(
    at::Tensor coords // N x 3 tensor of local space coordinates
) {
    int64_t num_coords = coords.size(0);
    at::Tensor jacobians = at::zeros({num_coords, 8, 3}, at::device(at::kCUDA).dtype(at::kFloat));
    coords_to_trilinear_jacobian_cuda_kernel<<<(num_coords + 1023) / 1024, 1024>>>(
        reinterpret_cast<float3*>(coords.data_ptr<float>()),
        jacobians.data_ptr<float>(),
        num_coords
    );
}

void points_to_corners_cuda_impl(
    at::Tensor points,
    at::Tensor corners
) {
    int64_t num_points = points.size(0);
    points_to_corners_cuda_kernel<<<(num_points + 1023) / 1024, 1024>>>(
        reinterpret_cast<point_data*>(points.data_ptr<short>()),
        reinterpret_cast<point_data*>(corners.data_ptr<short>()),
        num_points
    );
}

} // namespace kaolin

