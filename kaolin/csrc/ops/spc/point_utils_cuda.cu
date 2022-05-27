// Copyright (c) 2021,22 NVIDIA CORPORATION & AFFILIATES.
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

template<typename scalar_t>
__global__ void interpolate_trilinear_cuda_kernel(
    const float3* coords, // num_voxels, num_samples, 3
    const int32_t* pidx, // num_voxels
    const point_data* points, // point_hierarchy_size, 3
    const int32_t* trinkets, // point_hierarchy_size, 8
    const scalar_t* feature_in, // num_feats, feature_dim
    scalar_t* feature_out, // num_voxels, num_samples, feature_dim
    const int64_t feature_dim, 
    const int32_t resolution, 
    const int64_t num_samples,
    const int64_t num
){
    int64_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x*gridDim.x;
    if (idx > num) return;

    for (int32_t i=idx; i<num; i+=stride) { 

        int32_t _i = pidx[i / num_samples];

        if (_i > -1) {
            point_data point = points[_i];
            int32_t trinket[8]; 
            
            memcpy(&trinket, trinkets + (_i*8), sizeof(int32_t)*8);

            float3 x_ = make_float3(resolution * (coords[i].x * 0.5 + 0.5) - point.x, 
                                    resolution * (coords[i].y * 0.5 + 0.5) - point.y, 
                                    resolution * (coords[i].z * 0.5 + 0.5) - point.z);
            float3 _x = make_float3(1.0 - x_.x, 1.0 - x_.y, 1.0 - x_.z);
            
            float c000 = _x.x * _x.y * _x.z;
            float c001 = _x.x * _x.y * x_.z;
            float c010 = _x.x * x_.y * _x.z;
            float c011 = _x.x * x_.y * x_.z;
            float c100 = x_.x * _x.y * _x.z;
            float c101 = x_.x * _x.y * x_.z;
            float c110 = x_.x * x_.y * _x.z;
            float c111 = x_.x * x_.y * x_.z;
            
            for (uint64_t j=0; j<feature_dim; ++j) {
                scalar_t feat =
                    feature_in[trinket[0]*feature_dim+j] * c000 + 
                    feature_in[trinket[1]*feature_dim+j] * c001 + 
                    feature_in[trinket[2]*feature_dim+j] * c010 + 
                    feature_in[trinket[3]*feature_dim+j] * c011 + 
                    feature_in[trinket[4]*feature_dim+j] * c100 + 
                    feature_in[trinket[5]*feature_dim+j] * c101 + 
                    feature_in[trinket[6]*feature_dim+j] * c110 + 
                    feature_in[trinket[7]*feature_dim+j] * c111; 
                feature_out[i*feature_dim+j] = feat;
            }
        }
    }
}

template<typename scalar_t>
__global__ void coords_to_trilinear_cuda_kernel(
    const float3* coords,
    const point_data* points,
    scalar_t* coeffs,
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

void interpolate_trilinear_cuda_impl(
    at::Tensor coords,
    at::Tensor pidx,
    at::Tensor points,
    at::Tensor trinkets,
    at::Tensor feats_in,
    at::Tensor feats_out,
    int32_t level
){
    int64_t num_voxels = coords.size(0);
    int64_t num_samples = coords.size(1);
    int64_t feat_dim = feats_in.size(1);
    int64_t num = num_voxels * num_samples;
    int32_t resolution = 1 << level;

    int num_threads = 128;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.type(), "interpolate_trilinear_cuda", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
        auto stream = at::cuda::getCurrentCUDAStream();
        interpolate_trilinear_cuda_kernel<scalar_t><<<(num + num_threads - 1) / num_threads, num_threads, 0, stream>>>(
            reinterpret_cast<float3*>(coords.data_ptr<float>()),
            pidx.data_ptr<int32_t>(),
            reinterpret_cast<point_data*>(points.data_ptr<short>()),
            trinkets.data_ptr<int32_t>(),
            feats_in.data_ptr<scalar_t>(),
            feats_out.data_ptr<scalar_t>(),
            feat_dim,
            resolution,
            num_samples,
            num
        );
    }));
}


void coords_to_trilinear_cuda_impl(
    at::Tensor coords,
    at::Tensor points,
    at::Tensor coeffs
) {
    int64_t num_coords = coords.size(0);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(coeffs.type(), "coords_to_trilinear_cuda", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(coeffs));
        auto stream = at::cuda::getCurrentCUDAStream();
        coords_to_trilinear_cuda_kernel<scalar_t><<<(num_coords + 1023) / 1024, 1024, 0, stream>>>(
            reinterpret_cast<float3*>(coords.data_ptr<float>()),
            reinterpret_cast<point_data*>(points.data_ptr<short>()),
            coeffs.data_ptr<scalar_t>(),
            num_coords
        );
    }));
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

