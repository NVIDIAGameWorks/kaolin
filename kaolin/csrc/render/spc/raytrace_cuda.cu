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

#define CUB_NS_PREFIX namespace kaolin {
#define CUB_NS_POSTFIX }
#define CUB_NS_QUALIFIER ::kaolin::cub

#include <stdio.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#ifdef EXPERIMENTAL
    #include <ATen/native/cuda/KernelUtils.cuh>
#else 
    #include <THC/THCAtomics.cuh>
#endif
// TODO(ttakikawa): newer versions of PyTorch will migrate to <ATen/cuda/Atomics.cuh>. 
// How do we manage these dependencies?

#define CUB_STDERR
#include <cub/device/device_scan.cuh>

#include "../../spc_math.h"
#include "../../spc_utils.cuh"
#include "spc_render_utils.cuh"

namespace kaolin {

using namespace at::indexing;

#define RT_NUM_THREADS 1024

////////////////////////////////////////////////////////////////////////////////////////////////
/// Constants
////////////////////////////////////////////////////////////////////////////////////////////////

__constant__ uint8_t VOXEL_ORDER[8][8] = {
    { 0, 1, 2, 4, 3, 5, 6, 7 },
    { 1, 0, 3, 5, 2, 4, 7, 6 },
    { 2, 0, 3, 6, 1, 4, 7, 5 },
    { 3, 1, 2, 7, 0, 5, 6, 4 },
    { 4, 0, 5, 6, 1, 2, 7, 3 },
    { 5, 1, 4, 7, 0, 3, 6, 2 },
    { 6, 2, 4, 7, 0, 3, 5, 1 },
    { 7, 3, 5, 6, 1, 2, 4, 0 }
};

////////////////////////////////////////////////////////////////////////////////////////////////
/// Kernels
////////////////////////////////////////////////////////////////////////////////////////////////

// This function will initialize the nuggets array with each ray pointing to the octree root
__global__ void
init_nuggets_cuda_kernel(
    const uint num, 
    uint2* nuggets) {
  
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num) {
    nuggets[tidx].x = tidx; // ray idx
    nuggets[tidx].y = 0;    // point idx
  }
}

// This function will iterate over the nuggets (ray intersection proposals) and determine if they 
// result in an intersection. If they do, the info tensor is populated with the # of child nodes
// as determined by the input octree.
__global__ void
decide_cuda_kernel(
    const uint num, 
    const point_data* __restrict__ points, 
    const float3* __restrict__ ray_o, 
    const float3* __restrict__ ray_d,
    const uint2* __restrict__ nuggets, 
    uint* __restrict__ info, 
    const uint8_t* __restrict__ octree, 
    const uint32_t level, 
    const uint32_t not_done) {

  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num) {
    uint ridx = nuggets[tidx].x;
    uint pidx = nuggets[tidx].y;
    point_data p = points[pidx];
    float3 o = ray_o[ridx];
    float3 d = ray_d[ridx];

    // Radius of voxel
    float r = 1.0 / ((float)(0x1 << level)); 
    
    // Transform to [-1, 1]
    const float3 vc = make_float3(
        fmaf(r, fmaf(2.0, p.x, 1.0), -1.0f),
        fmaf(r, fmaf(2.0, p.y, 1.0), -1.0f),
        fmaf(r, fmaf(2.0, p.z, 1.0), -1.0f));

    // Compute aux info (precompute to optimize)
    float3 sgn = ray_sgn(d);
    float3 ray_inv = make_float3(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);

    float depth = ray_aabb(o, d, ray_inv, sgn, vc, r);

    if (not_done){
      if (depth != 0.0)
        info[tidx] = __popc(octree[pidx]);
      else
        info[tidx] = 0;
    }
    else { // at bottom
      if (depth > 0.0)
        info[tidx] = 1;
      else
        info[tidx] = 0;
    }
  }
}

// Overloaded version of function above that returns depth of voxel/ ray entry points
__global__ void
decide_cuda_kernel(
    const uint num, 
    const point_data* __restrict__ points, 
    const float3* __restrict__ ray_o, 
    const float3* __restrict__ ray_d,
    const uint2* __restrict__ nuggets, 
    float* depth,
    uint* __restrict__ info, 
    const uint8_t* __restrict__ octree, 
    const uint32_t level) {

  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num) {
    uint ridx = nuggets[tidx].x;
    uint pidx = nuggets[tidx].y;
    point_data p = points[pidx];
    float3 o = ray_o[ridx];
    float3 d = ray_d[ridx];

    // Radius of voxel
    float r = 1.0 / ((float)(0x1 << level)); 
    
    // Transform to [-1, 1]
    const float3 vc = make_float3(
        fmaf(r, fmaf(2.0, p.x, 1.0), -1.0f),
        fmaf(r, fmaf(2.0, p.y, 1.0), -1.0f),
        fmaf(r, fmaf(2.0, p.z, 1.0), -1.0f));

    // Compute aux info (precompute to optimize)
    float3 sgn = ray_sgn(d);
    float3 ray_inv = make_float3(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);

    depth[tidx] = ray_aabb(o, d, ray_inv, sgn, vc, r);

    // Perform AABB check
    if (depth[tidx] > 0.0){
      info[tidx] = 1; // mark to keep
    } else {
      info[tidx] = 0;
    }
  }
}

// Overloaded version of function above that returns depth of voxel/ ray entry and exit points
__global__ void
decide_cuda_kernel(
    const uint num, 
    const point_data* __restrict__ points, 
    const float3* __restrict__ ray_o, 
    const float3* __restrict__ ray_d,
    const uint2* __restrict__ nuggets, 
    float2* __restrict__ depth,
    uint* __restrict__ info, 
    const uint8_t* __restrict__ octree, 
    const uint32_t level) {

  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num) {
    uint ridx = nuggets[tidx].x;
    uint pidx = nuggets[tidx].y;
    point_data p = points[pidx];
    float3 o = ray_o[ridx];
    float3 d = ray_d[ridx];

    // Radius of voxel
    float r = 1.0 / ((float)(0x1 << level));
    
    // Transform to [-1, 1]
    const float3 vc = make_float3(
        fmaf(r, fmaf(2.0, p.x, 1.0), -1.0f),
        fmaf(r, fmaf(2.0, p.y, 1.0), -1.0f),
        fmaf(r, fmaf(2.0, p.z, 1.0), -1.0f));

    // Compute aux info (precompute to optimize)
    float3 sgn = ray_sgn(d);
    float3 ray_inv = make_float3(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);

    depth[tidx] = ray_aabb_with_exit(o, d, ray_inv, vc, r);

    // Perform AABB check
    if (depth[tidx].x > 0.0 && depth[tidx].y > 0.0){
      info[tidx] = 1; // mark to keep
    } else {
      info[tidx] = 0;
    }
  }
}

// This function will iterate over the nugget array, and for each nuggets stores the child indices of the
// nuggets (as defined by the octree tensor) 
__global__ void
subdivide_cuda_kernel(
    const uint num, 
    const uint2* __restrict__ nuggets_in, 
    uint2* __restrict__ nuggets_out, 
    const float3* __restrict__ ray_o,
    const point_data* __restrict__ points, 
    const uint8_t* __restrict__ octree, 
    const uint* __restrict__ exclusive_sum, 
    const uint* __restrict__ info,
    const uint* __restrict__ prefix_sum, 
    const uint32_t level) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num && info[tidx]) {
    uint ridx = nuggets_in[tidx].x;
    int pidx = nuggets_in[tidx].y;
    point_data p = points[pidx];

    uint base_idx = prefix_sum[tidx];

    uint8_t o = octree[pidx];
    uint s = exclusive_sum[pidx];

    float scale = 1.0 / ((float)(0x1 << level));
    float3 org = ray_o[ridx];
    float x = (0.5f * org.x + 0.5f) - scale*((float)p.x + 0.5);
    float y = (0.5f * org.y + 0.5f) - scale*((float)p.y + 0.5);
    float z = (0.5f * org.z + 0.5f) - scale*((float)p.z + 0.5);

    uint code = 0;
    if (x > 0) code = 4;
    if (y > 0) code += 2;
    if (z > 0) code += 1;

    for (uint i = 0; i < 8; i++) {
      uint j = VOXEL_ORDER[code][i];
      if (o&(0x1 << j)) {
        uint cnt = __popc(o&((0x2 << j) - 1)); // count set bits up to child - inclusive sum
        nuggets_out[base_idx].y = s + cnt;
        nuggets_out[base_idx++].x = ridx;
      }
    }
  }
}

template<typename scalar_t>
__global__ void
mark_pack_boundaries_cuda_kernel(
    const int64_t num, 
    const scalar_t* __restrict__ pack_ids, 
    uint* __restrict__ boundaries) {

  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num) {
    if (tidx == 0) {
      boundaries[tidx] = 1;
    } else {
      boundaries[tidx] = pack_ids[tidx - 1] == pack_ids[tidx] ? 0 : 1;
    }
  }
}

// This function will take a buffer and remove the zero pads
template<typename scalar_t>
__global__ void
compactify_cuda_kernel(
    const uint num, 
    const scalar_t* __restrict__ buffer_in, 
    scalar_t* __restrict__ buffer_out,
    const uint* __restrict__ info, 
    const uint* __restrict__ prefix_sum) {

  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num && info[tidx]) { 
    buffer_out[prefix_sum[tidx]] = buffer_in[tidx];
  }
}

template<typename scalar_t>
__global__ void
diff_cuda_kernel(
    const int64_t num_packs,
    const int64_t num_feats,
    const int64_t feat_dim,
    const scalar_t* __restrict__ feats_in,
    scalar_t* __restrict__ feats_out,
    const int64_t* __restrict__ pack_indices) {

  int64_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_packs) {
    int64_t upper_bound = (tidx == num_packs-1) ? num_feats : pack_indices[tidx+1];
    for (int64_t i=pack_indices[tidx]; i<upper_bound-1; ++i) {
      for (int64_t j=0; j<feat_dim; ++j) {
        feats_out[i * feat_dim + j] = feats_in[(i+1) * feat_dim + j] - feats_in[i * feat_dim + j];
      }
    }
  }
}

template<typename scalar_t>
__global__ void
sum_reduce_cuda_kernel(
    const int64_t num_feats, 
    const int64_t feat_dim, 
    const scalar_t* __restrict__ feats_in, 
    scalar_t* __restrict__ feats_out, 
    const int32_t* __restrict__ inclusive_sum) {

  int tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_feats) {
    for (int i=0; i<feat_dim; ++i) {
      int idx = (inclusive_sum[tidx]-1) * feat_dim + i;
#     ifdef EXPERIMENTAL
      int numel = num_feats*feat_dim;
      at::native::fastAtomicAdd(feats_out, idx, numel, feats_in[tidx * feat_dim + i], true); 
#     else
      gpuAtomicAdd(feats_out + idx, feats_in[tidx * feat_dim + i]); 
#     endif
    }
  }
}

// This kernel is the same as sum_reduce but avoids atomic add by packing the ops. 
// It however will cause thread divergence.
template<typename scalar_t>
__global__ void
packed_sum_reduce_cuda_kernel(
    const int64_t num_packs,
    const int64_t num_feats, 
    const int64_t feat_dim, 
    const scalar_t* __restrict__ feats_in, 
    scalar_t* __restrict__ feats_out, 
    const int64_t* __restrict__ pack_indices) {

  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_packs) {
    int64_t upper_bound = (tidx == num_packs-1) ? num_feats*feat_dim : pack_indices[tidx+1];    
    for (int i=pack_indices[tidx]; i<upper_bound-1; ++i) {
      for (int j=0; j<feat_dim; ++j) {
        feats_out[i * feat_dim + j] += feats_in[i * feat_dim + j];
      }
    }
  }
}

template<typename scalar_t>
__global__ void
cumprod_cuda_kernel(
    const int64_t num_packs,
    const int64_t num_feats,
    const int64_t feat_dim, 
    const scalar_t* __restrict__ feats_in, 
    scalar_t* __restrict__ feats_out, 
    int32_t* __restrict__ pack_indices,  // maps idx of pack -> beginning of global idx
    int32_t offset) {
  
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_packs) {
    int upper_bound = (tidx == num_packs-1) ? num_feats : pack_indices[tidx+1];
    int begin = pack_indices[tidx];
    if (offset == 0) {
      for (int j=0; j<feat_dim; ++j){
        feats_out[begin * feat_dim + j] = feats_in[begin * feat_dim + j];
      }
    }
    for (int i=begin+1; i<upper_bound; ++i) {
      for (int j=0; j<feat_dim; ++j){
        feats_out[i * feat_dim + j] = feats_in[(i-offset) * feat_dim + j] * feats_out[(i-1) * feat_dim + j];
      }
    }  
  } 
}

template<typename scalar_t>
__global__ void
cumprod_reverse_cuda_kernel(
    const int64_t num_packs,
    const int64_t num_feats,
    const int64_t feat_dim, 
    const scalar_t* __restrict__ feats_in, 
    scalar_t* __restrict__ feats_out, 
    int32_t* __restrict__ pack_indices,  // maps idx of pack -> beginning of global idx
    int32_t offset) {
  
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_packs) {
    int upper_bound = (tidx == num_packs-1) ? num_feats : pack_indices[tidx+1];
    int begin = pack_indices[tidx];
    if (offset == 0) {
      for (int j=0; j<feat_dim; ++j){
        feats_out[(upper_bound-1) * feat_dim + j] = feats_in[(upper_bound-1) * feat_dim + j];
      }
    }
    for (int i=upper_bound-2; i>=begin; --i) {
      for (int j=0; j<feat_dim; ++j){
        feats_out[i * feat_dim + j] = feats_in[(i+offset) * feat_dim + j] * feats_out[(i+1) * feat_dim + j];
      }
    }  
  } 
}

template<typename scalar_t>
__global__ void
cumsum_cuda_kernel(
    const int64_t num_packs,
    const int64_t num_feats,
    const int64_t feat_dim, 
    const scalar_t* __restrict__ feats_in, 
    scalar_t* __restrict__ feats_out, 
    int32_t* __restrict__ pack_indices,  // maps idx of pack -> beginning of global idx
    int32_t offset) {
  
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_packs) {
    int upper_bound = (tidx == num_packs-1) ? num_feats : pack_indices[tidx+1];
    int begin = pack_indices[tidx];
    if (offset == 0) {
      for (int j=0; j<feat_dim; ++j){
        feats_out[begin * feat_dim + j] = feats_in[begin * feat_dim + j];
      }
    }
    for (int i=begin+1; i<upper_bound; ++i) {
      for (int j=0; j<feat_dim; ++j){
        feats_out[i * feat_dim + j] = feats_in[(i-offset) * feat_dim + j] + feats_out[(i-1) * feat_dim + j];
      }
    }  
  } 
}

template<typename scalar_t>
__global__ void
cumsum_reverse_cuda_kernel(
    const int64_t num_packs,
    const int64_t num_feats,
    const int64_t feat_dim, 
    const scalar_t* __restrict__ feats_in, 
    scalar_t* __restrict__ feats_out, 
    int32_t* __restrict__ pack_indices,  // maps idx of pack -> beginning of global idx
    int32_t offset) {
  
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;
  if (tidx < num_packs) {
    int upper_bound = (tidx == num_packs-1) ? num_feats : pack_indices[tidx+1];
    int begin = pack_indices[tidx];
    if (offset == 0) {
      for (int j=0; j<feat_dim; ++j){
        feats_out[(upper_bound-1) * feat_dim + j] = feats_in[(upper_bound-1) * feat_dim + j];
      }
    }
    for (int i=upper_bound-2; i>=begin; --i) {
      for (int j=0; j<feat_dim; ++j){
        feats_out[i * feat_dim + j] = feats_in[(i+offset) * feat_dim + j] + feats_out[(i+1) * feat_dim + j];
      }
    }  
  } 
}

std::vector<at::Tensor> raytrace_cuda_impl(
    at::Tensor octree,
    at::Tensor points,
    at::Tensor pyramid,
    at::Tensor exclusive_sum,
    at::Tensor ray_o,
    at::Tensor ray_d,
    uint32_t max_level,
    uint32_t target_level,
    bool return_depth,
    bool with_exit) {

  uint num = ray_o.size(0);
  
  uint8_t* octree_ptr = octree.data_ptr<uint8_t>();
  point_data* points_ptr = reinterpret_cast<point_data*>(points.data_ptr<short>());
  uint*  exclusive_sum_ptr = reinterpret_cast<uint*>(exclusive_sum.data_ptr<int>());
  float3* ray_o_ptr = reinterpret_cast<float3*>(ray_o.data_ptr<float>());
  float3* ray_d_ptr = reinterpret_cast<float3*>(ray_d.data_ptr<float>());

  // allocate local GPU storage
  at::Tensor nuggets0 = at::empty({num, 2}, octree.options().dtype(at::kInt));
  at::Tensor nuggets1;

  uint depth_dim = with_exit ? 2 : 1;
  at::Tensor depths0;
  at::Tensor depths1;

  // Generate proposals (first proposal is root node)
  init_nuggets_cuda_kernel<<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
    num, reinterpret_cast<uint2*>(nuggets0.data_ptr<int>()));

  uint cnt, buffer = 0;
  for (uint32_t l = 0; l <= target_level; l++) {

    at::Tensor info = at::empty({num+1}, octree.options().dtype(at::kInt));
    uint* info_ptr = reinterpret_cast<uint*>(info.data_ptr<int>());

    // Do the proposals hit?
    if (l == target_level && return_depth) {
        depths0 = at::empty({num, depth_dim}, octree.options().dtype(at::kFloat));

        if (with_exit) {
            decide_cuda_kernel<<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
                num, points_ptr, ray_o_ptr, ray_d_ptr, reinterpret_cast<uint2*>(nuggets0.data_ptr<int>()), 
                reinterpret_cast<float2*>(l == target_level ? depths0.data_ptr<float>() : 0),
                info_ptr, octree_ptr, l);
        } else {
            decide_cuda_kernel<<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
                num, points_ptr, ray_o_ptr, ray_d_ptr, reinterpret_cast<uint2*>(nuggets0.data_ptr<int>()), 
                l == target_level ? depths0.data_ptr<float>() : 0, info_ptr, octree_ptr, l);
        }
    }
    else {
      decide_cuda_kernel<<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
        num, points_ptr, ray_o_ptr, ray_d_ptr, reinterpret_cast<uint2*>(nuggets0.data_ptr<int>()), 
        info_ptr, octree_ptr, l, target_level - l);   
    }

    at::Tensor prefix_sum = at::empty({num+1}, octree.options().dtype(at::kInt));
    uint*  prefix_sum_ptr = reinterpret_cast<uint*>(prefix_sum.data_ptr<int>());

    // set first element to zero
    CubDebugExit(cudaMemcpy(prefix_sum_ptr, &buffer, sizeof(uint), cudaMemcpyHostToDevice));

    // set up memory for DeviceScan calls
    void* temp_storage_ptr = NULL;
    uint64_t temp_storage_bytes = get_cub_storage_bytes(
      temp_storage_ptr, info_ptr, prefix_sum_ptr, num+1);
    at::Tensor temp_storage = at::empty({(int64_t)temp_storage_bytes}, octree.options());
    temp_storage_ptr = (void*)temp_storage.data_ptr<uint8_t>();

    CubDebugExit(cub::DeviceScan::InclusiveSum(
        temp_storage_ptr, temp_storage_bytes, info_ptr,
        prefix_sum_ptr + 1, num)); //start sum on second element
    cudaMemcpy(&cnt, prefix_sum_ptr + num, sizeof(uint), cudaMemcpyDeviceToHost);   

    // allocate local GPU storage
    nuggets1 = at::empty({cnt, 2}, octree.options().dtype(at::kInt));

    // miss everything
    if (cnt == 0) {
      num = 0;
      nuggets0 = nuggets1;
      if (return_depth) depths1 = at::empty({0, depth_dim}, octree.options().dtype(at::kFloat));
      break; 
    }

    // Subdivide if more levels remain, repeat
    if (l < target_level) {
      subdivide_cuda_kernel<<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
          num, reinterpret_cast<uint2*>(nuggets0.data_ptr<int>()), reinterpret_cast<uint2*>(nuggets1.data_ptr<int>()), ray_o_ptr, points_ptr,
          octree_ptr, exclusive_sum_ptr, info_ptr, prefix_sum_ptr, l);
    } else {
      compactify_cuda_kernel<uint2><<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
          num, reinterpret_cast<uint2*>(nuggets0.data_ptr<int>()), reinterpret_cast<uint2*>(nuggets1.data_ptr<int>()),
          info_ptr, prefix_sum_ptr);
      if (return_depth) {
          depths1 = at::empty({cnt, depth_dim}, octree.options().dtype(at::kFloat));

          if (with_exit) {
            compactify_cuda_kernel<float2><<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
                num, reinterpret_cast<float2*>(depths0.data_ptr<float>()), 
                reinterpret_cast<float2*>(depths1.data_ptr<float>()),
                info_ptr, prefix_sum_ptr);
          } else {
            compactify_cuda_kernel<float><<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
                num, depths0.data_ptr<float>(), depths1.data_ptr<float>(),
                info_ptr, prefix_sum_ptr);
          }
      }
    }
    nuggets0 = nuggets1;
    num = cnt;
  }

  if (return_depth) {
    return { nuggets0.index({Slice(0, num)}).contiguous(),
             depths1.index({Slice(0, num)}).contiguous() };
  } else {
    return { nuggets0.index({Slice(0, num)}).contiguous() };
  }
}

void mark_pack_boundaries_cuda_impl(
    at::Tensor pack_ids,
    at::Tensor boundaries) {
    int64_t num = pack_ids.size(0);
    AT_DISPATCH_INTEGRAL_TYPES(pack_ids.type(), "mark_pack_boundaries_cuda", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(boundaries));
        auto stream = at::cuda::getCurrentCUDAStream();
        mark_pack_boundaries_cuda_kernel<<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS, 0, stream>>>(
            num,
            pack_ids.data_ptr<scalar_t>(),
            reinterpret_cast<uint*>(boundaries.data_ptr<int>()));
    }));
}

void diff_cuda_impl(
    int64_t num_packs, 
    int64_t num_feats,
    int64_t feat_dim,
    at::Tensor feats_in,
    at::Tensor feats_out,
    at::Tensor pack_indices){ 
    
    int64_t* pack_indices_ptr = pack_indices.data_ptr<int64_t>();
    const int num_threads = 256;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.type(), "diff_cuda", ([&] {
        const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
        auto stream = at::cuda::getCurrentCUDAStream();
        diff_cuda_kernel<scalar_t><<<(num_packs+num_threads-1)/num_threads, num_threads, 0, stream>>>(
            num_packs, num_feats, feat_dim, 
            feats_in.data_ptr<scalar_t>(), 
            feats_out.data_ptr<scalar_t>(), 
            pack_indices_ptr);
    }));
}

void inclusive_sum_cuda_impl(
    int64_t num,
    at::Tensor info,
    at::Tensor inclusive_sum){ 

    int* info_ptr = info.data_ptr<int>();
    int* inclusive_sum_ptr = inclusive_sum.data_ptr<int>();

    void* temp_storage_ptr = NULL;
    uint64_t temp_storage_bytes = get_cub_storage_bytes(
            temp_storage_ptr, reinterpret_cast<uint*>(info_ptr), reinterpret_cast<uint*>(inclusive_sum_ptr), num);
    at::Tensor temp_storage = at::zeros({(int64_t)temp_storage_bytes}, device(at::DeviceType::CUDA).dtype(at::kByte));
    temp_storage_ptr = (void*)temp_storage.data_ptr<uint8_t>();

    CubDebugExit(cub::DeviceScan::InclusiveSum(temp_storage_ptr, temp_storage_bytes, info_ptr, inclusive_sum_ptr, num));
}

int sum_reduce_cuda_impl(
    int64_t num_feats,
    int64_t feat_dim,
    at::Tensor feats_in,
    at::Tensor feats_out,
    at::Tensor inclusive_sum) {

  int* inclusive_sum_ptr = inclusive_sum.data_ptr<int>();
  int cnt;

  const int num_threads = 1024;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.type(), "sum_reduce_cuda", ([&] {
      const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
      auto stream = at::cuda::getCurrentCUDAStream();
      cudaMemcpyAsync(&cnt, inclusive_sum_ptr + num_feats - 1, sizeof(int), cudaMemcpyDeviceToHost, stream);
      sum_reduce_cuda_kernel<scalar_t><<<(num_feats+num_threads-1)/num_threads, num_threads, 0, stream>>>(
          num_feats, feat_dim, 
          feats_in.data_ptr<scalar_t>(), 
          feats_out.data_ptr<scalar_t>(), 
          inclusive_sum_ptr);
  }));
  return cnt;
}

void cumsum_cuda_impl(
    int64_t num_feats,
    int64_t feat_dim,
    at::Tensor feats_in,
    at::Tensor feats_out,
    at::Tensor pack_indices,
    bool exclusive, 
    bool reverse) {
    
    int64_t num_packs = pack_indices.size(0);
    int* pack_indices_ptr = pack_indices.data_ptr<int>();

    int offset = exclusive ? 1 : 0;
    
    const int num_threads = 256;
    if (reverse) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.type(), "cumsum_cuda", ([&] {
            const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
            auto stream = at::cuda::getCurrentCUDAStream();
            cumsum_reverse_cuda_kernel<scalar_t><<<(num_packs+num_threads) / num_threads, num_threads, 0, stream>>>(
                num_packs, num_feats, feat_dim, 
                feats_in.data_ptr<scalar_t>(), 
                feats_out.data_ptr<scalar_t>(), 
                pack_indices_ptr, offset);
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.type(), "cumsum_cuda", ([&] {
            const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
            auto stream = at::cuda::getCurrentCUDAStream();
            cumsum_cuda_kernel<scalar_t><<<(num_packs+num_threads) / num_threads, num_threads, 0, stream>>>(
                num_packs, num_feats, feat_dim, 
                feats_in.data_ptr<scalar_t>(), 
                feats_out.data_ptr<scalar_t>(), 
                pack_indices_ptr, offset);
        }));
    }
}

void cumprod_cuda_impl(
    int64_t num_feats,
    int64_t feat_dim,
    at::Tensor feats_in,
    at::Tensor feats_out,
    at::Tensor pack_indices,
    bool exclusive,
    bool reverse) {
    
    int64_t num_packs = pack_indices.size(0);
    int* pack_indices_ptr = pack_indices.data_ptr<int>();

    int offset = exclusive ? 1 : 0;
    
    const int num_threads = 256;
    if (reverse) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.type(), "cumprod_reverse_cuda", ([&] {
            const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
            auto stream = at::cuda::getCurrentCUDAStream();
            cumprod_reverse_cuda_kernel<scalar_t><<<(num_packs+num_threads) / num_threads, num_threads, 0, stream>>>(
                num_packs, num_feats, feat_dim, 
                feats_in.data_ptr<scalar_t>(), 
                feats_out.data_ptr<scalar_t>(), 
                pack_indices_ptr, offset);
        }));
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(feats_in.type(), "cumprod_cuda", ([&] {
            const at::cuda::OptionalCUDAGuard device_guard(at::device_of(feats_out));
            auto stream = at::cuda::getCurrentCUDAStream();
            cumprod_cuda_kernel<scalar_t><<<(num_packs+num_threads) / num_threads, num_threads, 0, stream>>>(
                num_packs, num_feats, feat_dim, 
                feats_in.data_ptr<scalar_t>(), 
                feats_out.data_ptr<scalar_t>(), 
                pack_indices_ptr, offset);
        }));
    }
}

////////// generate rays //////////////////////////////////////////////////////////////////////////

__global__ void
generate_rays_cuda_kernel(
    uint num, 
    uint width, 
    uint height, 
    float4x4 tf,
    float3* ray_o, 
    float3* ray_d) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num) {
    uint px = tidx % width;
    uint py = tidx / height;

    float4 a = mul4x4(make_float4(0.0f, 0.0f, 1.0f, 0.0f), tf);
    float4 b = mul4x4(make_float4(px, py, 0.0f, 1.0f), tf);
    // float3 org = make_float3(M.m[3][0], M.m[3][1], M.m[3][2]);

    ray_o[tidx] = make_float3(a.x, a.y, a.z);
    ray_d[tidx] = make_float3(b.x, b.y, b.z);
  }
}


void generate_primary_rays_cuda_impl(
    uint width, 
    uint height, 
    float4x4& tf,
    float3* ray_o, 
    float3* ray_d) {
  uint num = width*height;

  generate_rays_cuda_kernel<<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(num, width, height, tf, ray_o, ray_d);
}


////////// generate shadow rays /////////


__global__ void
plane_intersect_rays_cuda_kernel(
    uint num, 
    float3* ray_o, 
    float3* ray_d,
    float3* output, 
    float4 plane, 
    uint* info) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num) {
    float3 org = ray_o[tidx];
    float3 dir = ray_d[tidx];

    float a = org.x*plane.x +  org.y*plane.y +  org.z*plane.z +  plane.w;
    float b = dir.x*plane.x +  dir.y*plane.y +  dir.z*plane.z;

    if (fabs(b) > 1e-3) {
      float t = - a / b;
      if (t > 0.0f) {
        output[tidx] = make_float3(org.x + t*dir.x, org.y + t*dir.y, org.z + t*dir.z);
        info[tidx] = 1;
      } else {
        info[tidx] = 0;
      }
    } else {
      info[tidx] = 0;
    }
  }
}

__global__ void
compactify_shadow_rays_cuda_kernel(
    uint num, 
    float3* p_in, 
    float3* p_out, 
    uint* map,
    uint* info, 
    uint* prefix_sum) {

  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num && info[tidx]) {
    p_out[prefix_sum[tidx]] = p_in[tidx];
    map[prefix_sum[tidx]] = tidx;
  }
}

__global__ void
set_shadow_rays_cuda_kernel(
    uint num, 
    float3* src, 
    float3* dst, 
    float3 light) {

  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num) {
    dst[tidx] = normalize(src[tidx] - light);
    src[tidx] = light;
  }
}

uint generate_shadow_rays_cuda_impl(
    uint num,
    float3* ray_o,
    float3* ray_d,
    float3* src,
    float3* dst,
    uint* map,
    float3& light,
    float4& plane,
    uint* info,
    uint* prefix_sum) {
  
  // set up memory for DeviceScan calls
  void* temp_storage_ptr = NULL;
  uint64_t temp_storage_bytes = get_cub_storage_bytes(temp_storage_ptr, info, prefix_sum, num);
  at::Tensor temp_storage = at::zeros({(int64_t)temp_storage_bytes}, device(at::DeviceType::CUDA).dtype(at::kByte));
  temp_storage_ptr = (void*)temp_storage.data_ptr<uint8_t>();

  uint cnt = 0;
  plane_intersect_rays_cuda_kernel<<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
      num, ray_o, ray_d, dst, plane, info);
  CubDebugExit(cub::DeviceScan::ExclusiveSum(
      temp_storage_ptr, temp_storage_bytes, info, prefix_sum, num));
  cudaMemcpy(&cnt, prefix_sum + num - 1, sizeof(uint), cudaMemcpyDeviceToHost);
  compactify_shadow_rays_cuda_kernel<<<(num + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(
      num, dst, src, map, info, prefix_sum);
  set_shadow_rays_cuda_kernel<<<(cnt + RT_NUM_THREADS - 1) / RT_NUM_THREADS, RT_NUM_THREADS>>>(cnt, src, dst, light);

  return cnt;
}

} // namespace kaolin

