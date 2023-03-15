// Copyright (c) 2021,23 NVIDIA CORPORATION & AFFILIATES.
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

#include <ATen/ATen.h>

#define CUB_STDERR
#include <cub/device/device_scan.cuh>
#include <cub/device/device_radix_sort.cuh>

#include "math_constants.h"

#include "../../spc/spc.h"
#include "../../../spc_math.h"
#include "../../../spc_utils.cuh"


namespace kaolin {

using namespace std;
using namespace at::indexing;

#define NUM_THREADS 64


size_t get_cub_storage_bytes_sort_pairs(
  void* d_temp_storage, 
  const uint64_t* d_morton_codes_in, 
  uint64_t* d_morton_codes_out, 
  const uint64_t* d_values_in, 
  uint64_t* d_values_out, 
  uint32_t num_items)
{
    size_t    temp_storage_bytes = 0;
    CubDebugExit(
        cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
                                        d_morton_codes_in, d_morton_codes_out, 
                                        d_values_in, d_values_out, num_items)
    );
    return temp_storage_bytes;
}


__global__ void
d_MarkDuplicates(
  uint num, 
  morton_code* mcode, 
  uint* occupancy)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num)
  {
    if (tidx == 0)
      occupancy[tidx] = 1;
    else
      occupancy[tidx] = mcode[tidx - 1] == mcode[tidx] ? 0 : 1;
  }
}


__global__ void
d_Compactify(
  uint num, 
  morton_code* mIn, 
  morton_code* mOut, 
  uint* occupancy, 
  uint* prefix_sum, 
  uint64_t* idx_in, 
  uint64_t* idx_out)
{
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num && occupancy[tidx]) {
    mOut[prefix_sum[tidx]] = mIn[tidx];
    idx_out[prefix_sum[tidx]] = idx_in[tidx];
  }
}

// determine if triangle and voxel overlap w.r.t. axis using separating axis theorem
__device__ bool 
TriangleVoxelSAT(double3 v0, double3 v1, double3 v2, float voxelHalfSize, double3 axis)
{
    double d0 = dot(v0, axis);
    double d1 = dot(v1, axis);
    double d2 = dot(v2, axis);

    double maxd = max(d0, max(d1, d2));
    double mind = min(d0, min(d1, d2));

    double r = voxelHalfSize * (abs(axis.x) + abs(axis.y) + abs(axis.z));

    float fd = (float)max(-maxd, mind);
    float fr = (float)r;

    return fd <= fr;
}


// determine if triangle and voxel overlap by testing against 13 critical axis
__device__ bool 
TriangleVoxelTest(float3 fva, float3 fvb, float3 fvc, float3 voxelcenter, float voxelHalfSize)
{
    double3 va = make_double3(fva.x-voxelcenter.x, fva.y-voxelcenter.y, fva.z-voxelcenter.z);
    double3 vb = make_double3(fvb.x-voxelcenter.x, fvb.y-voxelcenter.y, fvb.z-voxelcenter.z);
    double3 vc = make_double3(fvc.x-voxelcenter.x, fvc.y-voxelcenter.y, fvc.z-voxelcenter.z);

    double3 ab = normalize(vb - va);
    double3 bc = normalize(vc - vb);
    double3 ca = normalize(va - vc);

    //Cross ab, bc, and ca with (1, 0, 0)
    double3 a00 = make_double3(0.0, -ab.z, ab.y);
    double3 a01 = make_double3(0.0, -bc.z, bc.y);
    double3 a02 = make_double3(0.0, -ca.z, ca.y);

    //Cross ab, bc, and ca with (0, 1, 0)
    double3 a10 = make_double3(ab.z, 0.0, -ab.x);
    double3 a11 = make_double3(bc.z, 0.0, -bc.x);
    double3 a12 = make_double3(ca.z, 0.0, -ca.x);

    //Cross ab, bc, and ca with (0, 0, 1)
    double3 a20 = make_double3(-ab.y, ab.x, 0.0);
    double3 a21 = make_double3(-bc.y, bc.x, 0.0);
    double3 a22 = make_double3(-ca.y, ca.x, 0.0);
    
    if (!TriangleVoxelSAT(va, vb, vc, voxelHalfSize, a00) ||
        !TriangleVoxelSAT(va, vb, vc, voxelHalfSize, a01) ||
        !TriangleVoxelSAT(va, vb, vc, voxelHalfSize, a02) ||
        !TriangleVoxelSAT(va, vb, vc, voxelHalfSize, a10) ||
        !TriangleVoxelSAT(va, vb, vc, voxelHalfSize, a11) ||
        !TriangleVoxelSAT(va, vb, vc, voxelHalfSize, a12) ||
        !TriangleVoxelSAT(va, vb, vc, voxelHalfSize, a20) ||
        !TriangleVoxelSAT(va, vb, vc, voxelHalfSize, a21) ||
        !TriangleVoxelSAT(va, vb, vc, voxelHalfSize, a22) ||
        !TriangleVoxelSAT(va, vb, vc, voxelHalfSize, make_double3(1, 0, 0)) ||
        !TriangleVoxelSAT(va, vb, vc, voxelHalfSize, make_double3(0, 1, 0)) ||
        !TriangleVoxelSAT(va, vb, vc, voxelHalfSize, make_double3(0, 0, 1)) ||
        !TriangleVoxelSAT(va, vb, vc, voxelHalfSize, cross(ab, bc))) {
        return false;
    }

    return true;
}


// This function will iterate over t(triangle intersection proposals) and determine if they result in an 
// intersection. If they do, the occupancy tensor is set to number of voxels in subdivision or compaction
__global__ void
decide_cuda_kernel(
  const uint num, 
  const float3* __restrict__ face_vertices, 
  const uint64_t* __restrict__ morton_codes, // voxel morton codes
  const uint64_t* __restrict__ triangle_id, // face ids
  uint* __restrict__ occupancy, 
  const uint32_t level, 
  const uint32_t not_done) {

  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num) {
    occupancy[tidx] = 0;

    uint64_t tri_idx = triangle_id[tidx];

    float two_level = (float)(0x1 << level);
    float voxelSize = 2.0f/two_level;
    float voxelHalfSize = 0.5*voxelSize;

    point_data gridPos = to_point(morton_codes[tidx]);

    float3 p;
    p.x = fmaf(gridPos.x, voxelSize, voxelHalfSize-1.0f);
    p.y = fmaf(gridPos.y, voxelSize, voxelHalfSize-1.0f);
    p.z = fmaf(gridPos.z, voxelSize, voxelHalfSize-1.0f);

    if (TriangleVoxelTest(face_vertices[3*tri_idx], face_vertices[3*tri_idx+1], face_vertices[3*tri_idx+2], p, voxelHalfSize))
      occupancy[tidx] = not_done ? 8 : 1;

  }
}


// This function will subdivide input voxel to out voxels 
__global__ void
subdivide_cuda_kernel(
    const uint num, 
    const uint64_t* __restrict__ morton_codes_in, // input voxel morton codes
    const uint64_t* __restrict__ triangle_id_in, // input face ids
    uint64_t* __restrict__ morton_codes_out, // output voxel morton codes
    uint64_t* __restrict__ triangle_id_out, // output face ids
    const uint* __restrict__ occupancy,  
    const uint* __restrict__ prefix_sum) {

  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num && occupancy[tidx]) {
    uint64_t triangle_id = triangle_id_in[tidx]; //triangle id

    point_data p_in = to_point(morton_codes_in[tidx]);
    point_data p_out = make_point_data(2 * p_in.x, 2 * p_in.y, 2*p_in.z);

    uint base_idx = prefix_sum[tidx];

    for (uint i = 0; i < 8; i++) {
      point_data p = make_point_data(p_out.x+(i>>2), p_out.y+((i>>1)&0x1), p_out.z+(i&0x1));
      morton_codes_out[base_idx] = to_morton(p);
      triangle_id_out[base_idx] = triangle_id;
      base_idx++;
    }
  }
}


// compacify input buffers
__global__ void
compactify_cuda_kernel(
    const uint num, 
    const uint64_t* __restrict__ morton_codes_in, 
    const uint64_t* __restrict__ triangle_id_in, 
    uint64_t* __restrict__ morton_codes_out, 
    uint64_t* __restrict__ triangle_id_out, 
    const uint* __restrict__ occupancy,
    const uint* __restrict__ prefix_sum) {

  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num && occupancy[tidx]) { 
    morton_codes_out[prefix_sum[tidx]] = morton_codes_in[tidx];
    triangle_id_out[prefix_sum[tidx]] = triangle_id_in[tidx];
  }
}


// compute 3x3 matrix that maps 3d cartesian coords to 3d barycentriuc coods, w.r.t a given triangle
__global__ void
d_ComputeBaryCoords(
    uint num, 
    const float3* __restrict__ face_vertices,
    const uint64_t* __restrict__ morton_codes, 
    const uint64_t* __restrict__ triangle_id, 
    float2* barycoords,
    const uint32_t level) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num) {
    float two_level = (float)(0x1 << level);
    float voxelSize = 2.0f / two_level;
    float voxelHalfSize = 0.5 * voxelSize;

    point_data gridPos = to_point(morton_codes[tidx]);

    float3 p; // voxel centriod
    p.x = fmaf(gridPos.x, voxelSize, voxelHalfSize-1.0f);
    p.y = fmaf(gridPos.y, voxelSize, voxelHalfSize-1.0f);
    p.z = fmaf(gridPos.z, voxelSize, voxelHalfSize-1.0f);


    uint64_t face_id = triangle_id[tidx];

    float3 v1 = face_vertices[face_id * 3 + 0];
    float3 v2 = face_vertices[face_id * 3 + 1];
    float3 v3 = face_vertices[face_id * 3 + 2];
    float3 closest_p = triangle_closest_point(v1, v2, v3, p);
    float delta = dot2(cross(v1 - v2, v1 - v3));
    float3 d1 = closest_p - v1;
    float3 d2 = closest_p - v2;
    float3 d3 = closest_p - v3;
    float da = sqrtf(dot2(cross(d2, d3)));
    float db = sqrtf(dot2(cross(d1, d3)));
    float dc = sqrtf(dot2(cross(d1, d2)));
    float3 b = make_float3(da * rsqrtf(delta),
                           db * rsqrtf(delta),
                           dc * rsqrtf(delta));
    // handle negative barycentric coordinates
    if (b.x < 0.0f) {
      b.x = 0.;
    }

    if (b.y < 0.0f) {
      b.y = 0.;
    }

    if (b.z < 0.0f) {
      b.z = 0.;
    }
    b *= 1. / (b.x + b.y + b.z); // normalize to x+y+z==1 plane
    
    barycoords[tidx] = make_float2(b.x, b.y);
  }
}


std::vector<at::Tensor> mesh_to_spc_cuda_impl(
    at::Tensor face_vertices, 
    uint target_level) {
  // number of triangles, and indices pointer 
  int ntris = face_vertices.size(0);
 
  // allocate local GPU storage
  at::Tensor morton_codes0 = at::zeros({ntris}, face_vertices.options().dtype(at::kLong));
  at::Tensor triangle_id0 = at::arange(ntris, face_vertices.options().dtype(at::kLong));
  at::Tensor morton_codes1;
  at::Tensor triangle_id1;

  at::Tensor occupancy;
  at::Tensor prefix_sum;
  at::Tensor temp_storage;
 
  uint64_t temp_storage_bytes;

  uint next_cnt, buffer = 0; 
  uint curr_cnt = ntris;
  for (uint32_t l = 0; l <= target_level; l++) {

    occupancy = at::empty({curr_cnt + 1}, face_vertices.options().dtype(at::kInt));
    prefix_sum = at::empty({curr_cnt + 1}, face_vertices.options().dtype(at::kInt));

    // Do the proposals hit?
    decide_cuda_kernel<<<(curr_cnt + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(
      curr_cnt, 
      reinterpret_cast<float3*>(face_vertices.data_ptr<float>()), 
      reinterpret_cast<uint64_t*>(morton_codes0.data_ptr<int64_t>()), 
      reinterpret_cast<uint64_t*>(triangle_id0.data_ptr<int64_t>()), 
      reinterpret_cast<uint*>(occupancy.data_ptr<int>()), l, target_level - l);

    // set up memory for DeviceScan calls
    temp_storage_bytes = get_cub_storage_bytes(NULL, reinterpret_cast<uint*>(occupancy.data_ptr<int>()), 
      reinterpret_cast<uint*>(prefix_sum.data_ptr<int>()), curr_cnt+1);
    temp_storage = at::empty({(int64_t)temp_storage_bytes}, face_vertices.options().dtype(at::kByte));

    CubDebugExit(cub::DeviceScan::ExclusiveSum(
      (void*)temp_storage.data_ptr<uint8_t>(), temp_storage_bytes, 
      reinterpret_cast<uint*>(occupancy.data_ptr<int>()), 
      reinterpret_cast<uint*>(prefix_sum.data_ptr<int>()), curr_cnt+1)); //start sum on second element
    cudaMemcpy(&next_cnt, reinterpret_cast<uint*>(prefix_sum.data_ptr<int>()) + curr_cnt, 
               sizeof(uint), cudaMemcpyDeviceToHost);

    if (next_cnt == 0) {
      at::Tensor octree = at::empty({0}, face_vertices.options().dtype(at::kByte));
      at::Tensor face_ids = at::empty({0}, face_vertices.options().dtype(at::kLong));
      at::Tensor bary_coords = at::zeros({0, 3}, face_vertices.options().dtype(at::kFloat));
      return { octree, face_ids, bary_coords };
    }
    else {
      // allocate local GPU storage
      morton_codes1 = at::empty({next_cnt}, face_vertices.options().dtype(at::kLong));
      triangle_id1 = at::empty({next_cnt}, face_vertices.options().dtype(at::kLong));
    }

    // Subdivide if more levels remain, repeat
    if (l < target_level) {
      subdivide_cuda_kernel<<<(curr_cnt + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(
        curr_cnt, 
        reinterpret_cast<uint64_t*>(morton_codes0.data_ptr<int64_t>()), 
        reinterpret_cast<uint64_t*>(triangle_id0.data_ptr<int64_t>()), 
        reinterpret_cast<uint64_t*>(morton_codes1.data_ptr<int64_t>()), 
        reinterpret_cast<uint64_t*>(triangle_id1.data_ptr<int64_t>()), 
        reinterpret_cast<uint*>(occupancy.data_ptr<int>()), 
        reinterpret_cast<uint*>(prefix_sum.data_ptr<int>()));
    } else {
      compactify_cuda_kernel<<<(curr_cnt + NUM_THREADS - 1) / NUM_THREADS, NUM_THREADS>>>(
        curr_cnt, 
        reinterpret_cast<uint64_t*>(morton_codes0.data_ptr<int64_t>()), 
        reinterpret_cast<uint64_t*>(triangle_id0.data_ptr<int64_t>()), 
        reinterpret_cast<uint64_t*>(morton_codes1.data_ptr<int64_t>()), 
        reinterpret_cast<uint64_t*>(triangle_id1.data_ptr<int64_t>()), 
        reinterpret_cast<uint*>(occupancy.data_ptr<int>()), 
        reinterpret_cast<uint*>(prefix_sum.data_ptr<int>()));
    }

    morton_codes0 = morton_codes1;
    triangle_id0 = triangle_id1;
    curr_cnt = next_cnt;
  }

  // allocate local GPU storage
  morton_codes1 = at::empty({curr_cnt}, face_vertices.options().dtype(at::kLong));
  triangle_id1 = at::empty({curr_cnt}, face_vertices.options().dtype(at::kLong));

  // set up memory for DeviceScan calls
  temp_storage_bytes = get_cub_storage_bytes_sort_pairs(
    NULL, 
    reinterpret_cast<uint64_t*>(morton_codes0.data_ptr<int64_t>()), 
    reinterpret_cast<uint64_t*>(morton_codes1.data_ptr<int64_t>()), 
    reinterpret_cast<uint64_t*>(triangle_id0.data_ptr<int64_t>()), 
    reinterpret_cast<uint64_t*>(triangle_id1.data_ptr<int64_t>()), curr_cnt);
  temp_storage = at::empty({(int64_t)temp_storage_bytes}, face_vertices.options().dtype(at::kByte));

  CubDebugExit(cub::DeviceRadixSort::SortPairs(
    (void*)temp_storage.data_ptr<uint8_t>(), 
    temp_storage_bytes, 
    reinterpret_cast<uint64_t*>(morton_codes0.data_ptr<int64_t>()), 
    reinterpret_cast<uint64_t*>(morton_codes1.data_ptr<int64_t>()), 
    reinterpret_cast<uint64_t*>(triangle_id0.data_ptr<int64_t>()), 
    reinterpret_cast<uint64_t*>(triangle_id1.data_ptr<int64_t>()), curr_cnt));

  occupancy = at::empty({curr_cnt+1}, face_vertices.options().dtype(at::kInt));
  prefix_sum = at::empty({curr_cnt+1}, face_vertices.options().dtype(at::kInt));

  // set first element to zero
  CubDebugExit(cudaMemcpy(reinterpret_cast<uint*>(prefix_sum.data_ptr<int>()), &buffer, sizeof(uint), cudaMemcpyHostToDevice));

  // Mark boundaries of unique  
  d_MarkDuplicates << <(curr_cnt + 63) / 64, 64 >> > (
    curr_cnt, 
    reinterpret_cast<uint64_t*>(morton_codes1.data_ptr<int64_t>()), 
    reinterpret_cast<uint*>(occupancy.data_ptr<int>()));

  // set up memory for DeviceScan calls
  temp_storage_bytes = get_cub_storage_bytes(
    NULL, 
    reinterpret_cast<uint*>(occupancy.data_ptr<int>()), 
    reinterpret_cast<uint*>(prefix_sum.data_ptr<int>()), curr_cnt+1);
  temp_storage = at::empty({(int64_t)temp_storage_bytes}, face_vertices.options().dtype(at::kByte));

  uint psize;
  cub::DeviceScan::ExclusiveSum(
    (void*)temp_storage.data_ptr<uint8_t>(), 
    temp_storage_bytes, 
    reinterpret_cast<uint*>(occupancy.data_ptr<int>()), 
    reinterpret_cast<uint*>(prefix_sum.data_ptr<int>()), curr_cnt+1);
  CubDebugExit(cudaMemcpy(&psize, reinterpret_cast<uint*>(prefix_sum.data_ptr<int>()) + curr_cnt, sizeof(uint), cudaMemcpyDeviceToHost));

  d_Compactify << <(curr_cnt + 63) / 64, 64 >> > (
    curr_cnt, 
    reinterpret_cast<uint64_t*>(morton_codes1.data_ptr<int64_t>()), 
    reinterpret_cast<uint64_t*>(morton_codes0.data_ptr<int64_t>()), 
    reinterpret_cast<uint*>(occupancy.data_ptr<int>()), 
    reinterpret_cast<uint*>(prefix_sum.data_ptr<int>()), 
    reinterpret_cast<uint64_t*>(triangle_id1.data_ptr<int64_t>()), 
    reinterpret_cast<uint64_t*>(triangle_id0.data_ptr<int64_t>()));
  at::Tensor bary_coords = at::zeros({psize, 2}, face_vertices.options().dtype(at::kFloat));

  d_ComputeBaryCoords << <(psize + 63) / 64, 64 >> > (
    psize,
    reinterpret_cast<float3*>(face_vertices.data_ptr<float>()),
    reinterpret_cast<uint64_t*>(morton_codes0.data_ptr<int64_t>()), 
    reinterpret_cast<uint64_t*>(triangle_id0.data_ptr<int64_t>()), 
    reinterpret_cast<float2*>(bary_coords.data_ptr<float>()), target_level);

  morton_codes1 = morton_codes0.index({Slice(None, psize)}).contiguous();   
  triangle_id1 = triangle_id0.index({Slice(None, psize)}).contiguous();

  at::Tensor octree = morton_to_octree(morton_codes1, target_level);

  return { octree, triangle_id1, bary_coords };
}



}  // namespace kaolin
