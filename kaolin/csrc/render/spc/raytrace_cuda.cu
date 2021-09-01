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

#include <stdio.h>
#include <ATen/ATen.h>

#define CUB_STDERR
#include <cub/device/device_scan.cuh>

#include "../../spc_math.h"
#include "spc_render_utils.h"

namespace kaolin {

using namespace cub;
using namespace std;
using namespace at::indexing;

__constant__ uint Order[8][8] = {
    { 0, 1, 2, 4, 3, 5, 6, 7 },
    { 1, 0, 3, 5, 2, 4, 7, 6 },
    { 2, 0, 3, 6, 1, 4, 7, 5 },
    { 3, 1, 2, 7, 0, 5, 6, 4 },
    { 4, 0, 5, 6, 1, 2, 7, 3 },
    { 5, 1, 4, 7, 0, 3, 6, 2 },
    { 6, 2, 4, 7, 0, 3, 5, 1 },
    { 7, 3, 5, 6, 1, 2, 4, 0 }
};

uint64_t GetStorageBytes(void* d_temp_storage, uint* d_Info, uint* d_PrefixSum,
                      uint max_total_points) {
    uint64_t temp_storage_bytes = 0;
    CubDebugExit(DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes, d_Info,
        d_PrefixSum, max_total_points));
    return temp_storage_bytes;
}


__global__ void
d_InitNuggets(uint num, uint2* nuggets) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num) {
    nuggets[tidx].x = tidx; //ray idx
    nuggets[tidx].y = 0;
  }
}

__device__ bool
d_FaceEval(ushort i, ushort j, float a, float b, float c) {
  float result[4];

  result[0] = a*i + b*j + c;
  result[1] = result[0] + a;
  result[2] = result[0] + b;
  result[3] = result[1] + b;

  float min = 1;
  float max = -1;

  for (int i = 0; i < 4; i++) {
    if (result[i] < min) min = result[i];
    if (result[i] > max) max = result[i];
  }

  return (min <= 0.0f && max >= 0.0f);
}

// This function will iterate over the nuggets (ray intersection proposals) and determine if they 
// result in an intersection. If they do, the info tensor is populated with the # of child nodes
// as determined by the input octree.
__global__ void
d_Decide(uint num, point_data* points, float3* rorg, float3* rdir,
         uint2* nuggets, uint* info, uchar* O, uint Level, uint notDone) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num) {
    uint ridx = nuggets[tidx].x;
    uint pidx = nuggets[tidx].y;
    point_data p = points[pidx];
    float3 o = rorg[ridx];
    float3 d = rdir[ridx];

    // Radius of voxel
    float s1 = 1.0 / ((float)(0x1 << Level));
    
    // Transform to [-1, 1]
    const float3 vc = make_float3(
        fmaf(s1, fmaf(2.0, p.x, 1.0), -1.0f),
        fmaf(s1, fmaf(2.0, p.y, 1.0), -1.0f),
        fmaf(s1, fmaf(2.0, p.z, 1.0), -1.0f));

    // Compute aux info (precompute to optimize)
    float3 sgn = ray_sgn(d);
    float3 ray_inv = make_float3(1.0 / d.x, 1.0 / d.y, 1.0 / d.z);

    // Perform AABB check
    if (ray_aabb(o, d, ray_inv, sgn, vc, s1) > 0.0){
      // Count # of occupied voxels for expansion, if more levels are left
      info[tidx] = notDone ? __popc(O[pidx]) : 1;      
    } else {
      info[tidx] = 0;
    }
  }
}


__global__ void
d_Subdivide(uint num, uint2* nuggetsIn, uint2* nuggetsOut, float3* rorg,
            point_data* points, uchar* O, uint* S, uint* info,
            uint* prefix_sum, uint Level) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num && info[tidx]) {
    uint ridx = nuggetsIn[tidx].x;
    int pidx = nuggetsIn[tidx].y;
    point_data p = points[pidx];

    uint IdxBase = prefix_sum[tidx];

    uchar o = O[pidx];
    uint s = S[pidx];

    float scale = 1.0 / ((float)(0x1 << Level));
    float3 org = rorg[ridx];
    float x = (0.5f * org.x + 0.5f) - scale*((float)p.x + 0.5);
    float y = (0.5f * org.y + 0.5f) - scale*((float)p.y + 0.5);
    float z = (0.5f * org.z + 0.5f) - scale*((float)p.z + 0.5);

    uint code = 0;
    if (x > 0) code = 4;
    if (y > 0) code += 2;
    if (z > 0) code += 1;

    for (uint i = 0; i < 8; i++) {
      uint j = Order[code][i];
      if (o&(0x1 << j)) {
        uint cnt = __popc(o&((0x2 << j) - 1)); // count set bits up to child - inclusive sum
        nuggetsOut[IdxBase].y = s + cnt;
        nuggetsOut[IdxBase++].x = ridx;
      }
    }
  }
}


__global__ void
d_RemoveDuplicateRays(uint num, uint2* nuggets, uint* info) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num) {
    if (tidx == 0)
      info[tidx] = 1;
    else
      info[tidx] = nuggets[tidx - 1].x == nuggets[tidx].x ? 0 : 1;
  }
}


__global__ void
d_Compactify(uint num, uint2* nuggetsIn, uint2* nuggetsOut,
             uint* info, uint* prefix_sum) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num && info[tidx])
    nuggetsOut[prefix_sum[tidx]] = nuggetsIn[tidx];
}

uint spc_raytrace_cuda(
    uchar* d_octree,
    uint Level,
    uint targetLevel,
    point_data* d_points,
    uint* h_pyramid,
    uint* d_exsum,
    uint num,
    float3* d_Org,
    float3* d_Dir,
    uint2*  d_NuggetBuffers,
    uint*  d_Info,
    uint*  d_PrefixSum,
    void* d_temp_storage,
    uint64_t temp_storage_bytes) {

  uint* PyramidSum = h_pyramid + Level + 2;

  uint2*  d_Nuggets[2];
  d_Nuggets[0] = d_NuggetBuffers;
  d_Nuggets[1] = d_NuggetBuffers + KAOLIN_SPC_MAX_POINTS;

  int osize = PyramidSum[Level];

  d_InitNuggets<<<(num + 1023) / 1024, 1024>>>(num, d_Nuggets[0]);

  uint cnt, buffer = 0;

  // set first element to zero
  CubDebugExit(cudaMemcpy(d_PrefixSum, &buffer, sizeof(uint),
                          cudaMemcpyHostToDevice));

  for (uint l = 0; l <= targetLevel; l++) {
    d_Decide<<<(num + 1023) / 1024, 1024>>>(
        num, d_points, d_Org, d_Dir, d_Nuggets[buffer], d_Info, d_octree, l,
        targetLevel - l);
    CubDebugExit(DeviceScan::InclusiveSum(
        d_temp_storage, temp_storage_bytes, d_Info,
        d_PrefixSum + 1, num));//start sum on second element
    cudaMemcpy(&cnt, d_PrefixSum + num, sizeof(uint), cudaMemcpyDeviceToHost);

    if (cnt == 0 || cnt > KAOLIN_SPC_MAX_POINTS)
      break; // either miss everything, or exceed memory allocation

    if (l < targetLevel) {
      d_Subdivide<<<(num + 1023) / 1024, 1024>>>(
          num, d_Nuggets[buffer], d_Nuggets[(buffer + 1) % 2], d_Org, d_points,
          d_octree, d_exsum, d_Info, d_PrefixSum, l);
    } else {
      d_Compactify<<<(num + 1023) / 1024, 1024>>>(
          num, d_Nuggets[buffer], d_Nuggets[(buffer + 1) % 2],
          d_Info, d_PrefixSum);
    }

    CubDebugExit(cudaGetLastError());

    buffer = (buffer + 1) % 2;
    num = cnt;
  }

  return cnt;
}


uint remove_duplicate_rays_cuda(
    uint num,
    uint2*  d_Nuggets0,
    uint2*  d_Nuggets1,
    uint*  d_Info,
    uint*  d_PrefixSum,
    void* d_temp_storage,
    uint64_t temp_storage_bytes) {
  uint cnt = 0;

  d_RemoveDuplicateRays << <(num + 1023) / 1024, 1024 >> > (num, d_Nuggets0, d_Info);
  CubDebugExit(DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_Info, d_PrefixSum, num+1));
  cudaMemcpy(&cnt, d_PrefixSum + num, sizeof(uint), cudaMemcpyDeviceToHost);
  d_Compactify << <(num + 1023) / 1024, 1024 >> > (num, d_Nuggets0, d_Nuggets1, d_Info, d_PrefixSum);

  return cnt;
}

void mark_first_hit_cuda(
    uint num,
    uint2* d_Nuggets,
    uint* d_Info) {
    d_RemoveDuplicateRays << <(num + 1023) / 1024, 1024 >> > (num, d_Nuggets, d_Info);
}

////////// generate rays //////////////////////////////////////////////////////////////////////////

__global__ void
d_generate_rays(uint num, uint imageW, uint imageH, float4x4 mM,
                float3* rayorg, float3* raydir) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num) {
    uint px = tidx % imageW;
    uint py = tidx / imageW;

    float4 a = mul4x4(make_float4(0.0f, 0.0f, 1.0f, 0.0f), mM);
    float4 b = mul4x4(make_float4(px, py, 0.0f, 1.0f), mM);
    // float3 org = make_float3(M.m[3][0], M.m[3][1], M.m[3][2]);

    rayorg[tidx] = make_float3(a.x, a.y, a.z);
    raydir[tidx] = make_float3(b.x, b.y, b.z);
  }
}


void generate_primary_rays_cuda(uint imageW, uint imageH, float4x4& mM,
                                float3* d_Org, float3* d_Dir) {
  uint num = imageW*imageH;

  d_generate_rays<<<(num + 1023) / 1024, 1024>>>(num, imageW, imageH, mM, d_Org, d_Dir);
}


////////// generate shadow rays /////////


__global__ void
d_plane_intersect_rays(uint num, float3* d_Org, float3* d_Dir,
                       float3* d_Dst, float4 plane, uint* info) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num) {
    float3 org = d_Org[tidx];
    float3 dir = d_Dir[tidx];

    float a = org.x*plane.x +  org.y*plane.y +  org.z*plane.z +  plane.w;
    float b = dir.x*plane.x +  dir.y*plane.y +  dir.z*plane.z;

    if (fabs(b) > 1e-3) {
      float t = - a / b;
      if (t > 0.0f) {
        d_Dst[tidx] = make_float3(org.x + t*dir.x, org.y + t*dir.y, org.z + t*dir.z);
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
d_Compactify2(uint num, float3* pIn, float3* pOut, uint* map,
              uint* info, uint* prefix_sum) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num && info[tidx]) {
    pOut[prefix_sum[tidx]] = pIn[tidx];
    map[prefix_sum[tidx]] = tidx;
  }
}

__global__ void
d_SetShadowRays(uint num, float3* src, float3* dst, float3 light) {
  uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

  if (tidx < num) {
    dst[tidx] = normalize(src[tidx] - light);
    src[tidx] = light;
  }
}

uint generate_shadow_rays_cuda(
    uint num,
    float3* org,
    float3* dir,
    float3* src,
    float3* dst,
    uint* map,
    float3& light,
    float4& plane,
    uint* info,
    uint*prefixSum,
    void* d_temp_storage,
    uint64_t temp_storage_bytes) {
  uint cnt = 0;
  d_plane_intersect_rays<<<(num + 1023) / 1024, 1024>>>(
      num, org, dir, dst, plane, info);
  CubDebugExit(DeviceScan::ExclusiveSum(
      d_temp_storage, temp_storage_bytes, info, prefixSum, num));
  cudaMemcpy(&cnt, prefixSum + num - 1, sizeof(uint), cudaMemcpyDeviceToHost);
  d_Compactify2<<<(num + 1023) / 1024, 1024>>>(
      num, dst, src, map, info, prefixSum);
  d_SetShadowRays<<<(cnt + 1023) / 1024, 1024>>>(cnt, src, dst, light);

  return cnt;
}

// This kernel will iterate over Nuggets, instead of iterating over rays
__global__ void ray_aabb_kernel(
    const float3* __restrict__ query,     // ray query array
    const float3* __restrict__ ray_d,     // ray direction array
    const float3* __restrict__ ray_inv,   // inverse ray direction array
    const int2* __restrict__ nuggets,     // nugget array (ray-aabb correspondences)
    const float3* __restrict__ points,    // 3d coord array
    const int* __restrict__ info,         // binary array denoting beginning of nugget group
    const int* __restrict__  info_idxes,  // array of active nugget indices
    const float r,                        // radius of aabb
    const bool init,                      // first run?
    float* __restrict__ d,                // distance
    bool* __restrict__ cond,              // true if hit
    int* __restrict__ pidx,               // index of 3d coord array
    const int num_nuggets,                // # of nugget indices
    const int n                           // # of active nugget indices
){
    
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x*gridDim.x;
    if (idx > n) return;

    for (int _i=idx; _i<n; _i+=stride) {
        // Get index of corresponding nugget
        int i = info_idxes[_i];
        
        // Get index of ray
        uint ridx = nuggets[i].x;

        // If this ray is already terminated, continue
        if (!cond[ridx] && !init) continue;

        bool _hit = false;
        
        // Sign bit
        const float3 sgn = ray_sgn(ray_d[ridx]);
        
        int j = 0;
        // In order traversal of the voxels
        do {
            // Get the vc from the nugget
            uint _pidx = nuggets[i].y; // Index of points

            // Center of voxel
            const float3 vc = make_float3(
                fmaf(r, fmaf(2.0, points[_pidx].x, 1.0), -1.0f),
                fmaf(r, fmaf(2.0, points[_pidx].y, 1.0), -1.0f),
                fmaf(r, fmaf(2.0, points[_pidx].z, 1.0), -1.0f));

            float _d = ray_aabb(query[ridx], ray_d[ridx], ray_inv[ridx], sgn, vc, r);

            if (_d != 0.0) {
                _hit = true;
                pidx[ridx] = _pidx;
                cond[ridx] = _hit;
                if (_d > 0.0) {
                    d[ridx] = _d;
                }
            } 
           
            ++i;
            ++j;
            
        } while (i < num_nuggets && info[i] != 1 && _hit == false);

        if (!_hit) {
            // Should only reach here if it misses
            cond[ridx] = false;
            d[ridx] = 100;
        }
        
    }
}

void ray_aabb_cuda(
    const float3* query,     // ray query array
    const float3* ray_d,     // ray direction array
    const float3* ray_inv,   // inverse ray direction array
    const int2*  nuggets,    // nugget array (ray-aabb correspondences)
    const float3* points,    // 3d coord array
    const int* info,         // binary array denoting beginning of nugget group
    const int* info_idxes,   // array of active nugget indices
    const float r,           // radius of aabb
    const bool init,         // first run?
    float* d,                // distance
    bool* cond,              // true if hit
    int* pidx,               // index of 3d coord array
    const int num_nuggets,   // # of nugget indices
    const int n){            // # of active nugget indices

    const int threads = 128;
    const int blocks = (n + threads - 1) / threads;
    ray_aabb_kernel<<<blocks, threads>>>(
        query, ray_d, ray_inv, nuggets, points, info, info_idxes, r, init, d, cond, pidx, num_nuggets, n);
}

}  // namespace kaolin
