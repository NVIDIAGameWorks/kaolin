// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

#include <torch/extension.h>
#include <stdio.h>

#include <math_constants.h>

#define CUB_STDERR
#include <cub/device/device_scan.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include "../../spc_math.h"
#include "../../spc_utils.cuh"

namespace kaolin {

using namespace cub;
using namespace std;

#define NEAR_CLIPPING 0.15

// definitions used in profile curve evaluation
#define TWOFIFTYFIVE_OVER_NINTYSIX 2.65625f

// Bayesian Fusion profile curve using quadratic B-spline noise model
__device__ inline float BQ(cudaTextureObject_t ProfileCurve, float x)
{
	if (x <= -3.0f)
	{
		return 0.0f;
	}
	else if (x >= 6.0f)
	{
		return 0.5f;
	}
	else
	{
		float u = x + 3.0f;
		float iu = truncf(u);
		float t = u - iu;
		float s = 1.0f - t;

		// read in Bezier ordinates of needed curve segment
		float4 C = tex1D<float4>(ProfileCurve, iu);

		// evaluate Bernstein-Bezier basis functions, multiple by ordinates, and scale result
		return TWOFIFTYFIVE_OVER_NINTYSIX * (s*s*(s*C.x + 3.0f*t*C.y) + t*t*(3.0f*s*C.z + t*C.w));
	}
}

// Derivative of Bayesian Fusion profile curve using quadratic B-spline noise model
__device__ inline float DBQ(cudaTextureObject_t ProfileCurve, float x)
{
	if (x <= -3.0f)
	{
		return 0.0;
	}
	else if (x >= 6.0f)
	{
		return 0.0f;
	}
	else
	{
		float u = x + 3.0f;
		float iu = truncf(u);
		float t = u - iu;
		float s = 1.0f - t;

		// read in and scale Bezier ordinates of needed curve segment
		float4 cp0 = tex1D<float4>(ProfileCurve, iu);

		/// deCastlejau's Algorithm
		float3 cp1 = make_float3(s*cp0.x + t*cp0.y, s*cp0.y + t*cp0.z, s*cp0.z + t*cp0.w);
		float2 cp2 = make_float2(s*cp1.x + t*cp1.y, s*cp1.y + t*cp1.z);

    return TWOFIFTYFIVE_OVER_NINTYSIX*3.0*(cp2.y - cp2.x);
	}
}


__global__ void d_CompacifyNodes(uint32_t num, uint32_t* sum, 
                  uint8_t* in_occ, uint8_t* in_empty, 
                  uint8_t* out_occ, uint8_t* out_empty)
{
  uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx < num)
  {
    uint32_t new_index = (tidx == 0u) ? 0 : sum[tidx-1];
    if (new_index != sum[tidx])
    {
      out_occ[new_index] = in_occ[tidx];
      out_empty[new_index] = in_empty[tidx];
    }
  }
}

void compactify_nodes_cuda(uint32_t num_nodes, uint32_t* d_insum, uint8_t* d_occ_ptr, uint8_t* d_emp_ptr, uint8_t* d_octree, uint8_t* d_empty)
{

	if (num_nodes == 0) return;

	d_CompacifyNodes << <(num_nodes + 1023) / 1024, 1024 >> > (num_nodes, d_insum, d_occ_ptr, d_emp_ptr, d_octree, d_empty);

  AT_CUDA_CHECK(cudaGetLastError());

}


__global__ void d_OracleB(
	const uint32_t num, 
	const point_data* points, 
	const float4x4 T,  
  const float sigma,
        float2* mip,
  const int32_t depth_height,
  const int32_t depth_width,
  const int32_t num_levels,
  const int32_t hw,
	      uint32_t* occupied, 
	      uint32_t* state)
{
	uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < num)
	{
    point_data V = points[tidx];

    uint32_t result = 1;
    uint32_t result2 = 1;

    // x,y coordinates in pixel space
    float3 minExtent, maxExtent;
    voxel_extent(V, T, &minExtent, &maxExtent);

    // frustum cull
    if ((minExtent.x >= 0.0f) && (maxExtent.x < depth_width) && (minExtent.y >= 0.0f) && (maxExtent.y < depth_height) && minExtent.z > NEAR_CLIPPING)
    {
      // compute mip level
      uint32_t miplevel = max(ceil(log2(max(maxExtent.x - minExtent.x, maxExtent.y - minExtent.y))), 0.0);

      if (miplevel < num_levels)
      {
        float v0 = minExtent.z;
        float v1 = maxExtent.z;

        // scale according to miplevel
        float adaptInv = 1.0f / pow(2, miplevel);

        uint32_t xmin = (uint32_t)(adaptInv*minExtent.x);
        uint32_t ymin = (uint32_t)(adaptInv*minExtent.y);
        uint32_t xmax = (uint32_t)(adaptInv*maxExtent.x);
        uint32_t ymax = (uint32_t)(adaptInv*maxExtent.y);

        uint32_t stride = (uint32_t)(adaptInv*depth_width);

        float z0, z1;

        uint64_t offset = (uint64_t)(hw*(pow(4, num_levels-miplevel-1)-1)/3);
        float2* dmap = mip + offset;

        float2 d00 = dmap[ymin*stride + xmin];
        float2 d10 = dmap[ymin*stride + xmax];
        float2 d01 = dmap[ymax*stride + xmin];
        float2 d11 = dmap[ymax*stride + xmax];

        z0 = fmin(fmin(d00.x, d10.x), fmin(d01.x, d11.x)) - sigma;
        z1 = fmax(fmax(d00.y, d10.y), fmax(d01.y, d11.y)) + 2.0f*sigma;

        result = z0 <= v1 && v0 <= z1 ? 1 : 0;

        if (z0 > v1)
          result2 = 0; //empty
        else if (z1 < v0)
          result2 = 1; // unseen
        else
          result2 = 2; // occupied

      }
      else //too high up the pyramid
      {
        result = 1; // keep, and split for next time
        result2 = 2;
      }
    }
    else
    {
      if ((maxExtent.x < 0.0f) || (minExtent.x > depth_width) || (maxExtent.y < 0.0f) || (minExtent.y > depth_height) || (maxExtent.z < NEAR_CLIPPING))
      {
        result = 0;
        result2 = 1;
      }
      else
      {
        result = 1;
        result2 = 2;
      }
    }

  	occupied[tidx] = result;
  	state[tidx] = result2;
  }
}
  

void oracleB_cuda(uint32_t num, 
                  point_data* points, 
                  float4x4 T, 
                  float sigma, 
                  float2* mip, 
                  int32_t depth_height,
                  int32_t depth_width,
                  int32_t mip_levels,
                  int32_t hw,
                  uint32_t* occ, 
                  uint32_t* estate)
{
    if (num == 0) return;

	  d_OracleB << <(num + 1023) / 1024, 1024 >> > (num, points, T, sigma, mip, depth_height, depth_width, mip_levels, hw, occ, estate);

    AT_CUDA_CHECK(cudaGetLastError());
}


__global__ void d_OracleBFinal(
	const uint32_t num, 
	const point_data* __restrict__ points, 
	const float4x4 T,  
  const float one_over_sigma,
  const float* __restrict__ dmap, 
  float2* __restrict__ mipmap, 
  const int32_t depth_height,
  const int32_t depth_width,
  uint32_t* __restrict__ occupied, 
  uint32_t* __restrict__ state, 
  float* __restrict__ out_probs,
  const cudaTextureObject_t		ProfileCurve,
  const float scale,
  const int32_t num_levels,
  const int32_t hw)
{
	uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < num)
	{
    point_data V = points[tidx];
    float prob = 0.5f;

    // x,y coordinates in pixel space
    float3 minExtent, maxExtent;
    voxel_extent(V, T, &minExtent, &maxExtent);

    // frustum cull
    if ((minExtent.x >= 0.0f) && (maxExtent.x < depth_width) && (minExtent.y >= 0.0f) && (maxExtent.y < depth_height) && minExtent.z > NEAR_CLIPPING)
    {
      // compute mip level
      uint32_t miplevel = max(ceil(log2(max(maxExtent.x - minExtent.x, maxExtent.y - minExtent.y))), 0.0);

      if (miplevel < num_levels)
      {
        float4 P = make_float4((float)V.x, (float)V.y, (float)V.z, 1.0f) * T;
        float3 q = make_float3(P.x / P.z, P.y / P.z, P.z);

        uint32_t x = (uint32_t)floorf(q.x);
        uint32_t y = (uint32_t)floorf(q.y);

        float d = dmap[y*depth_width + x];

        // scale according to miplevel
        float adaptInv = 1.0f / pow(2, miplevel);

        uint32_t xmin = (uint32_t)(adaptInv*minExtent.x);
        uint32_t ymin = (uint32_t)(adaptInv*minExtent.y);
        uint32_t xmax = (uint32_t)(adaptInv*maxExtent.x);
        uint32_t ymax = (uint32_t)(adaptInv*maxExtent.y);

        uint32_t stride = (uint32_t)(adaptInv*depth_width);
        uint64_t offset = (uint64_t)(hw*(pow(4, num_levels-miplevel-1)-1)/3);
        float2* mmmap = mipmap + offset;

        float2 d00 = mmmap[ymin*stride + xmin];
        float2 d10 = mmmap[ymin*stride + xmax];
        float2 d01 = mmmap[ymax*stride + xmin];
        float2 d11 = mmmap[ymax*stride + xmax];

        float z0 = fmin(fmin(d00.x, d10.x), fmin(d01.x, d11.x));
        float z1 = fmax(fmax(d00.y, d10.y), fmax(d01.y, d11.y));

        if (z1-z0 < 10*scale)
          prob = BQ(ProfileCurve, one_over_sigma*(q.z-d));

      }
      else //too high up the pyramid
      {
        // this should not happen
      }
    }

    out_probs[tidx] = prob;
  }
}
  

void oracleB_final_cuda(
  int32_t num, 
  point_data* points, 
  float4x4 T, 
  float one_over_sigma,
  float* dmap, 
  float2* mipmap, 
  int32_t depth_height,
  int32_t depth_width,
  uint32_t* occ, 
  uint32_t* estate, 
  float* out_probs,
  cudaTextureObject_t		ProfileCurve,
  float scale,
  int32_t mip_levels,
  int32_t hw)
{
	if (num == 0) return;

	d_OracleBFinal << <(num + 1023) / 1024, 1024 >> > (
    num, points, T, one_over_sigma, dmap, mipmap, depth_height, depth_width, 
    occ, estate, out_probs, ProfileCurve, scale, mip_levels, hw);

  AT_CUDA_CHECK(cudaGetLastError());
}


__global__ void d_ProcessFinalVoxels(
  uint32_t 	num_nodes, 
  uint32_t* 	in_state, 
  uint32_t* 	in_nvsum, 
  uint32_t* 	out_occ, 
  uint32_t* 	out_state, 
  uint8_t* 	out_octree, 
  uint8_t* 	out_empty)
{
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_nodes)
    return;

  uint32_t base = 8 * index;
  uint32_t cnt = 0;
  uint32_t occupancy = 0;
  uint32_t emptiness = 0;

  for (int32_t i=0; i < 8; i++)
  {
    if (in_state[base+i] == 2)
    {
      occupancy |= 1 << i;
      emptiness |= 1 << i;
      cnt++;	
    }
    else if (in_state[base+i] == 1)
    {
      emptiness |= 1 << i;
    }
  }

  uint32_t up_index = in_nvsum[index];

  if (cnt == 0)
  {
    out_occ[index] = 0;
    out_state[up_index] = emptiness == 0 ? 0 : 1;
  }
  else
  {
    out_occ[index] = 1;
    out_state[up_index] = 2; // added line
  }

  out_octree[index] = occupancy;
  out_empty[index] = emptiness;
}

void process_final_voxels_cuda(uint32_t num_nodes, uint32_t* state, uint32_t* nvsum, uint32_t* occup,  uint32_t* prev_state, uint8_t* octree, uint8_t* empty)
{
	if (num_nodes == 0) return;

	d_ProcessFinalVoxels << <(num_nodes + 1023) / 1024, 1024 >> > (num_nodes, state, nvsum, occup, prev_state, octree, empty);

  AT_CUDA_CHECK(cudaGetLastError());
}


__global__ void d_ColorsBFinal(
	const uint32_t num, 
	const point_data* points, 
	const float4x4 T,  
  const float3* image, 
  const float* Dmap, 
  const float max_depth,
  const int32_t depth_height,
  const int32_t depth_width,
  float3* out_colors,
  float4* out_normals)
{
	uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < num)
	{
    out_colors[tidx] = make_float3(0.0f, 0.0f, 0.0f);
    out_normals[tidx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    point_data V = points[tidx];

    float4 Q = make_float4((float)V.x+0.5f, (float)V.y+0.5f, (float)V.z+0.5f, 1.0f) * T; // voxel center
    float3 q = make_float3(Q.x / Q.z, Q.y / Q.z, Q.z);

    uint32_t x = (uint32_t)q.x;
    uint32_t y = (uint32_t)q.y;

    if ((x > 0) && (x < depth_width-1) && (y > 0) && (y < depth_height-1) && q.z > NEAR_CLIPPING) //inbounds
    {
      float3 color = image[y*depth_width + x];

      // gradient
      uint32_t imx = max(x - 1, 0);
      uint32_t imy = max(y - 1, 0);
      uint32_t ipx = min(x + 1, depth_width);
      uint32_t ipy = min(y + 1, depth_height);
      
      // float d00 = Dmap[y*depth_width + x];
      float dp0 = Dmap[y*depth_width + ipx];
      float dm0 = Dmap[y*depth_width + imx];
      float d0p = Dmap[ipy*depth_width + x];
      float d0m = Dmap[imy*depth_width + x];

      if (dp0 != max_depth && dm0 != max_depth && d0p != max_depth && d0m != max_depth)
      {
          float du = 0.5f*(dp0 - dm0);
          float dv = 0.5f*(d0p - d0m);

          float4 h = make_float4(-du, -dv, 1.0f, 0.0f);

          float z = powf(rsqrt(du*du + dv*dv + 1.0f), 4.0); // z component of normal in camera space

          // gradient in VC
          float4 f = T * h;
          float3 normal = normalize(make_float3(f.x, f.y, f.z));

          out_normals[tidx] = make_float4(z*normal.x, z*normal.y, z*normal.z, z); 
          // out_colors[tidx] = z*color;
          out_colors[tidx] = make_float3(z*color.z, z*color.y, z*color.x); 
      }
    }
  }
}


void colorsB_final_cuda(
  const int32_t num, 
  const point_data* points, 
  const float4x4 T, 
  const float3* image, 
  const float* Dmap, 
  const float max_depth,
  const int32_t depth_height,
  const int32_t depth_width,
  float3* out_colors,
  float4* out_normals)
{
	if (num == 0) return;

	d_ColorsBFinal << <(num + 1023) / 1024, 1024 >> > (
    num, 
    points, 
    T, 
    image, 
    Dmap, 
    max_depth,
    depth_height, 
    depth_width, 
    out_colors, 
    out_normals);

  AT_CUDA_CHECK(cudaGetLastError());
}


__global__ void d_MergeEmpty(
  const uint32_t num, 
  const point_data* d_points, 
  const uint32_t  level,
  const uint8_t* d_octree0, 
  const uint8_t* d_octree1, 
  const uint8_t* d_empty0, 
  const uint8_t* d_empty1, 
  const int32_t* d_exsum0, 
  const int32_t* d_exsum1, 
	uint32_t* occupied, 
	uint32_t* state)
{
  uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx < num)
  {

    point_data V = d_points[tidx];

    int32_t id0 = identify(V, level, d_exsum0, d_octree0, d_empty0);
    int32_t id1 = identify(V, level, d_exsum1, d_octree1, d_empty1);

    if (id0 == -1 || id1 == -1) // empty 
    {
      occupied[tidx] = 0;
      state[tidx] = 0;   
    }
    else if (id0 < -1 && id1 < -1) // unseen 
    {
      occupied[tidx] = 0;
      state[tidx] = 1;   
    }
    else // occupied
    {
      occupied[tidx] = 1;
      state[tidx] = 2;     
    }
  }
}


void merge_empty_cuda(
  uint32_t num, 
  point_data* d_points, 
  uint32_t level, 
  uint8_t* d_octree0, 
  uint8_t* d_octree1, 
  uint8_t* d_empty0, 
  uint8_t* d_empty1, 
  int32_t* d_exsum0, 
  int32_t* d_exsum1, 
  uint32_t* occ,
  uint32_t* estate)
{
	if (num == 0) return;

	d_MergeEmpty << <(num + 1023) / 1024, 1024 >> > (num, d_points, level, d_octree0, d_octree1, d_empty0, d_empty1, d_exsum0, d_exsum1, occ, estate);

  AT_CUDA_CHECK(cudaGetLastError());
}


__global__ void d_BQMerge(
  const uint32_t num, 
  const point_data* d_points, 
  const uint32_t  level,
  const uint8_t* d_octree0, 
  const uint8_t* d_octree1, 
  const uint8_t* d_empty0, 
  const uint8_t* d_empty1, 
  const float* d_probs0,
  const float* d_probs1,
  const float3* d_colors0,
  const float3* d_colors1,  
  const float4* d_normals0,
  const float4* d_normals1,  
  const int32_t* d_exsum0, 
  const int32_t* d_exsum1, 
  const uint32_t offset0,
  const uint32_t offset1,
	uint32_t* occupied, 
	uint32_t* state,
  float* d_out_probs,
  float3* d_out_colors,
  float4* d_out_normals)
{
  uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx < num)
  {
    point_data V = d_points[tidx];

    int32_t id0 = identify(V, level, d_exsum0, d_octree0, d_empty0);
    int32_t id1 = identify(V, level, d_exsum1, d_octree1, d_empty1);

    if (id0 == -1 || id1 == -1) // empty 
    {
      occupied[tidx] = 0;
      state[tidx] = 0;   
    }
    else if (id0 < -1 && id1 < -1) // unseen 
    {
      occupied[tidx] = 0;
      state[tidx] = 1;   
    }
    else // occupied
    {
      occupied[tidx] = 1;
      state[tidx] = 2;     

      int32_t idx0 = id0 - offset0;
      int32_t idx1 = id1 - offset1;

      if (id0 < -1) // input 1 must be good
      {
        d_out_probs[tidx] = d_probs1[idx1];
        d_out_colors[tidx] = d_colors1[idx1];
        d_out_normals[tidx] = d_normals1[idx1];
      }
      else if (id1 < -1) // input 0 must be good
      {
        d_out_probs[tidx] = d_probs0[idx0];
        d_out_colors[tidx] = d_colors0[idx0];
        d_out_normals[tidx] = d_normals0[idx0];
      }
      else // both inputs are good, blend
      {
        float p0 = d_probs0[idx0];
        float p1 = d_probs1[idx1];
        float pn = p0*p1;
        float pd = (p0*p1 + (1.0f-p0)*(1.0f-p1));

        d_out_probs[tidx] = pn/pd;  

        d_out_colors[tidx] = d_colors0[idx0] + d_colors1[idx1];
        d_out_normals[tidx] = d_normals0[idx0] + d_normals1[idx1];
      }
    }
  }
}


void bq_merge_cuda(
  uint32_t num, 
  point_data* d_points, 
  uint32_t level, 
  uint8_t* d_octree0, 
  uint8_t* d_octree1, 
  uint8_t* d_empty0, 
  uint8_t* d_empty1, 
  float* d_probs0,
  float* d_probs1,
  float3* d_color0,
  float3* d_color1,
  float4* d_normals0,
  float4* d_normals1,  
  int32_t* d_exsum0, 
  int32_t* d_exsum1, 
  uint32_t offset0,
  uint32_t offset1,
  uint32_t* occ,
  uint32_t* estate,
  float* d_out_probs,
  float3* d_out_colors,
  float4* d_out_normals)
{
	if (num == 0) return;

	d_BQMerge << <(num + 1023) / 1024, 1024 >> > (num, d_points, level, d_octree0, d_octree1, d_empty0, d_empty1, 
                                                d_probs0, d_probs1, d_color0, d_color1, d_normals0, d_normals1, 
                                                d_exsum0, d_exsum1, offset0, offset1, occ, estate, d_out_probs, 
                                                d_out_colors, d_out_normals);
  AT_CUDA_CHECK(cudaGetLastError());
}


__global__ void d_TouchExtract(
  uint32_t 	num_nodes, 
  uint32_t* 	in_state, 
  uint32_t* 	in_nvsum, 
  uint32_t* 	out_state)
{
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_nodes)
    return;

  uint32_t up_index = in_nvsum[index];
  out_state[up_index] = in_state[index];

}


void touch_extract_cuda(uint32_t num, uint32_t* state, uint32_t* nvsum, uint32_t* prev_state)
{
	if (num == 0) return;

	d_TouchExtract << <(num + 1023) / 1024, 1024 >> > (num, state, nvsum, prev_state);

  AT_CUDA_CHECK(cudaGetLastError());
}


__global__ void d_BQExtract(
  uint32_t num, 
  point_data* d_points, 
  uint32_t level, 
  uint8_t* d_octree, 
  uint8_t* d_empty, 
  float* d_probs,
  int32_t* d_exsum, 
  uint32_t offset,
  uint32_t* occupied,
  uint32_t* state)
{
  uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx < num)
  {
    point_data V = d_points[tidx];

    float pmin = 1;
    float pmax = -1;

    for (uint32_t i = 0; i < 8; i++)
    {
      point_data P = make_point_data(V.x + (i >> 2), V.y + ((i >> 1) & 0x1), V.z + (i & 0x1));

      int32_t id = identify(P, level, d_exsum, d_octree, d_empty);

      float prob;
      if (id >= 0)
        prob = d_probs[id-offset];
      else if (id == -1)
        prob = 0.0f;
      else
        prob = 0.5f;

      pmin = fmin(pmin, prob);
      pmax = fmax(pmax, prob);
    }


    if (pmin < 0.5f && 0.5f < pmax) // occupied 
    {
      occupied[tidx] = 1;
      state[tidx] = 2;   
    }
    else if (pmax <= 0.5f) // empty 
    {
      occupied[tidx] = 0;
      state[tidx] = 0;   
    }
    else // unseen
    {
      occupied[tidx] = 0;
      state[tidx] = 1;     
    }
  }
}


void bq_extract_cuda(
  uint32_t num, 
  point_data* d_points, 
  uint32_t level, 
  uint8_t* d_octree, 
  uint8_t* d_empty, 
  float* d_probs,
  int32_t* d_exsum, 
  uint32_t offset,
  uint32_t* occ,
  uint32_t* estate)
{
	if (num == 0) return;

	d_BQExtract << <(num + 1023) / 1024, 1024 >> > (num, d_points, level, d_octree, d_empty, d_probs, d_exsum, offset, occ, estate);

  AT_CUDA_CHECK(cudaGetLastError());
}


__global__ void d_BQTouch(
  uint32_t num, 
  uint8_t* d_octree, 
  uint8_t* d_empty, 
  uint32_t* occupied,
  uint32_t* state)
{
  uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx < num)
  {
    uint8_t obits = d_octree[tidx];
    uint8_t ebits = d_empty[tidx];
    uint32_t bidx = 8*tidx;

    for (uint32_t i = 0; i < 8; i++)
    {
      uint32_t o = obits&(0x1 << i) ? 1 : 0;
      uint32_t e = ebits&(0x1 << i) ? 1 : 0;

      if (o == 0)
      {
        if (e == 0) // empty
        {
          occupied[bidx+i] = 0;
          state[bidx+i] = 0;
        }
        else // unseen
        {
          occupied[bidx+i] = 0;
          state[bidx+i] = 1;         
        }
      }
      else // occupied
      {
        occupied[bidx+i] = 1;
        state[bidx+i] = 2;
      }
    }
  }
}


void bq_touch_cuda(
  uint32_t num, 
  uint8_t* d_octree, 
  uint8_t* d_empty, 
  uint32_t* occ,
  uint32_t* estate)
{
	if (num == 0) return;

	d_BQTouch << <(num + 1023) / 1024, 1024 >> > (num, d_octree, d_empty, occ, estate);

  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace kaolin

