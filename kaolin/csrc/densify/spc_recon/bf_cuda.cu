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

inline __device__ int Identify(
  const short3 	k,
  const uint 		Level,
  const uint* 	Exsum,
  const uchar* 	Oroot,
  const uchar* 	Eroot,
  const uint 		offset)
{
  int maxval = (0x1 << Level) - 1;
  if (k.x < 0 || k.y < 0 || k.z < 0 || k.x > maxval || k.y > maxval || k.z > maxval)
    return -1;

  int ord = 0;
  int prev = 0;
  for (uint l = 0; l < Level; l++)
  {
    uint depth = Level - l - 1;
    uint mask = (0x1 << depth);
    uint child_idx = ((mask&k.x) << 2 | (mask&k.y) << 1 | (mask&k.z)) >> depth;
    uint bits = (uint)Oroot[ord];
    uint mpty = (uint)Eroot[ord];

    // count set bits up to child - inclusive sum
    uint cnt = __popc(bits&((0x2 << child_idx) - 1));
    ord = Exsum[prev];

    // if bit set, keep going
    if (bits&(0x1 << child_idx))
    {
      ord += cnt;

      if (depth == 0)
        return ord - offset;
    }
    else
    {
      if (mpty&(0x1 << child_idx))
        return -2 - depth;
      else
        return -1;
    }

    prev = ord;
  }

  return ord; // only if called with Level=0
}

__global__ void d_CompacifyNodes(uint num, uint* sum, 
                  uchar* in_occ, uchar* in_empty, 
                  uchar* out_occ, uchar* out_empty)
{
  uint tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx < num)
  {
    uint new_index = (tidx == 0u) ? 0 : sum[tidx-1];
    if (new_index != sum[tidx])
    {
      out_occ[new_index] = in_occ[tidx];
      out_empty[new_index] = in_empty[tidx];
    }
  }
}

void CompactifyNodes_cuda(uint num_nodes, uint* d_insum, uchar* d_occ_ptr, uchar* d_emp_ptr, uchar* d_octree, uchar* d_empty)
{

	if (num_nodes == 0) return;

	d_CompacifyNodes << <(num_nodes + 1023) / 1024, 1024 >> > (num_nodes, d_insum, d_occ_ptr, d_emp_ptr, d_octree, d_empty);

  AT_CUDA_CHECK(cudaGetLastError());

}

__global__ void d_OracleB(
	uint num, 
	point_data* points, 
	float4x4 T,  
    float sigma,
    float* dimg,
    int depth_height,
    int depth_width,
    int num_levels,
	float2* mip,
    int hw,
	uint* occupied, 
	uint* state)
{
	uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < num)
	{
    point_data V = points[tidx];

    uint result = 1;
    uint result2 = 1;

    // x,y coordinates in pixel space
    float3 minExtent, maxExtent;
    voxel_extent(V, T, &minExtent, &maxExtent);

    // frustum cull
    if ((minExtent.x >= 0.0f) && (maxExtent.x < depth_width) && (minExtent.y >= 0.0f) && (maxExtent.y < depth_height) && minExtent.z > NEAR_CLIPPING)
    {
      // compute mip level
      uint miplevel = max(ceil(log2(max(maxExtent.x - minExtent.x, maxExtent.y - minExtent.y))), 0.0);

      if (miplevel <= num_levels)
      {
        float v0 = minExtent.z;
        float v1 = maxExtent.z;

        // scale according to miplevel
        float adaptInv = 1.0f / pow(2, miplevel);

        uint xmin = (uint)(adaptInv*minExtent.x);
        uint ymin = (uint)(adaptInv*minExtent.y);
        uint xmax = (uint)(adaptInv*maxExtent.x);
        uint ymax = (uint)(adaptInv*maxExtent.y);

        uint stride = (uint)(adaptInv*depth_width);

        float z0, z1;

        if (miplevel > 0)
        {
          uint64_t offset = (uint64_t)(hw*(pow(4, num_levels-miplevel)-1)/3);
          float2* dmap = mip + offset;

          float2 d00 = dmap[ymin*stride + xmin];
          float2 d10 = dmap[ymin*stride + xmax];
          float2 d01 = dmap[ymax*stride + xmin];
          float2 d11 = dmap[ymax*stride + xmax];

          z0 = fmin(fmin(d00.x, d10.x), fmin(d01.x, d11.x)) - sigma;
          z1 = fmax(fmax(d00.y, d10.y), fmax(d01.y, d11.y)) + 2.0f*sigma;
        }
        else
        {
          float d00 = dimg[ymin*stride + xmin];
          float d10 = dimg[ymin*stride + xmax];
          float d01 = dimg[ymax*stride + xmin];
          float d11 = dimg[ymax*stride + xmax];

          z0 = fmin(fmin(d00, d10), fmin(d01, d11)) - sigma;
          z1 = fmax(fmax(d00, d10), fmax(d01, d11)) + 2.0f*sigma;
        }

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

void OracleB_cuda(uint num, 
                  point_data* points, 
                  float4x4 T, 
                  float sigma, 
                  float* Dmap, 
                  int depth_height,
                  int depth_width,
                  int mip_levels,
                  float2* mip, 
                  int hw,
                  uint* occ, 
                  uint* estate)
{
    if (num == 0) return;

	d_OracleB << <(num + 1023) / 1024, 1024 >> > (num, points, T, sigma, Dmap, depth_height, depth_width, mip_levels, mip, hw, occ, estate);

    (cudaGetLastError());
}

__global__ void d_OracleBFinal(
	uint num, 
	point_data* points, 
	float4x4 T,  
  float one_over_sigma,
  float* Dmap, 
  int depth_height,
  int depth_width,
    uint* occupied, 
  uint* state, 
  float* out_probs,
  cudaTextureObject_t		ProfileCurve)
{
	uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < num)
	{
    point_data V = points[tidx];
    float4 P[8];
    transform_corners(V, T, P);

    float pmin = 1.0f;
    float pmax = -1.0f;
    for (uint idx = 0; idx < 8; idx++)
    {
      float3 q = make_float3(P[idx].x / P[idx].z, P[idx].y / P[idx].z, P[idx].z);
      float prob = 0.5f;

      if ((q.x >= 0.0f) && (q.x <= depth_width) && 
          (q.y >= 0.0f) && (q.y <= depth_height) && q.z > NEAR_CLIPPING)
      {
        uint x = (uint)q.x;
        uint y = (uint)q.y;

        float d = Dmap[y*depth_width + x];
        prob = BQ(ProfileCurve, one_over_sigma*(q.z-d));
      }

      pmin = fmin(pmin, prob);
      pmax = fmax(pmax, prob);
      if (idx == 0) out_probs[tidx] = prob;
    }

    if (pmax == 0.0f)  //empty
    {
      occupied[tidx] = 0;
      state[tidx] = 0;
    }
    else if (pmin == 0.5f && pmax == 0.5f) // unseen
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

void OracleBFinal_cuda(
  int num, 
  point_data* points, 
  float4x4 T, 
  float one_over_sigma,
  float* Dmap, 
  int depth_height,
  int depth_width,
  uint* occ, 
  uint* estate, 
  float* out_probs,
  cudaTextureObject_t		ProfileCurve)
{
	if (num == 0) return;

	d_OracleBFinal << <(num + 1023) / 1024, 1024 >> > (num, points, T, one_over_sigma, Dmap, depth_height, depth_width, occ, estate, out_probs, ProfileCurve);

  AT_CUDA_CHECK(cudaGetLastError());
}

__global__ void d_ProcessFinalVoxels(
  uint 	num_nodes, 
  uint* 	in_state, 
  uint* 	in_nvsum, 
  uint* 	out_occ, 
  uint* 	out_state, 
  uchar* 	out_octree, 
  uchar* 	out_empty)
{
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_nodes)
    return;

  uint base = 8 * index;
  uint cnt = 0;
  uint occupancy = 0;
  uint emptiness = 0;

  for (int i=0; i < 8; i++)
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

  uint up_index = in_nvsum[index];

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

void ProcessFinalVoxels_cuda(uint num_nodes, uint* state, uint* nvsum, uint* occup,  uint* prev_state, uchar* octree, uchar* empty)
{
	if (num_nodes == 0) return;

	d_ProcessFinalVoxels << <(num_nodes + 1023) / 1024, 1024 >> > (num_nodes, state, nvsum, occup, prev_state, octree, empty);

  AT_CUDA_CHECK(cudaGetLastError());
}

__global__ void d_ColorsBFinal(
	uint num, 
	point_data* points, 
	float4x4 T,  
  float one_over_sigma,
  float3* image, 
  float* Dmap, 
  int depth_height,
  int depth_width,
  float* probs,
  uchar4* out_colors,
  float3* out_normals,
  cudaTextureObject_t		ProfileCurve)
{
	uint tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < num)
	{
    point_data V = points[tidx];

    float4 Q = make_float4((float)V.x, (float)V.y, (float)V.z, 1.0f) * T;
    float3 q = make_float3(Q.x / Q.z, Q.y / Q.z, Q.z);

    uint x = (uint)q.x;
    uint y = (uint)q.y;


    if ((x > 0) && (x < depth_width-1) && (y > 0) && (y < depth_height-1) && q.z > NEAR_CLIPPING) //inbounds
    {

      float prob = probs[tidx];

      if (prob == 0.0f)
      {
        out_colors[tidx] = make_uchar4(0, 0, 0, 0);
        out_normals[tidx] = make_float3(0.0f, 0.0f, 0.0f);
      }
      else if (prob == 0.5f)
      {
        out_colors[tidx] = make_uchar4(64, 64, 64, 64);
        out_normals[tidx] = make_float3(0.0f, 0.0f, 0.0f);
      }
      else
      {
        float3 color = image[y*depth_width + x];
        uchar r = (uchar)(255*color.x);
        uchar g = (uchar)(255*color.y);
        uchar b = (uchar)(255*color.z);
        out_colors[tidx] = make_uchar4(b, g, r, 0);

        // gradient
        uint imx = max(x - 1, 0);
        uint imy = max(y - 1, 0);
        uint ipx = min(x + 1, depth_width);
        uint ipy = min(y + 1, depth_height);
        
        float d00 = Dmap[y*depth_width + x];
        float dp0 = Dmap[y*depth_width + ipx];
        float dm0 = Dmap[y*depth_width + imx];
        float d0p = Dmap[ipy*depth_width + x];
        float d0m = Dmap[imy*depth_width + x];

        float du = 0.5f*(dp0 - dm0);
        float dv = 0.5f*(d0p - d0m);

        float dprob = DBQ(ProfileCurve, one_over_sigma*(Q.z-d00));

        float zi = 1.0f/Q.z;
        float w = one_over_sigma*dprob*zi;

        float4 h;

        h.z = w*zi*(Q.z*Q.z + Q.x*du + Q.y*dv);
        h.x = -w * du;
        h.y = -w * dv;
        h.w = 0.0;

        // gradient in VC
        float4 f = T * h;
        float3 grad = make_float3(f.x, f.y, f.z);
        float3 normal = normalize(grad);
        out_normals[tidx] = normal;

      }
    }
    else
    {
        out_colors[tidx] = make_uchar4(64, 64, 64, 64);
        out_normals[tidx] = make_float3(0.0f, 0.0f, 0.0f);
    }
  }
}

void ColorsBFinal_cuda(
  int num, 
  point_data* points, 
  float4x4 T, 
  float one_over_sigma,
  float3* image, 
  float* Dmap, 
  int depth_height,
  int depth_width,
  float* probs,
  uchar4* out_colors,
  float3* out_normals,
  cudaTextureObject_t		ProfileCurve)
{
	if (num == 0) return;

	d_ColorsBFinal << <(num + 1023) / 1024, 1024 >> > (num, points, T, one_over_sigma, image, Dmap, depth_height, depth_width, probs, out_colors, out_normals, ProfileCurve);

  AT_CUDA_CHECK(cudaGetLastError());
}

__global__ void d_MergeEmpty(
  const uint num, 
  const point_data* d_points, 
  const uint  level,
  const uchar* d_octree0, 
  const uchar* d_octree1, 
  const uchar* d_empty0, 
  const uchar* d_empty1, 
  const uint* d_exsum0, 
  const uint* d_exsum1, 
	uint* occupied, 
	uint* state)
{
  uint tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx < num)
  {

    point_data V = d_points[tidx];

    int id0 = Identify(V, level, d_exsum0, d_octree0, d_empty0, 0);
    int id1 = Identify(V, level, d_exsum1, d_octree1, d_empty1, 0);

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

void MergeEmpty_cuda(
  uint num, 
  point_data* d_points, 
  uint level, 
  uchar* d_octree0, 
  uchar* d_octree1, 
  uchar* d_empty0, 
  uchar* d_empty1, 
  uint* d_exsum0, 
  uint* d_exsum1, 
  uint* occ,
  uint* estate)
{
	if (num == 0) return;

	d_MergeEmpty << <(num + 1023) / 1024, 1024 >> > (num, d_points, level, d_octree0, d_octree1, d_empty0, d_empty1, d_exsum0, d_exsum1, occ, estate);

  AT_CUDA_CHECK(cudaGetLastError());
}

__global__ void d_BQMerge(
  const uint num, 
  const point_data* d_points, 
  const uint  level,
  const uchar* d_octree0, 
  const uchar* d_octree1, 
  const uchar* d_empty0, 
  const uchar* d_empty1, 
  const float* d_probs0,
  const float* d_probs1,
  const uchar4* d_colors0,
  const uchar4* d_colors1,  
  float3* d_normals0,
  float3* d_normals1,  
  const uint* d_exsum0, 
  const uint* d_exsum1, 
  const uint offset0,
  const uint offset1,
	uint* occupied, 
	uint* state,
  float* d_out_probs,
  uchar4* d_out_colors)
{
  uint tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx < num)
  {
    point_data V = d_points[tidx];

    int id0 = Identify(V, level, d_exsum0, d_octree0, d_empty0, offset0);
    int id1 = Identify(V, level, d_exsum1, d_octree1, d_empty1, offset1);

    float p0 = id0 >= 0 ? d_probs0[id0] : id0 < -1 ? 0.5f : 0.0f;
    float p1 = id1 >= 0 ? d_probs1[id1] : id1 < -1 ? 0.5f : 0.0f;   

    float p = p0*p1 / (p0*p1 + (1.0f-p0)*(1.0f-p1));

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
      d_out_colors[tidx] = id0 >= 0 ? d_colors0[id0] : d_colors1[id1];
      d_out_probs[tidx] = p;  
    }

  }
}

void BQMerge_cuda(
  uint num, 
  point_data* d_points, 
  uint level, 
  uchar* d_octree0, 
  uchar* d_octree1, 
  uchar* d_empty0, 
  uchar* d_empty1, 
  float* d_probs0,
  float* d_probs1,
  uchar4* d_color0,
  uchar4* d_color1,
  float3* d_normals0,
  float3* d_normals1,  
  uint* d_exsum0, 
  uint* d_exsum1, 
  uint offset0,
  uint offset1,
  uint* occ,
  uint* estate,
  float* d_out_probs,
  uchar4* d_out_colors)
{
	if (num == 0) return;

	d_BQMerge << <(num + 1023) / 1024, 1024 >> > (num, d_points, level, d_octree0, d_octree1, d_empty0, d_empty1, 
                                                d_probs0, d_probs1, d_color0, d_color1, d_normals0, d_normals1, 
                                                d_exsum0, d_exsum1, offset0, offset1, occ, estate, d_out_probs, d_out_colors);

  AT_CUDA_CHECK(cudaGetLastError());
}

__global__ void d_TouchExtract(
  uint 	num_nodes, 
  uint* 	in_state, 
  uint* 	in_nvsum, 
  uint* 	out_state)
{
  uint index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_nodes)
    return;

  uint up_index = in_nvsum[index];
  out_state[up_index] = in_state[index];

}

void TouchExtract_cuda(uint num, uint* state, uint* nvsum, uint* prev_state)
{
	if (num == 0) return;

	d_TouchExtract << <(num + 1023) / 1024, 1024 >> > (num, state, nvsum, prev_state);

  AT_CUDA_CHECK(cudaGetLastError());
}

__global__ void d_BQExtract(
  uint num, 
  point_data* d_points, 
  uint level, 
  uchar* d_octree, 
  uchar* d_empty, 
  float* d_probs,
  uint* d_exsum, 
  uint offset,
  uint* occupied,
  uint* state)
{
  uint tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx < num)
  {
    point_data V = d_points[tidx];

    float pmin = 1;
    float pmax = -1;

    for (uint i = 0; i < 8; i++)
    {
      point_data P = make_point_data(V.x + (i >> 2), V.y + ((i >> 1) & 0x1), V.z + (i & 0x1));

      int id = Identify(P, level, d_exsum, d_octree, d_empty, offset);

      float prob;
      if (id >= 0)
        prob = d_probs[id];
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

void BQExtract_cuda(
  uint num, 
  point_data* d_points, 
  uint level, 
  uchar* d_octree, 
  uchar* d_empty, 
  float* d_probs,
  uint* d_exsum, 
  uint offset,
  uint* occ,
  uint* estate)
{
	if (num == 0) return;

	d_BQExtract << <(num + 1023) / 1024, 1024 >> > (num, d_points, level, d_octree, d_empty, d_probs, d_exsum, offset, occ, estate);

  AT_CUDA_CHECK(cudaGetLastError());
}

__global__ void d_BQTouch(
  uint num, 
  uchar* d_octree, 
  uchar* d_empty, 
  uint* occupied,
  uint* state)
{
  uint tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx < num)
  {
    uchar obits = d_octree[tidx];
    uchar ebits = d_empty[tidx];

    uint bidx = 8*tidx;

    for (uint i = 0; i < 8; i++)
    {
      uint o = obits&(0x1 << i) ? 1 : 0;
      uint e = ebits&(0x1 << i) ? 1 : 0;

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

void BQTouch_cuda(
  uint num, 
  uchar* d_octree, 
  uchar* d_empty, 
  uint* occ,
  uint* estate)
{
	if (num == 0) return;

	d_BQTouch << <(num + 1023) / 1024, 1024 >> > (num, d_octree, d_empty, occ, estate);

  AT_CUDA_CHECK(cudaGetLastError());
}

}  // namespace kaolin

