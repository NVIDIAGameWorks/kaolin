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
  const float* dimg,
  const int depth_height,
  const int depth_width,
  const int num_levels,	
  float2* mip,
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

      if (miplevel <= num_levels)
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
  

void oracleB_cuda(uint32_t num, 
                  point_data* points, 
                  float4x4 T, 
                  float sigma, 
                  float* Dmap, 
                  int32_t depth_height,
                  int32_t depth_width,
                  int32_t mip_levels,
                  float2* mip, 
                  int32_t hw,
                  uint32_t* occ, 
                  uint32_t* estate)
{
    if (num == 0) return;

	  d_OracleB << <(num + 1023) / 1024, 1024 >> > (num, points, T, sigma, Dmap, depth_height, depth_width, mip_levels, mip, hw, occ, estate);

    AT_CUDA_CHECK(cudaGetLastError());
}


__global__ void d_OracleBFinal(
	const uint32_t num, 
	const point_data* __restrict__ points, 
	const float4x4 T,  
  const float one_over_sigma,
  const float* __restrict__ Dmap, 
  const int32_t depth_height,
  const int32_t depth_width,
  uint32_t* __restrict__ occupied, 
  uint32_t* __restrict__ state, 
  float* __restrict__ out_probs,
  const cudaTextureObject_t		ProfileCurve)
{
	uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < num)
	{
    point_data V = points[tidx];

    float4 P[8];
    transform_corners(V, T, P);

    float pmin = 1.0f;
    float pmax = -1.0f;
    for (uint32_t idx = 0; idx < 8; idx++)
    {
      float3 q = make_float3(P[idx].x / P[idx].z, P[idx].y / P[idx].z, P[idx].z);
      float prob = 0.5f;

      if ((q.x >= 0.0f) && (q.x < depth_width) && 
          (q.y >= 0.0f) && (q.y < depth_height) && q.z > NEAR_CLIPPING)
      {
        uint32_t x = (uint32_t)q.x;
        uint32_t y = (uint32_t)q.y;

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

  
void oracleB_final_cuda(
  int32_t num, 
  point_data* points, 
  float4x4 T, 
  float one_over_sigma,
  float* dmap, 
  int32_t depth_height,
  int32_t depth_width,
  uint32_t* occ, 
  uint32_t* estate, 
  float* out_probs,
  cudaTextureObject_t		ProfileCurve)
{
	if (num == 0) return;

	d_OracleBFinal << <(num + 1023) / 1024, 1024 >> > (
    num, points, T, one_over_sigma, dmap, depth_height, depth_width, 
    occ, estate, out_probs, ProfileCurve);

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
  const float one_over_sigma,
  const float3* image, 
  const float* Dmap, 
  const int32_t depth_height,
  const int32_t depth_width,
  const float* probs,
  uchar4* out_colors,
  float3* out_normals,
  cudaTextureObject_t		ProfileCurve)
{
	uint32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (tidx < num)
	{
    point_data V = points[tidx];

    float4 Q = make_float4((float)V.x, (float)V.y, (float)V.z, 1.0f) * T;
    float3 q = make_float3(Q.x / Q.z, Q.y / Q.z, Q.z);

    uint32_t x = (uint32_t)q.x;
    uint32_t y = (uint32_t)q.y;

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
        uint32_t imx = max(x - 1, 0);
        uint32_t imy = max(y - 1, 0);
        uint32_t ipx = min(x + 1, depth_width);
        uint32_t ipy = min(y + 1, depth_height);
        
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


void colorsB_final_cuda(
  const int32_t num, 
  const point_data* points, 
  const float4x4 T, 
  const float one_over_sigma,
  const float3* image, 
  const float* dmap, 
  const int32_t depth_height,
  const int32_t depth_width,
  const float* probs,
  uchar4* out_colors,
  float3* out_normals,
  cudaTextureObject_t		ProfileCurve)
{
	if (num == 0) return;

	d_ColorsBFinal << <(num + 1023) / 1024, 1024 >> > (
    num, 
    points, 
    T, 
    one_over_sigma,
    image, 
    dmap, 
    depth_height, 
    depth_width, 
    probs,
    out_colors, 
    out_normals,
    ProfileCurve);

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
  const uchar* d_octree0, 
  const uint8_t* d_octree1, 
  const uint8_t* d_empty0, 
  const uint8_t* d_empty1, 
  const uint32_t offset0,
  const uint32_t offset1,
  const float* d_probs0,
  const float* d_probs1,
  const uchar4* d_colors0,
  const uchar4* d_colors1,  
  const float3* d_normals0,
  const float3* d_normals1,  
  const int32_t* d_exsum0, 
  const int32_t* d_exsum1, 
	uint32_t* occupied, 
	uint32_t* state,
  float* d_out_probs,
  uchar4* d_out_colors)
{
  uint32_t tidx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tidx < num)
  {
    point_data V = d_points[tidx];

    int32_t id0 = identify(V, level, d_exsum0, d_octree0, d_empty0);
    int32_t id1 = identify(V, level, d_exsum1, d_octree1, d_empty1);

    float p0 = id0 >= 0 ? d_probs0[id0-offset0] : id0 < -1 ? 0.5f : 0.0f;
    float p1 = id1 >= 0 ? d_probs1[id1-offset1] : id1 < -1 ? 0.5f : 0.0f;   

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
      d_out_colors[tidx] = id0 >= 0 ? d_colors0[id0-offset0] : d_colors1[id1-offset1];
      d_out_probs[tidx] = p;  
    }
  }
}


std::vector<at::Tensor>  bq_merge_cuda(
  at::Tensor points,
  uint32_t level,
  at::Tensor octree0,
  at::Tensor octree1,  
  at::Tensor empty0,
  at::Tensor empty1,
  at::Tensor pyramid0,
  at::Tensor pyramid1,
  at::Tensor probs0,
  at::Tensor probs1,  
  at::Tensor colors0,
  at::Tensor colors1,
  at::Tensor normals0,
  at::Tensor normals1,
  at::Tensor exsum0,
  at::Tensor exsum1)
{
  uint32_t num = points.size(0);

  at::Tensor occupancy = at::zeros({num}, points.options().dtype(at::kInt));
  at::Tensor empty_state = at::zeros({num}, points.options().dtype(at::kInt));
  at::Tensor out_colors = at::zeros({num, 4}, points.options().dtype(at::kByte));
  at::Tensor out_probs = at::zeros({num}, points.options().dtype(at::kFloat));
  at::Tensor out_normals = at::zeros({num, 3}, points.options().dtype(at::kFloat));

  auto pyramid0_a = pyramid0.accessor<int32_t, 3>();
  uint32_t offset0 = pyramid0_a[0][1][level];

  auto pyramid1_a = pyramid1.accessor<int32_t, 3>();
  uint32_t offset1 = pyramid1_a[0][1][level];
 
  if (num > 0)
    d_BQMerge << <(num + 1023) / 1024, 1024 >> > (
      num, 
      reinterpret_cast<point_data*>(points.data_ptr<short>()), 
      level, 
      octree0.data_ptr<uchar>(), 
      octree1.data_ptr<uchar>(), 
      empty0.data_ptr<uchar>(), 
      empty1.data_ptr<uchar>(), 
      offset0, offset1, 
      probs0.data_ptr<float>(), 
      probs1.data_ptr<float>(), 
      reinterpret_cast<uchar4*>(colors0.data_ptr<uchar>()), 
      reinterpret_cast<uchar4*>(colors1.data_ptr<uchar>()), 
      reinterpret_cast<float3*>(normals0.data_ptr<float>()),
      reinterpret_cast<float3*>(normals1.data_ptr<float>()), 
      exsum0.data_ptr<int32_t>(), 
      exsum1.data_ptr<int32_t>(), 
      reinterpret_cast<uint32_t*>(occupancy.data_ptr<int32_t>()), 
      reinterpret_cast<uint32_t*>(empty_state.data_ptr<int32_t>()), 
      out_probs.data_ptr<float>(), 
      reinterpret_cast<uchar4*>(out_colors.data_ptr<uchar>()));

  AT_CUDA_CHECK(cudaGetLastError());

  return { occupancy, empty_state, out_probs, out_colors, out_normals };

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

